# CartPole as ag_* ops (R/resident_cartpole.R), recorded with ag_capture():
# short trajectories against an f64 reference of the same formulas, the strict
# thresholds, truncation, the device resets, and when PPO picks the adapter.

skip_if(length(rltoolsR:::ggmlr_missing(rltoolsR:::RES_TRIG)) > 0, "ggmlR without ag_sin / ag_cos")

has_vk_cart <- function() ggml_vulkan_available() && ggml_vulkan_device_count() > 0L
cart_devices <- function() if (has_vk_cart()) c("cpu", "gpu") else "cpu"

cart_res_capture <- function(rs) {
  ag_capture(function(a, cols, key) rltoolsR:::cart_res_step(rs, a, cols, key),
             params = rltoolsR:::cart_res_params(rs))
}

# f64 reference step: s [4, n], a 0 / 1 [n]
cart_ref_step <- function(s, a) {
  p <- rltoolsR:::CARTPOLE
  tm <- p$masspole + p$masscart; pml <- p$masspole * p$length
  x <- s[1, ]; xd <- s[2, ]; th <- s[3, ]; thd <- s[4, ]
  force <- ifelse(a == 1, p$force_mag, -p$force_mag)
  temp <- (force + pml * thd^2 * sin(th)) / tm
  ta <- (p$gravity * sin(th) - cos(th) * temp) /
    (p$length * (4 / 3 - p$masspole * cos(th)^2 / tm))
  xa <- temp - pml * ta * cos(th) / tm
  rbind(x + p$tau * xd, xd + p$tau * xa, th + p$tau * thd, thd + p$tau * ta)
}

cart_run <- function(f, a, n, t) {
  o <- f(a = matrix(a, 1, n), cols = matrix(seq_len(n) - 1, n, 1),
         key = matrix(c(0.25, t), 2, 1))
  list(obs = matrix(o$obs, 4, n), final = matrix(o$final_obs, 4, n),
       term = as.vector(o$terminated) > 0.5, trunc = as.vector(o$truncated) > 0.5,
       done = as.vector(o$done) > 0.5, reward = as.vector(o$reward))
}

cart_with <- function(dev, n, code) {
  ag_device(dev)
  on.exit(ag_device("cpu"), add = TRUE)
  rs <- rltoolsR:::cart_res_new(n)
  on.exit(rltoolsR:::cart_res_free(rs), add = TRUE)
  f <- cart_res_capture(rs)
  on.exit(ag_capture_free(f), add = TRUE)
  code(rs, f)
}

test_that("resident cartpole follows the f64 physics for 50 steps", {
  for (dev in cart_devices()) cart_with(dev, 64L, function(rs, f) {
    set.seed(11L)
    n <- rs$n
    s <- ag_buffer_read(rs$state)
    alive <- rep(TRUE, n)
    p <- rltoolsR:::CARTPOLE
    got <- want <- list(obs = NULL, term = NULL, reward = NULL)
    for (t in seq_len(50L)) {
      a <- sample(0:1, n, replace = TRUE)
      ref <- cart_ref_step(s, a)
      r <- cart_run(f, a, n, t)
      ref_term <- abs(ref[1, ]) > p$x_threshold | abs(ref[3, ]) > p$theta_threshold
      # compared until each game's first end
      got$obs <- c(got$obs, r$final[, alive]); want$obs <- c(want$obs, ref[, alive])
      got$term <- c(got$term, r$term[alive]); want$term <- c(want$term, ref_term[alive])
      got$reward <- c(got$reward, r$reward)
      alive <- alive & !r$done
      s <- r$obs
    }
    expect_equal(got$obs, want$obs, tolerance = 1e-4, info = dev)
    expect_identical(got$term, want$term, info = dev)
    expect_true(all(got$reward == 1), info = dev)
    expect_true(any(!alive))                  # some games ended on the way
  })
})

test_that("resident cartpole thresholds are strict, both signs", {
  p <- rltoolsR:::CARTPOLE
  up <- function(v) v * (1 + 1e-6)            # the next f32 values above
  # x_dot = theta_dot = 0: the next x, theta equal the current ones
  x  <- c(p$x_threshold, -p$x_threshold, up(p$x_threshold), -up(p$x_threshold), 0, 0, 0, 0)
  th <- c(0, 0, 0, 0, p$theta_threshold, -p$theta_threshold,
          up(p$theta_threshold), -up(p$theta_threshold))
  want <- c(FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, TRUE, TRUE)
  n <- length(x)
  for (dev in cart_devices()) cart_with(dev, n, function(rs, f) {
    ag_buffer_write(rs$state, rbind(x, 0, th, 0), col_offset = 0)
    # action 1 pushes x_dot, theta_dot only: the positions of this step stay
    r <- cart_run(f, rep(1L, n), n, 0)
    expect_identical(r$term, want, info = dev)
    expect_identical(r$trunc, rep(FALSE, n), info = dev)
  })
})

test_that("resident cartpole truncates at 500 steps and resets the counter", {
  p <- rltoolsR:::CARTPOLE
  n <- 4L
  for (dev in cart_devices()) cart_with(dev, n, function(rs, f) {
    # games 1, 2 at step 499; game 2 also leaves the track; 3, 4 at 10
    st <- matrix(0, 4, n); st[1, 2] <- 2 * p$x_threshold
    ag_buffer_write(rs$state, st, col_offset = 0)
    ag_buffer_write(rs$steps, matrix(c(p$max_steps - 1, p$max_steps - 1, 10, 10), 1, n),
                    col_offset = 0)
    r <- cart_run(f, rep(0L, n), n, 0)
    expect_identical(r$trunc, c(TRUE, FALSE, FALSE, FALSE), info = dev)
    expect_identical(r$term, c(FALSE, TRUE, FALSE, FALSE), info = dev)
    expect_identical(r$done, c(TRUE, TRUE, FALSE, FALSE), info = dev)
    expect_equal(as.vector(ag_buffer_read(rs$steps)), c(0, 0, 11, 11), info = dev)
    # the ended games restarted; final_obs is the state before the restart
    expect_true(all(abs(r$obs[, 1:2]) <= 0.05), info = dev)
    expect_equal(r$final[, 3:4], r$obs[, 3:4], info = dev)
    expect_gt(abs(r$final[1, 2]), p$x_threshold)
  })
})

test_that("resident cartpole resets are uniform and differ per game and step", {
  n <- 2000L
  for (dev in cart_devices()) cart_with(dev, n, function(rs, f) {
    st <- matrix(0, 4, n); st[1, ] <- 10           # every game off the track
    draw <- function(t) {
      ag_buffer_write(rs$state, st, col_offset = 0)
      cart_run(f, rep(0L, n), n, t)$obs
    }
    o1 <- draw(1); o2 <- draw(2)
    expect_true(all(o1 >= -0.05 & o1 <= 0.05), info = dev)
    expect_equal(mean(o1), 0, tolerance = 0.002, info = dev)
    expect_equal(stats::var(as.vector(o1)), 0.1^2 / 12, tolerance = 0.05, info = dev)
    expect_lt(abs(stats::cor(o1[1, ], o1[2, ])), 0.1)      # components
    expect_lt(abs(stats::cor(o1[1, -1], o1[1, -n])), 0.1)  # neighbour games
    expect_lt(abs(stats::cor(as.vector(o1), as.vector(o2))), 0.1)  # steps
  })
})

test_that("PPO takes the resident cartpole for the GPU env, the C++ one only on request", {
  meta <- list(normalize_obs = FALSE)
  adapter <- function(env, rollout)
    rltoolsR:::ppo_res_adapter(env, FALSE, FALSE, TRUE, meta, "gpu", rollout)
  v <- rl_vec_env("cartpole", n = 2)
  expect_null(adapter(v, "auto"))
  expect_identical(adapter(v, "resident")$obs_dim, 4L)
  skip_if_not(has_vk_cart())
  g <- rl_gpu_env("cartpole", n = 2, device = "gpu")
  expect_identical(adapter(g, "auto")$obs_dim, 4L)
})
