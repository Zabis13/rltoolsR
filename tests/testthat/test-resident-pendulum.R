# Pendulum as ag_* ops (R/resident_pendulum.R), recorded with ag_capture():
# 50 steps against an f64 reference that keeps theta unwrapped (Gymnasium),
# the wrap across +-pi, the reward at theta = +-pi, the torque of both action
# scales, truncation at 200, the device resets and their noise stream.

skip_if(length(rltoolsR:::ggmlr_missing(rltoolsR:::RES_TRIG)) > 0, "ggmlR without ag_sin / ag_cos")

has_vk_pend <- function() ggml_vulkan_available() && ggml_vulkan_device_count() > 0L
pend_devices <- function() if (has_vk_pend()) c("cpu", "gpu") else "cpu"

# f64 reference step: s [2, n] (theta unwrapped), a [n]; reward from s
pend_ref_step <- function(s, a, hi) {
  p <- rltoolsR:::PENDULUM_GYM
  u <- pmin(pmax(a, -hi), hi) * p$max_torque / hi
  th <- s[1, ]; thd <- s[2, ]
  thn <- ((th + pi) %% (2 * pi)) - pi
  r <- -(thn^2 + 0.1 * thd^2 + 0.001 * u^2)
  thd1 <- pmin(pmax(thd + (3 * p$g / (2 * p$l) * sin(th) + 3 / (p$m * p$l^2) * u) * p$dt,
                    -p$max_speed), p$max_speed)
  list(s = rbind(th + thd1 * p$dt, thd1), reward = r)
}
pend_ref_obs <- function(s) rbind(cos(s[1, ]), sin(s[1, ]), s[2, ])

pend_run <- function(f, a, n, t) {
  o <- f(a = matrix(a, 1, n), cols = matrix(seq_len(n) - 1, n, 1),
         key = matrix(c(0.375, t), 2, 1))
  list(obs = matrix(o$obs, 3, n), final = matrix(o$final_obs, 3, n),
       reward = as.vector(o$reward), term = as.vector(o$terminated) > 0.5,
       trunc = as.vector(o$truncated) > 0.5, done = as.vector(o$done) > 0.5)
}

pend_with <- function(dev, n, raw, code) {
  ag_device(dev)
  on.exit(ag_device("cpu"), add = TRUE)
  rs <- rltoolsR:::pend_res_new(n, raw)
  on.exit(rltoolsR:::pend_res_free(rs), add = TRUE)
  f <- ag_capture(function(a, cols, key) rltoolsR:::pend_res_step(rs, a, cols, key),
                  params = rltoolsR:::pend_res_params(rs))
  on.exit(ag_capture_free(f), add = TRUE)
  code(rs, f)
}

test_that("resident pendulum follows the f64 physics for 50 steps, theta wrapped", {
  for (raw in c(FALSE, TRUE)) for (dev in pend_devices()) pend_with(dev, 64L, raw, function(rs, f) {
    set.seed(21L)
    n <- rs$n
    s <- ag_buffer_read(rs$state)
    wrapped <- 0
    got <- want <- list(reward = NULL, obs = NULL)
    done <- FALSE; th_max <- 0
    for (t in seq_len(50L)) {
      a <- stats::runif(n, -1.5 * rs$hi, 1.5 * rs$hi)    # past the bounds too
      ref <- pend_ref_step(s, a, rs$hi)
      r <- pend_run(f, a, n, t)
      got$reward <- c(got$reward, r$reward); want$reward <- c(want$reward, ref$reward)
      got$obs <- c(got$obs, r$final); want$obs <- c(want$obs, pend_ref_obs(ref$s))
      done <- done || any(r$done)
      th_max <- max(th_max, abs(ag_buffer_read(rs$state)[1, ]))
      wrapped <- wrapped + sum(abs(ref$s[1, ]) > pi)
      s <- ref$s
    }
    info <- paste(dev, if (raw) "raw" else "scaled")
    expect_equal(got$reward, want$reward, tolerance = 1e-4, info = info)
    expect_equal(got$obs, want$obs, tolerance = 1e-4, info = info)
    expect_false(done, info = info)
    expect_lte(th_max, pi + 1e-6)            # theta stays wrapped
    expect_gt(wrapped, 0)                    # the reference left [-pi, pi]
  })
})

test_that("resident pendulum: the wrap across +-pi, the reward at +-pi", {
  p <- rltoolsR:::PENDULUM_GYM
  # games 1, 2 cross pi upwards / -pi downwards; 3, 4 sit at +-pi
  s0 <- rbind(c(pi - 0.01, -pi + 0.01, pi, -pi), c(2, -2, 0, 0))
  n <- ncol(s0)
  for (dev in pend_devices()) pend_with(dev, n, FALSE, function(rs, f) {
    ag_buffer_write(rs$state, s0, col_offset = 0)
    ref1 <- pend_ref_step(s0, rep(0, n), rs$hi)
    r1 <- pend_run(f, rep(0, n), n, 1)
    expect_equal(r1$reward[3:4], rep(-pi^2, 2), tolerance = 1e-6, info = dev)
    expect_equal(r1$reward, ref1$reward, tolerance = 1e-5, info = dev)
    th <- ag_buffer_read(rs$state)[1, ]
    expect_true(th[1] < 0 && th[2] > 0, info = dev)        # wrapped to the other side
    expect_equal(r1$final, pend_ref_obs(ref1$s), tolerance = 1e-5, info = dev)
    # the next reward uses the wrapped theta: no jump of 2 pi
    ref2 <- pend_ref_step(ref1$s, rep(0, n), rs$hi)
    r2 <- pend_run(f, rep(0, n), n, 2)
    expect_equal(r2$reward, ref2$reward, tolerance = 1e-4, info = dev)
  })
})

test_that("resident pendulum torque: 2 clip(a, -1, 1) scaled, clip(a, -2, 2) raw", {
  a <- c(-3, -1.5, -0.7, 0, 0.7, 1.5, 3)
  n <- length(a)
  for (raw in c(FALSE, TRUE)) for (dev in pend_devices()) pend_with(dev, n, raw, function(rs, f) {
    ag_buffer_write(rs$state, matrix(0, 2, n), col_offset = 0)
    u <- if (raw) pmin(pmax(a, -2), 2) else 2 * pmin(pmax(a, -1), 1)
    r <- pend_run(f, a, n, 1)
    # theta = theta_dot = 0: the reward is the torque term alone
    expect_equal(r$reward, -0.001 * u^2, tolerance = 1e-6, info = paste(dev, raw))
    expect_equal(r$final[3, ], 3 * u * rltoolsR:::PENDULUM_GYM$dt, tolerance = 1e-6,
                 info = paste(dev, raw))
  })
})

test_that("resident pendulum truncates at 200 steps, never terminates", {
  p <- rltoolsR:::PENDULUM_GYM
  n <- 3L
  for (dev in pend_devices()) pend_with(dev, n, FALSE, function(rs, f) {
    ag_buffer_write(rs$state, rbind(c(3, 3, 3), c(0, 0, 0)), col_offset = 0)
    ag_buffer_write(rs$steps, matrix(c(p$max_steps - 1, p$max_steps - 2, 0), 1, n),
                    col_offset = 0)
    r <- pend_run(f, rep(0, n), n, 1)
    expect_identical(r$trunc, c(TRUE, FALSE, FALSE), info = dev)
    expect_identical(r$term, rep(FALSE, n), info = dev)
    expect_identical(r$done, r$trunc, info = dev)
    expect_equal(as.vector(ag_buffer_read(rs$steps)), c(0, p$max_steps - 1, 1), info = dev)
    expect_equal(r$final[, 2:3], r$obs[, 2:3], info = dev)
    expect_gt(abs(r$final[1, 1] - r$obs[1, 1]) + abs(r$final[2, 1] - r$obs[2, 1]), 0)
  })
})

test_that("resident pendulum resets are uniform, apart from the policy's stream", {
  n <- 2000L
  for (dev in pend_devices()) pend_with(dev, n, FALSE, function(rs, f) {
    ag_buffer_write(rs$state, matrix(0, 2, n), col_offset = 0)
    ag_buffer_write(rs$steps, matrix(199, 1, n), col_offset = 0)   # all truncate now
    key <- matrix(c(0.375, 1), 2, 1)
    r <- pend_run(f, rep(0, n), n, 1)
    s <- ag_buffer_read(rs$state)
    expect_true(all(abs(s[1, ]) <= pi & abs(s[2, ]) <= 1), info = dev)
    expect_equal(mean(s[1, ]), 0, tolerance = 0.1, info = dev)
    expect_equal(stats::var(s[1, ]), (2 * pi)^2 / 12, tolerance = 0.05, info = dev)
    expect_equal(stats::var(s[2, ]), 4 / 12, tolerance = 0.05, info = dev)
    expect_lt(abs(stats::cor(s[1, ], s[2, ])), 0.1)
    # the policy's noise: stream 0 of the same key
    eps <- as.vector(as.matrix(ag_rand(ag_tensor(key), "normal", c(1L, n))))
    expect_lt(abs(stats::cor(eps, s[1, ])), 0.1)
    expect_lt(abs(stats::cor(eps, s[2, ])), 0.1)
  })
})
