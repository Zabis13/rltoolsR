# Tabular environments as ag_* ops (R/resident_tabular.R), recorded with
# ag_capture(): deterministic variants against rl_env step by step, slippery
# ones by the distribution of outcomes (the RNGs differ).

has_vk_res <- function() ggml_vulkan_available() && ggml_vulkan_device_count() > 0L
res_devices <- function() if (has_vk_res()) c("cpu", "gpu") else "cpu"

tab_res_capture <- function(rs) {
  ag_capture(function(a, cols, key) rltoolsR:::tab_res_step(rs, a, cols, key),
             params = rltoolsR:::tab_res_params(rs))
}

# 250 steps: past the 8x8 limit of 200, so truncation happens too
tab_res_check_det <- function(nm, dev, n = 16L, steps = 250L) {
  set.seed(3L)
  ag_device(dev)
  rs <- rltoolsR:::tab_res_new(rl_env(nm, seed = 1), n)
  on.exit(rltoolsR:::tab_res_free(rs), add = TRUE)
  f <- tab_res_capture(rs)
  on.exit(ag_capture_free(f), add = TRUE)
  singles <- lapply(seq_len(n), function(i) { e <- rl_env(nm, seed = i); rl_reset(e); e })
  cols <- matrix(seq_len(n) - 1, n, 1)
  S <- rs$S
  res <- ref <- list()
  for (t in seq_len(steps)) {
    a <- sample(0:3, n, replace = TRUE)
    o <- f(a = matrix(a, 1, n), cols = cols, key = matrix(c(stats::runif(1), t), 2, 1))
    rc <- lapply(seq_len(n), function(i) {
      x <- rl_step(singles[[i]], a[i])
      if (x$terminated || x$truncated) x$obs <- rl_reset(singles[[i]])
      x
    })
    obs_c <- vapply(rc, function(x) as.numeric(x$obs), numeric(if (rs$onehot) S else 1))
    res[[t]] <- list(obs = matrix(o$obs, ncol = n), reward = as.vector(o$reward),
                     term = as.vector(o$terminated) > 0.5, trunc = as.vector(o$truncated) > 0.5)
    ref[[t]] <- list(obs = matrix(obs_c, ncol = n), reward = vapply(rc, `[[`, 0, "reward"),
                     term = vapply(rc, `[[`, NA, "terminated"), trunc = vapply(rc, `[[`, NA, "truncated"))
  }
  expect_equal(res, ref, info = paste(nm, dev))
  ends <- sum(vapply(ref, function(x) sum(x$term | x$trunc), 0))
  expect_gte(ends, n)                      # games ended and restarted
}

tab_res_check_slip <- function(nm, dev, n = 3000L) {
  set.seed(4L)
  ag_device(dev)
  rs <- rltoolsR:::tab_res_new(rl_env(nm, seed = 1), n)
  on.exit(rltoolsR:::tab_res_free(rs), add = TRUE)
  f <- tab_res_capture(rs)
  on.exit(ag_capture_free(f), add = TRUE)
  # state 6 (row 1, col 2), action right: up -> 2, right -> 7 (hole), down -> 10
  ag_buffer_write(rs$state, matrix(6, 1, n), col_offset = 0)
  o <- f(a = matrix(2, 1, n), cols = matrix(seq_len(n) - 1, n, 1),
         key = matrix(c(stats::runif(1), 0), 2, 1))
  s <- if (rs$onehot) apply(o$obs, 2, which.max) - 1 else as.vector(o$obs)
  term <- as.vector(o$terminated) > 0.5
  # the hole ends the game, which restarts at 0 in the same step
  expect_identical(term, s == 0)
  s[term] <- 7
  cnt <- tabulate(match(s, c(2, 7, 10)), 3)
  expect_equal(sum(cnt), n)
  chi2 <- sum((cnt - n / 3)^2 / (n / 3))
  expect_lt(chi2, stats::qchisq(0.999, 2), label = paste(nm, dev, "chi-square"))
}

test_that("resident tables hold the C++ table for every (s, a, k)", {
  on.exit(ag_device("cpu"), add = TRUE)
  ag_device("cpu")
  for (nm in c("frozen_lake", "frozen_lake_8x8_det")) {
    env <- rl_env(nm, seed = 1)
    tab <- rltoolsR:::rl_env_table(env)
    rs <- rltoolsR:::tab_res_new(env, 2L)
    S <- rs$S; A <- rs$A; K <- rs$K
    out <- ag_buffer_read(rs$OUT)
    lp <- ag_buffer_read(rs$LP)
    rltoolsR:::tab_res_free(rs)
    # column (s A + a) K + k  <->  [s, a, k]
    s <- rep(0:(S - 1), each = A * K); a <- rep(rep(0:(A - 1), each = K), S); k <- rep(0:(K - 1), S * A)
    i <- cbind(s + 1, a + 1, k + 1)
    expect_equal(out[1, ], tab$`next`[i] + 0, info = nm)
    expect_equal(out[2, ], tab$reward[i], info = nm)
    expect_equal(out[3, ], tab$terminated[i] + 0, info = nm)
    p <- matrix(exp(lp), K)                 # padding: exp(-1e30) = 0
    expect_equal(as.vector(p), tab$prob[i], tolerance = 1e-6, info = nm)
  }
})

# Taxi starts at random: the device draws restarts with its own RNG, so after
# every step the singles take over the device's state for games that ended;
# all other games must agree step by step.
tab_res_check_taxi <- function(dev, n = 32L, steps = 250L) {
  set.seed(7L)
  ag_device(dev)
  rs <- rltoolsR:::tab_res_new(rl_env("taxi", seed = 1), n)
  on.exit(rltoolsR:::tab_res_free(rs), add = TRUE)
  f <- tab_res_capture(rs)
  on.exit(ag_capture_free(f), add = TRUE)
  load <- function(e, s, k = 0L) rl_load_state(e, c(writeBin(as.integer(k), raw()),
                                                     writeBin(as.integer(s), raw())))
  s0 <- as.vector(ag_buffer_read(rs$state))
  singles <- lapply(seq_len(n), function(i) { e <- rl_env("taxi", seed = i); load(e, s0[i]); e })
  isd_ok <- which(rltoolsR:::rl_env_table(singles[[1]])$isd > 0) - 1
  cols <- matrix(seq_len(n) - 1, n, 1)
  bad <- 0L; ends <- 0L; restarts <- NULL
  for (t in seq_len(steps)) {
    a <- sample(0:5, n, replace = TRUE)
    o <- f(a = matrix(a, 1, n), cols = cols, key = matrix(c(stats::runif(1), t), 2, 1))
    s_dev <- as.vector(o$obs)
    for (i in seq_len(n)) {
      x <- rl_step(singles[[i]], a[i])
      same <- x$reward == o$reward[i] && x$terminated == (o$terminated[i] > 0.5) &&
        x$truncated == (o$truncated[i] > 0.5)
      if (x$terminated || x$truncated) {
        ends <- ends + 1L
        restarts <- c(restarts, s_dev[i])
        load(singles[[i]], s_dev[i])           # the device's restart
      } else {
        same <- same && x$obs == s_dev[i]
      }
      if (!same) bad <- bad + 1L
    }
  }
  expect_identical(bad, 0L, label = paste("taxi", dev, "mismatched steps"))
  expect_gte(ends, n)
  expect_true(all(restarts %in% isd_ok))
}

# Restarts drawn on the device follow isd: every game truncates on its next
# step (steps = 199), so all n restart at once.
tab_res_check_taxi_isd <- function(dev, n = 30000L) {
  set.seed(8L)
  ag_device(dev)
  rs <- rltoolsR:::tab_res_new(rl_env("taxi", seed = 1), n)
  on.exit(rltoolsR:::tab_res_free(rs), add = TRUE)
  f <- tab_res_capture(rs)
  on.exit(ag_capture_free(f), add = TRUE)
  ag_buffer_write(rs$steps, matrix(199, 1, n), col_offset = 0)
  o <- f(a = matrix(0, 1, n), cols = matrix(seq_len(n) - 1, n, 1),
         key = matrix(c(stats::runif(1), 0), 2, 1))
  done <- as.vector(o$done) > 0.5
  expect_true(all(done))
  ok <- which(rs$isd > 0) - 1
  s <- as.vector(o$obs)
  expect_true(all(s %in% ok))
  cnt <- tabulate(match(s, ok), length(ok))
  chi2 <- sum((cnt - n / 300)^2 / (n / 300))
  expect_lt(chi2, stats::qchisq(0.999, 299), label = paste("taxi", dev, "chi-square"))
}

# Blackjack: observations from the environment's table (3 numbers, or 45 for
# the one-hots); hit from hard 20 busts with probability 12/13, the bust shows
# its sum in final_obs and the game restarts with a valid hand.
tab_res_check_blackjack <- function(nm, dev, n = 30000L) {
  set.seed(10L)
  ag_device(dev)
  env <- rl_env(nm, seed = 1)
  rs <- rltoolsR:::tab_res_new(env, n)
  on.exit(rltoolsR:::tab_res_free(rs), add = TRUE)
  f <- ag_capture(function(a, cols, key) rltoolsR:::tab_res_step(rs, a, cols, key),
                  params = rltoolsR:::tab_res_params(rs))
  on.exit(ag_capture_free(f), add = TRUE)
  ag_buffer_write(rs$state, matrix((20 * 11 + 5) * 2, 1, n), col_offset = 0)
  o <- f(a = matrix(1, 1, n), cols = matrix(seq_len(n) - 1, n, 1),
         key = matrix(c(stats::runif(1), 0), 2, 1))
  onehot <- nm == "blackjack_onehot"
  dec <- function(x) if (onehot) rbind(apply(x[1:32, ], 2, which.max) - 1,
                                       apply(x[33:43, ], 2, which.max) - 1,
                                       apply(x[44:45, ], 2, which.max) - 1) else x
  fin <- dec(o$final_obs); obs <- dec(o$obs)
  bust <- as.vector(o$terminated) > 0.5
  expect_identical(bust, fin[1, ] > 21)
  expect_true(all(as.vector(o$reward)[bust] == -1))
  expect_true(all(fin[2, ] == 5 & fin[3, ] == 0))
  # within 4.5 binomial standard deviations of 12/13
  expect_lt(abs(sum(bust) - n * 12 / 13), 4.5 * sqrt(n * 12 / 13 / 13))
  # restarted hands: sums 4..21, a dealer card, ace only from soft 12 up
  r <- obs[, bust, drop = FALSE]
  expect_true(all(r[1, ] >= 4 & r[1, ] <= 21 & r[2, ] >= 1 & r[2, ] <= 10 & (r[3, ] == 0 | r[1, ] >= 12)))
}

for (dev in res_devices()) {
  test_that(paste("resident blackjack hits and restarts on", dev), {
    on.exit(ag_device("cpu"), add = TRUE)
    for (nm in c("blackjack", "blackjack_onehot")) tab_res_check_blackjack(nm, dev)
  })

  test_that(paste("resident taxi matches rl_env and restarts by isd on", dev), {
    on.exit(ag_device("cpu"), add = TRUE)
    tab_res_check_taxi(dev)
    tab_res_check_taxi_isd(dev)
  })
  test_that(paste("resident deterministic tabular envs match rl_env on", dev), {
    on.exit(ag_device("cpu"), add = TRUE)
    for (nm in c("frozen_lake_det", "frozen_lake_det_onehot",
                 "frozen_lake_8x8_det", "frozen_lake_8x8_det_onehot",
                 "cliff_walking", "cliff_walking_onehot")) tab_res_check_det(nm, dev)
  })

  test_that(paste("resident slippery frozen_lake slips 1/3 each way on", dev), {
    on.exit(ag_device("cpu"), add = TRUE)
    for (nm in c("frozen_lake", "frozen_lake_onehot")) tab_res_check_slip(nm, dev)
  })
}
