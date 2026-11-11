# rollout = "resident" (R/ppo_resident.R): the policy and tic-tac-toe in one
# recorded graph per step. Checked against a host loop -- rl_gpu_env on the CPU,
# policy_act() with the same keys (k1, k2 + t) -- over two rollouts in a row
# (the second continues the games), then end to end through rl_ppo().

has_vk_rr <- function() ggml_vulkan_available() && ggml_vulkan_device_count() > 0L

test_that("the resident rollout equals a host loop on the same keys", {
  skip_if_not(has_vk_rr(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(3L)
  n <- 32L; ns <- 9L
  ag_device("gpu")
  ag_local_mode(graph = TRUE, matmul_precision = "f32")
  env <- rl_gpu_env("tictactoe", n = n, device = "gpu")
  state <- rl_ppo(n_steps = ns)$init(env, "gpu")
  net <- state$net
  cap <- new.env(parent = emptyenv())
  rltoolsR:::ppo_res_setup(net, cap, n, ns)
  on.exit(rltoolsR:::ppo_free_captures(cap), add = TRUE)

  ref <- rl_gpu_env("tictactoe", n = n, device = "cpu")
  obs <- rl_reset(ref)
  legal <- rl_legal_actions(ref)
  for (round in 1:2) {
    key <- c(stats::runif(1), floor(stats::runif(1) * 4096) / 4096)
    r <- rltoolsR:::ppo_res_rollout(cap, n, ns, key)
    h_obs <- h_legal <- NULL
    h <- list(act = NULL, logp = NULL, val = NULL, rew = NULL, done = NULL)
    for (t in seq_len(ns)) {
      x <- t(obs)
      lg <- t(legal) * 1
      o <- rltoolsR:::policy_act(net, x, NULL, lg, key = c(key[1], key[2] + t - 1))
      s <- rl_step(ref, o$a)
      h_obs <- cbind(h_obs, x); h_legal <- cbind(h_legal, lg)
      h$act <- cbind(h$act, o$a); h$logp <- cbind(h$logp, o$logp)
      h$val <- cbind(h$val, o$value); h$rew <- cbind(h$rew, s$reward)
      h$done <- cbind(h$done, s$terminated * 1)
      obs <- s$obs; legal <- s$legal
    }
    last <- as.numeric(rltoolsR:::policy_forward(net, t(obs))$value)
    info <- paste("rollout", round)
    expect_identical(r$act, h$act + 0, info = info)
    expect_identical(r$rew, h$rew + 0, info = info)
    expect_identical(r$done, h$done, info = info)
    expect_identical(r$legal, h_legal, info = info)
    expect_identical(ag_buffer_read(cap$obs_buf), h_obs, info = info)
    expect_equal(r$logp, h$logp, tolerance = 1e-6, info = info)
    expect_equal(r$val, h$val, tolerance = 1e-6, info = info)
    expect_equal(r$last_val, last, tolerance = 1e-6, info = info)
  }
})

test_that("rl_ppo picks the resident rollout and trains on it", {
  skip_if_not(has_vk_rr(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(4L)
  env <- rl_gpu_env("tictactoe", n = 64, device = "gpu")
  ag_device("gpu")
  algo <- rl_ppo(n_steps = 9, minibatch = 576)
  state <- algo$init(env, "gpu")
  col <- algo$collect(state, env)
  expect_identical(col$state$obs_path, "resident")
  expect_identical(col$steps, 576)
  expect_gt(col$info[["episodes"]], 0)
  m <- algo$update(col$state)$metrics
  expect_true(all(is.finite(m)))
  # a CPU environment cannot take it
  cpu_env <- rl_gpu_env("tictactoe", n = 8, device = "cpu")
  bad <- rl_ppo(n_steps = 4, rollout = "resident")
  expect_error(bad$collect(bad$init(cpu_env, "gpu"), cpu_env), "needs rl_gpu_env")
})
