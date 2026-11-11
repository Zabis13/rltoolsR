# Observations on the environment's device (rl_device_obs) and the PPO
# collection that reads them from an ag_buffer without the host. The host path
# is the reference: a CPU environment for the observations themselves, the
# same GPU environment with capture = FALSE (host path) for the collection.

has_vk_dev <- function() ggml_vulkan_available() && ggml_vulkan_device_count() > 0L

test_that("CPU environments keep returning matrices", {
  env <- rl_gpu_env("tictactoe", n = 2, device = "cpu")
  expect_true(is.matrix(rl_reset(env)))
  expect_true(is.matrix(rl_step(env, c(0, 4))$obs))
})

test_that("tictactoe device obs equals the host obs bit for bit", {
  skip_if_not(has_vk_dev(), "no Vulkan device")
  n <- 8L
  set.seed(3L)
  moves <- matrix(sample(0:8, n * 12, TRUE), n)
  run <- function(dev) {
    env <- rl_gpu_env("tictactoe", n = n, opponent = "random", device = dev)
    set.seed(5L)
    obs <- list(as.matrix(rl_reset(env)))
    for (t in seq_len(ncol(moves))) obs[[t + 1L]] <- as.matrix(rl_step(env, moves[, t])$obs)
    obs
  }
  env <- rl_gpu_env("tictactoe", n = n, device = "gpu")
  o <- rl_reset(env)
  expect_s3_class(o, "rl_device_obs")
  expect_identical(dim(o), c(n, 9L))
  expect_identical(run("gpu"), run("cpu"))
})

test_that("cube device obs and final_obs equal the host ones", {
  skip_if_not(has_vk_dev(), "no Vulkan device")
  n <- 6L
  run <- function(dev) {
    set.seed(7L)
    env <- rl_gpu_env("cube", n = n, size = 2, scramble = 3, max_steps = 4, device = dev)
    out <- list(obs = list(as.matrix(rl_reset(env))), fin = list())
    for (t in 1:10) {
      r <- rl_step(env, sample.int(env$action_space$n, n, TRUE) - 1L)
      out$obs[[t + 1L]] <- as.matrix(r$obs)
      tr <- r$truncated & !r$terminated
      if (any(tr)) out$fin[[length(out$fin) + 1L]] <- r$final_obs[tr, , drop = FALSE]
    }
    out
  }
  g <- run("gpu")
  h <- run("cpu")
  expect_identical(g$obs, h$obs)
  expect_true(length(h$fin) > 0)                 # max_steps = 4 truncates in 10 steps
  expect_identical(g$fin, h$fin)
})

test_that("a copy from another device's tensor is an error, not a host copy", {
  skip_if_not(has_vk_dev(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  ag_device("gpu")
  env <- rl_gpu_env("tictactoe", n = 4, device = "cpu")
  buf <- ag_buffer(9, 8)
  on.exit(ag_buffer_free(buf), add = TRUE)
  expect_error(ag_buffer_copy_from(buf, env$st$board[[1]], env$st$backend))
})

# one collection of `algo` on a fresh GPU tic-tac-toe environment
collect_once <- function(algo, seed = 21L) {
  set.seed(seed)
  env <- rl_gpu_env("tictactoe", n = 16, device = "gpu")
  ag_device("gpu")
  state <- algo$init(env, "gpu")
  algo$collect(state, env)
}

test_that("the path is chosen from the environment and the settings", {
  skip_if_not(has_vk_dev(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  path_of <- function(col) col$state$obs_path
  expect_identical(path_of(collect_once(rl_ppo(n_steps = 4))), "resident")
  expect_identical(path_of(collect_once(rl_ppo(n_steps = 4, rollout = "host"))), "device")
  expect_identical(path_of(collect_once(rl_ppo(n_steps = 4, capture = FALSE))), "host")
  expect_identical(path_of(collect_once(rl_ppo(n_steps = 4, normalize_obs = TRUE))), "host")
  expect_false("obs_device" %in% names(collect_once(rl_ppo(n_steps = 4, rollout = "host"))$info))
  # a CPU environment: matrices, the host path
  set.seed(1L)
  env <- rl_gpu_env("tictactoe", n = 16, device = "cpu")
  algo <- rl_ppo(n_steps = 4)
  ag_device("gpu")
  expect_identical(path_of(algo$collect(algo$init(env, "gpu"), env)), "host")
})

test_that("device and host collection give the same rollout", {
  skip_if_not(has_vk_dev(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  d <- collect_once(rl_ppo(n_steps = 5, rollout = "host"))
  h <- collect_once(rl_ppo(n_steps = 5, capture = FALSE))
  rd <- d$state$rollout
  rh <- h$state$rollout
  expect_null(rd$obs)
  N <- length(rh$logp)
  expect_identical(rd$cols, seq_len(N) - 1)
  # the buffer columns t * n_env + i hold the host b_obs layout
  expect_identical(ag_buffer_read(d$state$captures$obs_buf, 0, N), rh$obs)
  expect_identical(rd$act, rh$act)
  expect_identical(rd$legal, rh$legal)
  expect_equal(rd$logp, rh$logp, tolerance = 1e-6)
  expect_equal(rd$adv, rh$adv, tolerance = 1e-5)
  expect_equal(rd$ret, rh$ret, tolerance = 1e-5)
})

test_that("training on the device path matches the host path", {
  skip_if_not(has_vk_dev(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  run <- function(path) {
    # the host path from a CPU copy of the environment: same games, matrices
    # (normalize_obs = TRUE would also take it, but changes the data)
    env <- rl_gpu_env("tictactoe", n = 32, device = if (path == "device") "gpu" else "cpu")
    set.seed(31L)
    m <- rl_ppo_train(env, iterations = 2, n_steps = 9, minibatch = 96, device = "gpu",
                      verbose = FALSE, rollout = "host")
    list(h = m$history, w = lapply(m$net$parameters(), as.matrix))
  }
  d <- run("device")
  h <- run("host")
  # the paths themselves (default settings: device; CPU environment: host)
  # are checked in "the path is chosen from the environment and the settings"
  for (cl in c("pg", "vf", "ent")) expect_equal(d$h[[cl]], h$h[[cl]], tolerance = 1e-4, info = cl)
  for (nm in names(h$w)) expect_equal(d$w[[nm]], h$w[[nm]], tolerance = 1e-4, info = nm)
})
