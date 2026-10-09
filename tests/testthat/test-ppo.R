test_that("PPO act respects the legal mask", {
  set.seed(2)
  env <- rl_gpu_env("tictactoe", n = 64, device = "cpu")
  m <- rl_ppo_train(env, iterations = 1, device = "cpu", verbose = FALSE)
  legal <- matrix(FALSE, 64, 9)
  legal[cbind(1:64, sample(1:9, 64, TRUE))] <- TRUE
  obs <- matrix(0, 64, 9)
  a <- rl_act(m, obs, legal, greedy = FALSE)
  expect_true(all(legal[cbind(1:64, a + 1L)]))
  a <- rl_act(m, obs, legal, greedy = TRUE)
  expect_true(all(legal[cbind(1:64, a + 1L)]))
})

test_that("PPO trains on GPU with losses close to CPU", {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() > 0L, "no Vulkan device")
  run <- function(dev) {
    set.seed(5)
    env <- rl_gpu_env("tictactoe", n = 128, device = dev)
    rl_ppo_train(env, iterations = 3, minibatch = 1152, device = dev, verbose = FALSE)$history
  }
  h_cpu <- run("cpu"); h_gpu <- run("gpu")
  expect_true(all(is.finite(as.matrix(h_gpu))))
  # first iteration sees identical rollouts; f32 GPU math drifts afterwards
  expect_equal(h_gpu$vf[1], h_cpu$vf[1], tolerance = 1e-3)
  expect_equal(h_gpu$ent[1], h_cpu$ent[1], tolerance = 1e-3)
})

test_that("a discrete PPO update moves the policy along the advantage", {
  # catches sign errors in the policy gradient without training in an env;
  # learning itself is checked by inst/examples/check_learning_tictactoe.R
  set.seed(10)
  net <- rltoolsR:::ppo_net(9L, 9L, 16L)
  opt <- optimizer_adam(net$parameters(), lr = 1e-2)
  B <- 64
  obs <- matrix(sample(c(-1, 0, 1), 9 * B, TRUE), 9)
  legal <- matrix(1, 9, B)
  logp_all <- function() {
    ref_masked_log_softmax(as.matrix(net$forward(ag_tensor(obs))$logits), legal)
  }
  value_of <- function() as.numeric(as.matrix(net$forward(ag_tensor(obs))$value))
  act <- ag_sample_categorical(logp_all())
  pick <- cbind(act + 1L, seq_len(B))
  old_logp <- logp_all()[pick]
  adv <- rep(c(1, -1), each = B / 2)
  ret <- rep(0.5, B)
  v0 <- mean(abs(value_of() - ret))
  for (i in 1:5) {
    rltoolsR:::ppo_update(net, opt, obs, legal, act, old_logp, adv, ret,
                          clip = 0.2, vf_coef = 0.5, ent_coef = 0)
  }
  d <- logp_all()[pick] - old_logp
  expect_gt(mean(d[adv > 0]), 0)
  expect_lt(mean(d[adv < 0]), 0)
  expect_lt(mean(abs(value_of() - ret)), v0)
})

has_vk <- function() ggml_vulkan_available() && ggml_vulkan_device_count() > 0L

test_that("policy_logp matches an independent reference, masked and extreme", {
  set.seed(21)
  legal <- matrix(rbinom(9 * 40, 1, 0.6), 9, 40); legal[1, ] <- 1
  logits <- cbind(matrix(rnorm(9 * 20, sd = 3), 9, 20),
                  matrix(runif(9 * 20, -60, 60), 9, 20))   # spread up to ~100
  ref <- ref_masked_log_softmax(logits, legal)
  for (dev in c("cpu", if (has_vk()) "gpu")) {
    prev <- ag_device(dev)
    got <- as.matrix(rltoolsR:::policy_logp(ag_tensor(logits), legal))
    ag_device(prev)
    expect_true(all(is.finite(got)), info = dev)
    # absolute error, with |logp| up to ~100: double on the CPU, f32 on the GPU
    expect_lt(max(abs(got - ref)[legal == 1]), if (dev == "cpu") 1e-10 else 1e-4)
    expect_true(all(exp(got[legal == 0]) == 0), info = dev)
  }
})

test_that("greedy rl_act masks before argmax, on CPU and GPU", {
  set.seed(22)
  env <- rl_gpu_env("tictactoe", n = 16, device = "cpu")
  m <- rl_ppo_train(env, iterations = 1, device = "cpu", verbose = FALSE)
  # make action 4 the overwhelming favourite everywhere
  sd <- ag_state_dict(m$net)
  sd$parameters$pi_b[5, 1] <- 1e3
  ag_load_state_dict(m$net, sd)
  f <- tempfile(fileext = ".rds"); on.exit(unlink(f), add = TRUE)
  rl_save_model(m, f)
  obs <- matrix(sample(c(-1, 0, 1), 32 * 9, TRUE), 32, 9)
  only_not4 <- matrix(TRUE, 32, 9); only_not4[, 5] <- FALSE
  for (dev in c("cpu", if (has_vk()) "gpu")) {
    md <- rl_load_model(f, device = dev)
    expect_true(all(rl_act(md, obs) == 4L), info = dev)                # legal: it wins
    a <- rl_act(md, obs, only_not4)
    expect_false(any(a == 4L), info = dev)                             # illegal: never
    # and the choice is the reference argmax among the legal actions
    prev <- ag_device(dev); prec <- ag_matmul_precision("f32")   # as rl_act()
    lg <- as.matrix(md$net$forward(ag_tensor(t(obs)))$logits)
    ag_matmul_precision(prec); ag_device(prev)
    ref <- apply(ref_masked_log_softmax(lg, t(only_not4) * 1), 2, which.max) - 1L
    expect_identical(a, ref, info = dev)
  }
})

test_that("sampling never draws a masked action, even when exp(logp) sums below 1", {
  set.seed(23)
  n <- 20000L
  # f32-like logp whose probabilities sum to 0.9; the last action is masked
  logp <- matrix(log(c(0.5, 0.4, 0, 0)), 4, n)
  logp[3:4, ] <- -1e4
  a <- ag_sample_categorical(logp)
  expect_true(all(a %in% c(0L, 1L)))
  expect_equal(mean(a == 0L), 0.5 / 0.9, tolerance = 0.02)
})

test_that("logp at collection equals the update's first pass (f32 matmuls)", {
  # collection runs at batch n_env, the update at the minibatch: with 8 envs and
  # a 256 minibatch the two cross the GPU's mat-vec / mat-mat border (n > 8).
  # One function (policy_logp) with f32-tile matmuls must give the same numbers.
  for (dev in c("cpu", if (has_vk()) "gpu")) {
    set.seed(24)
    env <- rl_gpu_env("tictactoe", n = 8L, opponent = "self", device = dev)
    algo <- rl_ppo(n_steps = 64, minibatch = 256, hidden = 64)
    prev <- ag_device(dev)
    prec <- ag_matmul_precision("f32")
    state <- algo$init(env, dev)
    ro <- algo$collect(state, env)$state$rollout
    idx <- seq_len(256L)
    out <- state$net$forward(rltoolsR:::obs_input(ro$obs[, idx], ro$norm))
    lp <- as.matrix(rltoolsR:::policy_logp(out$logits, ro$legal[, idx] * 1))
    new <- lp[cbind(ro$act[idx] + 1L, seq_along(idx))]
    ag_matmul_precision(prec); ag_device(prev)
    expect_lt(max(abs(new - ro$logp[idx])), if (dev == "cpu") 1e-12 else 1e-6)
  }
})
