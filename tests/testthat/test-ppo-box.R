test_that("running stats match the statistics of all data seen", {
  set.seed(1)
  x1 <- matrix(rnorm(3 * 50, 2, 3), 3)
  x2 <- matrix(rnorm(3 * 70, -1, 0.5), 3)
  rs <- rltoolsR:::running_stats(3L)
  expect_equal(rs$var, rep(1, 3))
  rs <- rltoolsR:::running_stats_update(rs, x1)
  # count 0: the prior has no weight after the first batch
  expect_equal(rs$mean, rowMeans(x1))
  rs <- rltoolsR:::running_stats_update(rs, x2)
  all <- cbind(x1, x2)
  expect_equal(rs$count, 120)
  expect_equal(rs$mean, rowMeans(all))
  expect_equal(rs$var, rowMeans((all - rowMeans(all))^2))
})

test_that("observation normalization in the graph follows the stored form", {
  norm <- list(mean = c(1, -2), var = c(4, 0.25), epsilon = 1e-8, clip_obs = 3)
  x <- matrix(c(3, -2, 100, -2.5), 2)
  got <- as.matrix(rltoolsR:::obs_input(x, norm))
  want <- pmin(pmax((x - norm$mean) / sqrt(norm$var + norm$epsilon), -3), 3)
  expect_equal(got, want, tolerance = 1e-6)
})

test_that("box net layers get orthogonal init with their gains", {
  # Orthogonal init lives in ggmlR::ag_linear(init = "orthogonal") now; this
  # checks the net wires the gains: sqrt(2) hidden, 0.01 policy, 1 value.
  set.seed(2)
  p <- rltoolsR:::ppo_box_net(3L, 2L, 8L)$parameters()
  W <- function(nm) as.matrix(p[[nm]])
  expect_equal(crossprod(W("a1_W")), 2 * diag(3))        # 8x3, tall
  expect_equal(tcrossprod(W("a2_W")), 2 * diag(8))       # 8x8
  expect_equal(tcrossprod(W("mu_W")), 1e-4 * diag(2))    # 2x8, wide
  expect_equal(tcrossprod(W("v_W")), diag(1))            # 1x8
})

test_that("rollout logp equals the update's graph logp under the rollout normalization", {
  set.seed(3)
  env <- rl_vec_env("pendulum", n = 4, seed = 3)
  algo <- rl_ppo(n_steps = 32, minibatch = 64)
  state <- algo$init(env, "cpu")
  # the second rollout is the first one normalized with non-trivial statistics
  state <- algo$collect(state, env)$state
  state <- algo$update(state)$state
  state <- algo$collect(state, env)$state
  ro <- state$rollout
  expect_false(isTRUE(all.equal(ro$norm$mean, numeric(3))))
  out <- state$net$forward(rltoolsR:::obs_input(ro$obs, ro$norm))
  lp <- as.numeric(as.matrix(rltoolsR:::box_logp_tensor(out, ag_tensor(ro$act))))
  expect_equal(lp - 0.5 * log(2 * pi), ro$logp, tolerance = 1e-4)
})

test_that("statistics are frozen during a rollout and move after it", {
  set.seed(4)
  env <- rl_vec_env("pendulum", n = 2, seed = 4)
  algo <- rl_ppo(n_steps = 16, minibatch = 32)
  state <- algo$init(env, "cpu")
  s1 <- algo$collect(state, env)$state
  # first rollout seen with the prior (identity) statistics
  expect_equal(s1$rollout$norm$mean, numeric(3))
  expect_equal(s1$rollout$norm$var, rep(1, 3))
  expect_equal(s1$obs_rms$count, 32)
  expect_equal(s1$obs_rms$mean, rowMeans(s1$rollout$obs))
  expect_equal(s1$rew_rms$count, 32)
  m <- algo$model(s1)
  expect_equal(m$meta$obs_norm$mean, s1$obs_rms$mean)
  expect_identical(m$meta$obs_norm$form, "clip((x - mean) / sqrt(var + epsilon), -clip_obs, clip_obs)")
  # acting does not move the statistics
  rl_act(m, matrix(rnorm(6), 2), greedy = FALSE)
  expect_equal(algo$model(s1)$meta$obs_norm, m$meta$obs_norm)
})

test_that("continuous rl_act returns clipped n x act_dim actions", {
  set.seed(5)
  env <- rl_vec_env("pendulum", n = 4, seed = 5)
  m <- rl_train(env, rl_ppo(n_steps = 16, minibatch = 64), total_steps = 64,
                device = "cpu", verbose = FALSE)
  expect_true(m$meta$normalize_obs)
  expect_true(m$meta$normalize_reward)
  obs <- matrix(rnorm(40 * 3, sd = 5), 40)
  for (g in c(TRUE, FALSE)) {
    a <- rl_act(m, obs, greedy = g)
    expect_equal(dim(a), c(40L, 1L))
    expect_true(all(a >= -2 & a <= 2))
  }
})

test_that("continuous model saves and loads with its normalization", {
  set.seed(6)
  env <- rl_vec_env("car", n = 4, seed = 6)
  m <- rl_train(env, rl_ppo(n_steps = 16, minibatch = 64), total_steps = 128,
                device = "cpu", verbose = FALSE)
  f <- tempfile(fileext = ".rds")
  on.exit(unlink(f))
  rl_save_model(m, f)
  m2 <- rl_load_model(f, device = "cpu")
  expect_equal(m2$meta$obs_norm, m$meta$obs_norm)
  obs <- matrix(rnorm(10 * 6), 10)
  expect_equal(rl_act(m2, obs), rl_act(m, obs), tolerance = 1e-6)
})

test_that("set_train_state restores normalizers and the policy", {
  set.seed(7)
  env <- rl_vec_env("pendulum", n = 4, seed = 7)
  algo <- rl_ppo(n_steps = 16, minibatch = 64)
  m <- rl_train(env, algo, total_steps = 128, device = "cpu", verbose = FALSE)
  fresh <- algo$init(env, "cpu")
  restored <- algo$set_train_state(fresh, m$train_state$algo_state)
  expect_equal(restored$obs_rms, m$train_state$algo_state$normalizers$obs)
  expect_equal(restored$rew_rms, m$train_state$algo_state$normalizers$reward)
  expect_null(restored$ret_track)
  obs <- matrix(rnorm(8 * 3), 8)
  expect_equal(rl_act(algo$model(restored), obs), rl_act(m, obs), tolerance = 1e-6)
  # collect after a restore starts from a reset
  col <- algo$collect(restored, env)
  expect_equal(dim(col$state[["obs"]]), c(4L, 3L))
  expect_equal(col$steps, 64)
})

test_that("discrete PPO keeps normalization off by default", {
  env <- rl_gpu_env("tictactoe", n = 16, device = "cpu")
  m <- rl_ppo_train(env, iterations = 1, minibatch = 144, device = "cpu", verbose = FALSE)
  expect_false(m$meta$normalize_obs)
  expect_false(m$meta$normalize_reward)
  expect_null(m$meta$obs_norm)
})

test_that("a continuous PPO update moves the policy along the advantage", {
  # catches sign errors in the policy gradient without training in an env
  set.seed(9)
  net <- rltoolsR:::ppo_box_net(3L, 1L, 16L)
  opt <- optimizer_adam(net$parameters(), lr = 1e-2)
  B <- 64
  obs <- matrix(rnorm(3 * B), 3)
  logp_of <- function(act) {
    out <- net$forward(ag_tensor(obs))
    rltoolsR:::gauss_logp(act, as.matrix(out$mu), as.numeric(as.matrix(out$log_std)))
  }
  value_of <- function() as.numeric(as.matrix(net$forward(ag_tensor(obs))$value))
  out <- net$forward(ag_tensor(obs))
  act <- as.matrix(ag_sample_normal(as.matrix(out$mu), as.matrix(out$log_std)))
  old_logp <- logp_of(act)
  adv <- rep(c(1, -1), each = B / 2)
  ret <- rep(5, B)
  v0 <- mean(abs(value_of() - ret))
  for (i in 1:5) {
    rltoolsR:::ppo_update_box(net, opt, obs, act, old_logp, adv, ret,
                              clip = 0.2, vf_coef = 0.5, ent_coef = 0)
  }
  d <- logp_of(act) - old_logp
  expect_gt(mean(d[adv > 0]), 0)
  expect_lt(mean(d[adv < 0]), 0)
  expect_lt(mean(abs(value_of() - ret)), v0)
})
