# rollout = "resident" (R/ppo_resident.R): the policy and tic-tac-toe in one
# recorded graph per step. Checked against a host loop -- rl_gpu_env on the CPU,
# policy_act() with the same keys (k1, k2 + t) -- over two rollouts in a row
# (the second continues the games), then end to end through rl_ppo().

has_vk_rr <- function() ggml_vulkan_available() && ggml_vulkan_device_count() > 0L
# cartpole / pendulum adapters need ag_sin / ag_cos, absent in older ggmlR
has_trig <- function() length(rltoolsR:::ggmlr_missing(rltoolsR:::RES_TRIG)) == 0L

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
  ad <- rltoolsR:::ppo_res_adapter_ttt()
  rltoolsR:::ppo_res_setup(net, cap, ad, n, ns, 0.99)
  on.exit(rltoolsR:::ppo_free_captures(cap), add = TRUE)

  ref <- rl_gpu_env("tictactoe", n = n, device = "cpu")
  obs <- rl_reset(ref)
  legal <- rl_legal_actions(ref)
  for (round in 1:2) {
    key <- c(stats::runif(1), floor(stats::runif(1) * 4096) / 4096)
    r <- rltoolsR:::ppo_res_rollout(cap, ad, n, ns, key)
    expect_identical(r$trunc, matrix(0, n, ns))        # tic-tac-toe never truncates
    expect_identical(r$next_val, matrix(0, n, ns))
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

# Tabular adapter (R/resident_tabular.R): frozen_lake_det_onehot against rl_env
# singles and policy_act() with the same keys. Every game starts 95 steps into
# its episode, so all games still running reach the 100-step limit within the
# rollout: the truncation rows (trunc, next_val = V(final obs)) are checked too.
test_that("the resident tabular rollout equals a host loop, truncation included", {
  skip_if_not(has_vk_rr(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(5L)
  nm <- "frozen_lake_det_onehot"
  n <- 16L; ns <- 9L; gamma <- 0.99
  ag_device("gpu")
  ag_local_mode(graph = TRUE, matmul_precision = "f32")
  env <- rl_vec_env(nm, n = n, seed = 1)
  algo <- rl_ppo(n_steps = ns)
  state <- algo$init(env, "gpu")
  net <- state$net
  ad <- rltoolsR:::ppo_res_adapter(env, FALSE, FALSE, TRUE, state$meta, "gpu")
  expect_false(is.null(ad))
  cap <- new.env(parent = emptyenv())
  rs <- rltoolsR:::ppo_res_setup(net, cap, ad, n, ns, gamma)
  on.exit(rltoolsR:::ppo_free_captures(cap), add = TRUE)
  ag_buffer_write(rs$steps, matrix(95, 1, n), col_offset = 0)

  singles <- lapply(seq_len(n), function(i) {
    e <- rl_env(nm, seed = i)
    rl_reset(e)
    # snapshot = step counter (int) + state (int): 95 steps taken, at the start
    rl_load_state(e, c(writeBin(95L, raw()), writeBin(0L, raw())))
    e
  })
  obs <- matrix(0, 16, n); obs[1, ] <- 1
  key <- c(stats::runif(1), floor(stats::runif(1) * 4096) / 4096)
  r <- rltoolsR:::ppo_res_rollout(cap, ad, n, ns, key)
  h <- list(act = NULL, rew = NULL, done = NULL, trunc = NULL, next_val = NULL)
  for (t in seq_len(ns)) {
    o <- rltoolsR:::policy_act(net, obs, NULL, matrix(1, 4, n), key = c(key[1], key[2] + t - 1))
    s <- lapply(seq_len(n), function(i) rl_step(singles[[i]], o$a[i]))
    fin <- vapply(s, function(x) x$obs, numeric(16))
    term <- vapply(s, `[[`, NA, "terminated")
    trunc <- vapply(s, `[[`, NA, "truncated") & !term
    done <- trunc | term
    v_fin <- as.numeric(rltoolsR:::policy_forward(net, fin)$value)
    h$act <- cbind(h$act, o$a)
    h$rew <- cbind(h$rew, vapply(s, `[[`, 0, "reward"))
    h$done <- cbind(h$done, done * 1)
    h$trunc <- cbind(h$trunc, trunc * 1)
    h$next_val <- cbind(h$next_val, trunc * v_fin)
    for (i in which(done)) fin[, i] <- rl_reset(singles[[i]])
    obs <- fin
  }
  expect_identical(r$act, h$act + 0)
  expect_identical(r$rew, h$rew)
  expect_identical(r$done, h$done)
  expect_identical(r$trunc, h$trunc)
  expect_equal(r$next_val, h$next_val, tolerance = 1e-5)
  expect_gt(sum(h$trunc), 0)                           # truncations happened
  expect_identical(r$legal, matrix(1, 4, n * ns))
})

# Short training through rl_train(): frozen_lake_det_onehot is solved (return 1)
test_that("rl_ppo learns frozen_lake_det on the resident rollout", {
  skip_if_not(has_vk_rr(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(6L)
  env <- rl_vec_env("frozen_lake_det_onehot", n = 64, seed = 1)
  algo <- rl_ppo(n_steps = 32, minibatch = 512, lr = 3e-3, ent_coef = 0.01)
  m <- rl_train(env, algo, total_steps = 64 * 32 * 60, device = "gpu", verbose = FALSE)
  res <- rl_evaluate(m, "frozen_lake_det_onehot", n_episodes = 4)
  expect_equal(res[["return_mean"]], 1)
})

# A resumed state (set_train_state frees the recordings) rebuilds the adapter
# and the recording on its next collect and keeps the resident path.
test_that("the resident rollout survives set_train_state (resume)", {
  skip_if_not(has_vk_rr(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(7L)
  env <- rl_gpu_env("tictactoe", n = 64, device = "gpu")
  ag_device("gpu")
  algo <- rl_ppo(n_steps = 9, minibatch = 576)
  state <- algo$init(env, "gpu")
  state <- algo$update(algo$collect(state, env)$state)$state
  expect_identical(state$obs_path, "resident")
  ts <- algo$get_train_state(state)
  st2 <- algo$set_train_state(algo$init(env, "gpu"), ts)
  expect_null(st2$captures$res_step)
  col <- algo$collect(st2, env)
  expect_identical(col$state$obs_path, "resident")
  expect_identical(col$steps, 576)
  expect_false(is.null(col$state$captures$res_step))
  m <- algo$update(col$state)$metrics
  expect_true(all(is.finite(m)))
})

# normalize_reward on the resident path: the return tracker spans the games,
# the reward statistics count every step of every rollout, and the scaled
# advantages stay finite.
test_that("normalize_reward works on the resident rollout", {
  skip_if_not(has_vk_rr(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(8L)
  n <- 16L; ns <- 32L
  env <- rl_vec_env("frozen_lake_det_onehot", n = n, seed = 1)
  ag_device("gpu")
  algo <- rl_ppo(n_steps = ns, minibatch = n * ns, normalize_reward = TRUE)
  state <- algo$init(env, "gpu")
  for (i in 1:2) {
    col <- algo$collect(state, env)
    state <- col$state
    expect_identical(state$obs_path, "resident")
    expect_true(all(is.finite(state$rollout$adv)))
  }
  expect_length(state$ret_track, n)
  expect_identical(state$rew_rms$count, 2 * n * ns)
  m <- algo$update(state)$metrics
  expect_true(all(is.finite(m)))
})

# A different env$n between collects rebuilds the recording, its buffers and
# the return tracker instead of reusing the ones sized for the old n.
test_that("the resident rollout follows a change of env$n", {
  skip_if_not(has_vk_rr(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(9L)
  ag_device("gpu")
  algo <- rl_ppo(n_steps = 9, minibatch = 288, normalize_reward = TRUE)
  env64 <- rl_gpu_env("tictactoe", n = 64, device = "gpu")
  env32 <- rl_gpu_env("tictactoe", n = 32, device = "gpu")
  state <- algo$init(env64, "gpu")
  for (e in list(env64, env32, env64)) {
    col <- algo$collect(state, e)
    state <- col$state
    expect_identical(state$obs_path, "resident")
    expect_identical(col$steps, e$n * 9)
    expect_length(state$rollout$logp, e$n * 9)
    expect_length(state$ret_track, e$n)
    expect_identical(state$res_n, e$n)
    up <- algo$update(state)
    expect_true(all(is.finite(up$metrics)))
    state <- up$state
  }
})

# normalize_obs on the resident path (cartpole): the rollout's moments from the
# device equal running_stats_update() on the downloaded obs_buf, obs_buf keeps
# the raw observations, the policy saw the normalized ones (frozen statistics
# of the rollout), and the statistics count every step of every rollout.
test_that("normalize_obs works on the resident rollout", {
  skip_if_not(has_vk_rr(), "no Vulkan device")
  skip_if_not(has_trig(), "ggmlR without ag_sin / ag_cos")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(10L)
  n <- 32L; ns <- 16L; N <- n * ns
  env <- rl_gpu_env("cartpole", n = n, device = "gpu")
  ag_device("gpu")
  algo <- rl_ppo(n_steps = ns, minibatch = N, normalize_obs = TRUE)
  state <- algo$init(env, "gpu")
  for (i in 1:3) {
    rms_before <- state$obs_rms
    col <- algo$collect(state, env)
    state <- col$state
    info <- paste("rollout", i)
    expect_identical(state$obs_path, "resident", info = info)
    ro <- state$rollout
    expect_false(is.null(ro$norm), info = info)
    expect_equal(ro$norm$mean, rms_before$mean, info = info)   # frozen during the rollout
    raw <- ag_buffer_read(state$captures$obs_buf)
    expect_true(max(abs(raw[3, ])) < 0.25, info = info)         # raw theta, not normalized
    ref <- rltoolsR:::running_stats_update(rms_before, raw)
    expect_equal(state$obs_rms$mean, ref$mean, tolerance = 1e-5, info = info)
    expect_equal(state$obs_rms$var, ref$var, tolerance = 1e-4, info = info)
    expect_equal(state$obs_rms$count, i * N, info = info)
    # the values of the first step: the network on the normalized observations
    v <- as.numeric(rltoolsR:::policy_forward(state$net, raw[, seq_len(n)], ro$norm)$value)
    expect_equal(ro$ret[seq_len(n)] - ro$adv[seq_len(n)], v, tolerance = 1e-4, info = info)
    up <- algo$update(state)
    expect_true(all(is.finite(up$metrics)), info = info)
    state <- up$state
  }
  expect_gt(max(abs(state$obs_rms$mean)), 0)                   # the statistics moved
})

# Box actions on the resident path (pendulum, normalize_obs and
# normalize_reward on by default for box): unclipped actions [act_dim, N],
# logp equal to gauss_logp() of the network on the normalized observations
# (the host adds the constant of the density), no legal mask, finite update.
test_that("box actions work on the resident rollout (pendulum)", {
  skip_if_not(has_vk_rr(), "no Vulkan device")
  skip_if_not(has_trig(), "ggmlR without ag_sin / ag_cos")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(11L)
  n <- 32L; ns <- 16L; N <- n * ns
  env <- rl_gpu_env("pendulum_gym", n = n, device = "gpu")
  ag_device("gpu")
  algo <- rl_ppo(n_steps = ns, minibatch = N)
  state <- algo$init(env, "gpu")
  expect_true(state$meta$normalize_obs)
  clipped <- 0
  for (i in 1:2) {
    col <- algo$collect(state, env)
    state <- col$state
    info <- paste("rollout", i)
    expect_identical(state$obs_path, "resident", info = info)
    ro <- state$rollout
    expect_identical(dim(ro$act), c(1L, N), info = info)
    expect_null(ro$legal, info = info)
    clipped <- clipped + sum(abs(ro$act) > 1)                   # stored unclipped
    raw <- ag_buffer_read(state$captures$obs_buf)
    cols <- seq_len(n)                                          # the first step
    out <- rltoolsR:::policy_forward(state$net, raw[, cols], ro$norm)
    lp <- rltoolsR:::gauss_logp(ro$act[, cols, drop = FALSE], out$mu, as.numeric(out$log_std))
    expect_equal(ro$logp[cols], lp, tolerance = 1e-4, info = info)
    expect_equal(ro$ret[cols] - ro$adv[cols], as.numeric(out$value), tolerance = 1e-4,
                 info = info)
    up <- algo$update(state)
    expect_true(all(is.finite(up$metrics)), info = info)
    state <- up$state
  }
  expect_gt(clipped, 0)
})

# An older ggmlR (ggmlr_missing() mocked): "auto" falls back to the host
# path, rollout = "resident" names the missing functions.
test_that("without ag_sin / ag_cos the trig adapters fall back or name what is missing", {
  local_mocked_bindings(ggmlr_missing = function(fns) fns)
  meta <- list(normalize_obs = FALSE)
  v <- rl_vec_env("cartpole", n = 2)
  expect_error(rltoolsR:::ppo_res_adapter(v, FALSE, FALSE, TRUE, meta, "gpu", "resident"),
               "ag_sin\\(\\), ag_cos\\(\\) from ggmlR")
  expect_error(rltoolsR:::cart_res_new(2), "ag_sin")
  # the other adapters do not need them
  expect_false(is.null(rltoolsR:::ppo_res_adapter(rl_vec_env("frozen_lake", n = 2, seed = 1),
                                                  FALSE, FALSE, TRUE, meta, "gpu")))
  skip_if_not(has_vk_rr(), "no Vulkan device")
  g <- rl_gpu_env("cartpole", n = 2, device = "gpu")
  expect_null(rltoolsR:::ppo_res_adapter(g, FALSE, FALSE, TRUE, meta, "gpu", "auto"))
})

test_that("the resident adapter follows the policy's action type", {
  skip_if_not(has_trig(), "ggmlR without ag_sin / ag_cos")
  meta <- list(normalize_obs = FALSE)
  adapter <- function(env, box, rollout = "auto")
    rltoolsR:::ppo_res_adapter(env, box, FALSE, TRUE, meta, "gpu", rollout)
  v <- rl_vec_env("pendulum_gym", n = 2)
  expect_null(adapter(v, TRUE))
  expect_true(adapter(v, TRUE, "resident")$box)
  expect_null(adapter(v, FALSE, "resident"))                   # box env, discrete policy
  expect_null(adapter(rl_vec_env("cartpole", n = 2), TRUE, "resident"))
  skip_if_not(has_vk_rr(), "no Vulkan device")
  g <- rl_gpu_env("pendulum_gym_raw", n = 2, device = "gpu")
  expect_true(adapter(g, TRUE)$box)
})

# Fictitious self-play on the resident path: against a host loop with the same
# keys, where the opponent moves when vs_pool and the move's parity differs
# from the learner's side (side 1: the opponent opens every game). vs_pool /
# side of each step come from the record (the device redraws them at the end
# of a game): they stay fixed within a game, and the redraws follow p_self.
test_that("the resident fsp rollout equals a host loop on the same keys", {
  skip_if_not(has_vk_rr(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(12L)
  n <- 256L; ns <- 9L; p_self <- 0.5
  ag_device("gpu")
  ag_local_mode(graph = TRUE, matmul_precision = "f32")
  env <- rl_gpu_env("tictactoe", n = n, device = "gpu")
  net <- rl_ppo(n_steps = ns)$init(env, "gpu")$net
  opp_net <- rl_ppo(n_steps = ns)$init(env, "gpu")$net         # other weights
  vs0 <- rep(c(TRUE, FALSE), length.out = n); side0 <- rep(c(0L, 0L, 1L, 1L), length.out = n)
  cap <- new.env(parent = emptyenv())
  ad <- rltoolsR:::ppo_res_adapter_ttt()
  rltoolsR:::ppo_res_setup(net, cap, ad, n, ns, 0.99,
                           opp = list(net = opp_net, p_self = p_self, vs_pool = vs0, side = side0))
  on.exit(rltoolsR:::ppo_free_captures(cap), add = TRUE)
  key <- c(stats::runif(1), floor(stats::runif(1) * 4096) / 4096)
  r <- rltoolsR:::ppo_res_rollout(cap, ad, n, ns, key)

  ref <- rl_gpu_env("tictactoe", n = n, device = "cpu")
  obs <- rl_reset(ref); legal <- rl_legal_actions(ref)
  ply <- integer(n)
  h <- list(act = NULL, opp = NULL, val = NULL)
  for (t in seq_len(ns)) {
    x <- t(obs); lg <- t(legal) * 1
    k <- c(key[1], key[2] + t - 1)
    o1 <- rltoolsR:::policy_act(net, x, NULL, lg, key = k)
    o2 <- rltoolsR:::policy_act(opp_net, x, NULL, lg, key = k)
    opp <- r$vs_pool[, t] > 0.5 & (ply %% 2L) != r$side[, t]
    a <- ifelse(opp, o2$a, o1$a)
    s <- rl_step(ref, a)
    ply <- ifelse(s$terminated, 0L, ply + 1L)
    h$act <- cbind(h$act, a); h$opp <- cbind(h$opp, opp); h$val <- cbind(h$val, o1$value)
    obs <- s$obs; legal <- s$legal
  }
  expect_identical(r$vs_pool[, 1] > 0.5, vs0)
  expect_identical(r$side[, 1], side0 + 0)
  expect_identical(r$act, unname(h$act) + 0)
  expect_identical(r$keep, 1 - unname(h$opp))
  expect_equal(r$val, h$val, tolerance = 1e-6)                # always the learner's value
  expect_true(all(h$opp[side0 == 1L & vs0, 1]))               # side 1: the opponent opens
  # vs_pool / side change only after a game ended
  done <- r$done[, -ns] > 0.5
  same <- r$vs_pool[, -1] == r$vs_pool[, -ns] & r$side[, -1] == r$side[, -ns]
  expect_true(all(same[!done]))
  redraw_vs <- r$vs_pool[, -1][done]; redraw_side <- r$side[, -1][done]
  expect_gt(length(redraw_vs), 100)
  # absolute: > 3 binomial sd for the ~200 redraws
  expect_lt(abs(mean(redraw_vs) - (1 - p_self)), 0.12)
  expect_lt(abs(mean(redraw_side) - 0.5), 0.12)
})

test_that("rl_ppo takes the resident rollout with a pool and drops the snapshot's moves", {
  skip_if_not(has_vk_rr(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(13L)
  env <- rl_gpu_env("tictactoe", n = 64, device = "gpu")
  ag_device("gpu")
  algo <- rl_ppo(n_steps = 9, minibatch = 128, pool = list(p_self = 0.5, every = 1))
  state <- algo$init(env, "gpu")
  for (i in 1:2) {
    col <- algo$collect(state, env)
    state <- col$state
    expect_identical(state$obs_path, "resident")
    used <- col$info[["used"]]
    expect_lt(used, 576)                                       # the opponent's moves dropped
    expect_equal(used %% 128, 0)
    expect_gt(col$info[["pool_games"]], 0)
    up <- algo$update(state)
    expect_true(all(is.finite(up$metrics)))
    state <- up$state
  }
  expect_length(state$pool, 2L)
  # p_self = 1: no opponent in the recording, every move kept
  algo1 <- rl_ppo(n_steps = 9, minibatch = 64, pool = list(p_self = 1))
  col <- algo1$collect(algo1$init(env, "gpu"), env)
  expect_identical(col$state$obs_path, "resident")
  expect_null(col$state$captures$res_vs)
  expect_equal(col$info[["pool_games"]], 0)
  expect_equal(col$info[["used"]], 576)
})
