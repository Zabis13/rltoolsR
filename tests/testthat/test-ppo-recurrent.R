# Recurrent PPO (R/ppo_recurrent.R): the update's sequence pass reproduces the
# collection's log-probabilities (ratio = 1), episode starts reset the state,
# truncation is bootstrapped from the state the episode reached, models
# save/load and act with their state.

# A memory task: the first observation of an episode shows a cue (+1 or -1),
# the next ones show nothing but the step; at the last step action 1 earns +1
# if the cue was +1 and action 0 earns +1 if it was -1 (-1 otherwise). An MLP
# sees the same observation at the last step whatever the cue: 50% at best.
# obs = (cue at the first step else 0, first step, last step). `truncate`:
# the episode ends by the step limit (truncated, final_obs) instead.
cue_env <- function(n, len = 4L, truncate = FALSE) {
  e <- new.env(parent = emptyenv())
  e$n <- as.integer(n); e$len <- as.integer(len); e$truncate <- truncate
  e$name <- "cue"; e$n_players <- 1L; e$max_steps <- as.integer(len)
  e$observation_space <- space_box(rep(-1, 3), rep(1, 3))
  e$action_space <- space_discrete(2L)
  class(e) <- "rl_test_cue"
  e
}
cue_obs <- function(e, t, cue) cbind(ifelse(t == 0L, cue, 0), (t == 0L) * 1,
                                     (t == e$len - 1L) * 1)
ns <- asNamespace("rltoolsR")
registerS3method("rl_reset", "rl_test_cue", function(env, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  env$t <- integer(env$n)
  env$cue <- sample(c(-1, 1), env$n, replace = TRUE)
  cue_obs(env, env$t, env$cue)
}, envir = ns)
registerS3method("rl_step", "rl_test_cue", function(env, action) {
  last <- env$t == env$len - 1L
  right <- (as.integer(action) == 1L) == (env$cue > 0)
  reward <- ifelse(last, ifelse(right, 1, -1), 0)
  env$t <- env$t + 1L
  fin <- cue_obs(env, env$t, env$cue)        # the observation after the step
  fin[!last, ] <- NA
  if (any(last)) {
    env$t[last] <- 0L
    env$cue[last] <- sample(c(-1, 1), sum(last), replace = TRUE)
  }
  list(obs = cue_obs(env, env$t, env$cue), reward = reward,
       terminated = if (env$truncate) logical(env$n) else last,
       truncated = if (env$truncate) last else logical(env$n), final_obs = fin)
}, envir = ns)
registerS3method("rl_legal_actions", "rl_test_cue", function(env) NULL, envir = ns)

rec_devices <- function() c("cpu", if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0L) "gpu")

# Two collections without an update in between: the second starts mid-episode
# from the state the first one left.
rec_rollouts <- function(dev, truncate = FALSE, seed = 3L) {
  set.seed(seed)
  ag_device(dev)
  ag_local_mode(graph = TRUE, matmul_precision = "f32")
  env <- cue_env(4L, len = 3L, truncate = truncate)
  algo <- rl_ppo_recurrent(n_steps = 10L, n_minibatches = 2L, hidden = 8L)
  state <- algo$init(env, dev)
  st <- algo$collect(state, env)$state
  r1 <- st$rollout
  st$rollout <- NULL
  st <- algo$collect(st, env)$state
  list(state = st, r1 = r1, r2 = st$rollout)
}

test_that("the update's sequence pass reproduces the collection (ratio = 1)", {
  on.exit(ag_device("cpu"), add = TRUE)
  for (dev in rec_devices()) {
    r <- rec_rollouts(dev)
    tol <- if (dev == "gpu") 1e-5 else 1e-10
    expect_gt(max(abs(r$r2$h0)), 0)               # the second rollout starts mid-episode
    for (ro in list(r$r1, r$r2))
      expect_equal(ppo_rec_rollout_logp(r$state$net, ro), ro$logp, tolerance = tol,
                   info = dev)
    # the reset is what makes them agree: without it the sequences differ
    ro <- r$r2
    ro$starts[] <- FALSE
    expect_false(isTRUE(all.equal(ppo_rec_rollout_logp(r$state$net, ro), r$r2$logp,
                                  tolerance = 1e-6)))
  }
})

test_that("starts mark the first observation of every episode", {
  ag_device("cpu")
  r <- rec_rollouts("cpu")
  # len 3, 10 steps: episodes start at steps 1, 4, 7, 10; the second rollout
  # continues at step 11 of the run = step 2 of an episode
  expect_identical(unname(r$r1$starts[1, ]), (0:9) %% 3L == 0L)
  expect_identical(unname(r$r2$starts[1, ]), (10:19) %% 3L == 0L)
  expect_true(all(r$r1$starts == r$r1$starts[rep(1L, 4L), ]))
})

test_that("a truncated episode is bootstrapped from the state it reached", {
  ag_device("cpu")
  r <- rec_rollouts("cpu", truncate = TRUE)
  ro <- r$r2
  net <- r$state$net
  expect_true(any(ro$trunc == 1))
  # replay copy j step by step from the rollout's initial state
  value_after <- function(j, t, zero = FALSE) {
    h <- ro$h0[, j, drop = FALSE]; c <- ro$c0[, j, drop = FALSE]
    for (s in seq_len(t)) {
      if (ro$starts[j, s]) { h[] <- 0; c[] <- 0 }
      o <- net$step(matrix(ro$obs[, j, s]), list(h = h, c = c))
      h <- as.matrix(o$state$h); c <- as.matrix(o$state$c)
    }
    if (zero) { h[] <- 0; c[] <- 0 }
    fin <- matrix(c(0, 0, 0))                    # cue_obs at t = len: all zero
    as.numeric(as.matrix(net$step(fin, list(h = h, c = c))$value))
  }
  for (k in which(ro$trunc == 1)) {
    j <- (k - 1L) %% 4L + 1L; t <- (k - 1L) %/% 4L + 1L
    v <- value_after(j, t)
    expect_equal(ro$next_val[j, t], v, tolerance = 1e-10)
    expect_false(isTRUE(all.equal(v, value_after(j, t, zero = TRUE), tolerance = 1e-8)))
  }
})

test_that("a recurrent model saves, loads and acts with its state", {
  ag_device("cpu")
  set.seed(12L)
  env <- cue_env(4L, len = 3L)
  m <- rl_train(env, rl_ppo_recurrent(n_steps = 6L, n_minibatches = 2L, hidden = 8L),
                total_steps = 4L * 6L, device = "cpu", verbose = FALSE)
  expect_identical(m$meta$net_arch, "lstm")
  f <- tempfile(fileext = ".rds"); on.exit(unlink(f), add = TRUE)
  rl_save_model(m, f)
  m2 <- rl_load_model(f, device = "cpu")
  obs <- rl_reset(env)
  a <- rl_act(m, obs); b <- rl_act(m2, obs)
  expect_identical(a$action, b$action)
  expect_equal(a$state, b$state)
  a2 <- rl_act(m, obs, state = a$state, starts = c(TRUE, FALSE, FALSE, FALSE))
  fresh <- rl_act(m, obs)
  expect_equal(a2$state$h[, 1], fresh$state$h[, 1])   # copy 1 reset
  expect_false(isTRUE(all.equal(a2$state$h[, 2], fresh$state$h[, 2])))
  expect_error(rl_act(m, obs, state = list(h = matrix(0, 3, 4), c = matrix(0, 3, 4))),
               "state must be")
  expect_error(rl_evaluate(m, cue_env(4L), n_episodes = 4L, greedy = FALSE), "greedily")
})

test_that("an LSTM file does not load as another network, and back", {
  ag_device("cpu")
  set.seed(13L)
  env <- cue_env(2L, len = 3L)
  m <- rl_train(env, rl_ppo_recurrent(n_steps = 3L, n_minibatches = 1L, hidden = 4L),
                total_steps = 6L, device = "cpu", verbose = FALSE)
  f <- tempfile(fileext = ".rds"); on.exit(unlink(f), add = TRUE)
  obj <- model_file(m, "test")
  obj$meta$net_arch <- "separate"                 # LSTM weights, MLP metadata
  saveRDS(obj, f)
  expect_error(rl_load_model(f), "do not match")
  obj$meta$net_arch <- "transformer"
  expect_error(ppo_build(obj$meta), "unknown PPO network architecture")
  # an MLP model is not taken for a recurrent one
  mlp <- rl_train(env, rl_ppo(n_steps = 3L, capture = FALSE), total_steps = 6L,
                  device = "cpu", verbose = FALSE)
  expect_true(is.integer(rl_act(mlp, rl_reset(env))))
  expect_error(rl_act(mlp, rl_reset(env), state = list()), "recurrent models only")
})

test_that("settings are checked", {
  expect_error(rl_ppo_recurrent(n_minibatches = 0), "positive whole")
  algo <- rl_ppo_recurrent(n_minibatches = 3L)
  expect_error(algo$init(cue_env(4L), "cpu"), "multiple of n_minibatches")
  box <- cue_env(4L)
  box$action_space <- space_box(-1, 1)
  expect_error(rl_ppo_recurrent(n_minibatches = 2L)$init(box, "cpu"), "discrete actions only")
})

# ---- captured update ---------------------------------------------------------

# Two identical networks trained on the same truncated rollout (episode starts
# mid-sequence, truncations), one with ppo_rec_update(), one with the captured
# update; clipping active (max_grad_norm small), and clip / ent_coef changed
# on the third step, so a second recording is made.
rec_update_pair <- function(dev) {
  r <- rec_rollouts(dev, truncate = TRUE)
  ag_local_mode(graph = TRUE, matmul_precision = "f32")     # as rl_train()
  ro <- r$r2
  meta <- r$state$meta
  net_a <- r$state$net
  set.seed(1L)
  net_b <- ppo_build(meta)
  ag_load_state_dict(net_b, ag_state_dict(net_a))
  opt_a <- optimizer_adam(net_a$parameters(), lr = 0.01)
  opt_b <- optimizer_adam(net_b$parameters(), lr = 0.01)
  caps <- new.env(parent = emptyenv())
  on.exit(ppo_free_captures(caps), add = TRUE)
  sets <- list(list(e = 1:2, clip = 0.2, ent = 0.01), list(e = 3:4, clip = 0.2, ent = 0.01),
               list(e = c(1L, 3L), clip = 0.1, ent = 0.05))
  la <- lb <- list()
  for (k in seq_along(sets)) {
    s <- sets[[k]]
    la[[k]] <- ppo_rec_update(net_a, opt_a, ro, s$e, s$clip, 0.5, s$ent, 0.05)
    lb[[k]] <- ppo_rec_update_captured(net_b, opt_b, caps, ro, s$e, s$clip, 0.5, s$ent, 0.05)
  }
  list(ro = ro, la = la, lb = lb, n_rec = length(ls(caps)),
       wa = lapply(net_a$parameters(), as.matrix), wb = lapply(net_b$parameters(), as.matrix))
}

test_that("the captured update follows the ordinary one (starts, truncation, clipping)", {
  on.exit(ag_device("cpu"), add = TRUE)
  for (dev in rec_devices()) {
    p <- rec_update_pair(dev)
    tol <- if (dev == "gpu") 1e-5 else 1e-10
    expect_true(any(p$ro$starts[, -1L]))             # episode starts mid-sequence
    expect_true(any(p$ro$trunc == 1))
    expect_true(all(vapply(p$la, function(l) l[["grad_norm"]] > 0.05, NA)))  # clipping on
    expect_identical(p$n_rec, 2L)                    # the changed clip / ent_coef: a 2nd recording
    for (k in seq_along(p$la))
      expect_equal(p$lb[[k]], p$la[[k]], tolerance = tol, info = paste(dev, k))
    for (nm in names(p$wa))
      expect_equal(p$wb[[nm]], p$wa[[nm]], tolerance = tol, info = paste(dev, nm))
  }
})

test_that("rl_ppo_recurrent trains with and without capture alike", {
  on.exit(ag_device("cpu"), add = TRUE)
  for (dev in rec_devices()) {
    run <- function(capture) {
      set.seed(21L)
      env <- cue_env(4L, len = 3L, truncate = TRUE)
      m <- rl_train(env, rl_ppo_recurrent(n_steps = 8L, n_minibatches = 2L, hidden = 8L,
                                          epochs = 2L, capture = capture),
                    total_steps = 4L * 8L * 3L, device = dev, verbose = FALSE)
      list(w = lapply(m$net$parameters(), as.matrix), h = m$history)
    }
    a <- run(FALSE); b <- run(TRUE)
    tol <- if (dev == "gpu") 1e-4 else 1e-10
    expect_equal(b$w, a$w, tolerance = tol, info = dev)
  }
})
