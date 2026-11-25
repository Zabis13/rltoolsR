# Recurrent PPO: an LSTM policy for partially observed environments.
#
# Network (meta$net_arch = "lstm"): observation -> tanh encoder -> one LSTM
# shared by the policy and the value -> linear heads (CleanRL's
# ppo_atari_lstm layout). Stable-Baselines3's RecurrentPPO uses separate LSTMs
# for the actor and the critic by default, so learning curves are not directly
# comparable with it.
#
# Rollout (CleanRL's scheme): the state (h, c) of every environment at the
# start of the rollout, and per step `starts` -- TRUE where an episode begins
# at that observation (the done of the previous step under automatic resets).
# The update re-runs the LSTM over whole rollouts of a subset of the
# environments, zeroing the state where `starts` is TRUE; no padding, and the
# first pass of the first epoch reproduces the collection's log-probabilities.
#
# Truncated episodes are bootstrapped with V(final_obs) computed from the state
# BEFORE the reset (the state the episode actually reached), not from zeros.
#
# Collection runs the R loop (the host path of rl_ppo); on the GPU each step is
# one recorded graph (ggmlR::ag_capture) taking the observations and (h, c) and
# returning the action, its log-probability, the value and the new state.

#' Recurrent PPO (LSTM policy)
#'
#' PPO with an LSTM between an encoder and the policy and value heads, for
#' environments where the current observation does not tell the whole state
#' (partial observability, memory tasks). Use with [rl_train()].
#'
#' The network is `obs -> tanh layer (hidden) -> LSTM (hidden) -> heads`, one
#' LSTM shared by policy and value (as CleanRL's `ppo_atari_lstm`).
#' Stable-Baselines3's `RecurrentPPO` uses separate LSTMs for the actor and the
#' critic by default, so learning curves are not directly comparable with it.
#'
#' Minibatches are subsets of the environment copies, each with all
#' `n_steps` steps: `env$n` must be a multiple of `n_minibatches`. The state
#' of every copy at the start of the rollout is kept and the update re-runs
#' the LSTM over the rollout, resetting the state where an episode starts.
#' A truncated episode is bootstrapped with the value of its final observation
#' computed from the state the episode reached.
#'
#' Supported: discrete actions (with legal-action masks), the host collection
#' loop, observation and reward normalization, captured updates. Not supported
#' (yet): continuous actions, the device/resident collection paths, fictitious
#' self-play (`pool`).
#'
#' With `capture = TRUE` (the default) a minibatch update -- the forward over
#' the whole sequences, the backward and Adam -- is one recording
#' ([ggmlR::ag_capture_step()]) replayed for every minibatch: rollouts, episode
#' starts, the initial states and the normalization statistics are its inputs.
#' It needs ggmlR >= 0.8.9.4 (episode starts as a tensor in
#' [ggmlR::ag_lstm()]); with an older ggmlR the default falls back to the
#' ordinary update without a word, and an explicit `capture = TRUE` says so
#' once.
#'
#' A trained model acts with [rl_act()] and its `state` argument; see there.
#'
#' @inheritParams rl_ppo
#' @param n_minibatches Number of minibatches per epoch; each holds
#'   `env$n / n_minibatches` environment copies with all their steps.
#' @param hidden Width of the encoder and of the LSTM state.
#' @param lstm_init Initialisation of the LSTM, see [ggmlR::ag_lstm()]:
#'   `"rl"` (orthogonal recurrent weights, forget bias 1) or `"torch"`.
#' @param capture Record the minibatch update once and replay it (see
#'   Details); `FALSE` builds the tape every minibatch.
#' @return An `rl_algo` object for [rl_train()].
#' @export
#' @examples
#' \donttest{
#' env <- rl_vec_env("cartpole", n = 8)
#' m <- rl_train(env, rl_ppo_recurrent(n_steps = 32, n_minibatches = 2),
#'               total_steps = 8 * 32 * 2, device = "cpu", verbose = FALSE)
#' obs <- rl_reset(env)
#' r <- rl_act(m, obs)                 # list(action, state)
#' r <- rl_act(m, obs, state = r$state)
#' }
rl_ppo_recurrent <- function(n_steps = 128, epochs = 4, n_minibatches = 4, lr = 2.5e-4,
                             gamma = 0.99, lambda = 0.95, clip = 0.2, vf_coef = 0.5,
                             ent_coef = 0.01, hidden = 64, normalize_obs = FALSE,
                             normalize_reward = FALSE, clip_obs = 10, clip_reward = 10,
                             epsilon = 1e-8, max_grad_norm = 0.5,
                             lstm_init = c("rl", "torch"), capture = TRUE) {
  ggmlr_need(c("ag_lstm", "ag_lstm_cell"), "rl_ppo_recurrent()")
  lstm_init <- match.arg(lstm_init)
  # episode starts as a tensor in ag_lstm (inputs of the recording): ggmlR 0.8.9.4
  if (isTRUE(capture) && utils::packageVersion("ggmlR") < "0.8.9.4") {
    if (!missing(capture))
      message("rl_ppo_recurrent(): captured updates need ggmlR >= 0.8.9.4 (installed ",
              format(utils::packageVersion("ggmlR")), "); using capture = FALSE")
    capture <- FALSE
  }
  capture <- isTRUE(capture)
  whole <- function(v, nm) {
    if (!is.numeric(v) || length(v) != 1L || is.na(v) || v < 1 || v != round(v))
      stop(sprintf("rl_ppo_recurrent(): %s must be a positive whole number", nm), call. = FALSE)
    as.integer(v)
  }
  n_steps <- whole(n_steps, "n_steps")
  epochs <- whole(epochs, "epochs")
  n_minibatches <- whole(n_minibatches, "n_minibatches")
  hidden <- whole(hidden, "hidden")
  if (!is.null(max_grad_norm) &&
      (!is.numeric(max_grad_norm) || length(max_grad_norm) != 1L ||
       !is.finite(max_grad_norm) || max_grad_norm <= 0))
    stop("max_grad_norm must be NULL or one positive finite number", call. = FALSE)
  normalize_obs <- isTRUE(normalize_obs)
  normalize_reward <- isTRUE(normalize_reward)

  legal_of <- function(env, legal, n_act) {
    if (is.null(legal)) legal <- rl_legal_actions(env)
    if (is.null(legal)) matrix(TRUE, env$n, n_act) else legal
  }
  obs_norm <- function(state) {
    if (state$meta$normalize_obs) obs_norm_meta(state$obs_rms, epsilon, clip_obs)
  }

  init <- function(env, device) {
    sp <- env$action_space
    if (!identical(sp$type, "discrete"))
      stop("rl_ppo_recurrent() supports discrete actions only", call. = FALSE)
    if (env$n %% n_minibatches != 0L)
      stop(sprintf(paste0("rl_ppo_recurrent(): minibatches are subsets of the environment ",
                          "copies, so env$n (%d) must be a multiple of n_minibatches (%d)"),
                   env$n, n_minibatches), call. = FALSE)
    obs_dim <- space_size(env$observation_space)
    meta <- list(algo = "ppo", action_type = "discrete", obs_dim = obs_dim,
                 hidden = hidden, observation_space = env$observation_space,
                 action_space = sp, n_actions = sp$n, net_arch = "lstm",
                 lstm_init = lstm_init, normalize_obs = normalize_obs,
                 normalize_reward = normalize_reward)
    net <- ppo_build(meta)
    list(net = net, opt = optimizer_adam(net$parameters(), lr = lr),
         meta = meta, device = device,
         turn_sign = if (identical(env$opponent, "self")) -1 else 1,
         obs_rms = running_stats(obs_dim), rew_rms = running_stats(1L),
         obs = NULL, legal = NULL, ret_track = NULL, rollout = NULL, updates = 0,
         h = NULL, c = NULL, starts = NULL, captures = new.env(parent = emptyenv()))
  }

  collect <- function(state, env) {
    net <- state$net
    meta <- state$meta
    n_env <- env$n
    obs_dim <- meta$obs_dim
    n_act <- meta$n_actions
    H <- meta$hidden
    if (is.null(state[["obs"]])) {
      state$obs <- rl_reset(env)
      state$legal <- legal_of(env, NULL, n_act)
      state$ret_track <- numeric(n_env)
      state$h <- matrix(0, H, n_env)
      state$c <- matrix(0, H, n_env)
      state$starts <- rep(TRUE, n_env)
    }
    norm <- obs_norm(state)
    rew_rms <- state$rew_rms
    obs <- state[["obs"]]
    legal <- state$legal
    h <- state$h
    c <- state$c
    starts <- state$starts
    # the state the rollout starts from, before its first `starts` is applied
    h0 <- h
    c0 <- c

    n_t <- n_steps
    b_obs <- array(0, c(obs_dim, n_env, n_t))
    b_legal <- array(0, c(n_act, n_env, n_t))
    b_act <- matrix(0L, n_env, n_t)
    b_starts <- matrix(FALSE, n_env, n_t)
    b_logp <- b_val <- b_rew <- b_term <- b_trunc <- b_nv <- b_ret <- matrix(0, n_env, n_t)
    n_done <- 0
    for (t in seq_len(n_t)) {
      x <- t(obs_host(obs))
      lg <- t(legal) * 1
      b_starts[, t] <- starts
      h[, starts] <- 0
      c[, starts] <- 0
      out <- ppo_rec_act(net, x, h, c, norm, lg, key = stats::runif(2L))
      r <- rl_step(env, out$a)
      b_obs[, , t] <- x
      b_legal[, , t] <- lg
      b_act[, t] <- out$a
      b_logp[, t] <- out$logp
      b_val[, t] <- out$value
      h <- out$h
      c <- out$c
      done <- r$terminated | r$truncated
      rew <- r$reward
      if (meta$normalize_reward) {
        state$ret_track <- state$ret_track * gamma + rew
        b_ret[, t] <- state$ret_track
        state$ret_track[done] <- 0
        rew <- scale_reward(rew, rew_rms, epsilon, clip_reward)
      }
      b_rew[, t] <- rew
      trunc <- r$truncated & !r$terminated
      if (any(trunc) && !is.null(r$final_obs)) {
        if (anyNA(r$final_obs[trunc, ]))
          stop("environment returned NA final_obs for a truncated copy", call. = FALSE)
        # V(final_obs) from the state the episode reached (h, c after this
        # step, before the reset); all n_env columns, one recorded shape
        fin <- matrix(0, obs_dim, n_env)
        fin[, trunc] <- t(r$final_obs[trunc, , drop = FALSE])
        b_nv[trunc, t] <- ppo_rec_value(net, fin, h, c, norm)[trunc]
        b_trunc[trunc, t] <- 1
      }
      b_term[, t] <- as.numeric(done) - b_trunc[, t]
      n_done <- n_done + sum(done)
      starts <- done
      obs <- r$obs
      legal <- legal_of(env, r$legal, n_act)
    }
    hl <- h; cl <- c
    hl[, starts] <- 0
    cl[, starts] <- 0
    last_val <- ppo_rec_value(net, t(obs_host(obs)), hl, cl, norm)

    adv <- ppo_advantages(b_rew, b_val, b_term, b_trunc, b_nv, last_val, gamma, lambda,
                          state$turn_sign)
    state$rollout <- list(obs = b_obs, legal = b_legal, act = b_act, starts = b_starts,
                          h0 = h0, c0 = c0, logp = b_logp, adv = adv, ret = adv + b_val,
                          trunc = b_trunc, next_val = b_nv, norm = norm)
    if (meta$normalize_obs)
      state$obs_rms <- running_stats_update(state$obs_rms, matrix(b_obs, obs_dim))
    if (meta$normalize_reward)
      state$rew_rms <- running_stats_update(state$rew_rms, matrix(b_ret, 1L))
    state$obs <- obs
    state$legal <- legal
    state$h <- h
    state$c <- c
    state$starts <- starts
    list(state = state, steps = n_env * n_t, info = c(episodes = n_done))
  }

  update <- function(state) {
    ro <- state$rollout
    n_env <- ncol(ro$h0)
    per <- n_env %/% n_minibatches
    losses <- NULL
    n_mb <- 0
    for (ep in seq_len(epochs)) {
      perm <- sample.int(n_env)
      for (k in seq_len(n_minibatches)) {
        e <- sort(perm[(k - 1L) * per + seq_len(per)])
        l <- if (capture)
          ppo_rec_update_captured(state$net, state$opt, state$captures, ro, e, clip,
                                  vf_coef, ent_coef, max_grad_norm)
        else ppo_rec_update(state$net, state$opt, ro, e, clip, vf_coef, ent_coef,
                            max_grad_norm)
        losses <- if (is.null(losses)) l else losses + l
        n_mb <- n_mb + 1
      }
    }
    state$rollout <- NULL
    state$updates <- state$updates + n_mb
    list(state = state, metrics = losses / n_mb)
  }

  model <- function(state) {
    meta <- state$meta
    if (meta$normalize_obs) meta$obs_norm <- obs_norm(state)
    structure(list(net = state$net, device = state$device, meta = meta, history = NULL),
              class = "rl_ppo")
  }

  get_train_state <- function(state) {
    list(weights = save_net(state$net), optimizer_state = state$opt$state(),
         normalizers = list(obs = state$obs_rms, reward = state$rew_rms),
         counters = list(updates = state$updates))
  }

  set_train_state <- function(state, train_state) {
    net <- load_net(train_state$weights, function() ppo_build(state$meta), state$device)
    state$net <- net
    state$opt <- optimizer_adam(net$parameters(), lr = lr)
    # recordings reference the old network and optimizer: never carried over
    state$captures <- ppo_free_captures(state$captures)
    if (!is.null(train_state$optimizer_state)) {
      withCallingHandlers(
        state$opt$load_state(train_state$optimizer_state),
        ggmlR_opt_hyper_warning = function(w) invokeRestart("muffleWarning"))
    }
    state$obs_rms <- train_state$normalizers$obs
    state$rew_rms <- train_state$normalizers$reward
    state$updates <- train_state$counters$updates
    # fresh episodes after a resume: environments and LSTM states start anew
    state[c("obs", "legal", "ret_track", "rollout", "h", "c", "starts")] <- list(NULL)
    state
  }

  rl_algo("ppo_recurrent", init = init, collect = collect, update = update, model = model,
          get_train_state = get_train_state, set_train_state = set_train_state,
          params = list(n_steps = n_steps, epochs = epochs, n_minibatches = n_minibatches,
                        lr = lr, gamma = gamma, lambda = lambda, clip = clip,
                        vf_coef = vf_coef, ent_coef = ent_coef, hidden = hidden,
                        normalize_obs = normalize_obs, normalize_reward = normalize_reward,
                        clip_obs = clip_obs, clip_reward = clip_reward, epsilon = epsilon,
                        max_grad_norm = max_grad_norm, lstm_init = lstm_init,
                        capture = capture))
}

# ---- network ----------------------------------------------------------------

ppo_lstm_net <- function(obs_dim, n_actions, hidden, init = "rl") {
  enc  <- ag_linear(obs_dim, hidden, "tanh", init = "orthogonal", gain = sqrt(2))
  lstm <- ag_lstm(hidden, hidden, init = init)
  pi   <- ag_linear(hidden, n_actions, init = "orthogonal", gain = 0.01)
  v    <- ag_linear(hidden, 1L, init = "orthogonal", gain = 1)
  # one step: x [obs_dim, B] (already normalized), state list(h, c) [hidden, B]
  step <- function(x, state) {
    s <- lstm$cell$forward(enc$forward(x), state)
    list(logits = pi$forward(s$h), value = v$forward(s$h), state = s)
  }
  # a whole rollout: xs a list of T inputs [obs_dim, B], starts [T, B];
  # logits and values as [., T * B], column (t - 1) * B + j
  sequence <- function(xs, state, starts) {
    r <- lstm$forward(lapply(xs, enc$forward), state, starts)
    hs <- ppo_rec_cat_cols(r$outputs)
    list(logits = pi$forward(hs), value = v$forward(hs), state = r$state)
  }
  parameters <- function() list(
    enc_W = enc$W, enc_b = enc$b,
    lstm_weight_ih = lstm$weight_ih, lstm_weight_hh = lstm$weight_hh,
    lstm_bias_ih = lstm$bias_ih, lstm_bias_hh = lstm$bias_hh,
    pi_W = pi$W, pi_b = pi$b, v_W = v$W, v_b = v$b)
  structure(list(step = step, sequence = sequence, parameters = parameters,
                 hidden = hidden, captured = new.env(parent = emptyenv())),
            class = c("rl_ppo_net", "ag_layer"))
}

ppo_lstm_shapes <- function(meta) {
  h <- meta$hidden; o <- meta$obs_dim; a <- meta$n_actions
  list(enc_W = c(h, o), enc_b = c(h, 1L),
       lstm_weight_ih = c(4L * h, h), lstm_weight_hh = c(4L * h, h),
       lstm_bias_ih = c(4L * h, 1L), lstm_bias_hh = c(4L * h, 1L),
       pi_W = c(a, h), pi_b = c(a, 1L), v_W = c(1L, h), v_b = c(1L, 1L))
}

# cbind of a list of [n, B] tensors as ag ops: transpose, rbind in a balanced
# tree (each element copied log2(T) times, not T), transpose back.
ppo_rec_cat_cols <- function(xs) {
  ts <- lapply(xs, ag_transpose)
  while (length(ts) > 1L) {
    n <- length(ts)
    nxt <- lapply(seq(1L, n - 1L, by = 2L), function(i) ag_concat_rows(ts[[i]], ts[[i + 1L]]))
    if (n %% 2L == 1L) nxt <- c(nxt, ts[n])
    ts <- nxt
  }
  ag_transpose(ts[[1L]])
}

# ---- collection: one recorded graph per step ---------------------------------

# x [obs_dim, B], h, c [hidden, B] host matrices (copies whose episode starts
# already zeroed), legal [n_actions, B] 1 = allowed. Returns a (0-based), and
# unless greedy logp and value, plus the new h and c.
ppo_rec_act <- function(net, x, h, c, norm, legal, greedy = FALSE, key = stats::runif(2L)) {
  pen <- PPO_MASK * (1 - legal)
  key <- matrix(as.numeric(key)[1:2], 2L, 1L)
  body <- function(x, h, c, pen, key) {
    o <- net$step(x, list(h = h, c = c))
    z <- ag_add(o$logits, pen)
    if (greedy)
      return(list(a = ag_argmax(z, as_tensor = TRUE), h = o$state$h, c = o$state$c))
    lp <- ag_log_softmax(z)
    a <- ag_sample_categorical(lp, key = key)
    list(a = a, logp = ag_gather(lp, a), value = o$value, h = o$state$h, c = o$state$c)
  }
  res <- ppo_rec_run(net, paste0("act_", if (greedy) "greedy" else "sample"), norm,
                     body, list(x = x, h = h, c = c, pen = pen, key = key))
  out <- list(a = as.integer(round(as.numeric(res$a))), h = as.matrix(res$h),
              c = as.matrix(res$c))
  if (!greedy) {
    out$logp <- as.numeric(res$logp)
    out$value <- as.numeric(res$value)
  }
  out
}

# V(x) from the state (h, c).
ppo_rec_value <- function(net, x, h, c, norm) {
  body <- function(x, h, c) list(value = net$step(x, list(h = h, c = c))$value)
  as.numeric(ppo_rec_run(net, "value", norm, body, list(x = x, h = h, c = c))$value)
}

# Runs body(x, ...) as a recording cached in net$captured (one per id and
# normalization clip), normalizing x first when `norm` is set.
ppo_rec_run <- function(net, id, norm, body, args) {
  cache <- net$captured
  params <- unname(net$parameters())
  if (is.null(norm)) {
    if (is.null(cache[[id]])) cache[[id]] <- ag_capture(body, params = params)
    return(do.call(cache[[id]], args))
  }
  id <- paste0(id, "_norm_", format(norm$clip_obs, digits = 17))
  if (is.null(cache[[id]])) {
    clip <- norm$clip_obs
    f <- function(x, m, s, ...) body(obs_normalize(x, m, s, clip), ...)
    cache[[id]] <- ag_capture(f, params = params)
  }
  do.call(cache[[id]], c(args, list(m = matrix(norm$mean), s = obs_scale(norm))))
}

# ---- update -------------------------------------------------------------------

# The minibatch of environment copies `e` (1-based) of rollout `ro`: per-step
# inputs, the flat [., T * |e|] targets (column (t - 1) * |e| + j) and the
# initial state.
ppo_rec_batch <- function(ro, e) {
  n_t <- dim(ro$obs)[3L]
  B <- length(e)
  n_act <- dim(ro$legal)[1L]
  act <- as.vector(ro$act[e, , drop = FALSE])
  oh <- matrix(0, n_act, B * n_t)
  oh[cbind(act + 1L, seq_len(B * n_t))] <- 1
  list(xs = lapply(seq_len(n_t), function(t) matrix(ro$obs[, e, t], ncol = B)),
       starts = t(ro$starts[e, , drop = FALSE]) * 1,
       state = list(h = ro$h0[, e, drop = FALSE], c = ro$c0[, e, drop = FALSE]),
       legal = matrix(ro$legal[, e, , drop = FALSE], n_act),
       onehot = oh,
       old = as.vector(ro$logp[e, , drop = FALSE]),
       adv = as.vector(ro$adv[e, , drop = FALSE]),
       ret = as.vector(ro$ret[e, , drop = FALSE]))
}

# Forward over the minibatch's rollouts (inside a tape): masked log-softmax,
# log-probability of the taken actions [1, T * |e|] and the values.
ppo_rec_forward <- function(net, mb, norm) {
  xs <- lapply(mb$xs, function(x) obs_input(x, norm))
  out <- net$sequence(xs, mb$state, mb$starts)
  logp <- policy_logp(out$logits, mb$legal)
  list(logp = logp, logp_a = ag_sum(ag_mul(logp, ag_tensor(mb$onehot)), dim = 2L),
       value = out$value)
}

ppo_rec_update <- function(net, opt, ro, e, clip, vf_coef, ent_coef, max_grad_norm) {
  mb <- ppo_rec_batch(ro, e)
  adv <- (mb$adv - mean(mb$adv)) / (stats::sd(mb$adv) + 1e-8)
  old <- ag_tensor(matrix(mb$old, 1L))
  A <- ag_tensor(matrix(adv, 1L))
  R <- ag_tensor(matrix(mb$ret, 1L))
  with_grad_tape({
    f <- ppo_rec_forward(net, mb, ro$norm)
    ratio <- ag_exp(ag_sub(f$logp_a, old))
    s1 <- ag_mul(ratio, A)
    s2 <- ag_mul(ag_clamp(ratio, 1 - clip, 1 + clip), A)
    pg_loss <- ag_scale(ag_mean(ag_sub(s1, ag_relu(ag_sub(s1, s2)))), -1)
    vf_loss <- ag_mse_loss(f$value, R)
    ent <- ag_scale(ag_mean(ag_sum(ag_mul(ag_exp(f$logp), f$logp), dim = 2L)), -1)
    loss <- ag_add(ag_add(pg_loss, ag_scale(vf_loss, vf_coef)), ag_scale(ent, -ent_coef))
  })
  grads <- backward(loss)
  gn <- if (!is.null(max_grad_norm)) clip_grad_norm(net$parameters(), grads, max_grad_norm)
  opt$step(grads)
  opt$zero_grad()
  l <- c(pg = as.matrix(pg_loss)[1], vf = as.matrix(vf_loss)[1], ent = as.matrix(ent)[1])
  if (!is.null(gn)) l <- c(l, grad_norm = gn)
  l
}

# ppo_rec_update() as one ggmlR captured step (ag_capture_step), recorded once
# per key and replayed for every minibatch. Everything taken from the data is
# an input of the recording: the observations as one [T * obs_dim, B] matrix
# (step t = rows t * obs_dim + 1 ..., cut with ag_slice_rows), the episode
# starts [T, B], the initial state, the mask penalty, the taken actions as a
# one-hot matrix, old log-probabilities, advantages, returns and the
# normalization statistics (they move between iterations). Frozen, hence the
# key: B, T, clip, vf_coef, ent_coef, normalization on/off and its clip
# range, clipping on/off.
ppo_rec_update_captured <- function(net, opt, captures, ro, e, clip, vf_coef, ent_coef,
                                    max_grad_norm) {
  mb <- ppo_rec_batch(ro, e)
  n_t <- length(mb$xs)
  B <- length(e)
  obs_dim <- nrow(mb$xs[[1L]])
  # the recording does not check the starts on the device: do it here
  if (anyNA(mb$starts) || any(mb$starts != 0 & mb$starts != 1))
    stop("ppo update: episode starts must be 0/1", call. = FALSE)
  adv <- (mb$adv - mean(mb$adv)) / (stats::sd(mb$adv) + 1e-8)
  norm <- ro$norm
  norm_on <- !is.null(norm)
  inputs <- list(x = matrix(aperm(ro$obs[, e, , drop = FALSE], c(1L, 3L, 2L)), obs_dim * n_t),
                 st = mb$starts, h = mb$state$h, c = mb$state$c,
                 mask = PPO_MASK * (1 - mb$legal), onehot = mb$onehot,
                 old = matrix(mb$old, 1L), A = matrix(adv, 1L), R = matrix(mb$ret, 1L))
  if (norm_on) {
    inputs$m <- matrix(norm$mean)
    inputs$s <- obs_scale(norm)
  }
  key <- paste(c("rec", B, n_t, format(c(clip, vf_coef, ent_coef), digits = 17), norm_on,
                 if (norm_on) format(norm$clip_obs, digits = 17), !is.null(max_grad_norm)),
               collapse = "|")
  step <- captures[[key]]
  if (is.null(step)) {
    clip_obs <- if (norm_on) norm$clip_obs
    fn <- function(x, st, h, c, mask, onehot, old, A, R, m = NULL, s = NULL) {
      xs <- lapply(seq_len(n_t) - 1L, function(t) {
        xt <- ag_slice_rows(x, t * obs_dim, obs_dim)
        if (norm_on) obs_normalize(xt, m, s, clip_obs) else xt
      })
      out <- net$sequence(xs, list(h = h, c = c), st)
      logp <- ag_log_softmax(ag_add(out$logits, mask))
      logp_a <- ag_sum(ag_mul(logp, onehot), dim = 2L)
      ratio <- ag_exp(ag_sub(logp_a, old))
      s1 <- ag_mul(ratio, A)
      s2 <- ag_mul(ag_clamp(ratio, 1 - clip, 1 + clip), A)
      pg_loss <- ag_scale(ag_mean(ag_sub(s1, ag_relu(ag_sub(s1, s2)))), -1)
      vf_loss <- ag_mse_loss(out$value, R)
      ent <- ag_scale(ag_mean(ag_sum(ag_mul(ag_exp(logp), logp), dim = 2L)), -1)
      loss <- ag_add(ag_add(pg_loss, ag_scale(vf_loss, vf_coef)), ag_scale(ent, -ent_coef))
      list(loss = loss, pg = pg_loss, vf = vf_loss, ent = ent)
    }
    step <- ag_capture_step(fn, optimizer = opt, max_grad_norm = max_grad_norm)
    captures[[key]] <- step
  }
  out <- do.call(step, inputs)
  l <- c(pg = out$pg[1], vf = out$vf[1], ent = out$ent[1])
  if (!is.null(out$grad_norm)) l <- c(l, grad_norm = out$grad_norm[1])
  l
}

# Log-probabilities of the taken actions of a whole rollout, recomputed over
# the sequences as the update does ([n_env, T]); for tests and diagnostics.
ppo_rec_rollout_logp <- function(net, ro) {
  n_env <- ncol(ro$h0)
  mb <- ppo_rec_batch(ro, seq_len(n_env))
  lp <- NULL
  with_grad_tape(lp <- as.numeric(as.matrix(ppo_rec_forward(net, mb, ro$norm)$logp_a)))
  matrix(lp, n_env)
}

# ---- acting with a trained model ---------------------------------------------

# rl_act() for net_arch "lstm": obs [B, obs_dim]; state NULL (zeros) or
# list(h, c) [hidden, B]; starts zeroes the state of those copies first.
ppo_rec_act_model <- function(model, obs, legal, greedy, state, starts) {
  meta <- model$meta
  B <- nrow(obs)
  H <- meta$hidden
  if (is.null(state)) state <- list(h = matrix(0, H, B), c = matrix(0, H, B))
  if (!is.list(state) || !all(c("h", "c") %in% names(state)) ||
      !identical(dim(as.matrix(state$h)), c(H, B)) || !identical(dim(as.matrix(state$c)), c(H, B)))
    stop(sprintf("rl_act(): state must be list(h, c), each %d x %d (hidden x copies)", H, B),
         call. = FALSE)
  h <- as.matrix(state$h); c <- as.matrix(state$c)
  if (!is.null(starts)) {
    starts <- as.logical(starts)
    if (length(starts) != B || anyNA(starts))
      stop("rl_act(): starts must be ", B, " TRUE/FALSE values", call. = FALSE)
    h[, starts] <- 0
    c[, starts] <- 0
  }
  if (is.null(legal)) legal <- matrix(TRUE, B, meta$n_actions)
  norm <- if (isTRUE(meta$normalize_obs)) meta$obs_norm
  r <- with_policy_modes(model,
         ppo_rec_act(model$net, t(obs), h, c, norm, t(legal) * 1, greedy = greedy))
  list(action = r$a, state = list(h = r$h, c = r$c))
}
