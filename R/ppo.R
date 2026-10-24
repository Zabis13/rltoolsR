# Proximal policy optimization (PPO) with a ggmlR autograd network.
#
# Layout follows ggmlR's ag_* engine: features x batch (one column per sample).
# Rollouts run the network without a tape; sampling, masking and GAE are cheap
# R code on 9 x N matrices. Only the update builds a tape.

# Policy/value MLP as a ggmlR module (class "ag_layer" with named
# $parameters()), so it can be saved with ag_save_model(). The optimizer gets
# the same named list: ggmlR matches parameters by name, unnamed or duplicate
# names are silently not updated.
ppo_net <- function(obs_dim, n_actions, hidden) {
  l1 <- ag_linear(obs_dim, hidden, activation = "relu")
  l2 <- ag_linear(hidden, hidden, activation = "relu")
  pi <- ag_linear(hidden, n_actions)
  v  <- ag_linear(hidden, 1L, activation = "tanh")
  forward <- function(x) {
    h <- l2$forward(l1$forward(x))
    list(logits = pi$forward(h), value = v$forward(h))
  }
  parameters <- function() list(
    l1_W = l1$W, l1_b = l1$b, l2_W = l2$W, l2_b = l2$b,
    pi_W = pi$W, pi_b = pi$b, v_W = v$W, v_b = v$b
  )
  structure(list(forward = forward, parameters = parameters,
                 captured = new.env(parent = emptyenv())),
            class = c("rl_ppo_net", "ag_layer"))
}

# Large negative logit for illegal actions; finite so f16 stays finite too.
PPO_MASK <- -1e4

# The network's forward outside the update (collection, bootstrap, rl_act):
# features x batch observations -> list of output matrices.
#
# On the GPU it is a ggmlR capture (ag_capture): recorded once per batch shape,
# then one launch per call with no R-level op dispatch -- the per-op recording
# was ~0.25 ms an op while the device sat idle (inst/examples/
# profile_ppo_stages.R). Normalization statistics are graph inputs, not
# constants, since they move between iterations. On the CPU ag_capture just
# runs the forward.
policy_forward <- function(net, x, norm = NULL) {
  cache <- net$captured
  if (is.null(cache))                  # a net built without the cache
    return(lapply(net$forward(obs_input(x, norm)), as.matrix))
  params <- unname(net$parameters())
  if (is.null(norm)) {
    if (is.null(cache$plain))
      cache$plain <- ag_capture(function(x) net$forward(x), params = params)
    return(cache$plain(x = x))
  }
  # clip is a constant of the graph, so one recording per value
  key <- paste0("norm_", format(norm$clip_obs, digits = 17))
  if (is.null(cache[[key]])) {
    clip <- norm$clip_obs
    cache[[key]] <- ag_capture(function(x, m, s) net$forward(obs_normalize(x, m, s, clip)),
                               params = params)
  }
  cache[[key]](x = x, m = matrix(norm$mean), s = obs_scale(norm))
}

# Masked log-probabilities [n_actions, B] as an ag_tensor on the network's
# device. One function for collection, rl_act() and the update, so logp agrees
# across them (same op, same precision). `legal` is [n_actions, B], 1 = allowed;
# the mask is added before the softmax, so an illegal action never wins argmax.
policy_logp <- function(logits, legal) {
  ag_log_softmax(ag_add(logits, ag_tensor(PPO_MASK * (1 - legal))))
}

# The same masked log-softmax on the host, for collection: the logits are
# downloaded there anyway (logp of the taken action), and computing it on the
# device instead costs two more launches per step (+24% collect, measured)
# until the forward and action selection become one graph. With f32 matmuls it
# agrees with policy_logp() to ~1e-7.
host_logp <- function(logits, legal) {
  logits <- logits + PPO_MASK * (1 - legal)
  m <- apply(logits, 2, max)
  z <- sweep(logits, 2, m)
  sweep(z, 2, log(colSums(exp(z))))
}

#' Choose actions with a trained policy
#'
#' @param model Result of [rl_ppo_train()] or [rl_load_model()].
#' @param obs Observation matrix, one row per environment.
#' @param legal Optional logical matrix of legal actions (same rows as `obs`).
#'   Ignored for continuous actions.
#' @param greedy `TRUE` for the most likely (legal) action, `FALSE` to sample.
#' @details Observations are normalized with the statistics frozen in the
#'   model metadata when the model was trained with `normalize_obs`.
#'
#'   `rl_ppo_act()` is a deprecated alias kept for compatibility.
#' @return Discrete actions: integer vector of actions (`0`-based).
#'   Continuous actions: `n x act_dim` matrix clipped to the action space.
#' @export
#' @examples
#' \donttest{
#' env <- rl_gpu_env("tictactoe", n = 16, device = "cpu")
#' m <- rl_ppo_train(env, iterations = 1, device = "cpu", verbose = FALSE)
#' g <- rl_gpu_env("tictactoe", n = 1, opponent = "random", device = "cpu")
#' obs <- rl_reset(g)
#' rl_act(m, obs, rl_legal_actions(g))
#' }
rl_act <- function(model, obs, legal = NULL, greedy = TRUE) {
  prev <- ag_device(model$device)
  on.exit(ag_device(prev), add = TRUE)
  # same GPU modes as in training (see rl_train): one fused graph for the
  # forward (16 -> 2 launches per call, measured), and f32-tile matmuls so
  # batch 1 and a batch of more than 8 observations pick the same action
  ag_local_mode(graph = TRUE, matmul_precision = "f32")
  meta <- model$meta
  norm <- if (isTRUE(meta$normalize_obs)) meta$obs_norm
  # one captured launch; the outputs come back as small matrices, and the
  # action is picked from them on the host
  out <- policy_forward(model$net, t(obs), norm)
  if (identical(meta$action_type, "box")) {
    a <- if (greedy) out$mu else as.matrix(ag_sample_normal(out$mu, out$log_std))
    return(clip_to_space(a, meta$action_space))
  }
  if (is.null(legal)) legal <- matrix(TRUE, nrow(obs), meta$n_actions)
  masked <- out$logits + PPO_MASK * (1 - t(legal) * 1)
  # greedy: argmax of the masked logits (log-softmax does not move it), ties ->
  # first; sampling: Gumbel-max
  if (greedy) ag_argmax(masked) else ag_sample_categorical(masked)
}

#' @rdname rl_act
#' @export
rl_ppo_act <- rl_act


#' PPO algorithm for rl_train()
#'
#' Collects `n_steps` transitions from every copy of a vectorized
#' environment, computes GAE advantages and runs clipped PPO updates with a
#' ggmlR autograd network.
#'
#' Discrete actions use policy and value heads on a shared MLP. If the
#' environment provides a legal-action mask (in `rl_step()` or
#' [rl_legal_actions()]), illegal actions are masked out of the policy. For
#' two-player self-play environments (`rl_gpu_env(..., opponent = "self")`)
#' rewards and values are taken from the side to move, so the value of the
#' next state enters GAE with a minus sign.
#'
#' Continuous (`box`) actions use a Gaussian policy with a state-independent
#' `log_std` and separate actor and critic MLPs (tanh, orthogonal
#' initialization). Actions are sampled unbounded and clipped to the action
#' space only when sent to the environment.
#'
#' On truncation the value of `final_obs` is bootstrapped into the reward.
#'
#' Observations are normalized inside the network graph as
#' `clip((x - mean) / sqrt(var + epsilon), -clip_obs, clip_obs)`; the
#' statistics are stored in the model metadata and frozen in [rl_act()].
#' Rewards are divided by the standard deviation of the running discounted
#' return and clipped to `[-clip_reward, clip_reward]` (training only). Both
#' sets of statistics are fixed during a rollout and its update, and are
#' updated from the rollout once it is complete, so they lag by one
#' iteration; the first rollout is seen unnormalized.
#'
#' @param n_steps Steps collected per environment copy per iteration.
#' @param epochs PPO epochs per iteration.
#' @param minibatch Minibatch size.
#' @param lr Adam learning rate.
#' @param gamma,lambda Discount and GAE parameters.
#' @param clip PPO clip range.
#' @param vf_coef,ent_coef Value-loss and entropy coefficients.
#' @param hidden Hidden layer width.
#' @param normalize_obs,normalize_reward Normalize observations / rewards.
#'   `NULL` means `TRUE` for continuous and `FALSE` for discrete actions.
#' @param clip_obs,clip_reward Clip ranges after normalization.
#' @param epsilon Added to the variance before the square root.
#' @param max_grad_norm `NULL` (no clipping, the default) or a positive number:
#'   clip the gradients to this global L2 norm before each Adam step
#'   (0.5 is the Stable-Baselines3 default). The pre-clipping norm is reported as
#'   the `grad_norm` metric.
#' @param capture Run each update step as a ggmlR captured step
#'   (`ggmlR::ag_capture_step()`): recorded once per minibatch size, then one
#'   upload, two graph launches and one download per step instead of R-level op
#'   dispatch. Same results as the ordinary path (to rounding; checked on
#'   tic-tac-toe and Pendulum, several seeds). Measured on the GPU: update 3.6x
#'   (tic-tac-toe) and 5.4x (Pendulum) faster. `FALSE` runs the ordinary path.
#' @return An [rl_algo()].
#' @seealso [rl_train()], [rl_ppo_train()]
#' @export
#' @examples
#' algo <- rl_ppo(n_steps = 64, minibatch = 128, lr = 3e-4)
#' \donttest{
#' env <- rl_vec_env("pendulum", n = 4, seed = 1)
#' m <- rl_train(env, algo, total_steps = 1024, device = "cpu", verbose = FALSE)
#' tail(m$history)
#' }
rl_ppo <- function(n_steps = 9, epochs = 4, minibatch = 2048, lr = 1e-3,
                   gamma = 0.99, lambda = 0.95, clip = 0.2, vf_coef = 0.5,
                   ent_coef = 0.03, hidden = 64, normalize_obs = NULL,
                   normalize_reward = NULL, clip_obs = 10, clip_reward = 10,
                   epsilon = 1e-8, max_grad_norm = NULL, capture = TRUE) {
  if (!is.null(max_grad_norm) &&
      (!is.numeric(max_grad_norm) || length(max_grad_norm) != 1L ||
       !is.finite(max_grad_norm) || max_grad_norm <= 0))
    stop("max_grad_norm must be NULL or one positive finite number", call. = FALSE)
  capture <- isTRUE(capture)
  hidden <- as.integer(hidden)
  legal_of <- function(env, legal, n_act) {
    if (is.null(legal)) legal <- rl_legal_actions(env)
    if (is.null(legal)) matrix(TRUE, env$n, n_act) else legal
  }
  obs_norm <- function(state) {
    if (state$meta$normalize_obs) obs_norm_meta(state$obs_rms, epsilon, clip_obs)
  }

  init <- function(env, device) {
    sp <- env$action_space
    if (!sp$type %in% c("discrete", "box")) stop("PPO supports discrete and box actions only")
    box <- sp$type == "box"
    obs_dim <- space_size(env$observation_space)
    meta <- list(algo = "ppo", action_type = sp$type, obs_dim = obs_dim,
                 hidden = hidden, observation_space = env$observation_space,
                 action_space = sp,
                 normalize_obs = if (is.null(normalize_obs)) box else isTRUE(normalize_obs),
                 normalize_reward = if (is.null(normalize_reward)) box else isTRUE(normalize_reward))
    if (box) meta$act_dim <- space_size(sp) else meta$n_actions <- sp$n
    net <- ppo_build(meta)
    list(net = net, opt = optimizer_adam(net$parameters(), lr = lr),
         meta = meta, device = device, box = box,
         turn_sign = if (identical(env$opponent, "self")) -1 else 1,
         obs_rms = running_stats(obs_dim), rew_rms = running_stats(1L),
         obs = NULL, legal = NULL, ret_track = NULL, rollout = NULL, updates = 0,
         captures = new.env(parent = emptyenv()))
  }

  collect <- function(state, env) {
    net <- state$net
    meta <- state$meta
    box <- state$box
    n_env <- env$n
    obs_dim <- meta$obs_dim
    turn_sign <- state$turn_sign
    if (is.null(state[["obs"]])) {
      state$obs <- rl_reset(env)
      state$legal <- if (!box) legal_of(env, NULL, meta$n_actions)
      state$ret_track <- numeric(n_env)
    }
    # one normalization for the whole rollout, used again in update()
    norm <- obs_norm(state)
    rew_rms <- state$rew_rms
    obs <- state[["obs"]]
    legal <- state$legal

    n_t <- n_steps
    b_obs <- array(0, c(obs_dim, n_env, n_t))
    if (box) {
      b_act <- array(0, c(meta$act_dim, n_env, n_t))
    } else {
      b_act <- matrix(0L, n_env, n_t)
      b_legal <- array(0, c(meta$n_actions, n_env, n_t))
    }
    b_logp <- b_val <- b_rew <- b_done <- b_ret <- matrix(0, n_env, n_t)
    n_done <- 0
    tk <- prof_start()
    for (t in seq_len(n_t)) {
      x <- t(obs)
      out <- policy_forward(net, x, norm)          # captured: launch + download
      tk <- prof_tick(tk, "collect: forward")
      if (box) {
        mu <- out$mu
        log_std <- as.numeric(out$log_std)
        tk <- prof_tick(tk, "collect: download")
        a <- as.matrix(ag_sample_normal(mu, matrix(log_std, ncol = 1L)))
        tk <- prof_tick(tk, "collect: select")
        r <- rl_step(env, clip_to_space(a, meta$action_space))
        tk <- prof_tick(tk, "collect: env")
        b_act[, , t] <- a
        b_logp[, t] <- gauss_logp(a, mu, log_std)
      } else {
        logits <- out$logits
        tk <- prof_tick(tk, "collect: download")
        logp_all <- host_logp(logits, t(legal) * 1)
        a <- ag_sample_categorical(logp_all)
        tk <- prof_tick(tk, "collect: select")
        r <- rl_step(env, a)
        tk <- prof_tick(tk, "collect: env")
        b_legal[, , t] <- t(legal) * 1
        b_act[, t] <- a
        b_logp[, t] <- logp_all[cbind(a + 1L, seq_len(n_env))]
      }
      b_obs[, , t] <- x
      v_t <- as.numeric(out$value)
      tk <- prof_tick(tk, "collect: download")
      b_val[, t] <- v_t
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
        # padded to n_env columns: one recorded shape instead of one per
        # number of truncated environments
        fin <- matrix(0, obs_dim, n_env)
        fin[, seq_len(sum(trunc))] <- t(r$final_obs[trunc, , drop = FALSE])
        v_fin <- as.numeric(policy_forward(net, fin, norm)$value)[seq_len(sum(trunc))]
        b_rew[trunc, t] <- b_rew[trunc, t] + gamma * turn_sign * v_fin
      }
      b_done[, t] <- as.numeric(done)
      n_done <- n_done + sum(done)
      obs <- r$obs
      if (!box) legal <- legal_of(env, r$legal, meta$n_actions)
      tk <- prof_tick(tk, "collect: buffer")
    }
    last_val <- as.numeric(policy_forward(net, t(obs), norm)$value)
    tk <- prof_tick(tk, "collect: bootstrap")

    # ---- GAE (value of the next state is from the next mover's view) ----
    adv <- matrix(0, n_env, n_t)
    next_val <- last_val
    next_adv <- numeric(n_env)
    for (t in n_t:1) {
      nd <- 1 - b_done[, t]
      delta <- b_rew[, t] + gamma * nd * turn_sign * next_val - b_val[, t]
      adv[, t] <- delta + gamma * lambda * nd * turn_sign * next_adv
      next_val <- b_val[, t]
      next_adv <- adv[, t]
    }
    ret <- adv + b_val

    N <- n_env * n_t
    state$rollout <- list(
      obs = matrix(b_obs, obs_dim, N),
      act = if (box) matrix(b_act, meta$act_dim, N) else as.integer(b_act),
      legal = if (!box) matrix(b_legal, meta$n_actions, N),
      logp = as.numeric(b_logp), adv = as.numeric(adv), ret = as.numeric(ret),
      norm = norm)
    # statistics move only now, after GAE and the bootstrap used the old ones
    if (meta$normalize_obs) state$obs_rms <- running_stats_update(state$obs_rms, state$rollout$obs)
    if (meta$normalize_reward) state$rew_rms <- running_stats_update(state$rew_rms, matrix(b_ret, 1L))
    state$obs <- obs
    state$legal <- legal
    prof_tick(tk, "collect: gae")
    list(state = state, steps = N, info = c(episodes = n_done))
  }

  update <- function(state) {
    ro <- state$rollout
    N <- length(ro$logp)
    losses <- c(pg = 0, vf = 0, ent = 0)
    n_mb <- 0
    for (ep in seq_len(epochs)) {
      perm <- sample.int(N)
      for (s in seq(1, N, by = minibatch)) {
        tk <- prof_start()
        idx <- perm[s:min(s + minibatch - 1L, N)]
        # minibatch slicing (the argument promises) is forced inside the
        # update and charged to its "prep" stage
        prof_tick(tk, "update: prep")
        l <- if (capture) {
          ppo_update_captured(state, ro$obs[, idx, drop = FALSE],
                              if (!state$box) ro$legal[, idx, drop = FALSE],
                              if (state$box) ro$act[, idx, drop = FALSE] else ro$act[idx],
                              ro$logp[idx], ro$adv[idx], ro$ret[idx], clip, vf_coef,
                              ent_coef, ro$norm, max_grad_norm)
        } else if (state$box) {
          ppo_update_box(state$net, state$opt, ro$obs[, idx, drop = FALSE],
                         ro$act[, idx, drop = FALSE], ro$logp[idx], ro$adv[idx],
                         ro$ret[idx], clip, vf_coef, ent_coef, ro$norm, max_grad_norm)
        } else {
          ppo_update(state$net, state$opt, ro$obs[, idx, drop = FALSE],
                     ro$legal[, idx, drop = FALSE], ro$act[idx], ro$logp[idx],
                     ro$adv[idx], ro$ret[idx], clip, vf_coef, ent_coef, ro$norm,
                     max_grad_norm)
        }
        # grad_norm joins the metrics when clipping is on
        losses <- if (n_mb == 0) l else losses + l
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
    structure(list(net = state$net, device = state$device, meta = meta,
                   history = NULL), class = "rl_ppo")
  }

  # Adam moments and step count travel with the weights, so a resume continues
  # the optimizer instead of restarting it.
  get_train_state <- function(state) {
    list(weights = save_net(state$net), optimizer_state = state$opt$state(),
         normalizers = list(obs = state$obs_rms, reward = state$rew_rms),
         counters = list(updates = state$updates))
  }

  set_train_state <- function(state, train_state) {
    net <- load_net(train_state$weights, function() ppo_build(state$meta), state$device)
    state$net <- net
    state$opt <- optimizer_adam(net$parameters(), lr = lr)
    # Recordings reference the old network and optimizer: never carried over.
    state$captures <- ppo_free_captures(state$captures)
    if (is.null(train_state$optimizer_state)) {
      warning("checkpoint has no optimizer state (written by an older rltoolsR): ",
              "Adam starts fresh", call. = FALSE)
    } else {
      # rl_train() already warns about changed hyperparameters, comparing all of
      # algo$params (lr included; Adam's betas/eps are not settable here), so
      # ggmlR's own report of the same change is dropped -- by class, and only
      # that condition: shape/type errors and other warnings pass through.
      withCallingHandlers(
        state$opt$load_state(train_state$optimizer_state),
        ggmlR_opt_hyper_warning = function(w) invokeRestart("muffleWarning"))
    }
    state$obs_rms <- train_state$normalizers$obs
    state$rew_rms <- train_state$normalizers$reward
    state$updates <- train_state$counters$updates
    # keep the fields (as NULL): `$` would partially match obs -> obs_rms
    state[c("obs", "legal", "ret_track", "rollout")] <- list(NULL)
    state
  }

  rl_algo("ppo", init = init, collect = collect, update = update, model = model,
          get_train_state = get_train_state, set_train_state = set_train_state,
          params = list(n_steps = n_steps, epochs = epochs, minibatch = minibatch,
                        lr = lr, gamma = gamma, lambda = lambda, clip = clip,
                        vf_coef = vf_coef, ent_coef = ent_coef, hidden = hidden,
                        normalize_obs = normalize_obs,
                        normalize_reward = normalize_reward, clip_obs = clip_obs,
                        clip_reward = clip_reward, epsilon = epsilon,
                        max_grad_norm = max_grad_norm, capture = capture))
}

# PPO network for model metadata; files without action_type are discrete.
ppo_build <- function(meta) {
  if (identical(meta$action_type, "box")) {
    ppo_box_net(meta$obs_dim, meta$act_dim, meta$hidden)
  } else {
    ppo_net(meta$obs_dim, meta$n_actions, meta$hidden)
  }
}

#' Train a policy with PPO
#'
#' Shortcut for [rl_train()] with [rl_ppo()], counting the budget and the
#' evaluation interval in PPO iterations (`n_steps` steps of every
#' environment copy each).
#'
#' @inheritParams rl_ppo
#' @inheritParams rl_train
#' @param env A vectorized environment (`rl_gpu_env` or `rl_vec_env`) or a
#'   single [rl_env()].
#' @param iterations Number of collect + update iterations.
#' @param device `"auto"` (GPU if a Vulkan device is available, else CPU),
#'   `"gpu"` or `"cpu"` for the network. Pass it explicitly when results
#'   must be reproducible.
#' @param checkpoint_every Checkpoint every this many iterations (0 = only
#'   at the end), see `checkpoint` in [rl_train()].
#' @param eval_every Evaluate every this many iterations (0 = never), using
#'   `eval_fn(model)`. Evaluation uses its own RNG stream, so it does not
#'   change the training trajectory.
#' @param eval_fn Function of the model returning a named numeric vector.
#' @param verbose Print progress.
#' @return A list of class `rl_ppo` with `net`, `device`, `meta`
#'   (architecture and spaces, used by [rl_save_model()]), `history`
#'   (losses per iteration), `eval` (results of `eval_fn`, or `NULL`) and
#'   `train_state` (see [rl_train()]).
#' @export
#' @examples
#' \donttest{
#' env <- rl_gpu_env("tictactoe", n = 64, device = "cpu")
#' m <- rl_ppo_train(env, iterations = 5, device = "cpu", verbose = FALSE,
#'                   eval_every = 5,
#'                   eval_fn = function(m) rl_ppo_eval_tictactoe(m, n = 100))
#' m$eval
#' }
rl_ppo_train <- function(env, iterations = 100, n_steps = 9, epochs = 4,
                         minibatch = 2048, lr = 1e-3, gamma = 0.99,
                         lambda = 0.95, clip = 0.2, vf_coef = 0.5,
                         ent_coef = 0.03, hidden = 64, normalize_obs = NULL,
                         normalize_reward = NULL, clip_obs = 10,
                         clip_reward = 10, epsilon = 1e-8,
                         device = c("auto", "gpu", "cpu"), eval_every = 0,
                         eval_fn = NULL, verbose = TRUE, log = NULL,
                         overwrite = FALSE, checkpoint = NULL,
                         checkpoint_every = 0, resume = NULL, append = FALSE,
                         max_grad_norm = NULL, capture = TRUE) {
  env <- as_vec_env(env)
  per_iter <- n_steps * env$n
  algo <- rl_ppo(n_steps = n_steps, epochs = epochs, minibatch = minibatch,
                 lr = lr, gamma = gamma, lambda = lambda, clip = clip,
                 vf_coef = vf_coef, ent_coef = ent_coef, hidden = hidden,
                 normalize_obs = normalize_obs,
                 normalize_reward = normalize_reward, clip_obs = clip_obs,
                 clip_reward = clip_reward, epsilon = epsilon,
                 max_grad_norm = max_grad_norm, capture = capture)
  rl_train(env, algo, total_steps = iterations * per_iter, device = device,
           eval_every_steps = eval_every * per_iter, eval_fn = eval_fn,
           verbose = verbose, log = log, overwrite = overwrite,
           checkpoint = checkpoint,
           checkpoint_every_steps = checkpoint_every * per_iter,
           resume = resume, append = append)
}

ppo_update <- function(net, opt, obs, legal, act, old_logp, adv, ret,
                       clip, vf_coef, ent_coef, norm = NULL, max_grad_norm = NULL) {
  tk <- prof_start()
  adv <- (adv - mean(adv)) / (stats::sd(adv) + 1e-8)
  old <- ag_tensor(matrix(old_logp, 1))
  A <- ag_tensor(matrix(adv, 1))
  R <- ag_tensor(matrix(ret, 1))
  force(obs); force(legal); force(act)
  tk <- prof_tick(tk, "update: prep")

  with_grad_tape({
    # Built INSIDE the tape: with normalization it is computed by ag_* ops on
    # the device and lives only in the pass pool, which with_grad_tape() resets
    # on entry -- built before, it was destroyed before the forward read it.
    x <- obs_input(obs, norm)
    out <- net$forward(x)
    # exact log-probabilities (no clamp): masked actions get logp ~ -1e4 and
    # p = 0, so p * logp = 0 and their gradient is ~0
    logp <- policy_logp(out$logits, legal)
    logp_a <- ag_gather(logp, act)                              # [1, B]
    p <- ag_exp(logp)
    ratio <- ag_exp(ag_sub(logp_a, old))
    s1 <- ag_mul(ratio, A)
    s2 <- ag_mul(ag_clamp(ratio, 1 - clip, 1 + clip), A)
    surr <- ag_sub(s1, ag_relu(ag_sub(s1, s2)))                 # min(s1, s2)
    pg_loss <- ag_scale(ag_mean(surr), -1)
    vf_loss <- ag_mse_loss(out$value, R)
    # entropy per sample is -colSums(p * logp); averaged over the batch
    ent <- ag_scale(ag_mean(ag_sum(ag_mul(p, logp), dim = 2L)), -1)
    loss <- ag_add(ag_add(pg_loss, ag_scale(vf_loss, vf_coef)), ag_scale(ent, -ent_coef))
  })
  tk <- prof_tick(tk, "update: forward")
  grads <- backward(loss)
  tk <- prof_tick(tk, "update: backward")
  gn <- if (!is.null(max_grad_norm))
    clip_grad_norm(net$parameters(), grads, max_grad_norm)
  opt$step(grads)
  opt$zero_grad()
  tk <- prof_tick(tk, "update: adam")
  l <- c(pg = as.matrix(pg_loss)[1], vf = as.matrix(vf_loss)[1], ent = as.matrix(ent)[1])
  if (!is.null(gn)) l <- c(l, grad_norm = gn)
  prof_tick(tk, "update: losses")
  l
}

# ppo_update() / ppo_update_box() as one ggmlR captured step
# (ag_capture_step): recorded once per minibatch size, then upload the
# minibatch, two graph launches (forward+backward, Adam) and one download of the
# losses -- no R-level op dispatch. Same losses and metrics as the ordinary
# path, and the same optimizer: every recording updates state$opt's tensors.
#
# Everything the step reads from the data is an input of the recording, never
# a constant: the legality mask, and the taken actions as a one-hot matrix
# (ag_gather() takes its indices from R, which would freeze the first
# minibatch's actions into the graph). With finite masked log-probabilities,
# colSums(logp * onehot) is exactly the gathered value.
#
# What IS frozen -- B, clip, vf_coef, ent_coef, normalization on/off and its
# clip range, clipping on/off -- is the cache key, so a change records anew.
ppo_update_captured <- function(state, obs, legal, act, old_logp, adv, ret,
                                clip, vf_coef, ent_coef, norm = NULL,
                                max_grad_norm = NULL) {
  tk <- prof_start()
  net <- state$net
  B <- ncol(obs)
  adv <- (adv - mean(adv)) / (stats::sd(adv) + 1e-8)
  inputs <- list(x = obs, old = matrix(old_logp, 1L), A = matrix(adv, 1L),
                 R = matrix(ret, 1L))
  if (state$box) {
    d <- nrow(act)
    # the graph drops the constant -d/2 log(2 pi) of the log-density
    inputs$old <- inputs$old + 0.5 * d * log(2 * pi)
    inputs$act <- act
  } else {
    n_act <- state$meta$n_actions
    # A bad index would give an all-zero column: a silently zero contribution.
    if (!is.numeric(act) || length(act) != B || anyNA(act) ||
        any(act != round(act)) || any(act < 0 | act > n_act - 1))
      stop("ppo update: actions must be ", B, " whole numbers in 0..", n_act - 1,
           call. = FALSE)
    oh <- matrix(0, n_act, B)
    oh[cbind(act + 1, seq_len(B))] <- 1
    inputs$onehot <- oh
    inputs$mask <- PPO_MASK * (1 - legal)
  }
  norm_on <- !is.null(norm)
  if (norm_on) {
    inputs$m <- matrix(norm$mean)
    inputs$s <- obs_scale(norm)
  }
  key <- paste(c(B, format(c(clip, vf_coef, ent_coef), digits = 17), norm_on,
                 if (norm_on) format(norm$clip_obs, digits = 17),
                 !is.null(max_grad_norm)), collapse = "|")
  step <- state$captures[[key]]
  if (is.null(step)) {
    clip_obs <- if (norm_on) norm$clip_obs
    x_in <- function(x, m, s) if (norm_on) obs_normalize(x, m, s, clip_obs) else x
    surrogate <- function(logp_a, old, A) {
      ratio <- ag_exp(ag_sub(logp_a, old))
      s1 <- ag_mul(ratio, A)
      s2 <- ag_mul(ag_clamp(ratio, 1 - clip, 1 + clip), A)
      ag_scale(ag_mean(ag_sub(s1, ag_relu(ag_sub(s1, s2)))), -1)   # -mean(min(s1, s2))
    }
    fn <- if (state$box) {
      function(x, act, old, A, R, m = NULL, s = NULL) {
        out <- net$forward(x_in(x, m, s))
        sum_ls <- ag_sum(out$log_std)
        pg_loss <- surrogate(box_logp_tensor(out, act), old, A)
        vf_loss <- ag_mse_loss(out$value, R)
        loss <- ag_add(ag_add(pg_loss, ag_scale(vf_loss, vf_coef)),
                       ag_scale(sum_ls, -ent_coef))
        list(loss = loss, pg = pg_loss, vf = vf_loss, ent = sum_ls)
      }
    } else {
      function(x, mask, onehot, old, A, R, m = NULL, s = NULL) {
        out <- net$forward(x_in(x, m, s))
        logp <- ag_log_softmax(ag_add(out$logits, mask))
        logp_a <- ag_sum(ag_mul(logp, onehot), dim = 2L)              # [1, B]
        pg_loss <- surrogate(logp_a, old, A)
        vf_loss <- ag_mse_loss(out$value, R)
        ent <- ag_scale(ag_mean(ag_sum(ag_mul(ag_exp(logp), logp), dim = 2L)), -1)
        loss <- ag_add(ag_add(pg_loss, ag_scale(vf_loss, vf_coef)),
                       ag_scale(ent, -ent_coef))
        list(loss = loss, pg = pg_loss, vf = vf_loss, ent = ent)
      }
    }
    step <- ag_capture_step(fn, optimizer = state$opt, max_grad_norm = max_grad_norm)
    state$captures[[key]] <- step
  }
  tk <- prof_tick(tk, "update: prep")
  out <- do.call(step, inputs)
  tk <- prof_tick(tk, "update: step")
  ent <- out$ent[1]
  if (state$box) ent <- ent + 0.5 * d * (1 + log(2 * pi))
  l <- c(pg = out$pg[1], vf = out$vf[1], ent = ent)
  if (!is.null(out$grad_norm)) l <- c(l, grad_norm = out$grad_norm[1])
  l
}

# Release the device memory of every recorded update step; returns a fresh cache.
ppo_free_captures <- function(captures) {
  if (is.environment(captures))
    for (k in ls(captures, all.names = TRUE)) ag_capture_free(captures[[k]])
  new.env(parent = emptyenv())
}

#' Evaluate a tic-tac-toe policy against a random opponent
#'
#' Plays `n` games as X against a uniformly random opponent with the greedy
#' policy and reports the fractions of wins, draws and losses.
#'
#' @param model Result of [rl_ppo_train()] or [rl_load_model()].
#' @param n Number of games.
#' @param device Device for the evaluation environment: the CPU by default,
#'   whatever the model's device (see [rl_gpu_env()] for why).
#' @return Named numeric vector `win`, `draw`, `loss`.
#' @export
#' @examples
#' \donttest{
#' env <- rl_gpu_env("tictactoe", n = 64, device = "cpu")
#' m <- rl_ppo_train(env, iterations = 5, device = "cpu", verbose = FALSE)
#' rl_ppo_eval_tictactoe(m, n = 200)
#' }
rl_ppo_eval_tictactoe <- function(model, n = 512, device = "cpu") {
  env <- rl_gpu_env("tictactoe", n = n, opponent = "random", device = device)
  obs <- rl_reset(env)
  legal <- rl_legal_actions(env)
  result <- rep(NA_real_, n)
  for (i in 1:5) {
    a <- rl_act(model, obs, legal, greedy = TRUE)
    r <- rl_step(env, a)
    first <- is.na(result) & r$terminated
    result[first] <- r$reward[first]
    if (!anyNA(result)) break
    obs <- r$obs
    legal <- r$legal
  }
  c(win = mean(result == 1), draw = mean(result == 0), loss = mean(result == -1))
}
