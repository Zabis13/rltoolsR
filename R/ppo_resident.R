# Resident PPO collection: the policy and the game are one ag_capture()
# recording, replayed once per step with no data crossing the bus; the n_steps
# replays are queued together (ag_capture_repeat) and R waits once. Every step
# reads a counter t on the device and writes the step into columns
# t * n_env + i of the rollout buffers:
#   obs_buf   [obs_dim, N]    observations, read by the update (as on the device path)
#   res_legal [n_actions, N]  legal actions (environments with a legal mask only)
#   res_rec   [6, N]          action, log-probability, value, reward, done, boot
# and t <- t + 1. boot is gamma * V(final observation) for truncated games, 0
# otherwise; the host adds it to the reward after reward scaling, as the host
# path does. After the rollout the bootstrap value is one more small recording,
# then res_rec (and res_legal) come down; GAE runs on the host.
#
# The games live in the collection's own ag_buffers, built by an adapter, not
# in `env`, which is not stepped (it gives the specification). The noise key of
# step t is (k1, k2 + t) with k2 on a 2^-12 grid, so k2 + t is exact in f32 for
# t < 2048: the same key on the host reproduces the step. The policy samples
# from stream 0 of the key; adapters draw their own noise from other streams.
#
# Adapter contract (a list):
#   obs_dim, n_actions, legal (TRUE when read() gives a legal mask)
#   new(n)                 state for n games (an environment rs)
#   buffers(rs)            its ag_buffers (freed with the captures)
#   params(rs)             everything a recording over rs must list in params
#   read(rs, cols)         list(obs [obs_dim, N], legal [n_actions, N] and pen
#                          (PPO_MASK on illegal actions) or NULL for both,
#                          state): the one read of each state buffer
#   step(rs, a, cols, key, s)  writes the next state; finished games restart in
#                          the same step (obs is then the new game's). Returns
#                          reward, done (terminated or truncated) [1, N], and
#                          truncated [1, N] with final_obs [obs_dim, N] (the
#                          observation before the restart), or NULL for both
#                          when the environment never truncates.
#   box, act_dim           box actions (Gaussian policy): `a` is then the
#                          unclipped action [act_dim, N], clipped by step()
#                          itself; res_rec holds act_dim action rows. Absent
#                          for discrete actions.

# ggmlR functions an adapter needs beyond the ggmlR in DESCRIPTION: the
# package loads with an older ggmlR (import(ggmlR)), only these paths need it.
ggmlr_missing <- function(fns) {
  ns <- asNamespace("ggmlR")
  fns[!vapply(fns, exists, NA, envir = ns, inherits = FALSE)]
}

ggmlr_need <- function(fns, what) {
  miss <- ggmlr_missing(fns)
  if (length(miss))
    stop(sprintf("%s needs %s from ggmlR, which the installed ggmlR %s does not have; update ggmlR",
                 what, paste0(miss, "()", collapse = ", "), format(utils::packageVersion("ggmlR"))),
         call. = FALSE)
  invisible(TRUE)
}

RES_TRIG <- c("ag_sin", "ag_cos")

ppo_res_adapter_ttt <- function() list(
  obs_dim = 9L, n_actions = 9L, legal = TRUE, fsp = TRUE,
  new = function(n) ttt_res_new(n),
  buffers = function(rs) list(rs$board, rs$player, rs$I9),
  params = ttt_res_params,
  read = function(rs, cols) {
    s <- ttt_res_read(rs, cols)
    list(obs = s$obs, legal = s$legal, pen = ag_scale(s$occupied, PPO_MASK), state = s)
  },
  step = function(rs, a, cols, key, s) {
    r <- ttt_res_step(rs, a, cols, s$state)
    list(obs = r$obs, reward = r$reward, done = r$done, truncated = NULL, final_obs = NULL)
  })

# Tabular (toy_text) environments: `env` is an rl_env / rl_vec_env of one of
# them, used only for its table.
ppo_res_adapter_tab <- function(env) {
  proto <- rl_env(env$name, seed = 1)
  list(
    obs_dim = space_size(proto$observation_space), n_actions = proto$action_space$n,
    legal = FALSE,
    new = function(n) tab_res_new(proto, n),
    buffers = tab_res_buffers,
    params = tab_res_params,
    read = function(rs, cols) {
      s <- tab_res_read(rs, cols)
      list(obs = s$obs, legal = NULL, pen = NULL, state = s)
    },
    step = function(rs, a, cols, key, s) {
      r <- tab_res_step(rs, a, cols, key, s$state)
      list(obs = r$obs, reward = r$reward, done = r$done,
           truncated = if (rs$max_steps > 0) r$truncated, final_obs = if (rs$max_steps > 0) r$final_obs)
    })
}

ppo_res_adapter_cart <- function() list(
  obs_dim = 4L, n_actions = 2L, legal = FALSE, needs = RES_TRIG,
  new = cart_res_new,
  buffers = cart_res_buffers,
  params = cart_res_params,
  read = function(rs, cols) {
    s <- cart_res_read(rs, cols)
    list(obs = s$obs, legal = NULL, pen = NULL, state = s)
  },
  step = function(rs, a, cols, key, s) {
    r <- cart_res_step(rs, a, cols, key, s$state)
    list(obs = r$obs, reward = r$reward, done = r$done,
         truncated = r$truncated, final_obs = r$final_obs)
  })

ppo_res_adapter_pend <- function(raw) list(
  obs_dim = 3L, box = TRUE, act_dim = 1L, legal = FALSE, needs = RES_TRIG,
  new = function(n) pend_res_new(n, raw),
  buffers = pend_res_buffers,
  params = pend_res_params,
  read = function(rs, cols) {
    s <- pend_res_read(rs, cols)
    list(obs = s$obs, legal = NULL, pen = NULL, state = s)
  },
  step = function(rs, a, cols, key, s) {
    r <- pend_res_step(rs, a, cols, key, s$state)
    list(obs = r$obs, reward = r$reward, done = r$done,
         truncated = r$truncated, final_obs = r$final_obs)
  })

RES_PENDULUM <- c("pendulum_gym", "pendulum_gym_raw")

RES_TABULAR <- "^(frozen_lake|cliff_walking|taxi|blackjack)"

# The adapter for `env`, or NULL when the resident rollout does not apply.
# The C++ cartpole (rl_env / rl_vec_env, f64) goes resident (f32, device
# resets) only on an explicit rollout = "resident": "auto" does not change the
# environment's implementation; rl_gpu_env("cartpole") is f32 already.
ppo_res_adapter <- function(env, box, fsp, capture, meta, device, rollout = "auto") {
  if (!capture || !identical(device, "gpu")) return(NULL)
  ad <- ppo_res_adapter_of(env, rollout)
  # fictitious self-play needs a two-player self-play adapter
  if (fsp && !isTRUE(ad$fsp)) return(NULL)
  # an older ggmlR: "auto" stays on the host path, "resident" says what is missing
  if (length(ggmlr_missing(ad$needs))) {
    if (identical(rollout, "resident"))
      ggmlr_need(ad$needs, sprintf("rl_ppo(rollout = \"resident\") for %s", env$name))
    return(NULL)
  }
  # the adapter's actions must be the policy's
  if (!is.null(ad) && !identical(isTRUE(ad$box), isTRUE(box))) return(NULL)
  ad
}

# The same C++ / GPU rule as cartpole for pendulum.
ppo_res_adapter_of <- function(env, rollout) {
  if (inherits(env, "rl_gpu_env") && env$name %in% RES_PENDULUM &&
      identical(env$device, "gpu"))
    return(ppo_res_adapter_pend(identical(env$name, "pendulum_gym_raw")))
  if (identical(rollout, "resident") && inherits(env, c("rl_env", "rl_vec_env")) &&
      env$name %in% RES_PENDULUM)
    return(ppo_res_adapter_pend(identical(env$name, "pendulum_gym_raw")))
  if (inherits(env, "rl_gpu_env") && identical(env$name, "tictactoe") &&
      identical(env$opponent, "self") && identical(env$device, "gpu"))
    return(ppo_res_adapter_ttt())
  if (inherits(env, "rl_gpu_env") && identical(env$name, "cartpole") &&
      identical(env$device, "gpu"))
    return(ppo_res_adapter_cart())
  if (identical(rollout, "resident") && inherits(env, c("rl_env", "rl_vec_env")) &&
      identical(env$name, "cartpole"))
    return(ppo_res_adapter_cart())
  if (inherits(env, c("rl_env", "rl_vec_env")) && grepl(RES_TABULAR, env$name))
    return(ppo_res_adapter_tab(env))
  NULL
}

ppo_res_key <- function() c(stats::runif(1L), floor(stats::runif(1L) * 4096) / 4096)

# Buffers and recordings, built once per training (and again after the
# captures are freed): everything lives in `cap`, freed by ppo_free_captures().
# clip_obs: NULL without observation normalization; otherwise the recordings
# take the frozen statistics as inputs m (mean) and sc (1 / sqrt(var + eps)),
# [obs_dim, 1] each, the policy and the bootstrap see the normalized
# observation, and obs_buf keeps the raw one (the update normalizes it, as on
# the host path). res_moments then gives the rollout's mean and centred
# variance per feature, two passes over obs_buf in f32.
#
# opp: NULL, or fictitious self-play as list(net, p_self, vs_pool, side): the
# opponent network (the trainer's opp_net, reloaded in place per rollout, so
# the recording stays valid) moves where vs_pool and the parity of the move
# differs from the learner's side. Both networks run on every game with the
# same key, as on the host, and the action and logp are chosen by that mask;
# the value is always the learner's. Per game on the device: vs_pool, side
# (0: X) and par (moves since the game started, mod 2). A finished game draws
# vs_pool = u1 >= p_self and side = u2 < 0.5 from stream 1 of the step's key;
# par restarts at 0 (X to move). res_rec gets three rows more: keep (1 - the
# opponent moved), vs_pool and side before the step.
ppo_res_setup <- function(net, cap, ad, n_env, n_steps, gamma, clip_obs = NULL,
                          opp = NULL) {
  if (n_steps >= 2048L) stop("rollout = \"resident\": n_steps must be below 2048")
  rs <- ad$new(n_env)
  bufs <- ad$buffers(rs)
  for (i in seq_along(bufs)) cap[[paste0("res_env", i)]] <- bufs[[i]]
  N <- n_env * n_steps
  cap$obs_buf <- ag_buffer(ad$obs_dim, N)
  if (ad$legal) cap$res_legal <- ag_buffer(ad$n_actions, N)
  box <- isTRUE(ad$box)
  fsp <- !is.null(opp)
  cap$res_rec <- ag_buffer(ppo_res_act_rows(ad) + 5L + if (fsp) 3L else 0L, N)
  cap$res_cnt <- ag_buffer(1, 1)
  e2 <- ag_tensor(matrix(c(0, 1), 2, 1))
  one <- ag_tensor(matrix(1, 1, 1))
  no_pen <- if (!box) ag_tensor(matrix(0, ad$n_actions, 1))
  zero_n <- ag_tensor(matrix(0, 1, n_env))
  # discrete: the body of policy_act(); box: the one of policy_act_box() (eps
  # from stream 0 of the step's key, logp without -d/2 log(2 pi))
  mk_body <- function(nt) if (box) {
    function(h, pen, key) {
      out <- nt$forward(h)
      eps <- ag_rand(key, "normal", c(ad$act_dim, n_env))
      a <- ag_add(out$mu, ag_mul(eps, ag_exp(out$log_std)))
      logp <- ag_sub(ag_scale(ag_sum(ag_mul(eps, eps), dim = 2L), -0.5), ag_sum(out$log_std))
      list(a = a, logp = logp, value = out$value)
    }
  } else policy_act_body(nt, greedy = FALSE)
  body <- mk_body(net)
  if (fsp) {
    body_opp <- mk_body(opp$net)
    cap$res_vs <- ag_buffer(1, n_env)
    cap$res_side <- ag_buffer(1, n_env)
    cap$res_par <- ag_buffer(1, n_env)
    ag_buffer_write(cap$res_vs, matrix(as.numeric(opp$vs_pool), 1, n_env), col_offset = 0)
    ag_buffer_write(cap$res_side, matrix(as.numeric(opp$side), 1, n_env), col_offset = 0)
    ag_buffer_write(cap$res_par, matrix(0, 1, n_env), col_offset = 0)
    mp_self <- ag_tensor(matrix(-opp$p_self, 1, 1))
    half <- ag_tensor(matrix(0.5, 1, 1))
  }
  params <- c(unname(net$parameters()), ad$params(rs),
              list(cap$obs_buf, cap$res_rec, cap$res_cnt, e2, one, zero_n),
              if (!box) list(no_pen), if (ad$legal) list(cap$res_legal),
              if (fsp) c(unname(opp$net$parameters()),
                         list(cap$res_vs, cap$res_side, cap$res_par, mp_self, half)))
  # the opponent's move: action and logp by the mask, the learner's value;
  # then the next vs_pool / side / par. Returns the actions and the record rows.
  fsp_act <- function(h, pen, kt, cols) {
    vs <- ag_get_rows(cap$res_vs, cols)                 # one read of each
    side <- ag_get_rows(cap$res_side, cols)
    par <- ag_get_rows(cap$res_par, cols)
    out <- body(h, pen, kt)
    o2 <- body_opp(h, pen, kt)
    # vs_pool [par != side]
    opp_m <- ag_mul(vs, ag_sub(ag_add(par, side), ag_scale(ag_mul(par, side), 2)))
    pick <- function(x, y) ag_add(x, ag_mul(opp_m, ag_sub(y, x)))
    list(out = list(a = pick(out$a, o2$a), logp = pick(out$logp, o2$logp), value = out$value),
         vs = vs, side = side, par = par, opp = opp_m)
  }
  fsp_next <- function(f, done, kt, cols) {
    u <- ag_rand(kt, "uniform", c(2L, n_env), stream = 1L)
    vs_new <- pend_res_pos(ag_add(ag_slice_rows(u, 0L, 1L), mp_self))   # u1 >= p_self
    side_new <- pend_res_pos(ag_add(ag_scale(ag_slice_rows(u, 1L, 1L), -1), half))  # u2 < 0.5
    not_done <- ag_add(ag_scale(done, -1), one)
    ag_set_rows(cap$res_vs, cols, ag_add(f$vs, ag_mul(done, ag_sub(vs_new, f$vs))))
    ag_set_rows(cap$res_side, cols, ag_add(f$side, ag_mul(done, ag_sub(side_new, f$side))))
    ag_set_rows(cap$res_par, cols, ag_mul(ag_add(ag_scale(f$par, -1), one), not_done))
    ag_concat_rows(ag_concat_rows(ag_add(ag_scale(f$opp, -1), one), f$vs), f$side)
  }
  norm_on <- !is.null(clip_obs)
  nz <- function(x, m, sc) if (norm_on) obs_normalize(x, m, sc, clip_obs) else x
  step_body <- function(cols, key, z, m, sc) {
    t <- ag_get_rows(cap$res_cnt, z)                    # the one read of t
    idx <- ag_add(cols, ag_scale(t, n_env))
    kt <- ag_add(key, ag_matmul(e2, t))
    s <- ad$read(rs, cols)
    h <- nz(s$obs, m, sc)
    pen <- if (is.null(s$pen)) no_pen else s$pen
    if (fsp) {
      f <- fsp_act(h, pen, kt, cols)
      out <- f$out
    } else out <- body(h, pen, kt)
    r <- ad$step(rs, out$a, cols, kt, s)
    boot <- if (is.null(r$truncated)) zero_n else
      ag_scale(ag_mul(r$truncated, net$forward(nz(r$final_obs, m, sc))$value), gamma)
    ag_set_rows(cap$obs_buf, idx, s$obs)
    if (ad$legal) ag_set_rows(cap$res_legal, idx, s$legal)
    rec <- ag_concat_rows(ag_concat_rows(ag_concat_rows(ag_concat_rows(ag_concat_rows(
      out$a, out$logp), out$value), r$reward), r$done), boot)
    if (fsp) rec <- ag_concat_rows(rec, fsp_next(f, r$done, kt, cols))
    ag_set_rows(cap$res_rec, idx, rec)
    ag_set_rows(cap$res_cnt, z, ag_add(t, one))
    NULL
  }
  value_body <- function(cols, m, sc)
    list(value = net$forward(nz(ad$read(rs, cols)$obs, m, sc))$value)
  vparams <- c(unname(net$parameters()), ad$params(rs))
  if (norm_on) {
    cap$res_step <- ag_capture(function(cols, key, z, m, sc) step_body(cols, key, z, m, sc),
                               params = params)
    cap$res_value <- ag_capture(function(cols, m, sc) value_body(cols, m, sc), params = vparams)
    # w [1, N]: 1 / n on the n columns the update keeps, 0 elsewhere -- one
    # shape for any n, so the recording is not redone
    cap$res_moments <- ag_capture(function(cols, w) {
      x <- ag_get_rows(cap$obs_buf, cols)               # [obs_dim, N]
      bm <- ag_sum(ag_mul(x, w), dim = 1L)              # [obs_dim, 1]
      d <- ag_sub(x, bm)
      list(mean = bm, var = ag_sum(ag_mul(ag_mul(d, d), w), dim = 1L))
    }, params = list(cap$obs_buf))
  } else {
    cap$res_step <- ag_capture(function(cols, key, z) step_body(cols, key, z, NULL, NULL),
                               params = params)
    cap$res_value <- ag_capture(function(cols) value_body(cols, NULL, NULL), params = vparams)
  }
  rs
}

# Rows of res_rec holding the action: act_dim for box actions, else 1.
ppo_res_act_rows <- function(ad) if (isTRUE(ad$box)) as.integer(ad$act_dim) else 1L

# Mean and population variance per feature of the rollout's columns `keep`
# (0-based, the ones the update trains on) out of N, from the device.
ppo_res_moments <- function(cap, N, keep = seq_len(N) - 1) {
  w <- numeric(N)
  w[keep + 1] <- 1 / length(keep)
  mo <- cap$res_moments(cols = matrix(seq_len(N) - 1, N, 1), w = matrix(w, 1, N))
  list(mean = as.numeric(mo$mean), var = as.numeric(mo$var), n = length(keep))
}

# One rollout. Returns the [n_env, n_steps] matrices the host path builds
# (boot: the truncation bootstrap, added to rew after scaling), plus the
# legal-action matrix [n_actions, N] (NULL for box actions; act is then
# [act_dim, N]) and the bootstrap value. `norm` (obs_norm_meta()) when the
# recordings normalize observations; the caller then takes the moments of the
# columns it keeps with ppo_res_moments().
ppo_res_rollout <- function(cap, ad, n_env, n_steps, key = ppo_res_key(), norm = NULL) {
  cols <- matrix(seq_len(n_env) - 1, n_env, 1)
  z <- matrix(0, 1, 1)
  key <- matrix(key, 2, 1)
  ag_buffer_write(cap$res_cnt, z, col_offset = 0)
  # n_steps runs queued at once, one wait (ggmlR::ag_capture_repeat)
  if (is.null(norm)) {
    ag_capture_repeat(cap$res_step, n_steps, cols = cols, key = key, z = z)
    last_val <- as.numeric(cap$res_value(cols = cols)$value)
  } else {
    m <- matrix(norm$mean); sc <- obs_scale(norm)
    ag_capture_repeat(cap$res_step, n_steps, cols = cols, key = key, z = z, m = m, sc = sc)
    last_val <- as.numeric(cap$res_value(cols = cols, m = m, sc = sc)$value)
  }
  rec <- ag_buffer_read(cap$res_rec)
  ar <- ppo_res_act_rows(ad)
  m <- function(i) matrix(rec[ar + i, ], n_env, n_steps)
  box <- isTRUE(ad$box)
  legal <- if (ad$legal) ag_buffer_read(cap$res_legal) else if (!box)
    matrix(1, ad$n_actions, n_env * n_steps)
  # box: act [act_dim, N] (unclipped), logp with the constant of the density
  act <- if (box) rec[seq_len(ar), , drop = FALSE] else matrix(rec[1L, ], n_env, n_steps)
  logp <- m(1)
  if (box) logp <- logp - 0.5 * ar * log(2 * pi)
  fsp <- nrow(rec) > ar + 5L
  list(act = act, logp = logp, val = m(2), rew = m(3), done = m(4), boot = m(5),
       keep = if (fsp) m(6), vs_pool = if (fsp) m(7), side = if (fsp) m(8),
       legal = legal, last_val = last_val)
}
