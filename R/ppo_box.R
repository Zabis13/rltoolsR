# PPO for continuous (Box) actions: Gaussian policy with a state-independent
# log_std, separate actor and critic MLPs, orthogonal initialization.
#
# Actions are sampled unbounded and clipped to the action space only when
# sent to the environment; log-probabilities are of the unclipped action.

# Gains: sqrt(2) hidden, 0.01 policy head, 1 value head; log_std starts at 0.
ppo_box_net <- function(obs_dim, act_dim, hidden) {
  a1 <- ag_linear(obs_dim, hidden, "tanh", init = "orthogonal", gain = sqrt(2))
  a2 <- ag_linear(hidden, hidden, "tanh", init = "orthogonal", gain = sqrt(2))
  mu <- ag_linear(hidden, act_dim, init = "orthogonal", gain = 0.01)
  c1 <- ag_linear(obs_dim, hidden, "tanh", init = "orthogonal", gain = sqrt(2))
  c2 <- ag_linear(hidden, hidden, "tanh", init = "orthogonal", gain = sqrt(2))
  v  <- ag_linear(hidden, 1L, init = "orthogonal", gain = 1)
  log_std <- ag_param(matrix(0, act_dim, 1L))
  forward <- function(x) {
    list(mu = mu$forward(a2$forward(a1$forward(x))),
         value = v$forward(c2$forward(c1$forward(x))),
         log_std = log_std)
  }
  parameters <- function() list(
    a1_W = a1$W, a1_b = a1$b, a2_W = a2$W, a2_b = a2$b, mu_W = mu$W, mu_b = mu$b,
    c1_W = c1$W, c1_b = c1$b, c2_W = c2$W, c2_b = c2$b, v_W = v$W, v_b = v$b,
    log_std = log_std
  )
  structure(list(forward = forward, parameters = parameters,
                 captured = new.env(parent = emptyenv())),
            class = c("rl_ppo_box_net", "ag_layer"))
}

ppo_box_shapes <- function(meta) {
  h <- meta$hidden
  o <- meta$obs_dim
  a <- meta$act_dim
  list(a1_W = c(h, o), a1_b = c(h, 1L), a2_W = c(h, h), a2_b = c(h, 1L),
       mu_W = c(a, h), mu_b = c(a, 1L),
       c1_W = c(h, o), c1_b = c(h, 1L), c2_W = c(h, h), c2_b = c(h, 1L),
       v_W = c(1L, h), v_b = c(1L, 1L), log_std = c(a, 1L))
}

# Gaussian log-density of actions `a` (act_dim x B) per column.
gauss_logp <- function(a, mu, log_std) {
  z <- (a - mu) * exp(-log_std)
  -0.5 * colSums(z^2) - sum(log_std) - 0.5 * nrow(a) * log(2 * pi)
}

# act_dim x B actions -> B x act_dim matrix inside the action space.
clip_to_space <- function(a, space) {
  t(pmin(pmax(a, space$low), space$high))
}

# Box policies at collection: forward, noise, action, its log-density and the
# value as one recorded graph. The host sends the observations and a two-number
# key and gets back `a` [act_dim, B] (unclipped), `logp` and `value` per
# observation; mu and log_std stay on the device.
#
# The noise eps ~ N(0, 1) is drawn on the device from `key` (ggmlR::ag_rand),
# a = mu + eps * exp(log_std), and the density is taken from eps itself:
# logp = -1/2 sum(eps^2) - sum(log_std) - d/2 log(2 pi). The update recomputes
# it from the stored action as (a - mu) / exp(log_std) (box_logp_tensor()),
# which equals eps up to f32 rounding of a.
policy_act_box <- function(net, x, norm = NULL, act_dim, key = stats::runif(2L)) {
  key <- matrix(as.numeric(key)[1:2], 2L, 1L)
  B <- ncol(x)
  body <- function(h, key, B) {
    out <- net$forward(h)
    eps <- ag_rand(key, "normal", c(act_dim, B))
    a <- ag_add(out$mu, ag_mul(eps, ag_exp(out$log_std)))
    logp <- ag_sub(ag_scale(ag_sum(ag_mul(eps, eps), dim = 2L), -0.5), ag_sum(out$log_std))
    list(a = a, logp = logp, value = out$value)
  }
  cache <- net$captured
  if (is.null(cache)) {                # a net built without the cache
    res <- lapply(body(obs_input(x, norm), key, B), as.matrix)
  } else {
    params <- unname(net$parameters())
    # the batch size is a constant of the recording (the shape of eps); a
    # recording is redone exactly when the input shapes change, and reads it
    # from here
    cache$act_box_B <- B
    if (is.null(norm)) {
      if (is.null(cache$act_box))
        cache$act_box <- ag_capture(function(x, key) body(x, key, cache$act_box_B),
                                    params = params)
      res <- cache$act_box(x = x, key = key)
    } else {
      # clip is a constant of the graph, so one recording per value
      id <- paste0("act_box_norm_", format(norm$clip_obs, digits = 17))
      if (is.null(cache[[id]])) {
        clip <- norm$clip_obs
        cache[[id]] <- ag_capture(
          function(x, m, s, key) body(obs_normalize(x, m, s, clip), key, cache$act_box_B),
          params = params)
      }
      res <- cache[[id]](x = x, m = matrix(norm$mean), s = obs_scale(norm), key = key)
    }
  }
  a <- res$a
  list(a = a, logp = as.numeric(res$logp) - 0.5 * nrow(a) * log(2 * pi),
       value = as.numeric(res$value))
}

# Gaussian log-density in the graph, without the constant -d/2 log(2 pi):
# [1, B] tensor for actions `Aa` (act_dim x B tensor) under the forward output.
box_logp_tensor <- function(out, Aa) {
  z <- ag_mul(ag_sub(Aa, out$mu), ag_exp(ag_scale(out$log_std, -1)))
  ag_sub(ag_scale(ag_sum(ag_mul(z, z), dim = 2L), -0.5), ag_sum(out$log_std))
}

ppo_update_box <- function(net, opt, obs, act, old_logp, adv, ret,
                           clip, vf_coef, ent_coef, norm = NULL, max_grad_norm = NULL) {
  tk <- prof_start()
  B <- ncol(obs)
  d <- nrow(act)
  adv <- (adv - mean(adv)) / (stats::sd(adv) + 1e-8)
  # the graph drops the constant -d/2 log(2 pi) of the log-density
  const <- 0.5 * d * log(2 * pi)

  Aa <- ag_tensor(act)
  old <- ag_tensor(matrix(old_logp + const, 1))
  A <- ag_tensor(matrix(adv, 1))
  R <- ag_tensor(matrix(ret, 1))
  force(obs)
  tk <- prof_tick(tk, "update: prep")

  with_grad_tape({
    # Built INSIDE the tape: with normalization it is computed by ag_* ops on
    # the device and lives only in the pass pool, which with_grad_tape() resets
    # on entry -- built before, it was destroyed before the forward read it.
    x <- obs_input(obs, norm)
    out <- net$forward(x)
    sum_ls <- ag_sum(out$log_std)
    logp_a <- box_logp_tensor(out, Aa)
    ratio <- ag_exp(ag_sub(logp_a, old))
    s1 <- ag_mul(ratio, A)
    s2 <- ag_mul(ag_clamp(ratio, 1 - clip, 1 + clip), A)
    surr <- ag_sub(s1, ag_relu(ag_sub(s1, s2)))                 # min(s1, s2)
    pg_loss <- ag_scale(ag_mean(surr), -1)
    vf_loss <- ag_mse_loss(out$value, R)
    # entropy is sum(log_std) + const: only sum(log_std) carries a gradient
    loss <- ag_add(ag_add(pg_loss, ag_scale(vf_loss, vf_coef)), ag_scale(sum_ls, -ent_coef))
  })
  tk <- prof_tick(tk, "update: forward")
  grads <- backward(loss)
  tk <- prof_tick(tk, "update: backward")
  gn <- if (!is.null(max_grad_norm))
    clip_grad_norm(net$parameters(), grads, max_grad_norm)
  opt$step(grads)
  opt$zero_grad()
  tk <- prof_tick(tk, "update: adam")
  l <- c(pg = as.matrix(pg_loss)[1], vf = as.matrix(vf_loss)[1],
         ent = as.matrix(sum_ls)[1] + 0.5 * d * (1 + log(2 * pi)))
  if (!is.null(gn)) l <- c(l, grad_norm = gn)
  prof_tick(tk, "update: losses")
  l
}
