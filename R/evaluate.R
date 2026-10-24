# Policy evaluation: mean return over a fixed number of episodes
# (rl_tools/rl/utils/evaluation), run on a vectorized environment.
#
# Every episode is a slot with its own seed. A batch of slots is reset
# together and each copy counts only its first episode: the autoreset that
# follows is ignored, so short episodes are not over-represented. Results are
# stored by slot index, not by the order in which episodes finish.

#' Evaluate a policy
#'
#' Runs `n_episodes` episodes with [rl_act()] and summarizes their returns
#' and lengths. Episodes run in batches of `n_envs` copies of a vectorized
#' environment, one batched forward pass per step.
#'
#' Episode `i` is reset with seed `seed + i - 1` and counts exactly one
#' episode, from reset to its end: copies that finish early keep stepping
#' (one batch, one forward pass) but their rewards and steps after the first
#' episode are not counted, and a batch stops as soon as all its episodes
#' have ended. Every copy has its own RNG, so an autoreset of one copy does
#' not change the episodes of the others. The result therefore does not
#' depend on `n_envs`, and short episodes are not over-represented.
#'
#' With `greedy = FALSE` every episode also samples its actions from its own
#' RNG stream (L'Ecuyer-CMRG, independent streams from
#' [parallel::nextRNGStream()]), advanced only while the episode runs, so the
#' sampled episodes do not depend on `n_envs` either. The forward pass stays
#' one per step for the whole batch. The caller's RNG state (kind and
#' `.Random.seed`) is left as it was. On the GPU, batches of different sizes
#' can differ in the last float32 digit, which may flip a sampled action that
#' lies on a boundary; on the CPU the results match exactly.
#'
#' An episode ends when it is terminated, or truncated by the step limit of
#' the environment or by `max_steps`. Truncated episodes count with the
#' return accumulated so far and are marked in the `episodes` attribute.
#'
#' Observation normalization uses the statistics frozen in the model
#' metadata; evaluation never updates them.
#'
#' @param model A model accepted by [rl_act()].
#' @param env Environment name (see [rl_env_names()]), or an `rl_env` /
#'   `rl_vec_env` used only for evaluation: it is reset and stepped here, so
#'   do not pass the training environment.
#' @param n_episodes Number of episodes.
#' @param max_steps Step limit per episode. `NULL` (default) uses the
#'   environment's own limit `env$max_steps`, which keeps results comparable
#'   with the literature (e.g. 200 for `pendulum`).
#' @param greedy `TRUE` (default) for the deterministic policy (mean or most
#'   likely action), `FALSE` to sample actions.
#' @param seed Non-negative integer seed of the first episode (default
#'   `1000000`, offset from typical training seeds); episodes use
#'   seeds `seed` to `seed + n_episodes - 1`, the same initial states at
#'   every evaluation. Keep this range apart from the training seeds (a
#'   training `rl_vec_env` with seed `s` and `n` copies uses `s` to
#'   `s + n - 1`). The sampling streams of `greedy = FALSE` are derived from
#'   `seed` too. `NULL` continues the environment RNG streams: a
#'   non-reproducible mode with different initial states at every call; with
#'   `greedy = FALSE` the sampling streams are then drawn from R's RNG, so
#'   [set.seed()] before the call reproduces the sampled actions.
#' @param n_envs Copies stepped together when `env` is a name. Ignored for an
#'   environment object, which uses its own number of copies.
#' @return Named numeric vector `return_mean`, `return_std`, `length_mean`,
#'   `length_std`, usable as the result of `eval_fn` in [rl_train()]. The
#'   standard deviations are population ones (divided by `n_episodes`, as in
#'   RLtools), so one episode gives 0, not `NA`. Attribute `episodes` is a
#'   data frame with one row per episode in slot order: `return`, `length`
#'   and `truncated`; compute sample statistics and confidence intervals
#'   from it.
#' @seealso [rl_act()], [rl_train()]
#' @export
#' @examples
#' \donttest{
#' env <- rl_vec_env("pendulum", n = 4, seed = 1)
#' m <- rl_ppo_train(env, iterations = 1, device = "cpu", verbose = FALSE)
#' res <- rl_evaluate(m, "pendulum", n_episodes = 8)
#' res
#' attr(res, "episodes")
#' }
rl_evaluate <- function(model, env, n_episodes = 10, max_steps = NULL,
                        greedy = TRUE, seed = 1000000L, n_envs = min(n_episodes, 256)) {
  n_episodes <- as.integer(n_episodes)
  if (length(n_episodes) != 1L || is.na(n_episodes) || n_episodes < 1L)
    stop("rl_evaluate(): `n_episodes` must be a positive integer", call. = FALSE)
  if (is.character(env)) {
    env <- rl_vec_env(env, n = n_envs)
  } else {
    env <- as_vec_env(env)
    if (inherits(env, "rl_gpu_env"))
      stop("rl_evaluate(): rl_gpu_env is not supported", call. = FALSE)
  }
  if (env$n_players != 1L)
    stop("rl_evaluate(): only single-player environments are supported", call. = FALSE)
  if (is.null(max_steps)) max_steps <- env$max_steps
  if (length(max_steps) != 1L || is.na(max_steps) || max_steps < 1)
    stop("rl_evaluate(): the environment has no step limit, set `max_steps`", call. = FALSE)
  discrete <- !identical(model$meta$action_type, "box")

  if (!greedy) {
    # seed = NULL: the streams follow the caller's RNG, drawn before it is saved
    base <- if (is.null(seed)) sample.int(.Machine$integer.max, 1L) else seed
    restore <- rng_snapshot()
    on.exit(restore(), add = TRUE)
    streams <- slot_streams(base, n_episodes)
  }

  ret <- numeric(n_episodes)
  len <- integer(n_episodes)
  trunc <- logical(n_episodes)
  n <- env$n
  for (start in seq(1L, n_episodes, by = n)) {
    k <- min(n, n_episodes - start + 1L)
    obs <- rl_reset(env, if (!is.null(seed)) seed + start - 1L)
    r_b <- numeric(n)
    l_b <- integer(n)
    tr_b <- logical(n)
    done <- seq_len(n) > k                   # copies beyond the last slot
    t <- 0L
    while (!all(done) && t < max_steps) {
      t <- t + 1L
      legal <- if (discrete) rl_legal_actions(env)
      live <- !done
      act <- if (greedy) {
        rl_act(model, obs, legal, greedy = TRUE)
      } else {
        # one batched forward, then each live slot samples on the host
        slots <- which(live)
        ep <- start - 1L + slots
        d <- with_policy_modes(model, policy_dist(model, obs, legal))
        s <- sample_slots(d, streams[ep], slots)
        streams[ep] <- s$streams
        if (discrete) s$act else clip_to_space(s$act, model$meta$action_space)
      }
      res <- rl_step(env, act)
      r_b[live] <- r_b[live] + res$reward[live]
      l_b[live] <- l_b[live] + 1L
      tr_b <- tr_b | (live & res$truncated & !res$terminated)
      done <- done | res$terminated | res$truncated
      obs <- res$obs
    }
    tr_b[!done] <- TRUE                      # cut by `max_steps`
    idx <- start - 1L + seq_len(k)
    ret[idx] <- r_b[seq_len(k)]
    len[idx] <- l_b[seq_len(k)]
    trunc[idx] <- tr_b[seq_len(k)]
  }

  out <- c(return_mean = mean(ret), return_std = pop_sd(ret),
           length_mean = mean(len), length_std = pop_sd(len))
  attr(out, "episodes") <- data.frame(return = ret, length = len, truncated = trunc)
  out
}

# Sampled actions for one batch, each live slot from its own RNG stream.
# `d` is a policy_dist() result for the whole batch; `streams` are the
# .Random.seed vectors of the slots in `slots` (column indices of `d`). A slot
# draws only while it is live, so its stream does not depend on how long the
# other slots of the batch run. Other columns get the greedy action (they are
# stepped but not counted). Returns the actions (0-based integers, or the
# [act_dim, B] matrix before clipping) and the advanced streams.
sample_slots <- function(d, streams, slots) {
  draw <- function(i, f) {
    assign(".Random.seed", streams[[i]], envir = globalenv())
    x <- f()
    streams[[i]] <<- get(".Random.seed", envir = globalenv())
    x
  }
  if (!is.null(d$logits)) {
    x <- d$logits
    act <- max.col(t(x), ties.method = "first") - 1L
    for (i in seq_along(slots)) {
      j <- slots[i]
      # Gumbel-max, as ag_sample_categorical()
      g <- draw(i, function() -log(-log(stats::runif(nrow(x)))))
      act[j] <- which.max(x[, j] + g) - 1L
    }
  } else {
    act <- d$mu
    log_std <- matrix(d$log_std, nrow = nrow(act))
    for (i in seq_along(slots)) {
      j <- slots[i]
      ls <- log_std[, min(j, ncol(log_std))]
      # mu + eps * exp(log_std), as ag_sample_normal()
      act[, j] <- act[, j] + draw(i, function() stats::rnorm(nrow(act))) * exp(ls)
    }
  }
  list(act = act, streams = streams)
}

# Independent L'Ecuyer-CMRG streams, one per episode, from `base`: episode i
# gets the i-th stream after the one set.seed(base) gives, so it depends only
# on `base` and i (not on the batching), and its RNG is not the seed used to
# reset the episode. Call it between rng_snapshot() and its restore.
slot_streams <- function(base, n) {
  set.seed(base, kind = "L'Ecuyer-CMRG")
  s <- get(".Random.seed", envir = globalenv())
  out <- vector("list", n)
  for (i in seq_len(n)) {
    s <- parallel::nextRNGStream(s)
    out[[i]] <- s
  }
  out
}

# Snapshot of the caller's RNG: kind and .Random.seed. The returned function
# puts both back, or removes .Random.seed if there was none, for on.exit().
rng_snapshot <- function() {
  kind <- RNGkind()
  has_seed <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  if (has_seed) seed <- get(".Random.seed", envir = globalenv())
  function() {
    # RNGkind() reseeds, so the saved .Random.seed is written after it
    RNGkind(kind[1], kind[2], kind[3])
    if (has_seed) assign(".Random.seed", seed, envir = globalenv())
    else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE))
      rm(".Random.seed", envir = globalenv())
  }
}

# Population standard deviation (divides by n), as in RLtools.
pop_sd <- function(x) sqrt(mean((x - mean(x))^2))
