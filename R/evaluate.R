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
#' not change the episodes of the others. With `greedy = TRUE` the result
#' therefore does not depend on `n_envs`, and short episodes are not
#' over-represented. With `greedy = FALSE` actions are sampled from R's RNG
#' for the whole batch, so the sampled actions do depend on `n_envs`:
#' reproducibility is guaranteed only for `greedy = TRUE`.
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
#'   `s + n - 1`). `NULL` continues the environment RNG streams: a
#'   non-reproducible mode with different initial states at every call.
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
      res <- rl_step(env, rl_act(model, obs, legal, greedy))
      live <- !done
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

# Population standard deviation (divides by n), as in RLtools.
pop_sd <- function(x) sqrt(mean((x - mean(x))^2))
