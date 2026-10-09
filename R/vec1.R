# A single rl_env seen as a vectorized environment with one copy.
#
# Same contract as rl_vec_env: batched reset/step/legal_actions, same-step
# autoreset on terminated or truncated, and final_obs with the last
# observation of the finished episode (needed to bootstrap on truncation).
# The wrapped environment is owned by the wrapper: its state is not copied,
# so the caller must not step it directly afterwards.
rl_vec1 <- function(env) {
  check_env(env)
  structure(list(env = env, n = 1L, name = env$name, mode = env$mode,
                 n_players = env$n_players, max_steps = env$max_steps,
                 observation_space = env$observation_space,
                 action_space = env$action_space),
            class = "rl_vec1")
}

# The one place where a training loop turns its `env` argument into a
# vectorized environment.
as_vec_env <- function(env) {
  if (inherits(env, "rl_env")) return(rl_vec1(env))
  if (is.null(env$n)) stop("`env` must be an rl_env or a vectorized environment")
  env
}

#' @export
rl_reset.rl_vec1 <- function(env, seed = NULL) {
  matrix(rl_reset(env$env, seed), 1L)
}

#' @export
rl_step.rl_vec1 <- function(env, action) {
  r <- rl_step(env$env, as.numeric(action))
  final_obs <- matrix(r$obs, 1L)
  obs <- if (r$terminated || r$truncated) matrix(rl_reset(env$env), 1L) else final_obs
  list(obs = obs, reward = r$reward, terminated = r$terminated,
       truncated = r$truncated, final_obs = final_obs)
}

#' @export
rl_legal_actions.rl_vec1 <- function(env) {
  m <- rl_legal_actions(env$env)
  if (is.null(m)) NULL else matrix(m, 1L)
}

#' @export
print.rl_vec1 <- function(x, ...) {
  cat("<rl_vec1> ", x$name, " x 1\n", sep = "")
  cat("  observation: ", format(x$observation_space), "\n", sep = "")
  cat("  action:      ", format(x$action_space), "\n", sep = "")
  invisible(x)
}
