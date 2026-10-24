#' Create an environment
#'
#' Environments run as compiled C++ code behind a shared contract:
#' `reset`, `step`, `legal_actions`, `save_state` and `load_state`.
#'
#' @param name Environment name, see [rl_env_names()].
#' @param seed Optional non-negative integer seed for the environment RNG.
#' @return An object of class `rl_env` with fields `name`, `mode`,
#'   `n_players`, `max_steps`, `observation_space` and `action_space`.
#' @export
#' @examples
#' env <- rl_env("pendulum", seed = 1)
#' obs <- rl_reset(env)
#' res <- rl_step(env, 0.5)
#' res$reward
rl_env <- function(name, seed = NULL) {
  stopifnot(is.character(name), length(name) == 1)
  ptr <- .Call(rlt_env_make, name, seed)
  spec <- .Call(rlt_env_spec, ptr)
  structure(c(list(ptr = ptr), spec), class = "rl_env")
}

#' List available environments
#'
#' @return Character vector of environment names.
#' @export
#' @examples
#' rl_env_names()
rl_env_names <- function() {
  .Call(rlt_env_names)
}

#' Reset an environment
#'
#' @param env An `rl_env`, `rl_vec_env` or `rl_gpu_env`.
#' @param seed Optional seed; `NULL` continues the current RNG stream.
#'   For `rl_vec_env`, copy `i` uses `seed + i - 1`. Ignored by
#'   `rl_gpu_env` (its random opponent uses R's RNG).
#' @return Numeric observation vector (`rl_env`) or `n x obs_dim` matrix
#'   (`rl_vec_env`, `rl_gpu_env`).
#' @export
#' @examples
#' env <- rl_env("pendulum", seed = 1)
#' rl_reset(env)
#' rl_reset(env, seed = 2)
rl_reset <- function(env, seed = NULL) UseMethod("rl_reset")

#' @export
rl_reset.rl_env <- function(env, seed = NULL) {
  .Call(rlt_env_reset, env$ptr, seed)
}

#' @export
rl_reset.rl_vec_env <- function(env, seed = NULL) {
  .Call(rlt_vec_reset, env$ptr, seed)
}

#' Take one step in an environment
#'
#' Episode ends when `terminated` (terminal state) or `truncated`
#' (step limit `env$max_steps` reached) is `TRUE`.
#'
#' A `rl_vec_env` resets finished copies in the same step: `obs` holds
#' the first observation of the new episode and `final_obs` the last
#' observation of the finished one (use it to bootstrap on truncation).
#'
#' @param env An `rl_env`, `rl_vec_env` or `rl_gpu_env`.
#' @param action For `rl_env`, numeric vector matching `env$action_space`.
#'   For `rl_vec_env`, `n x act_dim` matrix (or length-`n` vector when
#'   `act_dim` is 1). For `rl_gpu_env`, length-`n` vector of cell indices
#'   `0..8`.
#' @return List with `obs`, `reward`, `terminated` and `truncated`;
#'   `rl_vec_env` adds `final_obs`, `rl_gpu_env` adds `legal` (legal-action
#'   mask for the next move) and `player` (side to move). Vectorized fields
#'   have one row or element per copy.
#' @export
#' @examples
#' env <- rl_env("pendulum", seed = 1)
#' obs <- rl_reset(env)
#' res <- rl_step(env, 0.5)
#' str(res)
rl_step <- function(env, action) UseMethod("rl_step")

#' @export
rl_step.rl_env <- function(env, action) {
  .Call(rlt_env_step, env$ptr, action)
}

#' @export
rl_step.rl_vec_env <- function(env, action) {
  .Call(rlt_vec_step, env$ptr, action)
}

#' Legal actions in the current state
#'
#' @param env An `rl_env`, `rl_vec_env` or `rl_gpu_env`.
#' @return Logical mask over discrete actions (`n x n_actions` matrix for
#'   `rl_vec_env` and `rl_gpu_env`), or `NULL` when the action space has no notion of legal
#'   actions (e.g. continuous `Box`).
#' @export
#' @examples
#' # continuous action space: no legal-action mask
#' env <- rl_env("pendulum", seed = 1)
#' rl_legal_actions(env)
#' \donttest{
#' g <- rl_gpu_env("tictactoe", n = 2, device = "cpu")
#' obs <- rl_reset(g)
#' rl_legal_actions(g)
#' }
rl_legal_actions <- function(env) UseMethod("rl_legal_actions")

#' @export
rl_legal_actions.rl_env <- function(env) {
  .Call(rlt_env_legal_mask, env$ptr)
}

#' @export
rl_legal_actions.rl_vec_env <- function(env) {
  .Call(rlt_vec_legal_mask, env$ptr)
}

#' Create a vectorized environment
#'
#' Runs `n` copies of an environment in C++ and steps them with one call,
#' returning batched matrices ready for a neural network.
#'
#' @inheritParams rl_env
#' @param n Number of copies.
#' @param seed Optional non-negative integer seed; copy `i` uses
#'   `seed + i - 1`.
#' @return An object of class `rl_vec_env` with the fields of [rl_env()]
#'   plus `n`.
#' @export
#' @examples
#' venv <- rl_vec_env("pendulum", n = 4, seed = 1)
#' obs <- rl_reset(venv)
#' res <- rl_step(venv, rep(0, 4))
#' dim(res$obs)
rl_vec_env <- function(name, n, seed = NULL) {
  stopifnot(is.character(name), length(name) == 1)
  n <- as.integer(n)
  stopifnot(length(n) == 1, !is.na(n), n >= 1)
  ptr <- .Call(rlt_vec_make, name, n, seed)
  spec <- .Call(rlt_vec_spec, ptr)
  structure(c(list(ptr = ptr, n = n), spec), class = "rl_vec_env")
}

#' @export
print.rl_vec_env <- function(x, ...) {
  cat("<rl_vec_env> ", x$name, " x ", x$n, "\n", sep = "")
  cat("  mode:        ", x$mode, " (players: ", x$n_players, ")\n", sep = "")
  cat("  max_steps:   ", if (x$max_steps > 0) x$max_steps else "none", "\n", sep = "")
  cat("  observation: ", format(x$observation_space), "\n", sep = "")
  cat("  action:      ", format(x$action_space), "\n", sep = "")
  invisible(x)
}

#' Save and restore environment state
#'
#' Snapshots hold the environment state and step counter, not the RNG
#' state. Used by search algorithms such as MCTS.
#'
#' @param env An `rl_env`.
#' @param state Raw vector returned by `rl_save_state()`.
#' @return `rl_save_state()` returns a raw vector; `rl_load_state()`
#'   returns `env` invisibly.
#' @export
#' @examples
#' env <- rl_env("pendulum", seed = 1)
#' obs <- rl_reset(env)
#' s <- rl_save_state(env)
#' r1 <- rl_step(env, 1)$obs
#' rl_load_state(env, s)
#' r2 <- rl_step(env, 1)$obs
#' all.equal(r1, r2)
rl_save_state <- function(env) {
  check_env(env)
  .Call(rlt_env_save_state, env$ptr)
}

#' @rdname rl_save_state
#' @export
rl_load_state <- function(env, state) {
  check_env(env)
  .Call(rlt_env_load_state, env$ptr, state)
  invisible(env)
}

#' @export
print.rl_env <- function(x, ...) {
  cat("<rl_env> ", x$name, "\n", sep = "")
  cat("  mode:        ", x$mode, " (players: ", x$n_players, ")\n", sep = "")
  cat("  max_steps:   ", if (x$max_steps > 0) x$max_steps else "none", "\n", sep = "")
  cat("  observation: ", format(x$observation_space), "\n", sep = "")
  cat("  action:      ", format(x$action_space), "\n", sep = "")
  invisible(x)
}

check_env <- function(env) {
  if (!inherits(env, "rl_env")) stop("`env` must be an rl_env object")
}
