# Observations that stay on the environment's device.

#' Observations held on the environment's device
#'
#' GPU environments created with `device = "gpu"` (tic-tac-toe, cube) return
#' their observations from [rl_reset()] and [rl_step()] as an
#' `rl_device_obs`: a handle to the observation tensor in device memory, not
#' a matrix. Training reads it on the device (PPO copies it into its rollout
#' buffer without the host); code that needs the values on the host calls
#' `as.matrix()`, which downloads them. Nothing downloads implicitly.
#'
#' The tensor belongs to the environment and is overwritten by later steps:
#' an `rl_device_obs` is valid until the next [rl_step()] or [rl_reset()] of
#' its environment. Copy (or download) it before stepping again.
#'
#' @param x An `rl_device_obs`.
#' @param ... Unused.
#' @return `as.matrix()`: the `n x obs_dim` matrix the host path returns, one
#'   row per environment copy (same values, same order). `dim()`: `c(n, obs_dim)`.
#' @name rl_device_obs
#' @examples
#' \donttest{
#' if (ggmlR::ggml_vulkan_available()) {
#'   env <- rl_gpu_env("tictactoe", n = 4, device = "gpu")
#'   obs <- rl_reset(env)
#'   as.matrix(obs)
#' }
#' }
NULL

# `tensor`: f32 ggml tensor laid out [obs_dim, n] (column = environment copy),
# owned by `backend`.
new_rl_device_obs <- function(tensor, backend, n, obs_dim) {
  structure(list(tensor = tensor, backend = backend, n = as.integer(n),
                 obs_dim = as.integer(obs_dim)),
            class = "rl_device_obs")
}

#' @rdname rl_device_obs
#' @export
as.matrix.rl_device_obs <- function(x, ...) {
  t(matrix(ggml_backend_tensor_get_data(x$tensor, 0, x$n * x$obs_dim),
           x$obs_dim, x$n))
}

#' @rdname rl_device_obs
#' @export
dim.rl_device_obs <- function(x) c(x$n, x$obs_dim)

#' @rdname rl_device_obs
#' @export
print.rl_device_obs <- function(x, ...) {
  cat("<rl_device_obs> ", x$n, " x ", x$obs_dim,
      " on the environment's device (as.matrix() downloads)\n", sep = "")
  invisible(x)
}

# Host matrix of observations: downloads an rl_device_obs, passes a matrix.
obs_host <- function(obs) if (inherits(obs, "rl_device_obs")) as.matrix(obs) else obs
