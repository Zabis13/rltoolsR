# Pendulum-v1 as a batched ggml graph: the same physics as the C++
# "pendulum_gym" / "pendulum_gym_raw" (Gymnasium classic_control), in f32 on
# the device.
#
# State [n, 2] (theta, theta_dot as two contiguous rows of n). The host clips
# the action to a torque per game (raw: clip(a, -2, 2), scaled: 2 clip(a, -1, 1)),
# one graph per step computes the next state into the other buffer. The host
# keeps the downloaded state: the reward uses the state before the step, the
# observation [cos, sin, theta_dot] the state after it. Pendulum never
# terminates; truncation and auto-reset are decided on the host.

PENDULUM_GYM <- list(g = 10, max_speed = 8, max_torque = 2, dt = 0.05, m = 1, l = 1,
                     max_steps = 200L)

pendulum_gpu_env <- function(n, device, raw) {
  k <- PENDULUM_GYM
  st <- new.env(parent = emptyenv())
  st$n <- n
  st$raw <- raw
  st$steps_done <- integer(n)
  st$backend <- ttt_backend(device)

  st$ctx_state <- ggml_init(1024 * 1024, no_alloc = TRUE)
  cs <- st$ctx_state
  st$state <- lapply(1:2, function(i) ggml_new_tensor_2d(cs, GGML_TYPE_F32, n, 2))
  st$torque <- ggml_new_tensor_1d(cs, GGML_TYPE_F32, n)
  st$buffer <- ggml_backend_alloc_ctx_tensors(cs, st$backend)

  st$ctx_graph <- ggml_init(8 * 1024 * 1024, no_alloc = TRUE)
  st$steps <- lapply(1:2, function(i) pendulum_build_step(st, src = i, dst = 3L - i))
  st$cur <- 1L

  reg.finalizer(st, ttt_free, onexit = TRUE)

  a <- if (raw) k$max_torque else 1
  env <- structure(list(
    st = st, name = if (raw) "pendulum_gym_raw" else "pendulum_gym", n = n, device = device,
    mode = "single", n_players = 1L, max_steps = k$max_steps,
    observation_space = space_box(c(-1, -1, -k$max_speed), c(1, 1, k$max_speed)),
    action_space = space_box(-a, a)
  ), class = c("rl_gpu_pendulum", "rl_gpu_env"))
  rl_reset(env)
  env
}

pendulum_build_step <- function(st, src, dst) {
  g <- st$ctx_graph
  n <- st$n
  k <- PENDULUM_GYM
  row <- function(t, j) ggml_view_1d(g, t, n, (j - 1) * n * 4)
  s <- st$state[[src]]
  th <- row(s, 1); thd <- row(s, 2)

  # thd + (3 g / (2 l) sin(th) + 3 / (m l^2) u) dt
  acc <- ggml_add(g, ggml_scale(g, ggml_sin(g, th), 3 * k$g / (2 * k$l) * k$dt),
                  ggml_scale(g, st$torque, 3 / (k$m * k$l^2) * k$dt))
  thd1 <- ggml_clamp(g, ggml_add(g, thd, acc), -k$max_speed, k$max_speed)
  th1 <- ggml_add(g, th, ggml_scale(g, thd1, k$dt))

  d <- st$state[[dst]]
  graph <- ggml_build_forward_expand(g, ggml_cpy(g, th1, row(d, 1)))
  ggml_graph_expand(graph, ggml_cpy(g, thd1, row(d, 2)))
  sched <- ggml_backend_sched_new(list(st$backend), parallel = FALSE)
  ggml_backend_sched_alloc_graph(sched, graph)
  list(graph = graph, sched = sched)
}

# n x 2 initial states: theta uniform in [-pi, pi], theta_dot in [-1, 1] (R's RNG)
pendulum_initial <- function(k) cbind(stats::runif(k, -pi, pi), stats::runif(k, -1, 1))

pendulum_observe <- function(state) cbind(cos(state[, 1]), sin(state[, 1]), state[, 2])

# writes an n x 2 state matrix (theta, theta_dot) to the current buffer (also used by checks)
pendulum_set_state <- function(env, state) {
  st <- env$st
  ggml_backend_tensor_set_data(st$state[[st$cur]], as.numeric(state))
  st$host <- matrix(ggml_backend_tensor_get_data(st$state[[st$cur]]), st$n, 2)
  st$steps_done <- integer(st$n)
  invisible(env)
}

#' @export
rl_reset.rl_gpu_pendulum <- function(env, seed = NULL) {
  if (!is.null(seed)) gpu_rng_seed(env$st, seed)
  pendulum_set_state(env, gpu_rng(env$st, function() pendulum_initial(env$n)))
  pendulum_observe(env$st$host)
}

#' @export
rl_step.rl_gpu_pendulum <- function(env, action) {
  st <- env$st
  k <- PENDULUM_GYM
  action <- as.numeric(action)
  if (length(action) != st$n) stop("action must have length ", st$n)
  if (anyNA(action)) stop(env$name, ": actions must not be NA")
  u <- if (st$raw) pmin(pmax(action, -k$max_torque), k$max_torque) else
    k$max_torque * pmin(pmax(action, -1), 1)
  th <- ((st$host[, 1] + pi) %% (2 * pi)) - pi
  reward <- -(th^2 + 0.1 * st$host[, 2]^2 + 0.001 * u^2)

  ggml_backend_tensor_set_data(st$torque, u)
  step <- st$steps[[st$cur]]
  ggml_backend_sched_graph_compute(step$sched, step$graph)
  st$cur <- 3L - st$cur
  st$host <- matrix(ggml_backend_tensor_get_data(st$state[[st$cur]]), st$n, 2)
  st$steps_done <- st$steps_done + 1L
  truncated <- st$steps_done >= k$max_steps
  # final_obs: the last observation of the games that ended, NA for the others
  res <- list(obs = pendulum_observe(st$host), reward = reward,
              terminated = logical(st$n), truncated = truncated)
  res$final_obs <- res$obs * NA_real_
  done <- which(truncated)
  if (length(done)) {
    res$final_obs[done, ] <- res$obs[done, ]
    st$host[done, ] <- gpu_rng(st, function() pendulum_initial(length(done)))
    ggml_backend_tensor_set_data(st$state[[st$cur]], as.numeric(st$host))
    st$steps_done[done] <- 0L
    res$obs <- pendulum_observe(st$host)
  }
  res
}

#' @export
rl_legal_actions.rl_gpu_pendulum <- function(env) NULL

#' @export
print.rl_gpu_pendulum <- function(x, ...) {
  cat("<rl_gpu_env> ", x$name, " x ", x$n, " on ", x$device, "\n", sep = "")
  cat("  observation: ", format(x$observation_space), "\n", sep = "")
  cat("  action:      ", format(x$action_space), "\n", sep = "")
  invisible(x)
}
