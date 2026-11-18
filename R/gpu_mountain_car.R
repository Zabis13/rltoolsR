# MountainCar-v0 and MountainCarContinuous-v0 as a batched ggml graph: the
# same physics as the C++ "mountain_car" / "mountain_car_continuous"
# (Gymnasium classic_control), in f32 on the device.
#
# State [n, 2] (x, velocity as two contiguous rows of n), so the downloaded
# state is the n x 2 observation matrix. The host turns actions into a force
# per game (discrete: (a - 1) * force, continuous: clip(a, -1, 1) * power), one
# graph per step computes the next state into the other buffer; reward,
# termination, truncation and auto-reset are decided on the host.

MOUNTAIN_CAR <- list(min_position = -1.2, max_position = 0.6, max_speed = 0.07,
                     force = 0.001, power = 0.0015, gravity = 0.0025)

mountain_car_gpu_env <- function(n, device, continuous) {
  k <- MOUNTAIN_CAR
  st <- new.env(parent = emptyenv())
  st$n <- n
  st$continuous <- continuous
  st$goal <- if (continuous) 0.45 else 0.5
  st$max_steps <- if (continuous) 999L else 200L
  st$steps_done <- integer(n)
  st$backend <- ttt_backend(device)

  st$ctx_state <- ggml_init(1024 * 1024, no_alloc = TRUE)
  cs <- st$ctx_state
  st$state <- lapply(1:2, function(i) ggml_new_tensor_2d(cs, GGML_TYPE_F32, n, 2))
  st$force <- ggml_new_tensor_1d(cs, GGML_TYPE_F32, n)
  st$buffer <- ggml_backend_alloc_ctx_tensors(cs, st$backend)

  st$ctx_graph <- ggml_init(8 * 1024 * 1024, no_alloc = TRUE)
  st$steps <- lapply(1:2, function(i) mountain_car_build_step(st, src = i, dst = 3L - i))
  st$cur <- 1L

  reg.finalizer(st, ttt_free, onexit = TRUE)

  name <- if (continuous) "mountain_car_continuous" else "mountain_car"
  env <- structure(list(
    st = st, name = name, n = n, device = device,
    mode = "single", n_players = 1L, max_steps = st$max_steps,
    observation_space = space_box(c(k$min_position, -k$max_speed),
                                  c(k$max_position, k$max_speed)),
    action_space = if (continuous) space_box(-1, 1) else space_discrete(3)
  ), class = c("rl_gpu_mountain_car", "rl_gpu_env"))
  rl_reset(env)
  env
}

mountain_car_build_step <- function(st, src, dst) {
  g <- st$ctx_graph
  n <- st$n
  k <- MOUNTAIN_CAR
  row <- function(t, j) ggml_view_1d(g, t, n, (j - 1) * n * 4)
  s <- st$state[[src]]
  x <- row(s, 1); v <- row(s, 2)

  v1 <- ggml_add(g, ggml_add(g, v, st$force),
                 ggml_scale(g, ggml_cos(g, ggml_scale(g, x, 3)), -k$gravity))
  v1 <- ggml_clamp(g, v1, -k$max_speed, k$max_speed)
  x1 <- ggml_clamp(g, ggml_add(g, x, v1), k$min_position, k$max_position)
  # at the left wall a negative velocity becomes 0
  at_wall <- ggml_step(g, ggml_scale_bias(g, x1, -1, k$min_position + 1e-6))
  backward <- ggml_step(g, ggml_neg(g, v1))
  v1 <- ggml_sub(g, v1, ggml_mul(g, v1, ggml_mul(g, at_wall, backward)))

  d <- st$state[[dst]]
  graph <- ggml_build_forward_expand(g, ggml_cpy(g, x1, row(d, 1)))
  ggml_graph_expand(graph, ggml_cpy(g, v1, row(d, 2)))
  sched <- ggml_backend_sched_new(list(st$backend), parallel = FALSE)
  ggml_backend_sched_alloc_graph(sched, graph)
  list(graph = graph, sched = sched)
}

# n x 2 initial states: x uniform in [-0.6, -0.4], velocity 0 (R's RNG)
mountain_car_initial <- function(k) cbind(stats::runif(k, -0.6, -0.4), 0)

# writes an n x 2 state matrix to the current buffer (also used by checks)
mountain_car_set_state <- function(env, state) {
  st <- env$st
  ggml_backend_tensor_set_data(st$state[[st$cur]], as.numeric(state))
  st$steps_done <- integer(st$n)
  invisible(env)
}

#' @export
rl_reset.rl_gpu_mountain_car <- function(env, seed = NULL) {
  if (!is.null(seed)) gpu_rng_seed(env$st, seed)
  obs <- gpu_rng(env$st, function() mountain_car_initial(env$n))
  mountain_car_set_state(env, obs)
  obs
}

#' @export
rl_step.rl_gpu_mountain_car <- function(env, action) {
  st <- env$st
  k <- MOUNTAIN_CAR
  if (length(action) != st$n) stop("action must have length ", st$n)
  if (st$continuous) {
    action <- as.numeric(action)
    if (anyNA(action)) stop("mountain_car_continuous: actions must not be NA")
    force <- pmin(pmax(action, -1), 1) * k$power
  } else {
    action <- as.integer(action)
    if (anyNA(action) || any(action < 0L | action > 2L)) stop("mountain_car: actions must be 0, 1 or 2")
    force <- (action - 1) * k$force
  }
  ggml_backend_tensor_set_data(st$force, force)
  step <- st$steps[[st$cur]]
  ggml_backend_sched_graph_compute(step$sched, step$graph)
  st$cur <- 3L - st$cur
  obs <- matrix(ggml_backend_tensor_get_data(st$state[[st$cur]]), st$n, 2)
  terminated <- obs[, 1] >= st$goal & obs[, 2] >= 0
  reward <- if (st$continuous) 100 * terminated - 0.1 * action^2 else rep(-1, st$n)
  st$steps_done <- st$steps_done + 1L
  truncated <- !terminated & st$steps_done >= st$max_steps
  # final_obs: the last observation of the games that ended, NA for the others
  res <- list(obs = obs, reward = reward, terminated = terminated, truncated = truncated,
              final_obs = obs * NA_real_)
  done <- which(terminated | truncated)
  if (length(done)) {
    res$final_obs[done, ] <- obs[done, ]
    obs[done, ] <- gpu_rng(st, function() mountain_car_initial(length(done)))
    ggml_backend_tensor_set_data(st$state[[st$cur]], as.numeric(obs))
    st$steps_done[done] <- 0L
    res$obs <- obs
  }
  res
}

#' @export
rl_legal_actions.rl_gpu_mountain_car <- function(env) NULL

#' @export
print.rl_gpu_mountain_car <- function(x, ...) {
  cat("<rl_gpu_env> ", x$name, " x ", x$n, " on ", x$device, "\n", sep = "")
  cat("  observation: ", format(x$observation_space), "\n", sep = "")
  cat("  action:      ", format(x$action_space), "\n", sep = "")
  invisible(x)
}
