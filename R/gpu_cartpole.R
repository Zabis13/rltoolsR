# CartPole-v1 as a batched ggml graph: the same physics as the C++ "cartpole"
# (Gymnasium classic_control/cartpole.py, Euler), in f32 on the device.
#
# State [n, 4] (x, x_dot, theta, theta_dot as four contiguous rows of n), so
# every component is a plain 1-d view and the downloaded state is already the
# n x 4 observation matrix. One graph per step computes the next state into
# the other buffer; termination, truncation and auto-reset are decided on the
# host from the downloaded observation (four floats per game), and reset games
# are written back with one upload.

CARTPOLE <- list(gravity = 9.8, masscart = 1.0, masspole = 0.1, length = 0.5,
                 force_mag = 10.0, tau = 0.02, x_threshold = 2.4,
                 theta_threshold = 12 * 2 * pi / 360, max_steps = 500L)

cartpole_gpu_env <- function(n, device) {
  k <- CARTPOLE
  st <- new.env(parent = emptyenv())
  st$n <- n
  st$steps_done <- integer(n)
  st$backend <- ttt_backend(device)

  st$ctx_state <- ggml_init(1024 * 1024, no_alloc = TRUE)
  cs <- st$ctx_state
  st$state  <- lapply(1:2, function(i) ggml_new_tensor_2d(cs, GGML_TYPE_F32, n, 4))
  st$F      <- ggml_new_tensor_2d(cs, GGML_TYPE_F32, 1, 2)
  st$action <- ggml_new_tensor_1d(cs, GGML_TYPE_I32, n)
  st$buffer <- ggml_backend_alloc_ctx_tensors(cs, st$backend)
  ggml_backend_tensor_set_data(st$F, c(-k$force_mag, k$force_mag))

  st$ctx_graph <- ggml_init(8 * 1024 * 1024, no_alloc = TRUE)
  st$steps <- lapply(1:2, function(i) cartpole_build_step(st, src = i, dst = 3L - i))
  st$cur <- 1L

  reg.finalizer(st, ttt_free, onexit = TRUE)

  env <- structure(list(
    st = st, name = "cartpole", n = n, device = device,
    mode = "single", n_players = 1L, max_steps = k$max_steps,
    observation_space = space_box(c(-2 * k$x_threshold, -Inf, -2 * k$theta_threshold, -Inf),
                                  c(2 * k$x_threshold, Inf, 2 * k$theta_threshold, Inf)),
    action_space = space_discrete(2)
  ), class = c("rl_gpu_cartpole", "rl_gpu_env"))
  rl_reset(env)
  env
}

cartpole_build_step <- function(st, src, dst) {
  g <- st$ctx_graph
  n <- st$n
  k <- CARTPOLE
  total_mass <- k$masspole + k$masscart
  pml <- k$masspole * k$length
  row <- function(t, j) ggml_view_1d(g, t, n, (j - 1) * n * 4)
  s <- st$state[[src]]
  x <- row(s, 1); xd <- row(s, 2); th <- row(s, 3); thd <- row(s, 4)

  force <- ggml_reshape_1d(g, ggml_get_rows(g, st$F, st$action), n)
  c_ <- ggml_cos(g, th)
  s_ <- ggml_sin(g, th)
  temp <- ggml_scale(g, ggml_add(g, force, ggml_scale(g, ggml_mul(g, ggml_sqr(g, thd), s_), pml)),
                     1 / total_mass)
  num <- ggml_sub(g, ggml_scale(g, s_, k$gravity), ggml_mul(g, c_, temp))
  den <- ggml_scale_bias(g, ggml_sqr(g, c_), -k$length * k$masspole / total_mass,
                         k$length * 4 / 3)
  thetaacc <- ggml_div(g, num, den)
  xacc <- ggml_sub(g, temp, ggml_scale(g, ggml_mul(g, thetaacc, c_), pml / total_mass))

  nxt <- list(ggml_add(g, x, ggml_scale(g, xd, k$tau)),
              ggml_add(g, xd, ggml_scale(g, xacc, k$tau)),
              ggml_add(g, th, ggml_scale(g, thd, k$tau)),
              ggml_add(g, thd, ggml_scale(g, thetaacc, k$tau)))
  d <- st$state[[dst]]
  graph <- NULL
  for (j in 1:4) {
    cp <- ggml_cpy(g, nxt[[j]], row(d, j))
    if (is.null(graph)) graph <- ggml_build_forward_expand(g, cp) else ggml_graph_expand(graph, cp)
  }
  sched <- ggml_backend_sched_new(list(st$backend), parallel = FALSE)
  ggml_backend_sched_alloc_graph(sched, graph)
  list(graph = graph, sched = sched)
}

# n x 4 initial states, uniform in [-0.05, 0.05] (R's RNG)
cartpole_initial <- function(k) matrix(stats::runif(4 * k, -0.05, 0.05), k, 4)

# writes an n x 4 state matrix to the current buffer (also used by checks)
cartpole_set_state <- function(env, state) {
  st <- env$st
  ggml_backend_tensor_set_data(st$state[[st$cur]], as.numeric(state))
  st$steps_done <- integer(st$n)
  invisible(env)
}

#' @export
rl_reset.rl_gpu_cartpole <- function(env, seed = NULL) {
  obs <- cartpole_initial(env$n)
  cartpole_set_state(env, obs)
  obs
}

#' @export
rl_step.rl_gpu_cartpole <- function(env, action) {
  st <- env$st
  action <- as.integer(action)
  if (length(action) != st$n) stop("action must have length ", st$n)
  if (anyNA(action) || any(action < 0L | action > 1L)) stop("cartpole: actions must be 0 or 1")
  ggml_backend_tensor_set_data(st$action, action)
  step <- st$steps[[st$cur]]
  ggml_backend_sched_graph_compute(step$sched, step$graph)
  st$cur <- 3L - st$cur
  obs <- matrix(ggml_backend_tensor_get_data(st$state[[st$cur]]), st$n, 4)
  k <- CARTPOLE
  terminated <- abs(obs[, 1]) > k$x_threshold | abs(obs[, 3]) > k$theta_threshold
  st$steps_done <- st$steps_done + 1L
  truncated <- !terminated & st$steps_done >= k$max_steps
  res <- list(obs = obs, reward = rep(1, st$n), terminated = terminated,
              truncated = truncated)
  done <- which(terminated | truncated)
  if (length(done)) {
    res$final_obs <- obs
    obs[done, ] <- cartpole_initial(length(done))
    ggml_backend_tensor_set_data(st$state[[st$cur]], as.numeric(obs))
    st$steps_done[done] <- 0L
    res$obs <- obs
  }
  res
}

#' @export
rl_legal_actions.rl_gpu_cartpole <- function(env) NULL

#' @export
print.rl_gpu_cartpole <- function(x, ...) {
  cat("<rl_gpu_env> cartpole x ", x$n, " on ", x$device, "\n", sep = "")
  cat("  observation: ", format(x$observation_space), "\n", sep = "")
  cat("  action:      ", format(x$action_space), "\n", sep = "")
  invisible(x)
}
