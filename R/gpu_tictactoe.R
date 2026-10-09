# Tic-tac-toe as a batched ggml graph.
#
# N boards live on the device as a [9, N] f32 tensor (+1 = X, -1 = O, 0 =
# empty) next to a [1, N] tensor with the side to move. One step is one graph:
# the move is a one-hot row of a 9x9 identity (get_rows), wins come from one
# matmul against the 9x8 line matrix, and the next state is written back into
# the state tensors with ggml_cpy, so boards never leave the device between
# steps. Only actions go up and observations/rewards come down.
#
# Every quantity the graph tests is an integer, so indicators are clamp(k, 0, 1)
# instead of ggml_step: exact, and supported by every backend.

ttt_lines <- function() {
  lines <- list(c(1, 2, 3), c(4, 5, 6), c(7, 8, 9),
                c(1, 4, 7), c(2, 5, 8), c(3, 6, 9),
                c(1, 5, 9), c(3, 5, 7))
  m <- matrix(0, 9, 8)
  for (j in seq_along(lines)) m[lines[[j]], j] <- 1
  m
}

ttt_backend <- function(device) {
  if (device == "gpu") ggml_vulkan_init(0L) else ggml_backend_cpu_init()
}

# indicator of an integer-valued tensor being >= 1
ttt_ind <- function(ctx, x) ggml_clamp(ctx, x, 0, 1)

# Apply `onehot` moves for side `sign` ([1, N]) where `active` ([1, N]) is 1.
# Returns the new board, whether the move was illegal and whether it won.
ttt_move <- function(ctx, board, onehot, sign, active, L) {
  occupied <- ggml_sqr(ctx, board)
  illegal <- ggml_mul(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, onehot, occupied)), active)
  apply_mask <- ggml_mul(ctx, ggml_mul(ctx, sign, active), ggml_scale_bias(ctx, illegal, -1, 1))
  new_board <- ggml_add(ctx, board, ggml_mul(ctx, onehot, apply_mask))
  lines <- ggml_mul_mat(ctx, L, new_board)                      # [8, N]
  own <- ggml_mul(ctx, lines, sign)
  win <- ttt_ind(ctx, ggml_sum_rows(ctx, ggml_clamp(ctx, ggml_scale_bias(ctx, own, 1, -2), 0, 1)))
  win <- ggml_mul(ctx, win, active)
  list(board = new_board, illegal = illegal, win = win)
}

ttt_full <- function(ctx, board) {
  ggml_clamp(ctx, ggml_scale_bias(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, board)), 1, -8), 0, 1)
}

#' Create a batched GPU environment
#'
#' Runs `n` copies of an environment as a ggml compute graph on the GPU
#' (Vulkan) or CPU. The state stays on the device between steps.
#'
#' Currently available: `"tictactoe"`. Observations are canonical: the
#' side to move sees its own marks as `+1` and the opponent's as `-1`.
#' Actions are cell indices `0..8` (row-major). An illegal move ends the
#' game with reward `-1`. Finished games are reset in the same step.
#'
#' @param name Environment name.
#' @param n Number of parallel games.
#' @param opponent `"self"`: both sides are played by the caller, the side
#'   to move alternates and `reward` is for the side that just moved.
#'   `"random"`: the caller always plays X and a uniformly random legal
#'   opponent answers inside the same graph; `reward` is for X. Its moves
#'   use R's RNG, so `set.seed()` makes them reproducible.
#' @param device `"cpu"` (the default), `"auto"` (GPU if a Vulkan device is
#'   available, else CPU) or `"gpu"`. CPU is the default even when the policy
#'   trains on the GPU: observations go through R to the network anyway, and a
#'   step on the GPU pays an upload, a launch and five downloads. Measured (PPO,
#'   256 games, 9 steps per iteration, RX 9070): env 2.0 ms per iteration on the
#'   CPU against 8.7 ms on the GPU, which made GPU collection slower than CPU
#'   collection (18.7 against 15.2 ms). The GPU pays off once the rollout stays
#'   on the device.
#' @return An object of class `rl_gpu_env`.
#' @export
#' @examples
#' \donttest{
#' env <- rl_gpu_env("tictactoe", n = 4, device = "cpu")
#' obs <- rl_reset(env)
#' res <- rl_step(env, c(0, 4, 8, 2))
#' }
rl_gpu_env <- function(name, n, opponent = c("self", "random"),
                       device = c("cpu", "auto", "gpu")) {
  name <- match.arg(name, "tictactoe")
  opponent <- match.arg(opponent)
  device <- rl_resolve_device(match.arg(device))
  n <- as.integer(n)
  stopifnot(length(n) == 1, !is.na(n), n >= 1)

  st <- new.env(parent = emptyenv())
  st$n <- n
  st$opponent <- opponent
  st$backend <- ttt_backend(device)

  # Persistent tensors: double-buffered state, constants and per-step inputs.
  st$ctx_state <- ggml_init(4 * 1024 * 1024, no_alloc = TRUE)
  cs <- st$ctx_state
  st$board  <- lapply(1:2, function(i) ggml_new_tensor_2d(cs, GGML_TYPE_F32, 9, n))
  st$player <- lapply(1:2, function(i) ggml_new_tensor_2d(cs, GGML_TYPE_F32, 1, n))
  st$L      <- ggml_new_tensor_2d(cs, GGML_TYPE_F32, 9, 8)
  st$I9     <- ggml_new_tensor_2d(cs, GGML_TYPE_F32, 9, 9)
  st$action <- ggml_new_tensor_1d(cs, GGML_TYPE_I32, n)
  st$noise  <- ggml_new_tensor_2d(cs, GGML_TYPE_F32, 9, n)
  st$buffer <- ggml_backend_alloc_ctx_tensors(cs, st$backend)
  ggml_backend_tensor_set_data(st$L, as.numeric(ttt_lines()))
  ggml_backend_tensor_set_data(st$I9, as.numeric(diag(9)))

  # Two step graphs, 1 -> 2 and 2 -> 1, each with its own scheduler. A graph
  # never writes the tensors it reads: the Vulkan graph optimizer reorders
  # nodes by data dependencies only and does not see a write-after-read on a
  # leaf, so an in-place ggml_cpy back into the state is unsafe.
  st$ctx_graph <- ggml_init(32 * 1024 * 1024, no_alloc = TRUE)
  st$steps <- lapply(1:2, function(i) ttt_build_step(st, src = i, dst = 3L - i))
  st$cur <- 1L

  reg.finalizer(st, ttt_free, onexit = TRUE)

  env <- structure(list(
    st = st, name = name, n = n, device = device, opponent = opponent,
    mode = "sequential", n_players = 2L, max_steps = 9L,
    observation_space = space_box(-1, 1, shape = 9),
    action_space = space_discrete(9)
  ), class = "rl_gpu_env")
  rl_reset(env)
  env
}

# Build the graph that reads state buffer `src` and writes buffer `dst`.
ttt_build_step <- function(st, src, dst) {
  g <- st$ctx_graph
  board0 <- st$board[[src]]
  player <- st$player[[src]]
  ones <- ggml_sqr(g, player)  # [1, N] of 1: player is always +-1
  onehot <- ggml_get_rows(g, st$I9, st$action)
  m1 <- ttt_move(g, board0, onehot, player, ones, st$L)
  done <- ttt_ind(g, ggml_add(g, ggml_add(g, m1$illegal, m1$win), ttt_full(g, m1$board)))
  reward <- ggml_sub(g, m1$win, m1$illegal)
  board <- m1$board

  if (st$opponent == "random") {
    legal <- ggml_scale_bias(g, ggml_sqr(g, board), -1, 1)
    score <- ggml_add(g, ggml_mul(g, st$noise, legal), legal)
    onehot2 <- ggml_get_rows(g, st$I9, ggml_argmax(g, score))
    active2 <- ggml_scale_bias(g, done, -1, 1)
    m2 <- ttt_move(g, board, onehot2, ggml_neg(g, player), active2, st$L)
    board <- m2$board
    done <- ttt_ind(g, ggml_add(g, ggml_add(g, done, m2$win), ttt_full(g, board)))
    reward <- ggml_sub(g, reward, m2$win)
    next_player <- ones
  } else {
    not_done <- ggml_scale_bias(g, done, -1, 1)
    next_player <- ggml_add(g, ggml_mul(g, ggml_neg(g, player), not_done), done)
  }

  next_board <- ggml_mul(g, board, ggml_scale_bias(g, done, -1, 1))
  out <- list(
    obs = ggml_mul(g, next_board, next_player),
    legal = ggml_scale_bias(g, ggml_sqr(g, next_board), -1, 1),
    reward = reward,
    done = done,
    player = next_player
  )
  for (t in out) ggml_set_output(t)

  graph <- ggml_build_forward_expand(g, out$obs)
  for (t in out[-1]) ggml_graph_expand(graph, t)
  ggml_graph_expand(graph, ggml_cpy(g, next_board, st$board[[dst]]))
  ggml_graph_expand(graph, ggml_cpy(g, next_player, st$player[[dst]]))

  sched <-ggml_backend_sched_new(list(st$backend), parallel = FALSE)
  ggml_backend_sched_alloc_graph(sched, graph)
  list(graph = graph, sched = sched, out = out)
}

ttt_free <- function(st) {
  for (s in st$steps) ggml_backend_sched_free(s$sched)
  st$steps <- NULL
  if (!is.null(st$buffer)) ggml_backend_buffer_free(st$buffer)
  if (!is.null(st$ctx_graph)) ggml_free(st$ctx_graph)
  if (!is.null(st$ctx_state)) ggml_free(st$ctx_state)
  if (!is.null(st$backend)) ggml_backend_free(st$backend)
  st$buffer <- st$ctx_graph <- st$ctx_state <- st$backend <- NULL
}

#' @export
rl_reset.rl_gpu_env <- function(env, seed = NULL) {
  st <- env$st
  ggml_backend_tensor_set_data(st$board[[st$cur]], numeric(9 * st$n))
  ggml_backend_tensor_set_data(st$player[[st$cur]], rep(1, st$n))
  matrix(0, st$n, 9)
}

#' @export
rl_step.rl_gpu_env <- function(env, action) {
  st <- env$st
  action <- as.integer(action)
  if (length(action) != st$n) stop("action must have length ", st$n)
  if (anyNA(action) || any(action < 0L | action > 8L)) stop("actions must be cell indices 0..8")
  ggml_backend_tensor_set_data(st$action, action)
  if (st$opponent == "random") {
    ggml_backend_tensor_set_data(st$noise, stats::runif(9 * st$n))
  }
  step <- st$steps[[st$cur]]
  ggml_backend_sched_graph_compute(step$sched, step$graph)
  st$cur <- 3L - st$cur
  out <- step$out
  list(
    obs = t(matrix(ggml_backend_tensor_get_data(out$obs), 9, st$n)),
    reward = ggml_backend_tensor_get_data(out$reward),
    terminated = ggml_backend_tensor_get_data(out$done) > 0.5,
    truncated = logical(st$n),
    legal = t(matrix(ggml_backend_tensor_get_data(out$legal), 9, st$n) > 0.5),
    player = as.integer(ggml_backend_tensor_get_data(out$player))
  )
}

#' @export
rl_legal_actions.rl_gpu_env <- function(env) {
  st <- env$st
  board <- matrix(ggml_backend_tensor_get_data(st$board[[st$cur]]), 9, st$n)
  t(board == 0)
}

#' @export
print.rl_gpu_env <- function(x, ...) {
  cat("<rl_gpu_env> ", x$name, " x ", x$n, " on ", x$device,
      " (opponent: ", x$opponent, ")\n", sep = "")
  cat("  observation: ", format(x$observation_space), "\n", sep = "")
  cat("  action:      ", format(x$action_space), "\n", sep = "")
  invisible(x)
}
