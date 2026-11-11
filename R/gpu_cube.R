# N x N x N cube as a batched ggml environment.
#
# Geometry ported from cayleyR (src/cube_nnn.h, MIT, same author): a move is
# an axis, a layer along it and a quarter turn; stickers are numbered face by
# face in the order U R F D L B, within a face row by row as seen from outside.
# Moves are 6N quarter turns, in pairs (move, inverse): U U' R R' F F' D D'
# L L' B B', then the inner layers (on 3x3 M M' E E' S S'). The inverse of
# move a (0-based) is bitwXor(a, 1). Index 6N is the identity, used only
# internally (walks of different lengths in one batch).
#
# State on the device: one-hot colours [S, 6, n] with S = 6N^2 stickers, which
# is also the observation (S * 6 values per game). A move is a permutation
# matrix per game: get_rows from the table of all moves by action, then one
# batched mul_mat -- only ops every backend has, no index casts.

cube_faces <- c(U = 0L, R = 1L, F = 2L, D = 3L, L = 4L, B = 5L)

# 0-based flat index of stickers at (x, y, z) facing `face`
cube_sticker_index <- function(n, x, y, z, face) {
  m <- n - 1L
  row <- integer(length(face)); col <- integer(length(face))
  f <- face == 0L; row[f] <- z[f];     col[f] <- x[f]       # U
  f <- face == 3L; row[f] <- m - z[f]; col[f] <- x[f]       # D
  f <- face == 2L; row[f] <- m - y[f]; col[f] <- x[f]       # F
  f <- face == 5L; row[f] <- m - y[f]; col[f] <- m - x[f]   # B
  f <- face == 1L; row[f] <- m - y[f]; col[f] <- m - z[f]   # R
  f <- face == 4L; row[f] <- m - y[f]; col[f] <- z[f]       # L
  face * n * n + row * n + col
}

# all stickers in index order: data frame x, y, z, face
cube_stickers <- function(n) {
  g <- expand.grid(x = 0:(n - 1L), y = 0:(n - 1L), z = 0:(n - 1L))
  m <- n - 1L
  parts <- list(
    cbind(g[g$y == m, ], face = 0L), cbind(g[g$y == 0, ], face = 3L),
    cbind(g[g$z == m, ], face = 2L), cbind(g[g$z == 0, ], face = 5L),
    cbind(g[g$x == m, ], face = 1L), cbind(g[g$x == 0, ], face = 4L))
  s <- do.call(rbind, parts)
  idx <- cube_sticker_index(n, s$x, s$y, s$z, s$face)
  s <- s[order(idx), ]
  rownames(s) <- NULL
  s
}

# one clockwise quarter turn about `axis` (0 x, 1 y, 2 z), seen from the
# positive end, applied to the cubie and to the facing direction
cube_rotate_once <- function(n, axis, s) {
  m <- n - 1L
  r <- s
  map <- function(face, from, to) { k <- match(face, from); face[!is.na(k)] <- to[k[!is.na(k)]]; face }
  if (axis == 0L) {          # R: U<-F, F<-D, D<-B, B<-U
    r$y <- m - s$z; r$z <- s$y
    r$face <- map(s$face, c(0L, 2L, 3L, 5L), c(2L, 3L, 5L, 0L))
  } else if (axis == 1L) {   # U: F<-R, R<-B, B<-L, L<-F
    r$x <- s$z; r$z <- m - s$x
    r$face <- map(s$face, c(2L, 1L, 5L, 4L), c(1L, 5L, 4L, 2L))
  } else {                   # F: U<-L, L<-D, D<-R, R<-U
    r$x <- m - s$y; r$y <- s$x
    r$face <- map(s$face, c(0L, 4L, 3L, 1L), c(4L, 3L, 1L, 0L))
  }
  r
}

# one layer move as a 1-based permutation, new[i] = state[perm[i]]; built
# backwards (where each destination's sticker came from), as in cayleyR
cube_layer_move <- function(n, axis, layer, turns, st = cube_stickers(n)) {
  lay <- c(st$x, st$y, st$z)[axis * nrow(st) + seq_len(nrow(st))]
  sel <- lay == layer
  s <- st[sel, ]
  for (t in seq_len(4L - turns)) s <- cube_rotate_once(n, axis, s)
  perm <- seq_len(nrow(st))
  perm[sel] <- cube_sticker_index(n, s$x, s$y, s$z, s$face) + 1L
  perm
}

# the alphabet: names, permutations (6N x S, 1-based), axis and layer
cube_alphabet <- function(n) {
  m <- n - 1L
  slots <- list(c(1L, m), c(0L, m), c(2L, m), c(1L, 0L), c(0L, 0L), c(2L, 0L))
  if (n > 2L) for (l in seq_len(n - 2L)) slots <- c(slots, list(c(0L, l), c(1L, l), c(2L, l)))
  base_name <- function(axis, layer) {
    letters3 <- list(c("L", "M", "R"), c("D", "E", "U"), c("B", "S", "F"))[[axis + 1L]]
    if (layer == m) letters3[3] else if (layer == 0L) letters3[1] else
      if (n == 3L) letters3[2] else paste0(layer, c("x", "y", "z")[axis + 1L])
  }
  reversed <- function(axis, layer) {
    if (layer == m) FALSE else if (layer == 0L) TRUE else if (n == 3L) axis != 2L else FALSE
  }
  st <- cube_stickers(n)
  nm <- character(); perms <- list(); ax <- integer(); ly <- integer()
  for (sl in slots) {
    for (turns in c(1L, 3L)) {
      eff <- if (reversed(sl[1], sl[2])) turns else 4L - turns
      nm <- c(nm, paste0(base_name(sl[1], sl[2]), if (turns == 3L) "'" else ""))
      perms <- c(perms, list(cube_layer_move(n, sl[1], sl[2], eff, st)))
      ax <- c(ax, sl[1]); ly <- c(ly, sl[2])
    }
  }
  list(n = n, names = nm, perms = do.call(rbind, perms), axis = ax, layer = ly)
}

# ---- host-side helpers ---------------------------------------------------

# Which moves commute and the order of each, read off the permutations
# (perms: M x S, 1-based), as cayleyR's generate_walk(no_cancel = TRUE).
cube_move_algebra <- function(perms) {
  M <- nrow(perms); S <- ncol(perms)
  compose <- function(s, m) s[perms[m, ]]
  commute <- matrix(FALSE, M, M)
  for (a in seq_len(M)) for (b in a:M) {
    ab <- compose(compose(seq_len(S), a), b)
    ba <- compose(compose(seq_len(S), b), a)
    commute[a, b] <- commute[b, a] <- all(ab == ba)
  }
  order <- integer(M)
  for (a in seq_len(M)) {
    s <- seq_len(S)
    for (k in 1:1000) { s <- compose(s, a); if (all(s == seq_len(S))) { order[a] <- k; break } }
  }
  list(commute = commute, order = order)
}

# k walks of `depth` moves: a k x depth matrix of 0-based moves. no_undo
# refuses the inverse of the last move; with `alg` (cube_move_algebra) it is
# cayleyR's no_cancel instead: walking back over the tail while it commutes
# with the candidate, meeting its inverse bans it, and so does meeting the
# candidate itself so often that one more would be shorter the other way.
cube_walk_actions <- function(k, depth, n_moves, alg = NULL) {
  a <- matrix(0L, k, depth)
  if (depth < 1L || k < 1L) return(a)
  for (t in seq_len(depth)) {
    allowed <- matrix(TRUE, k, n_moves)
    if (t > 1L) {
      if (is.null(alg)) {
        allowed[cbind(seq_len(k), bitwXor(a[, t - 1L], 1L) + 1L)] <- FALSE
      } else {
        inv_c <- matrix(bitwXor(seq_len(n_moves) - 1L, 1L), k, n_moves, byrow = TRUE)
        cand <- matrix(seq_len(n_moves) - 1L, k, n_moves, byrow = TRUE)
        active <- matrix(TRUE, k, n_moves)
        ban <- matrix(FALSE, k, n_moves)
        cnt <- matrix(0L, k, n_moves)
        for (j in (t - 1L):1) {
          b <- a[, j]
          hit <- active & (b == inv_c)
          ban <- ban | hit
          active <- active & !hit
          active <- active & t(alg$commute[, b + 1L, drop = FALSE])
          cnt <- cnt + (active & (b == cand))
          if (!any(active)) break
        }
        ord <- matrix(alg$order, k, n_moves, byrow = TRUE)
        ban <- ban | (ord > 0L & 2L * (cnt + 1L) > ord)
        allowed <- !ban
      }
    }
    if (any(rowSums(allowed) == 0L)) stop("no move left to take at step ", t)
    # uniform over the allowed moves of each walk
    u <- matrix(stats::runif(k * n_moves), k, n_moves)
    u[!allowed] <- -1
    a[, t] <- max.col(u, "first") - 1L
  }
  a
}

# colours (k x S, 0..5) after applying the moves in `acts` (k x d) to solved
cube_host_scramble <- function(perms, acts) {
  S <- ncol(perms)
  k <- nrow(acts)
  col <- matrix((seq_len(S) - 1L) %/% (S %/% 6L), k, S, byrow = TRUE)
  for (t in seq_len(ncol(acts))) {
    p <- perms[acts[, t] + 1L, , drop = FALSE]
    col <- matrix(col[cbind(rep(seq_len(k), S), as.vector(p))], k, S)
  }
  col
}

# colours (k x S) -> one-hot columns (S * 6 x k), sticker index fastest
cube_onehot <- function(col) {
  k <- nrow(col); S <- ncol(col)
  oh <- matrix(0, S * 6L, k)
  oh[cbind(as.vector(t(col)) * S + rep(seq_len(S), k), rep(seq_len(k), each = S))] <- 1
  oh
}

# ---- the environment -----------------------------------------------------

cube_gpu_env <- function(n, size, scramble, max_steps, device) {
  size <- as.integer(size); scramble <- as.integer(scramble)
  max_steps <- as.integer(max_steps)
  stopifnot(length(size) == 1, !is.na(size), size >= 2,
            length(scramble) == 1, !is.na(scramble), scramble >= 1,
            length(max_steps) == 1, !is.na(max_steps), max_steps >= 1)
  alpha <- cube_alphabet(size)
  S <- 6L * size * size
  M <- length(alpha$names)

  st <- new.env(parent = emptyenv())
  st$n <- n; st$size <- size; st$S <- S; st$M <- M
  st$scramble <- scramble; st$max_steps <- max_steps
  st$perms <- rbind(alpha$perms, seq_len(S))          # + identity
  st$alg <- cube_move_algebra(alpha$perms)            # for no_cancel walks
  st$steps_done <- integer(n)
  st$backend <- ttt_backend(device)

  # transposed permutation matrices, one column of S*S per move:
  # P[j, i, m] = 1 where new sticker i comes from old sticker j
  P <- array(0, c(S, S, M + 1L))
  for (m in seq_len(M + 1L)) P[cbind(st$perms[m, ], seq_len(S), m)] <- 1

  st$ctx_state <- ggml_init(4 * 1024 * 1024, no_alloc = TRUE)
  cs <- st$ctx_state
  # 2-d [S*6, n] (one column per game, as ag_buffer_copy_from() takes it);
  # the step graph views it as [S, 6, n]
  st$state  <- lapply(1:2, function(i) ggml_new_tensor_2d(cs, GGML_TYPE_F32, S * 6L, n))
  st$P      <- ggml_new_tensor_2d(cs, GGML_TYPE_F32, S * S, M + 1L)
  st$action <- ggml_new_tensor_1d(cs, GGML_TYPE_I32, n)
  st$buffer <- ggml_backend_alloc_ctx_tensors(cs, st$backend)
  ggml_backend_tensor_set_data(st$P, as.numeric(P))

  st$ctx_graph <- ggml_init(32 * 1024 * 1024, no_alloc = TRUE)
  st$steps <- lapply(1:2, function(i) cube_build_step(st, src = i, dst = 3L - i))
  st$cur <- 1L

  reg.finalizer(st, ttt_free, onexit = TRUE)

  env <- structure(list(
    st = st, name = "cube", n = n, device = device, size = size,
    move_names = alpha$names, mode = "sequential", n_players = 1L,
    max_steps = max_steps,
    observation_space = space_box(0, 1, shape = S * 6L),
    action_space = space_discrete(M)
  ), class = c("rl_gpu_cube", "rl_gpu_env"))
  rl_reset(env)
  env
}

# Graph: state[src] -> move by action -> state[dst]; outputs the new state
# (observation) and solved (1 when every face is one colour).
cube_build_step <- function(st, src, dst) {
  g <- st$ctx_graph
  S <- st$S; n <- st$n; f <- S %/% 6L
  p <- ggml_reshape_3d(g, ggml_get_rows(g, st$P, st$action), S, S, n)
  x <- ggml_mul_mat(g, p, ggml_reshape_3d(g, st$state[[src]], S, 6, n))   # [S, 6, n]
  # stickers of each colour per face: [1, 6 faces, 6 colours, n]
  cnt <- ggml_sum_rows(g, ggml_reshape_4d(g, x, f, 6, 6, n))
  full <- ttt_ind(g, ggml_scale_bias(g, cnt, 1, -(f - 1)))     # 1 where count == f
  faces <- ggml_sum_rows(g, ggml_reshape_2d(g, full, 36, n))  # [1, n]
  solved <- ttt_ind(g, ggml_scale_bias(g, faces, 1, -5))
  out <- list(obs = ggml_reshape_2d(g, x, S * 6L, n), solved = solved)
  for (t in out) ggml_set_output(t)

  graph <- ggml_build_forward_expand(g, out$obs)
  ggml_graph_expand(graph, out$solved)
  ggml_graph_expand(graph, ggml_cpy(g, out$obs, st$state[[dst]]))
  sched <- ggml_backend_sched_new(list(st$backend), parallel = FALSE)
  ggml_backend_sched_alloc_graph(sched, graph)
  list(graph = graph, sched = sched, out = out)
}

# one move for every game on the device (0-based, M = identity)
cube_apply <- function(st, action) {
  ggml_backend_tensor_set_data(st$action, as.integer(action))
  step <- st$steps[[st$cur]]
  ggml_backend_sched_graph_compute(step$sched, step$graph)
  st$cur <- 3L - st$cur
  step$out
}

# all games to solved, then the moves of `acts` (n x d) on the device;
# returns the observations as S*6 x n (NULL with download = FALSE: they are
# then only in st$state[[st$cur]])
cube_device_walk <- function(st, acts, download = TRUE) {
  solved <- cube_onehot(matrix((seq_len(st$S) - 1L) %/% (st$S %/% 6L), st$n, st$S,
                               byrow = TRUE))
  ggml_backend_tensor_set_data(st$state[[st$cur]], as.numeric(solved))
  if (!ncol(acts)) return(if (download) solved)
  for (t in seq_len(ncol(acts))) out <- cube_apply(st, acts[, t])
  if (download) matrix(ggml_backend_tensor_get_data(out$obs), st$S * 6L, st$n)
}

# the current state [S*6, n] as the device observation; it also
# holds the resets written on the host, unlike the step graph's output
cube_device_obs <- function(st) {
  new_rl_device_obs(st$state[[st$cur]], st$backend, st$n, st$S * 6L)
}

#' @export
rl_reset.rl_gpu_cube <- function(env, seed = NULL) {
  st <- env$st
  acts <- cube_walk_actions(st$n, st$scramble, st$M)
  st$steps_done <- integer(st$n)
  if (env$device == "gpu") {
    cube_device_walk(st, acts, download = FALSE)
    return(cube_device_obs(st))
  }
  t(cube_device_walk(st, acts))
}

#' @export
rl_step.rl_gpu_cube <- function(env, action) {
  st <- env$st
  action <- as.integer(action)
  if (length(action) != st$n) stop("action must have length ", st$n)
  if (anyNA(action) || any(action < 0L | action >= st$M))
    stop("actions must be move indices 0..", st$M - 1L)
  out <- cube_apply(st, action)
  dev <- env$device == "gpu"
  per <- st$S * 6L
  # on the GPU the observation stays on the device (see cube_device_obs())
  obs <- if (!dev) matrix(ggml_backend_tensor_get_data(out$obs), per, st$n)
  terminated <- ggml_backend_tensor_get_data(out$solved) > 0.5
  st$steps_done <- st$steps_done + 1L
  truncated <- !terminated & st$steps_done >= st$max_steps
  final_obs <- NULL
  done <- which(terminated | truncated)
  if (length(done)) {
    if (any(truncated)) {
      if (dev) {
        # only the truncated games' columns come down; the other rows are NA
        final_obs <- matrix(NA_real_, st$n, per)
        for (k in which(truncated)) {
          final_obs[k, ] <- ggml_backend_tensor_get_data(
            out$obs, offset = as.double(k - 1L) * per * 4, n_elements = per)
        }
      } else {
        final_obs <- t(obs)
      }
    }
    # new scrambles for the finished games, written in place on the device
    col <- cube_host_scramble(st$perms, cube_walk_actions(length(done), st$scramble, st$M))
    oh <- cube_onehot(col)
    for (k in seq_along(done)) {
      ggml_backend_tensor_set_data(st$state[[st$cur]], oh[, k],
                                   offset = as.double(done[k] - 1L) * per * 4)
    }
    if (!dev) obs[, done] <- oh
    st$steps_done[done] <- 0L
  }
  res <- list(obs = if (dev) cube_device_obs(st) else t(obs), reward = as.numeric(terminated),
              terminated = terminated, truncated = truncated,
              legal = matrix(TRUE, st$n, st$M))
  if (!is.null(final_obs)) res$final_obs <- final_obs
  res
}

#' @export
rl_legal_actions.rl_gpu_cube <- function(env) matrix(TRUE, env$n, env$st$M)

#' @export
print.rl_gpu_cube <- function(x, ...) {
  cat("<rl_gpu_env> cube ", x$size, "x", x$size, "x", x$size, " x ", x$n,
      " on ", x$device, " (scramble ", x$st$scramble, ", max_steps ",
      x$max_steps, ")\n", sep = "")
  cat("  observation: ", format(x$observation_space), "\n", sep = "")
  cat("  action:      ", format(x$action_space), " (",
      paste(x$move_names, collapse = " "), ")\n", sep = "")
  invisible(x)
}

#' Random walks on the cube with prev/next labels
#'
#' Training data for a move-prediction model, as `generate_walk(pivot = TRUE)`
#' in cayleyR: for each game a cut depth is drawn uniformly from
#' `k_min..n_moves - 1`, a walk without immediate undo is made from the solved
#' cube to that depth on the device, and the state there is labelled with the
#' move back toward solved (`prev`, the inverse of the last move) and a move
#' further away (`next`, a move that does not undo the last one).
#'
#' Uses the games of `env` as the batch: their states are replaced by the
#' walks' end points and their step counters reset.
#'
#' With `no_cancel = TRUE` a walk also refuses a move that cancels against
#' its tail through moves it commutes with (`U D U'`) or that would make a
#' run longer than half the move's order (`U U U`), as cayleyR's
#' `no_cancel`; the depth label is then a tighter upper bound on the distance.
#' `next` obeys the same rule.
#'
#' @param env A cube environment from `rl_gpu_env("cube", ...)`.
#' @param n_moves Walk length bound; cuts go up to `n_moves - 1`.
#' @param k_min Shallowest cut (default 2).
#' @param no_cancel Refuse cancelling moves (see above); `FALSE` refuses only
#'   the immediate undo.
#' @return A list with `obs` (`env$n` x `6 S` one-hot colours, as from
#'   [rl_step()]), `prev`, `next` (0-based move indices) and `depth`.
#' @export
#' @examples
#' \donttest{
#' env <- rl_gpu_env("cube", n = 8, size = 3, device = "cpu")
#' w <- rl_cube_walks(env, n_moves = 10)
#' table(w$depth)
#' }
rl_cube_walks <- function(env, n_moves = 20L, k_min = 2L, no_cancel = FALSE) {
  if (!inherits(env, "rl_gpu_cube")) stop("rl_cube_walks() needs a cube environment")
  n_moves <- as.integer(n_moves); k_min <- as.integer(k_min)
  if (k_min < 1L || k_min > n_moves - 1L)
    stop("need 1 <= k_min <= n_moves - 1")
  st <- env$st
  n <- st$n
  span <- n_moves - k_min
  cut <- pmin(k_min + as.integer(floor(stats::runif(n) * span)), n_moves - 1L)
  # one move past the deepest cut: the move after the cut is `next`
  acts <- cube_walk_actions(n, max(cut) + 1L, st$M, if (isTRUE(no_cancel)) st$alg)
  last <- acts[cbind(seq_len(n), cut)]
  nxt <- acts[cbind(seq_len(n), cut + 1L)]
  acts <- acts[, seq_len(max(cut)), drop = FALSE]
  acts[col(acts) > cut] <- st$M                         # identity past the cut
  obs <- cube_device_walk(st, acts)
  st$steps_done <- integer(n)
  list(obs = t(obs), prev = bitwXor(last, 1L), `next` = nxt, depth = cut)
}
