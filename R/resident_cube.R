# Cube walks as ag_* operations, for rl_cube_q(): the moves are drawn and
# made on the device. A walk is one ag_capture() recording replayed once per
# move, all replays queued together (ag_capture_repeat), and the training step
# reads the end states and the labels on the device -- per walk only a noise
# key goes up and nothing comes down.
#
# State. The colour of every sticker, not its one-hot: a move is then a
# gather, new[i, g] = old[perm[a_g][i], g]. ag_* tensors are 2-d, so the
# colours live one per column in `state` [1, 2 S n]:
#   columns 0 .. S n - 1      the games, sticker i of game g at i + S g
#   columns S n .. 2 S n - 1  the solved cube, never written
# and a move is one ag_get_rows() over an index tensor [S, n]. The first move
# of a walk reads the solved half (the index is shifted by S n when the step
# counter is 0), so a new walk needs no reset of the state. PI [S, M + 1]
# holds the source sticker of every sticker for every move; its last column
# is the identity, the move of a walk past its cut.
#
# Which moves a walk may take (as cube_walk_actions(), R/gpu_cube.R). The scan
# back over the tail of the walk is kept as two matrices [M, n], one row per
# candidate move c, updated by the move b just made:
#   ban' = H[c, b] + K[c, b] * ban          H: b is the inverse of c
#   cnt' = K[c, b] * (E[c, b] + cnt)        E: b is c itself
# K[c, b] = 1 where the scan goes on past b: with no_cancel, b commutes with c
# and is not its inverse; without it never, which leaves ban = "the inverse of
# the last move". c is refused when ban is set or 2 (cnt + 1) > order(c).
# The identity leaves both as they are (H = E = 0, K = 1).
#
# Labels. lab [3, n]: the cut depth of each walk (drawn at its start), its
# last move and `next`, the move drawn at the cut and not made. The step
# counter t is on the device: moves t < cut are made, the draw at t == cut is
# `next`, later steps are the identity. H and E are also the one-hots of
# prev (the inverse of the last move) and next for the targets.
#
# As in R/resident_ttt.R: every state buffer is read by ONE ag_get_rows node
# in a recording and the write depends on it; broadcasts are [m, 1] and
# [1, n] only. The noise key of step t is (k1, k2 + t), see R/ppo_resident.R.

cube_res_new <- function(st, n_moves, device, k_min = 2L, no_cancel = FALSE) {
  S <- st$S; n <- st$n; M <- st$M
  n_moves <- as.integer(n_moves); k_min <- as.integer(k_min)
  if (n_moves >= 2048L) stop("resident cube walks: n_moves must be below 2048")
  span <- max(1L, n_moves - k_min)
  rs <- new.env(parent = emptyenv())
  rs$n <- n; rs$S <- S; rs$M <- M; rs$depth <- n_moves
  rs$gpu <- identical(device, "gpu")
  rs$state <- ag_buffer(1, 2 * S * n)
  rs$PI <- ag_buffer(S, M + 1L)
  rs$acts <- ag_buffer(n, n_moves)
  rs$cnt <- ag_buffer(1, 1)
  rs$ban <- ag_buffer(M, n)
  rs$rep <- ag_buffer(M, n)
  rs$lab <- ag_buffer(3, n)
  rs$H <- ag_buffer(M, M + 1L)
  rs$K <- ag_buffer(M, M + 1L)
  rs$E <- ag_buffer(M, M + 1L)
  ag_buffer_write(rs$PI, t(st$perms) - 1, col_offset = 0)
  solved <- (seq_len(S) - 1) %/% (S %/% 6L)
  ag_buffer_write(rs$state, matrix(rep(solved, 2L * n), 1L), col_offset = 0)

  mv <- seq_len(M)                                      # 1-based moves
  inv <- bitwXor(mv - 1L, 1L) + 1L
  H <- E <- K <- matrix(0, M, M + 1L)
  H[cbind(inv, mv)] <- 1
  E[cbind(mv, mv)] <- 1
  if (isTRUE(no_cancel)) K[, mv] <- st$alg$commute * (1 - H[, mv])
  K[, M + 1L] <- 1
  ord <- if (isTRUE(no_cancel)) ifelse(st$alg$order > 0L, 2 - st$alg$order, -1e6) else
    rep(-1e6, M)
  ag_buffer_write(rs$H, H, col_offset = 0)
  ag_buffer_write(rs$K, K, col_offset = 0)
  ag_buffer_write(rs$E, E, col_offset = 0)

  tn <- function(v, r, c) ag_tensor(matrix(v, r, c))
  rs$k <- list(
    one = tn(1, 1, 1), zero = tn(0, 1, 1), ones_n = tn(1, 1, n), zeros_n = tn(0, 1, n),
    one_s = tn(1, S, 1), zeros_mn = tn(0, M, n), ord = tn(ord, M, 1),
    kmin_n = tn(k_min, 1, n), e2 = tn(c(0, 1), 2, 1),
    s1 = tn(c(1, 0, 0), 1, 3), s2 = tn(c(0, 1, 0), 1, 3), s3 = tn(c(0, 0, 1), 1, 3))
  rs$neg <- lapply(1:5, function(c) tn(-c, S, 1))
  # inputs of the recordings: the same values on every call, sent once
  rs$off <- matrix((seq_len(n) - 1) * S, 1, n)
  rs$all <- matrix(seq_len(S * n) - 1, S * n, 1)
  rs$cols <- matrix(seq_len(n) - 1, n, 1)
  rs$z <- matrix(0, 1, 1)
  k <- rs$k
  params <- cube_res_params(rs)

  # the move `a` ([n, 1] or [1, n]) at step t, for every game
  move <- function(a, t, off, all) {
    src <- ag_get_rows(rs$PI, a)                        # [S, n]
    first <- ag_add(ag_scale(ag_clamp(t, 0, 1), -1), k$one)       # 1 at t = 0
    shift <- ag_matmul(ag_scale(first, S * n), k$ones_n)          # [1, n]
    moved <- ag_get_rows(rs$state, ag_add(src, ag_add(off, shift)))
    ag_set_rows(rs$state, all, moved)
  }

  # given moves: column t of `acts` (cube_res_walk)
  rs$step <- ag_capture(function(off, all, z) {
    t <- ag_get_rows(rs$cnt, z)                         # the one read of t
    move(ag_get_rows(rs$acts, t), t, off, all)
    ag_set_rows(rs$cnt, z, ag_add(t, k$one))
    NULL
  }, params = params)

  # a new walk: cut depths, an empty tail, t = 0
  rs$start <- ag_capture(function(cols, z, key) {
    cut <- ag_add(ag_argmax(ag_rand(key, "uniform", c(span, n), stream = 1L),
                            as_tensor = TRUE), k$kmin_n)
    ag_set_rows(rs$lab, cols, ag_concat_rows(ag_concat_rows(cut, k$zeros_n), k$zeros_n))
    ag_set_rows(rs$ban, cols, k$zeros_mn)
    ag_set_rows(rs$rep, cols, k$zeros_mn)
    ag_set_rows(rs$cnt, z, k$zero)
    NULL
  }, params = params)

  # drawn moves (cube_res_sample)
  rs$draw <- ag_capture(function(off, all, cols, z, key) {
    t <- ag_get_rows(rs$cnt, z)
    lab <- ag_get_rows(rs$lab, cols)
    cut <- ag_matmul(k$s1, lab); last <- ag_matmul(k$s2, lab); nxt <- ag_matmul(k$s3, lab)
    ban <- ag_get_rows(rs$ban, cols)
    rp <- ag_get_rows(rs$rep, cols)
    over <- ag_clamp(ag_add(ag_scale(rp, 2), k$ord), 0, 1)        # 2 (cnt + 1) > order
    refused <- ag_clamp(ag_add(ban, over), 0, 1)
    u <- ag_rand(ag_add(key, ag_matmul(k$e2, t)), "uniform", c(M, n))
    a_s <- ag_argmax(ag_add(u, ag_scale(refused, -2)), as_tensor = TRUE)   # [1, n]
    d <- ag_sub(cut, ag_matmul(t, k$ones_n))            # moves left before the cut
    active <- ag_clamp(d, 0, 1)
    rest <- ag_sub(k$ones_n, active)
    at_cut <- ag_sub(k$ones_n, ag_clamp(ag_mul(d, d), 0, 1))
    a <- ag_add(ag_mul(active, a_s), ag_scale(rest, M))          # identity past the cut
    move(a, t, off, all)
    Ka <- ag_get_rows(rs$K, a)
    ag_set_rows(rs$ban, cols, ag_add(ag_get_rows(rs$H, a), ag_mul(Ka, ban)))
    ag_set_rows(rs$rep, cols, ag_mul(Ka, ag_add(ag_get_rows(rs$E, a), rp)))
    last <- ag_add(ag_mul(active, a_s), ag_mul(rest, last))
    nxt <- ag_add(ag_mul(at_cut, a_s), ag_mul(ag_sub(k$ones_n, at_cut), nxt))
    ag_set_rows(rs$lab, cols, ag_concat_rows(ag_concat_rows(cut, last), nxt))
    ag_set_rows(rs$acts, t, ag_reshape(a, n, 1L))       # kept for inspection
    ag_set_rows(rs$cnt, z, ag_add(t, k$one))
    NULL
  }, params = params)

  # what a walk left, for tests and inspection
  rs$peek <- ag_capture(function(all, cols) list(
    obs = cube_res_obs(rs, all), lab = ag_get_rows(rs$lab, cols),
    ban = ag_get_rows(rs$ban, cols), rep = ag_get_rows(rs$rep, cols)), params = params)
  rs
}

# Everything a recording over this state must list in params.
cube_res_params <- function(rs) {
  c(list(rs$state, rs$PI, rs$acts, rs$cnt, rs$ban, rs$rep, rs$lab, rs$H, rs$K, rs$E),
    unname(rs$k), rs$neg)
}

cube_res_free <- function(rs) {
  for (f in list(rs$step, rs$start, rs$draw, rs$peek)) ag_capture_free(f)
  for (b in list(rs$state, rs$PI, rs$acts, rs$cnt, rs$ban, rs$rep, rs$lab,
                 rs$H, rs$K, rs$E)) ag_buffer_free(b)
  invisible(NULL)
}

cube_res_run <- function(rs, f, times, args) {
  if (rs$gpu) {
    # all runs queued at once, one wait (ggmlR::ag_capture_repeat)
    do.call(ag_capture_repeat, c(list(f, times), args))
  } else {
    for (i in seq_len(times)) do.call(f, args)
  }
  invisible(rs)
}

# The walks of `acts` (n x d, 0-based moves, M = identity; 1 <= d <= n_moves)
# from the solved cube, on the device: the end states stay in rs$state.
cube_res_walk <- function(rs, acts) {
  d <- ncol(acts)
  if (nrow(acts) != rs$n || d < 1L || d > rs$depth)
    stop("cube_res_walk: acts must be ", rs$n, " x 1..", rs$depth)
  ag_buffer_write(rs$cnt, rs$z, col_offset = 0)
  ag_buffer_write(rs$acts, acts + 0, col_offset = 0)
  cube_res_run(rs, rs$step, d, list(off = rs$off, all = rs$all, z = rs$z))
}

# New walks drawn on the device: cut depths, then n_moves steps (the moves up
# to each walk's cut and `next` at it). End states in rs$state, labels in
# rs$lab, the moves made in rs$acts.
cube_res_sample <- function(rs, key = ppo_res_key()) {
  key <- matrix(key, 2, 1)
  rs$start(cols = rs$cols, z = rs$z, key = key)
  cube_res_run(rs, rs$draw, rs$depth,
               list(off = rs$off, all = rs$all, cols = rs$cols, z = rs$z, key = key))
}

# The observation [6 S, n] of the current states, inside a recording: one-hot
# colours, sticker index fastest (as rl_gpu_env("cube") returns it). `all` is
# rs$all, passed as an input of the recording.
cube_res_obs <- function(rs, all) {
  col <- ag_reshape(ag_get_rows(rs$state, all), rs$S, rs$n)
  ind <- function(c) {                                  # 1 where colour == c
    d <- if (c > 0L) ag_add(col, rs$neg[[c]]) else col
    ag_add(ag_scale(ag_clamp(ag_mul(d, d), 0, 1), -1), rs$k$one_s)
  }
  x <- ind(0L)
  for (c in 1:5) x <- ag_concat_rows(x, ind(c))
  x
}

# Targets and mask [M, n] of the drawn walks, inside a recording: prev gets
# (depth - 1 - mu) / sd and next (depth + 1 - mu) / sd. `cols` is rs$cols;
# c_prev, c_next are [1, n] inputs holding (-1 - mu) / sd and (1 - mu) / sd.
cube_res_labels <- function(rs, cols, c_prev, c_next, inv_sd) {
  k <- rs$k
  lab <- ag_get_rows(rs$lab, cols)
  depth <- ag_scale(ag_matmul(k$s1, lab), inv_sd)
  prev <- ag_get_rows(rs$H, ag_matmul(k$s2, lab))       # one-hot of the inverse
  nxt <- ag_get_rows(rs$E, ag_matmul(k$s3, lab))
  list(y = ag_add(ag_mul(prev, ag_add(depth, c_prev)), ag_mul(nxt, ag_add(depth, c_next))),
       mask = ag_add(prev, nxt))
}
