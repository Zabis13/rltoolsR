#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# The scramble against the solve, one move at a time -- for one Q-model.
#
# As cayleyR's diag_scramble_vs_solve.R, with the solver replaced by a beam
# search over a cube Q-model (rl_cube_q): a solved 3x3x3 is walked away from
# by random moves, and every position along the walk is solved from scratch.
#
# The search: each beam state goes through the network once, its 18 q values
# score its 18 children (predicted distance to solved after that move), the
# move that undoes the last one is skipped, children already visited are
# dropped, and the `beam` lowest-scored distinct children go on. A solve is
# the first child that is solved by colours.
#
# Usage:  Rscript diag_cube3_q_solve.R [depth] [seed] [trials] [beam] [steps] [model] [device]
# Scrambles are walks with no_undo and no_cancel (cayleyR generate_walk).
#   e.g.  Rscript diag_cube3_q_solve.R 11 1             100 walks, summary
#         Rscript diag_cube3_q_solve.R 11 1 500         500 walks
#         Rscript diag_cube3_q_solve.R 11 1 100 1024 30 /tmp/cube3_q.rds gpu
# ---------------------------------------------------------------------------
suppressMessages(devtools::load_all(quiet = TRUE))

args    <- commandArgs(trailingOnly = TRUE)
DEPTH   <- if (length(args) >= 1) as.integer(args[1]) else 11L
SEED    <- if (length(args) >= 2) as.integer(args[2]) else 1L
TRIALS  <- if (length(args) >= 3) as.integer(args[3]) else 100L
BEAM    <- if (length(args) >= 4) as.integer(args[4]) else 256L
STEPS   <- if (length(args) >= 5) as.integer(args[5]) else 30L
MODEL   <- if (length(args) >= 6) args[6] else "/tmp/cube3_q.rds"
DEVICE  <- if (length(args) >= 7) args[7] else "gpu"

## ---- the model -----------------------------------------------------------
saved <- readRDS(MODEL)
meta <- saved$meta
if (!identical(meta$size, 3L)) stop("the model is not for the 3x3x3")
net <- rltoolsR:::load_net(saved$weights, function() rltoolsR:::cube_q_build(meta), DEVICE)
model <- structure(list(net = net, device = DEVICE, meta = meta), class = "rl_cube_q")

alpha <- rltoolsR:::cube_alphabet(3L)
P  <- alpha$perms                       # 18 x 54, new[i] = old[P[m, i]]
nm <- alpha$names
S  <- ncol(P)
ALG <- rltoolsR:::cube_move_algebra(P)   # for no_cancel scrambles, as cayleyR
solved_col <- (seq_len(S) - 1L) %/% 9L

apply_moves <- function(col, moves) {   # col: k x 54, moves: k (0-based)
  k <- nrow(col)
  matrix(col[cbind(rep(seq_len(k), S), as.vector(P[moves + 1L, , drop = FALSE]))], k, S)
}
replay <- function(col1, path) {        # one state, a word of 0-based moves
  for (m in path) col1 <- apply_moves(col1, m)
  col1
}
is_solved <- function(col) {            # by colours: every face one colour
  ok <- rep(TRUE, nrow(col))
  for (f in 0:5) {
    blk <- col[, f * 9L + 1:9, drop = FALSE]
    ok <- ok & rowSums(blk == blk[, 1]) == 9L
  }
  ok
}
# exact key of a state: 54 base-6 digits in three exact doubles
pw <- 6^(0:17)
state_key <- function(col) {
  paste(col[, 1:18, drop = FALSE] %*% pw, col[, 19:36, drop = FALSE] %*% pw,
        col[, 37:54, drop = FALSE] %*% pw)
}
inv <- function(m) bitwXor(m, 1L)
word <- function(path) if (length(path)) paste(nm[path + 1L], collapse = " ") else "(none)"

## ---- the solve -----------------------------------------------------------
solve_one <- function(col1) {
  t0 <- proc.time()[["elapsed"]]
  if (is_solved(col1)) return(list(ok = TRUE, path = integer(), secs = 0, expanded = 0))
  beam <- col1
  paths <- list(integer())
  seen <- new.env(hash = TRUE, size = 65536L)
  seen[[state_key(col1)]] <- TRUE
  expanded <- 0
  for (step in seq_len(STEPS)) {
    nb <- nrow(beam)
    q <- rl_cube_q_values(model, t(cube_onehot(beam)))           # nb x 18
    expanded <- expanded + nb
    last <- vapply(paths, function(p) if (length(p)) p[length(p)] else -1L, 0L)
    undo <- cbind(seq_len(nb), inv(last) + 1L)[last >= 0, , drop = FALSE]
    q[undo] <- Inf
    o <- order(q)                                                 # column-major
    o <- o[is.finite(q[o])]
    par <- (o - 1L) %% nb + 1L
    mov <- (o - 1L) %/% nb
    child <- apply_moves(beam[par, , drop = FALSE], mov)
    done <- which(is_solved(child))
    if (length(done)) {
      j <- done[1]
      return(list(ok = TRUE, path = c(paths[[par[j]]], mov[j]),
                  secs = proc.time()[["elapsed"]] - t0, expanded = expanded))
    }
    keys <- state_key(child)
    keep <- !duplicated(keys) & !vapply(keys, exists, NA, envir = seen, inherits = FALSE)
    keep <- which(keep)[seq_len(min(BEAM, sum(keep)))]
    if (!length(keep)) break
    for (k in keys[keep]) seen[[k]] <- TRUE
    beam <- child[keep, , drop = FALSE]
    paths <- Map(function(p, m) c(paths[[p]], m), par[keep], mov[keep])
  }
  list(ok = FALSE, secs = proc.time()[["elapsed"]] - t0, expanded = expanded)
}

## ---- one walk ------------------------------------------------------------
walk <- function(seed) {
  set.seed(seed)
  scramble <- rltoolsR:::cube_walk_actions(1L, DEPTH, 18L, ALG)[1, ]
  rows <- vector("list", DEPTH)
  st <- matrix(solved_col, 1L)
  for (k in seq_len(DEPTH)) {
    st <- apply_moves(st, scramble[k])
    rows[[k]] <- list(depth = k, scramble = scramble[seq_len(k)],
                      back = rev(inv(scramble[seq_len(k)])), state = st,
                      res = solve_one(st))
  }
  list(scramble = scramble, rows = rows)
}

head_line <- function() {
  cat(sprintf("model: %s  (hidden %s, blocks %d, trained to depth %d)\n", MODEL,
              paste(meta$hidden, collapse = "x"), meta$blocks, meta$n_moves))
  cat(sprintf("search: beam %d, up to %d moves, device %s\n\n", BEAM, STEPS, DEVICE))
}

## ---- report --------------------------------------------------------------
cat(sprintf("scramble vs solve | n=3, depth %d, seeds %d..%d, %d walks\n",
            DEPTH, SEED, SEED + TRIALS - 1L, TRIALS))
head_line()
solved <- moves <- secs <- expd <- numeric(DEPTH)
shorter <- numeric(DEPTH)
bad <- 0L
for (t in seq_len(TRIALS)) {
  w <- walk(SEED + t - 1L)
  for (row in w$rows) {
    r <- row$res; k <- row$depth
    secs[k] <- secs[k] + r$secs
    expd[k] <- expd[k] + r$expanded
    if (!isTRUE(r$ok)) next
    if (!is_solved(replay(row$state, r$path))) { bad <- bad + 1L; next }
    solved[k] <- solved[k] + 1
    moves[k] <- moves[k] + length(r$path)
    shorter[k] <- shorter[k] + (length(r$path) <= k)
  }
  if (t %% 10L == 0L) { cat(sprintf("  ... %d walks\n", t)); flush(stdout()) }
}
cat(sprintf("\n%6s %8s %8s %9s %8s %8s %10s %8s\n", "depth", "walk", "solved",
            "moves", "ratio", "<=walk", "expanded", "secs"))
for (k in seq_len(DEPTH)) {
  s <- max(solved[k], 1)
  cat(sprintf("%6d %8d %7.0f%% %9.1f %7.2fx %7.0f%% %10.0f %8.2f\n", k, k,
              100 * solved[k] / TRIALS, moves[k] / s, moves[k] / s / k,
              100 * shorter[k] / s, expd[k] / TRIALS, secs[k] / TRIALS))
}
if (bad) cat(sprintf("\n  WARNING: %d solutions did not verify\n", bad))
cat("\n  walk      scramble length (no_undo + no_cancel), an upper bound to undo\n")
cat("  solved    share of walks solved within the move limit\n")
cat("  moves     mean solution length over the solved ones\n")
cat("  <=walk    share of solutions no longer than the walk\n")
cat("  expanded  beam states through the network per solve\n")
