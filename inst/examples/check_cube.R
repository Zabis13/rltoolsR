# Checks of the cube environment (rl_gpu_env("cube")) and rl_cube_walks().
# Rscript inst/examples/check_cube.R [cpu|gpu]
devtools::load_all(quiet = TRUE)
device <- if (length(commandArgs(TRUE))) commandArgs(TRUE)[1] else "cpu"
set.seed(1)
ok <- function(cond, msg) cat(sprintf("%-55s %s\n", msg, if (isTRUE(cond)) "OK" else "FAIL"))

# ---- 1. move tables ------------------------------------------------------
for (N in 2:5) {
  a <- rltoolsR:::cube_alphabet(N)
  P <- a$perms
  S <- 6 * N * N
  ok(nrow(P) == 6 * N && all(apply(P, 1, function(p) setequal(p, 1:S))),
     sprintf("N=%d: %d moves, all permutations", N, 6 * N))
  inv_ok <- all(sapply(seq(1, nrow(P), 2), function(i) all(P[i, ][P[i + 1, ]] == 1:S)))
  ok(inv_ok, sprintf("N=%d: move a and a xor 1 are inverses", N))
  ord4 <- all(apply(P, 1, function(p) { s <- 1:S; for (k in 1:4) s <- s[p]; all(s == 1:S) }))
  ok(ord4, sprintf("N=%d: every move has order 4", N))
  if (requireNamespace("cayleyR", quietly = TRUE)) {
    ref <- cayleyR::cube_moves(N)
    same <- identical(names(ref), a$names) &&
      all(sapply(seq_along(ref), function(i) all(ref[[i]] == P[i, ])))
    ok(same, sprintf("N=%d: names and perms match cayleyR::cube_moves", N))
  }
}

# ---- 2. environment ------------------------------------------------------
n <- 64
env <- rl_gpu_env("cube", n = n, size = 3, scramble = 1, max_steps = 30, device = device)
print(env)
obs <- as.matrix(rl_reset(env))   # device = "gpu" returns an rl_device_obs
ok(all(dim(obs) == c(n, 324)) && all(rowSums(obs) == 54), "reset: one-hot 324 per game")

# scramble 1: the inverse of the scramble solves -- find it by trying all
st <- env$st
solved_by <- matrix(FALSE, n, 18)
for (m in 0:17) {
  col <- rltoolsR:::cube_host_scramble(st$perms, matrix(m, 1, 1))
  solved_by[, m + 1] <- apply(obs, 1, function(o) all(o == rltoolsR:::cube_onehot(col)))
}
ok(all(rowSums(solved_by) >= 1), "reset at depth 1 equals one host move")
first <- max.col(solved_by, "first") - 1L
r <- rl_step(env, bitwXor(first, 1L)); r$obs <- as.matrix(r$obs)
ok(all(r$terminated) && all(r$reward == 1), "inverse move solves: terminated, reward 1")
ok(all(rowSums(r$obs) == 54) && all(!r$truncated), "solved games auto-reset")

# device step equals host step
acts <- sample(0:17, n, replace = TRUE)
prev <- r$obs
r2 <- rl_step(env, acts); r2$obs <- as.matrix(r2$obs)
host <- t(sapply(seq_len(n), function(i) {
  o <- matrix(prev[i, ], 54, 6); col <- max.col(o) - 1L
  as.vector(rltoolsR:::cube_onehot(matrix(col[st$perms[acts[i] + 1, ]], 1)))
}))
nd <- !(r2$terminated | r2$truncated)
ok(all(r2$obs[nd, ] == host[nd, ]), "device move equals host permutation")

# truncation
env2 <- rl_gpu_env("cube", n = 8, size = 3, scramble = 10, max_steps = 3, device = device)
invisible(rl_reset(env2))
for (k in 1:3) r <- rl_step(env2, rep(0L, 8))
ok(all(r$truncated | r$terminated) && !is.null(r$final_obs), "truncated at max_steps, final_obs set")

# ---- 3. walks ------------------------------------------------------------
env3 <- rl_gpu_env("cube", n = 4096, size = 3, device = device)
t0 <- Sys.time()
w <- rl_cube_walks(env3, n_moves = 20, k_min = 2)
cat(sprintf("rl_cube_walks 4096 x up to 19 moves: %.0f ms\n",
            1000 * as.numeric(difftime(Sys.time(), t0, units = "secs"))))
print(table(w$depth))
ok(all(bitwXor(w$`next`, 1L) != bitwXor(w$prev, 1L)), "next never undoes the last move")
# prev from the pivot returns to the walk one step shorter: depth-2 states
# after prev are one move from solved
i <- which(w$depth == 2)[1:min(20, sum(w$depth == 2))]
back <- sapply(i, function(j) {
  o <- matrix(w$obs[j, ], 54, 6); col <- max.col(o) - 1L
  col <- col[st$perms[w$prev[j] + 1, ]]
  any(sapply(0:17, function(m) all(col[st$perms[m + 1, ]] == 0:53 %/% 9)))
})
ok(all(back), "depth 2: prev leads to a state one move from solved")

# ---- 4. no_cancel walks (cayleyR generate_walk) -------------------------
alg <- rltoolsR:::cube_move_algebra(st$perms[1:18, ])
ok(all(rowSums(alg$commute) == 6) && all(alg$order == 4),
   "3x3: each move commutes with its axis (6), order 4")
A <- rltoolsR:::cube_walk_actions(2000, 12, 18, alg)
ax <- matrix(rltoolsR:::cube_alphabet(3)$axis[A + 1L], nrow(A))
# along a run of one axis: no move with its inverse, no move three times
viol <- 0L
for (i in seq_len(nrow(A))) {
  run <- A[i, 1]
  for (t in 2:ncol(A)) {
    run <- if (ax[i, t] == ax[i, t - 1]) c(run, A[i, t]) else A[i, t]
    m <- A[i, t]
    if (bitwXor(m, 1L) %in% run[-length(run)] || sum(run == m) > 2) viol <- viol + 1L
  }
}
ok(viol == 0L, "no_cancel: no cancelling move along a commuting run")
w2 <- rl_cube_walks(env3, n_moves = 20, k_min = 2, no_cancel = TRUE)
ok(all(w2$`next` != w2$prev), "no_cancel walks: next never undoes the last move")
