# Does a tic-tac-toe policy block? Every legal position where the opponent has
# exactly one open two-in-a-row (one cell saves the game) and the side to move
# has no win of its own, split by the side to move. As a control, positions
# where the side to move can win at once.
#
#   Rscript inst/examples/diag_tictactoe_block.R [model.rds ...]
#
# Columns:
#   n         positions
#   greedy    the greedy move is the block (or, for "win", a winning move)
#   p_move    mean policy probability on that move (legal moves only)
#   value     the move value is highest for that move among the legal ones, i.e.
#             the value net alone would choose it. Move value: 1 for a move that
#             wins, 0 for one that fills the board, else -V(child) (V is from the side to move,
#             so the mover's value of a move is minus the child's value)
#   q_good    mean -V(child) of that move
#   q_other   mean -V(child) of the other legal moves
#   v_state   mean V of the position itself (a lost-if-wrong position)
suppressMessages(devtools::load_all(quiet = TRUE))

paths <- commandArgs(trailingOnly = TRUE)
if (!length(paths)) paths <- "/tmp/tictactoe_ppo.rds"

lines <- rbind(c(1, 2, 3), c(4, 5, 6), c(7, 8, 9), c(1, 4, 7), c(2, 5, 8),
               c(3, 6, 9), c(1, 5, 9), c(3, 5, 7))

# all boards, X = +1, O = -1
B <- as.matrix(expand.grid(rep(list(c(-1, 0, 1)), 9)))
dimnames(B) <- NULL
nx <- rowSums(B == 1); no <- rowSums(B == -1)
won <- function(b, s) Reduce(`|`, lapply(seq_len(nrow(lines)), function(l)
  rowSums(b[, lines[l, ], drop = FALSE] == s) == 3))
ok <- (nx == no | nx == no + 1) & !won(B, 1) & !won(B, -1) & rowSums(B == 0) > 0
B <- B[ok, , drop = FALSE]
mover <- ifelse(rowSums(B == 1) == rowSums(B == -1), 1, -1)   # X moves on equal counts
obs <- B * mover                                              # canonical: mover = +1

# cells that complete a line for side s (+1 mover, -1 opponent), as a logical n x 9
completes <- function(o, s) {
  m <- matrix(FALSE, nrow(o), 9)
  for (l in seq_len(nrow(lines))) {
    v <- o[, lines[l, ], drop = FALSE]
    hit <- rowSums(v == s) == 2 & rowSums(v == 0) == 1
    for (k in 1:3) m[cbind(which(hit & v[, k] == 0), lines[l, k])] <- TRUE
  }
  m
}
win_cells <- completes(obs, 1)
threat <- completes(obs, -1)
sets <- list(
  block = which(rowSums(win_cells) == 0 & rowSums(threat) == 1),
  win   = which(rowSums(win_cells) >= 1))

score <- function(model) {
  out <- with_policy_modes(model, policy_forward(model$net, t(obs)))
  legal <- t(obs == 0)                                         # 9 x n
  logits <- out$logits + PPO_MASK * (1 - legal)
  p <- exp(sweep(logits, 2, apply(logits, 2, max)))
  p <- sweep(p, 2, colSums(p), "/")
  greedy <- max.col(t(logits), "first")
  v_state <- as.numeric(out$value)

  # values of all children, from the mover's view: -V(child)
  n <- nrow(obs)
  q <- matrix(-Inf, n, 9)
  for (c in 1:9) {
    i <- which(obs[, c] == 0)
    if (!length(i)) next
    child <- obs[i, , drop = FALSE]
    child[, c] <- 1
    child <- -child                                             # opponent's view
    q[i, c] <- -as.numeric(with_policy_modes(model, policy_forward(model$net, t(child)))$value)
  }
  # a move that ends the game is worth its reward, not -V of a terminal
  # position (the value net is never trained on those: the env resets)
  q[win_cells] <- 1
  last <- rowSums(obs == 0) == 1 & rowSums(win_cells) == 0
  q[last, ][obs[last, , drop = FALSE] == 0] <- 0                 # board full: draw

  rows <- list()
  for (kind in names(sets)) {
    good <- if (kind == "block") threat else win_cells
    for (side in c(1, -1)) {
      i <- sets[[kind]][mover[sets[[kind]]] == side]
      if (!length(i)) next
      g <- good[i, , drop = FALSE]
      qi <- q[i, , drop = FALSE]
      leg <- obs[i, , drop = FALSE] == 0
      q_good <- rowSums(ifelse(g, qi, 0)) / rowSums(g)
      oth <- leg & !g
      q_oth <- ifelse(rowSums(oth) > 0, rowSums(ifelse(oth, qi, 0)) / pmax(rowSums(oth), 1), NA)
      rows[[length(rows) + 1L]] <- data.frame(
        kind = kind, mover = if (side == 1) "X" else "O", n = length(i),
        greedy = mean(g[cbind(seq_along(i), greedy[i])]),
        p_move = mean(rowSums(t(p[, i, drop = FALSE]) * g)),
        value = mean(g[cbind(seq_along(i), max.col(qi, "first"))]),
        q_good = mean(q_good), q_other = mean(q_oth, na.rm = TRUE),
        v_state = mean(v_state[i]))
    }
  }
  do.call(rbind, rows)
}

for (path in paths) {
  m <- rl_load_model(path, device = "gpu")
  cat(sprintf("\n== %s  (net %s)\n", path,
              if (is.null(m$meta$net_arch)) "shared" else m$meta$net_arch))
  print(score(m), digits = 3, row.names = FALSE)
}
