# Tic-tac-toe self-play as ag_* operations, for the resident rollout: the step
# is part of one ag_capture() recording together with the policy, so actions
# never leave the device.
#
# The same game as rl_gpu_env("tictactoe", opponent = "self") (R/gpu_tictactoe.R):
# boards [9, N] (+1 = X, -1 = O, 0 = empty) and the side to move [1, N] live in
# ag_buffers; finished games are reset in the same step; the observation is
# canonical (board times the side to move). Every quantity is an integer, so
# the results are exact on the CPU and on the GPU.
#
# Broadcasts are explicit (a matmul with a ones column, constants as columns of
# the operand's height): ag_* on the CPU broadcasts only [m, 1] and [1, n].
#
# State written in a recording: each state buffer is read by ONE ag_get_rows
# node and the write depends on it. The Vulkan graph optimizer orders nodes by
# data dependencies only; a second read of the buffer that the write does not
# depend on could run after the write.

ttt_res_new <- function(n) {
  k <- function(v, r) ag_tensor(matrix(v, r, 1L))
  rs <- new.env(parent = emptyenv())
  rs$n <- as.integer(n)
  rs$board <- ag_buffer(9, n)
  rs$player <- ag_buffer(1, n)
  rs$I9 <- ag_buffer(9, 9)
  ag_buffer_write(rs$I9, diag(9), col_offset = 0)
  rs$k <- list(
    Lt = ag_tensor(t(ttt_lines())),           # [8, 9]
    one9c = k(1, 9), one1r = k(1, 1),
    sum9 = ag_tensor(matrix(1, 1, 9)), sum8 = ag_tensor(matrix(1, 1, 8)),
    rep9 = ag_tensor(matrix(1, 9, 1)), rep8 = ag_tensor(matrix(1, 8, 1)),
    m2_8 = k(-2, 8), m8_1 = k(-8, 1)
  )
  ttt_res_reset(rs)
  rs
}

ttt_res_reset <- function(rs) {
  ag_buffer_write(rs$board, matrix(0, 9, rs$n), col_offset = 0)
  ag_buffer_write(rs$player, matrix(1, 1, rs$n), col_offset = 0)
  invisible(rs)
}

ttt_res_free <- function(rs) {
  for (b in list(rs$board, rs$player, rs$I9)) ag_buffer_free(b)
  invisible(NULL)
}

# Everything a recording over this state must list in params.
ttt_res_params <- function(rs) c(list(rs$board, rs$player, rs$I9), unname(rs$k))

# The current state of games `cols` (f32 [N, 1], 0..N-1): the one read of
# each state buffer in a recording, plus what the policy sees -- the canonical
# observation (as rl_gpu_env returns it) and the occupied cells (1 - legal).
ttt_res_read <- function(rs, cols) {
  board <- ag_get_rows(rs$board, cols)
  player <- ag_get_rows(rs$player, cols)
  occupied <- ag_mul(board, board)
  list(board = board, player = player, occupied = occupied,
       obs = ag_mul(board, ag_matmul(rs$k$rep9, player)),
       legal = ag_add(ag_scale(occupied, -1), rs$k$one9c))
}

# One step for actions `a` (f32 cell indices, N values) on games `cols`, from
# the state `s` read by ttt_res_read() in the same recording. Writes the next
# state; returns obs [9, N], legal [9, N], reward and done [1, N]. Reward is
# for the side that just moved.
ttt_res_step <- function(rs, a, cols, s = ttt_res_read(rs, cols)) {
  k <- rs$k
  board <- s$board
  player <- s$player
  onehot <- ag_get_rows(rs$I9, a)
  illegal <- ag_matmul(k$sum9, ag_mul(onehot, s$occupied))
  mask <- ag_sub(player, ag_mul(player, illegal))      # sign, 0 on illegal
  nb <- ag_add(board, ag_mul(onehot, ag_matmul(k$rep9, mask)))
  own <- ag_mul(ag_matmul(k$Lt, nb), ag_matmul(k$rep8, player))
  win <- ag_clamp(ag_matmul(k$sum8, ag_clamp(ag_add(own, k$m2_8), 0, 1)), 0, 1)
  full <- ag_clamp(ag_add(ag_matmul(k$sum9, ag_mul(nb, nb)), k$m8_1), 0, 1)
  done <- ag_clamp(ag_add(ag_add(illegal, win), full), 0, 1)
  reward <- ag_sub(win, illegal)
  not_done <- ag_add(ag_scale(done, -1), k$one1r)     # 1 - done
  next_player <- ag_add(ag_mul(ag_scale(player, -1), not_done), done)
  next_board <- ag_mul(nb, ag_matmul(k$rep9, not_done))
  ag_set_rows(rs$board, cols, next_board)
  ag_set_rows(rs$player, cols, next_player)
  list(
    obs = ag_mul(next_board, ag_matmul(k$rep9, next_player)),
    legal = ag_add(ag_scale(ag_mul(next_board, next_board), -1), k$one9c),
    reward = reward,
    done = done
  )
}
