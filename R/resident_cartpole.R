# CartPole-v1 as ag_* operations, for a resident rollout: the step is recorded
# with ag_capture() together with the policy, as R/resident_tabular.R, so the
# physics, termination, truncation and resets stay on the device.
#
# The physics is the one of R/gpu_cartpole.R and the C++ "cartpole"
# (Gymnasium classic_control/cartpole.py, Euler), in f32:
#   state [4, N]  x, x_dot, theta, theta_dot, one column per game
#   steps [1, N]  steps taken
# One step: physics -> flags -> reset by mask -> the step counter, zeroed for
# the games that ended. terminated is |x| > x_threshold or |theta| >
# theta_threshold, strictly, as Gymnasium; truncated is the step limit on a
# game that did not terminate. A reset game starts uniform in [-0.05, 0.05]
# from stream 1 of the step's noise key (stream 0 is the policy's).
#
# As in R/resident_ttt.R: every state buffer is read by ONE ag_get_rows node in
# a recording and the write depends on it; broadcasts are [m, 1] and [1, n].

# 1e12 * (the smallest f32 step above a threshold) is far above 1: the
# clamped indicator of relu(v - t) is exactly 0 or 1
CART_RES_IND <- 1e12

cart_res_new <- function(n) {
  ggmlr_need(RES_TRIG, "the resident cartpole")
  k <- CARTPOLE
  rs <- new.env(parent = emptyenv())
  rs$n <- as.integer(n)
  rs$max_steps <- k$max_steps
  rs$state <- ag_buffer(4, n)
  rs$steps <- ag_buffer(1, n)
  tn <- function(v, r, c) ag_tensor(matrix(v, r, c))
  rs$k <- list(
    one = tn(1, 1, 1),
    ones = tn(1, 1, n),
    mfm = tn(-k$force_mag, 1, 1),
    l43 = tn(k$length * 4 / 3, 1, 1),
    mxt = tn(-k$x_threshold, 1, 1),
    mtt = tn(-k$theta_threshold, 1, 1),
    lim = tn(-(k$max_steps - 1), 1, 1),
    m05 = tn(-0.05, 4, 1))     # [4, 1]: added to [4, N]
  cart_res_reset(rs)
  rs
}

# New games for all N, uniform in [-0.05, 0.05] (R's RNG).
cart_res_reset <- function(rs) {
  ag_buffer_write(rs$state, matrix(stats::runif(4 * rs$n, -0.05, 0.05), 4, rs$n),
                  col_offset = 0)
  ag_buffer_write(rs$steps, matrix(0, 1, rs$n), col_offset = 0)
  invisible(rs)
}

cart_res_buffers <- function(rs) list(rs$state, rs$steps)

cart_res_free <- function(rs) {
  for (b in cart_res_buffers(rs)) ag_buffer_free(b)
  invisible(NULL)
}

# Everything a recording over this state must list in params.
cart_res_params <- function(rs) c(cart_res_buffers(rs), unname(rs$k))

# The current state of games `cols` (f32 [N, 1], 0..N-1): the one read of each
# state buffer in a recording; the observation is the state.
cart_res_read <- function(rs, cols) {
  s <- ag_get_rows(rs$state, cols)
  list(s = s, steps = ag_get_rows(rs$steps, cols), obs = s)
}

# |v| > t as 0 / 1, strictly (|v| = t is 0); `mt` holds -t
cart_res_over <- function(v, mt) {
  ag_relu(ag_add(ag_add(ag_relu(v), ag_relu(ag_scale(v, -1))), mt))
}

# One step for actions `a` (f32 0 / 1 [1, N]) on games `cols`, from the state
# `st` read by cart_res_read() in the same recording, with noise `key`
# ([2, 1]). Writes the next state; finished games restart in the same step.
# Returns obs [4, N] (of the restarted game for finished ones), final_obs
# [4, N] (before the restart), reward, terminated, truncated and done [1, N].
cart_res_step <- function(rs, a, cols, key, st = cart_res_read(rs, cols)) {
  k <- rs$k
  p <- CARTPOLE
  n <- rs$n
  total_mass <- p$masspole + p$masscart
  pml <- p$masspole * p$length
  row <- function(j) ag_slice_rows(st$s, j - 1L, 1L)
  x <- row(1); xd <- row(2); th <- row(3); thd <- row(4)

  # physics
  force <- ag_add(ag_scale(a, 2 * p$force_mag), k$mfm)              # -F / +F
  c_ <- ag_cos(th)
  s_ <- ag_sin(th)
  temp <- ag_scale(ag_add(force, ag_scale(ag_mul(ag_mul(thd, thd), s_), pml)),
                   1 / total_mass)
  num <- ag_sub(ag_scale(s_, p$gravity), ag_mul(c_, temp))
  # l (4/3 - m_p cos^2 / M) >= l / 3 > 0: the power -1 is safe
  den <- ag_add(ag_scale(ag_mul(c_, c_), -p$length * p$masspole / total_mass), k$l43)
  thetaacc <- ag_mul(num, ag_pow(den, -1))
  xacc <- ag_sub(temp, ag_scale(ag_mul(thetaacc, c_), pml / total_mass))
  x1 <- ag_add(x, ag_scale(xd, p$tau))
  th1 <- ag_add(th, ag_scale(thd, p$tau))
  nxt <- ag_concat_rows(ag_concat_rows(ag_concat_rows(
    x1, ag_add(xd, ag_scale(xacc, p$tau))), th1), ag_add(thd, ag_scale(thetaacc, p$tau)))

  # flags
  over <- ag_add(cart_res_over(x1, k$mxt), cart_res_over(th1, k$mtt))
  term <- ag_clamp(ag_scale(over, CART_RES_IND), 0, 1)
  steps1 <- ag_add(st$steps, k$one)
  not_term <- ag_add(ag_scale(term, -1), k$one)
  trunc <- ag_mul(ag_clamp(ag_add(steps1, k$lim), 0, 1), not_term)
  done <- ag_clamp(ag_add(term, trunc), 0, 1)

  # reset by mask, then the step counter
  init <- ag_add(ag_scale(ag_rand(key, "uniform", c(4L, n), stream = 1L), 0.1), k$m05)
  s_new <- ag_add(nxt, ag_mul(done, ag_sub(init, nxt)))
  ag_set_rows(rs$state, cols, s_new)
  ag_set_rows(rs$steps, cols, ag_mul(steps1, ag_add(ag_scale(done, -1), k$one)))
  list(obs = s_new, final_obs = nxt, reward = k$ones,
       terminated = term, truncated = trunc, done = done)
}
