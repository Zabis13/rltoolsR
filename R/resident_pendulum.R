# Pendulum-v1 as ag_* operations, for a resident rollout: the step is recorded
# with ag_capture() together with the policy, as R/resident_cartpole.R.
#
# The physics is the one of R/gpu_pendulum.R and the C++ "pendulum_gym" /
# "pendulum_gym_raw" (Gymnasium classic_control/pendulum.py), in f32:
#   state [2, N]  theta, theta_dot, one column per game
#   steps [1, N]  steps taken
# theta is kept wrapped to [-pi, pi]: one correction of 2 pi after each step
# is enough (|d theta| <= max_speed dt = 0.4). sin and cos are periodic, so
# the dynamics and the observation [cos, sin, theta_dot] do not change, and
# angle_normalize(theta) of the reward is theta itself (theta = pi gives pi^2
# as angle_normalize(pi) = -pi does). Gymnasium keeps theta unwrapped; the
# stored value differs from it by a multiple of 2 pi, which never leaves.
#
# One step: torque u = 2 / hi clip(a, -hi, hi) (hi = 2 for _raw: clip(a, -2, 2);
# hi = 1: 2 clip(a, -1, 1)), the reward from the state before the step
# -(theta^2 + 0.1 theta_dot^2 + 0.001 u^2), the physics, then truncation at
# 200 steps (never terminated) and the reset by mask: theta uniform in
# [-pi, pi], theta_dot in [-1, 1], from stream 1 of the step's noise key
# (stream 0 is the policy's).
#
# As in R/resident_ttt.R: every state buffer is read by ONE ag_get_rows node in
# a recording and the write depends on it; broadcasts are [m, 1] and [1, n].

pend_res_new <- function(n, raw) {
  ggmlr_need(RES_TRIG, "the resident pendulum")
  k <- PENDULUM_GYM
  rs <- new.env(parent = emptyenv())
  rs$n <- as.integer(n)
  rs$raw <- raw
  rs$hi <- if (raw) k$max_torque else 1
  rs$max_steps <- k$max_steps
  rs$state <- ag_buffer(2, n)
  rs$steps <- ag_buffer(1, n)
  tn <- function(v, r, c) ag_tensor(matrix(v, r, c))
  rs$k <- list(
    one = tn(1, 1, 1),
    mpi = tn(-pi, 1, 1),
    lim = tn(-(k$max_steps - 1), 1, 1),
    span = tn(c(2 * pi, 2), 2, 1),     # [2, 1]: reset ranges
    lo = tn(c(-pi, -1), 2, 1))
  pend_res_reset(rs)
  rs
}

# New games for all N (R's RNG), as pendulum_initial().
pend_res_reset <- function(rs) {
  ag_buffer_write(rs$state, rbind(stats::runif(rs$n, -pi, pi), stats::runif(rs$n, -1, 1)),
                  col_offset = 0)
  ag_buffer_write(rs$steps, matrix(0, 1, rs$n), col_offset = 0)
  invisible(rs)
}

pend_res_buffers <- function(rs) list(rs$state, rs$steps)

pend_res_free <- function(rs) {
  for (b in pend_res_buffers(rs)) ag_buffer_free(b)
  invisible(NULL)
}

# Everything a recording over this state must list in params.
pend_res_params <- function(rs) c(pend_res_buffers(rs), unname(rs$k))

# [cos theta, sin theta, theta_dot] [3, N] of a state [2, N]
pend_res_obs <- function(s) {
  th <- ag_slice_rows(s, 0L, 1L)
  ag_concat_rows(ag_concat_rows(ag_cos(th), ag_sin(th)), ag_slice_rows(s, 1L, 1L))
}

# The current state of games `cols` (f32 [N, 1], 0..N-1): the one read of each
# state buffer in a recording, plus the observation the policy sees.
pend_res_read <- function(rs, cols) {
  s <- ag_get_rows(rs$state, cols)
  list(s = s, steps = ag_get_rows(rs$steps, cols), obs = pend_res_obs(s))
}

# v > 0 as 0 / 1, strictly (see CART_RES_IND)
pend_res_pos <- function(v) ag_clamp(ag_scale(ag_relu(v), CART_RES_IND), 0, 1)

# One step for actions `a` (unclipped, [1, N]) on games `cols`, from the state
# `st` read by pend_res_read() in the same recording, with noise `key`
# ([2, 1]). Writes the next state; finished games restart in the same step.
# Returns obs [3, N] (of the restarted game for finished ones), final_obs
# [3, N] (before the restart), reward, terminated, truncated and done [1, N].
pend_res_step <- function(rs, a, cols, key, st = pend_res_read(rs, cols)) {
  k <- rs$k
  p <- PENDULUM_GYM
  th <- ag_slice_rows(st$s, 0L, 1L)
  thd <- ag_slice_rows(st$s, 1L, 1L)

  # torque and reward (the state before the step; theta is already wrapped)
  u <- ag_scale(ag_clamp(a, -rs$hi, rs$hi), p$max_torque / rs$hi)
  reward <- ag_scale(ag_add(ag_add(ag_mul(th, th), ag_scale(ag_mul(thd, thd), 0.1)),
                            ag_scale(ag_mul(u, u), 0.001)), -1)

  # physics, then theta back into [-pi, pi]
  acc <- ag_add(ag_scale(ag_sin(th), 3 * p$g / (2 * p$l) * p$dt),
                ag_scale(u, 3 / (p$m * p$l^2) * p$dt))
  thd1 <- ag_clamp(ag_add(thd, acc), -p$max_speed, p$max_speed)
  th1 <- ag_add(th, ag_scale(thd1, p$dt))
  over <- pend_res_pos(ag_add(th1, k$mpi))                       # th1 > pi
  under <- pend_res_pos(ag_add(ag_scale(th1, -1), k$mpi))        # th1 < -pi
  th1 <- ag_add(th1, ag_scale(ag_sub(under, over), 2 * pi))
  nxt <- ag_concat_rows(th1, thd1)

  # truncation only
  steps1 <- ag_add(st$steps, k$one)
  trunc <- ag_clamp(ag_add(steps1, k$lim), 0, 1)
  term <- ag_scale(trunc, 0)
  done <- trunc

  # reset by mask, then the step counter
  init <- ag_add(ag_mul(ag_rand(key, "uniform", c(2L, rs$n), stream = 1L), k$span), k$lo)
  s_new <- ag_add(nxt, ag_mul(done, ag_sub(init, nxt)))
  ag_set_rows(rs$state, cols, s_new)
  ag_set_rows(rs$steps, cols, ag_mul(steps1, ag_add(ag_scale(done, -1), k$one)))
  list(obs = pend_res_obs(s_new), final_obs = pend_res_obs(nxt), reward = reward,
       terminated = term, truncated = trunc, done = done)
}
