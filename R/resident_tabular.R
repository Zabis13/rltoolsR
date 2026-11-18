# Tabular (toy_text) environments as ag_* operations, for a resident rollout:
# the step is recorded with ag_capture() together with the policy, as
# R/resident_ttt.R, so state, sampling and resets stay on the device.
#
# The table is the one of the C++ environment (rl_env_table()), S states, A
# actions, K outcomes per (s, a). Indices are f32 values, exact below 2^24.
#   state [1, N], steps [1, N]   state index and steps taken, per game
#   LP  [K, S A]    log-probabilities of the outcomes, column s A + a
#                   (-1e30 for the padding)
#   OUT [3, S A K]  next state, reward, terminated, column (s A + a) K + k
#   I   [D, S]      observation of every state: the identity for the one-hot
#                   variants, the environment's observation table (blackjack)
# The outcome is argmax(LP + Gumbel), the start of a reset game
# argmax(log isd + Gumbel): exact categorical draws on the device.
#
# As in R/resident_ttt.R: every state buffer is read by ONE ag_get_rows node in
# a recording and the write depends on it; broadcasts are [m, 1] and [1, n].
# The noise key comes from the caller (step t uses (k1, k2 + t), see
# R/ppo_resident.R); the outcome and the reset use streams 1 and 2 of it.

tab_res_new <- function(env, n) {
  check_env(env)
  tab <- rl_env_table(env)
  S <- dim(tab$prob)[1]; A <- dim(tab$prob)[2]; K <- dim(tab$prob)[3]
  if (S * A * K >= 2^24) stop("resident tabular env: table too large for f32 indices")
  rs <- new.env(parent = emptyenv())
  rs$n <- as.integer(n); rs$S <- S; rs$A <- A; rs$K <- K
  rs$name <- env$name
  rs$max_steps <- env$max_steps
  rs$onehot <- identical(env$observation_space$type, "box")
  # observation by state from a table: the environment's own (blackjack) or
  # the identity for the one-hot variants
  rs$obs_rows <- !is.null(tab$obs) || rs$onehot
  rs$isd <- tab$isd

  # [S, A, K] -> [K, A, S]: column-major order k + K (a + A s)
  ka <- function(x) aperm(x, c(3, 2, 1))
  lp <- ka(ifelse(tab$prob > 0, log(tab$prob), -1e30))
  rs$LP <- ag_buffer(K, S * A)
  ag_buffer_write(rs$LP, matrix(lp, K, S * A), col_offset = 0)
  rs$OUT <- ag_buffer(3, S * A * K)
  ag_buffer_write(rs$OUT, rbind(as.vector(ka(tab$`next`)), as.vector(ka(tab$reward)),
                                as.vector(ka(tab$terminated)) + 0), col_offset = 0)
  if (rs$obs_rows) {
    O <- if (is.null(tab$obs)) diag(S) else t(tab$obs)
    rs$I <- ag_buffer(nrow(O), S)
    ag_buffer_write(rs$I, O, col_offset = 0)
  }
  rs$state <- ag_buffer(1, n)
  rs$steps <- ag_buffer(1, n)

  tn <- function(v, r, c) ag_tensor(matrix(v, r, c))
  rs$k <- list(
    one = tn(1, 1, 1),
    lisd = tn(ifelse(tab$isd > 0, log(tab$isd), -1e30), S, 1),
    lim = tn(-(rs$max_steps - 1), 1, 1),
    s1 = tn(c(1, 0, 0), 1, 3), s2 = tn(c(0, 1, 0), 1, 3), s3 = tn(c(0, 0, 1), 1, 3))
  tab_res_reset(rs)
  rs
}

# New games for all N, drawn from isd with R's RNG.
tab_res_reset <- function(rs) {
  s0 <- sample.int(rs$S, rs$n, replace = TRUE, prob = rs$isd) - 1
  ag_buffer_write(rs$state, matrix(s0, 1, rs$n), col_offset = 0)
  ag_buffer_write(rs$steps, matrix(0, 1, rs$n), col_offset = 0)
  invisible(rs)
}

tab_res_buffers <- function(rs) {
  c(list(rs$state, rs$steps, rs$LP, rs$OUT), if (rs$obs_rows) list(rs$I))
}

tab_res_free <- function(rs) {
  for (b in tab_res_buffers(rs)) ag_buffer_free(b)
  invisible(NULL)
}

# Everything a recording over this state must list in params.
tab_res_params <- function(rs) c(tab_res_buffers(rs), unname(rs$k))

tab_res_obs <- function(rs, s) if (rs$obs_rows) ag_get_rows(rs$I, s) else s

# The current state of games `cols` (f32 [N, 1], 0..N-1): the one read of each
# state buffer in a recording, plus the observation the policy sees.
tab_res_read <- function(rs, cols) {
  s <- ag_get_rows(rs$state, cols)
  list(s = s, steps = ag_get_rows(rs$steps, cols), obs = tab_res_obs(rs, s))
}

# One step for actions `a` (f32 action indices [1, N]) on games `cols`, from the
# state `st` read by tab_res_read() in the same recording, with noise `key`
# ([2, 1]). Writes the next state; finished games restart in the same step.
# Returns obs (of the restarted game for finished ones), final_obs (before the
# restart), reward, terminated, truncated and done, each [1, N] (obs and
# final_obs [S, N] for onehot).
tab_res_step <- function(rs, a, cols, key, st = tab_res_read(rs, cols)) {
  k <- rs$k
  n <- rs$n
  idx <- ag_add(ag_scale(st$s, rs$A), a)                       # s A + a
  col <- if (rs$K == 1L) idx else {
    g <- ag_rand(key, "gumbel", c(rs$K, n), stream = 1L)
    kk <- ag_argmax(ag_add(ag_get_rows(rs$LP, idx), g), as_tensor = TRUE)
    ag_add(ag_scale(idx, rs$K), kk)
  }
  o <- ag_get_rows(rs$OUT, col)                                 # [3, N]
  nxt <- ag_matmul(k$s1, o)
  reward <- ag_matmul(k$s2, o)
  term <- ag_matmul(k$s3, o)
  steps1 <- ag_add(st$steps, k$one)
  not_term <- ag_add(ag_scale(term, -1), k$one)
  trunc <- if (rs$max_steps > 0) ag_mul(ag_clamp(ag_add(steps1, k$lim), 0, 1), not_term) else
    ag_scale(term, 0)
  done <- ag_clamp(ag_add(term, trunc), 0, 1)
  init <- ag_argmax(ag_add(ag_rand(key, "gumbel", c(rs$S, n), stream = 2L), k$lisd),
                    as_tensor = TRUE)
  s_new <- ag_add(nxt, ag_mul(done, ag_sub(init, nxt)))
  ag_set_rows(rs$state, cols, s_new)
  ag_set_rows(rs$steps, cols, ag_mul(steps1, ag_add(ag_scale(done, -1), k$one)))
  list(obs = tab_res_obs(rs, s_new), final_obs = tab_res_obs(rs, nxt), reward = reward,
       terminated = term, truncated = trunc, done = done)
}
