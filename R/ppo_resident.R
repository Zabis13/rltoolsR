# Resident PPO collection (tic-tac-toe self-play on the GPU): the policy and
# the game are one ag_capture() recording, replayed once per step with no data
# crossing the bus; the n_steps replays are queued together (ag_capture_repeat)
# and R waits once. Every step reads a counter t on the device and writes the
# step into columns t * n_env + i of the rollout buffers:
#   obs_buf   [9, N]  observations, read by the update (as on the device path)
#   res_legal [9, N]  legal actions
#   res_rec   [5, N]  action, log-probability, value, reward, done
# and t <- t + 1. After the rollout the bootstrap value is one more small
# recording (observation from the boards), then res_rec and res_legal come down
# in two transfers; GAE runs on the host.
#
# The games live in the collection's own ag_buffers (R/resident_ttt.R), not in
# the rl_gpu_env object, which is not stepped. The noise key of step t is
# (k1, k2 + t) with k2 on a 2^-12 grid, so k2 + t is exact in f32 for t < 2048:
# the same key on the host reproduces the step.

ppo_res_ok <- function(env, box, fsp, capture, meta, device) {
  inherits(env, "rl_gpu_env") && identical(env$name, "tictactoe") &&
    identical(env$opponent, "self") && identical(env$device, "gpu") &&
    !box && !fsp && capture && !meta$normalize_obs && identical(device, "gpu")
}

ppo_res_key <- function() c(stats::runif(1L), floor(stats::runif(1L) * 4096) / 4096)

# Buffers and recordings, built once per training (and again after the
# captures are freed): everything lives in `cap`, freed by ppo_free_captures().
ppo_res_setup <- function(net, cap, n_env, n_steps) {
  if (n_steps >= 2048L) stop("rollout = \"resident\": n_steps must be below 2048")
  rs <- ttt_res_new(n_env)
  cap$res_board <- rs$board; cap$res_player <- rs$player; cap$res_I9 <- rs$I9
  N <- n_env * n_steps
  cap$obs_buf <- ag_buffer(9, N)
  cap$res_legal <- ag_buffer(9, N)
  cap$res_rec <- ag_buffer(5, N)
  cap$res_cnt <- ag_buffer(1, 1)
  e2 <- ag_tensor(matrix(c(0, 1), 2, 1))
  one <- ag_tensor(matrix(1, 1, 1))
  body <- policy_act_body(net, greedy = FALSE)
  params <- c(unname(net$parameters()), ttt_res_params(rs),
              list(cap$obs_buf, cap$res_legal, cap$res_rec, cap$res_cnt, e2, one))
  cap$res_step <- ag_capture(function(cols, key, z) {
    t <- ag_get_rows(cap$res_cnt, z)                    # the one read of t
    idx <- ag_add(cols, ag_scale(t, n_env))
    s <- ttt_res_read(rs, cols)
    out <- body(s$obs, ag_scale(s$occupied, PPO_MASK), ag_add(key, ag_matmul(e2, t)))
    r <- ttt_res_step(rs, out$a, cols, s)
    ag_set_rows(cap$obs_buf, idx, s$obs)
    ag_set_rows(cap$res_legal, idx, s$legal)
    rec <- ag_concat_rows(ag_concat_rows(ag_concat_rows(ag_concat_rows(
      out$a, out$logp), out$value), r$reward), r$done)
    ag_set_rows(cap$res_rec, idx, rec)
    ag_set_rows(cap$res_cnt, z, ag_add(t, one))
    NULL
  }, params = params)
  cap$res_value <- ag_capture(function(cols)
    list(value = net$forward(ttt_res_read(rs, cols)$obs)$value),
    params = c(unname(net$parameters()), ttt_res_params(rs)))
  rs
}

# One rollout. Returns the [n_env, n_steps] matrices the host path builds,
# plus the legal-action matrix [9, N] and the bootstrap value.
ppo_res_rollout <- function(cap, n_env, n_steps, key = ppo_res_key()) {
  cols <- matrix(seq_len(n_env) - 1, n_env, 1)
  z <- matrix(0, 1, 1)
  key <- matrix(key, 2, 1)
  ag_buffer_write(cap$res_cnt, z, col_offset = 0)
  # n_steps runs queued at once, one wait (ggmlR::ag_capture_repeat)
  ag_capture_repeat(cap$res_step, n_steps, cols = cols, key = key, z = z)
  last_val <- as.numeric(cap$res_value(cols = cols)$value)
  rec <- ag_buffer_read(cap$res_rec)
  m <- function(i) matrix(rec[i, ], n_env, n_steps)
  list(act = m(1), logp = m(2), val = m(3), rew = m(4), done = m(5),
       legal = ag_buffer_read(cap$res_legal), last_val = last_val)
}
