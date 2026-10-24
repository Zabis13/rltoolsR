# PPO collect on the GPU vs the CPU: same weights, same seed, one rollout each.
#
#   Rscript inst/examples/check_collect_cpu_gpu.R
#
# Each device runs in its own R process (the ag_* device state is global), in
# the modes rl_train() uses (graph mode, f32 matmuls). The CPU worker runs
# first and saves the initial weights; the GPU worker loads them, so both
# collect with the same network from the same R seed. Environments stay on the
# CPU on both devices: only the network's device differs.
#
# Checked, per environment:
#   tictactoe  terminated episodes, legal masks, self-play turn sign
#   pendulum   truncation at max_steps and the bootstrap of final_obs
# Compared: actions, logp, advantages and returns of the rollout (relative to
# their largest magnitude, since they scale with the rewards), returns at
# the truncation steps alone (where the bootstrap enters), and rl_act()
# (greedy) on the collected observations.
# Raw results: /tmp/check_collect_<env>_<device>.rds

args <- commandArgs(trailingOnly = TRUE)

setups <- list(
  tictactoe = list(n_env = 64L,
                   make_env = function()
                     rltoolsR::rl_gpu_env("tictactoe", n = 64L, opponent = "self", device = "cpu"),
                   algo = function(env) rltoolsR::rl_ppo(n_steps = 9, minibatch = 576)),
  pendulum  = list(n_env = 8L,
                   make_env = function() rltoolsR::rl_vec_env("pendulum", n = 8L, seed = 1),
                   # longer than an episode, so every copy is truncated once
                   algo = function(env) rltoolsR::rl_ppo(n_steps = env$max_steps + 8L,
                                                         minibatch = 256, gamma = 0.9))
)

# ---------------------------------------------------------------- worker ----
if (length(args) && args[1] == "--worker") {
  env_name <- args[2]; dev <- args[3]; w_file <- args[4]; out_file <- args[5]
  suppressPackageStartupMessages({ library(rltoolsR); library(ggmlR) })
  s <- setups[[env_name]]
  ag_device(dev)
  ag_graph_mode(TRUE)            # rl_train() modes; algo$collect bypasses it
  ag_matmul_precision("f32")

  set.seed(1)
  env <- s$make_env()
  algo <- s$algo(env)
  state <- algo$init(env, dev)
  if (dev == "cpu") {
    saveRDS(ag_state_dict(state$net), w_file)
  } else {
    ag_load_state_dict(state$net, readRDS(w_file))
  }

  set.seed(2)
  col <- algo$collect(state, env)
  ro <- col$state$rollout
  model <- algo$model(col$state)
  act <- rl_act(model, t(ro$obs), legal = if (!is.null(ro$legal)) t(ro$legal) > 0)

  saveRDS(list(rollout = ro, episodes = col$info[["episodes"]], act = act,
               n_env = env$n, max_steps = env$max_steps), out_file)
  quit(save = "no")
}

# ------------------------------------------------------------------ main ----
if (!ggmlR::ggml_vulkan_available()) stop("no Vulkan device: nothing to compare")

self <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
rscript <- file.path(R.home("bin"), "Rscript")

maxdiff <- function(a, b) max(abs(a - b))
# advantages and returns scale with the rewards (pendulum: returns ~100), so
# they are compared relative to their largest magnitude
reldiff <- function(a, b) maxdiff(a, b) / max(abs(a), .Machine$double.eps)

rows <- list()
for (e in names(setups)) {
  w_file <- sprintf("/tmp/check_collect_%s_weights.rds", e)
  res <- list()
  for (d in c("cpu", "gpu")) {
    f <- sprintf("/tmp/check_collect_%s_%s.rds", e, d)
    st <- system2(rscript, c(shQuote(self), "--worker", e, d, w_file, f))
    if (st != 0) stop("worker ", e, " ", d, " failed")
    res[[d]] <- readRDS(f)
  }
  a <- res$cpu$rollout; b <- res$gpu$rollout
  n_env <- res$cpu$n_env
  n_t <- length(a$logp) / n_env

  row <- data.frame(
    env = e,
    episodes = sprintf("%d / %d", res$cpu$episodes, res$gpu$episodes),
    act_mismatch = if (is.matrix(a$act)) sum(abs(a$act - b$act) > 1e-4) else sum(a$act != b$act),
    obs = maxdiff(a$obs, b$obs),
    logp = maxdiff(a$logp, b$logp),
    adv_rel = reldiff(a$adv, b$adv),
    ret_rel = reldiff(a$ret, b$ret),
    ret_trunc_rel = NA_real_,
    rl_act = if (is.matrix(res$cpu$act)) maxdiff(res$cpu$act, res$gpu$act)
             else sum(res$cpu$act != res$gpu$act)
  )
  # pendulum copies reset together, so all of them are truncated at step
  # max_steps; columns of the rollout are steps, rows environments
  if (e == "pendulum") {
    t_tr <- res$cpu$max_steps
    if (t_tr <= n_t) {
      idx <- (t_tr - 1L) * n_env + seq_len(n_env)
      row$ret_trunc_rel <- reldiff(a$ret[idx], b$ret[idx])
    }
  }
  rows[[e]] <- row
}

out <- do.call(rbind, rows)
rownames(out) <- NULL
cat("\nPPO collect, GPU vs CPU (max |difference|; *_rel divided by the largest",
    "|value|; act_mismatch and discrete rl_act are counts):\n\n")
print(out, digits = 3)
cat("\nExpected: act_mismatch 0, discrete rl_act 0; *_rel at float32 rounding",
    "(~1e-6 and below); obs, logp and continuous rl_act ~1e-5 and below",
    "(pendulum trajectories drift apart by rounding over an episode).\n")
