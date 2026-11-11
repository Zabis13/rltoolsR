# What a resident rollout step could save: launch floor vs the current step.
#
#   Rscript inst/examples/bench_step_floor.R [iters] [n_env]
#
# 1. Launch floor, per call (microseconds, median of `iters` batches of 50 calls):
#    cap_tiny   -- ag_capture of one ag_get_rows from an ag_buffer, 1 column out
#                  (upload of one index + launch + download of 9 numbers)
#    cap_obs    -- the same, n_env columns out (download of the observations)
#    env_graph  -- tic-tac-toe step graph alone (ggml_backend_sched_graph_compute,
#                  no uploads or downloads)
#    env_step   -- rl_step(): action upload + graph + reward/done/legal/player download
#    act_buf    -- PPO act graph from the buffer (policy_act_buf, as on the device path)
#    If cap_tiny is well below the current step (section 2), a step graph
#    replayed from R (one launch per step, no data) takes most of the gain; if
#    not, the replay belongs inside ggmlR (n steps in one submit).
#
# 2. Current device path (dev_buf in bench_obs_device.R): collect stages per
#    environment step, ms (rl_prof, R/prof.R), mean over `iters` collections.
#
# Raw results: /tmp/bench_step_floor.rds

args <- commandArgs(trailingOnly = TRUE)
iters <- if (length(args) >= 1) as.integer(args[1]) else 20L
n_env <- if (length(args) >= 2) as.integer(args[2]) else 256L

suppressPackageStartupMessages({ library(rltoolsR); library(ggmlR) })
if (!(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L)) stop("no Vulkan device")

ag_device("gpu")
ag_graph_mode(TRUE)            # rl_train() modes
ag_matmul_precision("f32")

# median microseconds per call of f() over `iters` batches of `k` calls
per_call <- function(f, k = 50L) {
  for (i in 1:5) f()
  us <- vapply(seq_len(iters), function(j) {
    t0 <- Sys.time()
    for (i in seq_len(k)) f()
    1e6 * as.numeric(difftime(Sys.time(), t0, units = "secs")) / k
  }, 0)
  stats::median(us)
}

# ---- 1. launch floor -------------------------------------------------------
set.seed(1)
buf <- ag_buffer(9, n_env)
ag_buffer_write(buf, matrix(stats::runif(9 * n_env), 9, n_env), col_offset = 0)
cap <- ag_capture(function(idx) list(y = ag_get_rows(buf, idx)), params = list(buf))
idx1 <- 0
idx_all <- seq_len(n_env) - 1

env <- rl_gpu_env("tictactoe", n = n_env, opponent = "self", device = "gpu")
invisible(rl_reset(env))
st <- env$st
step_graph <- st$steps[[st$cur]]
act0 <- rep(0L, n_env)

algo <- rl_ppo(n_steps = 9, minibatch = 2048, hidden = 64, rollout = "host")
state <- algo$init(env, "gpu")
caches <- new.env(parent = emptyenv())
legal <- matrix(1, 9, n_env)

floor_us <- c(
  cap_tiny  = per_call(function() as.numeric(cap(idx = idx1)$y)),
  cap_obs   = per_call(function() as.numeric(cap(idx = idx_all)$y)),
  env_graph = per_call(function() ggml_backend_sched_graph_compute(step_graph$sched,
                                                                   step_graph$graph)),
  env_step  = per_call(function() rl_step(env, act0)),
  act_buf   = per_call(function() rltoolsR:::policy_act_buf(state$net, buf, idx_all, legal,
                                                            caches, "act"))
)
ag_capture_free(cap)

# ---- 2. current device path, stages per step -------------------------------
env2 <- rl_gpu_env("tictactoe", n = n_env, opponent = "self", device = "gpu")
set.seed(1)
st2 <- algo$init(env2, "gpu")
for (i in 1:3) st2 <- algo$collect(st2, env2)$state
rltoolsR:::rl_prof(TRUE)
t0 <- Sys.time()
for (i in seq_len(iters)) st2 <- algo$collect(st2, env2)$state
wall <- 1000 * as.numeric(difftime(Sys.time(), t0, units = "secs")) / iters
pr <- rltoolsR:::rl_prof_get()
rltoolsR:::rl_prof(FALSE)
stopifnot(identical(st2$obs_path, "device"))

n_steps <- 9L
stage_ms <- sort(pr$ms / iters, decreasing = TRUE)
step_ms <- (wall - sum(stage_ms[grepl("bootstrap|gae", names(stage_ms))])) / n_steps

cat(sprintf("\ntictactoe, n_env = %d, %d batches / iterations\n", n_env, iters))
cat("\n1. launch floor, microseconds per call (median)\n")
print(round(floor_us, 1))
cat("\n2. device-path collect, ms per iteration (", n_steps, " steps)\n", sep = "")
print(round(stage_ms, 3))
cat(sprintf("collect wall %.2f ms; per step (without bootstrap/gae) %.3f ms\n", wall, step_ms))
cat(sprintf("cap_tiny / step: %.1f%%\n", 100 * floor_us[["cap_tiny"]] / (1000 * step_ms)))

out_file <- "/tmp/bench_step_floor.rds"
saveRDS(list(floor_us = floor_us, stage_ms = stage_ms, wall_ms = wall, step_ms = step_ms,
             n_env = n_env, iters = iters), out_file)
cat("saved", out_file, "\n")
