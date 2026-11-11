# Where the resident collection (rl_ppo(rollout = "resident"), tic-tac-toe
# self-play on the GPU) spends its time.
#
#   Rscript inst/examples/bench_resident_parts.R [iters] [time|perf]
#
# time (default): median microseconds per call over `iters` batches of 20 calls
#   collect     -- algo$collect(), the whole thing
#   cnt_reset   -- ag_buffer_write of the step counter (1 upload)
#   step_1      -- one replay of the step graph (upload check + launch + wait)
#   repeat_n    -- counter reset + ag_capture_repeat(step, n), n = 1, 9, 36:
#                  the slope over n is the cost of one more step when the runs
#                  are queued (command recording on the CPU and the graph on
#                  the GPU, overlapped); the intercept is the fixed cost
#   bootstrap   -- the value recording (observation from the boards)
#   dl_rec      -- download of the [5, N] record buffer
#   dl_legal    -- download of the [9, N] legal-action buffer
#   host        -- collect minus the device parts above (GAE, rollout list, R)
# then:
#   1. 36 steps queued straight through ggml: recording + submit on the CPU
#      vs the wait for the GPU after the last submit
#   2. the hidden-layer matmul [64,64] x [64,n_env], matmul_precision "f32"
#      (as rl_train() runs) vs "default", per matmul
#   3. collect stages from rl_prof, and ppo_gae() alone
# perf: GPU time of every node of ONE step graph (GGML_VK_PERF_LOGGER; it
#   synchronizes after each graph, so do not read wall times in this mode).
#
# Raw results: /tmp/bench_resident_parts.rds

args <- commandArgs(trailingOnly = TRUE)
iters <- if (length(args) >= 1) as.integer(args[1]) else 30L
mode <- if (length(args) >= 2) args[2] else "time"
stopifnot(mode %in% c("time", "perf"))
if (mode == "perf") Sys.setenv(GGML_VK_PERF_LOGGER = "1")

suppressPackageStartupMessages({ library(rltoolsR); library(ggmlR) })
if (!(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L)) stop("no Vulkan device")

ag_device("gpu")
ag_graph_mode(TRUE)            # rl_train() modes
ag_matmul_precision("f32")

n_env <- 256L
n_steps <- 9L
set.seed(1)
env <- rl_gpu_env("tictactoe", n = n_env, opponent = "self", device = "gpu")
algo <- rl_ppo(n_steps = n_steps, minibatch = 2048, hidden = 64, rollout = "resident")
state <- algo$init(env, "gpu")
warm <- function() for (i in 1:3) state <<- algo$update(algo$collect(state, env)$state)$state
if (mode == "perf") invisible(capture.output(warm())) else warm()
stopifnot(identical(state$obs_path, "resident"))

cap <- state$captures
cols <- matrix(seq_len(n_env) - 1, n_env, 1)
z <- matrix(0, 1, 1)
key <- matrix(c(0.3, 0.5), 2, 1)

if (mode == "perf") {
  # the counter past the rollout is clamped to its last column: harmless here
  invisible(capture.output(cap$res_step(cols = cols, key = key, z = z)))
  out <- capture.output(cap$res_step(cols = cols, key = key, z = z))
  cat(sprintf("\nGPU time per node of one step graph (%d nodes), n_env = %d\n",
              ggml_graph_n_nodes(attr(cap$res_step, "captures")[[
                ls(attr(cap$res_step, "captures"))[1]]]$graph), n_env))
  cat(out, sep = "\n")
  quit(save = "no")
}

# median microseconds per call of f() over `iters` batches of `k` calls
per_call <- function(f, k = 20L) {
  for (i in 1:3) f()
  us <- vapply(seq_len(iters), function(j) {
    t0 <- Sys.time()
    for (i in seq_len(k)) f()
    1e6 * as.numeric(difftime(Sys.time(), t0, units = "secs")) / k
  }, 0)
  stats::median(us)
}

rep_n <- function(n) function() {
  ag_buffer_write(cap$res_cnt, z, col_offset = 0)
  ag_capture_repeat(cap$res_step, n, cols = cols, key = key, z = z)
}

us <- c(
  collect   = per_call(function() algo$collect(state, env), k = 5L),
  cnt_reset = per_call(function() ag_buffer_write(cap$res_cnt, z, col_offset = 0)),
  step_1    = per_call(function() cap$res_step(cols = cols, key = key, z = z)),
  repeat_1  = per_call(rep_n(1L)),
  repeat_9  = per_call(rep_n(9L)),
  repeat_36 = per_call(rep_n(36L), k = 5L),
  bootstrap = per_call(function() cap$res_value(cols = cols)),
  dl_rec    = per_call(function() ag_buffer_read(cap$res_rec)),
  dl_legal  = per_call(function() ag_buffer_read(cap$res_legal))
)
slope <- (us[["repeat_36"]] - us[["repeat_9"]]) / 27
icpt <- us[["repeat_9"]] - 9 * slope
device <- us[["repeat_9"]] + us[["bootstrap"]] + us[["dl_rec"]] + us[["dl_legal"]]
host <- us[["collect"]] - device

cat(sprintf("\nresident collect, tic-tac-toe, n_env = %d, n_steps = %d, %d batches\n",
            n_env, n_steps, iters))
cat("\nmicroseconds per call (median)\n")
print(round(us, 1))
cat(sprintf("\nqueued step: %.1f us per extra step (slope 9 -> 36), fixed %.1f us\n",
            slope, icpt))
cat(sprintf("collect %.0f us = steps (repeat_9) %.0f + bootstrap %.0f + downloads %.0f + host %.0f\n",
            us[["collect"]], us[["repeat_9"]], us[["bootstrap"]],
            us[["dl_rec"]] + us[["dl_legal"]], host))
cat(sprintf("shares: steps %.0f%%, bootstrap %.0f%%, downloads %.0f%%, host %.0f%%\n",
            100 * us[["repeat_9"]] / us[["collect"]], 100 * us[["bootstrap"]] / us[["collect"]],
            100 * (us[["dl_rec"]] + us[["dl_legal"]]) / us[["collect"]],
            100 * host / us[["collect"]]))

# ---- 1. command recording on the CPU ---------------------------------------
# The step graph queued 36 times straight through ggml: the time until the last
# call returns is recording + submit on the CPU (the GPU runs behind it); the
# wait after it is what the GPU still has left.
recs <- attr(cap$res_step, "captures")
rc <- recs[[ls(recs)[1L]]]
cap$res_step(cols = cols, key = key, z = z)      # inputs on the device as below
rec_wait <- t(vapply(seq_len(iters), function(j) {
  ag_buffer_write(cap$res_cnt, z, col_offset = 0)
  t0 <- Sys.time()
  for (i in 1:36) ggml_backend_graph_compute_async(rc$backend, rc$graph)
  t1 <- Sys.time()
  ggml_backend_synchronize(rc$backend)
  t2 <- Sys.time()
  1e6 * c(record = as.numeric(difftime(t1, t0, units = "secs")),
          wait = as.numeric(difftime(t2, t1, units = "secs")))
}, c(record = 0, wait = 0)))
rw <- apply(rec_wait, 2, stats::median)
cat(sprintf("\n1. 36 queued steps: recording + submit on the CPU %.0f us (%.1f per step), then wait %.0f us\n",
            rw[["record"]], rw[["record"]] / 36, rw[["wait"]]))

# ---- 2. the hidden-layer matmul, f32 tiles vs default ----------------------
# [64, 64] x [64, 256] as in the network, 20 in a chain in one recording, so
# the launch is shared: time per call / 20 ~ one matmul
W <- ag_param(diag(64))
x0 <- matrix(stats::runif(64 * n_env), 64, n_env)
mm_us <- vapply(c("f32", "default"), function(prec) {
  old <- ag_matmul_precision(prec)
  on.exit(ag_matmul_precision(old))
  f <- ag_capture(function(x) {
    y <- x
    for (i in 1:20) y <- ag_matmul(W, y)
    list(y = y)
  }, params = list(W))
  on.exit(ag_capture_free(f), add = TRUE)
  one <- per_call(function() f(x = x0))
  # the same recording with 1 matmul: subtract the launch and transfers
  g <- ag_capture(function(x) list(y = ag_matmul(W, x)), params = list(W))
  on.exit(ag_capture_free(g), add = TRUE)
  (one - per_call(function() g(x = x0))) / 19
}, 0)
cat(sprintf("2. matmul [64,64] x [64,%d]: f32 tiles %.1f us, default %.1f us (ratio %.1f)\n",
            n_env, mm_us[["f32"]], mm_us[["default"]], mm_us[["f32"]] / mm_us[["default"]]))

# ---- 3. host stages of the collection --------------------------------------
rltoolsR:::rl_prof(TRUE)
for (i in seq_len(iters)) algo$collect(state, env)
pr <- rltoolsR:::rl_prof_get()
rltoolsR:::rl_prof(FALSE)
stage_us <- 1000 * pr$ms / iters
r <- rltoolsR:::ppo_res_rollout(cap, n_env, n_steps)
gae_us <- per_call(function() rltoolsR:::ppo_gae(r$rew, r$val, r$done, r$last_val,
                                                 0.99, 0.95, -1))
cat("3. collect stages (rl_prof), us per collection\n")
print(round(stage_us, 1))
cat(sprintf("   of which ppo_gae alone %.1f us; collect minus stages %.0f us\n",
            gae_us, us[["collect"]] - sum(stage_us)))

out_file <- "/tmp/bench_resident_parts.rds"
saveRDS(list(us = us, slope = slope, intercept = icpt, host = host,
             record_wait = rw, matmul_us = mm_us, stage_us = stage_us, gae_us = gae_us,
             n_env = n_env, n_steps = n_steps, iters = iters), out_file)
cat("saved", out_file, "\n")
