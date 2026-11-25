# Recurrent PPO: how the time to RECORD the captured update grows with the
# sequence length T. Linear growth is expected (the tape is ~22 ops per step);
# a slope above 1 would point at a per-op cost that grows with the graph
# (lookups in the deferred graph, the reachability count, ...).
#
#   Rscript inst/examples/bench_ppo_recurrent_record.R [device] [reps]
#
# device "gpu" (default) or "cpu"; reps fresh recordings per T (default 5).
# Fix the GPU clock first and restore it after:
#   echo high | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level
#   echo auto | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level
#
# For every T (n_envs 8, minibatch of 4 copies, hidden 64, cartpole): one
# recording is made first and dropped -- it compiles the pipelines for the new
# shapes, which is a one-off per process, not a cost of recording. Then `reps`
# times: an empty cache, the first call (record + one replay), and 5 further
# calls (replays). record = first call - median replay. Reported per T: the
# median and quartiles of record, the median replay, and at the end the slope
# of log(record) on log(T) (1 = linear).
# Raw results: /tmp/bench_ppo_recurrent_record_<device>.csv

suppressMessages({ library(ggmlR); library(rltoolsR) })
args <- commandArgs(trailingOnly = TRUE)
device <- if (length(args) >= 1L) args[[1L]] else "gpu"
reps <- if (length(args) >= 2L) as.integer(args[[2L]]) else 5L
if (device == "gpu" && !(ggml_vulkan_available() && ggml_vulkan_device_count() > 0L))
  stop("no Vulkan device")

ns <- asNamespace("rltoolsR")
ms <- function(expr) {
  t0 <- proc.time()[["elapsed"]]
  force(expr)
  (proc.time()[["elapsed"]] - t0) * 1000
}

point <- function(n_t, n_envs = 8L) {
  set.seed(1L)
  ag_device(device)
  ag_local_mode(graph = TRUE, matmul_precision = "f32")      # as rl_train()
  env <- rl_vec_env("cartpole", n = n_envs, seed = 1L)
  algo <- rl_ppo_recurrent(n_steps = n_t, n_minibatches = 2L, hidden = 64L)
  st <- algo$init(env, device)
  ro <- algo$collect(st, env)$state$rollout
  e <- seq_len(n_envs %/% 2L)
  opt <- optimizer_adam(st$net$parameters(), lr = 2.5e-4)
  one <- function(caps) ns$ppo_rec_update_captured(st$net, opt, caps, ro, e, 0.2, 0.5, 0.01, 0.5)
  # warm-up recording: pipeline compilation for these shapes
  caps <- new.env(parent = emptyenv())
  one(caps)
  ns$ppo_free_captures(caps)
  rows <- NULL
  for (r in seq_len(reps)) {
    caps <- new.env(parent = emptyenv())
    first <- ms(one(caps))
    replay <- stats::median(vapply(1:5, function(i) ms(one(caps)), numeric(1L)))
    ns$ppo_free_captures(caps)
    rows <- rbind(rows, data.frame(device = device, T = n_t, rep = r, first_ms = first,
                                   replay_ms = replay, record_ms = first - replay))
  }
  rows
}

res <- NULL
for (n_t in c(32L, 64L, 128L, 256L)) {
  r <- point(n_t)
  cat(sprintf("T = %3d  record median %7.1f ms (q25 %7.1f, q75 %7.1f)  replay %6.1f ms\n",
              n_t, stats::median(r$record_ms), stats::quantile(r$record_ms, 0.25),
              stats::quantile(r$record_ms, 0.75), stats::median(r$replay_ms)))
  res <- rbind(res, r)
}
f <- sprintf("/tmp/bench_ppo_recurrent_record_%s.csv", device)
utils::write.csv(res, f, row.names = FALSE)
agg <- stats::aggregate(record_ms ~ T, res, stats::median)
fit <- stats::lm(log(record_ms) ~ log(T), agg)
cat(sprintf("\nslope of log(record) on log(T): %.2f (1 = linear)\n", stats::coef(fit)[[2L]]))
cat("record per step:", paste(sprintf("T=%d %.2f ms", agg$T, agg$record_ms / agg$T),
                              collapse = ", "), "\n")
cat("written:", f, "\n")
