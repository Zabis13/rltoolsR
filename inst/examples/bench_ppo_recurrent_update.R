# Recurrent PPO: one minibatch update, captured (ag_capture_step) vs the tape
# built every minibatch (ppo_rec_update).
#
#   Rscript inst/examples/bench_ppo_recurrent_update.R [device] [reps]
#
# device "gpu" (default) or "cpu"; reps timed calls per point (default 5).
# Fix the GPU clock first and restore it after:
#   echo high | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level
#   echo auto | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level
#
# For every n_envs x n_steps (T) point: one rollout of cartpole is collected,
# then the same minibatch (n_envs / 2 copies with all T steps, hidden 64) is
# updated `reps` times by each path on its own copy of the network. Reported
# separately: the first call (the captured path records the graphs and, on a
# fresh shape, compiles pipelines) and the median / q25 / q75 of the following
# calls -- the gain of capturing is the steady state. The tape grows with T
# (~22 ops per step), so the gain is expected to grow with T.
# Raw results: /tmp/bench_ppo_recurrent_update_<device>.csv

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

point <- function(n_envs, n_t) {
  set.seed(1L)
  ag_device(device)
  ag_local_mode(graph = TRUE, matmul_precision = "f32")      # as rl_train()
  env <- rl_vec_env("cartpole", n = n_envs, seed = 1L)
  algo <- rl_ppo_recurrent(n_steps = n_t, n_minibatches = 2L, hidden = 64L)
  st <- algo$init(env, device)
  ro <- algo$collect(st, env)$state$rollout
  e <- seq_len(n_envs %/% 2L)
  net_b <- ns$ppo_build(st$meta)
  ag_load_state_dict(net_b, ag_state_dict(st$net))
  opt_a <- optimizer_adam(st$net$parameters(), lr = 2.5e-4)
  opt_b <- optimizer_adam(net_b$parameters(), lr = 2.5e-4)
  caps <- new.env(parent = emptyenv())
  on.exit(ns$ppo_free_captures(caps), add = TRUE)
  tape <- function() ns$ppo_rec_update(st$net, opt_a, ro, e, 0.2, 0.5, 0.01, 0.5)
  capd <- function() ns$ppo_rec_update_captured(net_b, opt_b, caps, ro, e, 0.2, 0.5, 0.01, 0.5)
  out <- NULL
  for (path in c("tape", "captured")) {
    f <- if (path == "tape") tape else capd
    first <- ms(f())
    t <- vapply(seq_len(reps), function(i) ms(f()), numeric(1L))
    out <- rbind(out, data.frame(device = device, n_envs = n_envs, T = n_t, B = length(e),
                                 path = path, first_ms = first, median_ms = stats::median(t),
                                 q25_ms = unname(stats::quantile(t, 0.25)),
                                 q75_ms = unname(stats::quantile(t, 0.75))))
  }
  out
}

res <- NULL
for (n_envs in c(8L, 32L)) for (n_t in c(32L, 128L)) {
  r <- point(n_envs, n_t)
  print(r, row.names = FALSE, digits = 4)
  res <- rbind(res, r)
}
f <- sprintf("/tmp/bench_ppo_recurrent_update_%s.csv", device)
utils::write.csv(res, f, row.names = FALSE)
cat("\nsteady-state speedup (tape median / captured median):\n")
w <- reshape(res[, c("n_envs", "T", "path", "median_ms")], idvar = c("n_envs", "T"),
             timevar = "path", direction = "wide")
w$speedup <- w$median_ms.tape / w$median_ms.captured
print(w, row.names = FALSE, digits = 4)
cat("written:", f, "\n")
