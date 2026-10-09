# PPO speed, CPU vs GPU, on both environments, in one table.
#
#   Rscript inst/examples/bench_cpu_gpu.R [iters_tictactoe] [iters_pendulum]
#
# Each (environment, device) runs in its own R process -- the ag_* device state
# is global -- in the modes rl_train() uses (graph mode, f32 matmuls), after two
# warm-up iterations, with the update both ordinary and captured
# (rl_ppo(capture = TRUE)), interleaved A/B in the same process. Reported:
# median ms per iteration for collect and update in each mode,
# and rl_act() per call at batch 1 and batch n_env (greedy). For where the time
# goes, stage by stage, use profile_ppo_stages.R.
# Raw results: /tmp/bench_cpu_gpu_<env>_<device>.rds

args <- commandArgs(trailingOnly = TRUE)

setups <- list(
  tictactoe = list(n_env = 256L, make_env = function(dev, n)
                     # env on the CPU on both devices: the rl_gpu_env() default (see its docs)
                     rltoolsR::rl_gpu_env("tictactoe", n = n, opponent = "self", device = "cpu"),
                   algo = function(capture) rltoolsR::rl_ppo(n_steps = 9, minibatch = 2048, hidden = 64,
                                                             capture = capture)),
  pendulum  = list(n_env = 16L, make_env = function(dev, n)
                     rltoolsR::rl_vec_env("pendulum", n = n, seed = 1),
                   algo = function(capture) rltoolsR::rl_ppo(n_steps = 128, epochs = 10, minibatch = 256,
                                                      lr = 1e-3, gamma = 0.9, lambda = 0.95,
                                                      ent_coef = 0, hidden = 64,
                                                      capture = capture))
)

# ---------------------------------------------------------------- worker ----
# Both update modes (capture = FALSE / TRUE) in ONE process, each with its own
# environment and state from the same seed, iterations interleaved A/B with the
# order alternating, so drift (clocks, thermals, allocator) hits both alike.
if (length(args) && args[1] == "--worker") {
  env_name <- args[2]; dev <- args[3]; iters <- as.integer(args[4]); out_file <- args[5]
  suppressPackageStartupMessages({ library(rltoolsR); library(ggmlR) })
  s <- setups[[env_name]]
  ag_device(dev)
  ag_graph_mode(TRUE)            # rl_train() modes; algo$collect/update bypass it
  ag_matmul_precision("f32")
  runs <- lapply(c(plain = FALSE, capture = TRUE), function(cap) {
    set.seed(1)
    env <- s$make_env(dev, s$n_env)
    algo <- s$algo(cap)
    r <- new.env()
    r$env <- env; r$algo <- algo; r$state <- algo$init(env, dev)
    r
  })
  run_iter <- function(r) {
    t1 <- system.time(col <- r$algo$collect(r$state, r$env))[["elapsed"]]
    t2 <- system.time(upd <- r$algo$update(col$state))[["elapsed"]]
    r$state <- upd$state
    1000 * c(collect = t1, update = t2)
  }
  for (i in 1:2) for (r in runs) run_iter(r)
  tm <- list(plain = NULL, capture = NULL)
  for (i in seq_len(iters)) {
    ord <- if (i %% 2L) c("plain", "capture") else c("capture", "plain")
    for (m in ord) tm[[m]] <- cbind(tm[[m]], run_iter(runs[[m]]))
  }
  it <- lapply(tm, function(x) apply(x, 1, stats::median))

  # rl_act is the collection path, which the update mode does not touch.
  r <- runs$plain
  model <- r$algo$model(r$state)
  obs <- rl_reset(r$env)
  act_ms <- function(o, reps) {
    rl_act(model, o)
    1000 * system.time(for (i in seq_len(reps)) rl_act(model, o))[["elapsed"]] / reps
  }
  act <- c(act_1 = act_ms(obs[1, , drop = FALSE], 100L), act_n = act_ms(obs, 20L))

  saveRDS(list(iter = it, act = act, n_env = s$n_env, iters = iters), out_file)
  quit(save = "no")
}

# ---------------------------------------------------------------- driver ----
iters <- c(tictactoe = if (length(args) >= 1) as.integer(args[1]) else 10L,
           pendulum  = if (length(args) >= 2) as.integer(args[2]) else 3L)
self <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
rscript <- file.path(R.home("bin"), "Rscript")
has_gpu <- tryCatch(ggmlR::ggml_vulkan_available() && ggmlR::ggml_vulkan_device_count() >= 1L,
                    error = function(e) FALSE)
devs <- if (has_gpu) c("cpu", "gpu") else "cpu"
if (!has_gpu) cat("no Vulkan device: CPU only\n")

res <- list()
for (e in names(setups)) for (d in devs) {
  f <- file.path("/tmp", sprintf("bench_cpu_gpu_%s_%s.rds", e, d))
  unlink(f)
  cat(sprintf("running %s on %s (%d iterations) ...\n", e, d, iters[[e]]))
  st <- system2(rscript, c(shQuote(self), "--worker", e, d, iters[[e]], f))
  if (st != 0 || !file.exists(f)) { cat(sprintf("  FAILED (status %d)\n", st)); next }
  res[[paste(e, d)]] <- readRDS(f)
}

get <- function(e, d, what, k, mode = NULL) {
  r <- res[[paste(e, d)]]
  if (is.null(r)) return(NA_real_)
  x <- if (is.null(mode)) r[[what]] else r[[what]][[mode]]
  x[[k]]
}
val <- function(e, d, k, mode) {
  if (k == "total") get(e, d, "iter", "collect", mode) + get(e, d, "iter", "update", mode)
  else if (k %in% c("act_1", "act_n")) get(e, d, "act", k)
  else get(e, d, "iter", k, mode)
}
rows <- list()
for (e in names(setups)) {
  for (mode in c("plain", "capture")) for (k in c("collect", "update", "total")) {
    v <- sapply(devs, function(d) val(e, d, k, mode))
    rows[[length(rows) + 1L]] <- data.frame(
      env = e, measure = if (k == "total") "iteration total" else k, update = mode,
      cpu_ms = round(v[["cpu"]], 2),
      gpu_ms = if (has_gpu) round(v[["gpu"]], 2) else NA,
      gpu_vs_cpu = if (has_gpu) sprintf("%.2fx", v[["gpu"]] / v[["cpu"]]) else NA)
  }
  for (k in c("act_1", "act_n")) {
    v <- sapply(devs, function(d) val(e, d, k, NULL))
    rows[[length(rows) + 1L]] <- data.frame(
      env = e, measure = if (k == "act_1") "rl_act batch 1"
                         else sprintf("rl_act batch %d", setups[[e]]$n_env),
      update = "-", cpu_ms = round(v[["cpu"]], 2),
      gpu_ms = if (has_gpu) round(v[["gpu"]], 2) else NA,
      gpu_vs_cpu = if (has_gpu) sprintf("%.2fx", v[["gpu"]] / v[["cpu"]]) else NA)
  }
}
tab <- do.call(rbind, rows)

# The capture's own effect, per device: plain / capture (> 1x: capture faster).
sp <- list()
for (e in names(setups)) for (k in c("update", "total")) {
  sp[[length(sp) + 1L]] <- data.frame(
    env = e, measure = if (k == "total") "iteration total" else k,
    t(sapply(devs, function(d)
      sprintf("%.2fx", val(e, d, k, "plain") / val(e, d, k, "capture")))))
}
sp <- do.call(rbind, sp)

cat(sprintf("\n==== PPO, ms (median per iteration / mean per rl_act call); ggmlR %s, rltoolsR %s ====\n",
            utils::packageVersion("ggmlR"), utils::packageVersion("rltoolsR")))
cat("update = plain: rl_ppo(capture = FALSE); capture: rl_ppo(capture = TRUE), interleaved A/B\n")
cat("gpu_vs_cpu > 1x: GPU slower\n\n")
print(tab, row.names = FALSE)
cat("\n==== capture speed-up, plain / capture (> 1x: capture faster) ====\n\n")
print(sp, row.names = FALSE)
