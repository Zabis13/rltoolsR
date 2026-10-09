# Where does a PPO iteration spend its time: CPU vs GPU, stage by stage.
#
#   Rscript inst/examples/profile_ppo_stages.R [tictactoe|pendulum] [n_env] [iterations]
#
# Runs three configurations, each in its own R process (the ag_* device state is
# global, so one process per configuration keeps them from leaking):
#   cpu        -- device "cpu"
#   gpu-graph  -- device "gpu", graph mode + f32 matmuls, as rl_train() runs
#   gpu-eager  -- device "gpu", graph mode off: every ag_* op computes before it
#                 returns, so its time lands in its own stage
# In graph mode the forward is deferred to the next materialization, so
# "collect: forward" / "update: forward" read ~0 and their cost shows up in
# "collect: download" / "update: backward" -- compare with gpu-eager for that.
#
# Stages are marked in R/ppo.R and R/ppo_box.R (prof_tick, off by default).
# Prints, per iteration:
#   1. collect / update time, clean run (profiling off);
#   2. time per stage, ms and share, and GPU/CPU ratio;
#   3. GPU launches and host<->device transfers per stage;
#   4. inside ggmlR: per-op stages of the eager forward (ctx, create, flush,
#      upload, graph, compute, download) and of the graph backward.
# Results: /tmp/profile_ppo_stages_<env>_<config>.rds

args <- commandArgs(trailingOnly = TRUE)

# ---------------------------------------------------------------- worker ----
if (length(args) && args[1] == "--worker") {
  cfg <- args[2]; env_name <- args[3]
  n_env <- as.integer(args[4]); iters <- as.integer(args[5]); out_file <- args[6]
  suppressPackageStartupMessages({ library(rltoolsR); library(ggmlR) })
  ns <- asNamespace("ggmlR")
  dev <- if (cfg == "cpu") "cpu" else "gpu"

  set.seed(1)
  if (env_name == "tictactoe") {
    # env on the CPU in every configuration: the rl_gpu_env() default (see its docs)
    env <- rl_gpu_env("tictactoe", n = n_env, opponent = "self", device = "cpu")
    algo <- rl_ppo(n_steps = 9, minibatch = 2048, hidden = 64)
  } else {
    env <- rl_vec_env("pendulum", n = n_env, seed = 1)
    algo <- rl_ppo(n_steps = 128, epochs = 10, minibatch = 256, lr = 1e-3,
                   gamma = 0.9, lambda = 0.95, ent_coef = 0, hidden = 64)
  }
  ag_device(dev)
  # algo$collect/update are called directly, bypassing rl_train(): set its modes
  ag_graph_mode(cfg == "gpu-graph")
  ag_matmul_precision("f32")
  state <- algo$init(env, dev)

  run_iter <- function() {
    t1 <- system.time(col <- algo$collect(state, env))[["elapsed"]]
    t2 <- system.time(upd <- algo$update(col$state))[["elapsed"]]
    state <<- upd$state
    1000 * c(collect = t1, update = t2)
  }
  for (i in 1:2) run_iter()                                   # warm-up

  # 1. clean timing
  clean <- rowMeans(sapply(seq_len(iters), function(i) run_iter()))

  # 2-3. stages, with launch and transfer counters as the probe
  xfer_tot <- function() {
    e <- ns$.ag_xfer$counts
    k <- if (is.null(e)) character(0) else ls(e, all.names = TRUE)
    r <- lapply(k, function(z) e[[z]])
    s <- function(d, f) sum(vapply(r, function(x) if (x$dir == d) as.double(f(x)) else 0, 0))
    c(up = s("up", function(x) x$n), down = s("down", function(x) x$n),
      up_mb = s("up", function(x) x$elems * 4 / 2^20),
      down_mb = s("down", function(x) x$elems * 4 / 2^20))
  }
  probe <- function() c(launch = ns$ag_launch_count()[["total"]], xfer_tot())
  ns$ag_xfer_count(TRUE)
  rltoolsR:::rl_prof(TRUE, probe = probe)
  staged <- rowMeans(sapply(seq_len(iters), function(i) run_iter()))
  pr <- rltoolsR:::rl_prof_get()
  rltoolsR:::rl_prof(FALSE)
  ns$ag_xfer_count(FALSE)
  stage_ms <- pr$ms / iters
  stage_cnt <- if (length(pr$cnt)) t(sapply(pr$cnt, function(v) v / iters)) else NULL

  # 4. inside ggmlR (GPU only; separate pass, its own overhead)
  fwd <- bwd <- NULL
  if (dev == "gpu") {
    ns$ag_forward_profile(TRUE); ns$ag_forward_profile_reset()
    ns$ag_backward_profile(TRUE)
    for (i in seq_len(iters)) run_iter()
    if (!is.null(ns$.ag_fwd$totals))
      fwd <- list(ms = ns$.ag_fwd$totals / iters, n_ops = ns$.ag_fwd$n / iters)
    if (!is.null(ns$.ag_bwd$prof_totals))
      bwd <- list(ms = ns$.ag_bwd$prof_totals / iters, n = ns$.ag_bwd$prof_n / iters)
    ns$ag_forward_profile(FALSE); ns$ag_backward_profile(FALSE)
  }

  saveRDS(list(cfg = cfg, clean = clean, staged = staged, stage_ms = stage_ms,
               stage_cnt = stage_cnt, fwd = fwd, bwd = bwd), out_file)
  quit(save = "no")
}

# ---------------------------------------------------------------- driver ----
env_name <- if (length(args) >= 1) args[1] else "tictactoe"
n_env <- if (length(args) >= 2) as.integer(args[2]) else 256L
iters <- if (length(args) >= 3) as.integer(args[3]) else 10L
stopifnot(env_name %in% c("tictactoe", "pendulum"))

self <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
rscript <- file.path(R.home("bin"), "Rscript")
has_gpu <- tryCatch(ggmlR::ggml_vulkan_available() && ggmlR::ggml_vulkan_device_count() >= 1L,
                    error = function(e) FALSE)
cfgs <- if (has_gpu) c("cpu", "gpu-graph", "gpu-eager") else "cpu"

res <- list()
for (cfg in cfgs) {
  f <- file.path("/tmp", sprintf("profile_ppo_stages_%s_%s.rds", env_name, cfg))
  unlink(f)
  cat(sprintf("running %s ...\n", cfg))
  st <- system2(rscript, c(shQuote(self), "--worker", cfg, env_name, n_env, iters, f))
  if (st != 0 || !file.exists(f)) { cat(sprintf("  %s FAILED (status %d)\n", cfg, st)); next }
  res[[cfg]] <- readRDS(f)
}
cfgs <- names(res)

cat(sprintf("\n==== %s, n_env = %d, %d iterations; ms per iteration ====\n",
            env_name, n_env, iters))

# 1. clean totals
tot <- sapply(res, function(r) c(r$clean, total = sum(r$clean)))
cat("\n-- 1. iteration time, profiling off --\n")
print(round(tot, 1))

# 2. stages
stages <- unique(unlist(lapply(res, function(r) names(r$stage_ms))))
ord <- c(grep("^collect", stages, value = TRUE), grep("^update", stages, value = TRUE))
ms <- sapply(res, function(r) { v <- r$stage_ms[ord]; v[is.na(v)] <- 0; v })
ms <- matrix(ms, length(ord), dimnames = list(ord, cfgs))
staged_tot <- sapply(res, function(r) sum(r$staged))
tab <- rbind(ms, `(unmarked)` = staged_tot - colSums(ms), `TOTAL (staged run)` = staged_tot)
pct <- sweep(tab, 2, staged_tot, "/") * 100
cat("\n-- 2. time per stage, ms (share of the staged run) --\n")
out <- matrix(sprintf("%8.1f (%4.1f%%)", tab, pct), nrow(tab),
              dimnames = dimnames(tab))
if ("cpu" %in% cfgs) for (g in setdiff(cfgs, "cpu"))
  out <- cbind(out, setNames(sprintf("%6.1fx", tab[, g] / pmax(tab[, "cpu"], 1e-3)),
                             rownames(tab)))
colnames(out) <- c(cfgs, if ("cpu" %in% cfgs) paste0(setdiff(cfgs, "cpu"), "/cpu"))
print(noquote(out))

# 3. launches and transfers per stage
for (cfg in setdiff(cfgs, "cpu")) {
  sc <- res[[cfg]]$stage_cnt
  if (is.null(sc)) next
  sc <- sc[intersect(ord, rownames(sc)), , drop = FALSE]
  sc <- rbind(sc, TOTAL = colSums(sc))
  cat(sprintf("\n-- 3. %s: GPU launches and transfers per stage --\n", cfg))
  print(round(sc, 1))
  per_launch <- tab[rownames(sc)[-nrow(sc)], cfg] / pmax(sc[-nrow(sc), "launch"], 1)
  cat("   ms per launch, stages with launches:\n")
  print(round(per_launch[sc[-nrow(sc), "launch"] > 0], 2))
}

# 4. inside ggmlR
for (cfg in setdiff(cfgs, "cpu")) {
  r <- res[[cfg]]
  cat(sprintf("\n-- 4. %s: inside ggmlR (separate profiled pass) --\n", cfg))
  if (!is.null(r$fwd)) {
    m <- sort(r$fwd$ms, decreasing = TRUE)
    cat(sprintf("eager forward ops: %.0f per iteration, %.1f ms, %.3f ms per op\n",
                r$fwd$n_ops, sum(m), sum(m) / max(r$fwd$n_ops, 1)))
    print(round(cbind(ms = m, pct = 100 * m / sum(m)), 1))
  } else cat("eager forward ops: none recorded\n")
  if (!is.null(r$bwd)) {
    m <- sort(r$bwd$ms, decreasing = TRUE)
    cat(sprintf("graph backward: %.0f per iteration, %.1f ms\n", r$bwd$n, sum(m)))
    print(round(cbind(ms = m, pct = 100 * m / sum(m)), 1))
  } else cat("graph backward: none recorded\n")
}
