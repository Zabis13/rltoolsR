# CPU or GPU for acting: where does the GPU start to pay?
#
#   Rscript inst/examples/bench_act_crossover.R [reps]
#
# Fix the GPU clock first and restore it after:
#   echo high | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level
#   echo auto | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level
#
# 1. rl_act() per call, greedy, tic-tac-toe observations (9 features, 9
#    actions), batch 1, 2, 4, ..., 1024 (a factor of 2 between points), hidden
#    64 and 256. CPU = the host path (policy_act_host), GPU = one captured
#    launch.
# 2. PPO collect per iteration (tic-tac-toe self-play, 9 steps), n_env 16 and
#    256, hidden 64 and 256: the whole loop -- environment, acting, transfers
#    -- since the crossover of one call and of the loop can differ. Note that
#    collect does not use the host path: on the CPU it runs the ag_* forward.
#
# Each device runs in its own R process (the ag_* device state is global),
# with OPENBLAS_NUM_THREADS=1 / OMP_NUM_THREADS=1 so the CPU numbers do not
# depend on the machine's BLAS threading. Every point: warm-up calls first
# (dropped), then `reps` timed chunks; reported median, q25 and q75 of the
# per-call (per-iteration) time.
# Raw results: /tmp/bench_act_crossover_<device>.rds

args <- commandArgs(trailingOnly = TRUE)
batches <- 2L^(0:10)
hiddens <- c(64L, 256L)
n_envs <- c(16L, 256L)

# ---------------------------------------------------------------- worker ----
if (length(args) && args[1] == "--worker") {
  dev <- args[2]; reps <- as.integer(args[3]); out_file <- args[4]
  suppressPackageStartupMessages({ library(rltoolsR); library(ggmlR) })
  ag_device(dev)
  ag_graph_mode(TRUE)
  ag_matmul_precision("f32")
  now <- function() as.numeric(Sys.time()) * 1000
  # per-call ms of f(), in `reps` chunks of `k` calls after `warm` calls
  chunks <- function(f, k, warm) {
    for (i in seq_len(warm)) f()
    vapply(seq_len(reps), function(r) {
      t0 <- now(); for (i in seq_len(k)) f(); (now() - t0) / k
    }, 0)
  }
  rows <- list()
  for (h in hiddens) {
    set.seed(1)
    env <- rl_gpu_env("tictactoe", n = 1L, opponent = "self", device = "cpu")
    algo <- rl_ppo(n_steps = 9, minibatch = 576, hidden = h)
    model <- algo$model(algo$init(env, dev))
    for (b in batches) {
      obs <- matrix(sample(c(-1, 0, 1), 9 * b, TRUE), b, 9)
      k <- max(5L, as.integer(2000 / b))          # ~ same work per chunk
      tm <- chunks(function() rl_act(model, obs), k, warm = 3L * k)
      rows[[length(rows) + 1L]] <- data.frame(part = "rl_act", device = dev, hidden = h,
                                              size = b, ms = tm)
    }
    for (n in n_envs) {
      set.seed(2)
      env <- rl_gpu_env("tictactoe", n = n, opponent = "self", device = "cpu")
      algo <- rl_ppo(n_steps = 9, minibatch = 9L * n, hidden = h)
      state <- algo$init(env, dev)
      tm <- chunks(function() state <<- algo$collect(state, env)$state, 3L, warm = 5L)
      rows[[length(rows) + 1L]] <- data.frame(part = "collect", device = dev, hidden = h,
                                              size = n, ms = tm)
    }
  }
  saveRDS(list(rows = do.call(rbind, rows),
               blas = c(extSoftVersion()[["BLAS"]], La_library()),
               threads = Sys.getenv(c("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"))),
          out_file)
  quit(save = "no")
}

# ------------------------------------------------------------------ main ----
reps <- if (length(args) >= 1) as.integer(args[1]) else 15L
if (!ggmlR::ggml_vulkan_available()) stop("no Vulkan device: nothing to compare")
self <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
rscript <- file.path(R.home("bin"), "Rscript")
res <- list()
for (d in c("cpu", "gpu")) {
  f <- sprintf("/tmp/bench_act_crossover_%s.rds", d)
  cat("running", d, "...\n")
  st <- system2(rscript, c(shQuote(self), "--worker", d, reps, f),
                env = c("OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1"))
  if (st != 0) stop("worker ", d, " failed")
  res[[d]] <- readRDS(f)
}
cat("BLAS:", res$cpu$blas, "; threads:", res$cpu$threads, "\n")

all <- rbind(res$cpu$rows, res$gpu$rows)
q <- aggregate(ms ~ part + device + hidden + size, all,
               function(v) stats::quantile(v, c(0.25, 0.5, 0.75)))
q <- data.frame(q[1:4], q25 = q$ms[, 1], med = q$ms[, 2], q75 = q$ms[, 3])
for (p in c("rl_act", "collect")) for (h in hiddens) {
  a <- q[q$part == p & q$hidden == h & q$device == "cpu", ]
  g <- q[q$part == p & q$hidden == h & q$device == "gpu", ]
  m <- merge(a, g, by = "size", suffixes = c("_cpu", "_gpu"))
  tab <- data.frame(size = m$size,
                    cpu = sprintf("%.3f [%.3f-%.3f]", m$med_cpu, m$q25_cpu, m$q75_cpu),
                    gpu = sprintf("%.3f [%.3f-%.3f]", m$med_gpu, m$q25_gpu, m$q75_gpu),
                    gpu_vs_cpu = sprintf("%.2fx", m$med_gpu / m$med_cpu))
  cat(sprintf("\n== %s, hidden %d: ms per %s, median [q25-q75]; %s ==\n", p, h,
              if (p == "rl_act") "call" else "iteration",
              if (p == "rl_act") "size = batch" else "size = n_env"))
  print(tab, row.names = FALSE)
  win <- m$size[m$med_gpu < m$med_cpu]
  cat(if (length(win)) sprintf("GPU faster from size %d\n", min(win)) else "GPU never faster\n")
}
