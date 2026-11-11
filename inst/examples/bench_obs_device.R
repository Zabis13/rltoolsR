# PPO collect time by observation path, on one GPU environment (stage 4).
#
#   Rscript inst/examples/bench_obs_device.R [iters] [env]
#
#   iters  timed iterations per point (default 30), after 3 warm-up ones
#   env    "tictactoe" (default) or "cube"
#
# Points, all with the network on the GPU in the modes rl_train() uses:
#   1 host        GPU environment, normalize_obs = TRUE: forced onto the host
#                 path (obs downloaded every step). The base; note it also
#                 adds the normalization to the act graph.
#   2 dev_nocap   GPU environment, capture = FALSE: the environment returns
#                 rl_device_obs, the collection calls as.matrix() every step
#                 (host path). Cost of the object itself: should equal 1
#                 within noise.
#   3 dev_buf     GPU environment, capture = TRUE: ag_buffer_copy_from +
#                 ag_get_rows. 2 vs 3 is the gain of the device buffer.
#   4 cpu_env     reference only: the environment on the CPU (other backend
#                 for the env graph; not part of the path comparison).
# Every point has its own environment and state from the same seed; the
# iterations are interleaved round-robin with a rotating order, so drift hits
# all points alike. Each iteration is collect + update; only collect is timed.
# The column path (state$obs_path) confirms the path each point took.
#
# cube: also the share of steps with a truncation (final_obs downloads on the
# device path), from a separate environment stepped with random moves for as
# many steps as the timed collections (an untrained policy rarely solves, so
# the share is close to what the collection sees).
# Raw results: /tmp/bench_obs_device_<env>.rds

args <- commandArgs(trailingOnly = TRUE)
iters <- if (length(args) >= 1) as.integer(args[1]) else 30L
env_name <- if (length(args) >= 2) args[2] else "tictactoe"
stopifnot(env_name %in% c("tictactoe", "cube"))

suppressPackageStartupMessages({ library(rltoolsR); library(ggmlR) })
if (!(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L)) stop("no Vulkan device")

n_env <- 256L
make_env <- function(dev) {
  if (env_name == "tictactoe") rl_gpu_env("tictactoe", n = n_env, opponent = "self", device = dev)
  else rl_gpu_env("cube", n = n_env, size = 3, scramble = 20, max_steps = 50, device = dev)
}
n_steps <- if (env_name == "tictactoe") 9L else 16L
algo_of <- function(...) rl_ppo(n_steps = n_steps, minibatch = 2048, hidden = 64, ...)

points <- list(
  host      = list(env = "gpu", algo = algo_of(normalize_obs = TRUE, rollout = "host")),
  dev_nocap = list(env = "gpu", algo = algo_of(capture = FALSE, rollout = "host")),
  dev_buf   = list(env = "gpu", algo = algo_of(capture = TRUE, rollout = "host")),
  cpu_env   = list(env = "cpu", algo = algo_of(capture = TRUE, rollout = "host"))
)
# tic-tac-toe self-play only: the whole rollout as one recorded graph per step
if (env_name == "tictactoe")
  points$resident <- list(env = "gpu", algo = algo_of(capture = TRUE, rollout = "resident"))

ag_device("gpu")
ag_graph_mode(TRUE)            # rl_train() modes; algo$collect/update bypass it
ag_matmul_precision("f32")

runs <- lapply(points, function(p) {
  set.seed(1)
  r <- new.env()
  r$env <- make_env(p$env)
  r$algo <- p$algo
  r$state <- p$algo$init(r$env, "gpu")
  r
})

run_iter <- function(r) {
  # Sys.time(): microseconds; system.time() rounds to ~1 ms, coarser than the
  # differences between the paths
  t0 <- Sys.time()
  col <- r$algo$collect(r$state, r$env)
  t1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  r$state <- r$algo$update(col$state)$state
  1000 * t1
}

for (i in 1:3) for (r in runs) run_iter(r)
tm <- matrix(NA_real_, iters, length(runs), dimnames = list(NULL, names(runs)))
for (i in seq_len(iters)) {
  ord <- names(runs)[(seq_along(runs) + i - 2L) %% length(runs) + 1L]
  for (m in ord) tm[i, m] <- run_iter(runs[[m]])
}
path <- vapply(runs, function(r) r$state$obs_path, "")

q <- apply(tm, 2, stats::quantile, c(0.25, 0.5, 0.75))
tab <- data.frame(point = colnames(tm), path = path,
                  median_ms = q[2, ], q25 = q[1, ], q75 = q[3, ],
                  vs_host = q[2, ] / q[2, "host"], row.names = NULL)
cat(sprintf("\n%s, n_env = %d, n_steps = %d, %d iterations per point (collect, ms)\n",
            env_name, n_env, n_steps, iters))
tab[, c("median_ms", "q25", "q75")] <- round(tab[, c("median_ms", "q25", "q75")], 2)
tab$vs_host <- round(tab$vs_host, 3)
print(tab, row.names = FALSE)
cat(sprintf("dev_buf / dev_nocap: %.3f\n", q[2, "dev_buf"] / q[2, "dev_nocap"]))
if (!is.null(runs$resident)) {
  cat(sprintf("resident / dev_buf: %.3f\n", q[2, "resident"] / q[2, "dev_buf"]))
  recs <- attr(runs$resident$state$captures$res_step, "captures")
  for (k in ls(recs))
    cat(sprintf("resident step graph: %d nodes\n", ggml_graph_n_nodes(recs[[k]]$graph)))
}

res <- list(times = tm, table = tab, env = env_name, n_env = n_env, n_steps = n_steps)
if (env_name == "cube") {
  # same seed, same schedule as the collection: a fresh environment stepped
  # with random moves for as many steps as the timed collections took
  set.seed(1)
  env <- make_env("gpu")
  n_total <- (iters + 3L) * n_steps
  any_tr <- frac_tr <- numeric(n_total)
  for (t in seq_len(n_total)) {
    r <- rl_step(env, sample.int(env$action_space$n, n_env, TRUE) - 1L)
    tr <- r$truncated & !r$terminated
    any_tr[t] <- any(tr)
    frac_tr[t] <- mean(tr)
  }
  cat(sprintf("cube truncation: %.1f%% of steps have one (final_obs download), %.2f%% of env-steps\n",
              100 * mean(any_tr), 100 * mean(frac_tr)))
  res$trunc <- c(steps_with = mean(any_tr), env_steps = mean(frac_tr))
}
out_file <- sprintf("/tmp/bench_obs_device_%s.rds", env_name)
saveRDS(res, out_file)
cat("saved", out_file, "\n")
