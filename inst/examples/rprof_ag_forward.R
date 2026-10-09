# Where do the ~0.25 ms per ag_* op go in graph mode, when the GPU is idle?
#
#   Rscript inst/examples/rprof_ag_forward.R [n_env] [iterations] [cpu|gpu]
#
# profile_ppo_stages.R measured, on tic-tac-toe (n_env 256, gpu-graph):
# "update: forward" 91 ms per iteration with 0 launches, of which ggmlR's own
# .ag_run_op stages (ctx + create) are ~40 ms; the rest is unattributed.
# Here: Rprof over whole PPO iterations in the rl_train() modes, then only the
# samples taken INSIDE the forward stages are kept:
#   update forward  -- stack contains with_grad_tape (the loss is built in it)
#   collect forward -- stack contains the policy forward called from collect,
#                      outside as.matrix() (materialization = launch, not record)
# For each: self time (innermost function) and inclusive time per function.
# Raw file: /tmp/rprof_ag_forward_<device>.out

args <- commandArgs(trailingOnly = TRUE)
n_env <- if (length(args) >= 1) as.integer(args[1]) else 256L
iters <- if (length(args) >= 2) as.integer(args[2]) else 20L
dev <- if (length(args) >= 3) args[3] else "gpu"

suppressPackageStartupMessages({ library(rltoolsR); library(ggmlR) })
set.seed(1)
env <- rl_gpu_env("tictactoe", n = n_env, opponent = "self", device = dev)
algo <- rl_ppo(n_steps = 9, minibatch = 2048, hidden = 64)
ag_device(dev)
ag_graph_mode(TRUE)
ag_matmul_precision("f32")
state <- algo$init(env, dev)
run_iter <- function() {
  col <- algo$collect(state, env)
  state <<- algo$update(col$state)$state
}
for (i in 1:2) run_iter()                                      # warm-up

interval <- 0.001
f <- file.path("/tmp", sprintf("rprof_ag_forward_%s.out", dev))
unlink(f)
t0 <- proc.time()[["elapsed"]]
Rprof(f, interval = interval)
for (i in seq_len(iters)) run_iter()
Rprof(NULL)
wall <- 1000 * (proc.time()[["elapsed"]] - t0) / iters

# Each sample line is a call stack, innermost first, names quoted.
lines <- readLines(f)
# Rprof may clamp the interval (10 ms on Linux): take the real one from the header
interval <- as.numeric(sub(".*sample.interval=([0-9]+).*", "\\1", lines[1])) / 1e6
lines <- lines[-1]
lines <- lines[!grepl("^sample.interval", lines)]
stacks <- lapply(regmatches(lines, gregexpr('"[^"]*"', lines)),
                 function(s) gsub('"', "", s))
ms_per_sample <- 1000 * interval
per_iter <- function(n) n * ms_per_sample / iters

in_update_fwd <- vapply(stacks, function(s) "with_grad_tape" %in% s, logical(1))
in_collect_fwd <- vapply(stacks, function(s) {
  any(grepl("collect", s)) && any(s %in% c("forward", "net$forward")) &&
    !("as.matrix" %in% s) && !("with_grad_tape" %in% s)
}, logical(1))

report <- function(sel, title, top = 25L) {
  st <- stacks[sel]
  cat(sprintf("\n==== %s: %.1f ms per iteration (%d samples) ====\n",
              title, per_iter(length(st)), length(st)))
  if (!length(st)) return(invisible())
  self <- sort(table(vapply(st, `[`, "", 1L)), decreasing = TRUE)
  incl <- sort(table(unlist(lapply(st, unique))), decreasing = TRUE)
  fmt <- function(tb) {
    tb <- head(tb, top)
    data.frame(fun = names(tb), ms_iter = round(per_iter(as.numeric(tb)), 2),
               pct = round(100 * as.numeric(tb) / length(st), 1))
  }
  cat("-- self time (innermost function) --\n")
  print(fmt(self), row.names = FALSE)
  cat("-- inclusive time --\n")
  print(fmt(incl), row.names = FALSE)
  # ag_* entry points: inclusive time of each public op
  ag <- incl[grepl("^ag_", names(incl))]
  if (length(ag)) {
    cat("-- inclusive time of ag_* functions --\n")
    print(fmt(ag), row.names = FALSE)
  }
}

cat(sprintf("\n%s, n_env = %d, %d iterations, graph mode; wall %.1f ms per iteration (under Rprof)\n",
            dev, n_env, iters, wall))
report(in_update_fwd, "update forward (inside with_grad_tape)")
report(in_collect_fwd, "collect forward (policy forward, before materialization)")
cat("\nraw:", f, "\n")
