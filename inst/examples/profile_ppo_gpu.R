# Profile one PPO iteration on tic-tac-toe: where the time goes and how much
# data crosses host <-> device.
#
#   Rscript inst/examples/profile_ppo_gpu.R [gpu|cpu] [n_env] [iterations]
#
# Prints:
#   1. time of collect and update per iteration;
#   2. Rprof top functions (self time) for collect and update;
#   3. counts of ggmlR host <-> device transfers and GPU op launches per
#      iteration (via trace() on ggmlR internals; separate pass, not timed);
#   4. rl_act() time per call on CPU and GPU for batch 1 and batch n_env.
# Results are saved to /tmp/profile_ppo_<device>.rds.

library(rltoolsR)

args <- commandArgs(trailingOnly = TRUE)
dev <- if (length(args) >= 1) args[1] else "gpu"
n_env <- if (length(args) >= 2) as.integer(args[2]) else 256L
iters <- if (length(args) >= 3) as.integer(args[3]) else 10L

set.seed(1)
env <- rl_gpu_env("tictactoe", n = n_env, opponent = "self", device = dev)
algo <- rl_ppo(n_steps = 9, minibatch = 2048, hidden = 64)
prev <- ag_device(dev)
# same GPU modes as rl_train() (R/train.R): calling algo$collect/update
# directly bypasses rl_train, so without these the profile measures eager mode
prev_graph <- ag_graph_mode(TRUE)
prev_prec <- ag_matmul_precision("f32")
state <- algo$init(env, dev)

run_iter <- function(state) {
  t1 <- system.time(col <- algo$collect(state, env))[["elapsed"]]
  t2 <- system.time(upd <- algo$update(col$state))[["elapsed"]]
  list(state = upd$state, t = c(collect = t1, update = t2))
}

# ---- warm-up ----
for (i in 1:2) state <- run_iter(state)$state

# ---- 1. timing ----
tm <- matrix(0, iters, 2, dimnames = list(NULL, c("collect", "update")))
for (i in seq_len(iters)) {
  r <- run_iter(state)
  state <- r$state
  tm[i, ] <- r$t
}
cat(sprintf("\n== %s, n_env = %d, %d iterations ==\n", dev, n_env, iters))
cat(sprintf("per iteration: collect %.1f ms, update %.1f ms (collect %.0f%%)\n",
            1000 * mean(tm[, 1]), 1000 * mean(tm[, 2]),
            100 * sum(tm[, 1]) / sum(tm)))

# ---- 2. Rprof ----
prof_phase <- function(phase) {
  f <- file.path("/tmp", sprintf("ppo_%s_%s.Rprof", phase, dev))
  unlink(f)
  for (i in seq_len(iters)) {
    col <- if (phase == "collect") {
      Rprof(f, interval = 0.01, append = TRUE)
      x <- algo$collect(state, env)
      Rprof(NULL)
      x
    } else algo$collect(state, env)
    upd <- if (phase == "update") {
      Rprof(f, interval = 0.01, append = TRUE)
      x <- algo$update(col$state)
      Rprof(NULL)
      x
    } else algo$update(col$state)
    state <<- upd$state
  }
  s <- summaryRprof(f)$by.self
  head(s[order(-s$self.time), c("self.time", "self.pct")], 15)
}
p_collect <- prof_phase("collect")
p_update <- prof_phase("update")
cat("\n-- Rprof collect (self time) --\n"); print(p_collect)
cat("\n-- Rprof update (self time) --\n"); print(p_update)

# ---- 3. transfer counts ----
ns <- asNamespace("ggmlR")
traced <- intersect(c(".ag_gpu_to_r", ".ag_r_to_gpu", ".ag_run_op"), ls(ns, all.names = TRUE))
.prof_cnt <- new.env()
for (f in traced) {
  assign(f, 0L, envir = .prof_cnt)
  trace(f, where = ns, print = FALSE,
        tracer = substitute(assign(F, get(F, envir = E) + 1L, envir = E),
                            list(F = f, E = .prof_cnt)))
}
count <- function(expr) {
  for (f in traced) assign(f, 0L, envir = .prof_cnt)
  ns$ag_launch_count(reset = TRUE)
  force(expr)
  # launch = real GPU submissions (gpu_op counts .ag_run_op calls, which graph
  # mode only records)
  c(vapply(traced, function(f) get(f, envir = .prof_cnt), integer(1)),
    launch = ns$ag_launch_count()[["total"]])
}
col <- NULL
c_collect <- count(col <- algo$collect(state, env))
c_update <- count(upd <- algo$update(col$state))
launch_sites <- ns$ag_launch_count()   # still holds the update pass, by call site
state <- upd$state
for (f in traced) untrace(f, where = ns)
cnt <- rbind(collect = c_collect, update = c_update)
colnames(cnt) <- c(.ag_gpu_to_r = "download", .ag_r_to_gpu = "upload",
                   .ag_run_op = "gpu_op", launch = "launch")[colnames(cnt)]
cat("\n-- ggmlR calls per iteration --\n"); print(cnt)
cat("\n-- update launches by call site --\n"); print(launch_sites)

# ---- 3b. which update ops miss graph mode ----
# One update pass with .ag_run_op traced: per call, the two callers above it,
# scope, whether it writes into `out`, `resident`, and whether it was deferred
# (same rule as .ag_run_op: resident && .ag_defer_ok(scope, out)). Not timed.
.run_op_log <- new.env()
.run_op_log$keys <- character(0)
trace(".ag_run_op", where = ns, print = FALSE, tracer = quote(local({
  cs <- sys.calls()
  hit <- which(vapply(cs, function(cl) identical(cl[[1]], as.name(".ag_run_op")),
                      logical(1)))
  k <- if (length(hit)) max(hit) else 0L
  who <- function(j) if (j >= 1L) deparse(cs[[j]][[1]])[1] else "-"
  deferred <- isTRUE(resident) && .ag_defer_ok(scope[1], out)
  key <- paste(who(k - 2L), who(k - 1L), scope[1],
               if (is.null(out)) "-" else "out", resident,
               if (deferred) "deferred" else "PER-OP", sep = " | ")
  .GlobalEnv$.run_op_log$keys <- c(.GlobalEnv$.run_op_log$keys, key)
})))
col <- algo$collect(state, env)
upd <- algo$update(col$state)
state <- upd$state
untrace(".ag_run_op", where = ns)
run_op_tab <- sort(table(.run_op_log$keys), decreasing = TRUE)
cat("\n-- update .ag_run_op calls: caller2 | caller1 | scope | out | resident | path --\n")
print(as.data.frame(run_op_tab, responseName = "n"), right = FALSE, row.names = FALSE)

# ---- 4. rl_act CPU vs GPU ----
ag_graph_mode(prev_graph)
ag_matmul_precision(prev_prec)
ag_device(prev)
model <- algo$model(state)
f <- tempfile(fileext = ".rds")
rl_save_model(model, f)
obs_of <- function(n) matrix(sample(c(-1, 0, 1), 9 * n, replace = TRUE), n, 9)
bench <- function(m, n, reps) {
  obs <- obs_of(n)
  rl_act(m, obs)
  1000 * system.time(for (i in seq_len(reps)) rl_act(m, obs))[["elapsed"]] / reps
}
devs <- if (dev == "gpu") c("cpu", "gpu") else "cpu"
act <- sapply(devs, function(d) {
  m <- rl_load_model(f, device = d)
  c(batch_1 = bench(m, 1L, 200L), batch_n = bench(m, n_env, 50L))
})
cat(sprintf("\n-- rl_act, ms per call (batch_n = %d) --\n", n_env)); print(round(act, 3))

out <- file.path("/tmp", sprintf("profile_ppo_%s.rds", dev))
saveRDS(list(device = dev, n_env = n_env, iters = iters, time = tm,
             rprof_collect = p_collect, rprof_update = p_update,
             calls = cnt, launch_sites = launch_sites, run_op_tab = run_op_tab, rl_act_ms = act), out)
cat("\nsaved:", out, "\n")
