# Where does one captured PPO update step (ggmlR::ag_capture_step) spend its
# time: ggmlR's opt-in replay timer (GGMLR_AG_CAPTURE_PROF=1), stage by stage,
# against the whole update divided by the number of steps -- the difference is
# rltoolsR's own per-step work (advantage normalization, one-hot, mask, calls).
#
#   Rscript inst/examples/profile_capture_replay.R [tictactoe|pendulum] [n_env] [iterations] [--vk-perf | --nodes]
#
# --nodes: list the nodes of the recorded step graphs (op counts, which ops feed
# the common ones); full tables in /tmp/nodes_<env>_<recording>_<graph>.csv.
#
# --vk-perf: instead, time every op of the step graphs ON THE DEVICE with
# GGML_VK_PERF_LOGGER (GPU timestamps per node). The update is repeated on ONE
# collected rollout, so the window holds only step graphs; the logger prints a
# summary after every graph, which is captured and summed per op. Graph 1
# (forward+backward) and graph 2 (Adam: the blocks with OPT_STEP_ADAMW) are
# reported apart. The logger waits for the device after every graph, so wall
# times in this mode are not the real ones -- only the per-op device times are.
#
# GPU only; settings as in bench_cpu_gpu.R. The recording (first step of each
# minibatch size) is not timed.

args <- commandArgs(trailingOnly = TRUE)
vk_perf <- "--vk-perf" %in% args
nodes   <- "--nodes" %in% args
args <- setdiff(args, c("--vk-perf", "--nodes"))
# Read when the Vulkan instance is created: set before ggmlR starts it.
if (vk_perf) Sys.setenv(GGML_VK_PERF_LOGGER = "1") else Sys.setenv(GGMLR_AG_CAPTURE_PROF = "1")
suppressPackageStartupMessages({ library(rltoolsR); library(ggmlR) })
env_name <- if (length(args) >= 1) args[1] else "pendulum"
n_env <- if (length(args) >= 2) as.integer(args[2]) else if (env_name == "pendulum") 16L else 256L
iters <- if (length(args) >= 3) as.integer(args[3]) else 5L
stopifnot(ggml_vulkan_available(), ggml_vulkan_device_count() >= 1L)

set.seed(1)
if (env_name == "tictactoe") {
  env <- rl_gpu_env("tictactoe", n = n_env, opponent = "self")
  algo <- rl_ppo(n_steps = 9, minibatch = 2048, hidden = 64)
} else {
  env <- rl_vec_env("pendulum", n = n_env, seed = 1)
  algo <- rl_ppo(n_steps = 128, epochs = 10, minibatch = 256, lr = 1e-3, gamma = 0.9,
                 lambda = 0.95, ent_coef = 0, hidden = 64)
}
ag_device("gpu")
ag_graph_mode(TRUE)
ag_matmul_precision("f32")
state <- algo$init(env, "gpu")
# warm-up, records (with --vk-perf the logger prints after every graph: kept quiet)
quiet <- function(expr) if (vk_perf) invisible(utils::capture.output(expr)) else expr
quiet(for (i in 1:2) state <- algo$update(algo$collect(state, env)$state)$state)

if (nodes) {
  # What the recorded step graphs are made of: every node with its sources.
  # Full tables in /tmp; printed: op counts, and for the common ops which ops
  # feed them (SCALE fed by SCALE = a foldable chain, REPEAT feeding MUL/ADD =
  # a broadcast ggml could do itself, CONT after TRANSPOSE, ...).
  for (k in ls(state$captures)) {
    cap <- attr(state$captures[[k]], "captures")$step
    if (is.null(cap)) next
    for (g in c("graph", "graph_opt")) {
      d <- ggmlR:::.ag_graph_dump(cap[[g]])
      csv <- file.path("/tmp", sprintf("nodes_%s_%s_%s.csv", env_name, gsub("[^0-9A-Za-z]+", "_", k), g))
      utils::write.csv(d, csv, row.names = FALSE)
      cat(sprintf("\n==== recording %s, %s: %d nodes (full table: %s) ====\n", k, g, nrow(d), csv))
      print(sort(table(d$op), decreasing = TRUE))
      if (g == "graph") {
        d$feeds <- paste0(d$src0_op, ifelse(nzchar(d$src1_op), paste0(" , ", d$src1_op), ""))
        for (o in names(sort(table(d$op), decreasing = TRUE))[1:6]) {
          cat(sprintf("\n-- %s fed by (src0 , src1):\n", o))
          print(sort(table(d$feeds[d$op == o]), decreasing = TRUE))
        }
      }
    }
  }
  quit(save = "no")
}

if (vk_perf) {
  quiet(col <- algo$collect(state, env))
  # Each logger block: "<op>: <count> x <mean> us = <total> us[ (GFLOPS)]".
  out <- utils::capture.output(for (i in seq_len(iters)) algo$update(col$state))
  starts <- grep("^Vulkan Timings:", out)
  ends <- c(starts[-1L] - 1L, length(out))
  rx <- "^(.+): ([0-9]+) x ([0-9.]+) us = ([0-9.]+) us"
  per <- list(g1 = list(), g2 = list())
  nblk <- c(g1 = 0, g2 = 0)
  for (b in seq_along(starts)) {
    ln <- out[starts[b]:ends[b]]
    ln <- ln[grepl(rx, ln)]
    if (!length(ln)) next
    op  <- sub(rx, "\\1", ln)
    cnt <- as.numeric(sub(rx, "\\2", ln))
    tot <- as.numeric(sub(rx, "\\4", ln))
    g <- if (any(grepl("OPT_STEP_ADAMW", op))) "g2" else "g1"
    nblk[g] <- nblk[g] + 1
    for (k in seq_along(op)) {
      prev <- per[[g]][[op[k]]]
      if (is.null(prev)) prev <- c(n = 0, us = 0)
      per[[g]][[op[k]]] <- prev + c(cnt[k], tot[k])
    }
  }
  if (!nblk[["g1"]]) stop("no logger output captured: is GGML_VK_PERF_LOGGER honoured by this build?")
  cat(sprintf("\n==== %s, n_env = %d: device time per op, per step (%d steps) ====\n",
              env_name, n_env, as.integer(nblk[["g1"]])))
  for (g in c("g1", "g2")) {
    if (!length(per[[g]])) next
    m <- do.call(rbind, per[[g]])
    df <- data.frame(op = rownames(m), nodes_per_step = m[, "n"] / nblk[[g]],
                     us_per_node = m[, "us"] / m[, "n"], us_per_step = m[, "us"] / nblk[[g]])
    bad <- !is.finite(df$us_per_step)
    if (any(bad)) { cat("\nrows that did not parse:\n"); print(df[bad, ], row.names = FALSE) }
    df <- df[!bad, ]
    df <- df[order(-df$us_per_step), ]
    csv <- file.path("/tmp", sprintf("vkperf_%s_%s.csv", env_name, g))
    utils::write.csv(df, csv, row.names = FALSE)
    tot <- sum(df$us_per_step)
    # Compact: the 15 largest, the rest as one row; names cut to 45 characters.
    top <- head(df, 15L)
    if (nrow(df) > 15L) {
      rest <- df[-(1:15), ]
      top <- rbind(top, data.frame(op = sprintf("(other %d ops)", nrow(rest)),
                                   nodes_per_step = sum(rest$nodes_per_step), us_per_node = NA,
                                   us_per_step = sum(rest$us_per_step)))
    }
    top$op <- substr(top$op, 1L, 45L)
    top$share <- sprintf("%.1f%%", 100 * top$us_per_step / tot)
    top[c("nodes_per_step", "us_per_node", "us_per_step")] <-
      lapply(top[c("nodes_per_step", "us_per_node", "us_per_step")], round, 1)
    cat(sprintf("\n-- %s: %.0f nodes, %.3f ms on the device per step (full table: %s) --\n",
                if (g == "g1") "graph 1 (forward+backward)" else "graph 2 (Adam)",
                sum(df$nodes_per_step), tot / 1000, csv))
    cat("   note: each graph's FIRST node also carries the submit-to-start latency\n")
    print(top, row.names = FALSE)
  }
  quit(save = "no")
}
invisible(ggmlR:::.ag_capture_prof(reset = TRUE))

t_upd <- 0
for (i in seq_len(iters)) {
  col <- algo$collect(state, env)
  t_upd <- t_upd + system.time(upd <- algo$update(col$state))[["elapsed"]]
  state <- upd$state
}
pr <- ggmlR:::.ag_capture_prof()
n <- pr[["replays"]]
stages <- pr[setdiff(names(pr), c("total", "replays"))]
per_step <- 1000 * t_upd / n

cat(sprintf("\n==== %s, n_env = %d, %d iterations: %d captured steps ====\n",
            env_name, n_env, iters, n))
cat(sprintf("update per step, whole (system.time): %.3f ms\n\n", per_step))
tab <- data.frame(stage = c(names(stages), "ggmlR total", "rltoolsR + rest"),
                  ms = round(c(stages, pr[["total"]], per_step - pr[["total"]]), 3))
tab$share <- sprintf("%.1f%%", 100 * tab$ms / per_step)
print(tab, row.names = FALSE)

# Graph stages are split into recording+submit (CPU side) and the wait for the
# device. Recording cost per node tells whether reusing the recorded command
# buffer would pay; the wait is the device's own time (plus queue latency).
nd <- attr(pr, "nodes")
if (length(nd)) {
  cat("\nnodes per graph and CPU recording cost per node:\n")
  for (g in names(nd)) {
    rec <- pr[[paste(g, "record+submit")]]
    wt  <- pr[[paste(g, "wait")]]
    cat(sprintf("  %-28s %4d nodes | record+submit %.3f ms (%.1f us/node) | wait %.3f ms\n",
                g, as.integer(nd[[g]]), rec, 1000 * rec / nd[[g]], wt))
  }
}
