# What does one captured forward (ggmlR::ag_capture) cost, part by part?
#
#   Rscript inst/examples/bench_capture_replay.R [reps]
#
# policy_forward() on the GPU is one replay of a recorded graph. A replay is:
#   R wrapper   argument checks, shape key, validity check (param pointers)
#   upload      x only (unchanged inputs -- mean, scale -- are not re-sent)
#   compute     one graph launch (submit + wait)
#   download    one transfer: the outputs are joined into one tensor in the graph
# Each part is timed on its own, on the recording policy_forward() made, and
# the R wrapper is what is left of a full call. CPU: the same forward with
# as.matrix() on the outputs, for reference. Tic-tac-toe net (9 -> 64 -> 64,
# no normalization) and Pendulum net (3 -> 64 -> 64, normalized), at several
# batch sizes. Microseconds per call.

args <- commandArgs(trailingOnly = TRUE)
reps <- if (length(args) >= 1) as.integer(args[1]) else 2000L

suppressPackageStartupMessages({ library(rltoolsR); library(ggmlR) })
ns <- asNamespace("ggmlR")
rl <- asNamespace("rltoolsR")
us <- function(expr_fn, n = reps) {
  expr_fn()                                             # warm
  1e6 * system.time(for (i in seq_len(n)) expr_fn())[["elapsed"]] / n
}

nets <- list(
  tictactoe = list(build = function() rl$ppo_net(9L, 9L, 64L), obs = 9L, norm = NULL),
  pendulum  = list(build = function() rl$ppo_box_net(3L, 1L, 64L), obs = 3L,
                   norm = list(mean = c(0.1, -0.2, 0.3), var = c(0.5, 0.6, 2),
                               epsilon = 1e-8, clip_obs = 10))
)
batches <- c(1L, 16L, 256L)

# ---- CPU reference ----
ag_device("cpu")
cpu <- list()
for (nm in names(nets)) {
  s <- nets[[nm]]
  set.seed(1)
  net <- s$build()
  for (b in batches) {
    x <- matrix(rnorm(s$obs * b), s$obs, b)
    cpu[[paste(nm, b)]] <- us(function() lapply(net$forward(rl$obs_input(x, s$norm)), as.matrix))
  }
}

# ---- GPU replay, by part ----
ag_device("gpu")
ag_graph_mode(TRUE)
ag_matmul_precision("f32")
rows <- list()
for (nm in names(nets)) {
  s <- nets[[nm]]
  set.seed(1)
  net <- s$build()
  for (b in batches) {
    # two observation batches, alternated: x changes every call, as in a
    # rollout, while the normalization statistics stay the same
    xs <- list(matrix(rnorm(s$obs * b), s$obs, b), matrix(rnorm(s$obs * b), s$obs, b))
    k <- 0L
    full <- us(function() { k <<- k %% 2L + 1L; rl$policy_forward(net, xs[[k]], s$norm) })
    x <- xs[[1L]]

    # the recording policy_forward() just made for this shape
    caps <- ls(net$captured)
    f <- net$captured[[caps[1]]]
    cache <- attr(f, "captures")
    key <- grep(sprintf("^x:%dx%d[, ]", s$obs, b), ls(cache), value = TRUE)[1]
    cap <- cache[[key]]
    params <- unname(net$parameters())
    inputs <- if (is.null(s$norm)) list(x = x) else
      list(x = x, m = matrix(s$norm$mean), s = rl$obs_scale(s$norm))

    t_valid <- us(function() ns$.ag_capture_valid(cap, params))
    # only x is re-sent per call; unchanged inputs are skipped by the replay
    t_up <- us(function() ns$.ag_xfer_up(cap$in_ptrs[["x"]], as.numeric(x), "bench"))
    t_comp <- us(function() ns$.ag_graph_compute(cap$backend, cap$graph, "bench"))
    # outputs joined into one tensor when the recording could (cap$out_root)
    t_down <- if (!is.null(cap$out_root)) us(function() ns$.ag_xfer_down(cap$out_root, "bench")) else
      us(function() for (p in cap$out_ptrs) ns$.ag_xfer_down(p, "bench"))
    t_down1 <- us(function() ns$.ag_xfer_down(cap$out_ptrs[[1]], "bench"))

    rows[[length(rows) + 1L]] <- data.frame(
      net = nm, batch = b, cpu = round(cpu[[paste(nm, b)]]),
      gpu_full = round(full),
      r_wrapper = round(full - t_up - t_comp - t_down),
      of_which_valid = round(t_valid),
      upload = round(t_up), n_inputs = length(cap$in_ptrs),
      compute = round(t_comp),
      download = round(t_down), n_outputs = length(cap$out_ptrs),
      joined = !is.null(cap$out_root),
      one_down = round(t_down1))
  }
}
ag_device("cpu")

cat(sprintf("\n==== captured forward, microseconds per call (%d reps) ====\n", reps))
cat("r_wrapper = gpu_full - upload - compute - download; of_which_valid is part of it\n")
cat("one_down = one single-output download; joined = outputs read in one transfer\n\n")
print(do.call(rbind, rows), row.names = FALSE)
