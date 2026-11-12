# One iteration of rl_cube_q() on the GPU with host walks and with resident
# walks (R/resident_cube.R): the collection (walks + labels) and the update
# (one Adam step), milliseconds, median over `iters` iterations.
#
#   Rscript inst/examples/bench_cube_walks.R [iters] [size] [n_moves]
#
# host:     rl_cube_walks() -- one graph launch per move from R, the end
#           states downloaded and uploaded again as the input of the step
# resident: all moves queued in one go, the step reads the states on the device
#
# Compare timings only at a fixed GPU clock:
#   echo high | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level
#
# Raw results: /tmp/bench_cube_walks.rds

args <- commandArgs(trailingOnly = TRUE)
iters <- if (length(args) >= 1) as.integer(args[1]) else 30L
size <- if (length(args) >= 2) as.integer(args[2]) else 3L
n_moves <- if (length(args) >= 3) as.integer(args[3]) else 20L

suppressPackageStartupMessages({ library(rltoolsR); library(ggmlR) })
if (!(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L)) stop("no Vulkan device")

ag_device("gpu")
ag_graph_mode(TRUE)            # rl_train() modes
ag_matmul_precision("f32")

ms <- function(t0) 1e3 * as.numeric(difftime(Sys.time(), t0, units = "secs"))

run <- function(n, walks) {
  set.seed(1)
  env <- rl_gpu_env("cube", n = n, size = size, device = "gpu")
  algo <- rl_cube_q(n_moves = n_moves, walks = walks)
  state <- algo$init(env, "gpu")
  for (i in 1:5) state <- algo$update(algo$collect(state, env)$state)$state
  t <- vapply(seq_len(iters), function(j) {
    t0 <- Sys.time()
    state <<- algo$collect(state, env)$state
    tc <- ms(t0)
    t0 <- Sys.time()
    state <<- algo$update(state)$state
    c(collect = tc, update = ms(t0))
  }, c(collect = 0, update = 0))
  apply(t, 1, stats::median)
}

res <- NULL
for (n in c(256L, 4096L)) for (walks in c("host", "resident")) {
  r <- run(n, walks)
  res <- rbind(res, data.frame(n = n, walks = walks, collect = r[["collect"]],
                               update = r[["update"]], total = sum(r)))
  gc()
}

cat(sprintf("\nrl_cube_q on the GPU, cube %d, n_moves = %d, ms per iteration (median of %d)\n",
            size, n_moves, iters))
print(format(res, digits = 3), row.names = FALSE)
for (n in unique(res$n)) {
  h <- res$total[res$n == n & res$walks == "host"]
  r <- res$total[res$n == n & res$walks == "resident"]
  cat(sprintf("n = %d: resident / host = %.2f\n", n, r / h))
}

# ---- the walk on the device, by part ---------------------------------------
# plan:  cube_walk_plan() -- the moves of every walk drawn in R (the host
#        path's share; the resident path does not run it)
# given: cube_res_walk()  -- those moves uploaded and made on the device
# drawn: cube_res_sample() -- the resident collection: moves drawn and made on
#        the device (n_moves steps, one more than `given`)
parts <- function(n) {
  set.seed(1)
  env <- rl_gpu_env("cube", n = n, size = size, device = "gpu")
  st <- env$st
  rs <- rltoolsR:::cube_res_new(st, n_moves, "gpu")
  on.exit(rltoolsR:::cube_res_free(rs))
  one <- function() {
    t0 <- Sys.time()
    w <- rltoolsR:::cube_walk_plan(st, n_moves, 2L)
    tp <- ms(t0)
    t0 <- Sys.time()
    rltoolsR:::cube_res_walk(rs, w$acts)
    tw <- ms(t0)
    t0 <- Sys.time()
    rltoolsR:::cube_res_sample(rs)
    c(plan = tp, given = tw, drawn = ms(t0))
  }
  for (i in 1:3) one()
  apply(vapply(seq_len(iters), function(j) one(), c(plan = 0, given = 0, drawn = 0)),
        1, stats::median)
}
part_ms <- t(vapply(c(256L, 4096L), parts, c(plan = 0, given = 0, drawn = 0)))
rownames(part_ms) <- paste("n =", c(256L, 4096L))
cat("\nthe walk by part, ms (median)\n")
print(round(part_ms, 2))

out_file <- "/tmp/bench_cube_walks.rds"
saveRDS(list(res = res, part_ms = part_ms, iters = iters, size = size, n_moves = n_moves),
        out_file)
cat("saved", out_file, "\n")
