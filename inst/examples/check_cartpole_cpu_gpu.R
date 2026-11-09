# CartPole: the C++ environment (rl_env, float64) against the GPU graph
# (rl_gpu_env, f32) from the same start states with the same random actions.
#
#   Rscript inst/examples/check_cartpole_cpu_gpu.R [device=gpu|cpu] [n=16] [seed=1]
suppressMessages(devtools::load_all(quiet = TRUE))

opt <- list(device = "gpu", n = 16, seed = 1)
for (a in commandArgs(TRUE)) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (!kv[1] %in% names(opt)) stop("unknown key: ", kv[1])
  opt[[kv[1]]] <- if (is.numeric(opt[[kv[1]]])) as.numeric(kv[2]) else kv[2]
}
set.seed(opt$seed)
n <- as.integer(opt$n)

genv <- rl_gpu_env("cartpole", n = n, device = opt$device)
cat("GPU env backend:", ggml_backend_name(genv$st$backend), "\n")
s0 <- rl_reset(genv)
cenv <- lapply(seq_len(n), function(i) {
  e <- rl_env("cartpole", seed = i)
  rl_reset(e)
  # snapshot = step counter (int) + state (4 doubles)
  rl_load_state(e, c(writeBin(0L, raw()), writeBin(as.double(s0[i, ]), raw())))
  e
})
alive <- rep(TRUE, n)
max_diff <- 0
mismatch <- 0L
steps <- 0L
for (t in 1:500) {
  a <- sample(0:1, n, replace = TRUE)
  rg <- rl_step(genv, a)
  for (i in which(alive)) {
    rc <- rl_step(cenv[[i]], a[i])
    steps <- steps + 1L
    og <- if (rg$terminated[i] || rg$truncated[i]) rg$final_obs[i, ] else rg$obs[i, ]
    max_diff <- max(max_diff, abs(og - rc$obs))
    if (rc$terminated != rg$terminated[i]) mismatch <- mismatch + 1L
    if (rc$terminated || rc$truncated || rg$terminated[i]) alive[i] <- FALSE
  }
  if (!any(alive)) break
}
cat(sprintf("C++ vs GPU graph: %d episodes, %d steps, max |obs diff| %.2e, termination mismatches %d -> %s\n",
            n, steps, max_diff, mismatch, if (max_diff < 1e-3 && mismatch == 0L) "OK" else "CHECK"))
