# CartPole-v1, PPO with the resident rollout (R/resident_cartpole.R) against
# the host rollout: same settings, same seed, the return of the greedy policy
# and the training time of each.
#
#   Rscript inst/examples/check_learning_cartpole_resident.R [steps=1e5] [n=8] [seed=1]
#
# GPU only: rl_gpu_env("cartpole", device = "gpu") and a GPU network. Both runs
# have normalize_obs = FALSE: with it the resident rollout does not apply, and
# the two runs would not compare. rollout = "resident" stops if the resident
# path cannot be taken, so the first run is resident for sure.
# Settings of rl-baselines3-zoo for CartPole-v1 (as train_cartpole.R).
# Solved = mean return >= 475 over 100 episodes. The rollouts differ in the
# reset RNG, so the returns agree only within the seed-to-seed spread.
# Exits with status 1 when the resident run is not solved.
suppressMessages(devtools::load_all(quiet = TRUE))

opt <- list(steps = 1e5, n = 8, seed = 1)
for (a in commandArgs(TRUE)) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (!kv[1] %in% names(opt)) stop("unknown key: ", kv[1])
  opt[[kv[1]]] <- as.numeric(kv[2])
}
str(opt)
if (!(ggml_vulkan_available() && ggml_vulkan_device_count() > 0L)) stop("needs a Vulkan device")

n_steps <- 32L
iters <- max(1L, as.integer(round(opt$steps / (opt$n * n_steps))))

run <- function(rollout) {
  set.seed(opt$seed)
  env <- rl_gpu_env("cartpole", n = as.integer(opt$n), device = "gpu")
  t0 <- Sys.time()
  m <- rl_ppo_train(env, iterations = iters, n_steps = n_steps, epochs = 20,
                    minibatch = 256, lr = 1e-3, gamma = 0.98, lambda = 0.8,
                    ent_coef = 0, hidden = 64, normalize_obs = FALSE,
                    device = "gpu", verbose = FALSE, rollout = rollout)
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  ev <- rl_evaluate(m, "cartpole", n_episodes = 100, seed = 2000000L)
  cat(sprintf("%-8s %6.1f s (%.3f s / iteration): return %.1f +- %.1f, length %.1f\n",
              rollout, secs, secs / iters, ev[["return_mean"]], ev[["return_std"]],
              ev[["length_mean"]]))
  list(secs = secs, ret = ev[["return_mean"]])
}

cat(sprintf("%d iterations x %d envs x %d steps\n", iters, as.integer(opt$n), n_steps))
res <- run("resident")
hst <- run("host")
cat(sprintf("resident / host time: %.2f\n", res$secs / hst$secs))

ok <- res$ret >= 475
cat(if (ok) "OK: resident solved (>= 475)\n" else "FAIL: resident not solved (< 475)\n")
if (!ok) quit(status = 1)
