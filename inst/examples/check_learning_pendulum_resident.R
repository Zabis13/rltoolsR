# Pendulum-v1, PPO with box actions on the resident rollout
# (R/resident_pendulum.R) against the host rollout: same settings, same seed,
# the greedy return and the training time of each.
#
#   Rscript inst/examples/check_learning_pendulum_resident.R [steps=3e5] [n=16] [seed=1]
#
# GPU only: rl_gpu_env("pendulum_gym", device = "gpu") and a GPU network.
# rollout = "resident" stops if the resident path cannot be taken, so the
# first run is resident for sure. Observations and rewards are normalized (the
# default for box actions) in both runs. Settings of train_and_run_pendulum.R.
# A random policy scores about -1200 per episode, one that swings up and holds
# about -200; the check asks for > -400. The rollouts differ in the reset RNG,
# so the returns agree only within the seed-to-seed spread.
# Exits with status 1 when the resident run is below the threshold.
suppressMessages(devtools::load_all(quiet = TRUE))

opt <- list(steps = 3e5, n = 16, seed = 1)
for (a in commandArgs(TRUE)) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (!kv[1] %in% names(opt)) stop("unknown key: ", kv[1])
  opt[[kv[1]]] <- as.numeric(kv[2])
}
str(opt)
if (!(ggml_vulkan_available() && ggml_vulkan_device_count() > 0L)) stop("needs a Vulkan device")

n_steps <- 128L
iters <- max(1L, as.integer(round(opt$steps / (opt$n * n_steps))))

run <- function(rollout) {
  set.seed(opt$seed)
  env <- rl_gpu_env("pendulum_gym", n = as.integer(opt$n), device = "gpu")
  t0 <- Sys.time()
  m <- rl_ppo_train(env, iterations = iters, n_steps = n_steps, epochs = 10,
                    minibatch = 256, lr = 1e-3, gamma = 0.9, lambda = 0.95,
                    ent_coef = 0, hidden = 64, device = "gpu", verbose = FALSE,
                    rollout = rollout)
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  ev <- rl_evaluate(m, "pendulum_gym", n_episodes = 20, seed = 2000000L)
  cat(sprintf("%-8s %6.1f s (%.3f s / iteration): return %.1f +- %.1f\n",
              rollout, secs, secs / iters, ev[["return_mean"]], ev[["return_std"]]))
  list(secs = secs, ret = ev[["return_mean"]])
}

cat(sprintf("%d iterations x %d envs x %d steps\n", iters, as.integer(opt$n), n_steps))
res <- run("resident")
hst <- run("host")
cat(sprintf("resident / host time: %.2f\n", res$secs / hst$secs))

ok <- res$ret > -400
cat(if (ok) "OK: resident > -400\n" else "FAIL: resident <= -400\n")
if (!ok) quit(status = 1)
