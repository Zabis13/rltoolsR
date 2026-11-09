# CartPole-v1: sanity check of PPO (discrete actions, GAE, update) before
# harder environments. Solved = mean return >= 475 over 100 consecutive
# episodes (Gymnasium's threshold for v1).
#
#   Rscript inst/examples/train_cartpole.R [steps=1e5] [norm_reward=0|1] [seed=1]
#
# GPU only: the environment is rl_gpu_env("cartpole") on Vulkan and the
# network trains on the GPU; the script stops if either is not on the GPU.
# Every evaluation also records where the network ran (column `gpu`, 1 = the
# ag device was "gpu" on a Vulkan backend). Evaluation itself steps the C++
# environment (rl_evaluate() does not take GPU environments); the network
# forward there still runs on the GPU. Model saved to /tmp/cartpole_ppo.rds.
suppressMessages(devtools::load_all(quiet = TRUE))

opt <- list(steps = 1e5, norm_reward = 0, seed = 1)
for (a in commandArgs(TRUE)) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (!kv[1] %in% names(opt)) stop("unknown key: ", kv[1])
  opt[[kv[1]]] <- as.numeric(kv[2])
}
str(opt)
set.seed(opt$seed)

ag_state <- ggmlR:::.ag_device_state
on_gpu <- function() {
  identical(ag_state$device, "gpu") && !is.null(ag_state$backend) &&
    startsWith(ggml_backend_name(ag_state$backend), "Vulkan")
}
env_backend <- function(env) ggml_backend_name(env$st$backend)

# ---- 1. train PPO on the GPU ------------------------------------------------
# rl-baselines3-zoo settings for CartPole-v1: 8 envs x 32 steps, 20 epochs,
# minibatch 256, gamma 0.98, lambda 0.8, no entropy bonus
env <- rl_gpu_env("cartpole", n = 8L, device = "gpu")
print(env)
cat("environment backend:", env_backend(env), "\n")
stopifnot(startsWith(env_backend(env), "Vulkan"))
algo <- rl_ppo(n_steps = 32, epochs = 20, minibatch = 256, lr = 1e-3, gamma = 0.98,
               lambda = 0.8, ent_coef = 0, hidden = 64,
               normalize_reward = opt$norm_reward == 1)
# runs inside rl_train(): records where the network is at that moment
eval100 <- function(m) c(rl_evaluate(m, "cartpole", n_episodes = 100), gpu = as.numeric(on_gpu()))
t0 <- Sys.time()
m <- rl_train(env, algo, total_steps = opt$steps, device = "gpu",
              eval_every_steps = 1e4, eval_fn = eval100,
              log = "/tmp/cartpole_ppo.csv", overwrite = TRUE)
secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("\nnetwork device: %s; on the GPU at every evaluation: %s\n", m$device,
            all(m$eval$gpu == 1)))
if (m$device != "gpu" || !all(m$eval$gpu == 1)) stop("training did not run on the GPU")

# ---- 2. verdict -------------------------------------------------------------
ev <- rl_evaluate(m, "cartpole", n_episodes = 100, seed = 2000000L)
cat(sprintf("trained %g steps in %.0f s\n", opt$steps, secs))
cat(sprintf("100 episodes (greedy): return %.1f +- %.1f, length %.1f\n",
            ev[["return_mean"]], ev[["return_std"]], ev[["length_mean"]]))
cat(if (ev[["return_mean"]] >= 475) "SOLVED (>= 475)\n" else "not solved (< 475)\n")
rl_save_model(m, "/tmp/cartpole_ppo.rds")
cat("saved /tmp/cartpole_ppo.rds\n")
