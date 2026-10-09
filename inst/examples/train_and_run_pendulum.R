# Train PPO with a Gaussian policy on Pendulum, save the model, load it back
# and run it.
#
#   Rscript inst/examples/train_and_run_pendulum.R [model.rds] [total_steps]
#
# The model is always saved to /tmp (only the file name is taken from the
# argument).
#
# Training uses the GPU when a Vulkan device is available, otherwise the CPU.
# Observations and rewards are normalized (the default for continuous
# actions); the observation statistics are saved with the model and used by
# rl_act(). A random policy scores about -1200 per episode, a policy that
# swings the pendulum up and holds it about -200.

library(rltoolsR)

args <- commandArgs(trailingOnly = TRUE)
path <- file.path("/tmp", basename(if (length(args) >= 1) args[1] else "pendulum_ppo.rds"))
total_steps <- if (length(args) >= 2) as.numeric(args[2]) else 3e5

# mean episode return of the greedy policy over 8 fixed starts
pendulum_return <- function(m) {
  env <- rl_vec_env("pendulum", n = 8, seed = 100)
  obs <- rl_reset(env, seed = 100)
  total <- 0
  for (i in 1:200) {
    r <- rl_step(env, rl_act(m, obs, greedy = TRUE))
    total <- total + mean(r$reward)
    obs <- r$obs
  }
  c(return = total)
}

# ---- train ----
set.seed(1)
env <- rl_vec_env("pendulum", n = 16, seed = 1)
print(env)
algo <- rl_ppo(n_steps = 128, epochs = 10, minibatch = 256, lr = 1e-3,
               gamma = 0.9, lambda = 0.95, ent_coef = 0, hidden = 64)
t0 <- Sys.time()
model <- rl_train(env, algo, total_steps = total_steps, device = "auto",
                  eval_every_steps = 10 * 2048, eval_fn = pendulum_return,
                  verbose = TRUE)
cat(sprintf("trained %d steps in %.0f s on %s\n", model$train_state$steps,
            as.numeric(difftime(Sys.time(), t0, units = "secs")), model$device))
rl_save_model(model, path)
cat("saved:", path, "\n\n")

# ---- load and run on CPU ----
m <- rl_load_model(path, device = "cpu")
cat(sprintf("greedy return on CPU after load: %.1f\n", pendulum_return(m)))
