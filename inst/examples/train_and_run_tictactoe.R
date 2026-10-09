# Train a small PPO policy on tic-tac-toe, save it, load it back and run it.
#
#   Rscript inst/examples/train_and_run_tictactoe.R [model.rds]
#
# The model is always saved to /tmp (only the file name is taken from the
# argument).
#
# Training uses the GPU when a Vulkan device is available, otherwise the CPU.
# The saved model is then loaded on the CPU and played against a random
# opponent; one game is printed move by move.

library(rltoolsR)

args <- commandArgs(trailingOnly = TRUE)
path <- file.path("/tmp", basename(if (length(args)) args[1] else "tictactoe_ppo.rds"))

# ---- train ----
set.seed(1)
env <- rl_gpu_env("tictactoe", n = 256, opponent = "self")   # env on the CPU (default)
print(env)
model <- rl_ppo_train(env, iterations = 100, minibatch = 2024, hidden = 64,
                      device = env$device, verbose = TRUE)
rl_save_model(model, path)
cat("saved:", path, "\n\n")

# ---- load and run on CPU ----
m <- rl_load_model(path, device = "cpu")
res <- rl_ppo_eval_tictactoe(m, n = 1000, device = "cpu")
cat(sprintf("vs random (1000 games): win %.1f%%, draw %.1f%%, loss %.1f%%\n\n",
            100 * res["win"], 100 * res["draw"], 100 * res["loss"]))

show_board <- function(obs) {
  s <- ifelse(obs > 0, "X", ifelse(obs < 0, "O", "."))
  for (r in 0:2) cat(" ", s[r * 3 + 1:3], "\n")
  cat("\n")
}

# one game: the model plays X, the random opponent answers inside env
game <- rl_gpu_env("tictactoe", n = 1, opponent = "random", device = "cpu")
obs <- rl_reset(game)
legal <- rl_legal_actions(game)
for (move in 1:5) {
  a <- rl_act(m, obs, legal, greedy = TRUE)
  cat("X plays cell", a, "\n")
  r <- rl_step(game, a)
  if (r$terminated[1]) {
    cat(c("-1" = "X lost", "0" = "draw", "1" = "X won")[as.character(r$reward[1])], "\n")
    break
  }
  obs <- r$obs
  legal <- r$legal
  show_board(obs[1, ])
}
