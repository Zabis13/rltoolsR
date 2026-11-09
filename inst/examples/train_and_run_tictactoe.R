# Train a small PPO policy on tic-tac-toe, save it, load it back and run it.
#
#   Rscript inst/examples/train_and_run_tictactoe.R [model.rds] [ent_coef]
#
# The model is always saved to /tmp (only the file name is taken from the
# argument).
#
# The environment runs on the CPU (the rl_gpu_env() default: faster collect);
# the policy trains on the GPU when a Vulkan device is available, otherwise
# the CPU (bench_cpu_gpu.R: ~36 vs ~98 ms per iteration with capture).
# Every 10 iterations the policy plays 1000 games against a random opponent
# and the best one so far is saved (progress is printed at these evals only).
# The best model is then loaded on the CPU and played against a random
# opponent; one game is printed move by move.

library(rltoolsR)

args <- commandArgs(trailingOnly = TRUE)
path <- file.path("/tmp", basename(if (length(args)) args[1] else "tictactoe_ppo.rds"))
ent_coef <- if (length(args) >= 2) as.numeric(args[2]) else 0.03   # rl_ppo default
cat("ent_coef:", ent_coef, "\n")

# ---- train ----
set.seed(1)
env <- rl_gpu_env("tictactoe", n = 256, opponent = "self")   # env on the CPU (default)
print(env)
# every eval_every iterations: play 1000 games vs random, keep the best model
# (score = win - loss) in `path`; the last one goes to *_last.rds
best <- -Inf
eval_best <- function(m) {
  res <- rl_ppo_eval_tictactoe(m, n = 1000, device = "cpu")
  score <- res[["win"]] - res[["loss"]]
  if (score > best) {
    best <<- score
    rl_save_model(m, path)
  }
  res_o <- rl_ppo_eval_tictactoe(m, n = 1000, device = "cpu", side = "O")
  c(res, best = best, o_win = res_o[["win"]], o_draw = res_o[["draw"]],
    o_loss = res_o[["loss"]])
}
t0 <- Sys.time()
model <- rl_ppo_train(env, iterations = 300, device = "auto", verbose = TRUE,
                      pool = TRUE,   # fictitious self-play: 80% self, 20% past snapshots
                      ent_coef = ent_coef,
                      eval_every = 10, eval_fn = eval_best)
cat(sprintf("training (with evals): %.1f s\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))
path_last <- sub("(\\.rds)?$", "_last.rds", path)
rl_save_model(model, path_last)
cat(sprintf("saved: best (win - loss %.3f) %s, last %s\n\n", best, path, path_last))

# ---- load and run on CPU ----
m <- rl_load_model(path, device = "cpu")
res <- rl_ppo_eval_tictactoe(m, n = 1000, device = "cpu")
cat(sprintf("vs random as X (1000 games): win %.1f%%, draw %.1f%%, loss %.1f%%\n",
            100 * res["win"], 100 * res["draw"], 100 * res["loss"]))
res_o <- rl_ppo_eval_tictactoe(m, n = 1000, device = "cpu", side = "O")
cat(sprintf("vs random as O (1000 games): win %.1f%%, draw %.1f%%, loss %.1f%%\n\n",
            100 * res_o["win"], 100 * res_o["draw"], 100 * res_o["loss"]))

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
