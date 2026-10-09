# Check that PPO self-play learns tic-tac-toe (moved out of the test suite).
#
#   Rscript inst/examples/check_learning_tictactoe.R [seed]
#
# Trains on the CPU for 30 iterations and plays the greedy policy against a
# random opponent. Thresholds sit outside both ranges over 8 seeds (CPU,
# ent_coef 0.03, 30 iterations): trained win 0.91-0.97, loss 0.00-0.09;
# untrained win 0.53-0.73, loss 0.15-0.34. Exits with status 1 on failure.

library(rltoolsR)

args <- commandArgs(trailingOnly = TRUE)
seed <- if (length(args)) as.integer(args[1]) else 1L

set.seed(seed)
env <- rl_gpu_env("tictactoe", n = 256, opponent = "self", device = "cpu")
t0 <- Sys.time()
m <- rl_ppo_train(env, iterations = 30, minibatch = 1024, device = "cpu",
                  eval_every = 30, verbose = FALSE,
                  eval_fn = function(m) rl_ppo_eval_tictactoe(m, n = 512, device = "cpu"))
cat(sprintf("seed %d, %.1f s: win %.3f, draw %.3f, loss %.3f\n", seed,
            as.numeric(difftime(Sys.time(), t0, units = "secs")),
            m$eval$win[1], m$eval$draw[1], m$eval$loss[1]))

ok <- all(is.finite(as.matrix(m$history))) && m$eval$win[1] > 0.8 && m$eval$loss[1] < 0.15
cat(if (ok) "OK\n" else "FAIL: expected win > 0.8 and loss < 0.15\n")
if (!ok) quit(status = 1)
