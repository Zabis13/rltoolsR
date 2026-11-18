# Check that PPO self-play learns tic-tac-toe (moved out of the test suite).
#
#   Rscript inst/examples/check_learning_tictactoe.R [seed] [device]
#
# device: "cpu" (default) or "gpu". On the GPU the environment, the network and
# the rollout all run on the device (rollout = "resident" is picked by itself).
#
# Trains for 30 iterations and plays the greedy policy against a random
# opponent. Ranges over seeds 1-24 (ent_coef 0.03, 30 iterations, measured
# 2026-11-17):
#   CPU  win 0.78-0.96 (mean 0.888), loss 0.04-0.23; 1 of 24 below the thresholds
#   GPU  win 0.65-0.95 (mean 0.871), loss 0.04-0.30; 4 of 24 below the thresholds
#   untrained  win 0.53-0.73, loss 0.15-0.34
# CPU and GPU do not differ beyond the seed-to-seed spread (seeds 9-24: means
# 0.886 and 0.880). The thresholds sit above the untrained range, but a trained
# run with an unlucky seed can still fall below them: one FAIL is not a
# regression, a mean over many seeds near the untrained range is.
# Exits with status 1 on failure.

library(rltoolsR)

args <- commandArgs(trailingOnly = TRUE)
seed <- if (length(args) >= 1L) as.integer(args[1]) else 1L
device <- if (length(args) >= 2L) args[2] else "cpu"
stopifnot(device %in% c("cpu", "gpu"))

set.seed(seed)
env <- rl_gpu_env("tictactoe", n = 256, opponent = "self", device = device)
t0 <- Sys.time()
m <- rl_ppo_train(env, iterations = 30, minibatch = 1024, device = device,
                  eval_every = 30, verbose = FALSE,
                  eval_fn = function(m) rl_ppo_eval_tictactoe(m, n = 512, device = device))
cat(sprintf("seed %d, %s, %.1f s: win %.3f, draw %.3f, loss %.3f\n", seed, device,
            as.numeric(difftime(Sys.time(), t0, units = "secs")),
            m$eval$win[1], m$eval$draw[1], m$eval$loss[1]))

ok <- all(is.finite(as.matrix(m$history))) && m$eval$win[1] > 0.8 && m$eval$loss[1] < 0.15
cat(if (ok) "OK\n" else "FAIL: expected win > 0.8 and loss < 0.15\n")
if (!ok) quit(status = 1)
