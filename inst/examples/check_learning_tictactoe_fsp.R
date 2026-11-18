# Tic-tac-toe, PPO with fictitious self-play (pool = TRUE) on the resident
# rollout against the host rollout: same settings, same seed, win / draw /
# loss of the greedy policy against a random opponent and the training time.
#
#   Rscript inst/examples/check_learning_tictactoe_fsp.R [seed] [iterations]
#
# GPU only: rl_gpu_env("tictactoe", opponent = "self", device = "gpu"). The
# resident run has rollout = "resident" (an error if it cannot be taken); its
# opponent snapshot plays in the same recording, and vs_pool / side are
# redrawn on the device, so the runs agree only within the seed-to-seed
# spread. The thresholds are the ones of check_learning_tictactoe.R
# (win > 0.8, loss < 0.15): one FAIL on an unlucky seed is not a regression.
# Exits with status 1 when the resident run fails them.

suppressMessages(devtools::load_all(quiet = TRUE))

args <- commandArgs(trailingOnly = TRUE)
seed <- if (length(args) >= 1L) as.integer(args[1]) else 1L
iters <- if (length(args) >= 2L) as.integer(args[2]) else 30L
if (!(ggml_vulkan_available() && ggml_vulkan_device_count() > 0L)) stop("needs a Vulkan device")

run <- function(rollout) {
  set.seed(seed)
  env <- rl_gpu_env("tictactoe", n = 256, opponent = "self", device = "gpu")
  t0 <- Sys.time()
  m <- rl_ppo_train(env, iterations = iters, minibatch = 1024, device = "gpu",
                    pool = TRUE, rollout = rollout, eval_every = iters, verbose = FALSE,
                    eval_fn = function(m) rl_ppo_eval_tictactoe(m, n = 512, device = "gpu"))
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  h <- m$history
  pg <- if ("pool_games" %in% names(h)) sum(h$pool_games, na.rm = TRUE) else NA
  cat(sprintf("%-8s %6.1f s (%.3f s / iteration): win %.3f, draw %.3f, loss %.3f; pool games %g\n",
              rollout, secs, secs / iters, m$eval$win[1], m$eval$draw[1], m$eval$loss[1], pg))
  list(secs = secs, win = m$eval$win[1], loss = m$eval$loss[1],
       finite = all(is.finite(as.matrix(h[setdiff(names(h), grep("^pool_", names(h), value = TRUE))]))))
}

cat(sprintf("seed %d, %d iterations x 256 games x 9 steps\n", seed, iters))
res <- run("resident")
hst <- run("host")
cat(sprintf("resident / host time: %.2f\n", res$secs / hst$secs))

ok <- res$finite && res$win > 0.8 && res$loss < 0.15
cat(if (ok) "OK\n" else "FAIL: expected win > 0.8 and loss < 0.15 (resident)\n")
if (!ok) quit(status = 1)
