# PPO on FrozenLake with the resident rollout (R/resident_tabular.R): the
# policy and the lake are one recorded device graph per step. Prints the
# training history and a greedy evaluation; the model is saved to /tmp.
#
#   Rscript inst/examples/train_frozen_lake.R [env=frozen_lake_det_onehot] [n=64]
#     [n_steps=32] [iters=60] [lr=3e-3] [seed=6] [device=gpu]
suppressMessages(devtools::load_all(quiet = TRUE))

opt <- list(env = "frozen_lake_det_onehot", n = 64, n_steps = 32, iters = 60,
            lr = 3e-3, seed = 6, device = "gpu")
for (a in commandArgs(TRUE)) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (!kv[1] %in% names(opt)) stop("unknown key: ", kv[1])
  opt[[kv[1]]] <- if (is.numeric(opt[[kv[1]]])) as.numeric(kv[2]) else kv[2]
}
set.seed(opt$seed)
n <- as.integer(opt$n); ns <- as.integer(opt$n_steps)

env <- rl_vec_env(opt$env, n = n, seed = opt$seed)
algo <- rl_ppo(n_steps = ns, minibatch = n * ns / 4, lr = opt$lr, ent_coef = 0.01)
t0 <- Sys.time()
m <- rl_train(env, algo, total_steps = n * ns * opt$iters, device = opt$device, verbose = FALSE)
cat(sprintf("%s: %d iterations, %.1f s\n", opt$env, opt$iters,
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))
print(utils::tail(m$history, 5))

res <- rl_evaluate(m, opt$env, n_episodes = 100)
print(res[names(res)])                 # the summary, without the per-episode table
f <- file.path("/tmp", paste0("ppo_", opt$env, ".rds"))
rl_save_model(m, f)
cat("saved", f, "\n")
