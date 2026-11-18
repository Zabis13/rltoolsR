# Where PPO training on rl_mame_env("mk3") spends its time: stage timer
# rl_prof() over collect/update, same PPO settings as train_mk3.R, no
# evaluation, log or checkpoint.
#
#   Rscript inst/examples/profile_mk3.R [steps=2e4] [n=6] [frameskip=4]
#
# collect: env   - rl_step (MAME), collect: forward - policy on the GPU,
# update: *      - PPO update, other - rest of rl_train (wall minus stages)
suppressMessages(devtools::load_all(quiet = TRUE))

opt <- list(steps = 2e4, n = 6, frameskip = 4)
for (a in commandArgs(TRUE)) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (!kv[1] %in% names(opt)) stop("unknown key: ", kv[1])
  opt[[kv[1]]] <- as.numeric(kv[2])
}
str(opt)
set.seed(1)

env <- rl_mame_env("mk3", n = opt$n, cart = "/mnt/Data2/ROMs/Mortal Kombat 3 (USA).zip",
                   start = "/mnt/Data2/ROMs/states/genesis/sonya_sindel_cpu.sta",
                   frameskip = opt$frameskip, mame = "/usr/games/mame")
algo <- rl_ppo(n_steps = 128, epochs = 4, minibatch = 256, lr = 3e-4, gamma = 0.99,
               lambda = 0.95, ent_coef = 0.01, hidden = 128, normalize_reward = TRUE)

rl_prof(TRUE)
t0 <- Sys.time()
m <- tryCatch(rl_train(env, algo, total_steps = opt$steps, device = "gpu", verbose = FALSE),
              finally = rl_close(env))
wall <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
p <- rl_prof_get()
rl_prof(FALSE)

s <- sort(p$ms / 1000, decreasing = TRUE)
s <- c(s, other = wall - sum(s))
cat(sprintf("\n%g steps x %d processes in %.1f s: %.0f steps/s\n",
            opt$steps, opt$n, wall, opt$steps / wall))
cat(sprintf("%-20s %8s %6s\n", "stage", "s", "%"))
for (k in names(s)) cat(sprintf("%-20s %8.2f %6.1f\n", k, s[[k]], 100 * s[[k]] / wall))
