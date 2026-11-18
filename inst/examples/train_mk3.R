# PPO against the CPU in Mortal Kombat 3 (Genesis, MAME): one round per
# episode, every round from the round-1 snapshot.
#
#   Rscript inst/examples/train_mk3.R [steps=5e5] [n=4] [frameskip=4] [seed=1] [show=0] [fighter=] [resume=] [init=]
#
# Training: rl_mame_env("mk3", n) with the CPU as P2, PPO on the GPU.
# fighter=sonya: tower mode (no snapshot, tower 1 from the select screen,
# the game goes on); the evaluation environment plays the tower too.
# Evaluation every 5e4 steps on a separate environment (4 processes, greedy
# policy, 8 rounds): win rate, KO wins, mean HP difference at the end.
# Every run has its own files, nothing is overwritten:
#   /tmp/mk3_<snap|tower_<fighter>>_<YYYYmmdd_HHMMSS>.rds   model
#   ..._ckpt.rds   checkpoint,   ....csv   log
# resume=<run name> (e.g. mk3_snap_20261114_153000) goes on from that run's
# checkpoint and log; the model is saved under a new name.
# init=<model .rds> (e.g. /tmp/mk3_snap_20261114_153000.rds) starts a new run
# from that model's weights; Adam, step counter and log start fresh.
suppressMessages(devtools::load_all(quiet = TRUE))

opt <- list(steps = 5e5, n = 4, frameskip = 4, seed = 1, show = 0, fighter = "",
            resume = "", init = "")
for (a in commandArgs(TRUE)) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (!kv[1] %in% names(opt)) stop("unknown key: ", kv[1])
  opt[[kv[1]]] <- if (kv[1] %in% c("fighter", "resume", "init")) kv[2] else as.numeric(kv[2])
}
tower <- nzchar(opt$fighter)
resume <- nzchar(opt$resume)
init <- nzchar(opt$init)
if (resume && init) stop("resume= and init= are exclusive")
if (init && !file.exists(opt$init)) stop("no model ", opt$init)
str(opt)
set.seed(opt$seed)

cart  <- "/mnt/Data2/ROMs/Mortal Kombat 3 (USA).zip"
start <- "/mnt/Data2/ROMs/states/genesis/sonya_sindel_cpu.sta"
run   <- sprintf("mk3_%s_%s", if (tower) paste0("tower_", opt$fighter) else "snap",
                 format(Sys.time(), "%Y%m%d_%H%M%S"))
base  <- file.path("/tmp", if (resume) opt$resume else run)
ckpt  <- paste0(base, "_ckpt.rds")
log   <- paste0(base, ".csv")
model <- file.path("/tmp", paste0(run, ".rds"))
if (resume && !file.exists(ckpt)) stop("no checkpoint ", ckpt)
cat("run", run, if (resume) paste("(resumes", opt$resume, ")"), "\n")
mk_env <- function(n, show = FALSE)
  rl_mame_env("mk3", n = n, cart = cart, start = if (!tower) start,
              fighter = if (tower) opt$fighter, frameskip = opt$frameskip,
              mame = "/usr/games/mame", show = show)

env  <- mk_env(opt$n, show = opt$show == 1)
eenv <- mk_env(4L)
print(env)

# 8 rounds, greedy: share won, won by KO, mean (own HP - CPU HP) / 166 at the end
eval_rounds <- function(m, rounds = 8L) {
  obs <- rl_reset(eenv)
  won <- ko <- hp <- numeric(0)
  while (length(won) < rounds) {
    r <- rl_step(eenv, rl_act(m, obs, greedy = TRUE))
    d <- which(r$terminated | r$truncated)
    if (length(d)) {
      f <- r$final_obs[d, , drop = FALSE]
      won <- c(won, f[, 1] > f[, 2]); ko <- c(ko, f[, 1] > f[, 2] & r$end[d] == "ko")
      hp <- c(hp, f[, 1] - f[, 2])
    }
    obs <- r$obs
  }
  c(win = mean(won), win_ko = mean(ko), hp_diff = mean(hp))
}

algo <- rl_ppo(n_steps = 128, epochs = 4, minibatch = 256, lr = 3e-4, gamma = 0.99,
               lambda = 0.95, ent_coef = 0.01, hidden = 128, normalize_reward = TRUE)
if (init) {
  # weights from the saved model, a fresh Adam on them (as set_train_state does)
  m0 <- read_model_file(opt$init, "init=")
  init0 <- algo$init
  algo$init <- function(env, device) {
    st <- init0(env, device)
    st$net <- load_net(m0$net, function() ppo_build(st$meta), device)
    st$opt <- optimizer_adam(st$net$parameters(), lr = algo$params$lr)
    st
  }
  cat("init from", opt$init, "\n")
}
t0 <- Sys.time()
m <- tryCatch(
  rl_train(env, algo, total_steps = opt$steps, device = "gpu",
           eval_every_steps = 5e4, eval_fn = eval_rounds,
           log = log, overwrite = FALSE, append = resume,
           checkpoint = ckpt, checkpoint_every_steps = 5e4,
           resume = if (resume) ckpt),
  finally = { rl_close(env); rl_close(eenv) })
secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("trained %g steps in %.0f s (%.0f steps/s)\n", opt$steps, secs, opt$steps / secs))
rl_save_model(m, model)
cat("saved", model, "\n")
