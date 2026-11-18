# Watch a trained MK3 model fight the CPU in a MAME window (real-time speed).
#
#   Rscript inst/examples/watch_mk3.R [model=last] [rounds=5] [greedy=1] [frameskip=4]
#
# model=last (newest model of train_mk3.R in /tmp), snap or tower (newest of
# that mode), or a path.
#
# Model from train_mk3.R. The window is unthrottled, so every step is
# padded with Sys.sleep to frameskip / 60 s.
suppressMessages(devtools::load_all(quiet = TRUE))

opt <- list(model = "last", rounds = 5, greedy = 1, frameskip = 4)
for (a in commandArgs(TRUE)) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (!kv[1] %in% names(opt)) stop("unknown key: ", kv[1])
  opt[[kv[1]]] <- if (kv[1] == "model") kv[2] else as.numeric(kv[2])
}
# model=last / snap / tower: the newest /tmp/mk3_<...>.rds of train_mk3.R
if (opt$model %in% c("last", "snap", "tower")) {
  pat <- if (opt$model == "last") "^mk3_.*[0-9]\\.rds$" else sprintf("^mk3_%s.*[0-9]\\.rds$", opt$model)
  f <- list.files("/tmp", pattern = pat, full.names = TRUE)
  if (!length(f)) stop("no model /tmp/", pat)
  opt$model <- f[which.max(file.mtime(f))]
}
cat("model", opt$model, "\n")

m   <- rl_load_model(opt$model)
env <- rl_mame_env("mk3", n = 1L, cart = "/mnt/Data2/ROMs/Mortal Kombat 3 (USA).zip",
                   start = "/mnt/Data2/ROMs/states/genesis/sonya_sindel_cpu.sta",
                   frameskip = opt$frameskip, mame = "/usr/games/mame", show = TRUE)

dt  <- opt$frameskip / 60
obs <- rl_reset(env)
done <- 0
while (done < opt$rounds) {
  t0 <- Sys.time()
  r  <- rl_step(env, rl_act(m, obs, greedy = opt$greedy == 1))
  if (r$terminated[1] || r$truncated[1]) {
    f <- r$final_obs[1, ]
    done <- done + 1
    cat(sprintf("round %d: %s, own HP %.2f, CPU HP %.2f, %s\n", done,
                if (f[1] > f[2]) "WIN" else "loss", f[1], f[2], r$end[1]))
  }
  obs <- r$obs
  wait <- dt - as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (wait > 0) Sys.sleep(wait)
}
rl_close(env)
