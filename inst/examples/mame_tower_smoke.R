# Smoke test of the tower mode of rl_mame_env("mk3"): boot, select, tower 1,
# fights one after another, Continue after a lost match, console reset by
# the watchdog. Random actions, or a trained model (model=path, greedy).
#
#   Rscript inst/examples/mame_tower_smoke.R [steps=5000] [n=1] [fighter=sonya] [show=0] [model=]
#
# model=last (newest model of train_mk3.R in /tmp), snap or tower (newest of
# that mode), or a path.
#
# Prints every finished round (with rounds won by P1 / P2 at that moment,
# CPU fighter id) and steps where the CPU's HP went up inside a round (an
# endurance fight: a second opponent). MAME logs (boot / reboot lines):
# <work_dir>/pN/mame.log.
suppressMessages(devtools::load_all(quiet = TRUE))

opt <- list(steps = "5000", n = "1", fighter = "sonya", show = "0", model = "")
for (a in commandArgs(TRUE)) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (!kv[1] %in% names(opt)) stop("unknown key: ", kv[1])
  opt[[kv[1]]] <- if (length(kv) > 1) kv[2] else ""
}
str(opt)
set.seed(1)
steps <- as.numeric(opt$steps)

t0 <- Sys.time()
env <- rl_mame_env("mk3", n = as.integer(opt$n), cart = "/mnt/Data2/ROMs/Mortal Kombat 3 (USA).zip",
                   fighter = opt$fighter, mame = "/usr/games/mame", show = opt$show == "1")
cat(sprintf("boot to the first fight: %.1f s\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
print(env)
# model=last / snap / tower: the newest /tmp/mk3_<...>.rds of train_mk3.R
if (opt$model %in% c("last", "snap", "tower")) {
  pat <- if (opt$model == "last") "^mk3_.*[0-9]\\.rds$" else sprintf("^mk3_%s.*[0-9]\\.rds$", opt$model)
  f <- list.files("/tmp", pattern = pat, full.names = TRUE)
  if (!length(f)) stop("no model /tmp/", pat)
  opt$model <- f[which.max(file.mtime(f))]
}
if (nzchar(opt$model)) cat("model", opt$model, "\n")
m <- if (nzchar(opt$model)) rl_load_model(opt$model) else NULL

obs <- rl_reset(env)
rounds <- list(); hp_up <- list()
len <- integer(env$n); ret <- numeric(env$n)
prev <- env$st$s
prev_armed <- env$st$armed
paused <- 0
t0 <- Sys.time()
for (i in seq_len(steps)) {
  a <- if (is.null(m)) sample.int(63L, env$n, replace = TRUE) - 1L else rl_act(m, obs, greedy = TRUE)
  r <- rl_step(env, a)
  s <- env$st$s
  len <- len + 1L; ret <- ret + r$reward
  paused <- paused + sum(!r$active)
  up <- which(prev_armed & env$st$armed & s[, "hp2"] > prev[, "hp2"] & prev[, "hp2"] > 0)
  prev_armed <- env$st$armed
  for (k in up) hp_up[[length(hp_up) + 1L]] <- data.frame(step = i, row = k, from = prev[k, "hp2"],
                                                          to = s[k, "hp2"])
  for (k in which(r$terminated | r$truncated)) {
    rounds[[length(rounds) + 1L]] <- data.frame(
      step = i, row = k, end = r$end[k], len = len[k], hp1 = s[k, "hp1"], hp2 = s[k, "hp2"],
      won1 = s[k, "won1"], won2 = s[k, "won2"], ret = round(ret[k], 2))
    len[k] <- 0L; ret[k] <- 0
  }
  prev <- s
  obs <- r$obs
}
secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
work_dir <- env$work_dir
rl_close(env)

cat(sprintf("%g steps x %d rows in %.1f s: %.0f env steps/s, paused (not played) %d row-steps\n",
            steps, env$n, secs, steps * env$n / secs, paused))
cat("\nfinished rounds:\n")
if (length(rounds)) print(do.call(rbind, rounds), row.names = FALSE) else cat("  none\n")
cat("\nCPU HP up inside a round (endurance?):\n")
if (length(hp_up)) print(do.call(rbind, hp_up), row.names = FALSE) else cat("  none\n")
cat("\nMAME log lines:\n")
for (p in list.files(work_dir, pattern = "^p[0-9]+$", full.names = TRUE)) {
  l <- grep("^rlt:", readLines(file.path(p, "mame.log"), warn = FALSE), value = TRUE)
  cat(basename(p), ":\n", paste0("  ", l, "\n"), sep = "")
}
