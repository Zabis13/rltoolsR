# Smoke test of rl_mame_env("mk3") without training: random actions,
# checks of shapes, rewards, round end and same-step reset, speed.
#
#   Rscript inst/examples/mame_env_smoke.R [steps=3000] [n=2] [frameskip=4] [show=0]
#
# Log of finished rounds: /tmp/mk3_smoke.csv
suppressMessages(devtools::load_all(quiet = TRUE))

opt <- list(steps = 3000, n = 2, frameskip = 4, show = 0)
for (a in commandArgs(TRUE)) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (!kv[1] %in% names(opt)) stop("unknown key: ", kv[1])
  opt[[kv[1]]] <- as.numeric(kv[2])
}
str(opt)
set.seed(1)

env <- rl_mame_env("mk3", n = opt$n, cart = "/mnt/Data2/ROMs/Mortal Kombat 3 (USA).zip",
                   start = "/mnt/Data2/ROMs/states/genesis/sonya_sindel_cpu.sta",
                   frameskip = opt$frameskip, mame = "/usr/games/mame", show = opt$show == 1)
on.exit(rl_close(env))
print(env)

fail <- character(0)
check <- function(ok, msg) if (!isTRUE(ok)) fail <<- c(fail, msg)

t0 <- Sys.time()
obs <- rl_reset(env)
cat(sprintf("reset: %.2f s\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
print(round(obs, 3))
cat("raw state after reset:\n"); print(env$st$s)
check(identical(dim(obs), c(env$n, 9L)), "reset: obs dim")
check(all(obs[, 1:2] == 1), "reset: HP not full")

ret <- numeric(env$n); len <- integer(env$n)
rounds <- data.frame()
neg <- 0L; ground_log <- data.frame(); g_prev <- env$st$ground
t0 <- Sys.time()
for (t in seq_len(opt$steps)) {
  a <- sample.int(63L, env$n, replace = TRUE) - 1L
  r <- rl_step(env, a)
  check(identical(dim(r$obs), c(env$n, 9L)) && all(is.finite(r$obs)), sprintf("step %d: obs", t))
  neg <- neg + sum(r$final_obs[, 5] < 0)
  g <- env$st$ground; chg <- setdiff(which(g != g_prev), which(r$terminated | r$truncated))
  for (i in chg) {
    s <- env$st$s[i, ]
    ground_log <- rbind(ground_log, data.frame(step = t, row = i, from = g_prev[i], to = g[i],
                                               y1 = s[["y1"]], y2 = s[["y2"]],
                                               air1 = s[["air1"]], air2 = s[["air2"]]))
  }
  g_prev <- g
  check(length(r$reward) == env$n && all(is.finite(r$reward)), sprintf("step %d: reward", t))
  out <- which(abs(r$obs) > 2, arr.ind = TRUE)
  for (k in seq_len(nrow(out))) {
    i <- out[k, 1]; s <- env$st$s[i, ]
    check(FALSE, sprintf("step %d row %d: obs col %d = %.3f (x1 %d x2 %d y1 %d y2 %d)",
                         t, i, out[k, 2], r$obs[i, out[k, 2]], s["x1"], s["x2"], s["y1"], s["y2"]))
  }
  ret <- ret + r$reward; len <- len + 1L
  d <- which(r$terminated | r$truncated)
  for (i in d) {
    f <- r$final_obs[i, ]
    rounds <- rbind(rounds, data.frame(step = t, row = i, end = r$end[i], len = len[i],
                                       hp_own = f[1], hp_opp = f[2], timer = f[7],
                                       ret = ret[i], last_r = r$reward[i]))
    if (r$end[i] == "ko") check(min(f[1:2]) == 0, sprintf("step %d row %d: ko without HP 0", t, i))
    check(abs(r$reward[i]) >= 0.5, sprintf("step %d row %d: no win/loss bonus", t, i))
    check(all(r$obs[i, 1:2] == 1), sprintf("step %d row %d: obs after reset, HP not full", t, i))
    ret[i] <- 0; len[i] <- 0L
  }
}
secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("%d steps x %d rows in %.1f s: %.0f env steps/s, %.0f emulated fps per process\n",
            opt$steps, env$n, secs, opt$steps * env$n / secs, opt$steps * opt$frameskip / secs))

cat("\nfinished rounds:\n")
if (nrow(rounds)) print(rounds, digits = 3, row.names = FALSE) else cat("  none\n")
utils::write.csv(rounds, "/tmp/mk3_smoke.csv", row.names = FALSE)

cat(sprintf("\nown height < 0: %d of %d row-steps (%.2f%%)\n", neg, opt$steps * env$n,
            100 * neg / (opt$steps * env$n)))
cat("ground changes (not at a round reset):\n")
if (nrow(ground_log)) print(ground_log, row.names = FALSE) else cat("  none\n")

fail <- unique(fail)
if (length(fail)) {
  cat("\nFAIL (", length(fail), "):\n", sep = ""); cat(head(fail, 30), sep = "\n")
} else cat("\nOK\n")
