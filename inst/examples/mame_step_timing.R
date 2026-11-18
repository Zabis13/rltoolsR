# Where rl_step of rl_mame_env("mk3") spends its time: random actions,
# internal mame_* functions wrapped with wall-clock timers (no Rprof: its
# signals break the blocking FIFO reads).
#
#   Rscript inst/examples/mame_step_timing.R [steps=2000] [n=8] [frameskip=4]
#
# send    - writing actions to the FIFOs
# wait    - mame_read_exact: waiting for MAME + reading 18 bytes
# parse   - mame_read minus wait
# restart - mame_restart after a finished round (its send/read inside)
# step R  - rest of rl_step (rewards, ground, observation)
# outside - outside rl_step (random actions, loop)
suppressMessages(devtools::load_all(quiet = TRUE))

opt <- list(steps = 2000, n = 8, frameskip = 4)
for (a in commandArgs(TRUE)) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (!kv[1] %in% names(opt)) stop("unknown key: ", kv[1])
  opt[[kv[1]]] <- as.numeric(kv[2])
}
str(opt)
set.seed(1)

ns <- asNamespace("rltoolsR")
tm <- c(send = 0, wait = 0, read = 0, restart = 0)
n_restart <- 0
in_restart <- FALSE
now <- function() as.numeric(Sys.time())
wrap <- function(name, key, count = FALSE) {
  orig <- get(name, envir = ns)
  f <- function(...) {
    if (in_restart && key != "restart") return(orig(...))
    t0 <- now()
    if (key == "restart") { in_restart <<- TRUE; on.exit(in_restart <<- FALSE) }
    out <- orig(...)
    tm[[key]] <<- tm[[key]] + now() - t0
    if (count) n_restart <<- n_restart + 1
    out
  }
  if (bindingIsLocked(name, ns)) unlockBinding(name, ns)
  assign(name, f, envir = ns)
}
wrap("mame_send", "send")
wrap("mame_read_exact", "wait")
wrap("mame_read", "read")
wrap("mame_restart", "restart", count = TRUE)

env <- rl_mame_env("mk3", n = opt$n, cart = "/mnt/Data2/ROMs/Mortal Kombat 3 (USA).zip",
                   start = "/mnt/Data2/ROMs/states/genesis/sonya_sindel_cpu.sta",
                   frameskip = opt$frameskip, mame = "/usr/games/mame")
obs <- rl_reset(env)
tm[] <- 0; n_restart <- 0

t_step <- 0
t0 <- now()
for (i in seq_len(opt$steps)) {
  a <- sample.int(63L, env$n, replace = TRUE) - 1L
  ts <- now()
  r <- rl_step(env, a)
  t_step <- t_step + now() - ts
}
total <- now() - t0
rl_close(env)

parse <- tm[["read"]] - tm[["wait"]]
step_r <- t_step - tm[["send"]] - tm[["read"]] - tm[["restart"]]
res <- c(send = tm[["send"]], wait = tm[["wait"]], parse = parse,
         restart = tm[["restart"]], `step R` = step_r, outside = total - t_step)
cat(sprintf("\n%d steps x %d processes in %.1f s: %.0f env steps/s, %d restarts\n",
            opt$steps, opt$n, total, opt$steps * opt$n / total, n_restart))
cat(sprintf("%-8s %8s %8s %6s\n", "part", "s", "ms/step", "%"))
for (k in names(res))
  cat(sprintf("%-8s %8.2f %8.3f %6.1f\n", k, res[[k]], 1000 * res[[k]] / opt$steps,
              100 * res[[k]] / total))
