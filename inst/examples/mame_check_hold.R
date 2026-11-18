# Do block (pad B) and run (pad Y + forward) work with RLT_HOLD=2,5?
#
#   Rscript inst/examples/mame_check_hold.R <mame> <cart> [show|-] [start.sta]
#
# One MAME process, each scenario from the round-1 snapshot: 180 steps of
# one action from the first step (input unlocks after "Fight!", ~step 20);
# idle runs three times to check that a snapshot load resets everything:
#   idle    nothing
#   block   B held (action 2)
#   walk    right (action 21)
#   run     right + Y held (action 26)
# Prints per scenario: P1 HP lost, hits taken, P1 X after 10 / 20 / 40
# steps of the action. Expected: block loses less HP than idle; run moves
# faster than walk. Directions are relative (RLT_RELATIVE): right =
# towards the opponent on either side.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: mame_check_hold.R <mame> <cart> [show|-] [start.sta]")
mame  <- args[1]
cart  <- args[2]
show  <- length(args) >= 3 && args[3] == "show"
start <- if (length(args) >= 4) args[4] else "/mnt/Data2/ROMs/states/genesis/quick.sta"
stopifnot(file.exists(start))
k        <- 4L
ram_addr <- 0xFFCB00                     # region with X, timer and HP
ram_n    <- 0xFFFA00 - ram_addr
obs_size <- 4L + ram_n

lua <- "inst/examples/mame_fifo_probe.lua"
lua <- if (file.exists(lua)) normalizePath(lua) else
  system.file("examples", "mame_fifo_probe.lua", package = "rltoolsR")
root <- "/tmp/rlt_mame"
dir <- file.path(root, "hold")
unlink(dir, recursive = TRUE)
dir.create(file.path(dir, "genesis"), recursive = TRUE)
stopifnot(file.copy(start, file.path(dir, "genesis", "rlt.sta")))

if (grepl("\\.zip$", cart, ignore.case = TRUE)) {
  inner <- unzip(cart, list = TRUE)$Name
  inner <- inner[grepl("\\.(gen|md|bin|smd)$", inner, ignore.case = TRUE)][1]
  unzip(cart, files = inner, exdir = root, junkpaths = TRUE)
  cart <- file.path(root, basename(inner))
}
cart <- normalizePath(cart)

act <- file.path(dir, "act"); obs <- file.path(dir, "obs")
stopifnot(system2("mkfifo", c(act, obs)) == 0)
env <- c(sprintf("RLT_ACT=%s", act), sprintf("RLT_OBS=%s", obs),
         sprintf("RLT_K=%d", k), sprintf("RLT_RAM_ADDR=%d", ram_addr),
         sprintf("RLT_RAM_N=%d", ram_n), "RLT_START=1",
         "RLT_HOLD=2,5",                    # pad B = block, Y = run
         "RLT_DEBUG=1",                     # applied actions -> mame.log
         "RLT_RELATIVE=1",                  # right = towards the opponent
         sprintf("RLT_X1=%d", 0xFFCB1C), sprintf("RLT_X2=%d", 0xFFCB68),
         if (!show) "SDL_VIDEODRIVER=dummy")
video <- if (show) "-window" else c("-video", "none")
system2(mame, c("genesis", "-cart", shQuote(cart),
                "-ctrl1", "md6button", "-ctrl2", "md6button", "-nothrottle",
                video, "-sound", "none", "-skip_gameinfo",
                "-state_directory", dir, "-cfg_directory", dir,
                "-nvram_directory", dir, "-autoboot_script", lua),
        env = env, wait = FALSE,
        stdout = file.path(dir, "mame.log"), stderr = file.path(dir, "mame.log"))
act_con <- fifo(act, "wb", blocking = TRUE)
obs_con <- fifo(obs, "rb", blocking = TRUE)
if (show && nzchar(Sys.which("wmctrl"))) system2("wmctrl", c("-r", "MAME", "-b", "add,above"))

read_exact <- function(size) {
  out <- raw(0); t0 <- NULL
  while (length(out) < size) {
    chunk <- readBin(obs_con, "raw", size - length(out))
    if (length(chunk) == 0) {
      if (is.null(t0)) t0 <- proc.time()[["elapsed"]]
      if (proc.time()[["elapsed"]] - t0 > 5) stop("no data from MAME (see ", dir, "/mame.log)")
      Sys.sleep(0.0005); next
    }
    t0 <- NULL; out <- c(out, chunk)
  }
  out
}
send <- function(cmd, a1 = 0L) {
  writeBin(as.raw(c(cmd, a1, 0L, 0L)), act_con); flush(act_con)
  read_exact(obs_size)[-(1:4)]
}
at  <- function(r, a) as.integer(r[a - ram_addr + 1L])
u16 <- function(r, a) 256L * at(r, a) + at(r, a + 1L)
state <- function(r) c(hp1 = at(r, 0xFFF865), hp2 = at(r, 0xFFF9D7),
                       x1 = u16(r, 0xFFCB1C), x2 = u16(r, 0xFFCB68))

# The action is sent from the first step: the game ignores it during
# "Fight!" and acts on it as soon as input unlocks, before the first CPU hit.
run <- function(action, lock = 0L, n = 180L) {
  s <- list(state(send(2L)))               # load snapshot
  for (i in seq_len(lock)) s[[length(s) + 1L]] <- state(send(0L, 0L))
  for (i in seq_len(n)) s[[length(s) + 1L]] <- state(send(0L, action))
  do.call(rbind, s)
}

invisible(read_exact(obs_size))           # first observation, after the start load
# idle three times, between other actions: a snapshot load must give the
# same run regardless of what was pressed before.
# idle_b right after idle: same as idle -> pressed inputs in the past
# matter; same as idle2 -> elapsed frames matter.
acts <- c(idle = 0L, idle_b = 0L, block = 2L, idle2 = 0L, walk = 21L, run = 26L, idle3 = 0L)
res <- lapply(acts, run)
writeBin(as.raw(c(3L, 0L, 0L, 0L)), act_con); flush(act_con)
close(act_con); close(obs_con)

cat("deterministic reset: idle == idle_b", identical(res$idle, res$idle_b),
    " idle_b == idle2", identical(res$idle_b, res$idle2),
    " idle2 == idle3", identical(res$idle2, res$idle3), "\n")
if (!identical(res$idle, res$idle2)) {
  d <- which(rowSums(res$idle != res$idle2) > 0)[1]
  cat("  idle / idle2 first differ at step", d - 1L, ":", res$idle[d, ], "/", res$idle2[d, ], "\n")
}
cat(sprintf("%-6s %8s %5s %8s %6s %6s %6s %8s\n",
            "", "HP lost", "hits", "x start", "x+25", "x+40", "x+60", "min dist"))
for (nm in names(res)) {
  m <- res[[nm]]; b <- 1L                  # action from the first step; input unlocks ~step 20
  hits <- sum(diff(m[, "hp1"]) < 0)
  cat(sprintf("%-6s %8d %5d %8d %6d %6d %6d %8d\n", nm, m[1, "hp1"] - min(m[, "hp1"]), hits,
              m[b, "x1"], m[b + 25, "x1"], m[b + 40, "x1"], m[b + 60, "x1"],
              min(abs(m[, "x2"] - m[, "x1"]))))
}
