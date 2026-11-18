# Does the FIFO lockstep loop R <-> MAME work, and how fast is it?
#
#   Rscript inst/examples/mame_fifo_probe.R <mame> <cart> [n] [k] [steps] [show|-] [start.sta]
#
# start.sta: snapshot every process starts from (default: the manual one,
# /mnt/Data2/ROMs/states/genesis/quick.sta; power-on boot if it is missing).
# <cart>: Mortal Kombat 3 Genesis cartridge, .gen or a .zip holding one.
# Starts n MAME processes (genesis, no video/sound, unthrottled) running
# mame_fifo_probe.lua and talks to them over two FIFOs each. Checks:
#   1. lockstep   each step advances every process by exactly k frames
#   2. speed      steps/s for the whole batch: actions are written to all
#                 processes first, then all answers are read, so they emulate
#                 in parallel
#   3. save/load  steps after a load repeat the steps after the save
#                 (same frame numbers, same RAM bytes)
# Six-button pads on both ports; actions 0..62 are mapped to pad inputs in
# the .lua. The port/field names are dumped to <dir>/ports.tsv.
# Work files (FIFOs, states, logs) go to /tmp/rlt_mame.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: mame_fifo_probe.R <mame> <cart> [n] [k] [steps]")
mame     <- args[1]
cart     <- args[2]
n        <- if (length(args) >= 3) as.integer(args[3]) else 4L
k        <- if (length(args) >= 4) as.integer(args[4]) else 4L
steps    <- if (length(args) >= 5) as.integer(args[5]) else 2000L
show     <- length(args) >= 6 && args[6] == "show"   # process 1 in a window
# Start snapshot: every process loads it before the first observation.
# Must come from MAME run with the same cart and -ctrlN md6button.
start    <- if (length(args) >= 7) args[7] else "/mnt/Data2/ROMs/states/genesis/quick.sta"
if (!file.exists(start)) { cat("no start snapshot, booting from power-on\n"); start <- NULL }
# The window is unthrottled too (game runs ~10x speed); for benchmarks run
# without "show": all processes with -video none.
warmup   <- 300L                        # steps before the checks: game boot
check_n  <- 20L                         # steps compared after save and load
ram_addr <- 0xFF0000                    # start of work RAM, see the .lua header
ram_n    <- 16L
obs_size <- 4L + ram_n

# Prefer the source tree copy: the installed one may be stale.
lua <- "inst/examples/mame_fifo_probe.lua"
lua <- if (file.exists(lua)) normalizePath(lua) else
  system.file("examples", "mame_fifo_probe.lua", package = "rltoolsR")
root <- "/tmp/rlt_mame"
# Only the probe's own files are removed (root may hold other work files).
dir.create(root, showWarnings = FALSE)
unlink(c(Sys.glob(file.path(root, "p[0-9]*")), file.path(root, "ports.tsv")),
       recursive = TRUE)

# MAME gets a plain .gen: unpack a .zip into the work directory.
if (grepl("\\.zip$", cart, ignore.case = TRUE)) {
  inner <- unzip(cart, list = TRUE)$Name
  inner <- inner[grepl("\\.(gen|md|bin|smd)$", inner, ignore.case = TRUE)][1]
  stopifnot(!is.na(inner))
  unzip(cart, files = inner, exdir = root, junkpaths = TRUE)
  cart <- file.path(root, basename(inner))
}
cart <- normalizePath(cart)

# ---- start processes ----------------------------------------------------
procs <- lapply(seq_len(n), function(i) {
  dir <- file.path(root, sprintf("p%d", i))
  dir.create(dir)
  if (!is.null(start)) {
    dir.create(file.path(dir, "genesis"))
    stopifnot(file.copy(start, file.path(dir, "genesis", "rlt.sta")))
  }
  act <- file.path(dir, "act"); obs <- file.path(dir, "obs")
  stopifnot(system2("mkfifo", c(act, obs)) == 0)
  env <- c(sprintf("RLT_ACT=%s", act), sprintf("RLT_OBS=%s", obs),
           sprintf("RLT_K=%d", k), sprintf("RLT_RAM_ADDR=%d", ram_addr),
           sprintf("RLT_RAM_N=%d", ram_n),
           if (i == 1) sprintf("RLT_PORTS=%s", file.path(root, "ports.tsv")),
           if (!is.null(start)) "RLT_START=1",
           "RLT_HOLD=2,5",                  # pad B = block, Y = run in MK3: held all k frames
           # -video none still opens an empty SDL window: no window at all
           if (!(show && i == 1)) "SDL_VIDEODRIVER=dummy")
  video <- if (show && i == 1) c("-window", "-verbose") else c("-video", "none")
  system2(mame, c("genesis", "-cart", shQuote(cart),
                  "-ctrl1", "md6button", "-ctrl2", "md6button", "-nothrottle",
                  video, "-sound", "none", "-skip_gameinfo",
                  "-state_directory", dir, "-cfg_directory", dir,
                  "-nvram_directory", dir, "-autoboot_script", lua),
          env = env, wait = FALSE,
          stdout = file.path(dir, "mame.log"), stderr = file.path(dir, "mame.log"))
  list(dir = dir, act = act, obs = obs)
})

# Same open order as the Lua side (act, then obs). Blocks until MAME opens
# its end: if MAME fails to start, this hangs -- see <dir>/pN/mame.log.
for (i in seq_len(n)) {
  procs[[i]]$act_con <- fifo(procs[[i]]$act, "wb", blocking = TRUE)
  procs[[i]]$obs_con <- fifo(procs[[i]]$obs, "rb", blocking = TRUE)
}
cat("connected:", n, "processes\n")

# A window started from a background process opens behind the terminal:
# raise it and keep it on top (needs wmctrl; title contains "MAME").
if (show && nzchar(Sys.which("wmctrl"))) {
  system2("wmctrl", c("-r", "MAME", "-b", "add,above"))
  system2("wmctrl", c("-a", "MAME"))
}

# FIFOs are opened blocking, yet readBin() has been seen returning 0 bytes
# while MAME was alive. Empty reads are retried (with a short sleep) and
# counted; the count is printed at the end: 0 means reads really block.
empty_reads <- 0L
read_timeout <- 5                       # seconds without data -> error
read_exact <- function(con, size) {
  out <- raw(0)
  t0 <- NULL
  while (length(out) < size) {
    chunk <- readBin(con, "raw", size - length(out))
    if (length(chunk) == 0) {
      empty_reads <<- empty_reads + 1L
      if (is.null(t0)) t0 <- proc.time()[["elapsed"]]
      if (proc.time()[["elapsed"]] - t0 > read_timeout)
        stop("no data from MAME for ", read_timeout, " s (see ", root, "/pN/mame.log)")
      Sys.sleep(0.0005)
      next
    }
    t0 <- NULL
    out <- c(out, chunk)
  }
  out
}

parse_obs <- function(r) {
  list(frame = readBin(r[1:4], "integer", size = 4, endian = "little"),
       ram = r[-(1:4)])
}

# One batch step: cmd and actions per process. Write all, then read all.
step_all <- function(cmd, a1 = rep(0L, n), a2 = rep(0L, n)) {
  for (i in seq_len(n)) {
    writeBin(as.raw(c(cmd, a1[i], a2[i], 0L)), procs[[i]]$act_con)
    flush(procs[[i]]$act_con)
  }
  lapply(seq_len(n), function(i) parse_obs(read_exact(procs[[i]]$obs_con, obs_size)))
}

frames <- function(res) vapply(res, `[[`, integer(1), "frame")

# The first observation is sent before any command.
cur <- lapply(procs, function(p) parse_obs(read_exact(p$obs_con, obs_size)))

# ---- 1. lockstep during warmup ------------------------------------------
bad <- 0L
for (s in seq_len(warmup)) {
  nxt <- step_all(0L)
  bad <- bad + sum(frames(nxt) - frames(cur) != k)
  cur <- nxt
}
cat(sprintf("lockstep: %d / %d steps off by != %d frames\n", bad, warmup * n, k))

# ---- 2. speed -----------------------------------------------------------
t0 <- proc.time()[["elapsed"]]
for (s in seq_len(steps)) {
  cur <- step_all(0L, sample.int(63L, n, TRUE) - 1L, sample.int(63L, n, TRUE) - 1L)
}
dt <- proc.time()[["elapsed"]] - t0
cat(sprintf("speed: %d batch steps in %.2f s = %.1f batch steps/s, %.1f env steps/s, %.0f frames/s per process\n",
            steps, dt, steps / dt, steps * n / dt, steps * k / dt))

# ---- 3. save / load -----------------------------------------------------
run_check <- function(first) {
  out <- list(first)
  for (s in seq_len(check_n)) out[[s + 1L]] <- step_all(0L)
  out
}
after_save <- run_check(step_all(1L))
after_load <- run_check(step_all(2L))
same <- vapply(seq_len(n), function(i) {
  all(vapply(seq_along(after_save), function(s) {
    identical(after_save[[s]][[i]], after_load[[s]][[i]])
  }, logical(1)))
}, logical(1))
cat("save/load identical per process:", same, "\n")
if (!all(same)) {
  i <- which(!same)[1]
  cat("process", i, "frames after save:", vapply(after_save, function(r) r[[i]]$frame, 1L), "\n")
  cat("process", i, "frames after load:", vapply(after_load, function(r) r[[i]]$frame, 1L), "\n")
}

# ---- shutdown -----------------------------------------------------------
for (p in procs) {
  writeBin(as.raw(c(3L, 0L, 0L, 0L)), p$act_con)
  flush(p$act_con)
  close(p$act_con); close(p$obs_con)
}
cat("empty reads (retried):", empty_reads, "\n")
cat("ports dump:", file.path(root, "ports.tsv"), "\n")
