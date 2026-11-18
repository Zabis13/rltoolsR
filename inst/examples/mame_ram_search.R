# Where are HP, X, Y and the round timer in MK3 (Genesis) RAM?
#
#   Rscript inst/examples/mame_ram_search.R <mame> <cart> [show] [start.sta]
#
# One MAME process (same Lua as mame_fifo_probe.R) starts from the round-1
# snapshot and sends all 64 KB of work RAM (0xFF0000-0xFFFFFF) each step.
# Scenarios, each from the snapshot (k = 4 frames per step):
#   idle    P1 does nothing for 450 steps (the CPU opponent attacks)
#   walk    P1 right 60 steps, left 60 steps, right 60 steps
#   jump    P1 up (jump) 120 steps
#   attack  P1 right + random button, 450 steps
# Then prints:
#   - the openai/retro addresses (health, matches_won for both players)
#     at addr and addr ^ 1 (retro stores Genesis RAM byte-swapped)
#   - X candidates: 16-bit words that go up while walking right and down
#     while walking left
#   - Y candidates: words that move during jumps and stay put while walking
#   - timer candidates: bytes that only decrease, about once a second
# Dumps are saved to /tmp/rlt_mame/ram_search.rds.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: mame_ram_search.R <mame> <cart> [show|-] [start.sta]")
mame  <- args[1]
cart  <- args[2]
show  <- length(args) >= 3 && args[3] == "show"
start <- if (length(args) >= 4) args[4] else "/mnt/Data2/ROMs/states/genesis/quick.sta"
stopifnot(file.exists(start))
k        <- 4L
ram_addr <- 0xFF0000
ram_n    <- 65536L
obs_size <- 4L + ram_n

lua <- "inst/examples/mame_fifo_probe.lua"
lua <- if (file.exists(lua)) normalizePath(lua) else
  system.file("examples", "mame_fifo_probe.lua", package = "rltoolsR")
root <- "/tmp/rlt_mame"
dir <- file.path(root, "search")
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

# ---- start MAME -----------------------------------------------------------
act <- file.path(dir, "act"); obs <- file.path(dir, "obs")
stopifnot(system2("mkfifo", c(act, obs)) == 0)
env <- c(sprintf("RLT_ACT=%s", act), sprintf("RLT_OBS=%s", obs),
         sprintf("RLT_K=%d", k), sprintf("RLT_RAM_ADDR=%d", ram_addr),
         sprintf("RLT_RAM_N=%d", ram_n), "RLT_START=1",
         "RLT_HOLD=2,5",                    # pad B = block, Y = run in MK3: held all k frames
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

# Runs actions from the snapshot; returns a steps x 65536 raw matrix.
run <- function(actions) {
  first <- send(2L)                       # load snapshot
  m <- matrix(as.raw(0), length(actions) + 1L, ram_n)
  m[1, ] <- first
  for (s in seq_along(actions)) m[s + 1L, ] <- send(0L, actions[s])
  m
}

invisible(read_exact(obs_size))           # first observation, after the start load
RIGHT <- 3L * 7L; LEFT <- 7L * 7L; UP <- 1L * 7L
set.seed(1)
acts <- list(
  idle   = rep(0L, 450),
  walk   = rep(c(RIGHT, LEFT, RIGHT), each = 60),
  jump   = rep(UP, 120),
  attack = RIGHT + sample.int(6L, 450, TRUE)
)
dumps <- lapply(acts, run)
writeBin(as.raw(c(3L, 0L, 0L, 0L)), act_con); flush(act_con)   # quit, no answer
close(act_con); close(obs_con)
saveRDS(list(acts = acts, dumps = dumps), file.path(root, "ram_search.rds"))
cat("dumps saved:", file.path(root, "ram_search.rds"), "\n\n")

# ---- analysis -------------------------------------------------------------
u8  <- function(m, off) as.integer(m[, off + 1L])
u16 <- function(m, off) 256L * as.integer(m[, off + 1L]) + as.integer(m[, off + 2L])  # big-endian
hex <- function(off) sprintf("0x%06X", ram_addr + off)
rle_str <- function(v) { r <- rle(v); paste(sprintf("%d(x%d)", r$values, r$lengths), collapse = " ") }

cat("== openai/retro addresses, value runs over 'attack' (value(xsteps)) ==\n")
retro <- c(health = 0xFFF865, matches_won = 0xFFF86B,
           enemy_health = 0xFFF9D7, enemy_matches_won = 0xFFF9DD)
for (nm in names(retro)) for (sw in c(0L, 1L)) {
  off <- bitwXor(retro[[nm]] - ram_addr, sw)
  cat(sprintf("%-18s %s %s: %s\n", nm, hex(off), if (sw) "(^1)" else "    ",
              substr(rle_str(u8(dumps$attack, off)), 1, 150)))
}

# X: word changes correlate with walking direction (+1 right, -1 left).
walk <- dumps$walk
dir_sign <- c(rep(1, 60), rep(-1, 60), rep(1, 60))
offs <- seq(0L, ram_n - 2L, by = 2L)
score_x <- vapply(offs, function(o) {
  d <- diff(u16(walk, o))
  d[abs(d) > 1000] <- 0                   # wrap-around / unrelated jumps
  if (all(d == 0)) return(0)
  sum(sign(d) * dir_sign) / length(d)
}, numeric(1))
top <- order(-abs(score_x))[1:12]
cat("\n== X candidates (score: share of steps moving with the stick) ==\n")
for (i in top) {
  v <- u16(walk, offs[i])
  cat(sprintf("%s score %+.2f  start %d  after right %d  after left %d  end %d\n",
              hex(offs[i]), score_x[i], v[1], v[61], v[121], v[181]))
}

# Y: words that change a lot while jumping and barely while walking.
jump <- dumps$jump
score_y <- vapply(offs, function(o) {
  cj <- sum(diff(u16(jump, o)) != 0); cw <- sum(diff(u16(walk, o)) != 0)
  if (cj < 20) return(0)
  cj / length(jump[, 1]) - cw / length(walk[, 1])
}, numeric(1))
top <- order(-score_y)[1:12]
cat("\n== Y candidates (score: change rate jumping minus walking) ==\n")
for (i in top) {
  v <- u16(jump, offs[i])
  cat(sprintf("%s score %.2f  min %d  max %d  first: %s\n", hex(offs[i]), score_y[i],
              min(v), max(v), paste(head(v, 12), collapse = " ")))
}

# Timer: bytes that never increase over 'idle' and change about every
# 60 / k steps (once a game second; the timer may be BCD).
idle <- dumps$idle
n_sec <- (nrow(idle) - 1) * k / 60
score_t <- vapply(0:(ram_n - 1L), function(o) {
  d <- diff(u8(idle, o))
  if (any(d > 0)) return(Inf)
  abs(sum(d != 0) - n_sec)
}, numeric(1))
top <- order(score_t)[1:8]
cat(sprintf("\n== timer candidates (expect ~%.0f decrements over 'idle') ==\n", n_sec))
for (i in top) {
  v <- u8(idle, i - 1L)
  cat(sprintf("%s changes %d  values %s\n", hex(i - 1L), sum(diff(v) != 0),
              paste(sprintf("%02X", unique(v)), collapse = " ")))
}
