# MK3 (Genesis) RAM: "in the air" flag, fighter state, arena / ground level,
# fighter id candidates.
#
#   Rscript inst/examples/mame_ram_state.R <mame> <cart> [show|-] [start.sta] [random_steps]
#
# One MAME process (mame_fifo_probe.lua, relative directions) sends all
# 64 KB of work RAM every step (k = 4 frames). Scenarios from the snapshot:
#   idle 150, walk (fwd 60, back 60, fwd 60), jump (up 120), crouch (down
#   120), block (B held 120), random (forward / down + button / button /
#   up-forward; reload on KO) random_steps (default 3000)
# Prints:
#   1. air flag candidates: bytes != their ground value exactly when P1 is
#      in the air (Y < 103), balanced accuracy over idle/walk/jump/crouch/block
#   2. state candidates: bytes of the fighter objects (0xFFCB00 + 2 x 0x4C)
#      whose usual value differs between scenarios
#   3. arena change: steps where Y of both fighters jumps by > 150 (not a
#      reload); bytes constant before, constant after, different
#   4. fighter id candidates: bytes constant over all scenarios, 0..20, P1
#      differs from P2 (object +0x4C, record +0x172)
# Dumps: /tmp/rlt_mame/ram_state.rds.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: mame_ram_state.R <mame> <cart> [show|-] [start.sta] [random_steps]")
mame  <- args[1]
cart  <- args[2]
show  <- length(args) >= 3 && args[3] == "show"
start <- if (length(args) >= 4 && args[4] != "-") args[4] else "/mnt/Data2/ROMs/states/genesis/quick.sta"
n_rand <- if (length(args) >= 5) as.integer(args[5]) else 3000L
stopifnot(file.exists(start))
k        <- 4L
ram_addr <- 0xFF0000
ram_n    <- 65536L
obs_size <- 4L + ram_n
O_HP1 <- 0xF865; O_HP2 <- 0xF9D7; O_Y1 <- 0xCB20; O_Y2 <- 0xCB6C

lua <- "inst/examples/mame_fifo_probe.lua"
lua <- if (file.exists(lua)) normalizePath(lua) else
  system.file("examples", "mame_fifo_probe.lua", package = "rltoolsR")
root <- "/tmp/rlt_mame"
dir <- file.path(root, "state")
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
         sprintf("RLT_RAM_N=%d", ram_n), "RLT_START=1", "RLT_HOLD=2,5",
         "RLT_RELATIVE=1", sprintf("RLT_X1=%d", 0xFFCB1C), sprintf("RLT_X2=%d", 0xFFCB68),
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

# Runs actions from the snapshot (reloading on KO); returns a list with a
# steps x 65536 raw matrix and `reload` (TRUE where the row follows a load).
run <- function(actions) {
  m <- matrix(as.raw(0), length(actions) + 1L, ram_n)
  reload <- logical(length(actions) + 1L)
  m[1, ] <- send(2L); reload[1] <- TRUE
  for (s in seq_along(actions)) {
    prev <- m[s, ]
    ko <- as.integer(prev[O_HP1 + 1L]) == 0L || as.integer(prev[O_HP2 + 1L]) == 0L
    if (ko) { m[s + 1L, ] <- send(2L); reload[s + 1L] <- TRUE }
    else m[s + 1L, ] <- send(0L, actions[s])
  }
  list(m = m, reload = reload)
}

invisible(read_exact(obs_size))           # first observation, after the start load
FWD <- 3L * 7L; BACK <- 7L * 7L; UP <- 1L * 7L; DOWN <- 5L * 7L; UPF <- 2L * 7L
set.seed(1)
rand <- function(n) {
  kind <- sample.int(4L, n, TRUE, prob = c(0.35, 0.35, 0.2, 0.1))
  b <- sample.int(6L, n, TRUE)
  ifelse(kind == 1L, FWD, ifelse(kind == 2L, DOWN + b, ifelse(kind == 3L, b, UPF)))
}
acts <- list(
  idle   = rep(0L, 150),
  walk   = rep(c(FWD, BACK, FWD), each = 60),
  jump   = rep(UP, 120),
  crouch = rep(DOWN, 120),
  block  = rep(2L, 120),
  random = rand(n_rand)
)
t0 <- Sys.time()
dumps <- lapply(acts, run)
cat(sprintf("dumps: %.0f s\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
writeBin(as.raw(c(3L, 0L, 0L, 0L)), act_con); flush(act_con)   # quit, no answer
close(act_con); close(obs_con)
saveRDS(list(acts = acts, dumps = dumps), file.path(root, "ram_state.rds"))
cat("saved:", file.path(root, "ram_state.rds"), "\n")

# ---- helpers --------------------------------------------------------------
u8  <- function(m, off) as.integer(m[, off + 1L])
u16 <- function(m, off) 256L * u8(m, off) + u8(m, off + 1L)
s16 <- function(m, off) { v <- u16(m, off); ifelse(v >= 32768L, v - 65536L, v) }
hex <- function(off) sprintf("0x%06X", ram_addr + off)
vals <- function(v, n = 8) paste(head(sort(unique(v)), n), collapse = " ")
mode_of <- function(v) { t <- tabulate(v + 1L, 256L); which.max(t) - 1L }

# ---- 1. air flag ----------------------------------------------------------
cal <- c("idle", "walk", "jump", "crouch", "block")
M <- do.call(rbind, lapply(dumps[cal], `[[`, "m"))
air <- s16(M, O_Y1) < 103L
cat(sprintf("\n== 1. air flag (P1 in the air in %d of %d steps) ==\n", sum(air), length(air)))
bal <- vapply(0:(ram_n - 1L), function(o) {
  v <- u8(M, o)
  v0 <- mode_of(v[!air])
  p <- v != v0
  (mean(p[air]) + mean(!p[!air])) / 2
}, numeric(1))
top <- order(-bal)[1:15]
for (i in top) {
  o <- i - 1L; v <- u8(M, o)
  cat(sprintf("%s bal.acc %.3f  ground %d  air values %s\n", hex(o), bal[i],
              mode_of(v[!air]), vals(v[air])))
}

# ---- 2. fighter state -----------------------------------------------------
cat("\n== 2. fighter object bytes, usual value per scenario (differing only) ==\n")
obj <- 0xCB00:(0xCB00 + 2L * 0x4C - 1L)
tab <- t(vapply(obj, function(o) vapply(cal, function(s) mode_of(u8(dumps[[s]]$m, o)), integer(1)),
                integer(length(cal))))
nd_rand <- vapply(obj, function(o) length(unique(u8(dumps$random$m, o))), integer(1))
keep <- apply(tab, 1, function(r) length(unique(r)) > 1)
out <- data.frame(addr = hex(obj), who = ifelse(obj < 0xCB4C, "P1", "P2"),
                  rel = sprintf("+0x%02X", (obj - 0xCB00) %% 0x4C), tab, n_random = nd_rand)
print(out[keep, ], row.names = FALSE)

# ---- 3. arena change ------------------------------------------------------
cat("\n== 3. arena change ==\n")
R <- dumps$random$m; rl <- dumps$random$reload
y1 <- s16(R, O_Y1); y2 <- s16(R, O_Y2)
tr <- which(abs(diff(y1)) > 150 & abs(diff(y2)) > 150 & !rl[-1]) + 1L
cat("transitions at rows:", if (length(tr)) tr else "none", "\n")
ep <- cumsum(rl)
for (t in tr) {
  w <- max(1L, t - 8L):min(nrow(R), t + 4L)
  cat(sprintf("\nrow %d: y1 %s\n         y2 %s\n", t, paste(y1[w], collapse = " "),
              paste(y2[w], collapse = " ")))
  before <- max(1L, t - 60L):(t - 6L); after <- t:min(nrow(R), t + 60L)
  before <- before[ep[before] == ep[t]]; after <- after[ep[after] == ep[t]]
  if (length(before) < 10 || length(after) < 10) { cat("  window too short\n"); next }
  Rb <- R[before, , drop = FALSE]; Ra <- R[after, , drop = FALSE]
  cand <- which(vapply(0:(ram_n - 1L), function(o) {
    b <- u8(Rb, o); a <- u8(Ra, o)
    all(b == b[1]) && all(a == a[1]) && a[1] != b[1]
  }, logical(1))) - 1L
  cat(sprintf("  %d bytes constant before (%d rows), constant after (%d rows), changed:\n",
              length(cand), length(before), length(after)))
  for (o in head(cand, 40))
    cat(sprintf("  %s  %d -> %d\n", hex(o), u8(Rb, o)[1], u8(Ra, o)[1]))
  y_new <- y1[after[length(after)]]
  g <- which(vapply(seq(0L, ram_n - 2L, 2L), function(o) {
    b <- s16(Rb, o); a <- s16(Ra, o)
    all(b == 103L) && all(abs(a - y_new) <= 30L)
  }, logical(1)))
  cat("  words == 103 before and near the new ground after:",
      if (length(g)) hex(2L * (g - 1L)) else "none", "\n")
}

# ---- 4. fighter id --------------------------------------------------------
cat("\n== 4. fighter id candidates (constant everywhere, 0..20, P1 != P2) ==\n")
A <- do.call(rbind, lapply(dumps, `[[`, "m"))
pairs <- rbind(cbind(0xCB00:(0xCB00 + 0x4B), 0x4C), cbind(0xF700:(0xF9FF - 0x172), 0x172))
for (j in seq_len(nrow(pairs))) {
  o1 <- pairs[j, 1]; o2 <- o1 + pairs[j, 2]
  v1 <- u8(A, o1); v2 <- u8(A, o2)
  if (all(v1 == v1[1]) && all(v2 == v2[1]) && v1[1] <= 20 && v2[1] <= 20 && v1[1] != v2[1])
    cat(sprintf("P1 %s = %d   P2 %s = %d\n", hex(o1), v1[1], hex(o2), v2[1]))
}
