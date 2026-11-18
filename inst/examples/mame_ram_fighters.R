# MK3 (Genesis) RAM: which bytes hold the fighter id of P1 and of P2?
#
#   Rscript inst/examples/mame_ram_fighters.R <mame> <cart> [dir] [show]
#
# Snapshots <dir>/<p1>_<p2>.sta (default dir /mnt/Data2/ROMs/states/genesis),
# fighter names taken from the file name, e.g. subzero_sonya.sta. One MAME
# process (mame_fifo_probe.lua); for each snapshot: copy it over "rlt",
# load, 20 idle steps, all 64 KB of work RAM each step.
# A byte is a candidate for player p when it is constant within every
# snapshot and equal across snapshots exactly when p's fighter is the same.
# Dumps: /tmp/rlt_mame/ram_fighters.rds.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: mame_ram_fighters.R <mame> <cart> [dir] [show]")
mame <- args[1]
cart <- args[2]
sdir <- if (length(args) >= 3) args[3] else "/mnt/Data2/ROMs/states/genesis"
show <- length(args) >= 4 && args[4] == "show"
files <- list.files(sdir, "^[[:alnum:]]+_[[:alnum:]]+\\.sta$", full.names = TRUE)
if (length(files) < 2) stop("need at least 2 snapshots <p1>_<p2>.sta in ", sdir)
who <- do.call(rbind, strsplit(sub("\\.sta$", "", basename(files)), "_", fixed = TRUE))
colnames(who) <- c("P1", "P2")
print(data.frame(file = basename(files), who), row.names = FALSE)

k <- 4L; ram_addr <- 0xFF0000; ram_n <- 65536L; obs_size <- 4L + ram_n; n_idle <- 20L
lua <- "inst/examples/mame_fifo_probe.lua"
lua <- if (file.exists(lua)) normalizePath(lua) else
  system.file("examples", "mame_fifo_probe.lua", package = "rltoolsR")
root <- "/tmp/rlt_mame"
dir <- file.path(root, "fighters")
unlink(dir, recursive = TRUE)
dir.create(file.path(dir, "genesis"), recursive = TRUE)
rlt <- file.path(dir, "genesis", "rlt.sta")
stopifnot(file.copy(files[1], rlt))

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
send <- function(cmd) {
  writeBin(as.raw(c(cmd, 0L, 0L, 0L)), act_con); flush(act_con)
  read_exact(obs_size)[-(1:4)]
}

invisible(read_exact(obs_size))
dumps <- lapply(files, function(f) {
  stopifnot(file.copy(f, rlt, overwrite = TRUE))   # read by MAME at load time
  m <- matrix(as.raw(0), n_idle + 1L, ram_n)
  m[1, ] <- send(2L)
  for (s in seq_len(n_idle)) m[s + 1L, ] <- send(0L)
  m
})
writeBin(as.raw(c(3L, 0L, 0L, 0L)), act_con); flush(act_con)
close(act_con); close(obs_con)
names(dumps) <- basename(files)
saveRDS(list(who = who, dumps = dumps), file.path(root, "ram_fighters.rds"))
cat("saved:", file.path(root, "ram_fighters.rds"), "\n")

# HP check: full HP after every load means the snapshot loaded
hp <- vapply(dumps, function(m) as.integer(m[n_idle + 1L, c(0xF865, 0xF9D7) + 1L]), integer(2))
cat("\nHP after load (P1, P2):\n"); print(t(hp))

# Per snapshot value of each byte (NA where it changes during the idle steps)
V <- t(vapply(dumps, function(m) {
  v <- as.integer(m[2, ])
  same <- colSums(m[-1, , drop = FALSE] != m[rep(2L, n_idle), , drop = FALSE]) == 0
  ifelse(same, v, NA_integer_)
}, integer(ram_n)))
hex <- function(off) sprintf("0x%06X", ram_addr + off)
ok_p <- list()
for (p in c("P1", "P2")) {
  nm <- who[, p]
  if (length(unique(nm)) < 2) { cat("\n", p, ": the same fighter in all snapshots, skipped\n"); next }
  same_nm <- outer(nm, nm, `==`)
  ok <- vapply(seq_len(ram_n), function(j) {
    v <- unname(V[, j])
    !anyNA(v) && all(outer(v, v, `==`) == same_nm)
  }, logical(1))
  ok_p[[p]] <- ok
  cand <- which(ok) - 1L
  cat(sprintf("\n== %s: %d candidates (fighters: %s) ==\n", p, length(cand),
              paste(unique(nm), collapse = ", ")))
  for (o in head(cand, 60)) {
    v <- V[, o + 1L]; u <- !duplicated(nm)
    cat(sprintf("%s  %s\n", hex(o), paste(sprintf("%s=%d", nm[u], v[u]), collapse = "  ")))
  }
  obj <- cand[cand >= 0xCB00 & cand <= 0xCB97]
  cat(sprintf("-- %s candidates in the fighter objects 0xFFCB00..0xFFCB97: %d\n", p, length(obj)))
  for (o in obj) {
    v <- V[, o + 1L]; u <- !duplicated(nm)
    cat(sprintf("%s  %s +0x%02X  %s\n", hex(o), if (o < 0xCB4C) "P1" else "P2", (o - 0xCB00) %% 0x4C,
                paste(sprintf("%s=%d", nm[u], v[u]), collapse = "  ")))
  }
}

# ---- narrowing --------------------------------------------------------------
# Value of byte `o` per snapshot, named by fighter (first snapshot per fighter)
show_byte <- function(o, nm) {
  v <- V[, o + 1L]; u <- !duplicated(nm)
  paste(sprintf("%s=%d", nm[u], v[u]), collapse = "  ")
}
p2_const <- function(o) { v <- V[, o + 1L]; !anyNA(v) && all(v == v[1]) }

# 1. P1 candidates in the fighter object / record whose P2 pair (+0x4C /
#    +0x172) is the same in all snapshots and differs from every P1 value;
#    all values 0..30. Only when P2 is one fighter in all snapshots.
cat("\n== filter 1: fighter object / record, P2 pair constant ==\n")
if (!is.null(ok_p$P1) && length(unique(who[, "P2"])) == 1L) {
  regions <- rbind(cbind(0xCB00:(0xCB00 + 0x4B), 0x4C), cbind(0xF700:(0xF9FF - 0x172), 0x172))
  n1 <- 0L
  for (j in seq_len(nrow(regions))) {
    o1 <- regions[j, 1]; o2 <- o1 + regions[j, 2]
    if (!ok_p$P1[o1 + 1L] || !p2_const(o2)) next
    v1 <- V[, o1 + 1L]; v2 <- V[1, o2 + 1L]
    if (max(v1, v2) > 30 || v2 %in% v1) next
    n1 <- n1 + 1L
    cat(sprintf("P1 %s  %s   P2 %s  %s=%d\n", hex(o1), show_byte(o1, who[, "P1"]),
                hex(o2), who[1, "P2"], v2))
  }
  if (n1 == 0L) cat("none\n")
} else cat("needs several P1 fighters and one P2 fighter\n")

# 2. Arcade MK3 numbering (hypothesis): kano 0, sonya 1, jax 2, nightwolf 3,
#    subzero 4, ..., shangtsung 12; also x2 and x4 (word / long table index).
cat("\n== filter 2: arcade MK3 fighter numbers (x1, x2, x4) ==\n")
arcade <- c(kano = 0, sonya = 1, jax = 2, nightwolf = 3, subzero = 4, stryker = 5,
            sindel = 6, kunglao = 9, kabal = 10, sheeva = 11, shangtsung = 12,
            liukang = 13, smoke = 14)
if (all(who %in% names(arcade))) {
  for (mult in c(1L, 2L, 4L)) {
    e1 <- arcade[who[, "P1"]] * mult; e2 <- arcade[who[, "P2"]] * mult
    hit1 <- which(colSums(V == e1, na.rm = FALSE) == nrow(V)) - 1L
    hit1 <- hit1[!is.na(hit1)]
    cat(sprintf("x%d: %d bytes match P1\n", mult, length(hit1)))
    for (o in head(hit1, 30)) {
      pr <- vapply(c(0x4C, 0x172), function(d) {
        o2 <- o + d
        if (o2 >= ram_n) return("")
        v <- V[, o2 + 1L]
        if (!anyNA(v) && all(v == e2)) sprintf("  P2 pair +0x%X matches", d) else ""
      }, character(1))
      cat(sprintf("  %s  %s%s\n", hex(o), show_byte(o, who[, "P1"]), paste(pr, collapse = "")))
    }
  }
} else cat("fighter names outside the table: ", paste(setdiff(who, names(arcade)), collapse = ", "), "\n")
