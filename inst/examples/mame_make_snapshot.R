# Two-player (VS) snapshot of MK3 (Genesis) at the start of round 1, made
# by MAME from power-on (inst/mame/mk3_snapshot.lua).
#
#   Rscript inst/examples/mame_make_snapshot.R p1=sonya p2=kabal [show=0] [out=dir]
#   Rscript inst/examples/mame_make_snapshot.R p1=sonya mode=1p [show=0] [out=dir]
#
# mode=1p: P1 against the CPU, tower 1; the opponent is chosen by the game,
# the file is <p1>_<opponent>_cpu.sta.
#
# Fighters by name or id 0..13: kano 0, sonya 1, jax 2, nightwolf 3,
# subzero 4, stryker 5, sindel 6, sektor 7, cyrax 8, kunglao 9,
# kabal 10, sheeva 11, shangtsung 12, liukang 13 (Smoke, 14, is hidden).
# Writes <out>/<p1>_<p2>.sta (default out /mnt/Data2/ROMs/states/genesis)
# after checking the fighter ids in the saved round. show=1: MAME in a
# window, fast until the save, then normal speed; close it with Esc.

ids <- c(kano = 0, sonya = 1, jax = 2, nightwolf = 3, subzero = 4, stryker = 5,
         sindel = 6, sektor = 7, cyrax = 8, kunglao = 9, kabal = 10, sheeva = 11,
         shangtsung = 12, liukang = 13)
# Select screen cell (grid 5 x 3, cell 7 unused) of each fighter id 0..13;
# Smoke (14) is hidden, not on the grid.
cell_of <- c(3L, 5L, 2L, 11L, 8L, 6L, 1L, 10L, 9L, 13L, 14L, 12L, 0L, 4L)
opt <- list(p1 = NULL, p2 = NULL, mode = "vs", show = "0", out = "/mnt/Data2/ROMs/states/genesis")
for (a in commandArgs(TRUE)) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (!kv[1] %in% names(opt)) stop("unknown key: ", kv[1])
  opt[[kv[1]]] <- kv[2]
}
fighter <- function(x) {
  if (is.null(x)) stop("p1= and p2= are required")
  if (grepl("^[0-9]+$", x)) { i <- as.integer(x); nm <- names(ids)[ids == i] }
  else { i <- ids[[match.arg(x, names(ids))]]; nm <- names(ids)[ids == i] }
  if (!length(nm) || i > 13) stop("unknown fighter (0..13): ", x)
  list(id = as.integer(i), name = nm)
}
one <- match.arg(opt$mode, c("vs", "1p")) == "1p"
f1 <- fighter(opt$p1)
f2 <- if (one) list(id = NA_integer_, name = "cpu") else fighter(opt$p2)
show <- opt$show == "1"
mame <- "/usr/games/mame"
cart <- "/mnt/Data2/ROMs/Mortal Kombat 3 (USA).zip"

root <- "/tmp/rlt_mame"
dir <- file.path(root, "snapshot")
unlink(dir, recursive = TRUE)
dir.create(dir, recursive = TRUE)
inner <- unzip(cart, list = TRUE)$Name
inner <- inner[grepl("\\.(gen|md|bin|smd)$", inner, ignore.case = TRUE)][1]
unzip(cart, files = inner, exdir = root, junkpaths = TRUE)
gen <- normalizePath(file.path(root, basename(inner)))

lua <- "inst/mame/mk3_snapshot.lua"
lua <- if (file.exists(lua)) normalizePath(lua) else system.file("mame", "mk3_snapshot.lua", package = "rltoolsR")
env <- c(sprintf("RLT_P1=%d", cell_of[f1$id + 1L]),
         if (one) "RLT_MODE=1p" else sprintf("RLT_P2=%d", cell_of[f2$id + 1L]),
         "RLT_SAVE=snap",
         if (show) "RLT_KEEP=1" else "SDL_VIDEODRIVER=dummy")
log <- file.path(dir, "mame.log")
cat(sprintf("making %s (%d) vs %s...\n", f1$name, f1$id,
            if (one) "CPU, tower 1" else sprintf("%s (%d)", f2$name, f2$id)))
system2(mame, c("genesis", "-cart", shQuote(gen), "-ctrl1", "md6button", "-ctrl2", "md6button",
                if (show) "-window" else c("-video", "none", "-nothrottle"),
                "-sound", "none", "-skip_gameinfo",
                "-state_directory", dir, "-cfg_directory", dir, "-nvram_directory", dir,
                "-autoboot_script", lua),
        env = env, stdout = log, stderr = log)
lines <- readLines(log, warn = FALSE)
hit <- grep("^rlt_snap: saved", lines, value = TRUE)
if (!length(hit)) stop("no snapshot; MAME log:\n", paste(lines, collapse = "\n"))
got <- strsplit(regmatches(hit, regexpr("id1 [0-9]+ id2 [0-9]+", hit)), " ")[[1]]
got <- as.integer(got[c(2, 4)])
if (one) {
  if (!got[2] %in% ids) stop("unknown CPU fighter id ", got[2])
  f2 <- list(id = got[2], name = names(ids)[ids == got[2]])
  cat(sprintf("CPU opponent: %s (%d)\n", f2$name, f2$id))
}
if (!identical(got, c(f1$id, f2$id)))
  stop(sprintf("wrong fighters in the snapshot: id1 %d id2 %d, wanted %d %d", got[1], got[2], f1$id, f2$id))
dest <- file.path(opt$out, sprintf("%s_%s%s.sta", f1$name, f2$name, if (one) "_cpu" else ""))
if (file.exists(dest)) stop(dest, " exists; the new snapshot is in ", file.path(dir, "genesis", "snap.sta"))
stopifnot(file.copy(file.path(dir, "genesis", "snap.sta"), dest))
cat("saved", dest, "\n")
