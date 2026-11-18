# Checks on the dumps of mame_ram_state.R (no MAME):
#   1. fighter id: runs of 0xFFCB4B / 0xFFCB97 and the select-screen copies
#      0xFF0127 / 0xFF0129 / 0xFF9F19 per scenario; every change of P2 id
#      with P2 animation (+0x45) around it
#   2. air flag: bit 7 of +0x12 (and +0x22) vs Y, for P1 and P2
#
#   Rscript inst/examples/mame_ram_state_check.R [/tmp/rlt_mame/ram_state.rds]

args <- commandArgs(trailingOnly = TRUE)
f <- if (length(args)) args[1] else "/tmp/rlt_mame/ram_state.rds"
d <- readRDS(f)
u8  <- function(m, a) as.integer(m[, a - 0xFF0000 + 1L])
s16 <- function(m, a) { v <- 256L * u8(m, a) + u8(m, a + 1L); ifelse(v >= 32768L, v - 65536L, v) }
rle_str <- function(v, n = 25) {
  r <- rle(v)
  s <- paste(sprintf("%d(x%d)", head(r$values, n), head(r$lengths, n)), collapse = " ")
  if (length(r$values) > n) paste(s, "...") else s
}

# ---- 1. fighter id ------------------------------------------------------------
cat("== 1. fighter id, value runs value(xsteps) ==\n")
ids <- c(P1_obj = 0xFFCB4B, P2_obj = 0xFFCB97, P1_sel = 0xFF0127, P2_sel = 0xFF0129,
         P1_9F19 = 0xFF9F19)
for (s in names(d$dumps)) {
  m <- d$dumps[[s]]$m
  cat(sprintf("-- %s (%d steps)\n", s, nrow(m)))
  for (nm in names(ids)) cat(sprintf("  %-8s %s\n", nm, rle_str(u8(m, ids[[nm]]))))
}

R <- d$dumps$random$m; rl <- d$dumps$random$reload
id2 <- u8(R, 0xFFCB97); an2 <- u8(R, 0xFFCB91); sel2 <- u8(R, 0xFF0129)
ch <- which(diff(id2) != 0 & !rl[-1]) + 1L
cat(sprintf("\nP2 id changes in 'random' (not at reload): %d\n", length(ch)))
for (t in head(ch, 20)) {
  w <- max(1L, t - 3L):min(nrow(R), t + 3L)
  cat(sprintf("  row %d: id %d -> %d | id %s | anim %s | sel %s\n", t, id2[t - 1L], id2[t],
              paste(id2[w], collapse = " "), paste(an2[w], collapse = " "),
              paste(sel2[w], collapse = " ")))
}

# ---- 2. air flag --------------------------------------------------------------
cat("\n== 2. air flag: bit 7 of +0x12 / +0x22 vs Y (all scenarios) ==\n")
A <- do.call(rbind, lapply(d$dumps, `[[`, "m"))
ybin <- function(y) cut(y, c(-Inf, -1, 46, 47, 102, 103, Inf),
                        labels = c("<0", "0..46", "47", "48..102", "103", ">103"))
for (p in 1:2) {
  base <- 0xFFCB00 + (p - 1L) * 0x4C
  y <- s16(A, base + 0x20)
  for (off in c(0x12, 0x22)) {
    b7 <- bitwAnd(u8(A, base + off), 128L) > 0
    cat(sprintf("\nP%d +0x%02X (0x%06X) bit 7 by Y:\n", p, off, base + off))
    print(table(Y = ybin(y), bit7 = b7))
  }
}
