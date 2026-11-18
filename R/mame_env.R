# Environments emulated by MAME: n MAME processes, each driven in lockstep
# by a Lua script over two FIFOs (inst/mame/<game>.lua). Same contract as
# rl_vec_env: batched reset / step, same-step reset of finished games and
# final_obs. Linux / macOS only (mkfifo).

#' Mortal Kombat 3 under MAME
#'
#' Runs `n` MAME processes with the Sega Genesis cartridge of Mortal
#' Kombat 3, each starting every round from the snapshot `start`, or, in
#' tower mode (`start = NULL`, `fighter` given), playing the game on: tower
#' 1 against the CPU from the select screen, no snapshots. One step is
#' `frameskip` emulated frames.
#'
#' Actions: `Discrete(63)`, `a = 7 * direction + button`. Direction, relative
#' to the opponent: 0 none, 1 up, 2 up-forward, 3 forward, 4 down-forward,
#' 5 down, 6 down-back, 7 back, 8 up-back. Button: 0 none, 1..6 = pad A, B,
#' C, X, Y, Z. B (block) and Y (run) are held for the whole step, the other
#' buttons are pressed for one frame. Special moves are sequences of steps.
#'
#' Observation (9 numbers, from the row's own side): own HP / 166,
#' opponent HP / 166, (opponent X - own X) / 320, (own X - 500) / 500,
#' own height above the ground / 100, (own Y - opponent Y) / 200 (positive:
#' the opponent is higher), timer / 99, own and opponent "in the air" flag
#' (0 / 1). Heights are clipped to -2..2. The ground is 103 at the start of
#' a round and moves to the Y where a fighter has stood still on the ground
#' for about 64 frames (an uppercut through the floor moves the fight to
#' another level).
#'
#' Reward per step: (damage dealt - damage taken) / 166, plus 1 for winning
#' and -1 for losing the round. An episode is one round: `terminated` on a
#' knockout or when the timer reaches 0 (the fighter with more HP wins). The
#' finished copy is reset from the snapshot in the same step; the input lock
#' at the start of a round ("Fight!") is skipped.
#'
#' Tower mode: MAME boots, passes the 1P menu, selects `fighter`, picks
#' tower 1 and starts the first fight on its own (a few seconds); then every
#' step is played, between fights too (victory poses, "Finish him", tower
#' screen). `terminated` once at the end of each round, the next episode
#' starts at once, nothing is loaded; [rl_reset()] returns the current
#' observation. With `retry = "checkpoint"` the state is saved at the start
#' of every fight (round 1) and loaded after a lost match: the same opponent
#' again until the match is won, then the game goes on to the next one. With
#' `retry = "continue"` the script presses Start (Continue), selects
#' `fighter` again and starts the next fight. When the round timer
#' has not changed for `watchdog_frames` (end of the tower, game over) the
#' console is reset and boots again; with `"checkpoint"`, after a won match
#' (the tower is done) it picks the next tower, after the last of `n_towers`
#' tower 1 again. Meanwhile the process keeps answering
#' every step, paused: the observation and the state stay as they were,
#' reward 0, the action is ignored, and [rl_step()]'s `active` is `FALSE`
#' for its rows ([rl_ppo()] does not train on such steps).
#'
#' @param game Game; only `"mk3"`.
#' @param n Number of MAME processes.
#' @param cart Cartridge: a `.gen` file or a `.zip` holding one.
#' @param start Snapshot (`.sta`) to start every round from: made in MAME
#'   with the same cartridge and `-ctrl1 md6button -ctrl2 md6button`.
#'   `NULL`: tower mode, needs `fighter`.
#' @param fighter Tower mode: P1's fighter by name (`"kano"`, `"sonya"`,
#'   `"jax"`, `"nightwolf"`, `"subzero"`, `"stryker"`, `"sindel"`,
#'   `"sektor"`, `"cyrax"`, `"kunglao"`, `"kabal"`, `"sheeva"`,
#'   `"shangtsung"`, `"liukang"`).
#' @param frameskip Emulated frames per step.
#' @param opponent `"cpu"`: P2 is the game's AI, one row per process.
#'   `"policy"`: both fighters are played by the caller, rows `2i - 1` (P1)
#'   and `2i` (P2) for process `i`, rewards mirrored; needs a two-player
#'   snapshot.
#' @param mame Path to the MAME executable.
#' @param show Show process 1 in a window (unthrottled).
#' @param max_steps Optional step limit per round (`truncated`); `NULL` for
#'   none (the round timer ends every round).
#' @param intro_frames Frames run once at start after loading `start` (input
#'   lock at the start of a round); the state after them is saved and loaded
#'   on every reset.
#' @param watchdog_frames Tower mode: frames without a change of the round
#'   timer before the console is reset.
#' @param retry Tower mode, after a lost match: `"checkpoint"` (load the
#'   state saved at the start of the fight) or `"continue"` (the game's
#'   Continue).
#' @param n_towers Tower mode with `retry = "checkpoint"`: number of towers
#'   to go through in turn.
#' @param timeout Seconds to wait for an answer from MAME (in tower mode
#'   also the boot to the first fight).
#' @param work_dir Directory for FIFOs, snapshots and logs; a new temporary
#'   directory by default.
#' @return An object of class `rl_mame_env`; close it with [rl_close()].
#' @export
#' @examples
#' \dontrun{
#' env <- rl_mame_env("mk3", n = 4, cart = "Mortal Kombat 3 (USA).zip",
#'                    start = "states/genesis/sonya_sindel_cpu.sta")
#' obs <- rl_reset(env)
#' r <- rl_step(env, rep(21L, env$n))   # walk forward
#' rl_close(env)
#' }
rl_mame_env <- function(game = "mk3", n = 4L, cart, start = NULL, fighter = NULL,
                        frameskip = 4L, opponent = c("cpu", "policy"), mame = "mame",
                        show = FALSE, max_steps = NULL, intro_frames = 80L,
                        watchdog_frames = 3600L, retry = c("checkpoint", "continue"),
                        n_towers = 4L, timeout = 60, work_dir = NULL) {
  game <- match.arg(game, "mk3")
  opponent <- match.arg(opponent)
  retry <- match.arg(retry)
  n <- as.integer(n); frameskip <- as.integer(frameskip)
  stopifnot(length(n) == 1, n >= 1, length(frameskip) == 1, frameskip >= 1)
  if (!file.exists(cart)) stop("cartridge not found: ", cart, call. = FALSE)
  tower <- is.null(start)
  if (tower) {
    if (is.null(fighter)) stop("give `start` (a snapshot) or `fighter` (tower mode)", call. = FALSE)
    if (opponent != "cpu") stop("tower mode is against the CPU: opponent = \"cpu\"", call. = FALSE)
    fighter <- match.arg(tolower(fighter), names(MK3_FIGHTERS))
  } else {
    if (!is.null(fighter)) stop("`fighter` is for tower mode (start = NULL)", call. = FALSE)
    if (!file.exists(start)) stop("snapshot not found: ", start, call. = FALSE)
  }
  if (!nzchar(Sys.which("mkfifo"))) stop("rl_mame_env needs mkfifo (Linux / macOS)", call. = FALSE)
  mame_bin <- Sys.which(mame)
  if (!nzchar(mame_bin)) stop("MAME not found: ", mame, call. = FALSE)
  lua <- system.file("mame", paste0(game, ".lua"), package = "rltoolsR")
  if (!nzchar(lua)) stop("inst/mame/", game, ".lua not installed", call. = FALSE)

  if (is.null(work_dir)) work_dir <- tempfile("rlt_mame_")
  dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
  work_dir <- normalizePath(work_dir)
  if (grepl("\\.zip$", cart, ignore.case = TRUE)) {
    inner <- utils::unzip(cart, list = TRUE)$Name
    inner <- inner[grepl("\\.(gen|md|bin|smd)$", inner, ignore.case = TRUE)][1]
    if (is.na(inner)) stop("no .gen / .md / .bin in ", cart, call. = FALSE)
    utils::unzip(cart, files = inner, exdir = work_dir, junkpaths = TRUE)
    cart <- file.path(work_dir, basename(inner))
  }
  cart <- normalizePath(cart)

  st <- new.env(parent = emptyenv())
  st$n_proc <- n; st$timeout <- timeout; st$closed <- FALSE
  st$two <- opponent == "policy"
  st$tower <- tower
  st$max_steps <- if (is.null(max_steps)) Inf else as.integer(max_steps)
  st$intro <- ceiling(as.integer(intro_frames) / frameskip)
  st$procs <- vector("list", n)
  for (i in seq_len(n)) {
    dir <- file.path(work_dir, sprintf("p%d", i))
    unlink(dir, recursive = TRUE)
    dir.create(file.path(dir, "genesis"), recursive = TRUE)
    if (!tower) file.copy(start, file.path(dir, "genesis", "rlt.sta"))
    act <- file.path(dir, "act"); obs <- file.path(dir, "obs")
    if (system2("mkfifo", c(shQuote(act), shQuote(obs))) != 0) stop("mkfifo failed in ", dir, call. = FALSE)
    win <- show && i == 1L
    mode_vars <- if (tower) {
      c(sprintf("RLT_TOWER=%d", MK3_CELL[MK3_FIGHTERS[[fighter]] + 1L]),
        sprintf("RLT_WATCHDOG=%d", as.integer(watchdog_frames)),
        sprintf("RLT_RETRY=%s", retry), sprintf("RLT_NTOWERS=%d", as.integer(n_towers)))
    } else {
      c("RLT_START=1", sprintf("RLT_INTRO=%d", st$intro * frameskip))
    }
    env_vars <- c(sprintf("RLT_ACT=%s", shQuote(act)), sprintf("RLT_OBS=%s", shQuote(obs)),
                  sprintf("RLT_K=%d", frameskip), mode_vars,
                  if (!win) "SDL_VIDEODRIVER=dummy")
    system2(mame_bin, c("genesis", "-cart", shQuote(cart),
                        "-ctrl1", "md6button", "-ctrl2", "md6button", "-nothrottle",
                        if (win) "-window" else c("-video", "none"),
                        "-sound", "none", "-skip_gameinfo",
                        "-state_directory", shQuote(dir), "-cfg_directory", shQuote(dir),
                        "-nvram_directory", shQuote(dir), "-autoboot_script", shQuote(lua)),
            env = env_vars, wait = FALSE,
            stdout = file.path(dir, "mame.log"), stderr = file.path(dir, "mame.log"))
    st$procs[[i]] <- list(dir = dir, act = act, obs = obs)
  }
  # Opening blocks until MAME opens its end (same order as the Lua side);
  # if MAME fails to start this waits: see <work_dir>/pN/mame.log.
  for (i in seq_len(n)) {
    st$procs[[i]]$act_con <- fifo(st$procs[[i]]$act, "wb", blocking = TRUE)
    st$procs[[i]]$obs_con <- fifo(st$procs[[i]]$obs, "rb", blocking = TRUE)
  }
  reg.finalizer(st, mame_close, onexit = TRUE)
  if (show && nzchar(Sys.which("wmctrl"))) system2("wmctrl", c("-r", "MAME", "-b", "add,above"))
  # first observation: after the start load, or (tower) when the first fight is on
  st$s <- mame_read(st, seq_len(n))
  st$n_ground <- ceiling(64 / frameskip)
  st$ground <- integer(n); st$ground_updates <- integer(n)
  st$still <- matrix(0L, n, 2L); st$prev_y <- matrix(0L, n, 2L)
  mk3_ground_init(st, seq_len(n), st$s)
  st$steps <- integer(n)
  st$armed <- !mk3_over(st$s)              # tower: a round is going on

  rows <- if (st$two) 2L * n else n
  structure(list(st = st, name = game, n = rows, n_proc = n, frameskip = frameskip,
                 opponent = opponent, work_dir = work_dir,
                 start = if (tower) "tower" else "snapshot", fighter = fighter,
                 mode = if (st$two) "simultaneous" else "single",
                 n_players = if (st$two) 2L else 1L,
                 max_steps = if (is.null(max_steps)) 0L else as.integer(max_steps),
                 observation_space = space_box(rep(-2, 9), rep(2, 9)),
                 action_space = space_discrete(63)),
            class = "rl_mame_env")
}

#' Close an environment
#'
#' Stops the processes behind an environment (for [rl_mame_env()]: the
#' MAME processes; their logs stay in `env$work_dir`). Closing twice is a
#' no-op.
#'
#' @param env An environment.
#' @return `env`, invisibly.
#' @export
rl_close <- function(env) UseMethod("rl_close")

#' @export
rl_close.rl_mame_env <- function(env) {
  mame_close(env$st)
  invisible(env)
}

#' @export
rl_reset.rl_mame_env <- function(env, seed = NULL) {
  st <- env$st
  mame_check_open(st)
  idx <- seq_len(st$n_proc)
  st$steps[] <- 0L
  if (st$tower) return(mk3_obs(st$s, st$ground, st$two))   # the game goes on
  st$s <- mame_restart(st, idx)
  mk3_ground_init(st, idx, st$s)
  mk3_obs(st$s, st$ground, st$two)
}

#' @export
rl_step.rl_mame_env <- function(env, action) {
  st <- env$st
  mame_check_open(st)
  np <- st$n_proc
  a <- as.integer(action)
  if (length(a) != env$n || anyNA(a) || any(a < 0L | a > 62L))
    stop("`action` must be ", env$n, " integers in 0..62", call. = FALSE)
  a1 <- if (st$two) a[seq(1L, 2L * np, 2L)] else a
  a2 <- if (st$two) a[seq(2L, 2L * np, 2L)] else integer(np)
  idx <- seq_len(np)
  mame_send(st, idx, 0L, a1, a2)
  s <- mame_read(st, idx)
  prev <- st$s
  # tower mode, paused (the script plays a menu): the state stays as it was,
  # no reward, no round end; the step is not for training (active FALSE)
  active <- s[, "paused"] == 0L
  if (!all(active)) {
    p <- which(!active)
    s[p, setdiff(mk3_cols, "paused")] <- prev[p, setdiff(mk3_cols, "paused")]
  }
  taken1 <- pmax(prev[, "hp1"] - s[, "hp1"], 0)
  taken2 <- pmax(prev[, "hp2"] - s[, "hp2"], 0)
  ko <- s[, "hp1"] == 0 | s[, "hp2"] == 0
  time_up <- !ko & s[, "timer"] == 0
  over <- ko | time_up
  # tower: the end of a round counts once (the game stays "over" until the
  # next round); a new round re-inits the ground (another arena)
  done <- if (st$tower) st$armed & over else over
  begun <- st$tower & !st$armed & !over
  st$armed <- !over
  st$steps <- st$steps + active
  trunc <- !done & st$steps >= st$max_steps
  win <- (done & s[, "hp1"] > s[, "hp2"]) - (done & s[, "hp2"] > s[, "hp1"])
  r1 <- (taken2 - taken1) / 166 + win
  mk3_ground_update(st, s)
  if (any(begun)) mk3_ground_init(st, which(begun), s[begun, , drop = FALSE])
  final_obs <- mk3_obs(s, st$ground, st$two)
  end <- ifelse(done & ko, "ko", ifelse(done & time_up, "time", ""))
  fin <- which(done | trunc)
  st$steps[fin] <- 0L
  restart <- length(fin) && !st$tower
  if (restart) {
    s[fin, ] <- mame_restart(st, fin)
    mk3_ground_init(st, fin, s[fin, , drop = FALSE])
  }
  st$s <- s
  obs <- if (restart) mk3_obs(s, st$ground, st$two) else final_obs
  # final_obs: the last observation of the rounds that ended, NA for the others
  ended <- done | trunc
  final_obs[!(if (st$two) rep(ended, each = 2L) else ended), ] <- NA_real_
  if (st$two) {
    rep2 <- function(x) rep(x, each = 2L)
    list(obs = obs, reward = as.vector(rbind(r1, -r1)), terminated = rep2(done),
         truncated = rep2(trunc), final_obs = final_obs, end = rep2(end),
         active = rep2(active))
  } else {
    list(obs = obs, reward = r1, terminated = done, truncated = trunc,
         final_obs = final_obs, end = end, active = active)
  }
}

#' @export
rl_legal_actions.rl_mame_env <- function(env) NULL

#' @export
print.rl_mame_env <- function(x, ...) {
  cat("<rl_mame_env> ", x$name, " x ", x$n_proc, " MAME processes (", x$n, " rows)\n", sep = "")
  cat("  opponent:    ", x$opponent, ", frameskip ", x$frameskip, "\n", sep = "")
  cat("  start:       ", if (x$start == "tower") paste0("tower 1, ", x$fighter) else "snapshot",
      "\n", sep = "")
  cat("  observation: ", format(x$observation_space), "\n", sep = "")
  cat("  action:      ", format(x$action_space), "\n", sep = "")
  cat("  work_dir:    ", x$work_dir, if (x$st$closed) " (closed)", "\n", sep = "")
  invisible(x)
}

# ---- internals ---------------------------------------------------------------

mk3_cols <- c("hp1", "hp2", "x1", "x2", "y1", "y2", "timer", "won1", "won2", "air1", "air2",
              "paused")

# Fighter ids (RAM) and their select screen cells (grid 5 x 3, cell 7
# unused; Smoke, id 14, is hidden), as in inst/examples/mame_make_snapshot.R.
MK3_FIGHTERS <- c(kano = 0L, sonya = 1L, jax = 2L, nightwolf = 3L, subzero = 4L,
                  stryker = 5L, sindel = 6L, sektor = 7L, cyrax = 8L, kunglao = 9L,
                  kabal = 10L, sheeva = 11L, shangtsung = 12L, liukang = 13L)
MK3_CELL <- c(3L, 5L, 2L, 11L, 8L, 6L, 1L, 10L, 9L, 13L, 14L, 12L, 0L, 4L)

# Round over: a knockout or the timer at 0.
mk3_over <- function(s) s[, "hp1"] == 0 | s[, "hp2"] == 0 | s[, "timer"] == 0

# Ground level per process (Y grows downwards). 103 at the start of a round;
# an uppercut through the floor moves the fight to another level (e.g. 303).
# A fighter is "still" while not in the air and Y does not change; when a
# fighter has been still for `n_ground` steps (~64 frames) at a Y more than
# MK3_GROUND_TOL from the ground, that Y becomes the ground (the smaller Y
# when both are; a fighter lying down has a larger Y). A fighter flying
# without the air flag (thrown, hit) moves, so it never counts.
MK3_GROUND0 <- 103L
MK3_GROUND_TOL <- 8L

mk3_ground_init <- function(st, idx, s) {
  st$ground[idx] <- MK3_GROUND0
  st$still[idx, ] <- 0L
  st$prev_y[idx, ] <- s[, c("y1", "y2"), drop = FALSE]
}

mk3_ground_update <- function(st, s) {
  y <- s[, c("y1", "y2"), drop = FALSE]
  air <- s[, c("air1", "air2"), drop = FALSE] == 1L
  st$still[] <- ifelse(!air & y == st$prev_y, st$still + 1L, 0L)
  st$prev_y[] <- y
  cand <- ifelse(st$still >= st$n_ground, y, NA_integer_)
  cand <- pmin(cand[, 1], cand[, 2], na.rm = TRUE)
  upd <- which(!is.na(cand) & abs(cand - st$ground) > MK3_GROUND_TOL)
  st$ground[upd] <- cand[upd]
  st$ground_updates[upd] <- st$ground_updates[upd] + 1L
}

# Rows of observations from the raw state (one row per process, P1 view;
# with `two`, P1 and P2 rows interleaved). Heights are clipped to [-2, 2].
mk3_obs <- function(s, ground, two) {
  clip <- function(x) pmin(pmax(x, -2), 2)
  view <- function(hp_s, hp_o, x_s, x_o, y_s, y_o, air_s, air_o) {
    cbind(hp_s / 166, hp_o / 166, (x_o - x_s) / 320, (x_s - 500) / 500,
          clip((ground - y_s) / 100), clip((y_s - y_o) / 200), s[, "timer"] / 99,
          air_s, air_o)
  }
  p1 <- view(s[, "hp1"], s[, "hp2"], s[, "x1"], s[, "x2"], s[, "y1"], s[, "y2"],
             s[, "air1"], s[, "air2"])
  if (!two) return(unname(p1))
  p2 <- view(s[, "hp2"], s[, "hp1"], s[, "x2"], s[, "x1"], s[, "y2"], s[, "y1"],
             s[, "air2"], s[, "air1"])
  out <- matrix(0, 2L * nrow(s), 9L)
  out[seq(1L, nrow(out), 2L), ] <- p1
  out[seq(2L, nrow(out), 2L), ] <- p2
  out
}

mame_check_open <- function(st) {
  if (st$closed) stop("the environment is closed", call. = FALSE)
}

mame_send <- function(st, idx, cmd, a1 = integer(length(idx)), a2 = integer(length(idx))) {
  for (k in seq_along(idx)) {
    con <- st$procs[[idx[k]]]$act_con
    writeBin(as.raw(c(cmd, a1[k], a2[k], 0L)), con)
    flush(con)
  }
}

# Answers of processes `idx`: a length(idx) x 9 integer matrix (mk3_cols).
mame_read <- function(st, idx) {
  out <- matrix(0L, length(idx), length(mk3_cols), dimnames = list(NULL, mk3_cols))
  for (k in seq_along(idx)) {
    b <- as.integer(mame_read_exact(st, idx[k], 18L))
    i16 <- function(hi, lo) { v <- 256L * hi + lo; if (v >= 32768L) v - 65536L else v }
    out[k, ] <- c(b[5], b[6], 256L * b[7] + b[8], 256L * b[9] + b[10],
                  i16(b[11], b[12]), i16(b[13], b[14]), b[15], b[16], b[17],
                  bitwAnd(b[18], 1L), bitwAnd(b[18], 2L) %/% 2L, bitwAnd(b[18], 4L) %/% 4L)
  }
  out
}

mame_read_exact <- function(st, i, size) {
  con <- st$procs[[i]]$obs_con
  out <- raw(0)
  t0 <- NULL
  while (length(out) < size) {
    chunk <- readBin(con, "raw", size - length(out))
    if (length(chunk) == 0L) {
      if (is.null(t0)) t0 <- proc.time()[["elapsed"]]
      if (proc.time()[["elapsed"]] - t0 > st$timeout)
        stop("no answer from MAME process ", i, " for ", st$timeout, " s (see ",
             file.path(st$procs[[i]]$dir, "mame.log"), ")", call. = FALSE)
      Sys.sleep(0.0005)
      next
    }
    t0 <- NULL
    out <- c(out, chunk)
  }
  out
}

# Load the after-intro snapshot ("rltgo", saved by the Lua side at start)
# in processes `idx`: one round trip, no intro steps.
mame_restart <- function(st, idx) {
  mame_send(st, idx, 2L)
  mame_read(st, idx)
}

mame_close <- function(st) {
  if (isTRUE(st$closed) || is.null(st$procs)) return(invisible())
  st$closed <- TRUE
  for (p in st$procs) {
    if (is.null(p$act_con)) next
    try({
      writeBin(as.raw(c(3L, 0L, 0L, 0L)), p$act_con)
      flush(p$act_con)
    }, silent = TRUE)
    try(close(p$act_con), silent = TRUE)
    try(close(p$obs_con), silent = TRUE)
  }
  invisible()
}
