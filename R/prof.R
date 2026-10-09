# Stage timer for PPO collect/update (inst/examples/profile_ppo_stages.R).
#
# Off by default: a stage boundary is `tk <- prof_tick(tk, "name")`, and with
# profiling off prof_start() returns NULL, so each boundary costs one is.null().
# Wall-clock time: on the GPU an eager ag_* op computes before it returns, while
# graph mode defers the forward to the next materialization (as.matrix,
# backward), so its time lands in that stage.
#
# `probe` is an optional function() returning a named numeric of counters
# (launches, transfers, ...); its per-stage deltas are accumulated alongside the
# times. The script installs it, so this file needs no ggmlR internals.

.rl_prof <- new.env(parent = emptyenv())
.rl_prof$on <- FALSE

# Switch on/off (TRUE/FALSE) and clear; returns the previous state invisibly.
rl_prof <- function(on = TRUE, probe = NULL) {
  old <- .rl_prof$on
  .rl_prof$on <- isTRUE(on)
  .rl_prof$probe <- probe
  .rl_prof$ms <- numeric(0)
  .rl_prof$n <- numeric(0)
  .rl_prof$cnt <- list()
  invisible(old)
}

# list(ms = per-stage milliseconds, n = per-stage hits, cnt = per-stage probe deltas)
rl_prof_get <- function() list(ms = .rl_prof$ms, n = .rl_prof$n, cnt = .rl_prof$cnt)

prof_start <- function() {
  if (!.rl_prof$on) return(NULL)
  list(t = proc.time()[["elapsed"]],
       c = if (!is.null(.rl_prof$probe)) .rl_prof$probe())
}

prof_tick <- function(tk, stage) {
  if (is.null(tk)) return(NULL)
  now <- prof_start()
  add <- function(v, x) { v[stage] <- (if (is.na(v[stage])) 0 else v[stage]) + x; v }
  .rl_prof$ms <- add(.rl_prof$ms, 1000 * (now$t - tk$t))
  .rl_prof$n <- add(.rl_prof$n, 1)
  if (!is.null(now$c)) {
    d <- now$c - tk$c
    old <- .rl_prof$cnt[[stage]]
    .rl_prof$cnt[[stage]] <- if (is.null(old)) d else old + d
  }
  # restart after the bookkeeping, so it is not charged to the next stage
  prof_start()
}
