# LunarLander with Gymnasium's heuristic: how each episode ends and what the
# lander does in its last steps (landing, crash or still awake at the limit).
#
#   Rscript inst/examples/lunar_lander_diag.R [env=lunar_lander] [seeds=1:10] [tail=12]
suppressMessages(devtools::load_all(quiet = TRUE))

opt <- list(env = "lunar_lander", seeds = "1:10", tail = 12)
for (a in commandArgs(TRUE)) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (!kv[1] %in% names(opt)) stop("unknown key: ", kv[1])
  opt[[kv[1]]] <- if (is.numeric(opt[[kv[1]]])) as.numeric(kv[2]) else kv[2]
}
continuous <- opt$env == "lunar_lander_continuous"

# Gymnasium's heuristic() from lunar_lander.py
heuristic <- function(s) {
  angle_targ <- min(max(s[1] * 0.5 + s[3] * 1.0, -0.4), 0.4)
  hover_targ <- 0.55 * abs(s[1])
  angle_todo <- (angle_targ - s[5]) * 0.5 - s[6] * 1.0
  hover_todo <- (hover_targ - s[2]) * 0.5 - s[4] * 0.5
  if (s[7] > 0 || s[8] > 0) { angle_todo <- 0; hover_todo <- -s[4] * 0.5 }
  if (continuous) return(pmin(pmax(c(hover_todo * 20 - 1, -angle_todo * 20), -1), 1))
  if (hover_todo > abs(angle_todo) && hover_todo > 0.05) 2
  else if (angle_todo < -0.05) 3 else if (angle_todo > 0.05) 1 else 0
}

for (seed in eval(parse(text = opt$seeds))) {
  env <- rl_env(opt$env, seed = seed)
  s <- rl_reset(env, seed = seed)
  rows <- list(); total <- 0; first_contact <- NA
  for (t in seq_len(env$max_steps)) {
    a <- heuristic(s)
    r <- rl_step(env, a)
    total <- total + r$reward
    if (is.na(first_contact) && (r$obs[7] > 0 || r$obs[8] > 0)) first_contact <- t
    rows[[t]] <- c(t = t, a = if (continuous) a[1] else a, round(r$obs, 3), rew = round(r$reward, 2))
    s <- r$obs
    if (r$terminated || r$truncated) break
  }
  how <- if (r$truncated) "LIMIT (awake)" else if (r$reward == 100) "landed" else "crash"
  cat(sprintf("\n== seed %d: %s at step %d, return %.1f, first leg contact at %s\n",
              seed, how, t, total, first_contact))
  if (how != "landed") {
    m <- do.call(rbind, utils::tail(rows, opt$tail))
    colnames(m) <- c("t", "a", "x", "y", "vx", "vy", "ang", "angv", "legL", "legR", "rew")
    print(m, row.names = FALSE)
    # a leg flag that flips back and forth: contact begin / end chatter
    legs <- do.call(rbind, rows)[, c(9, 10)]
    flips <- sum(abs(diff(legs[, 1]))) + sum(abs(diff(legs[, 2])))
    cat("leg contact flag changes:", flips, "\n")
  }
}
