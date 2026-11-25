# Lunar lander free fall, step by step: where the body is and how the episode
# ends. For the test-box2d.R "free fall crashes with -100" failure on Linux CI.
#
#   Rscript inst/examples/diag_lunar_lander_fall.R [seed=1] [env=lunar_lander]
#
# Columns: x, y, vy as in the observation (y = 0 is the lander's centre at
# helipad height + LEG_DOWN, so the body touches flat ground near
# y = (10 - 18) / 30 / (400 / 30 / 2) = -0.04); leg1 / leg2 the legs' touching
# flags; reward, terminated, truncated. How it ends:
#   terminated, reward -100, |x| < 1 -> begin event of the body (game_over)
#   terminated, reward -100, |x| >= 1 -> flew off the side
#   terminated, reward +100           -> the body fell asleep (b2Body_IsAwake FALSE)
#   truncated                          -> none of these within 1000 steps

suppressMessages(devtools::load_all(quiet = TRUE))
opt <- list(seed = "1", env = "lunar_lander")
for (a in commandArgs(TRUE)) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (!kv[1] %in% names(opt)) stop("unknown key: ", kv[1])
  opt[[kv[1]]] <- kv[2]
}
seed <- as.integer(opt$seed)
noop <- if (opt$env == "lunar_lander_continuous") c(-1, 0) else 0

env <- rl_env(opt$env, seed = seed)
s <- rl_reset(env, seed = seed)
rows <- list()
for (t in seq_len(env$max_steps)) {
  r <- rl_step(env, noop)
  o <- r$obs
  rows[[t]] <- data.frame(t = t, x = o[1], y = o[2], vy = o[4], leg1 = o[7], leg2 = o[8],
                          reward = r$reward, terminated = r$terminated,
                          truncated = r$truncated)
  if (r$terminated || r$truncated) break
}
d <- do.call(rbind, rows)
print(d, digits = 4, row.names = FALSE)

last <- d[nrow(d), ]
cat(sprintf("\n%s seed %d: %d steps, min y %.3f, legs touched at step %s\n",
            opt$env, seed, nrow(d), min(d$y),
            if (any(d$leg1 > 0 | d$leg2 > 0)) d$t[which(d$leg1 > 0 | d$leg2 > 0)[1]] else "never"))
cat("end: ", if (last$truncated) "truncated (no crash, no sleep within the step limit)"
    else if (last$reward == 100) "asleep (+100)"
    else if (abs(last$x) >= 1) "off the side (-100)"
    else "body touched the ground: game_over (-100)", "\n", sep = "")
