# Where the dynamic linker binds Box2D's functions for this package
# (rlt_box2d_symbols(), dladdr()): all in this package's own library. Another
# file means a symbol of another library loaded in the process overrides
# Box2D's (ELF symbol interposition, Linux). Not on Windows (no dladdr).
test_that("Box2D's functions resolve to this package's library", {
  skip_on_os("windows")
  s <- .Call(rltoolsR:::rlt_box2d_symbols)
  message(paste(sprintf("  %-24s %s", names(s), s), collapse = "\n"))
  expect_true(all(s == s[["rlt_box2d_check"]]),
              info = paste(sprintf("%s: %s", names(s), s), collapse = "; "))
})

# The bundled Box2D v3 (src/box2d) builds, links and simulates: a box dropped
# from 10 m comes to rest on the ground at y = 0.5 (its half height).
test_that("bundled Box2D builds and simulates a falling box", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_equal(r[1:2], c(3, 2))
  expect_equal(r[4], 0.5, tolerance = 0.02)
})

# A polygon-segment collision in its stages (segment_checks()): narrow phase,
# the segment's bounds, broad phase.
test_that("Box2D: narrow phase, a box overlapping a segment has 2 contact points", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_equal(r[10], 2)
  expect_lt(abs(r[11] - (-0.1)), 1e-3)               # the box's bottom 0.1 below the segment
})

# How a segment's bounds are made, step by step (b2CreateSegmentShape ->
# body transform -> stored segment -> b2ComputeSegmentBounds -> shape->aabb ->
# broad-phase tree)
test_that("Box2D: b2ComputeSegmentBounds() alone, identity transform", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_equal(r[17:20], c(-50, 0, 50, 0), info = paste(r[17:20], collapse = " "))
})

test_that("Box2D: a static body's transform", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_equal(r[21:24], c(0, 0, 1, 0), info = paste(r[21:24], collapse = " "))
})

test_that("Box2D: the segment as stored in the shape", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_equal(r[25:28], c(-50, 0, 50, 0), info = paste(r[25:28], collapse = " "))
})

test_that("Box2D: b2ComputeSegmentBounds() with the body's transform", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_equal(r[38:41], c(-50, 0, 50, 0), info = paste(r[38:41], collapse = " "))
})

test_that("Box2D: the shape's AABB recomputed as at creation, margin 0", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_equal(r[42:45], c(-50, 0, 50, 0), info = paste(r[42:45], collapse = " "))
})

test_that("Box2D: the shape's AABB recomputed as at creation, margin 0.02", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_equal(r[46:49], c(-50.02, -0.02, 50.02, 0.02), tolerance = 1e-5,
               info = paste(r[46:49], collapse = " "))
})

test_that("Box2D: the broad-phase tree finds the segment near the origin", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_equal(r[29], 1)
})

test_that("Box2D: the broad phase's fat AABB of the segment", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_true(all(abs(r[30:33] - c(-50, 0, 50, 0)) < 0.1), info = paste(r[30:33], collapse = " "))
})

test_that("Box2D: the segment's node AABB in the broad-phase tree", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_true(all(abs(r[34:37] - c(-50, 0, 50, 0)) < 0.1), info = paste(r[34:37], collapse = " "))
})

test_that("Box2D: a segment shape's bounds", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_true(all(abs(r[12:15] - c(-50, 0, 50, 0)) < 0.1), info = paste(r[12:15], collapse = " "))
})

test_that("Box2D: broad phase, a box falling onto a segment gets a contact", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_gte(r[16], 1)
})

# Step by step towards the lunar lander (src/env_box2d.cpp, drop_box()): each
# drops a body from 10 m, stepped as the lander (50 Hz, 8 sub-steps). The first
# that fails tells what does not collide.
test_that("Box2D: a box rests on segment ground", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_equal(r[5], 0.5, tolerance = 0.02)
})

test_that("Box2D: a box with the lander's filter rests on polygon ground", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_equal(r[6], 0.5, tolerance = 0.02)
})

test_that("Box2D: a box with the lander's filter rests on segment ground", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_equal(r[7], 0.5, tolerance = 0.02)
})

test_that("Box2D: the lander's hull rests on segment ground", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_equal(r[8], 10 / 30, tolerance = 0.02)       # bottom 10 / LL_SCALE below the centre
})

test_that("Box2D: the lander's hull with legs stays on segment ground", {
  r <- .Call(rltoolsR:::rlt_box2d_check)
  expect_gt(r[9], 0.3)                               # not through the ground
  expect_lt(r[9], 1)                                 # not hanging in the air
})

# ---- LunarLander (src/env_box2d.cpp, from gymnasium/envs/box2d/lunar_lander.py)

# Gymnasium's heuristic() from lunar_lander.py, in R
ll_heuristic <- function(s, continuous) {
  angle_targ <- min(max(s[1] * 0.5 + s[3] * 1.0, -0.4), 0.4)
  hover_targ <- 0.55 * abs(s[1])
  angle_todo <- (angle_targ - s[5]) * 0.5 - s[6] * 1.0
  hover_todo <- (hover_targ - s[2]) * 0.5 - s[4] * 0.5
  if (s[7] > 0 || s[8] > 0) {
    angle_todo <- 0
    hover_todo <- -s[4] * 0.5
  }
  if (continuous) return(pmin(pmax(c(hover_todo * 20 - 1, -angle_todo * 20), -1), 1))
  if (hover_todo > abs(angle_todo) && hover_todo > 0.05) 2
  else if (angle_todo < -0.05) 3
  else if (angle_todo > 0.05) 1
  else 0
}

ll_episode <- function(name, seed, policy) {
  env <- rl_env(name, seed = seed)
  s <- rl_reset(env, seed = seed)
  total <- 0
  for (t in seq_len(env$max_steps)) {
    r <- rl_step(env, policy(s))
    total <- total + r$reward
    s <- r$obs
    if (r$terminated || r$truncated) break
  }
  list(total = total, last = r, steps = t)
}

# Free fall of seed 1 against a reference run (inst/examples/
# diag_lunar_lander_fall.R): start, flight, legs touching. Observation:
# x, y, vx, vy, angle, angular velocity, leg1, leg2.
ll_fall <- function(steps) {
  env <- rl_env("lunar_lander", seed = 1)
  obs <- list(rl_reset(env, seed = 1))
  for (t in seq_len(steps)) obs[[t + 1]] <- rl_step(env, 0)$obs
  obs                                                # obs[[t + 1]]: after step t
}

test_that("lunar_lander free fall: start position (after step 1)", {
  o <- ll_fall(1)[[2]]
  expect_lt(abs(o[2] - 1.4138), 1e-3)
})

test_that("lunar_lander free fall: flight at step 50", {
  o <- ll_fall(50)[[51]]
  expect_lt(abs(o[2] - 0.7355), 1e-3)
  expect_lt(abs(o[4] - (-1.267)), 1e-3)
})

test_that("lunar_lander free fall: legs touch the ground at step 73", {
  obs <- ll_fall(73)
  expect_equal(obs[[73]][7:8], c(0, 0))              # step 72: in the air
  expect_equal(obs[[74]][7:8], c(1, 1))              # step 73: both legs down
})

test_that("lunar_lander spec", {
  env <- rl_env("lunar_lander", seed = 1)
  expect_equal(env$max_steps, 1000L)
  expect_equal(env$observation_space$shape, 8L)
  expect_equal(env$action_space$n, 4L)
  envc <- rl_env("lunar_lander_continuous", seed = 1)
  expect_equal(envc$action_space$shape, 2L)
  o <- rl_reset(env, seed = 3)
  expect_length(o, 8)
  expect_equal(o[7:8], c(0, 0))                      # legs up in the air
  expect_lt(abs(o[1]), 0.1)                          # starts above the middle
  expect_error(rl_step(env, 4), "integer in 0..3")
  expect_error(rl_step(envc, c(0, NaN)), "NaN")
})

for (nm in c("lunar_lander", "lunar_lander_continuous")) {
  continuous <- nm == "lunar_lander_continuous"

  # The heuristic itself crashes now and then (in Gymnasium too), so no landing
  # is required of every seed: the mean return over 20 seeds, with a margin
  # (lunar_lander measured 244.7 on seeds 1-20, 240.8 on 1001-1200).
  test_that(paste(nm, "Gymnasium's heuristic: mean return above 220 over 20 seeds"), {
    ret <- vapply(1:20, function(seed) {
      ll_episode(nm, seed, function(s) ll_heuristic(s, continuous))$total
    }, 0)
    expect_gt(mean(ret), 220)
  })

  test_that(paste(nm, "free fall crashes with -100"), {
    noop <- if (continuous) function(s) c(-1, 0) else function(s) 0
    e <- ll_episode(nm, 1, noop)
    expect_true(e$last$terminated)
    expect_equal(e$last$reward, -100)
  })

  test_that(paste(nm, "same seed and actions, same trajectory; save / load"), {
    pol <- function(s) ll_heuristic(s, continuous)
    expect_equal(ll_episode(nm, 4, pol), ll_episode(nm, 4, pol))
    env <- rl_env(nm, seed = 5)
    s <- rl_reset(env, seed = 5)
    for (t in 1:30) s <- rl_step(env, pol(s))$obs
    snap <- rl_save_state(env)
    # engines off: their dispersion comes from the RNG, which a snapshot does
    # not hold; without them the restored world must step exactly the same
    a <- if (continuous) c(-1, 0) else 0
    r1 <- lapply(1:5, function(t) rl_step(env, a))
    rl_load_state(env, snap)
    r2 <- lapply(1:5, function(t) rl_step(env, a))
    expect_identical(r1, r2)
  })

  test_that(paste(nm, "vec env matches single envs"), {
    venv <- rl_vec_env(nm, n = 3, seed = 1)
    singles <- lapply(0:2, function(i) rl_env(nm, seed = 1 + i))
    o <- rl_reset(venv, seed = 1)
    expect_equal(o, do.call(rbind, lapply(1:3, function(i) rl_reset(singles[[i]], seed = i))))
    a <- if (continuous) matrix(c(0.5, 0, -0.5, 0.8, -0.8, 0), 3, 2) else c(0, 2, 3)
    vec <- sgl <- list()
    for (t in 1:20) {
      r <- rl_step(venv, a)
      done <- r$terminated | r$truncated
      r$obs[done, ] <- r$final_obs[done, ]          # the singles keep the last obs
      vec[[t]] <- r[c("obs", "reward", "terminated", "truncated")]
      ri <- lapply(1:3, function(i) rl_step(singles[[i]], if (continuous) a[i, ] else a[i]))
      sgl[[t]] <- list(obs = do.call(rbind, lapply(ri, `[[`, "obs")),
                       reward = vapply(ri, `[[`, 0, "reward"),
                       terminated = vapply(ri, `[[`, NA, "terminated"),
                       truncated = vapply(ri, `[[`, NA, "truncated"))
      if (any(r$terminated | r$truncated)) break      # singles do not auto-reset
    }
    for (f in c("obs", "reward", "terminated", "truncated"))
      expect_equal(lapply(vec, `[[`, f), lapply(sgl, `[[`, f), label = f)
  })
}
