envs <- list(
  pendulum        = list(obs = 3L, act = 1L, max_steps = 200L),
  pendulum_gym    = list(obs = 3L, act = 1L, max_steps = 200L),
  pendulum_gym_raw = list(obs = 3L, act = 1L, max_steps = 200L),
  cartpole        = list(obs = 4L, act = 1L, max_steps = 500L),
  mountain_car    = list(obs = 2L, act = 1L, max_steps = 200L),
  mountain_car_continuous = list(obs = 2L, act = 1L, max_steps = 999L),
  acrobot        = list(obs = 6L, act = 1L, max_steps = 500L),
  acrobot_swingup = list(obs = 6L, act = 1L, max_steps = 500L),
  car             = list(obs = 6L, act = 2L, max_steps = 1000L),
  car_track       = list(obs = 9L, act = 2L, max_steps = 1000L)
)

test_that("all RLtools environments are registered", {
  # toy_text environments are tested in test-env-toy-text.R
  expect_setequal(grep("^(frozen_lake|cliff_walking|taxi|blackjack|lunar_lander)", rl_env_names(), value = TRUE, invert = TRUE),
                  names(envs))
})

for (nm in names(envs)) {
  e <- envs[[nm]]
  # cartpole has discrete actions 0 / 1: positive values push right;
  # mountain_car has 0 / 1 / 2: x in [-1, 1] rounds to push left / none / right
  act <- switch(nm,
    cartpole = function(x) { x[] <- as.numeric(x > 0); x },
    mountain_car = function(x) { x[] <- round(x) + 1; x },
    identity)

  test_that(paste(nm, "spec, reset, step, save/load"), {
    env <- rl_env(nm, seed = 1)
    expect_equal(env$name, nm)
    expect_equal(env$max_steps, e$max_steps)
    expect_equal(space_size(env$observation_space), e$obs)
    expect_equal(space_size(env$action_space), e$act)

    o1 <- rl_reset(env, seed = 7)
    o2 <- rl_reset(env, seed = 7)
    expect_equal(o1, o2)
    expect_length(o1, e$obs)
    expect_true(all(is.finite(o1)))

    a <- act(rep(0.1, e$act))
    r <- rl_step(env, a)
    expect_length(r$obs, e$obs)
    expect_true(is.finite(r$reward))

    s <- rl_save_state(env)
    r1 <- rl_step(env, a)
    rl_load_state(env, s)
    r2 <- rl_step(env, a)
    expect_equal(r1, r2)

    expect_error(rl_step(env, rep(0, e$act + 1)), "length")
  })

  test_that(paste(nm, "step fields, legal actions and episode end"), {
    env <- rl_env(nm, seed = 1)
    expect_null(rl_legal_actions(env))
    rl_reset(env, seed = 3)
    a <- act(rep(0.1, e$act))
    bad <- NA_integer_   # first step with malformed fields
    for (i in seq_len(e$max_steps)) {
      r <- rl_step(env, a)
      ok <- is.numeric(r$reward) && length(r$reward) == 1L &&
        is.logical(r$terminated) && length(r$terminated) == 1L &&
        is.logical(r$truncated) && length(r$truncated) == 1L
      if (!ok) { bad <- i; break }
      if (r$terminated || r$truncated) break
    }
    expect_identical(bad, NA_integer_, label = "first step with malformed fields")
    # an episode ends by max_steps; truncation only at the limit itself
    expect_true(r$terminated || r$truncated)
    expect_equal(r$truncated, !r$terminated && i == e$max_steps)
  })

  test_that(paste(nm, "same seed and actions give the same trajectory"), {
    run <- function() {
      env <- rl_env(nm, seed = 1)
      rl_reset(env, seed = 5)
      acts <- seq(-1, 1, length.out = 20)
      out <- list()
      # up to the end of the episode: stepping past it is an error
      for (x in acts) {
        r <- rl_step(env, act(rep(x, e$act)))
        out[[length(out) + 1L]] <- r
        if (r$terminated || r$truncated) break
      }
      out
    }
    expect_equal(run(), run())
  })

  test_that(paste(nm, "vec env matches single envs step by step"), {
    venv <- rl_vec_env(nm, n = 3, seed = 1)
    singles <- lapply(0:2, function(i) rl_env(nm, seed = 1 + i))
    o <- rl_reset(venv, seed = 1)
    expect_equal(o, do.call(rbind, lapply(1:3, function(i) rl_reset(singles[[i]], seed = i))))
    a <- act(matrix(c(-0.5, 0, 0.5), 3, e$act))
    # results of all steps, compared once at the end
    vec <- sgl <- list()
    for (t in 1:5) {
      r <- rl_step(venv, if (e$act == 1L) a[, 1] else a)
      vec[[t]] <- r[c("obs", "reward", "terminated", "truncated")]
      ri <- lapply(1:3, function(i) rl_step(singles[[i]], a[i, ]))
      sgl[[t]] <- list(obs = do.call(rbind, lapply(ri, `[[`, "obs")),
                       reward = vapply(ri, `[[`, 0, "reward"),
                       terminated = vapply(ri, `[[`, NA, "terminated"),
                       truncated = vapply(ri, `[[`, NA, "truncated"))
    }
    for (f in c("obs", "reward", "terminated", "truncated")) {
      expect_equal(lapply(vec, `[[`, f), lapply(sgl, `[[`, f), label = f)
    }
  })
}

test_that("acrobot classic gives -1 per step", {
  env <- rl_env("acrobot", seed = 1)
  rl_reset(env)
  expect_equal(rl_step(env, 0)$reward, -1)
})

test_that("car terminates when leaving the arena", {
  env <- rl_env("car", seed = 1)
  rl_reset(env)
  done <- FALSE
  for (i in seq_len(1000)) {
    r <- rl_step(env, c(1, 0))
    if (r$terminated || r$truncated) break
  }
  expect_true(r$terminated)
})

test_that("car_track lidar observations are in [0, 1]", {
  env <- rl_env("car_track", seed = 1)
  o <- rl_reset(env)
  expect_true(all(o[7:9] >= 0 & o[7:9] <= 1))
})

test_that("cartpole gives 1 per step and terminates when the pole falls", {
  env <- rl_env("cartpole", seed = 1)
  rl_reset(env)
  for (i in seq_len(500)) {
    r <- rl_step(env, 1)
    expect_equal(r$reward, 1)
    if (r$terminated || r$truncated) break
  }
  expect_true(r$terminated)
  expect_lt(i, 100)
  rl_reset(env)
  expect_error(rl_step(env, 0.5), "0 or 1")
})

test_that("a single env refuses to step after the episode ended (Gymnasium contract)", {
  env <- rl_env("frozen_lake_det", seed = 1)
  rl_reset(env)
  r <- rl_step(env, 3)
  for (a in c(1, 1, 2, 1, 2, 2)) r <- rl_step(env, a)
  expect_true(r$terminated)
  expect_error(rl_step(env, 0), "episode has ended")
  rl_reset(env)
  expect_false(rl_step(env, 0)$terminated)
  # truncation ends the episode as well
  p <- rl_env("pendulum", seed = 1)
  rl_reset(p)
  for (i in 1:200) r <- rl_step(p, 0)
  expect_true(r$truncated)
  expect_error(rl_step(p, 0), "episode has ended")
  # a loaded snapshot is played from
  s <- rl_save_state(p)
  rl_load_state(p, s)
  expect_silent(rl_step(p, 0))
})

test_that("a single env step has an empty info", {
  env <- rl_env("pendulum", seed = 1)
  rl_reset(env)
  expect_identical(rl_step(env, 0)$info, list())
})

# Gymnasium classic_control/pendulum.py step by step in double: torque u in
# [-2, 2], the cost uses the state before the step and the clipped torque
gym_pendulum <- function(th, thd, u) {
  obs <- matrix(0, length(u), 3)
  rew <- numeric(length(u))
  for (t in seq_along(u)) {
    ut <- min(max(u[t], -2), 2)
    rew[t] <- -((((th + pi) %% (2 * pi)) - pi)^2 + 0.1 * thd^2 + 0.001 * ut^2)
    thd <- min(max(thd + (3 * 10 / 2 * sin(th) + 3 * ut) * 0.05, -8), 8)
    th <- th + thd * 0.05
    obs[t, ] <- c(cos(th), sin(th), thd)
  }
  list(obs = obs, reward = rew)
}

# snapshot = step counter (int) + state (theta, theta_dot as doubles)
pendulum_load <- function(env, th, thd) rl_load_state(env, c(writeBin(0L, raw()), writeBin(c(th, thd), raw())))

test_that("pendulum_gym and pendulum_gym_raw follow Gymnasium for 200 steps", {
  set.seed(1)
  # multiples of 1/1024: actions pass to C++ as float, these stay exact
  u <- round(stats::runif(200, -2.5, 2.5) * 1024) / 1024
  ref <- gym_pendulum(2.5, -0.7, u)
  # pendulum_gym takes u / 2: 2 clip(u / 2, -1, 1) is the same torque
  for (nm in c("pendulum_gym", "pendulum_gym_raw")) {
    env <- rl_env(nm, seed = 1)
    rl_reset(env)
    pendulum_load(env, 2.5, -0.7)
    a <- if (nm == "pendulum_gym") u / 2 else u
    rs <- lapply(a, function(x) rl_step(env, x))
    obs <- do.call(rbind, lapply(rs, `[[`, "obs"))
    expect_lt(max(abs(obs - ref$obs)), 1e-5, label = paste(nm, "max |obs diff|"))
    expect_lt(max(abs(vapply(rs, `[[`, 0, "reward") - ref$reward)), 1e-4,
              label = paste(nm, "max |reward diff|"))
    # never terminates, truncated only at the 200-step limit
    expect_false(any(vapply(rs, `[[`, NA, "terminated")))
    expect_identical(vapply(rs, `[[`, NA, "truncated"), c(rep(FALSE, 199), TRUE))
  }
})

test_that("pendulum_gym_raw clips the torque and its cost to 2", {
  env <- rl_env("pendulum_gym_raw", seed = 1)
  rl_reset(env)
  pendulum_load(env, 1, 0.5)
  r3 <- rl_step(env, 3)
  pendulum_load(env, 1, 0.5)
  r2 <- rl_step(env, 2)
  expect_equal(r3, r2)
  expect_equal(r3$reward, -(1 + 0.1 * 0.25 + 0.001 * 4), tolerance = 1e-6)
  expect_equal(env$action_space$high, 2)
  expect_equal(rl_env("pendulum_gym", seed = 1)$action_space$high, 1)
})

# pushes along the velocity: reaches the goal well before the step limit
mc_swing <- function(name, left, right) {
  env <- rl_env(name, seed = 1)
  o <- rl_reset(env)
  rs <- list()
  repeat {
    r <- rl_step(env, if (o[2] >= 0) right else left)
    rs[[length(rs) + 1L]] <- r
    o <- r$obs
    if (r$terminated || r$truncated) break
  }
  rs
}

test_that("mountain_car gives -1 per step and the swing reaches the goal", {
  rs <- mc_swing("mountain_car", 0, 2)
  last <- rs[[length(rs)]]
  expect_true(last$terminated)
  expect_lt(length(rs), 200)
  expect_true(all(vapply(rs, `[[`, 0, "reward") == -1))
  expect_gte(last$obs[1], 0.5)
  env <- rl_env("mountain_car", seed = 1)
  expect_error(rl_step(env, 3), "0, 1 or 2")
  expect_error(rl_step(env, 0.5), "0, 1 or 2")
})

test_that("mountain_car_continuous rewards -0.1 a^2 and 100 at the goal", {
  rs <- mc_swing("mountain_car_continuous", -1, 1)
  last <- rs[[length(rs)]]
  expect_true(last$terminated)
  expect_gte(last$obs[1], 0.45)
  expect_equal(last$reward, 100 - 0.1, tolerance = 1e-6)
  expect_equal(rs[[1]]$reward, -0.1, tolerance = 1e-6)
  # the torque cost uses the unclipped action
  env <- rl_env("mountain_car_continuous", seed = 1)
  rl_reset(env)
  expect_equal(rl_step(env, 2)$reward, -0.4, tolerance = 1e-6)
})

test_that("mountain_car stops at the left wall", {
  for (nm in c("mountain_car", "mountain_car_continuous")) {
    env <- rl_env(nm, seed = 1)
    rl_reset(env)
    # snapshot = step counter (int) + state (x, velocity as doubles)
    rl_load_state(env, c(writeBin(0L, raw()), writeBin(c(-1.19, -0.05), raw())))
    r <- rl_step(env, if (nm == "mountain_car") 0 else -1)
    expect_equal(r$obs, c(-1.2, 0), tolerance = 1e-6)
  }
})
