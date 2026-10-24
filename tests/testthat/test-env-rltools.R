envs <- list(
  pendulum        = list(obs = 3L, act = 1L, max_steps = 200L),
  acrobot         = list(obs = 6L, act = 1L, max_steps = 500L),
  acrobot_swingup = list(obs = 6L, act = 1L, max_steps = 500L),
  car             = list(obs = 6L, act = 2L, max_steps = 1000L),
  car_track       = list(obs = 9L, act = 2L, max_steps = 1000L)
)

test_that("all RLtools environments are registered", {
  expect_setequal(rl_env_names(), names(envs))
})

for (nm in names(envs)) {
  e <- envs[[nm]]

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

    a <- rep(0.1, e$act)
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
    a <- rep(0.1, e$act)
    for (i in seq_len(e$max_steps)) {
      r <- rl_step(env, a)
      expect_true(is.numeric(r$reward) && length(r$reward) == 1L)
      expect_true(is.logical(r$terminated) && length(r$terminated) == 1L)
      expect_true(is.logical(r$truncated) && length(r$truncated) == 1L)
      if (r$terminated || r$truncated) break
    }
    # an episode ends by max_steps; truncation only at the limit itself
    expect_true(r$terminated || r$truncated)
    expect_equal(r$truncated, !r$terminated && i == e$max_steps)
  })

  test_that(paste(nm, "same seed and actions give the same trajectory"), {
    run <- function() {
      env <- rl_env(nm, seed = 1)
      rl_reset(env, seed = 5)
      acts <- seq(-1, 1, length.out = 20)
      lapply(acts, function(x) rl_step(env, rep(x, e$act)))
    }
    expect_equal(run(), run())
  })

  test_that(paste(nm, "vec env matches single envs step by step"), {
    venv <- rl_vec_env(nm, n = 3, seed = 1)
    singles <- lapply(0:2, function(i) rl_env(nm, seed = 1 + i))
    o <- rl_reset(venv, seed = 1)
    for (i in 1:3) expect_equal(o[i, ], rl_reset(singles[[i]], seed = i))
    a <- matrix(c(-0.5, 0, 0.5), 3, e$act)
    for (t in 1:5) {
      r <- rl_step(venv, if (e$act == 1L) a[, 1] else a)
      for (i in 1:3) {
        ri <- rl_step(singles[[i]], a[i, ])
        expect_equal(r$obs[i, ], ri$obs)
        expect_equal(r$reward[i], ri$reward)
        expect_equal(r$terminated[i], ri$terminated)
        expect_equal(r$truncated[i], ri$truncated)
      }
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
