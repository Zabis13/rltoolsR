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
