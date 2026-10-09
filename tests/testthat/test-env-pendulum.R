test_that("pendulum is registered and has correct spec", {
  expect_true("pendulum" %in% rl_env_names())
  env <- rl_env("pendulum", seed = 1)
  expect_s3_class(env, "rl_env")
  expect_equal(env$mode, "single")
  expect_equal(env$max_steps, 200L)
  expect_s3_class(env$observation_space, "rl_space_box")
  expect_equal(env$observation_space$shape, 3L)
  expect_equal(env$observation_space$dtype, "float32")
  expect_equal(env$action_space$low, -1)
  expect_equal(env$action_space$high, 1)
  expect_null(rl_legal_actions(env))
})

test_that("reset is reproducible with a seed", {
  env <- rl_env("pendulum")
  o1 <- rl_reset(env, seed = 42)
  o2 <- rl_reset(env, seed = 42)
  expect_equal(o1, o2)
  expect_length(o1, 3)
  expect_equal(o1[1]^2 + o1[2]^2, 1, tolerance = 1e-6)
})

test_that("step returns reward and truncates at step limit", {
  env <- rl_env("pendulum", seed = 1)
  rl_reset(env)
  for (i in seq_len(199)) {
    r <- rl_step(env, 0)
    expect_false(r$truncated)
  }
  r <- rl_step(env, 0)
  expect_true(r$truncated)
  expect_false(r$terminated)
  expect_lte(r$reward, 0)
})

test_that("wrong action length is an error", {
  env <- rl_env("pendulum", seed = 1)
  rl_reset(env)
  expect_error(rl_step(env, c(0, 0)), "length 1")
})

test_that("save/load state restores trajectory", {
  env <- rl_env("pendulum", seed = 1)
  rl_reset(env)
  rl_step(env, 0.3)
  s <- rl_save_state(env)
  a <- rl_step(env, 1)
  rl_step(env, -1)
  rl_load_state(env, s)
  b <- rl_step(env, 1)
  expect_equal(a, b)
})

test_that("unknown environment is an error", {
  expect_error(rl_env("nope"), "unknown environment")
})
