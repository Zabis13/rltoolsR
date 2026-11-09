# Errors raised from C++ must not leak: Rf_error() longjmps past C++
# destructors, so these calls are here mainly for the valgrind run (a leak shows
# up there as "definitely lost"). Names are longer than the short-string buffer
# so std::string really allocates on the heap.
long_name <- strrep("x", 100)

test_that("unknown long names fail cleanly", {
  expect_error(rl_env(long_name), "unknown environment")
  expect_error(rl_vec_env(long_name, n = 2), "unknown environment")
})

test_that("bad actions fail cleanly", {
  env <- rl_env("pendulum", seed = 1)
  rl_reset(env, seed = 1)
  expect_error(rl_step(env, rep(0, 50)), "length 1")
  cp <- rl_env("cartpole", seed = 1)
  rl_reset(cp, seed = 1)
  expect_error(rl_step(cp, 0.5), "0 or 1")

  venv <- rl_vec_env("cartpole", n = 2, seed = 1)
  rl_reset(venv, seed = 1)
  expect_error(rl_step(venv, c(0, 0, 0)), "2 x 1 matrix")
  expect_error(rl_step(venv, c(0.5, 0.5)), "0 or 1")
})

test_that("an empty legal mask gives NULL", {
  expect_null(rl_legal_actions(rl_env("pendulum", seed = 1)))
  expect_null(rl_legal_actions(rl_vec_env("pendulum", n = 2, seed = 1)))
})

test_that("state snapshots round-trip through the two-phase copy", {
  env <- rl_env("pendulum", seed = 1)
  rl_reset(env, seed = 1)
  rl_step(env, 0.3)
  st <- rl_save_state(env)
  expect_true(is.raw(st) && length(st) > 0)
  expect_identical(rl_save_state(env), st)
})

test_that("bad state snapshots fail cleanly", {
  env <- rl_env("pendulum", seed = 1)
  expect_error(rl_load_state(env, as.raw(1:2)), "too short")
})
