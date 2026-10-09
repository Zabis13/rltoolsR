test_that("rl_vec1 wraps a single env with batched shapes", {
  venv <- rltoolsR:::rl_vec1(rl_env("pendulum", seed = 1))
  expect_s3_class(venv, "rl_vec1")
  expect_equal(venv$n, 1L)
  obs <- rl_reset(venv)
  expect_equal(dim(obs), c(1L, 3L))
  expect_equal(obs[1, ], rl_reset(rl_env("pendulum", seed = 1)))
  r <- rl_step(venv, 0)
  expect_equal(dim(r$obs), c(1L, 3L))
  expect_equal(dim(r$final_obs), c(1L, 3L))
  expect_length(r$reward, 1L)
  expect_false(r$terminated)
  expect_false(r$truncated)
  expect_equal(r$obs, r$final_obs)
  expect_null(rl_legal_actions(venv))
})

test_that("rl_vec1 matches rl_vec_env step by step", {
  single <- rltoolsR:::rl_vec1(rl_env("pendulum", seed = 4))
  vec <- rl_vec_env("pendulum", n = 1, seed = 4)
  expect_equal(rl_reset(single, seed = 4), rl_reset(vec, seed = 4))
  for (a in c(-1, 0.5, 2)) {
    expect_equal(rl_step(single, a), rl_step(vec, a))
  }
})

test_that("rl_vec1 autoresets on truncation and keeps final_obs", {
  venv <- rltoolsR:::rl_vec1(rl_env("pendulum", seed = 1))
  rl_reset(venv)
  for (i in seq_len(199)) r <- rl_step(venv, 0)
  expect_false(r$truncated)
  r <- rl_step(venv, 0)
  expect_true(r$truncated)
  expect_false(r$terminated)
  expect_false(isTRUE(all.equal(r$obs, r$final_obs)))
  r <- rl_step(venv, 0)
  expect_false(r$truncated)
})

test_that("as_vec_env wraps rl_env and passes vectorized envs through", {
  expect_s3_class(rltoolsR:::as_vec_env(rl_env("pendulum")), "rl_vec1")
  venv <- rl_vec_env("pendulum", n = 2)
  expect_identical(rltoolsR:::as_vec_env(venv), venv)
  expect_error(rltoolsR:::as_vec_env(list()), "vectorized")
})
