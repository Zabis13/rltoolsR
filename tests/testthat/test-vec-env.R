test_that("vec env shapes and seeding", {
  venv <- rl_vec_env("car", n = 4, seed = 10)
  expect_s3_class(venv, "rl_vec_env")
  expect_equal(venv$n, 4L)
  obs <- rl_reset(venv, seed = 10)
  expect_equal(dim(obs), c(4L, 6L))

  # copy i matches a single env seeded with seed + i - 1
  for (i in 1:4) {
    expect_equal(obs[i, ], rl_reset(rl_env("car"), seed = 10 + i - 1))
  }
  expect_false(isTRUE(all.equal(obs[1, ], obs[2, ])))

  r <- rl_step(venv, matrix(0, 4, 2))
  expect_equal(dim(r$obs), c(4L, 6L))
  expect_equal(dim(r$final_obs), c(4L, 6L))
  expect_length(r$reward, 4)
  expect_type(r$terminated, "logical")
  expect_null(rl_legal_actions(venv))
})

test_that("vec env matches single envs step by step", {
  venv <- rl_vec_env("pendulum", n = 3, seed = 1)
  envs <- lapply(0:2, function(i) rl_env("pendulum", seed = 1 + i))
  o <- rl_reset(venv, seed = 1)
  for (i in 1:3) expect_equal(o[i, ], rl_reset(envs[[i]], seed = i))
  a <- c(-0.5, 0, 0.5)
  r <- rl_step(venv, a)
  for (i in 1:3) {
    ri <- rl_step(envs[[i]], a[i])
    expect_equal(r$obs[i, ], ri$obs)
    expect_equal(r$reward[i], ri$reward)
  }
})

test_that("vec env autoresets on truncation", {
  venv <- rl_vec_env("pendulum", n = 2, seed = 1)
  rl_reset(venv)
  for (i in seq_len(199)) r <- rl_step(venv, c(0, 0))
  expect_false(any(r$truncated))
  r <- rl_step(venv, c(0, 0))
  expect_true(all(r$truncated))
  expect_false(isTRUE(all.equal(r$obs, r$final_obs)))
  r <- rl_step(venv, c(0, 0))
  expect_false(any(r$truncated))
})

test_that("vec env validates inputs", {
  expect_error(rl_vec_env("pendulum", n = 0))
  expect_error(rl_vec_env("nope", n = 2), "unknown environment")
  venv <- rl_vec_env("car", n = 2)
  rl_reset(venv)
  expect_error(rl_step(venv, c(0, 0)), "2 x 2 matrix")
})
