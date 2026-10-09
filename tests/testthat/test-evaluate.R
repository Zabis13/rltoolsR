# One PPO iteration on pendulum (4 copies x 32 steps), trained once for the file.
eval_model <- local({
  m <- NULL
  function() {
    if (is.null(m)) {
      set.seed(21)
      env <- rl_vec_env("pendulum", n = 4, seed = 21)
      m <<- rl_train(env, rl_ppo(n_steps = 32, minibatch = 64), total_steps = 128,
                     device = "cpu", verbose = FALSE)
    }
    m
  }
})

test_that("rl_evaluate returns the summary and one row per episode", {
  res <- rl_evaluate(eval_model(), "pendulum", n_episodes = 5, max_steps = 20)
  expect_named(res, c("return_mean", "return_std", "length_mean", "length_std"))
  ep <- attr(res, "episodes")
  expect_s3_class(ep, "data.frame")
  expect_named(ep, c("return", "length", "truncated"))
  expect_equal(nrow(ep), 5)
  expect_equal(res[["return_mean"]], mean(ep$return))
  expect_equal(res[["length_mean"]], mean(ep$length))
})

test_that("standard deviations are population ones", {
  res <- rl_evaluate(eval_model(), "pendulum", n_episodes = 5, max_steps = 20)
  ep <- attr(res, "episodes")
  pop <- function(x) sqrt(mean((x - mean(x))^2))
  expect_equal(res[["return_std"]], pop(ep$return))
  expect_equal(res[["length_std"]], pop(ep$length))
  one <- rl_evaluate(eval_model(), "pendulum", n_episodes = 1, max_steps = 20)
  expect_identical(unname(one[c("return_std", "length_std")]), c(0, 0))
})

test_that("greedy results do not depend on n_envs", {
  # 10 episodes: batches of 1, of 3 (last one partial) and of 8 (last one partial)
  run <- function(n_envs) attr(rl_evaluate(eval_model(), "pendulum", n_episodes = 10,
                                           max_steps = 20, n_envs = n_envs), "episodes")
  e1 <- run(1)
  for (n_envs in c(3, 8)) {
    e <- run(n_envs)
    expect_equal(e$return, e1$return, tolerance = 1e-6)
    expect_identical(e$length, e1$length)
    expect_identical(e$truncated, e1$truncated)
  }
})

test_that("the same seed reproduces the evaluation, another seed changes it", {
  run <- function(seed) rl_evaluate(eval_model(), "pendulum", n_episodes = 5,
                                    max_steps = 20, seed = seed)
  a <- run(7)
  expect_identical(run(7), a)
  # pendulum starts from a random angle, so the returns follow the seed
  expect_false(isTRUE(all.equal(attr(run(8), "episodes")$return,
                                attr(a, "episodes")$return)))
})

test_that("episodes cut by a step limit are counted and marked truncated", {
  # pendulum never terminates: every episode runs to the limit
  m <- eval_model()
  limit <- rl_env("pendulum")$max_steps
  ep <- attr(rl_evaluate(m, "pendulum", n_episodes = 3), "episodes")
  expect_equal(ep$length, rep(limit, 3))
  expect_true(all(ep$truncated))
  ep <- attr(rl_evaluate(m, "pendulum", n_episodes = 3, max_steps = 10), "episodes")
  expect_equal(ep$length, rep(10L, 3))
  expect_true(all(ep$truncated))
  expect_true(all(is.finite(ep$return) & ep$return != 0))
})

test_that("evaluation changes neither the model nor the training environment", {
  m <- eval_model()
  w0 <- lapply(m$net$parameters(), as.matrix)
  meta0 <- m$meta
  train <- rl_vec_env("pendulum", n = 2, seed = 5)
  twin <- rl_vec_env("pendulum", n = 2, seed = 5)
  rl_reset(train)
  rl_reset(twin)
  eval_env <- rl_vec_env("pendulum", n = 4)
  by_obj <- rl_evaluate(m, eval_env, n_episodes = 6, max_steps = 20)
  by_name <- rl_evaluate(m, "pendulum", n_episodes = 6, max_steps = 20, n_envs = 4)
  expect_equal(by_obj, by_name)
  expect_identical(lapply(m$net$parameters(), as.matrix), w0)
  expect_identical(m$meta, meta0)
  # the training environment continues as if no evaluation had run
  a <- matrix(0.3, 2, 1)
  expect_identical(rl_step(train, a)$obs, rl_step(twin, a)$obs)
})

test_that("rl_evaluate rejects bad arguments and unsupported environments", {
  m <- eval_model()
  expect_error(rl_evaluate(m, "pendulum", n_episodes = 0),
               "`n_episodes` must be a positive integer")
  expect_error(rl_evaluate(m, rl_gpu_env("tictactoe", n = 2, device = "cpu")),
               "rl_gpu_env is not supported")
  # no light multi-player or unlimited environment yet: minimal stand-ins,
  # rejected before they are reset or stepped
  two <- structure(list(n = 1L, n_players = 2L, max_steps = 10), class = "rl_vec_env")
  expect_error(rl_evaluate(m, two), "only single-player environments")
  open <- structure(list(n = 1L, n_players = 1L, max_steps = 0), class = "rl_vec_env")
  expect_error(rl_evaluate(m, open), "no step limit, set `max_steps`")
})

test_that("rl_evaluate works as eval_fn of rl_train", {
  set.seed(22)
  env <- rl_vec_env("pendulum", n = 4, seed = 22)
  m <- rl_train(env, rl_ppo(n_steps = 32, minibatch = 64), total_steps = 256,
                device = "cpu", verbose = FALSE, eval_every_steps = 128,
                eval_fn = function(m) rl_evaluate(m, "pendulum", n_episodes = 2,
                                                  max_steps = 10))
  expect_equal(nrow(m$eval), 2)
  expect_named(m$eval, c("iteration", "steps", "return_mean", "return_std",
                         "length_mean", "length_std"))
  expect_equal(m$eval$steps, c(128, 256))
  expect_equal(m$eval$length_mean, c(10, 10))
})
