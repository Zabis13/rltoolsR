ttt_run <- function(eval_every_steps, total_steps = 3 * 9 * 16) {
  set.seed(11)
  env <- rl_gpu_env("tictactoe", n = 16, opponent = "self", device = "cpu")
  rl_train(env, rl_ppo(minibatch = 144), total_steps = total_steps,
           device = "cpu", eval_every_steps = eval_every_steps, verbose = FALSE,
           eval_fn = function(m) rl_ppo_eval_tictactoe(m, n = 32, device = "cpu"))
}

test_that("rl_train counts the budget in environment steps", {
  m <- ttt_run(0)
  expect_s3_class(m, "rl_ppo")
  expect_equal(nrow(m$history), 3)
  expect_equal(m$history$steps, c(144, 288, 432))
  expect_null(m$eval)
  expect_equal(m$train_state$steps, 432)
  expect_equal(m$train_state$iteration, 3L)
  expect_identical(m$train_state$algo_state$optimizer_state$format, "ggmlR.ag_opt_state")
  expect_gt(m$train_state$algo_state$optimizer_state$t, 0)
  expect_named(m$train_state$rng, c("train", "eval"))
})

test_that("evaluation frequency does not change training", {
  h0 <- ttt_run(0)
  h1 <- ttt_run(144)
  h2 <- ttt_run(288)
  expect_equal(nrow(h1$eval), 3)
  expect_equal(nrow(h2$eval), 1)
  expect_identical(h1$history, h0$history)
  expect_identical(h2$history, h0$history)
  expect_identical(h1$train_state$rng$train, h0$train_state$rng$train)
})

test_that("evaluation stream advances between evaluations", {
  set.seed(11)
  s <- rltoolsR:::rng_streams()
  e0 <- s$eval
  rltoolsR:::with_eval_stream(s, stats::runif(1))
  expect_false(identical(s$eval, e0))
})

test_that("evaluation leaves the training stream untouched", {
  set.seed(5)
  s <- rltoolsR:::rng_streams()
  before <- .Random.seed
  rltoolsR:::with_eval_stream(s, stats::runif(10))
  expect_identical(.Random.seed, before)
})

test_that("set_train_state restores a policy that acts the same", {
  set.seed(3)
  env <- rl_gpu_env("tictactoe", n = 16, device = "cpu")
  algo <- rl_ppo(minibatch = 144)
  m <- rl_train(env, algo, total_steps = 2 * 9 * 16, device = "cpu", verbose = FALSE)

  fresh <- algo$init(rltoolsR:::as_vec_env(env), "cpu")
  restored <- algo$set_train_state(fresh, m$train_state$algo_state)
  m2 <- algo$model(restored)
  expect_null(restored[["obs"]])
  expect_equal(restored$updates, m$train_state$algo_state$counters$updates)

  obs <- matrix(sample(c(-1, 0, 1), 64 * 9, TRUE), 64, 9)
  legal <- obs == 0
  legal[rowSums(legal) == 0, 1] <- TRUE
  expect_identical(rl_act(m2, obs, legal, greedy = TRUE),
                   rl_act(m, obs, legal, greedy = TRUE))
  set.seed(9); a1 <- rl_act(m, obs, legal, greedy = FALSE)
  set.seed(9); a2 <- rl_act(m2, obs, legal, greedy = FALSE)
  expect_identical(a2, a1)
})

test_that("rl_ppo_train accepts a single rl_env", {
  set.seed(12)
  m <- rl_ppo_train(rl_env("pendulum", seed = 12), iterations = 2, n_steps = 32,
                    minibatch = 32, device = "cpu", verbose = FALSE)
  expect_equal(m$history$steps, c(32, 64))
  expect_true(all(is.finite(as.matrix(m$history))))
})

test_that("rl_train validates its arguments", {
  env <- rl_gpu_env("tictactoe", n = 4, device = "cpu")
  expect_error(rl_train(env, list(), total_steps = 1, device = "cpu"), "rl_algo")
  expect_error(rl_algo("x", init = 1, collect = identity, update = identity,
                       model = identity, get_train_state = identity,
                       set_train_state = identity), "`init`")
  idle <- rl_algo("idle", init = function(env, device) list(),
                  collect = function(state, env) list(state = state, steps = 0),
                  update = function(state) list(state = state),
                  model = function(state) list(),
                  get_train_state = function(state) list(),
                  set_train_state = function(state, ts) state)
  expect_error(rl_train(env, idle, total_steps = 1, device = "cpu", verbose = FALSE),
               "no environment steps")
})
