test_that("pool arguments are validated and filled with defaults", {
  pa <- rltoolsR:::ppo_pool_args
  expect_null(pa(NULL))
  expect_null(pa(FALSE))
  expect_identical(pa(TRUE), list(p_self = 0.8, every = 10L, size = 20L))
  expect_identical(pa(list(every = 5))$every, 5L)
  expect_error(pa(list(p_self = 2)), "p_self")
  expect_error(pa(list(every = 0)), "every")
  expect_error(pa(list(size = 1.5)), "size")
  expect_error(pa(list(foo = 1)), "pool")
})

test_that("pool needs a discrete self-play environment", {
  env <- rl_gpu_env("tictactoe", n = 4, opponent = "random", device = "cpu")
  expect_error(rl_ppo_train(env, iterations = 1, device = "cpu", verbose = FALSE,
                            pool = TRUE), "self-play")
})

test_that("the move counter follows the board through wins and auto-resets", {
  set.seed(3)
  env <- rl_gpu_env("tictactoe", n = 64, device = "cpu")
  algo <- rl_ppo(n_steps = 23, minibatch = 512, pool = list(p_self = 0, every = 1))
  prev <- ag_device("cpu")
  on.exit(ag_device(prev), add = TRUE)
  state <- algo$init(env, "cpu")
  for (i in 1:3) {
    col <- algo$collect(state, env)
    state <- col$state
    # marks on the board == moves since the last reset, in every game
    expect_identical(as.integer(rowSums(state$obs != 0)), state$ply)
    state <- algo$update(state)$state
  }
  expect_gt(col$info[["episodes"]], 64)    # games did end and restart
})

test_that("pool training reports its columns and keeps update shapes fixed", {
  set.seed(4)
  env <- rl_gpu_env("tictactoe", n = 64, device = "cpu")
  m <- rl_ppo_train(env, iterations = 4, minibatch = 256, device = "cpu",
                    verbose = FALSE, pool = list(p_self = 0.5, every = 2, size = 1))
  h <- m$history
  expect_true(all(c("used", "pool_games", "pool_win", "pool_loss", "pool_win_x", "pool_win_o") %in% names(h)))
  expect_true(all(h$used %% 256 == 0))
  expect_true(all(h$used < 64 * 9))        # the snapshot's moves are dropped
  expect_true(all(h$pool_games > 0))
  expect_true(all(h$pool_win >= 0 & h$pool_win <= 1))
  expect_true(all(h$pool_loss >= 0 & h$pool_win + h$pool_loss <= 1))
  ps <- m$train_state$algo_state$pool
  expect_length(ps$pool, 1L)               # size = 1: oldest dropped
  expect_identical(ps$pool_ids, 2L)        # snapshots at iterations 1 and 3
  expect_identical(ps$iters, 4L)
})

test_that("without pool the history has no pool columns", {
  set.seed(4)
  env <- rl_gpu_env("tictactoe", n = 16, device = "cpu")
  m <- rl_ppo_train(env, iterations = 1, minibatch = 144, device = "cpu", verbose = FALSE)
  expect_false(any(c("used", "pool_games", "pool_win", "pool_loss", "pool_win_x", "pool_win_o") %in% names(m$history)))
})

test_that("tic-tac-toe evaluation as O reports shares that sum to one", {
  set.seed(6)
  env <- rl_gpu_env("tictactoe", n = 32, device = "cpu")
  m <- rl_ppo_train(env, iterations = 1, minibatch = 288, device = "cpu", verbose = FALSE)
  res <- rl_ppo_eval_tictactoe(m, n = 200, side = "O")
  expect_named(res, c("win", "draw", "loss"))
  expect_equal(sum(res), 1)
  # an untrained O against a random X loses more often than X would
  expect_gt(res[["loss"]], rl_ppo_eval_tictactoe(m, n = 200)[["loss"]])
})
