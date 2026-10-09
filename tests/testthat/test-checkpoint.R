ck_dir <- function() {
  d <- tempfile("rlt_ck_")
  dir.create(d)
  d
}

ttt_env <- function() rl_gpu_env("tictactoe", n = 16, device = "cpu")
ITER <- 9 * 16   # steps per PPO iteration with these settings

counting <- function(algo) {
  calls <- new.env()
  calls$n <- 0
  collect <- algo$collect
  algo$collect <- function(state, env) { calls$n <- calls$n + 1; collect(state, env) }
  list(algo = algo, calls = calls)
}

test_that("a checkpoint is a model file with the training state", {
  dir <- ck_dir()
  f <- file.path(dir, "ck.rds")
  set.seed(1)
  m <- rl_train(ttt_env(), rl_ppo(minibatch = 144), total_steps = 2 * ITER,
                device = "cpu", verbose = FALSE, checkpoint = f)
  obj <- readRDS(f)
  expect_identical(obj$format_version, 1L)
  expect_true(obj$has_train_state)
  ts <- obj$train_state
  expect_equal(ts$steps, 2 * ITER)
  expect_equal(ts$iteration, 2L)
  expect_named(ts$rng, c("train", "eval"))
  expect_equal(ts$history, m$history)
  expect_false(ts$history_truncated)
  expect_equal(ts$algo_params$minibatch, 144)
  # rl_load_model() reads it as a plain model and ignores the state
  m2 <- rl_load_model(f, device = "cpu")
  obs <- matrix(0, 4, 9)
  expect_identical(rl_act(m2, obs), rl_act(m, obs))
  # a plain model file says it has no state
  g <- file.path(dir, "model.rds")
  rl_save_model(m, g)
  expect_false(readRDS(g)$has_train_state)
})

test_that("resume restores the policy and continues the counters", {
  dir <- ck_dir()
  f <- file.path(dir, "ck.rds")
  set.seed(2)
  m1 <- rl_train(ttt_env(), rl_ppo(minibatch = 144), total_steps = 2 * ITER,
                 device = "cpu", verbose = FALSE, checkpoint = f)
  algo <- rl_ppo(minibatch = 144)
  m2 <- rl_train(ttt_env(), algo, total_steps = 4 * ITER, device = "cpu",
                 verbose = FALSE, checkpoint = f, resume = f)
  expect_equal(m2$history$steps, (1:4) * ITER)
  expect_equal(m2$history$iteration, 1:4)
  expect_equal(m2$history[1:2, ], m1$history)
  expect_equal(m2$train_state$iteration, 4L)
  expect_gte(m2$train_state$time, m1$train_state$time)
  expect_equal(readRDS(f)$train_state$steps, 4 * ITER)

  # set_train_state on the checkpointed state gives the same actions
  state <- algo$init(rltoolsR:::as_vec_env(ttt_env()), "cpu")
  state <- algo$set_train_state(state, m1$train_state$algo_state)
  obs <- matrix(sample(c(-1, 0, 1), 20 * 9, TRUE), 20)
  legal <- obs == 0
  legal[rowSums(legal) == 0, 1] <- TRUE
  expect_identical(rl_act(algo$model(state), obs, legal), rl_act(m1, obs, legal))
})

test_that("resume restores both RNG streams at the iteration boundary", {
  dir <- ck_dir()
  f <- file.path(dir, "ck.rds")
  draw_u <- function(m) c(u = stats::runif(1))
  set.seed(3)
  m1 <- rl_train(ttt_env(), rl_ppo(minibatch = 144), total_steps = ITER,
                 device = "cpu", verbose = FALSE, checkpoint = f,
                 eval_every_steps = ITER, eval_fn = draw_u)
  ts <- m1$train_state
  set.seed(99)  # the caller's stream must not matter
  seen <- new.env()
  algo <- rl_ppo(minibatch = 144)
  collect <- algo$collect
  algo$collect <- function(state, env) {
    if (is.null(seen$seed)) seen$seed <- get(".Random.seed", envir = globalenv())
    collect(state, env)
  }
  m2 <- rl_train(ttt_env(), algo, total_steps = 2 * ITER, device = "cpu",
                 verbose = FALSE, resume = f, eval_every_steps = ITER, eval_fn = draw_u)
  # the first collect after resume starts from the saved training stream
  expect_identical(seen$seed, ts$rng$train)
  # the next evaluation continues the saved evaluation stream
  next_eval <- rltoolsR:::with_preserved_seed({
    assign(".Random.seed", ts$rng$eval, envir = globalenv())
    stats::runif(1)
  })
  expect_identical(m2$eval$u[2], next_eval)
})

test_that("normalizer statistics keep their count across resume", {
  dir <- ck_dir()
  f <- file.path(dir, "ck.rds")
  set.seed(4)
  env <- rl_vec_env("pendulum", n = 4, seed = 4)
  algo <- rl_ppo(n_steps = 16, minibatch = 64)
  rl_train(env, algo, total_steps = 128, device = "cpu", verbose = FALSE, checkpoint = f)
  m <- rl_train(env, algo, total_steps = 256, device = "cpu", verbose = FALSE,
                checkpoint = f, resume = f)
  expect_equal(m$train_state$algo_state$normalizers$obs$count, 256)
  expect_equal(m$train_state$algo_state$normalizers$reward$count, 256)
})

test_that("resume with the budget reached returns without training", {
  dir <- ck_dir()
  f <- file.path(dir, "ck.rds")
  set.seed(5)
  m1 <- rl_train(ttt_env(), rl_ppo(minibatch = 144), total_steps = 2 * ITER,
                 device = "cpu", verbose = FALSE, checkpoint = f)
  before <- readRDS(f)
  cnt <- counting(rl_ppo(minibatch = 144))
  expect_message(
    m2 <- rl_train(ttt_env(), cnt$algo, total_steps = 2 * ITER, device = "cpu",
                   verbose = FALSE, checkpoint = f, resume = f),
    "nothing to train")
  expect_equal(cnt$calls$n, 0)
  expect_equal(m2$history, m1$history)
  expect_identical(readRDS(f)$train_state$steps, before$train_state$steps)
})

test_that("an incompatible checkpoint fails before the first iteration", {
  dir <- ck_dir()
  f <- file.path(dir, "ck.rds")
  set.seed(6)
  rl_train(ttt_env(), rl_ppo(minibatch = 144), total_steps = ITER,
           device = "cpu", verbose = FALSE, checkpoint = f)

  cnt <- counting(rl_ppo(minibatch = 144, hidden = 32))
  expect_error(rl_train(ttt_env(), cnt$algo, total_steps = 2 * ITER, device = "cpu",
                        verbose = FALSE, resume = f), "other shapes")
  other <- rl_algo("other", init = function(env, device) list(),
                   collect = function(state, env) stop("collect called"),
                   update = function(state) list(state = state),
                   model = function(state) list(meta = list()),
                   get_train_state = function(state) list(),
                   set_train_state = function(state, ts) state)
  expect_error(rl_train(ttt_env(), other, total_steps = 2 * ITER, device = "cpu",
                        verbose = FALSE, resume = f), "from algorithm 'ppo'")
  expect_error(rl_train(rl_vec_env("pendulum", n = 2), cnt$algo, total_steps = 2 * ITER,
                        device = "cpu", verbose = FALSE, resume = f), "actions")
  expect_equal(cnt$calls$n, 0)

  # changed hyperparameters are allowed with a warning
  expect_warning(rl_train(ttt_env(), rl_ppo(minibatch = 144, lr = 5e-4),
                          total_steps = 2 * ITER, device = "cpu", verbose = FALSE,
                          resume = f), "changed hyperparameters: lr")

  # a plain model file cannot be resumed
  g <- file.path(dir, "model.rds")
  rl_save_model(rl_load_model(f, device = "cpu"), g)
  expect_error(rl_train(ttt_env(), rl_ppo(minibatch = 144), total_steps = 2 * ITER,
                        device = "cpu", verbose = FALSE, resume = g), "no training state")
})

test_that("an existing checkpoint is not overwritten by a new run", {
  dir <- ck_dir()
  f <- file.path(dir, "ck.rds")
  set.seed(7)
  rl_train(ttt_env(), rl_ppo(minibatch = 144), total_steps = ITER,
           device = "cpu", verbose = FALSE, checkpoint = f)
  cnt <- counting(rl_ppo(minibatch = 144))
  expect_error(rl_train(ttt_env(), cnt$algo, total_steps = ITER, device = "cpu",
                        verbose = FALSE, checkpoint = f), "already exists")
  expect_equal(cnt$calls$n, 0)
  expect_no_error(rl_train(ttt_env(), cnt$algo, total_steps = ITER, device = "cpu",
                           verbose = FALSE, checkpoint = f, overwrite = TRUE))
})

test_that("periodic checkpoints follow environment steps", {
  dir <- ck_dir()
  f <- file.path(dir, "ck.rds")
  seen <- c()
  set.seed(8)
  rl_train(ttt_env(), rl_ppo(minibatch = 144), total_steps = 5 * ITER,
           device = "cpu", verbose = FALSE, checkpoint = f,
           checkpoint_every_steps = 2 * ITER, eval_every_steps = ITER,
           eval_fn = function(m) {
             if (file.exists(f)) seen <<- c(seen, readRDS(f)$train_state$steps)
             c(x = 1)
           })
  # written after iterations 2 and 4 (seen by the next evaluation), then at the end
  expect_equal(unique(seen), c(2, 4) * ITER)
  expect_equal(readRDS(f)$train_state$steps, 5 * ITER)
  expect_equal(nrow(readRDS(f)$train_state$eval), 5)
})

test_that("an interrupted write keeps the previous checkpoint", {
  dir <- ck_dir()
  f <- file.path(dir, "ck.rds")
  rltoolsR:::write_rds_atomic(list(a = 1), f)
  broken <- function(obj, path) {
    writeBin(as.raw(1:10), path)
    stop("disk full")
  }
  expect_error(rltoolsR:::write_rds_atomic(list(a = 2), f, save = broken), "disk full")
  expect_identical(readRDS(f), list(a = 1))
  expect_identical(list.files(dir), "ck.rds")
})

test_that("resume with append drops log rows written after the checkpoint", {
  dir <- ck_dir()
  f <- file.path(dir, "ck.rds")
  log <- file.path(dir, "run.csv")
  n <- 0
  crash_at_3 <- function(m) {
    n <<- n + 1
    if (n == 3) stop("crash")
    c(x = n)
  }
  set.seed(9)
  expect_error(rl_train(ttt_env(), rl_ppo(minibatch = 144), total_steps = 4 * ITER,
                        device = "cpu", verbose = FALSE, checkpoint = f,
                        checkpoint_every_steps = 2 * ITER, eval_every_steps = ITER,
                        eval_fn = crash_at_3, log = log), "crash")
  # history row 3 was logged after the checkpoint at iteration 2
  expect_equal(utils::read.csv(log)$iteration, 1:3)

  expect_error(rl_train(ttt_env(), rl_ppo(minibatch = 144), total_steps = 4 * ITER,
                        device = "cpu", verbose = FALSE, resume = f, log = log),
               "already exist")
  expect_message(
    m <- rl_train(ttt_env(), rl_ppo(minibatch = 144), total_steps = 4 * ITER,
                  device = "cpu", verbose = FALSE, resume = f, log = log, append = TRUE,
                  eval_every_steps = ITER, eval_fn = function(m) c(x = 0)),
    "dropped 1 row")
  h <- utils::read.csv(log)
  expect_equal(h$iteration, 1:4)
  expect_equal(h$steps, m$history$steps)
  e <- utils::read.csv(file.path(dir, "run_eval.csv"))
  expect_equal(e$iteration, 1:4)
  expect_equal(e$x, c(1, 2, 0, 0))
  expect_error(rl_train(ttt_env(), rl_ppo(), total_steps = ITER, device = "cpu",
                        verbose = FALSE, log = log, append = TRUE), "needs `resume`")
})

test_that("resume restores the Adam state; checkpoints without it warn", {
  set.seed(1)
  m <- rl_train(ttt_env(), rl_ppo(minibatch = 144), total_steps = 2 * ITER,
                device = "cpu", verbose = FALSE)
  ts <- m$train_state$algo_state
  expect_identical(ts$optimizer_state$format, "ggmlR.ag_opt_state")
  expect_gt(ts$optimizer_state$t, 0)

  algo <- rl_ppo(minibatch = 144)
  env <- ttt_env()
  st <- algo$set_train_state(algo$init(env, "cpu"), ts)
  expect_identical(st$opt$t, ts$optimizer_state$t)
  expect_identical(st$opt$state()$slots, ts$optimizer_state$slots)

  old <- ts
  old$optimizer_state <- NULL
  expect_warning(algo$set_train_state(algo$init(env, "cpu"), old),
                 "no optimizer state")
})
