local_dir <- function() {
  d <- tempfile("rlt_log_")
  dir.create(d)
  d
}

log_run <- function(log, overwrite = FALSE, eval_fn = function(m) c(score = 1.5, miss = NA),
                    total_steps = 3 * 9 * 16, algo = rl_ppo(minibatch = 144)) {
  set.seed(13)
  env <- rl_gpu_env("tictactoe", n = 16, device = "cpu")
  rl_train(env, algo, total_steps = total_steps, device = "cpu",
           eval_every_steps = 2 * 144, eval_fn = eval_fn, verbose = FALSE,
           log = log, overwrite = overwrite)
}

test_that("the CSV logs match history and eval", {
  dir <- local_dir()
  f <- file.path(dir, "run.csv")
  m <- log_run(f)
  fe <- file.path(dir, "run_eval.csv")
  expect_true(file.exists(fe))

  h <- utils::read.csv(f)
  expect_identical(names(h), c("iteration", "steps", "time", names(m$history)[-(1:2)]))
  expect_equal(nrow(h), nrow(m$history))
  expect_equal(h[names(m$history)], m$history, tolerance = 1e-15)
  expect_true(all(diff(h$time) >= 0))

  e <- utils::read.csv(fe)
  expect_identical(names(e), c("iteration", "steps", "time", "score", "miss"))
  expect_equal(nrow(e), nrow(m$eval))
  expect_equal(e$steps, m$eval$steps)
  expect_true(all(is.na(e$miss)))
  expect_equal(m$train_state$time >= max(h$time), TRUE)
})

test_that("numbers are written in full precision with a dot in any locale", {
  dir <- local_dir()
  f <- file.path(dir, "run.csv")
  old <- options(OutDec = ",")
  m <- tryCatch(log_run(f), finally = options(old))
  expect_identical(utils::read.csv(f)$pg, m$history$pg)
})

test_that("existing logs stop rl_train before the first iteration", {
  dir <- local_dir()
  f <- file.path(dir, "run.csv")
  log_run(f)
  before <- readLines(f)
  calls <- 0
  algo <- rl_ppo(minibatch = 144)
  collect <- algo$collect
  algo$collect <- function(state, env) { calls <<- calls + 1; collect(state, env) }
  expect_error(log_run(f, algo = algo), "already exist")
  expect_equal(calls, 0)
  expect_identical(readLines(f), before)

  # either file alone blocks the run
  unlink(f)
  expect_error(log_run(f, algo = algo), "run_eval.csv")
  expect_equal(calls, 0)

  m <- log_run(f, overwrite = TRUE)
  expect_equal(nrow(utils::read.csv(f)), nrow(m$history))
})

test_that("an interrupted run leaves readable logs", {
  dir <- local_dir()
  f <- file.path(dir, "run.csv")
  n_eval <- 0
  boom <- function(m) {
    n_eval <<- n_eval + 1
    if (n_eval == 2) stop("interrupted")
    c(score = 1)
  }
  expect_error(log_run(f, eval_fn = boom, total_steps = 5 * 144), "interrupted")
  h <- utils::read.csv(f)
  expect_equal(h$iteration, 1:4)
  e <- utils::read.csv(file.path(dir, "run_eval.csv"))
  expect_equal(e$iteration, 2L)
})

test_that("a failed write warns and training continues", {
  dir <- local_dir()
  sub <- file.path(dir, "logs")
  dir.create(sub)
  f <- file.path(sub, "run.csv")
  gone <- function(m) {
    unlink(sub, recursive = TRUE)
    c(score = 1)
  }
  expect_warning(m <- log_run(f, eval_fn = gone), "logging is switched off")
  expect_equal(nrow(m$history), 3)
})

test_that("log paths are checked before training", {
  expect_equal(rltoolsR:::log_eval_path("a/run.csv"), "a/run_eval.csv")
  expect_equal(rltoolsR:::log_eval_path("a/run.CSV"), "a/run_eval.CSV")
  expect_equal(rltoolsR:::log_eval_path("a/run"), "a/run_eval.csv")
  expect_error(log_run(file.path(tempdir(), "no_such_dir", "run.csv")), "does not exist")
  expect_error(log_run(c("a", "b")), "file path")
})

test_that("changing metric names stops training", {
  n <- 0
  shifty <- rl_algo("shifty", init = function(env, device) list(),
                    collect = function(state, env) list(state = state, steps = 1),
                    update = function(state) {
                      n <<- n + 1
                      list(state = state, metrics = if (n == 1) c(a = 1) else c(b = 2))
                    },
                    model = function(state) list(),
                    get_train_state = function(state) list(),
                    set_train_state = function(state, ts) state)
  env <- rl_gpu_env("tictactoe", n = 1, device = "cpu")
  expect_error(rl_train(env, shifty, total_steps = 3, device = "cpu", verbose = FALSE),
               "same names every iteration")
})
