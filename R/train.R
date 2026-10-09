# Algorithm-independent training loop.
#
# rl_train() owns the environment, the device, the step budget, evaluation,
# logging and the two RNG streams. The algorithm owns everything else in an
# opaque `state` that rl_train() only passes along: rollouts or a replay
# buffer, networks, optimizers. No batch crosses the contract, so on-policy
# and off-policy algorithms fit the same loop.

#' Define a training algorithm
#'
#' Bundles the functions [rl_train()] calls. `state` is whatever the
#' algorithm needs (networks, optimizer, rollout or replay buffer, last
#' observation); `rl_train()` passes it along without looking inside.
#'
#' Everything random in `collect` and `update` (action sampling, minibatch
#' shuffling, replay sampling) must use R's RNG: it is the training stream,
#' which `rl_train()` keeps separate from evaluation.
#'
#' @param name Algorithm name, e.g. `"ppo"`.
#' @param init `function(env, device)` returning the initial state. `env` is
#'   vectorized, the device is already active.
#' @param collect `function(state, env)` stepping the environment and
#'   storing the data in the state. Returns `list(state, steps, info)`, where
#'   `steps` is the number of environment transitions taken (over all copies)
#'   and `info` an optional named numeric vector for the history.
#' @param update `function(state)` learning from the stored data. Returns
#'   `list(state, metrics)`, `metrics` an optional named numeric vector.
#' @param model `function(state)` returning the policy as a model object
#'   (usable with [rl_act()] and [rl_save_model()]).
#' @param get_train_state `function(state)` returning everything needed to
#'   resume: weights, normalizer statistics, `optimizer_state` and counters.
#' @param set_train_state `function(state, train_state)` restoring it.
#'   Environment-facing parts of the state (last observation, unfinished
#'   rollout) are not restored: training resumes from a reset.
#' @param params Named list of training hyperparameters, stored in
#'   checkpoints; resuming with different values gives a warning.
#' @return An object of class `rl_algo`.
#' @seealso [rl_train()], [rl_ppo()]
#' @export
#' @examples
#' # a random policy on Pendulum: no learning, only the interface
#' algo <- rl_algo("random",
#'   init = function(env, device) list(),
#'   collect = function(state, env) {
#'     r <- rl_step(env, stats::runif(env$n, -2, 2))
#'     list(state = state, steps = env$n, info = c(reward = mean(r$reward)))
#'   },
#'   update = function(state) list(state = state, metrics = NULL),
#'   model = function(state) list(meta = list()),
#'   get_train_state = function(state) list(optimizer_state = NULL),
#'   set_train_state = function(state, train_state) state)
#' env <- rl_vec_env("pendulum", n = 4, seed = 1)
#' rl_reset(env)
#' m <- rl_train(env, algo, total_steps = 400, device = "cpu", verbose = FALSE)
#' head(m$history)
rl_algo <- function(name, init, collect, update, model, get_train_state,
                    set_train_state, params = list()) {
  stopifnot(is.character(name), length(name) == 1L, is.list(params))
  fns <- list(init = init, collect = collect, update = update, model = model,
              get_train_state = get_train_state,
              set_train_state = set_train_state)
  for (nm in names(fns)) {
    if (!is.function(fns[[nm]])) stop("rl_algo(): `", nm, "` must be a function")
  }
  structure(c(list(name = name, params = params), fns), class = "rl_algo")
}

#' Train an agent
#'
#' Runs `collect` + `update` of `algo` until `total_steps` environment
#' transitions have been taken, evaluating every `eval_every_steps`.
#'
#' Training and evaluation use two separate RNG streams. The evaluation
#' stream is seeded from the state of the training stream at the start
#' (without drawing from it) and advances only during evaluation, so how
#' often the agent is evaluated does not change training. On return the
#' global RNG holds the training stream.
#'
#' @param env A vectorized environment (`rl_gpu_env`, `rl_vec_env`) or a
#'   single [rl_env()], which is wrapped as one copy and is then owned by
#'   the training loop.
#' @param algo An [rl_algo()], e.g. [rl_ppo()].
#' @param total_steps Budget in environment transitions over all copies.
#'   Training stops after the first iteration that reaches it.
#' @param device `"auto"` (GPU if a Vulkan device is available, else CPU),
#'   `"gpu"` or `"cpu"`. Pass it explicitly when results must be
#'   reproducible.
#' @param eval_every_steps Evaluate each time the step count crosses a
#'   multiple of this (0 = never), using `eval_fn(model)`.
#' @param eval_fn Function of the model returning a named numeric vector.
#' @param verbose Print progress.
#' @param log Optional CSV file path. The history is written to `log` and the
#'   evaluations to the same name with an `_eval` suffix (`run.csv` and
#'   `run_eval.csv`), one row per iteration / evaluation as soon as it is
#'   done. Both files start with `iteration`, `steps` and `time` (seconds
#'   since the start of `rl_train()`), followed by the history or `eval_fn`
#'   columns. Numbers are written in full precision with `.` as the decimal
#'   separator regardless of locale, missing values as `NA`. If a write
#'   fails, a warning is given and logging is switched off for the rest of
#'   the run instead of stopping a long training.
#' @param overwrite Replace existing log and checkpoint files. Without it
#'   `rl_train()` stops before training if any of them exists (the
#'   checkpoint being resumed from may be overwritten).
#' @param checkpoint Optional file path for checkpoints (see
#'   [rl_save_checkpoint()]). Written each time the step count crosses a
#'   multiple of `checkpoint_every_steps` (after that iteration's
#'   evaluation) and always at the end of training.
#' @param checkpoint_every_steps Checkpoint interval in environment steps
#'   (0 = only at the end).
#' @param resume Optional checkpoint file to continue from. Weights,
#'   normalizer statistics, counters, elapsed time, both RNG streams,
#'   history and evaluations are restored; `total_steps` is the total
#'   budget, so if the checkpoint already reached it the model is returned
#'   without training. The optimizer state (Adam moments and step count) is
#'   restored too; the environment is reset (an episode unfinished at the
#'   checkpoint is lost). The checkpoint must come from the same algorithm,
#'   network and spaces;
#'   changed hyperparameters give a warning.
#' @param append With `resume`, continue the existing log files: rows after
#'   the checkpoint's step (written after the last checkpoint) are dropped,
#'   then new rows are appended.
#' @return The model from `algo$model()` with `history` (one row per
#'   iteration), `eval` (results of `eval_fn`, or `NULL`) and `train_state`
#'   (algorithm state, hyperparameters, step counters, elapsed `time` and
#'   both RNG streams). `optimizer_state` holds the optimizer's moments and
#'   step count (`opt$state()` from ggmlR).
#'
#'   `info`, `metrics` and the result of `eval_fn` must have the same names
#'   in every iteration (use `NA` for a value that is missing); otherwise
#'   training stops.
#' @seealso [rl_algo()], [rl_ppo()], [rl_save_checkpoint()]
#' @export
#' @examples
#' \donttest{
#' env <- rl_vec_env("pendulum", n = 4, seed = 1)
#' m <- rl_train(env, rl_ppo(n_steps = 64, minibatch = 128), total_steps = 1024,
#'               device = "cpu", verbose = FALSE,
#'               checkpoint = tempfile(fileext = ".rds"))
#' head(m$history)
#' }
rl_train <- function(env, algo, total_steps, device = c("auto", "gpu", "cpu"),
                     eval_every_steps = 0, eval_fn = NULL, verbose = TRUE,
                     log = NULL, overwrite = FALSE, checkpoint = NULL,
                     checkpoint_every_steps = 0, resume = NULL, append = FALSE) {
  if (!inherits(algo, "rl_algo")) stop("rl_train(): `algo` must be an rl_algo object")
  device <- rl_resolve_device(match.arg(device))
  env <- as_vec_env(env)
  ck <- if (!is.null(resume)) read_checkpoint(resume)
  if (isTRUE(append) && is.null(ck)) {
    stop("rl_train(): append = TRUE continues the logs of a resumed run and needs `resume`")
  }
  if (!is.null(checkpoint)) check_checkpoint_path(checkpoint, overwrite, resume)
  prev <- ag_device(device)
  on.exit(ag_device(prev), add = TRUE)
  # GPU modes for this call, restored on exit (also on error):
  # - graph mode: forward ops run as one fused graph instead of one launch each
  #   (collect -31%, 140 -> 10 launches per iteration, measured);
  # - f32-tile matmuls: otherwise a Vulkan fp16/coopmat device computes f32 x f32
  #   with f16 tiles once a batch exceeds 8 columns, so logp at collection (few
  #   envs) and in the update (larger minibatch) disagree by up to ~1e-3. Free
  #   at these layer sizes (measured).
  ag_local_mode(graph = TRUE, matmul_precision = "f32")

  if (is.null(ck)) {
    streams <- rng_streams()
    state <- algo$init(env, device)
    steps <- 0
    it <- 0L
    t_offset <- 0
    history <- list()
    evals <- list()
  } else {
    # init draws from the RNG; the saved streams replace it afterwards
    state <- algo$init(env, device)
    check_resume(ck, algo, algo$model(state)$meta)
    ts <- ck$train_state
    state <- algo$set_train_state(state, ts$algo_state)
    assign(".Random.seed", ts$rng$train, envir = globalenv())
    streams <- new.env(parent = emptyenv())
    streams$eval <- ts$rng$eval
    steps <- ts$steps
    it <- ts$iteration
    t_offset <- ts$time
    history <- df_rows(ts$history)
    evals <- df_rows(ts$eval)
  }
  t0 <- proc.time()[["elapsed"]]
  elapsed <- function() t_offset + proc.time()[["elapsed"]] - t0
  finish <- function() finish_model(algo, state, history, evals, steps, it, elapsed(), streams)
  if (!is.null(ck) && steps >= total_steps) {
    message("rl_train(): the checkpoint already reached total_steps (", steps,
            " >= ", total_steps, "), nothing to train")
    return(finish())
  }

  logger <- if (!is.null(log)) csv_logger(log, overwrite, append, keep_steps = steps)
  eval_on <- eval_every_steps > 0 && !is.null(eval_fn)
  hooks <- train_hooks(verbose, eval_on, logger)
  ckpt_on <- !is.null(checkpoint)
  saved_at <- NA

  while (steps < total_steps) {
    it <- it + 1L
    col <- algo$collect(state, env)
    if (!isTRUE(col$steps > 0)) stop("rl_train(): `collect` took no environment steps")
    state <- col$state
    before <- steps
    steps <- steps + col$steps
    upd <- algo$update(state)
    state <- upd$state

    row <- c(iteration = it, steps = steps, col$info, upd$metrics)
    check_row_names(row, history, "`collect` info and `update` metrics")
    history[[length(history) + 1L]] <- row
    ctx <- list(iteration = it, steps = steps, row = row, time = elapsed())
    run_hooks(hooks, "after_iteration", ctx)
    if (eval_on && crossed(before, steps, eval_every_steps)) {
      res <- with_eval_stream(streams, eval_fn(algo$model(state)))
      ctx$eval <- c(iteration = it, steps = steps, res)
      check_row_names(ctx$eval, evals, "`eval_fn` results")
      ctx$time <- elapsed()
      evals[[length(evals) + 1L]] <- ctx$eval
      run_hooks(hooks, "after_eval", ctx)
    }
    # after the evaluation, so the eval stream and rows are current
    if (ckpt_on && checkpoint_every_steps > 0 && crossed(before, steps, checkpoint_every_steps)) {
      rl_save_checkpoint(finish(), checkpoint)
      saved_at <- steps
    }
  }

  model <- finish()
  if (ckpt_on && !identical(saved_at, steps)) rl_save_checkpoint(model, checkpoint)
  model
}

crossed <- function(before, after, every) after %/% every > before %/% every

finish_model <- function(algo, state, history, evals, steps, it, time, streams) {
  model <- algo$model(state)
  model$history <- as.data.frame(do.call(rbind, history))
  model$eval <- if (length(evals)) as.data.frame(do.call(rbind, evals)) else NULL
  model$train_state <- list(algo = algo$name, algo_params = algo$params,
                            steps = steps, iteration = it, time = time,
                            rng = rng_state(streams),
                            algo_state = algo$get_train_state(state))
  model
}

# data.frame -> list of named numeric rows, as the loop builds them
df_rows <- function(df) {
  if (is.null(df) || !nrow(df)) return(list())
  lapply(seq_len(nrow(df)), function(i) unlist(df[i, , drop = FALSE]))
}

check_checkpoint_path <- function(path, overwrite, resume) {
  if (!is.character(path) || length(path) != 1L || is.na(path) || !nzchar(path)) {
    stop("rl_train(): `checkpoint` must be a file path")
  }
  if (!dir.exists(dirname(path))) {
    stop("rl_train(): checkpoint directory '", dirname(path), "' does not exist")
  }
  same <- !is.null(resume) &&
    identical(normalizePath(path, mustWork = FALSE), normalizePath(resume, mustWork = FALSE))
  if (file.exists(path) && !same && !isTRUE(overwrite)) {
    stop("rl_train(): checkpoint file '", path, "' already exists. ",
         "Pass overwrite = TRUE to replace it.")
  }
}

# rbind() matches by position, so rows with other names would silently
# shift columns.
check_row_names <- function(row, rows, what) {
  if (length(rows) && !identical(names(row), names(rows[[1]]))) {
    stop("rl_train(): ", what, " must have the same names every iteration: (",
         paste(names(rows[[1]]), collapse = ", "), ") then (",
         paste(names(row), collapse = ", "), ")")
  }
}

# ---- extension points ----
#
# Hooks are lists of functions of `ctx` per event: "after_iteration" gets
# iteration, steps, time and the history row; "after_eval" also the eval
# row. Progress printing and the CSV logger are hooks; checkpoints plug in
# here.

train_hooks <- function(verbose, eval_on, logger = NULL) {
  show <- function(x) message(paste(sprintf("%s=%.4g", names(x), x), collapse = " "))
  hooks <- list(after_iteration = list(), after_eval = list())
  if (verbose && !eval_on) hooks$after_iteration <- list(function(ctx) show(ctx$row))
  if (verbose && eval_on) {
    hooks$after_eval <- list(function(ctx) show(c(ctx$row, ctx$eval[-(1:2)])))
  }
  if (!is.null(logger)) {
    hooks$after_iteration <- c(hooks$after_iteration, logger$after_iteration)
    hooks$after_eval <- c(hooks$after_eval, logger$after_eval)
  }
  hooks
}

run_hooks <- function(hooks, event, ctx) {
  for (h in hooks[[event]]) h(ctx)
  invisible(NULL)
}

# ---- RNG streams ----
#
# Both streams live in R's global generator. The training stream is the
# global .Random.seed itself; the evaluation stream is kept here and swapped
# in only for the duration of an evaluation.

rng_streams <- function() {
  if (!exists(".Random.seed", envir = globalenv(), inherits = FALSE)) set.seed(NULL)
  train <- get(".Random.seed", envir = globalenv())
  # derived from the training state without drawing from it
  eval_seed <- as.integer(sum(as.numeric(train) %% 2147483647) %% 2147483647)
  streams <- new.env(parent = emptyenv())
  streams$eval <- with_preserved_seed({
    set.seed(eval_seed)
    get(".Random.seed", envir = globalenv())
  })
  streams
}

with_eval_stream <- function(streams, expr) {
  train <- get(".Random.seed", envir = globalenv())
  assign(".Random.seed", streams$eval, envir = globalenv())
  on.exit({
    streams$eval <- get(".Random.seed", envir = globalenv())
    assign(".Random.seed", train, envir = globalenv())
  }, add = TRUE)
  expr
}

rng_state <- function(streams) {
  list(train = get(".Random.seed", envir = globalenv()), eval = streams$eval)
}
