# Checkpoints: a model file (same format as rl_save_model()) with the
# training state filled in, written atomically.

# Rows of history / eval kept in a checkpoint; older rows are dropped and
# history_truncated is set.
RL_CHECKPOINT_MAX_ROWS <- 100000L

#' Save a training checkpoint
#'
#' Writes the model together with its training state (see [rl_train()]):
#' algorithm state including weights and normalizer statistics, step and
#' iteration counters, elapsed time, both RNG streams, history and
#' evaluations. The file is a regular model file: [rl_load_model()] reads it
#' as a model, `rl_train(resume = path)` continues training from it.
#'
#' Resuming continues training: weights, normalizer statistics, the Adam
#' moments and step count, counters and both RNG streams are restored. What
#' is not: the environment is reset, so an episode unfinished at the
#' checkpoint is lost. Checkpoints written before the optimizer state was
#' saved still resume, with a warning and a fresh optimizer. Weights and
#' optimizer state come back exactly on the same device and within float
#' tolerance across CPU and GPU. Of `history` and `eval` the last 100000
#' rows are kept.
#'
#' The file is written to a temporary file in the same directory and then
#' renamed over `path`, so an interrupted write leaves the previous
#' checkpoint intact.
#'
#' @param model A model returned by [rl_train()] (with `train_state`).
#' @param path File path.
#' @return `path`, invisibly.
#' @seealso [rl_train()], [rl_load_model()]
#' @export
#' @examples
#' \donttest{
#' env <- rl_vec_env("pendulum", n = 4, seed = 1)
#' m <- rl_train(env, rl_ppo(n_steps = 64, minibatch = 128), total_steps = 512,
#'               device = "cpu", verbose = FALSE)
#' f <- tempfile(fileext = ".rds")
#' rl_save_checkpoint(m, f)
#' m2 <- rl_train(env, rl_ppo(n_steps = 64, minibatch = 128), total_steps = 1024,
#'                device = "cpu", verbose = FALSE, resume = f)
#' }
rl_save_checkpoint <- function(model, path) {
  ts <- model$train_state
  if (is.null(ts)) stop("rl_save_checkpoint(): `model` has no train_state (train it with rl_train())")
  ts$history_truncated <- (!is.null(model$history) && nrow(model$history) > RL_CHECKPOINT_MAX_ROWS) ||
    (!is.null(model$eval) && nrow(model$eval) > RL_CHECKPOINT_MAX_ROWS)
  ts$history <- tail_rows(model$history)
  ts$eval <- tail_rows(model$eval)
  write_rds_atomic(model_file(model, "rl_save_checkpoint()", train_state = ts), path)
  invisible(path)
}

tail_rows <- function(df, n = RL_CHECKPOINT_MAX_ROWS) {
  if (is.null(df) || nrow(df) <= n) return(df)
  df[seq.int(nrow(df) - n + 1L, nrow(df)), , drop = FALSE]
}

# Temporary file in the target directory (a rename across file systems is
# not atomic), then rename; where rename does not replace an existing file
# (Windows), copy over it instead. `save` is replaceable for tests.
write_rds_atomic <- function(obj, path, save = saveRDS) {
  tmp <- tempfile(".rltools_", tmpdir = dirname(path), fileext = ".tmp")
  on.exit(if (file.exists(tmp)) unlink(tmp), add = TRUE)
  save(obj, tmp)
  if (!suppressWarnings(file.rename(tmp, path))) {
    if (!file.copy(tmp, path, overwrite = TRUE)) {
      stop("could not write '", path, "'")
    }
  }
  invisible(path)
}

read_checkpoint <- function(path) {
  obj <- read_model_file(path, "rl_train()")
  if (!isTRUE(obj$has_train_state) || is.null(obj$train_state)) {
    stop("rl_train(): '", path, "' has no training state (saved with ",
         "rl_save_model()?); resume needs a file from rl_save_checkpoint()")
  }
  obj
}

# Called after algo$init(), before the first iteration: the checkpoint must
# come from the same algorithm with the same network and spaces. Changed
# training hyperparameters are allowed, with a warning.
check_resume <- function(ck, algo, meta) {
  ts <- ck$train_state
  old <- ck$meta
  if (!identical(ts$algo, algo$name)) {
    stop("rl_train(): the checkpoint is from algorithm '", format(ts$algo),
         "', `algo` is '", algo$name, "'")
  }
  if (!identical(old$action_space$type, meta$action_space$type)) {
    stop("rl_train(): the checkpoint has '", old$action_space$type,
         "' actions, the environment '", meta$action_space$type, "'")
  }
  if (!identical(unclass(old$observation_space), unclass(meta$observation_space)) ||
      !identical(unclass(old$action_space), unclass(meta$action_space))) {
    stop("rl_train(): observation or action space of the checkpoint differs from the environment")
  }
  a <- model_algo(meta)
  if (!is.null(a) && !identical(lapply(a$shapes(old), as.integer),
                                lapply(a$shapes(meta), as.integer))) {
    stop("rl_train(): the network in the checkpoint has other shapes than `algo` builds")
  }
  p_old <- ts$algo_params
  p_new <- algo$params
  nms <- union(names(p_old), names(p_new))
  changed <- nms[!vapply(nms, function(n) identical(p_old[[n]], p_new[[n]]), logical(1))]
  if (length(changed)) {
    warning("rl_train(): resuming with changed hyperparameters: ",
            paste(changed, collapse = ", "), call. = FALSE)
  }
  invisible(TRUE)
}
