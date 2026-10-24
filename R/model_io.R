# Saving and loading trained models.
#
# All contact with ggmlR's serialization lives in save_net()/load_net(), on
# ggmlR's ag_state_dict()/ag_load_state_dict().

RL_MODEL_FORMAT <- "rltoolsR_model"
RL_MODEL_FORMAT_VERSION <- 1L

# Per-algorithm network rebuild from metadata, expected parameter shapes and
# model class.
rl_model_algos <- list(
  ppo = list(
    build = function(meta) ppo_build(meta),
    shapes = function(meta) {
      if (identical(meta$action_type, "box")) return(ppo_box_shapes(meta))
      h <- meta$hidden
      list(l1_W = c(h, meta$obs_dim), l1_b = c(h, 1L),
           l2_W = c(h, h), l2_b = c(h, 1L),
           pi_W = c(meta$n_actions, h), pi_b = c(meta$n_actions, 1L),
           v_W = c(1L, h), v_b = c(1L, 1L))
    },
    class = "rl_ppo"
  )
)

model_algo <- function(meta) {
  a <- meta$algo
  if (!is.character(a) || length(a) != 1L || is.na(a)) return(NULL)
  rl_model_algos[[a]]
}

# Network -> ggmlR state dict (a plain list, weights as CPU matrices).
save_net <- function(net) ag_state_dict(net)

# State dict -> network on `device`. Also reads the ag_save_model() container
# that files written before ag_state_dict() hold (same $parameters/$buffers).
# The rebuild draws random initial weights before they are overwritten, so the
# caller's RNG state is restored.
load_net <- function(container, build, device) {
  prev <- ag_device(device)
  on.exit(ag_device(prev), add = TRUE)
  net <- with_preserved_seed(build())
  ag_load_state_dict(net, container)
  ag_eval(net)
  net
}

#' Save a trained model
#'
#' Writes the network weights together with the metadata needed to rebuild
#' it (algorithm, layer sizes, observation and action spaces) to a single
#' `.rds` file. The model can be loaded on any device with [rl_load_model()],
#' e.g. trained on the GPU and run on the CPU.
#'
#' Only the weights are saved. Optimizer state, step counters and the RNG
#' state are not, so the file is for running the policy, not for resuming
#' training.
#'
#' @param model A trained model, e.g. the result of [rl_ppo_train()].
#' @param path File path.
#' @return `path`, invisibly.
#' @seealso [rl_load_model()], [rl_save_checkpoint()] to resume training
#' @export
#' @examples
#' \donttest{
#' env <- rl_gpu_env("tictactoe", n = 16, device = "cpu")
#' m <- rl_ppo_train(env, iterations = 1, device = "cpu", verbose = FALSE)
#' f <- tempfile(fileext = ".rds")
#' rl_save_model(m, f)
#' m2 <- rl_load_model(f, device = "cpu")
#' }
rl_save_model <- function(model, path) {
  saveRDS(model_file(model, "rl_save_model()"), path)
  invisible(path)
}

# Model -> file object. `train_state` is NULL for a plain model; a checkpoint
# fills it. has_train_state tells the two apart without reading the field.
model_file <- function(model, who, train_state = NULL) {
  meta <- model$meta
  if (is.null(model_algo(meta))) {
    stop(who, ": `model` has no known algorithm metadata")
  }
  meta$pkg_version <- utils::packageVersion("rltoolsR")
  meta$ggmlR_version <- utils::packageVersion("ggmlR")
  list(
    format = RL_MODEL_FORMAT,
    format_version = RL_MODEL_FORMAT_VERSION,
    meta = meta,
    net = save_net(model$net),
    has_train_state = !is.null(train_state),
    train_state = train_state
  )
}

# Reads and checks the format fields of a model or checkpoint file.
read_model_file <- function(path, who) {
  obj <- readRDS(path)
  if (!is.list(obj) || !identical(obj$format, RL_MODEL_FORMAT)) {
    stop(who, ": '", path, "' is not an rltoolsR model file")
  }
  fv <- obj$format_version
  if (!is.numeric(fv) || length(fv) != 1L || is.na(fv) || fv < 1) {
    stop(who, ": invalid format_version in '", path, "'")
  }
  if (fv > RL_MODEL_FORMAT_VERSION) {
    stop(who, ": '", path, "' has format version ", fv,
         ", this rltoolsR supports up to ", RL_MODEL_FORMAT_VERSION,
         ". Update rltoolsR.")
  }
  obj
}

#' Load a trained model
#'
#' Reads a file written by [rl_save_model()], rebuilds the network from the
#' stored metadata and places it on `device`.
#'
#' The file is read with [readRDS()], which can deserialize arbitrary R
#' objects: load only files from trusted sources.
#'
#' @param path File path written by [rl_save_model()] or
#'   [rl_save_checkpoint()] (the training state is ignored).
#' @param device `"cpu"` (default), `"gpu"` or `"auto"` (GPU if a Vulkan
#'   device is available, else CPU). The CPU is the default because a small
#'   policy runs faster there: on tic-tac-toe PPO (hidden 64) one `rl_act()`
#'   call takes 0.36 ms on the CPU vs 8.6 ms on the GPU for batch 1 and
#'   1.6 ms vs 9.2 ms for batch 256 (`inst/examples/profile_ppo_gpu.R`).
#'   Use the GPU for large batches or large networks. The default used to
#'   be `"auto"`.
#' @return A model of the saved algorithm's class (e.g. `rl_ppo`), usable
#'   with [rl_act()].
#' @seealso [rl_save_model()]
#' @export
#' @examples
#' \donttest{
#' env <- rl_gpu_env("tictactoe", n = 16, device = "cpu")
#' m <- rl_ppo_train(env, iterations = 1, device = "cpu", verbose = FALSE)
#' f <- tempfile(fileext = ".rds")
#' rl_save_model(m, f)
#' m2 <- rl_load_model(f)
#' rl_ppo_eval_tictactoe(m2, n = 50)
#' }
rl_load_model <- function(path, device = c("cpu", "gpu", "auto")) {
  device <- rl_resolve_device(match.arg(device))
  obj <- read_model_file(path, "rl_load_model()")

  meta <- obj$meta
  algo <- model_algo(meta)
  if (is.null(algo)) {
    stop("rl_load_model(): unknown algorithm '", format(meta$algo), "'")
  }
  need <- meta$ggmlR_version
  if (!is.null(need) && utils::packageVersion("ggmlR") < need) {
    stop("rl_load_model(): the model was saved with ggmlR ", format(need),
         ", installed is ", format(utils::packageVersion("ggmlR")),
         ". Update ggmlR.")
  }

  expected <- algo$shapes(meta)
  saved <- obj$net$parameters
  if (!setequal(names(expected), names(saved))) {
    stop("rl_load_model(): parameters in the file (",
         paste(names(saved), collapse = ", "), ") do not match the '",
         meta$algo, "' network (", paste(names(expected), collapse = ", "), ")")
  }
  for (nm in names(expected)) {
    got <- dim(as.matrix(saved[[nm]]))
    if (!identical(as.integer(got), as.integer(expected[[nm]]))) {
      stop("rl_load_model(): parameter '", nm, "' has shape ",
           paste(got, collapse = "x"), ", metadata expects ",
           paste(expected[[nm]], collapse = "x"))
    }
  }

  net <- load_net(obj$net, function() algo$build(meta), device)
  structure(list(net = net, device = device, meta = meta, history = NULL),
            class = algo$class)
}
