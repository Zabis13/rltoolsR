# Replay buffer for off-policy algorithms (rl_tools/rl/components/replay_buffer).
#
# A ring of `capacity` transitions held in one ggmlR ag_buffer on the training
# device, one column per transition, packed as
#
#   [obs | act | reward | terminated | truncated | next_obs]
#
# so a batch is one ag_get_rows() and its fields are row slices. obs and act are
# adjacent: the critic input Q(obs, act) is one slice. next_obs holds the final
# observation of an episode on autoreset (final_obs), not the next reset
# observation. truncated is stored for diagnostics and sequential batches but
# is not read in training: only terminated stops the bootstrap.
#
# The layout is defined here and nowhere else; algorithms slice through
# rb_field() / rb_slice(), so a layout change does not reach them.
#
# Host-side state: position, size, and episode_start (the slot where the
# episode of each transition began, as in RLtools) for sequential batches.

RB_FIELDS <- c("obs", "act", "reward", "terminated", "truncated", "next_obs")

# Row ranges (0-based `from`, length `n`) of the packed transition.
rb_layout <- function(obs_dim, act_dim) {
  n <- c(obs = obs_dim, act = act_dim, reward = 1L, terminated = 1L,
         truncated = 1L, next_obs = obs_dim)
  from <- c(0L, cumsum(n)[-length(n)])
  names(from) <- names(n)
  list(from = from, n = n, rows = sum(n))
}

# A new, empty buffer on the current ag device. `n_envs` copies add one
# transition each per step.
rb_new <- function(obs_dim, act_dim, capacity, n_envs = 1L) {
  obs_dim <- as.integer(obs_dim); act_dim <- as.integer(act_dim)
  capacity <- as.integer(capacity); n_envs <- as.integer(n_envs)
  if (capacity < n_envs)
    stop("replay buffer: capacity (", capacity, ") must be at least the number ",
         "of environments (", n_envs, ")", call. = FALSE)
  layout <- rb_layout(obs_dim, act_dim)
  rb <- new.env(parent = emptyenv())
  rb$obs_dim <- obs_dim
  rb$act_dim <- act_dim
  rb$capacity <- capacity
  rb$n_envs <- n_envs
  rb$layout <- layout
  rb$buf <- ag_buffer(layout$rows, capacity)
  rb$pos <- 0L                             # next slot to write
  rb$size <- 0L                            # filled slots
  rb$episode_start <- rep(NA_integer_, capacity)
  rb$cur_start <- rep(NA_integer_, n_envs) # slot where each copy's episode began
  class(rb) <- "rl_replay_buffer"
  rb
}

rb_full <- function(rb) rb$size == rb$capacity

# Add one step of `n_envs` transitions. obs and next_obs are n x obs_dim (one
# row per copy, as rl_step() returns them), act n x act_dim, the rest length n.
rb_add <- function(rb, obs, act, reward, terminated, truncated, next_obs) {
  n <- rb$n_envs
  obs <- matrix(obs, n); act <- matrix(act, n); next_obs <- matrix(next_obs, n)
  if (ncol(obs) != rb$obs_dim || ncol(next_obs) != rb$obs_dim || ncol(act) != rb$act_dim ||
      length(reward) != n || length(terminated) != n || length(truncated) != n)
    stop("replay buffer: transition shapes do not match (", n, " environments, ",
         "obs_dim ", rb$obs_dim, ", act_dim ", rb$act_dim, ")", call. = FALSE)
  data <- rbind(t(obs), t(act), as.numeric(reward), as.numeric(terminated),
                as.numeric(truncated), t(next_obs))

  # ring write: two writes when the step crosses the end
  first <- min(n, rb$capacity - rb$pos)
  ag_buffer_write(rb$buf, data[, seq_len(first), drop = FALSE], rb$pos)
  if (first < n) ag_buffer_write(rb$buf, data[, -seq_len(first), drop = FALSE], 0L)

  slots <- (rb$pos + seq_len(n) - 1L) %% rb$capacity
  new_ep <- is.na(rb$cur_start)
  rb$cur_start[new_ep] <- slots[new_ep]
  rb$episode_start[slots + 1L] <- rb$cur_start
  rb$cur_start[as.logical(terminated) | as.logical(truncated)] <- NA_integer_

  rb$pos <- (rb$pos + n) %% rb$capacity
  rb$size <- min(rb$size + n, rb$capacity)
  invisible(rb)
}

# `batch` uniform slot indices (0-based, with replacement) from the filled
# part, drawn from R's RNG -- the training stream inside rl_train().
rb_sample_idx <- function(rb, batch) {
  if (rb$size < 1L) stop("replay buffer: sampling from an empty buffer", call. = FALSE)
  sample.int(rb$size, batch, replace = TRUE) - 1L
}

# A batch as one packed ag_tensor [rows, B] on the device. Inside ag_capture()
# pass `idx` as an argument of the captured function and list rb$buf in params.
rb_gather <- function(rb, idx) ag_get_rows(rb$buf, idx)

# One field of a packed batch.
rb_field <- function(rb, x, field) {
  field <- match.arg(field, RB_FIELDS)
  ag_slice_rows(x, rb$layout$from[[field]], rb$layout$n[[field]])
}

# Adjacent fields as one slice, e.g. c("obs", "act") for the critic input.
rb_slice <- function(rb, x, fields) {
  fields <- match.arg(fields, RB_FIELDS, several.ok = TRUE)
  i <- match(fields, RB_FIELDS)
  if (any(diff(sort(i)) != 1L))
    stop("replay buffer: fields ", paste(fields, collapse = ", "),
         " are not adjacent in the layout", call. = FALSE)
  first <- RB_FIELDS[min(i)]
  ag_slice_rows(x, rb$layout$from[[first]], sum(rb$layout$n[fields]))
}

rb_free <- function(rb) {
  if (!is.null(rb$buf)) ag_buffer_free(rb$buf)
  rb$buf <- NULL
  invisible(NULL)
}

# --- persistence ------------------------------------------------------------
#
# A file of its own next to the checkpoint, so the checkpoint stays small: a
# serialized header (dimensions, position, size, episode bookkeeping), then
# the filled slots as f32, downloaded and written in parts of `chunk` columns
# so the host never holds the whole buffer. The RNG state belongs to the
# checkpoint itself (rl_train's streams). Written atomically.

RB_FILE_FORMAT <- "rltoolsR.replay_buffer.v1"

rb_save <- function(rb, path, chunk = 65536L) {
  header <- list(format = RB_FILE_FORMAT, obs_dim = rb$obs_dim, act_dim = rb$act_dim,
                 capacity = rb$capacity, n_envs = rb$n_envs, rows = rb$layout$rows,
                 pos = rb$pos, size = rb$size, episode_start = rb$episode_start,
                 cur_start = rb$cur_start)
  write_rds_atomic(header, path, save = function(h, file) {
    con <- file(file, "wb")
    on.exit(close(con), add = TRUE)
    serialize(h, con)
    # slots 0..size-1 are the filled ones (a full ring is all of them)
    for (start in seq(0L, max(rb$size - 1L, 0L), by = chunk)) {
      if (rb$size == 0L) break
      k <- min(chunk, rb$size - start)
      writeBin(as.numeric(ag_buffer_read(rb$buf, start, k)), con, size = 4L)
    }
  })
}

# Restore into a new buffer on the current device.
rb_load <- function(path, chunk = 65536L) {
  con <- file(path, "rb")
  on.exit(close(con), add = TRUE)
  h <- unserialize(con)
  if (!identical(h$format, RB_FILE_FORMAT))
    stop("'", path, "' is not a replay buffer file", call. = FALSE)
  rb <- rb_new(h$obs_dim, h$act_dim, h$capacity, h$n_envs)
  for (start in seq(0L, max(h$size - 1L, 0L), by = chunk)) {
    if (h$size == 0L) break
    k <- min(chunk, h$size - start)
    v <- readBin(con, "double", n = h$rows * k, size = 4L)
    if (length(v) != h$rows * k) stop("'", path, "' is truncated", call. = FALSE)
    ag_buffer_write(rb$buf, matrix(v, h$rows, k), start)
  }
  rb$pos <- h$pos
  rb$size <- h$size
  rb$episode_start <- h$episode_start
  rb$cur_start <- h$cur_start
  rb
}

#' @export
print.rl_replay_buffer <- function(x, ...) {
  cat("<rl_replay_buffer> ", x$size, " / ", x$capacity, " transitions, obs ",
      x$obs_dim, ", act ", x$act_dim, ", ", x$n_envs, " env(s)\n", sep = "")
  invisible(x)
}
