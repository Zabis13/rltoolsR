# CSV logging for rl_train().
#
# One internal logger object writes two files: the history (one row per
# iteration) at `path` and the evaluations next to it with an "_eval"
# suffix. Columns are fixed by the first row; every row starts with
# iteration, steps and time (seconds since the start of rl_train()).
# Numbers are written with sprintf("%.17g"): full precision, "." as the
# decimal separator in any locale, NA as "NA".
#
# A failed write must not end a long run: it warns and switches the logger
# off. A change of columns is a programming error and stops training.

log_eval_path <- function(path) {
  if (grepl("\\.csv$", path, ignore.case = TRUE)) {
    sub("(\\.csv)$", "_eval\\1", path, ignore.case = TRUE)
  } else {
    paste0(path, "_eval.csv")
  }
}

# Checks run before training starts, so a refused log never costs a partial
# run: both paths distinct, directory present, no existing file unless
# `overwrite` (then both are removed) or `append` (resume: rows after
# `keep_steps` are dropped and the existing header fixes the columns).
csv_logger <- function(path, overwrite = FALSE, append = FALSE, keep_steps = 0) {
  if (!is.character(path) || length(path) != 1L || is.na(path) || !nzchar(path)) {
    stop("rl_train(): `log` must be a file path")
  }
  if (isTRUE(append) && isTRUE(overwrite)) {
    stop("rl_train(): `append` and `overwrite` exclude each other")
  }
  files <- c(history = path, eval = log_eval_path(path))
  norm <- normalizePath(files, mustWork = FALSE)
  if (identical(norm[[1]], norm[[2]])) {
    stop("rl_train(): history and evaluation logs would share the file '", path, "'")
  }
  if (!dir.exists(dirname(path))) {
    stop("rl_train(): log directory '", dirname(path), "' does not exist")
  }
  existing <- files[file.exists(files)]
  if (length(existing) && !isTRUE(overwrite) && !isTRUE(append)) {
    stop("rl_train(): log file(s) already exist: ", paste(existing, collapse = ", "),
         ". Pass overwrite = TRUE to replace them.")
  }
  if (isTRUE(overwrite)) unlink(existing)

  lg <- new.env(parent = emptyenv())
  lg$files <- files
  lg$active <- TRUE
  lg$cols <- list()
  if (isTRUE(append)) {
    for (kind in names(existing)) lg$cols[[kind]] <- trim_log(existing[[kind]], keep_steps)
  }

  write_row <- function(kind, row) {
    if (!lg$active) return(invisible(FALSE))
    cols <- lg$cols[[kind]]
    first <- is.null(cols)
    if (first) {
      lg$cols[[kind]] <- names(row)
    } else if (!identical(names(row), cols)) {
      stop("rl_train(): ", kind, " log columns changed from (",
           paste(cols, collapse = ", "), ") to (", paste(names(row), collapse = ", "), ")")
    }
    lines <- paste(sprintf("%.17g", unname(row)), collapse = ",")
    if (first) lines <- c(paste(names(row), collapse = ","), lines)
    ok <- tryCatch({
      suppressWarnings(cat(paste0(lines, "\n"), file = lg$files[[kind]], append = TRUE, sep = ""))
      TRUE
    }, error = function(e) {
      warning("rl_train(): writing '", lg$files[[kind]], "' failed (",
              conditionMessage(e), "); logging is switched off, training continues",
              call. = FALSE)
      FALSE
    })
    if (!ok) lg$active <- FALSE
    invisible(ok)
  }

  # row layout: iteration, steps, time, then the rest
  with_time <- function(row, time) c(row[1:2], time = time, row[-(1:2)])
  lg$after_iteration <- function(ctx) write_row("history", with_time(ctx$row, ctx$time))
  lg$after_eval <- function(ctx) write_row("eval", with_time(ctx$eval, ctx$time))
  lg
}

# Drops rows with steps > keep_steps (logged after the checkpoint being
# resumed) and malformed rows (a write cut short), rewriting the file
# atomically. Returns the header, or NULL for an empty file (removed, so a
# header is written again).
trim_log <- function(file, keep_steps) {
  lines <- readLines(file)
  if (!length(lines)) {
    unlink(file)
    return(NULL)
  }
  header <- strsplit(lines[1], ",", fixed = TRUE)[[1]]
  k <- match("steps", header)
  if (is.na(k)) stop("rl_train(): log '", file, "' has no steps column")
  fields <- strsplit(lines[-1], ",", fixed = TRUE)
  st <- suppressWarnings(as.numeric(vapply(fields, `[`, "", k)))
  keep <- lengths(fields) == length(header) & !is.na(st) & st <= keep_steps
  if (!all(keep)) {
    write_rds_atomic(c(lines[1], lines[-1][keep]), file,
                     save = function(obj, f) writeLines(obj, f))
    message("rl_train(): dropped ", sum(!keep), " row(s) after step ", keep_steps,
            " from '", file, "'")
  }
  header
}
