#' Action and observation spaces
#'
#' Gymnasium-style space descriptions. Values are stored row-major.
#'
#' @param low,high Lower and upper bounds, recycled to `prod(shape)`.
#' @param shape Integer dimensions.
#' @param dtype Element type: `"float32"`, `"int32"` or `"uint8"`.
#' @param n Number of discrete values (`0..n-1`).
#' @param nvec Number of values for each discrete component.
#' @param ... Named spaces for `space_dict()`.
#' @return An object of class `rl_space`.
#' @name spaces
#' @examples
#' space_box(-1, 1, shape = 3)
#' space_discrete(12)
NULL

new_space <- function(type, ...) {
  structure(list(type = type, ...), class = c(paste0("rl_space_", type), "rl_space"))
}

#' @rdname spaces
#' @export
space_box <- function(low, high, shape = length(low), dtype = "float32") {
  shape <- as.integer(shape)
  size <- prod(shape)
  dtype <- match.arg(dtype, c("float32", "int32", "uint8"))
  new_space("box", dtype = dtype, shape = shape,
            low = rep_len(as.numeric(low), size),
            high = rep_len(as.numeric(high), size))
}

#' @rdname spaces
#' @export
space_discrete <- function(n) {
  new_space("discrete", dtype = "int32", n = as.integer(n))
}

#' @rdname spaces
#' @export
space_multi_discrete <- function(nvec) {
  new_space("multi_discrete", dtype = "int32", nvec = as.integer(nvec))
}

#' @rdname spaces
#' @export
space_multi_binary <- function(shape) {
  new_space("multi_binary", dtype = "uint8", shape = as.integer(shape))
}

#' @rdname spaces
#' @export
space_dict <- function(...) {
  spaces <- list(...)
  if (length(spaces) == 0 || is.null(names(spaces)) || any(names(spaces) == "")) {
    stop("all spaces in space_dict() must be named")
  }
  if (!all(vapply(spaces, inherits, logical(1), "rl_space"))) {
    stop("space_dict() accepts only rl_space objects")
  }
  new_space("dict", spaces = spaces)
}

#' Size of a flat sample from a space
#'
#' @param space An `rl_space`.
#' @return Integer number of elements.
#' @export
#' @examples
#' space_size(space_box(-1, 1, shape = c(2, 3)))
#' space_size(space_dict(pos = space_box(0, 1, shape = 2), cell = space_discrete(9)))
space_size <- function(space) {
  switch(space$type,
    box = ,
    multi_binary = as.integer(prod(space$shape)),
    discrete = 1L,
    multi_discrete = length(space$nvec),
    dict = sum(vapply(space$spaces, space_size, integer(1))),
    stop("unknown space type: ", space$type)
  )
}

#' @export
format.rl_space <- function(x, ...) {
  switch(x$type,
    box = sprintf("Box(%s, shape = (%s), %s)",
                  if (length(x$low) <= 4) paste0("[", paste(signif(x$low, 4), collapse = ", "), "] .. [",
                                                 paste(signif(x$high, 4), collapse = ", "), "]")
                  else sprintf("[%g .. %g]", min(x$low), max(x$high)),
                  paste(x$shape, collapse = ", "), x$dtype),
    discrete = sprintf("Discrete(%d)", x$n),
    multi_discrete = sprintf("MultiDiscrete(%s)", paste(x$nvec, collapse = ", ")),
    multi_binary = sprintf("MultiBinary(%s)", paste(x$shape, collapse = ", ")),
    dict = sprintf("Dict(%s)", paste(names(x$spaces), vapply(x$spaces, format, character(1)),
                                     sep = ": ", collapse = ", ")),
    x$type
  )
}

#' @export
print.rl_space <- function(x, ...) {
  cat(format(x), "\n", sep = "")
  invisible(x)
}
