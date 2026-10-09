#' Version of the bundled RLtools headers
#'
#' @return Character string with the RLtools commit hash.
#' @export
#' @examples
#' rltools_version()
rltools_version <- function() {
  .Call(rlt_version)
}
