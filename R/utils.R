# Evaluate `expr` without changing the caller's RNG stream: the global
# .Random.seed is saved before and restored (or removed again) afterwards.
with_preserved_seed <- function(expr) {
  has_seed <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  if (has_seed) seed <- get(".Random.seed", envir = globalenv())
  on.exit({
    if (has_seed) assign(".Random.seed", seed, envir = globalenv())
    else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE))
      rm(".Random.seed", envir = globalenv())
  }, add = TRUE)
  expr
}
