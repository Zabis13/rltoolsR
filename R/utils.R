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

# Private RNG streams of rl_gpu_env (Gymnasium's reset(seed = ...)): after
# rl_reset(env, seed = s) the environment draws its random starts, scrambles
# and opponent moves from its own stream, seeded with s, and the caller's
# stream is left as it was. Without a seed ever given the environment draws
# from R's global stream (set.seed() reproduces it), as before.
gpu_rng_seed <- function(st, seed) {
  seed <- as.numeric(seed)
  if (length(seed) != 1L || is.na(seed) || seed < 0) stop("seed must be one non-negative number")
  st$rng <- with_preserved_seed({
    set.seed(seed)
    get(".Random.seed", envir = globalenv())
  })
  invisible(st)
}

# Runs f() on the environment's stream (when it has one) and keeps the
# advanced stream for the next draw.
gpu_rng <- function(st, f) {
  if (is.null(st$rng)) return(f())
  with_preserved_seed({
    assign(".Random.seed", st$rng, envir = globalenv())
    out <- f()
    st$rng <- get(".Random.seed", envir = globalenv())
    out
  })
}
