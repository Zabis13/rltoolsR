# Cube scrambles in C (src/cube_walk.cpp) against the R reference
# (cube_walk_actions_r, cube_host_scramble_r in R/gpu_cube.R). The random
# streams differ, so walks are compared by rule and by move frequencies, and
# the move application bit for bit on the same words.

cube_st <- function() rl_gpu_env("cube", n = 1, size = 3, scramble = 1, device = "cpu")$st

test_that("C move application equals the R reference, identity row included", {
  st <- cube_st()
  set.seed(1)
  acts <- matrix(sample.int(st$M + 1L, 50 * 12, TRUE) - 1L, 50, 12)   # M = identity
  expect_identical(rltoolsR:::cube_host_scramble(st$perms, acts),
                   rltoolsR:::cube_host_scramble_r(st$perms, acts))
  # no moves: solved
  expect_identical(rltoolsR:::cube_host_scramble(st$perms, matrix(0L, 3, 0)),
                   rltoolsR:::cube_host_scramble_r(st$perms, matrix(0L, 3, 0)))
})

test_that("cube_scramble returns the colours of its own walks", {
  st <- cube_st()
  set.seed(2)
  s <- rltoolsR:::cube_scramble(st$perms, 40, 20, st$M)
  expect_identical(dim(s$acts), c(40L, 20L))
  expect_identical(s$col, rltoolsR:::cube_host_scramble_r(st$perms, s$acts))
  set.seed(2)
  expect_identical(rltoolsR:::cube_walk_actions(40, 20, st$M), s$acts)   # same stream
})

# Per step and move, the share of walks taking that move: C against R. A move
# the rule forbids has share 0 in both.
freq <- function(a, M) sapply(seq_len(ncol(a)), function(t) tabulate(a[, t] + 1L, M) / nrow(a))

test_that("no_undo walks: never the inverse of the last move, uniform otherwise", {
  st <- cube_st()
  set.seed(3)
  a <- rltoolsR:::cube_walk_actions(20000, 6, st$M)
  expect_true(all(a >= 0L & a < st$M))
  expect_false(any(a[, -1] == bitwXor(a[, -ncol(a)], 1L)))
  r <- rltoolsR:::cube_walk_actions_r(20000, 6, st$M)
  fc <- freq(a, st$M); fr <- freq(r, st$M)
  expect_lt(max(abs(fc - fr)), 0.015)
})

test_that("no_cancel walks follow the R rule", {
  st <- cube_st()
  set.seed(4)
  a <- rltoolsR:::cube_walk_actions(20000, 8, st$M, st$alg)
  r <- rltoolsR:::cube_walk_actions_r(20000, 8, st$M, st$alg)
  fc <- freq(a, st$M); fr <- freq(r, st$M)
  expect_identical(fc == 0, fr == 0)
  expect_lt(max(abs(fc - fr)), 0.015)
  # the rule word by word: every move of 500 C walks is in the R reference's
  # allowed set for the prefix before it (U U U, U D U' and the like never)
  ok <- vapply(2:ncol(a), function(t) {
    allowed <- rltoolsR:::cube_allowed_r(a[1:500, seq_len(t - 1L), drop = FALSE],
                                         st$M, st$alg)
    all(allowed[cbind(1:500, a[1:500, t] + 1L)])
  }, NA)
  expect_true(all(ok))
})
