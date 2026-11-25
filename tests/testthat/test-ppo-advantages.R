# ppo_advantages(): ggmlR's ag_gae() and the local fallback (ppo_gae() with the
# truncation bootstrap folded into the reward) give the same advantages.

adv_data <- function(seed = 3, n_env = 4L, n_t = 10L) {
  set.seed(seed)
  term  <- matrix(runif(n_env * n_t) < 0.15, n_env, n_t) * 1
  trunc <- (matrix(runif(n_env * n_t) < 0.15, n_env, n_t) & term == 0) * 1
  list(rew = matrix(rnorm(n_env * n_t), n_env, n_t), val = matrix(rnorm(n_env * n_t), n_env, n_t),
       term = term, trunc = trunc, nv = trunc * rnorm(n_env * n_t), last = rnorm(n_env))
}

test_that("ag_gae path equals the folded-reward fallback", {
  skip_if(length(rltoolsR:::ggmlr_missing("ag_gae")) > 0L, "ggmlR without ag_gae()")
  d <- adv_data()
  for (sg in c(1, -1)) {
    got <- rltoolsR:::ppo_advantages(d$rew, d$val, d$term, d$trunc, d$nv, d$last,
                                     0.99, 0.95, sg)
    ref <- rltoolsR:::ppo_gae(d$rew + 0.99 * sg * d$trunc * d$nv, d$val, d$term + d$trunc,
                              d$last, 0.99, 0.95, sg)
    expect_equal(got, ref, tolerance = 1e-12, info = paste("turn_sign", sg))
  }
})

test_that("a truncated step bootstraps V(final obs), a terminated one does not", {
  rew <- matrix(c(1, 1), 1, 2); val <- matrix(0, 1, 2)
  # step 1 truncated with V(final) = 10, step 2 terminated; gamma 0.5, lambda 1
  a <- rltoolsR:::ppo_advantages(rew, val, matrix(c(0, 1), 1, 2), matrix(c(1, 0), 1, 2),
                                 matrix(c(10, 0), 1, 2), 99, 0.5, 1, 1)
  expect_equal(as.numeric(a), c(1 + 0.5 * 10, 1))
})
