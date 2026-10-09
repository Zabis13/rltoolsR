model_logits <- function(m, obs) {
  prev <- ag_device(m$device)
  on.exit(ag_device(prev), add = TRUE)
  as.matrix(m$net$forward(ag_tensor(t(obs)))$logits)
}

# rows where the best logit beats the runner-up by a clear margin, so greedy
# actions cannot flip on float noise
clear_rows <- function(logits, margin = 1e-3) {
  apply(logits, 2, function(z) { s <- sort(z, decreasing = TRUE); s[1] - s[2] > margin })
}

train_tiny <- function(seed = 1) {
  set.seed(seed)
  env <- rl_gpu_env("tictactoe", n = 16, device = "cpu")
  rl_ppo_train(env, iterations = 1, minibatch = 144, hidden = 16,
               device = "cpu", verbose = FALSE)
}

test_that("an optimizer step updates every named parameter", {
  set.seed(1)
  prev <- ag_device("cpu")
  on.exit(ag_device(prev), add = TRUE)
  net <- ppo_net(9L, 9L, 8L)
  q <- net$parameters()
  expect_named(q, c("l1_W", "l1_b", "l2_W", "l2_b", "pi_W", "pi_b", "v_W", "v_b"))
  w0 <- lapply(q, as.matrix)
  opt <- optimizer_adam(q, lr = 1e-2)
  x <- ag_tensor(matrix(runif(9 * 4), 9, 4))
  with_grad_tape({
    out <- net$forward(x)
    loss <- ag_add(ag_sum(out$logits), ag_sum(out$value))
  })
  opt$step(backward(loss))
  w1 <- lapply(net$parameters(), as.matrix)
  for (nm in names(w0)) expect_false(identical(w1[[nm]], w0[[nm]]), info = nm)
})

test_that("save -> load on CPU keeps weights, logits and greedy actions", {
  m <- train_tiny()
  f <- tempfile(fileext = ".rds")
  on.exit(unlink(f), add = TRUE)
  expect_identical(rl_save_model(m, f), f)

  m2 <- rl_load_model(f, device = "cpu")
  expect_s3_class(m2, "rl_ppo")
  expect_identical(m2$device, "cpu")
  expect_identical(m2$meta$algo, "ppo")
  expect_identical(lapply(m2$net$parameters(), as.matrix),
                   lapply(m$net$parameters(), as.matrix))

  set.seed(3)
  obs <- matrix(sample(c(-1, 0, 1), 64 * 9, TRUE), 64, 9)
  l1 <- model_logits(m, obs)
  expect_equal(model_logits(m2, obs), l1, tolerance = 1e-6)
  ok <- clear_rows(l1)
  expect_gt(sum(ok), 0)
  expect_identical(rl_act(m2, obs[ok, , drop = FALSE]),
                   rl_act(m, obs[ok, , drop = FALSE]))
})

test_that("loading does not touch the caller's RNG", {
  m <- train_tiny()
  f <- tempfile(fileext = ".rds")
  on.exit(unlink(f), add = TRUE)
  rl_save_model(m, f)
  set.seed(42)
  before <- .Random.seed
  rl_load_model(f, device = "cpu")
  expect_identical(.Random.seed, before)
})

test_that("bad files fail with clear errors", {
  m <- train_tiny()
  f <- tempfile(fileext = ".rds")
  on.exit(unlink(f), add = TRUE)
  rl_save_model(m, f)
  obj <- readRDS(f)
  expect_null(obj$train_state)
  expect_true("train_state" %in% names(obj))

  bad <- obj; bad$format <- "something"
  saveRDS(bad, f)
  expect_error(rl_load_model(f, device = "cpu"), "not an rltoolsR model file")

  bad <- obj; bad$format_version <- 99L
  saveRDS(bad, f)
  expect_error(rl_load_model(f, device = "cpu"), "format version 99")

  bad <- obj; bad$meta$algo <- "nope"
  saveRDS(bad, f)
  expect_error(rl_load_model(f, device = "cpu"), "unknown algorithm")

  bad <- obj; bad$meta$hidden <- bad$meta$hidden + 1L
  saveRDS(bad, f)
  expect_error(rl_load_model(f, device = "cpu"), "has shape")

  expect_error(rl_save_model(list(net = m$net), f), "algorithm metadata")
})
