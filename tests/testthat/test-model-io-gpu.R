test_that("trained on GPU, loaded on CPU (and back) gives the same policy", {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() > 0L, "no Vulkan device")
  logits <- function(m, obs) {
    prev <- ag_device(m$device)
    on.exit(ag_device(prev), add = TRUE)
    prev_prec <- ag_matmul_precision("f32")         # as rl_act() does
    on.exit(ag_matmul_precision(prev_prec), add = TRUE)
    as.matrix(m$net$forward(ag_tensor(t(obs)))$logits)
  }
  weights <- function(m) {
    prev <- ag_device(m$device)
    on.exit(ag_device(prev), add = TRUE)
    lapply(m$net$parameters(), as.matrix)
  }

  set.seed(1)
  env <- rl_gpu_env("tictactoe", n = 64, device = "gpu")
  m <- rl_ppo_train(env, iterations = 2, minibatch = 576, hidden = 32,
                    device = "gpu", verbose = FALSE)
  f <- tempfile(fileext = ".rds")
  on.exit(unlink(f), add = TRUE)
  rl_save_model(m, f)

  # weights round-trip exactly, whatever the device
  w <- weights(m)
  expect_identical(readRDS(f)$net$parameters, w)
  m_cpu <- rl_load_model(f, device = "cpu")
  m_gpu <- rl_load_model(f, device = "gpu")
  expect_identical(m_cpu$device, "cpu")
  expect_identical(m_gpu$device, "gpu")
  expect_identical(weights(m_cpu), w)
  expect_identical(weights(m_gpu), w)

  set.seed(3)
  obs <- matrix(sample(c(-1, 0, 1), 128 * 9, TRUE), 128, 9)
  l_gpu <- logits(m, obs)

  # same device, same kernels: bit-identical
  expect_identical(logits(m_gpu, obs), l_gpu)

  # with ag_matmul_precision("f32") the GPU matmul keeps f32 tiles, so CPU and
  # GPU agree to float rounding (without it a batch > 8 ran with f16 tiles,
  # ~1e-3); actions are compared wherever the best logit is not a float-level tie
  expect_equal(logits(m_cpu, obs), l_gpu, tolerance = 1e-5)
  ok <- apply(l_gpu, 2, function(z) { s <- sort(z, decreasing = TRUE); s[1] - s[2] > 1e-5 })
  expect_gt(sum(ok), 0)
  expect_identical(rl_act(m_cpu, obs[ok, , drop = FALSE]),
                   rl_act(m, obs[ok, , drop = FALSE]))
})
