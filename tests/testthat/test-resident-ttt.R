# Tic-tac-toe self-play as ag_* ops (R/resident_ttt.R), recorded with
# ag_capture(), against rl_gpu_env("tictactoe") on the same actions: legal
# moves mostly, illegal ones too, games ending and resetting.

has_vk_res <- function() ggml_vulkan_available() && ggml_vulkan_device_count() > 0L

ttt_res_check <- function(dev, n = 32L, steps = 60L) {
  set.seed(5L)
  ref <- rl_gpu_env("tictactoe", n = n, opponent = "self", device = "cpu")
  ref_obs <- rl_reset(ref)
  legal <- rl_legal_actions(ref)
  ag_device(dev)
  rs <- rltoolsR:::ttt_res_new(n)
  on.exit(rltoolsR:::ttt_res_free(rs), add = TRUE)
  f <- ag_capture(function(a, cols) rltoolsR:::ttt_res_step(rs, a, cols),
                  params = rltoolsR:::ttt_res_params(rs))
  on.exit(ag_capture_free(f), add = TRUE)
  cols <- matrix(seq_len(n) - 1, n, 1)
  n_done <- 0
  for (s in seq_len(steps)) {
    a <- vapply(seq_len(n), function(i) {
      if (stats::runif(1) < 0.1) sample.int(9, 1) - 1L else sample(which(legal[i, ]), 1) - 1L
    }, 0L)
    r <- rl_step(ref, a)
    o <- f(a = matrix(a, 1, n), cols = cols)
    expect_identical(o$obs, t(r$obs) + 0, info = paste(dev, "step", s))
    expect_identical(as.vector(o$reward), r$reward + 0, info = paste(dev, "step", s))
    expect_identical(as.vector(o$done) > 0.5, r$terminated, info = paste(dev, "step", s))
    expect_identical(o$legal > 0.5, t(r$legal), info = paste(dev, "step", s))
    legal <- r$legal
    n_done <- n_done + sum(r$terminated)
  }
  expect_gt(n_done, n)                      # games ended and restarted
}

test_that("resident tic-tac-toe matches rl_gpu_env on the CPU", {
  on.exit(ag_device("cpu"), add = TRUE)
  ttt_res_check("cpu")
})

test_that("resident tic-tac-toe matches rl_gpu_env on the GPU", {
  skip_if_not(has_vk_res(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  ttt_res_check("gpu")
})
