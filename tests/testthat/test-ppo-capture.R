# rl_ppo(capture = TRUE): every update step as a ggmlR captured step. The
# reference is the ordinary path (capture = FALSE) on the same device and seed:
# the capture uses no random numbers, so both see the same rollouts and
# minibatch permutations. Minibatch sizes are chosen so the rollout does not
# divide evenly -- two recordings (full and short minibatch) must update the
# SAME optimizer and parameters.
#
# Run lengths are the minimum that covers this: one tic-tac-toe iteration is
# already 4 epochs x 3 minibatches, so both recordings are reused within it;
# Pendulum needs a second iteration, the first one whose normalization
# statistics are not the identity (they enter the captured graph as inputs).

has_vk_cap <- function() ggml_vulkan_available() && ggml_vulkan_device_count() > 0L

metric_cols <- function(h) intersect(c("pg", "vf", "ent", "grad_norm"), names(h))

# Runs are cached by their arguments: several tests compare the same runs.
run_cache <- new.env()
cached_run <- function(key, f) {
  if (is.null(run_cache[[key]])) run_cache[[key]] <- f()
  run_cache[[key]]
}
run_key <- function(...) paste(vapply(list(...), function(x) if (is.null(x)) "NULL" else format(x),
                                      ""), collapse = "/")

run_ttt <- function(dev, capture, mgn = NULL, seed = 11L) {
  cached_run(run_key("ttt", dev, capture, mgn, seed), function() {
    set.seed(seed)
    env <- rl_gpu_env("tictactoe", n = 64, device = dev)     # N = 9 * 64 = 576
    m <- rl_ppo_train(env, iterations = 1, minibatch = 200, device = dev,
                      verbose = FALSE, max_grad_norm = mgn, capture = capture)
    list(h = m$history, w = lapply(m$net$parameters(), as.matrix))
  })
}

run_pendulum <- function(dev, capture, mgn = NULL, seed = 12L) {
  cached_run(run_key("pendulum", dev, capture, mgn, seed), function() {
    set.seed(seed)
    env <- rl_vec_env("pendulum", n = 4, seed = seed)         # N = 16 * 4 = 64
    m <- rl_ppo_train(env, iterations = 2, n_steps = 16, minibatch = 24, device = dev,
                      verbose = FALSE, max_grad_norm = mgn, capture = capture)
    list(h = m$history, w = lapply(m$net$parameters(), as.matrix))
  })
}

expect_same_run <- function(got, ref, tol, info) {
  cols <- metric_cols(ref$h)
  expect_identical(metric_cols(got$h), cols, info = info)
  for (cl in cols)
    expect_equal(got$h[[cl]], ref$h[[cl]], tolerance = tol, info = paste(info, cl))
  for (nm in names(ref$w))
    expect_equal(got$w[[nm]], ref$w[[nm]], tolerance = tol, info = paste(info, nm))
}

test_that("max_grad_norm is validated", {
  for (bad in list(0, -1, Inf, NA_real_, c(1, 2), "1"))
    expect_error(rl_ppo(max_grad_norm = bad), "max_grad_norm")
})

test_that("capture on the CPU equals the ordinary path (ttt, Pendulum, clipping)", {
  for (mgn in list(NULL, 0.05)) {
    info <- paste("max_grad_norm", if (is.null(mgn)) "NULL" else format(mgn))
    expect_same_run(run_ttt("cpu", TRUE, mgn), run_ttt("cpu", FALSE, mgn), 1e-8,
                    paste("ttt", info))
    expect_same_run(run_pendulum("cpu", TRUE, mgn), run_pendulum("cpu", FALSE, mgn), 1e-8,
                    paste("pendulum", info))
  }
})

test_that("clipping reports grad_norm, and a never-active limit changes nothing", {
  r_none <- run_ttt("cpu", FALSE)
  r_big  <- run_ttt("cpu", FALSE, 1e3)
  r_tiny <- run_ttt("cpu", FALSE, 0.05)
  expect_false("grad_norm" %in% names(r_none$h))
  expect_true(all(r_big$h$grad_norm > 0.05))       # so 0.05 clips every step
  for (cl in c("pg", "vf", "ent")) expect_equal(r_big$h[[cl]], r_none$h[[cl]])
  expect_false(isTRUE(all.equal(r_tiny$w, r_none$w)))
})

test_that("capture on the GPU matches the ordinary GPU path", {
  skip_if_not(has_vk_cap(), "no Vulkan device")
  for (mgn in list(NULL, 0.05)) {
    info <- paste("max_grad_norm", if (is.null(mgn)) "NULL" else format(mgn))
    expect_same_run(run_ttt("gpu", TRUE, mgn), run_ttt("gpu", FALSE, mgn),
                    1e-4, paste("ttt", info))
    expect_same_run(run_pendulum("gpu", TRUE, mgn), run_pendulum("gpu", FALSE, mgn),
                    1e-4, paste("pendulum", info))
  }
})

test_that("a changed graph constant records anew instead of reusing", {
  skip_if_not(has_vk_cap(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  one_update <- function(algos) {
    set.seed(41L)
    env <- rl_gpu_env("tictactoe", n = 64, device = "gpu")
    ag_device("gpu")
    state <- algos[[1L]]$init(env, "gpu")
    for (a in algos) state <- a$update(a$collect(state, env)$state)$state
    list(w = lapply(state$net$parameters(), as.matrix), captures = state$captures)
  }
  cap <- one_update(list(rl_ppo(clip = 0.2, minibatch = 576, capture = TRUE),
                         rl_ppo(clip = 0.1, minibatch = 576, capture = TRUE)))
  ref <- one_update(list(rl_ppo(clip = 0.2, minibatch = 576, capture = FALSE),
                         rl_ppo(clip = 0.1, minibatch = 576, capture = FALSE)))
  expect_length(ls(cap$captures), 2L)                 # one recording per clip value
  for (nm in names(ref$w))
    expect_equal(cap$w[[nm]], ref$w[[nm]], tolerance = 1e-4, info = nm)
})
