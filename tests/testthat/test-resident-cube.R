# Cube walks as ag_* ops (R/resident_cube.R) against rl_cube_walks() on the
# same random numbers: the same end states and labels, walk after walk (the
# second walk starts from the solved cube again without a reset).

has_vk_cube <- function() ggml_vulkan_available() && ggml_vulkan_device_count() > 0L

cube_res_check <- function(dev, size, n = 24L, n_moves = 9L, no_cancel = FALSE) {
  env <- rl_gpu_env("cube", n = n, size = size, device = "cpu")
  ag_device(dev)
  rs <- rltoolsR:::cube_res_new(env$st, n_moves - 1L, dev)
  on.exit(rltoolsR:::cube_res_free(rs), add = TRUE)
  for (seed in 1:3) {
    set.seed(seed)
    w <- rl_cube_walks(env, n_moves = n_moves, no_cancel = no_cancel)
    set.seed(seed)
    p <- rltoolsR:::cube_walk_plan(env$st, n_moves, 2L, no_cancel)
    expect_identical(p$depth, w$depth)
    expect_identical(p$prev, w$prev)
    expect_identical(p$`next`, w$`next`)
    rltoolsR:::cube_res_walk(rs, p$acts)
    o <- rs$peek(all = rs$all, cols = rs$cols)$obs
    expect_identical(o, t(w$obs) + 0, info = paste(dev, "size", size, "seed", seed))
  }
}

# Walks drawn on the device (cube_res_sample): the moves it recorded give the
# end states and the labels it left, every move was one the host rule allows
# after the moves before it, and the device's refusals at the cut are the
# host's.
cube_host_allowed <- function(prefix, M, alg) {
  if (!length(prefix)) return(rep(TRUE, M))
  if (is.null(alg)) return(seq_len(M) - 1L != bitwXor(prefix[length(prefix)], 1L))
  cand <- seq_len(M) - 1L
  active <- rep(TRUE, M); ban <- rep(FALSE, M); cnt <- integer(M)
  for (b in rev(prefix)) {
    hit <- active & (b == bitwXor(cand, 1L))
    ban <- ban | hit
    active <- active & !hit & alg$commute[, b + 1L]
    cnt <- cnt + (active & (b == cand))
  }
  !(ban | (alg$order > 0L & 2L * (cnt + 1L) > alg$order))
}

cube_sample_check <- function(dev, size, no_cancel, n = 48L, n_moves = 10L, k_min = 2L) {
  env <- rl_gpu_env("cube", n = n, size = size, device = "cpu")
  st <- env$st
  M <- st$M
  alg <- if (no_cancel) st$alg
  ag_device(dev)
  rs <- rltoolsR:::cube_res_new(st, n_moves, dev, k_min, no_cancel)
  on.exit(rltoolsR:::cube_res_free(rs), add = TRUE)
  out <- list()
  for (rep in 1:2) {
    set.seed(20L + rep)
    rltoolsR:::cube_res_sample(rs)
    p <- rs$peek(all = rs$all, cols = rs$cols)
    acts <- matrix(as.integer(round(ag_buffer_read(rs$acts))), n, n_moves)
    cut <- as.integer(round(p$lab[1, ])); last <- as.integer(round(p$lab[2, ]))
    nxt <- as.integer(round(p$lab[3, ]))
    info <- paste(dev, "size", size, "no_cancel", no_cancel, "walk", rep)
    expect_true(all(cut >= k_min & cut <= n_moves - 1L), info = info)
    expect_gt(length(unique(cut)), 2L)
    # identity from the cut on, real moves before it
    expect_true(all(acts[col(acts) > cut] == M), info = info)
    expect_true(all(acts[col(acts) <= cut] < M), info = info)
    expect_identical(last, acts[cbind(seq_len(n), cut)], info = info)
    expect_identical(p$obs, rltoolsR:::cube_onehot(rltoolsR:::cube_host_scramble(st$perms, acts)),
                     info = info)
    ok <- TRUE; same_mask <- TRUE
    for (g in seq_len(n)) {
      for (t in seq_len(cut[g]))
        ok <- ok && cube_host_allowed(acts[g, seq_len(t - 1L)], M, alg)[acts[g, t] + 1L]
      allowed <- cube_host_allowed(acts[g, seq_len(cut[g])], M, alg)
      ok <- ok && allowed[nxt[g] + 1L]
      ord <- if (no_cancel) st$alg$order else integer(M)
      refused <- p$ban[, g] > 0.5 | (ord > 0L & 2 * (p$rep[, g] + 1) > ord)
      same_mask <- same_mask && identical(!refused, allowed)
    }
    expect_true(ok, info = info)
    expect_true(same_mask, info = info)
    out[[rep]] <- list(acts = acts, lab = p$lab)
  }
  expect_false(identical(out[[1]]$acts, out[[2]]$acts))   # a new key, new walks
  out
}

test_that("walks drawn on the device obey the rules (CPU)", {
  on.exit(ag_device("cpu"), add = TRUE)
  cube_sample_check("cpu", 2L, FALSE)
  cube_sample_check("cpu", 3L, FALSE)
  cube_sample_check("cpu", 3L, TRUE)
})

test_that("walks drawn on the device obey the rules (GPU) and equal the CPU's", {
  skip_if_not(has_vk_cube(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  g <- cube_sample_check("gpu", 3L, TRUE)
  cube_sample_check("gpu", 4L, TRUE, n = 64L, n_moves = 20L)
  expect_identical(g, cube_sample_check("cpu", 3L, TRUE))
})

test_that("resident cube walks match rl_cube_walks on the CPU", {
  on.exit(ag_device("cpu"), add = TRUE)
  cube_res_check("cpu", 2L)
  cube_res_check("cpu", 3L)
  cube_res_check("cpu", 3L, no_cancel = TRUE)
})

test_that("resident cube walks match rl_cube_walks on the GPU", {
  skip_if_not(has_vk_cube(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  cube_res_check("gpu", 3L)
  cube_res_check("gpu", 4L, n = 64L, n_moves = 20L)
})

test_that("rl_cube_q trains on resident walks", {
  # the loss of the device's own targets and mask equals the one computed in
  # R from what the walk left (same network, before the step moves it)
  check <- function(dev, no_cancel) {
    set.seed(11L)
    env <- rl_gpu_env("cube", n = 32, size = 2, device = "cpu")
    ag_device(dev)
    algo <- rl_cube_q(n_moves = 8, hidden = c(32L, 16L), blocks = 1L, walks = "resident",
                      no_cancel = no_cancel)
    state <- algo$collect(algo$init(env, dev), env)$state
    rs <- state$res
    p <- rs$peek(all = rs$all, cols = rs$cols)
    meta <- state$meta
    q <- as.matrix(state$net$forward(ag_tensor(p$obs))$q)
    n <- ncol(q); M <- nrow(q)
    cut <- p$lab[1, ]; prev <- bitwXor(as.integer(round(p$lab[2, ])), 1L)
    nxt <- as.integer(round(p$lab[3, ]))
    ip <- cbind(prev + 1L, seq_len(n)); ix <- cbind(nxt + 1L, seq_len(n))
    host <- (sum((q[ip] - (cut - 1 - meta$y_mu) / meta$y_sd)^2) +
             sum((q[ix] - (cut + 1 - meta$y_mu) / meta$y_sd)^2)) / (2 * n)
    up <- algo$update(state)
    expect_equal(up$metrics[["loss"]], host, tolerance = 1e-4,
                 info = paste(dev, "no_cancel", no_cancel))
    state <- up$state
    for (i in 1:3) state <- algo$update(algo$collect(state, env)$state)$state
    expect_identical(state$updates, 4L)
  }
  on.exit(ag_device("cpu"), add = TRUE)
  check("cpu", FALSE)
  skip_if_not(has_vk_cube(), "no Vulkan device")
  check("gpu", FALSE)
  check("gpu", TRUE)
  # "auto" on the GPU is the resident path
  set.seed(1L)
  env <- rl_gpu_env("cube", n = 16, size = 2, device = "cpu")
  ag_device("gpu")
  algo <- rl_cube_q(n_moves = 6, hidden = c(16L, 16L), blocks = 0L)
  state <- algo$collect(algo$init(env, "gpu"), env)$state
  expect_false(is.null(state$res))
  expect_identical(names(state$batch), c("all", "cols", "c_prev", "c_next"))
})
