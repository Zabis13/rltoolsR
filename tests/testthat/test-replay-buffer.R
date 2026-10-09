# Replay buffer (R/replay_buffer.R): packed layout, ring writes across the end
# with a number of environments that does not divide the capacity, sampling
# only from the filled part, fields of a gathered batch, episode bookkeeping
# and the save/load round trip. On the CPU and, when present, the GPU.

rb_devices <- function() c("cpu", if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0L) "gpu")

# Step `t` of 3 environments with obs_dim 2, act_dim 1; values encode (t, env).
rb_step <- function(t, n = 3L) {
  e <- seq_len(n)
  list(obs = cbind(t * 10 + e, -(t * 10 + e)), act = matrix(t + e / 10, n),
       reward = t * 100 + e, terminated = e == 1L & t %% 2L == 0L,
       truncated = logical(n), next_obs = cbind(t * 10 + e + 0.5, 0))
}
rb_column <- function(s, j) c(s$obs[j, ], s$act[j, ], s$reward[j], s$terminated[j],
                              s$truncated[j], s$next_obs[j, ])
add_step <- function(rb, s) rltoolsR:::rb_add(rb, s$obs, s$act, s$reward, s$terminated,
                                              s$truncated, s$next_obs)

test_that("the layout is fixed in one place, obs and act adjacent", {
  l <- rltoolsR:::rb_layout(3L, 2L)
  expect_equal(l$rows, 3 + 2 + 3 + 3)
  expect_equal(unname(l$from), c(0, 3, 5, 6, 7, 8))
  expect_equal(l$from[["act"]], l$from[["obs"]] + l$n[["obs"]])
})

test_that("a ring with 3 environments and capacity 7 wraps across the end", {
  on.exit(ag_device("cpu"), add = TRUE)
  for (dev in rb_devices()) {
    ag_device(dev)
    rb <- rltoolsR:::rb_new(2L, 1L, capacity = 7L, n_envs = 3L)
    steps <- lapply(1:3, rb_step)
    for (s in steps) add_step(rb, s)
    # 9 transitions into 7 slots: step 3 wrote slots 6, 0, 1
    expect_equal(rb$size, 7L, info = dev)
    expect_equal(rb$pos, 2L, info = dev)
    want <- cbind(rb_column(steps[[3]], 2), rb_column(steps[[3]], 3),
                  rb_column(steps[[1]], 3), rb_column(steps[[2]], 1),
                  rb_column(steps[[2]], 2), rb_column(steps[[2]], 3),
                  rb_column(steps[[3]], 1))
    expect_equal(ag_buffer_read(rb$buf), want, tolerance = 1e-6, ignore_attr = TRUE,
                 info = dev)
    rltoolsR:::rb_free(rb)
  }
})

test_that("indices come from the filled part only", {
  ag_device("cpu")
  rb <- rltoolsR:::rb_new(2L, 1L, capacity = 100L, n_envs = 3L)
  expect_error(rltoolsR:::rb_sample_idx(rb, 4L), "empty")
  add_step(rb, rb_step(1)); add_step(rb, rb_step(2))
  set.seed(1)
  idx <- rltoolsR:::rb_sample_idx(rb, 1000L)
  expect_true(all(idx >= 0L & idx < 6L))
  expect_setequal(unique(idx), 0:5)
  # reproducible from R's RNG
  set.seed(1)
  expect_identical(rltoolsR:::rb_sample_idx(rb, 1000L), idx)
  rltoolsR:::rb_free(rb)
})

test_that("a gathered batch splits into its fields", {
  on.exit(ag_device("cpu"), add = TRUE)
  for (dev in rb_devices()) {
    ag_device(dev)
    rb <- rltoolsR:::rb_new(2L, 1L, capacity = 10L, n_envs = 3L)
    s1 <- rb_step(1); s2 <- rb_step(2)
    add_step(rb, s1); add_step(rb, s2)
    idx <- c(4L, 0L, 4L)                   # step 2 env 2, step 1 env 1, again
    x <- rltoolsR:::rb_gather(rb, idx)
    f <- function(field) as.matrix(rltoolsR:::rb_field(rb, x, field))
    expect_equal(f("obs"), cbind(s2$obs[2, ], s1$obs[1, ], s2$obs[2, ]), info = dev)
    expect_equal(f("act"), matrix(c(s2$act[2], s1$act[1], s2$act[2]), 1), info = dev,
                 tolerance = 1e-6)
    expect_equal(f("reward"), matrix(c(s2$reward[2], s1$reward[1], s2$reward[2]), 1), info = dev)
    expect_equal(f("terminated"), matrix(c(0, 0, 0), 1), info = dev)
    expect_equal(f("next_obs"), cbind(s2$next_obs[2, ], s1$next_obs[1, ], s2$next_obs[2, ]),
                 info = dev)
    oa <- as.matrix(rltoolsR:::rb_slice(rb, x, c("obs", "act")))
    expect_equal(oa, rbind(f("obs"), f("act")), info = dev)
    expect_error(rltoolsR:::rb_slice(rb, x, c("obs", "reward")), "not adjacent")
    rltoolsR:::rb_free(rb)
  }
})

test_that("episode_start records where each copy's episode began", {
  ag_device("cpu")
  rb <- rltoolsR:::rb_new(2L, 1L, capacity = 20L, n_envs = 3L)
  for (t in 1:3) add_step(rb, rb_step(t))
  # env 1 terminates at t = 2 (slot 3); its next episode starts at slot 6
  expect_equal(rb$episode_start[1:9], c(0L, 1L, 2L, 0L, 1L, 2L, 6L, 1L, 2L))
  rltoolsR:::rb_free(rb)
})

test_that("save and load restore the buffer in parts", {
  on.exit(ag_device("cpu"), add = TRUE)
  f <- tempfile(fileext = ".rb")
  on.exit(unlink(f), add = TRUE)
  for (dev in rb_devices()) {
    ag_device(dev)
    rb <- rltoolsR:::rb_new(2L, 1L, capacity = 7L, n_envs = 3L)
    for (t in 1:3) add_step(rb, rb_step(t))
    rltoolsR:::rb_save(rb, f, chunk = 3L)  # parts of 3, 3 and 1 columns
    rb2 <- rltoolsR:::rb_load(f, chunk = 2L)
    expect_equal(ag_buffer_read(rb2$buf), ag_buffer_read(rb$buf), tolerance = 1e-6,
                 info = dev)
    for (nm in c("pos", "size", "episode_start", "cur_start", "capacity", "n_envs"))
      expect_identical(rb2[[nm]], rb[[nm]], info = paste(dev, nm))
    rltoolsR:::rb_free(rb); rltoolsR:::rb_free(rb2)
  }
  # a partly filled buffer saves only its filled slots
  ag_device("cpu")
  rb <- rltoolsR:::rb_new(2L, 1L, capacity = 1000L, n_envs = 3L)
  add_step(rb, rb_step(1))
  rltoolsR:::rb_save(rb, f)
  rb2 <- rltoolsR:::rb_load(f)
  expect_equal(ag_buffer_read(rb2$buf, 0, 3), ag_buffer_read(rb$buf, 0, 3), tolerance = 1e-6)
  expect_lt(file.size(f), 1000 * 9 * 4)
  writeLines("x", f)
  expect_error(rltoolsR:::rb_load(f))
})

test_that("rb_add checks the shapes and the capacity", {
  ag_device("cpu")
  expect_error(rltoolsR:::rb_new(2L, 1L, capacity = 2L, n_envs = 3L), "capacity")
  rb <- rltoolsR:::rb_new(2L, 1L, capacity = 10L, n_envs = 3L)
  s <- rb_step(1)
  expect_error(rltoolsR:::rb_add(rb, s$obs, s$act, s$reward[1:2], s$terminated,
                                 s$truncated, s$next_obs), "shapes")
  rltoolsR:::rb_free(rb)
})
