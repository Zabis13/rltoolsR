devices <- "cpu"
if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0L) devices <- c(devices, "gpu")

mc_names <- c("mountain_car", "mountain_car_continuous")

for (dev in devices) for (nm in mc_names) {
  continuous <- nm == "mountain_car_continuous"

  test_that(paste(nm, "graph matches rl_env on", dev), {
    set.seed(1)
    n <- 8L
    genv <- rl_gpu_env(nm, n = n, device = dev)
    expect_equal(genv$max_steps, if (continuous) 999L else 200L)
    s0 <- rl_reset(genv)
    expect_equal(dim(s0), c(n, 2L))
    cenv <- lapply(seq_len(n), function(i) {
      e <- rl_env(nm, seed = i)
      rl_reset(e)
      # snapshot = step counter (int) + state (x, velocity as doubles)
      rl_load_state(e, c(writeBin(0L, raw()), writeBin(as.double(s0[i, ]), raw())))
      e
    })
    # 30 random steps stay far from the goal; game 1 starts at the left wall
    s0[1, ] <- c(-1.19, -0.05)
    mountain_car_set_state(genv, s0)
    rl_load_state(cenv[[1]], c(writeBin(0L, raw()), writeBin(s0[1, ], raw())))
    # results of all steps, compared once at the end
    obs_g <- obs_c <- array(0, c(30, n, 2))
    rew_g <- rew_c <- matrix(0, 30, n)
    end_g <- end_c <- matrix(FALSE, 30, 2 * n)
    for (t in 1:30) {
      a <- if (continuous) stats::runif(n, -1.5, 1.5) else sample(0:2, n, replace = TRUE)
      if (t == 1L) a[1] <- if (continuous) -1 else 0
      rg <- rl_step(genv, a)
      rc <- lapply(seq_len(n), function(i) rl_step(cenv[[i]], a[i]))
      obs_g[t, , ] <- rg$obs
      obs_c[t, , ] <- do.call(rbind, lapply(rc, `[[`, "obs"))
      rew_g[t, ] <- rg$reward
      rew_c[t, ] <- vapply(rc, `[[`, 0, "reward")
      end_g[t, ] <- c(rg$terminated, rg$truncated)
      end_c[t, ] <- c(vapply(rc, `[[`, NA, "terminated"), vapply(rc, `[[`, NA, "truncated"))
    }
    expect_lt(max(abs(obs_g - obs_c)), 1e-4)
    expect_lt(max(abs(rew_g - rew_c)), 1e-5)
    expect_identical(end_g, end_c)
  })

  test_that(paste(nm, "reaches the goal and auto-resets on", dev), {
    set.seed(2)
    env <- rl_gpu_env(nm, n = 2, device = dev)
    o <- rl_reset(env)
    lo <- if (continuous) -1 else 0
    hi <- if (continuous) 1 else 2
    for (t in seq_len(env$max_steps)) {
      r <- rl_step(env, ifelse(o[, 2] >= 0, hi, lo))
      o <- r$obs
      if (any(r$terminated)) break
    }
    i <- which(r$terminated)[1]
    expect_false(is.na(i))
    expect_gte(r$final_obs[i, 1], if (continuous) 0.45 else 0.5)
    expect_equal(r$reward[i], if (continuous) 100 - 0.1 else -1, tolerance = 1e-6)
    # the finished game restarts in [-0.6, -0.4] at rest
    expect_true(r$obs[i, 1] >= -0.6 && r$obs[i, 1] <= -0.4)
    expect_equal(r$obs[i, 2], 0)
  })
}

test_that("mountain_car GPU env rejects bad actions", {
  env <- rl_gpu_env("mountain_car", n = 2, device = "cpu")
  expect_error(rl_step(env, c(0, 3)), "0, 1 or 2")
  expect_error(rl_step(env, 1), "length")
  envc <- rl_gpu_env("mountain_car_continuous", n = 2, device = "cpu")
  expect_error(rl_step(envc, c(0, NA)), "NA")
})
