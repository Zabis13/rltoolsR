devices <- "cpu"
if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0L) devices <- c(devices, "gpu")

for (dev in devices) for (nm in c("pendulum_gym", "pendulum_gym_raw")) {
  test_that(paste(nm, "graph matches rl_env on", dev), {
    set.seed(1)
    n <- 8L
    genv <- rl_gpu_env(nm, n = n, device = dev)
    expect_equal(genv$max_steps, 200L)
    rl_reset(genv)
    s0 <- genv$st$host
    cenv <- lapply(seq_len(n), function(i) {
      e <- rl_env(nm, seed = i)
      rl_reset(e)
      # snapshot = step counter (int) + state (theta, theta_dot as doubles)
      rl_load_state(e, c(writeBin(0L, raw()), writeBin(as.double(s0[i, ]), raw())))
      e
    })
    # results of all steps, compared once at the end
    obs_g <- obs_c <- array(0, c(30, n, 3))
    rew_g <- rew_c <- matrix(0, 30, n)
    end_g <- end_c <- matrix(FALSE, 30, 2 * n)
    for (t in 1:30) {
      a <- stats::runif(n, -3, 3)
      rg <- rl_step(genv, a)
      rc <- lapply(seq_len(n), function(i) rl_step(cenv[[i]], a[i]))
      obs_g[t, , ] <- rg$obs
      obs_c[t, , ] <- do.call(rbind, lapply(rc, `[[`, "obs"))
      rew_g[t, ] <- rg$reward
      rew_c[t, ] <- vapply(rc, `[[`, 0, "reward")
      end_g[t, ] <- c(rg$terminated, rg$truncated)
      end_c[t, ] <- c(vapply(rc, `[[`, NA, "terminated"), vapply(rc, `[[`, NA, "truncated"))
    }
    # f32 graph against float64 C++: errors accumulate over 30 steps
    expect_lt(max(abs(obs_g - obs_c)), 1e-3)
    expect_lt(max(abs(rew_g - rew_c)), 1e-3)
    expect_identical(end_g, end_c)
  })

  test_that(paste(nm, "truncates at 200 steps and auto-resets on", dev), {
    set.seed(2)
    env <- rl_gpu_env(nm, n = 2, device = dev)
    rl_reset(env)
    for (t in 1:200) r <- rl_step(env, c(0, 1))
    expect_identical(r$truncated, c(TRUE, TRUE))
    expect_identical(r$terminated, c(FALSE, FALSE))
    expect_equal(dim(r$final_obs), c(2L, 3L))
    expect_false(isTRUE(all.equal(r$final_obs, r$obs)))
    expect_equal(env$st$steps_done, c(0L, 0L))
  })
}

test_that("pendulum GPU env rejects bad actions", {
  env <- rl_gpu_env("pendulum_gym", n = 2, device = "cpu")
  expect_error(rl_step(env, 1), "length")
  expect_error(rl_step(env, c(0, NA)), "NA")
})
