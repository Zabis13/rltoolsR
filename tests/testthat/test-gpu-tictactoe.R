devices <- "cpu"
if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0L) devices <- c(devices, "gpu")

# Play a sequence of moves in a single game; returns the list of step results.
play <- function(env, moves) lapply(moves, function(m) rl_step(env, m))

for (dev in devices) {
  test_that(paste("tictactoe win, reset and canonical obs on", dev), {
    env <- rl_gpu_env("tictactoe", n = 1, device = dev)
    r <- play(env, c(0, 3, 1, 4))
    # after X plays 0, O to move sees X as -1
    expect_equal(r[[1]]$obs[1, 1], -1)
    expect_equal(r[[1]]$player, -1L)
    expect_equal(r[[1]]$legal[1, ], c(FALSE, rep(TRUE, 8)))
    expect_false(any(sapply(r, `[[`, "terminated")))
    w <- rl_step(env, 2)                      # X completes the top row
    expect_equal(w$reward, 1)
    expect_true(w$terminated)
    expect_equal(as.numeric(w$obs), rep(0, 9))  # same-step reset
    expect_equal(w$player, 1L)
    expect_true(all(rl_legal_actions(env)))
  })

  test_that(paste("tictactoe illegal move and draw on", dev), {
    env <- rl_gpu_env("tictactoe", n = 1, device = dev)
    r <- play(env, c(0, 0))
    expect_equal(r[[2]]$reward, -1)
    expect_true(r[[2]]$terminated)

    r <- play(env, c(0, 1, 2, 4, 3, 5, 7, 6, 8))
    expect_false(any(sapply(r[1:8], `[[`, "terminated")))
    expect_true(r[[9]]$terminated)
    expect_equal(r[[9]]$reward, 0)
  })

  test_that(paste("tictactoe batch games are independent on", dev), {
    env <- rl_gpu_env("tictactoe", n = 3, device = dev)
    moves <- rbind(c(0, 3, 1, 4, 2),   # X wins
                   c(0, 0, 4, 4, 4),   # illegal at step 2, then new games
                   c(4, 0, 8, 2, 1))   # O to move after 5 plies, no result
    res <- lapply(1:5, function(i) rl_step(env, moves[, i]))
    term <- sapply(res, `[[`, "terminated")
    rew <- sapply(res, `[[`, "reward")
    expect_equal(term[1, ], c(FALSE, FALSE, FALSE, FALSE, TRUE))
    expect_equal(rew[1, 5], 1)
    expect_equal(term[2, 2], TRUE)
    expect_equal(rew[2, 2], -1)
    expect_false(any(term[3, ]))
    expect_equal(res[[5]]$player, c(1L, -1L, -1L))
  })

  test_that(paste("tictactoe random opponent on", dev), {
    set.seed(1)
    n <- 256
    env <- rl_gpu_env("tictactoe", n = n, opponent = "random", device = dev)
    legal <- rl_legal_actions(env)
    wins <- losses <- draws <- 0
    for (i in 1:50) {
      # X plays a random legal move
      a <- apply(legal, 1, function(l) { w <- which(l); w[sample.int(length(w), 1)] - 1L })
      r <- rl_step(env, a)
      expect_true(all(r$reward %in% c(-1, 0, 1)))
      expect_true(all(r$player == 1L))
      wins <- wins + sum(r$terminated & r$reward == 1)
      losses <- losses + sum(r$terminated & r$reward == -1)
      draws <- draws + sum(r$terminated & r$reward == 0)
      legal <- r$legal
    }
    # random vs random: X (first mover) wins ~58%, loses ~29%, draws ~13%
    total <- wins + losses + draws
    expect_gt(total, 1000)
    expect_gt(wins / total, 0.5)
    expect_lt(wins / total, 0.67)
    expect_gt(losses / total, 0.22)
    expect_lt(losses / total, 0.36)
  })
}

test_that("tictactoe gpu matches cpu", {
  skip_if_not(length(devices) == 2, "no Vulkan device")
  set.seed(3)
  moves <- matrix(sample(0:8, 64 * 12, replace = TRUE), 64)
  run <- function(dev) {
    env <- rl_gpu_env("tictactoe", n = 64, device = dev)
    lapply(1:12, function(i) rl_step(env, moves[, i]))
  }
  expect_equal(run("cpu"), run("gpu"))
})
