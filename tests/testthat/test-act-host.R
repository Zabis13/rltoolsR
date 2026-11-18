# rl_act() on the CPU takes a host path (policy_act_host(), plain R matrices)
# instead of the ag_* forward. Checked against the ag_* path
# (options(rltoolsR.host_act = FALSE)) on random weights: the network outputs
# themselves (logits, mu, value), greedy and sampled actions with the same
# seed, with and without a legal mask and observation normalization, for all
# three nets (separate, shared, box).

ag_path <- function(expr) {
  old <- options(rltoolsR.host_act = FALSE)
  on.exit(options(old))
  expr
}

# a CPU model; norm = TRUE gives it non-trivial observation statistics
cpu_model <- function(env, norm = FALSE, arch = NULL, ...) {
  ag_device("cpu")
  algo <- rl_ppo(normalize_obs = norm, ...)
  state <- algo$init(env, "cpu")
  if (norm) {
    d <- state$meta$obs_dim
    state$obs_rms <- list(mean = rnorm(d), var = runif(d, 0.3, 3), count = 100)
  }
  m <- algo$model(state)
  if (identical(arch, "shared")) {
    m$meta$net_arch <- "shared"
    m$net <- rltoolsR:::ppo_build(m$meta)
  }
  m
}

outputs_ag <- function(m, obs) {
  rltoolsR:::with_policy_modes(m, {
    norm <- if (isTRUE(m$meta$normalize_obs)) m$meta$obs_norm
    rltoolsR:::policy_forward(m$net, t(obs), norm)
  })
}

check_discrete <- function(m, obs, legal) {
  h <- rltoolsR:::policy_forward_host(m, t(obs))
  g <- outputs_ag(m, obs)
  expect_equal(h$logits, g$logits, tolerance = 1e-12)
  expect_equal(h$value, g$value, tolerance = 1e-12)
  for (lg in list(NULL, legal)) {
    expect_identical(rl_act(m, obs, lg), ag_path(rl_act(m, obs, lg)))
    set.seed(11); a_h <- rl_act(m, obs, lg, greedy = FALSE)
    set.seed(11); a_g <- ag_path(rl_act(m, obs, lg, greedy = FALSE))
    expect_identical(a_h, a_g)
    if (!is.null(lg)) expect_true(all(lg[cbind(seq_len(nrow(obs)), a_h + 1L)]))
  }
}

test_that("host rl_act equals the ag_* path: separate net (tictactoe)", {
  set.seed(1)
  env <- rl_gpu_env("tictactoe", n = 1, opponent = "self", device = "cpu")
  obs <- matrix(sample(c(-1, 0, 1), 9 * 40, TRUE), 40, 9)
  legal <- obs == 0
  legal[rowSums(legal) == 0, 1] <- TRUE
  for (norm in c(FALSE, TRUE)) {
    m <- cpu_model(env, norm = norm, hidden = 32)
    expect_identical(m$meta$net_arch, "separate")
    check_discrete(m, obs, legal)
    check_discrete(m, obs[1, , drop = FALSE], legal[1, , drop = FALSE])    # batch 1
  }
})

test_that("host rl_act equals the ag_* path: shared net (models from rltoolsR <= 0.1.1)", {
  set.seed(2)
  env <- rl_gpu_env("tictactoe", n = 1, opponent = "self", device = "cpu")
  obs <- matrix(sample(c(-1, 0, 1), 9 * 40, TRUE), 40, 9)
  legal <- obs == 0
  legal[rowSums(legal) == 0, 1] <- TRUE
  m <- cpu_model(env, arch = "shared", hidden = 32)
  check_discrete(m, obs, legal)
})

test_that("host rl_act equals the ag_* path: greedy box actions (pendulum)", {
  set.seed(3)
  env <- rl_vec_env("pendulum", n = 2, seed = 1)
  obs <- matrix(rnorm(3 * 40), 40, 3)
  for (norm in c(FALSE, TRUE)) {
    m <- cpu_model(env, norm = norm, hidden = 32)
    h <- rltoolsR:::policy_forward_host(m, t(obs))
    g <- outputs_ag(m, obs)
    expect_equal(h$mu, g$mu, tolerance = 1e-12)
    expect_equal(h$value, g$value, tolerance = 1e-12)
    expect_identical(rl_act(m, obs), ag_path(rl_act(m, obs)))
    # box sampling stays on the ag_* path: same draws either way
    set.seed(5); a1 <- rl_act(m, obs, greedy = FALSE)
    set.seed(5); a2 <- ag_path(rl_act(m, obs, greedy = FALSE))
    expect_identical(a1, a2)
  }
})

test_that("the host path follows the weights as they change", {
  set.seed(4)
  env <- rl_gpu_env("tictactoe", n = 1, opponent = "self", device = "cpu")
  m <- cpu_model(env, hidden = 16)
  obs <- matrix(sample(c(-1, 0, 1), 9 * 20, TRUE), 20, 9)
  p <- m$net$parameters()$pi_b
  p$data <- p$data + rnorm(length(p$data), sd = 5)        # as an optimizer step does
  expect_identical(rl_act(m, obs), ag_path(rl_act(m, obs)))
})
