# The Gymnasium contract on rl_gpu_env (see "Differences from Gymnasium" in
# ?rl_step): rl_reset(env, seed = s) reproduces the trajectory without
# touching the caller's RNG stream; final_obs is always there, the last
# observation of the games that ended in the step and NA for the others.

gpu_contract_envs <- list(
  cartpole = list(make = function() rl_gpu_env("cartpole", n = 4, device = "cpu"),
                  act = function(t) rep(1L, 4)),
  mountain_car = list(make = function() rl_gpu_env("mountain_car", n = 4, device = "cpu"),
                      act = function(t) rep(t %% 3L, 4)),
  pendulum_gym = list(make = function() rl_gpu_env("pendulum_gym", n = 4, device = "cpu"),
                      act = function(t) rep(0.5, 4)),
  cube = list(make = function() rl_gpu_env("cube", n = 4, size = 3, scramble = 4,
                                           max_steps = 5, device = "cpu"),
              act = function(t) rep(t %% 6L, 4)),
  tictactoe = list(make = function() rl_gpu_env("tictactoe", n = 4, opponent = "random",
                                                device = "cpu"),
                   act = function(t) rep(t %% 9L, 4))
)

run_seeded <- function(env, act, seed, steps = 20L) {
  o <- rl_reset(env, seed = seed)
  out <- lapply(seq_len(steps), function(t) {
    r <- rl_step(env, act(t))
    r[c("obs", "reward", "terminated", "truncated", "final_obs")]
  })
  list(o = o, steps = out)
}

for (nm in names(gpu_contract_envs)) {
  e <- gpu_contract_envs[[nm]]

  test_that(paste(nm, "rl_reset(seed) reproduces the trajectory, global RNG untouched"), {
    env <- e$make()
    set.seed(1)
    g0 <- .Random.seed
    a <- run_seeded(env, e$act, 5)
    expect_identical(.Random.seed, g0)
    expect_equal(run_seeded(env, e$act, 5), a)
    if (nm != "tictactoe") {               # tic-tac-toe starts from empty boards
      expect_false(isTRUE(all.equal(a$o, run_seeded(env, e$act, 6)$o)))
    }
  })

  test_that(paste(nm, "final_obs: ended games' last observation, NA for the others"), {
    env <- e$make()
    # pendulum and mountain car end only at the 200-step limit
    long <- nm %in% c("pendulum_gym", "mountain_car")
    steps <- run_seeded(env, e$act, 3, steps = if (long) 200L else 20L)$steps
    na_rows <- lapply(steps, function(r) apply(is.na(r$final_obs), 1, all))
    ended <- lapply(steps, function(r) r$terminated | r$truncated)
    expect_identical(na_rows, lapply(ended, `!`))
    any_na_in_ended <- vapply(steps, function(r) anyNA(r$final_obs[r$terminated | r$truncated, ]), NA)
    expect_false(any(any_na_in_ended))
    expect_true(any(unlist(ended)))        # games ended within the run
  })
}
