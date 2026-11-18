MAP_4X4 <- c("SFFF", "FHFH", "FFFH", "HFFG")
MAP_8X8 <- c("SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF",
             "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG")

# Gymnasium toy_text/frozen_lake.py transition table in R, same layout as
# rl_env_table(): S x A x K arrays, next 0-based, isd from the S cells
gym_frozen_lake <- function(desc, slippery) {
  n <- length(desc)
  m <- do.call(rbind, strsplit(desc, ""))
  S <- n * n
  K <- if (slippery) 3L else 1L
  prob <- rew <- array(0, c(S, 4, K))
  nxt <- array(0L, c(S, 4, K))
  term <- array(FALSE, c(S, 4, K))
  inc <- function(r, c, a) switch(a + 1, c(r, max(c - 1, 0)), c(min(r + 1, n - 1), c),
                                  c(r, min(c + 1, n - 1)), c(max(r - 1, 0), c))
  for (r in 0:(n - 1)) for (c in 0:(n - 1)) {
    s <- r * n + c
    l <- m[r + 1, c + 1]
    for (a in 0:3) {
      if (l %in% c("G", "H")) {
        prob[s + 1, a + 1, 1] <- 1; nxt[s + 1, a + 1, 1] <- s; term[s + 1, a + 1, 1] <- TRUE
        next
      }
      dirs <- if (slippery) c((a - 1) %% 4, a, (a + 1) %% 4) else a
      for (k in seq_along(dirs)) {
        rc <- inc(r, c, dirs[k])
        nl <- m[rc[1] + 1, rc[2] + 1]
        prob[s + 1, a + 1, k] <- if (slippery) 1 / 3 else 1
        nxt[s + 1, a + 1, k] <- as.integer(rc[1] * n + rc[2])
        rew[s + 1, a + 1, k] <- as.numeric(nl == "G")
        term[s + 1, a + 1, k] <- nl %in% c("G", "H")
      }
    }
  }
  isd <- as.numeric(as.vector(t(m)) == "S")
  list(prob = prob, `next` = nxt, reward = rew, terminated = term, isd = isd / sum(isd))
}

fl <- expand.grid(big = c(FALSE, TRUE), det = c(FALSE, TRUE), onehot = c(FALSE, TRUE))
fl$name <- paste0("frozen_lake", ifelse(fl$big, "_8x8", ""), ifelse(fl$det, "_det", ""),
                  ifelse(fl$onehot, "_onehot", ""))

test_that("all frozen_lake variants are registered", {
  expect_setequal(grep("^frozen_lake", rl_env_names(), value = TRUE), fl$name)
})

# snapshot = step counter (int) + state (int)
toy_load <- function(env, s) rl_load_state(env, c(writeBin(0L, raw()), writeBin(as.integer(s), raw())))

for (i in seq_len(nrow(fl))) {
  v <- fl[i, ]
  nm <- v$name
  S <- if (v$big) 64L else 16L

  test_that(paste(nm, "spec, table and observation"), {
    env <- rl_env(nm, seed = 1)
    expect_equal(env$max_steps, if (v$big) 200L else 100L)
    expect_equal(env$action_space$n, 4L)
    if (v$onehot) {
      expect_equal(env$observation_space$shape, S)
    } else {
      expect_equal(env$observation_space$n, S)
    }
    # the whole table, every state and action, against Gymnasium
    expect_equal(rl_env_table(env), gym_frozen_lake(if (v$big) MAP_8X8 else MAP_4X4, !v$det))

    o <- rl_reset(env, seed = 3)
    expect_equal(o, if (v$onehot) c(1, rep(0, S - 1)) else 0)
    s <- rl_save_state(env)
    r1 <- rl_step(env, 2)
    rl_load_state(env, s)
    # the snapshot holds no RNG state: slippery steps may differ after a load
    if (v$det) expect_equal(rl_step(env, 2), r1) else expect_equal(rl_save_state(env), s)
    expect_error(rl_step(env, 4), "integer in 0..3")
    expect_error(rl_step(env, 1.5), "integer in 0..3")
  })

  test_that(paste(nm, "vec env matches single envs"), {
    venv <- rl_vec_env(nm, n = 3, seed = 1)
    singles <- lapply(0:2, function(j) rl_env(nm, seed = 1 + j))
    o <- rl_reset(venv, seed = 1)
    expect_equal(o, do.call(rbind, lapply(1:3, function(j) rl_reset(singles[[j]], seed = j))))
    a <- c(1, 2, 3)
    vec <- sgl <- list()
    for (t in 1:10) {
      r <- rl_step(venv, a)
      vec[[t]] <- r[c("obs", "reward", "terminated", "truncated")]
      ri <- lapply(1:3, function(j) {
        x <- rl_step(singles[[j]], a[j])
        if (x$terminated || x$truncated) x$obs <- rl_reset(singles[[j]])
        x
      })
      sgl[[t]] <- list(obs = do.call(rbind, lapply(ri, `[[`, "obs")),
                       reward = vapply(ri, `[[`, 0, "reward"),
                       terminated = vapply(ri, `[[`, NA, "terminated"),
                       truncated = vapply(ri, `[[`, NA, "truncated"))
    }
    for (f in c("obs", "reward", "terminated", "truncated")) {
      expect_equal(lapply(vec, `[[`, f), lapply(sgl, `[[`, f), label = f)
    }
  })
}

test_that("frozen_lake_det: the shortest path reaches the goal with reward 1", {
  env <- rl_env("frozen_lake_det", seed = 1)
  rl_reset(env)
  rs <- lapply(c(1, 1, 2, 1, 2, 2), function(a) rl_step(env, a))
  expect_equal(vapply(rs, `[[`, 0, "obs"), c(4, 8, 9, 13, 14, 15))
  expect_equal(vapply(rs, `[[`, 0, "reward"), c(0, 0, 0, 0, 0, 1))
  expect_equal(vapply(rs, `[[`, NA, "terminated"), c(rep(FALSE, 5), TRUE))
})

test_that("frozen_lake_det: walking into a wall truncates at 100 steps", {
  env <- rl_env("frozen_lake_det", seed = 1)
  rl_reset(env)
  rs <- lapply(1:100, function(t) rl_step(env, 0))
  expect_false(any(vapply(rs, `[[`, NA, "terminated")))
  expect_identical(vapply(rs, `[[`, NA, "truncated"), c(rep(FALSE, 99), TRUE))
})

# Gymnasium toy_text/cliffwalking.py (is_slippery = False) transition table in R
gym_cliff_walking <- function() {
  nr <- 4; nc <- 12; S <- nr * nc
  start <- (nr - 1) * nc
  prob <- array(1, c(S, 4, 1)); rew <- array(0, c(S, 4, 1))
  nxt <- array(0L, c(S, 4, 1)); term <- array(FALSE, c(S, 4, 1))
  dr <- c(-1, 0, 1, 0); dc <- c(0, 1, 0, -1)
  for (r in 0:(nr - 1)) for (c in 0:(nc - 1)) for (a in 0:3) {
    s <- r * nc + c
    r1 <- min(max(r + dr[a + 1], 0), nr - 1)
    c1 <- min(max(c + dc[a + 1], 0), nc - 1)
    if (r1 == nr - 1 && c1 >= 1 && c1 <= nc - 2) {
      nxt[s + 1, a + 1, 1] <- start; rew[s + 1, a + 1, 1] <- -100
    } else {
      nxt[s + 1, a + 1, 1] <- as.integer(r1 * nc + c1); rew[s + 1, a + 1, 1] <- -1
      term[s + 1, a + 1, 1] <- r1 == nr - 1 && c1 == nc - 1
    }
  }
  isd <- numeric(S); isd[start + 1] <- 1
  list(prob = prob, `next` = nxt, reward = rew, terminated = term, isd = isd)
}

for (nm in c("cliff_walking", "cliff_walking_onehot")) {
  test_that(paste(nm, "spec, table, the cliff and the shortest path"), {
    env <- rl_env(nm, seed = 1)
    onehot <- grepl("onehot", nm)
    expect_equal(env$max_steps, 200L)
    expect_equal(env$action_space$n, 4L)
    if (onehot) expect_equal(env$observation_space$shape, 48L) else
      expect_equal(env$observation_space$n, 48L)
    expect_equal(rltoolsR:::rl_env_table(env), gym_cliff_walking())
    state <- function(o) if (onehot) which.max(o) - 1 else o

    expect_equal(state(rl_reset(env)), 36)
    # right from the start: the cliff, -100, back at the start, not the end
    r <- rl_step(env, 1)
    expect_equal(c(state(r$obs), r$reward), c(36, -100))
    expect_false(r$terminated || r$truncated)
    # up, 11 x right, down: 13 steps of -1 to the goal
    rs <- lapply(c(0, rep(1, 11), 2), function(a) rl_step(env, a))
    expect_equal(state(rs[[13]]$obs), 47)
    expect_equal(sum(vapply(rs, `[[`, 0, "reward")), -13)
    expect_identical(vapply(rs, `[[`, NA, "terminated"), c(rep(FALSE, 12), TRUE))
  })
}

test_that("cliff_walking truncates at 200 steps", {
  env <- rl_env("cliff_walking", seed = 1)
  rl_reset(env)
  rs <- lapply(1:200, function(t) rl_step(env, 3))       # left into the wall
  expect_false(any(vapply(rs, `[[`, NA, "terminated")))
  expect_identical(vapply(rs, `[[`, NA, "truncated"), c(rep(FALSE, 199), TRUE))
})

# Gymnasium toy_text/taxi.py (Taxi-v3, no rain, no fickle passenger) in R
TAXI_MAP <- c("+---------+", "|R: | : :G|", "| : | : : |", "| : : : : |",
              "| | : | : |", "|Y| : |B: |", "+---------+")
gym_taxi <- function() {
  desc <- do.call(rbind, strsplit(TAXI_MAP, ""))
  locs <- list(c(0, 0), c(0, 4), c(4, 0), c(4, 3))
  enc <- function(r, c, p, d) as.integer(((r * 5 + c) * 5 + p) * 4 + d)
  S <- 500L
  prob <- array(1, c(S, 6, 1)); rew <- array(0, c(S, 6, 1))
  nxt <- array(0L, c(S, 6, 1)); term <- array(FALSE, c(S, 6, 1))
  isd <- numeric(S)
  for (r in 0:4) for (c in 0:4) for (p in 0:4) for (d in 0:3) {
    s <- enc(r, c, p, d)
    if (p < 4 && p != d) isd[s + 1] <- 1
    at <- which(vapply(locs, function(l) all(l == c(r, c)), NA)) - 1
    for (a in 0:5) {
      nr <- r; nc <- c; np <- p; rw <- -1; tm <- FALSE
      if (a == 0) nr <- min(r + 1, 4)
      else if (a == 1) nr <- max(r - 1, 0)
      else if (a == 2 && desc[2 + r, 2 * c + 3] == ":") nc <- min(c + 1, 4)
      else if (a == 3 && desc[2 + r, 2 * c + 1] == ":") nc <- max(c - 1, 0)
      else if (a == 4) {
        if (p < 4 && length(at) && at == p) np <- 4 else rw <- -10
      } else if (a == 5) {
        if (length(at) && at == d && p == 4) { np <- d; tm <- TRUE; rw <- 20 }
        else if (length(at) && p == 4) np <- at
        else rw <- -10
      }
      nxt[s + 1, a + 1, 1] <- enc(nr, nc, np, d)
      rew[s + 1, a + 1, 1] <- rw
      term[s + 1, a + 1, 1] <- tm
    }
  }
  list(prob = prob, `next` = nxt, reward = rew, terminated = term, isd = isd / sum(isd))
}

taxi_state <- function(r, c, p, d) ((r * 5 + c) * 5 + p) * 4 + d

for (nm in c("taxi", "taxi_onehot")) {
  test_that(paste(nm, "spec and table"), {
    env <- rl_env(nm, seed = 1)
    expect_equal(env$max_steps, 200L)
    expect_equal(env$action_space$n, 6L)
    if (nm == "taxi_onehot") expect_equal(env$observation_space$shape, 500L) else
      expect_equal(env$observation_space$n, 500L)
    expect_equal(rltoolsR:::rl_env_table(env), gym_taxi())
  })
}

test_that("taxi pickup, dropoff and delivery", {
  env <- rl_env("taxi", seed = 1)
  rl_reset(env)
  step_from <- function(s, a) { toy_load(env, s); r <- rl_step(env, a); c(r$obs, r$reward, r$terminated) }
  # at R with the passenger at R, destination G: pickup puts them in the taxi
  expect_equal(step_from(taxi_state(0, 0, 0, 1), 4), c(taxi_state(0, 0, 4, 1), -1, 0))
  # a second pickup is illegal
  expect_equal(step_from(taxi_state(0, 0, 4, 1), 4), c(taxi_state(0, 0, 4, 1), -10, 0))
  # pickup where the passenger is not
  expect_equal(step_from(taxi_state(0, 0, 1, 2), 4), c(taxi_state(0, 0, 1, 2), -10, 0))
  # dropoff at R, not the destination: the passenger waits at R
  expect_equal(step_from(taxi_state(0, 0, 4, 1), 5), c(taxi_state(0, 0, 0, 1), -1, 0))
  # dropoff away from R, G, Y, B is illegal
  expect_equal(step_from(taxi_state(2, 2, 4, 1), 5), c(taxi_state(2, 2, 4, 1), -10, 0))
  # delivery at G ends the episode with 20
  expect_equal(step_from(taxi_state(0, 4, 4, 1), 5), c(taxi_state(0, 4, 1, 1), 20, 1))
  # the wall right of (0, 1) blocks east; (0, 2) -> (0, 3) is open
  expect_equal(step_from(taxi_state(0, 1, 0, 1), 2)[1], taxi_state(0, 1, 0, 1))
  expect_equal(step_from(taxi_state(0, 2, 0, 1), 2)[1], taxi_state(0, 3, 0, 1))
})

test_that("taxi starts uniformly over the 300 states of the initial distribution", {
  env <- rl_env("taxi", seed = 1)
  N <- 30000
  s0 <- vapply(seq_len(N), function(i) rl_reset(env), 0)
  ok <- which(gym_taxi()$isd > 0) - 1
  expect_true(all(s0 %in% ok))
  cnt <- tabulate(match(s0, ok), length(ok))
  chi2 <- sum((cnt - N / 300)^2 / (N / 300))
  expect_lt(chi2, stats::qchisq(0.999, 299))
})

# ---- blackjack: state (sum * 11 + dealer) * 2 + ace
bj_s <- function(sum, d, ace) (sum * 11 + d) * 2 + ace

# probability that hit / stick from `s` ends with `pred(next, reward, terminated)`
bj_prob <- function(tab, s, a, pred) {
  k <- seq_len(dim(tab$prob)[3])
  ok <- pred(tab$`next`[s + 1, a + 1, k], tab$reward[s + 1, a + 1, k], tab$terminated[s + 1, a + 1, k])
  sum(tab$prob[s + 1, a + 1, k][ok])
}

test_that("blackjack spec, observations and probabilities", {
  env <- rl_env("blackjack", seed = 1)
  expect_equal(env$observation_space$type, "multi_discrete")
  expect_equal(env$observation_space$nvec, c(32L, 11L, 2L))
  expect_equal(env$action_space$n, 2L)
  expect_equal(env$max_steps, 100L)
  envo <- rl_env("blackjack_onehot", seed = 1)
  expect_equal(envo$observation_space$shape, 45L)

  tab <- rltoolsR:::rl_env_table(env)
  expect_lt(max(abs(apply(tab$prob, c(1, 2), sum) - 1)), 1e-12)
  expect_equal(sum(tab$isd), 1, tolerance = 1e-12)
  expect_equal(tab$obs[bj_s(20, 5, 0) + 1, ], c(20, 5, 0))

  # the observation is (sum, dealer card, usable ace); one-hots at 0, 32, 43
  o <- rl_reset(env)
  expect_true(o[1] >= 4 && o[1] <= 21 && o[2] >= 1 && o[2] <= 10 && o[3] %in% 0:1)
  toy_load(envo, bj_s(20, 5, 0))
  r <- rl_step(envo, 0)
  expect_equal(which(r$obs > 0), c(21, 33 + 5, 44))
})

test_that("blackjack hits: busts, soft hands and no automatic stick at 21", {
  tab <- rltoolsR:::rl_env_table(rl_env("blackjack", seed = 1))
  bust <- function(n, r, t) t & r == -1
  expect_equal(bj_prob(tab, bj_s(20, 5, 0), 1, bust), 12 / 13)       # only an ace saves
  expect_equal(bj_prob(tab, bj_s(21, 5, 0), 1, bust), 1)             # hard 21 may hit
  expect_equal(bj_prob(tab, bj_s(21, 5, 1), 1, bust), 0)             # soft 21 never busts
  # soft 17 + 10 is hard 17, not a bust
  hard17 <- function(n, r, t) n == bj_s(17, 5, 0) & !t
  expect_equal(bj_prob(tab, bj_s(17, 5, 1), 1, hard17), 4 / 13)
  # a bust shows the bust sum: hard 20 + 10 -> 30
  expect_equal(bj_prob(tab, bj_s(20, 5, 0), 1, function(n, r, t) n == bj_s(30, 5, 0) & t), 4 / 13)
})

test_that("blackjack initial hands", {
  tab <- rltoolsR:::rl_env_table(rl_env("blackjack", seed = 1))
  p <- function(c) if (c == 10) 4 / 13 else 1 / 13
  expect_equal(tab$isd[bj_s(12, 5, 1) + 1], p(1)^2 * p(5))           # A, A: soft 12
  expect_equal(tab$isd[bj_s(21, 10, 1) + 1], 2 * p(1) * p(10) * p(10)) # A, 10: soft 21, no bonus
  expect_equal(tab$isd[bj_s(4, 3, 0) + 1], p(2)^2 * p(3))            # 2, 2: hard 4
  expect_true(all(tab$isd[bj_s(22:31, 1, 0) + 1] == 0))
})

# Gymnasium's dealer (blackjack.py: draw while sum_hand < 17, a usable ace as
# 11, so he stands on soft 17) played in R, against the dealer distribution in
# the stick outcomes: bust = win at hard 16, final k = draw at hard k. One
# chi-square per open card, Bonferroni over the 10 cards.
test_that("blackjack dealer matches a Monte Carlo port of blackjack.py", {
  set.seed(9)
  tab <- rltoolsR:::rl_env_table(rl_env("blackjack", seed = 1))
  draw <- function(n) pmin(sample.int(13, n, replace = TRUE), 10)
  sum_hand <- function(raw, ace) ifelse(ace & raw + 10 <= 21, raw + 10, raw)
  N <- 50000
  for (d in 1:10) {
    h <- draw(N)
    raw <- d + h; ace <- d == 1 | h == 1
    repeat {
      go <- sum_hand(raw, ace) < 17
      if (!any(go)) break
      c <- draw(sum(go))
      raw[go] <- raw[go] + c; ace[go] <- ace[go] | c == 1
    }
    fin <- sum_hand(raw, ace)
    cnt <- c(sum(fin > 21), vapply(17:21, function(k) sum(fin == k), 0))
    p <- c(bj_prob(tab, bj_s(16, d, 0), 0, function(n, r, t) r == 1),
           vapply(17:21, function(k) bj_prob(tab, bj_s(k, d, 0), 0, function(n, r, t) r == 0), 0))
    expect_equal(sum(p), 1, tolerance = 1e-12)
    chi2 <- sum((cnt - N * p)^2 / (N * p))
    expect_lt(chi2, stats::qchisq(1 - 0.001 / 10, 5), label = paste("dealer card", d))
  }
})

# Value iteration on the table: the known basic strategy (no doubling or
# splitting) stands on hard 17+ and hits hard 11 or less against every card.
test_that("blackjack table gives the basic strategy on hard hands", {
  tab <- rltoolsR:::rl_env_table(rl_env("blackjack", seed = 1))
  S <- dim(tab$prob)[1]
  q <- function(V, a) rowSums(tab$prob[, a, ] * (tab$reward[, a, ] + (!tab$terminated[, a, ]) *
                                                   matrix(V[tab$`next`[, a, ] + 1], S)))
  V <- numeric(S)
  for (i in 1:15) V <- pmax(q(V, 1), q(V, 2))
  qs <- q(V, 1); qh <- q(V, 2)
  for (d in 1:10) {
    expect_true(all(qs[bj_s(17:21, d, 0) + 1] > qh[bj_s(17:21, d, 0) + 1]), label = paste("stand, card", d))
    expect_true(all(qh[bj_s(4:11, d, 0) + 1] > qs[bj_s(4:11, d, 0) + 1]), label = paste("hit, card", d))
  }
})

test_that("frozen_lake slips to each of three directions with probability 1/3", {
  env <- rl_env("frozen_lake", seed = 1)
  rl_reset(env)
  N <- 3000
  # state 6 (row 1, col 2), action right: up -> 2, right -> 7 (hole), down -> 10
  nxt <- vapply(seq_len(N), function(t) { toy_load(env, 6); rl_step(env, 2)$obs }, 0)
  expect_setequal(unique(nxt), c(2, 7, 10))
  cnt <- tabulate(match(nxt, c(2, 7, 10)), 3)
  # within 4.5 binomial standard deviations of N / 3
  expect_lt(max(abs(cnt - N / 3)), 4.5 * sqrt(N * (1 / 3) * (2 / 3)))
})
