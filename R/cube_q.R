# Q-model for the cube: one output per move, the predicted distance to solved
# after making it. Trained online on rl_cube_walks() batches, as cayleyR's
# train_cube3_q_online.R: each state is labelled at two moves only -- prev
# (back toward solved, target depth - 1) and next (further away, depth + 1) --
# and a mask keeps the other outputs out of the loss. Every batch is fresh.
#
# Targets are scaled to (y - y_mu) / y_sd; with cuts uniform on
# k_min..n_moves-1 and the two targets depth -/+ 1, y_mu is the mean cut and
# y_sd^2 the variance of the cut plus 1. Unlabelled outputs get target 0 in
# scaled units (the mean): the mask zeroes their gradient, and the blandest
# value keeps the shared trunk out of trouble (measured in cayleyR).

# MLP with residual blocks (DeepCubeA's shape): in -> h1 -> h2, `blocks` x
# [h2 -> h2 relu -> h2, + skip, relu], -> n_actions.
cube_q_net <- function(obs_dim, n_actions, hidden, blocks) {
  l1 <- ag_linear(obs_dim, hidden[1], activation = "relu")
  l2 <- ag_linear(hidden[1], hidden[2], activation = "relu")
  res <- lapply(seq_len(blocks), function(i)
    list(a = ag_linear(hidden[2], hidden[2], activation = "relu"),
         b = ag_linear(hidden[2], hidden[2])))
  out <- ag_linear(hidden[2], n_actions)
  forward <- function(x) {
    h <- l2$forward(l1$forward(x))
    for (r in res) h <- ag_relu(ag_add(h, r$b$forward(r$a$forward(h))))
    list(q = out$forward(h))
  }
  parameters <- function() {
    p <- list(l1_W = l1$W, l1_b = l1$b, l2_W = l2$W, l2_b = l2$b)
    for (i in seq_along(res)) {
      p[[paste0("r", i, "a_W")]] <- res[[i]]$a$W
      p[[paste0("r", i, "a_b")]] <- res[[i]]$a$b
      p[[paste0("r", i, "b_W")]] <- res[[i]]$b$W
      p[[paste0("r", i, "b_b")]] <- res[[i]]$b$b
    }
    c(p, list(out_W = out$W, out_b = out$b))
  }
  structure(list(forward = forward, parameters = parameters,
                 captured = new.env(parent = emptyenv())),
            class = c("rl_cube_q_net", "ag_layer"))
}

cube_q_build <- function(meta) cube_q_net(meta$obs_dim, meta$n_actions, meta$hidden, meta$blocks)

#' Q-model training for the cube
#'
#' An [rl_algo()] for [rl_train()] on an environment from
#' `rl_gpu_env("cube", ...)`: every iteration draws `env$n` fresh labelled
#' states with [rl_cube_walks()] and makes one Adam step on them. The network
#' predicts, for each move, the distance to solved after making it; the loss
#' is the mean squared error on the two labelled moves of each state (`prev`:
#' depth - 1, `next`: depth + 1). Steps count states, so `total_steps` in
#' [rl_train()] is the number of states seen.
#'
#' The learning rate warms up linearly over `warmup` updates and then follows
#' a cosine down to `lr_min` at update `updates` (constant `lr` with
#' `sched = "const"`).
#'
#' @param n_moves,k_min Walk length bound and shallowest cut, see
#'   [rl_cube_walks()].
#' @param hidden Two layer widths: input layer and residual width.
#' @param blocks Number of residual blocks (0: plain MLP).
#' @param lr,lr_min,warmup,updates,sched Learning-rate schedule; `updates`
#'   is the length of the cosine, normally `total_steps / env$n`.
#' @param max_grad_norm Optional gradient clipping.
#' @param walks Where the walks are made. `"resident"`: on the training
#'   device -- the moves are drawn and made in one recorded graph, all steps
#'   of a walk queued in one go, and the training step reads the end states
#'   and the labels there; per iteration only a noise key crosses the bus.
#'   The walks then live in the algorithm's own device buffers and the games
#'   of `env` are left alone. `"host"`: with [rl_cube_walks()], the moves
#'   drawn in R, one graph launch per move and the states through R. `"auto"`
#'   (the default) picks `"resident"` when training on the GPU. The two obey
#'   the same rules and follow `set.seed()`, but draw different walks (the
#'   device has its own generator).
#' @param no_cancel Walks refuse cancelling moves, see [rl_cube_walks()].
#' @return An [rl_algo()]. The model from [rl_train()] has class
#'   `rl_cube_q`; see [rl_cube_q_values()] and [rl_cube_q_eval()].
#'   [rl_save_model()], [rl_load_model()] and the checkpoints of [rl_train()]
#'   (`checkpoint`, `resume`) do not support it yet.
#' @export
#' @examples
#' \donttest{
#' env <- rl_gpu_env("cube", n = 256, size = 3, device = "cpu")
#' algo <- rl_cube_q(n_moves = 8, hidden = c(128, 64), blocks = 1, updates = 20)
#' m <- rl_train(env, algo, total_steps = 256 * 20, device = "cpu", verbose = FALSE)
#' val <- rl_cube_walks(env, n_moves = 8)
#' rl_cube_q_eval(m, val)
#' }
rl_cube_q <- function(n_moves = 20L, k_min = 2L, hidden = c(1024L, 512L), blocks = 2L,
                      lr = 1e-3, lr_min = 1e-5, warmup = 500L, updates = 20000L,
                      sched = c("cosine", "const"), max_grad_norm = NULL,
                      walks = c("auto", "host", "resident"), no_cancel = FALSE) {
  sched <- match.arg(sched)
  walks <- match.arg(walks)
  no_cancel <- isTRUE(no_cancel)
  n_moves <- as.integer(n_moves); k_min <- as.integer(k_min)
  hidden <- as.integer(hidden); blocks <- as.integer(blocks)
  if (k_min < 1L || k_min > n_moves - 1L) stop("need 1 <= k_min <= n_moves - 1")
  if (length(hidden) != 2L || anyNA(hidden) || any(hidden < 1L))
    stop("`hidden` must be two positive widths")
  cuts <- k_min:(n_moves - 1L)
  y_mu <- mean(cuts)
  y_sd <- sqrt(mean((cuts - y_mu)^2) + 1)

  lr_at <- function(u) {
    if (sched == "const") return(lr)
    if (u <= warmup) return(lr * u / max(1L, warmup))
    t <- min(1, (u - warmup) / max(1, updates - warmup))
    lr_min + 0.5 * (lr - lr_min) * (1 + cos(pi * t))
  }

  init <- function(env, device) {
    if (!inherits(env, "rl_gpu_cube")) stop("rl_cube_q() needs rl_gpu_env(\"cube\", ...)")
    meta <- list(algo = "cube_q", obs_dim = env$observation_space$shape,
                 n_actions = env$action_space$n, hidden = hidden, blocks = blocks,
                 size = env$size, move_names = env$move_names,
                 y_mu = y_mu, y_sd = y_sd, n_moves = n_moves, k_min = k_min)
    net <- cube_q_build(meta)
    list(net = net, opt = optimizer_adam(net$parameters(), lr = lr_at(1L)),
         meta = meta, device = device, updates = 0L, batch = NULL,
         step = NULL)
  }

  collect <- function(state, env) {
    resident <- walks == "resident" || walks == "auto" && identical(state$device, "gpu")
    B <- env$n
    if (resident) {
      # R/resident_cube.R: walks and labels are made and stay on the device
      if (is.null(state$res))
        state$res <- cube_res_new(env$st, n_moves, state$device, k_min, no_cancel)
      cube_res_sample(state$res)
      state$batch <- list(all = state$res$all, cols = state$res$cols,
                          c_prev = matrix((-1 - y_mu) / y_sd, 1L, B),
                          c_next = matrix((1 - y_mu) / y_sd, 1L, B))
      return(list(state = state, steps = B, info = NULL))
    }
    w <- rl_cube_walks(env, n_moves = n_moves, k_min = k_min, no_cancel = no_cancel)
    M <- state$meta$n_actions
    y <- mask <- matrix(0, M, B)
    ip <- cbind(w$prev + 1L, seq_len(B))
    ix <- cbind(w$`next` + 1L, seq_len(B))
    y[ip] <- (w$depth - 1L - y_mu) / y_sd
    y[ix] <- (w$depth + 1L - y_mu) / y_sd
    mask[ip] <- 1
    mask[ix] <- 1
    state$batch <- list(x = t(w$obs), y = y, mask = mask)
    list(state = state, steps = B, info = NULL)
  }

  update <- function(state) {
    u <- state$updates + 1L
    state$opt$lr <- lr_at(u)
    if (is.null(state$step)) {
      net <- state$net
      M <- state$meta$n_actions
      # mean over the M x B grid, rescaled to the mean over the 2 B labels
      loss <- function(x, y, mask) {
        d <- ag_mul(ag_sub(net$forward(x)$q, y), mask)
        list(loss = ag_scale(ag_mean(ag_mul(d, d)), M / 2))
      }
      res <- state$res
      # resident walks: observations and labels are read from the walk's buffers
      fn <- if (is.null(res)) loss else function(all, cols, c_prev, c_next) {
        lb <- cube_res_labels(res, cols, c_prev, c_next, 1 / y_sd)
        loss(cube_res_obs(res, all), lb$y, lb$mask)
      }
      state$step <- ag_capture_step(fn, params = if (!is.null(res)) cube_res_params(res),
                                    optimizer = state$opt, max_grad_norm = max_grad_norm)
    }
    out <- do.call(state$step, state$batch)
    state$updates <- u
    m <- c(loss = out$loss[1], lr = state$opt$lr)
    if (!is.null(out$grad_norm)) m <- c(m, grad_norm = out$grad_norm[1])
    list(state = state, metrics = m)
  }

  model <- function(state) {
    structure(list(net = state$net, device = state$device, meta = state$meta,
                   history = NULL), class = "rl_cube_q")
  }

  get_train_state <- function(state) {
    list(weights = save_net(state$net), optimizer_state = state$opt$state(),
         counters = list(updates = state$updates))
  }

  set_train_state <- function(state, train_state) {
    net <- load_net(train_state$weights, function() cube_q_build(state$meta), state$device)
    state$net <- net
    state$opt <- optimizer_adam(net$parameters(), lr = lr)
    if (!is.null(state$step)) ag_capture_free(state$step)
    state["step"] <- list(NULL)
    if (!is.null(train_state$optimizer_state)) {
      withCallingHandlers(
        state$opt$load_state(train_state$optimizer_state),
        ggmlR_opt_hyper_warning = function(w) invokeRestart("muffleWarning"))
    }
    state$updates <- train_state$counters$updates
    state["batch"] <- list(NULL)
    state
  }

  rl_algo("cube_q", init = init, collect = collect, update = update, model = model,
          get_train_state = get_train_state, set_train_state = set_train_state,
          params = list(n_moves = n_moves, k_min = k_min, hidden = hidden,
                        blocks = blocks, lr = lr, lr_min = lr_min, warmup = warmup,
                        updates = updates, sched = sched,
                        max_grad_norm = max_grad_norm, walks = walks,
                        no_cancel = no_cancel))
}

#' Move values of a cube Q-model
#'
#' @param model Result of [rl_train()] with [rl_cube_q()].
#' @param obs Observations, one row per state (as from [rl_step()] or
#'   [rl_cube_walks()]).
#' @return Matrix `nrow(obs)` x moves: predicted distance to solved after
#'   each move, in moves.
#' @export
#' @examples
#' \donttest{
#' env <- rl_gpu_env("cube", n = 64, size = 3, device = "cpu")
#' algo <- rl_cube_q(n_moves = 6, hidden = c(64, 32), blocks = 0, updates = 5)
#' m <- rl_train(env, algo, total_steps = 64 * 5, device = "cpu", verbose = FALSE)
#' obs <- rl_reset(env)
#' q <- rl_cube_q_values(m, obs)
#' env$move_names[max.col(-q, "first")]   # the move the model rates best
#' }
rl_cube_q_values <- function(model, obs) {
  obs <- obs_host(obs)
  meta <- model$meta
  q <- with_policy_modes(model, policy_forward(model$net, t(obs))$q)
  t(q) * meta$y_sd + meta$y_mu
}

#' Score a cube Q-model on labelled states
#'
#' `pair`: share of states where the model puts `prev` below `next` (the
#' question a descent asks); `top1`: share where `prev` has the lowest value
#' of all moves; `mae_prev`, `mae_next`: mean absolute error of the two
#' labelled values.
#'
#' @param model Result of [rl_train()] with [rl_cube_q()].
#' @param val A fixed labelled set from [rl_cube_walks()].
#' @param by_depth Return one row per depth instead of the overall numbers.
#' @return Named numeric vector (usable as `eval_fn` in [rl_train()]), or a
#'   data frame with `depth`, `n` and the same columns.
#' @export
#' @examples
#' \donttest{
#' env <- rl_gpu_env("cube", n = 64, size = 3, device = "cpu")
#' val <- rl_cube_walks(env, n_moves = 6)             # fixed held-out set
#' algo <- rl_cube_q(n_moves = 6, hidden = c(64, 32), blocks = 0, updates = 5)
#' m <- rl_train(env, algo, total_steps = 64 * 5, device = "cpu", verbose = FALSE,
#'               eval_every_steps = 64 * 5, eval_fn = function(m) rl_cube_q_eval(m, val))
#' rl_cube_q_eval(m, val, by_depth = TRUE)
#' }
rl_cube_q_eval <- function(model, val, by_depth = FALSE) {
  vp <- rl_cube_q_values(model, val$obs)
  rows <- seq_len(nrow(vp))
  q_prev <- vp[cbind(rows, val$prev + 1L)]
  q_next <- vp[cbind(rows, val$`next` + 1L)]
  d <- val$depth
  one <- function(i) c(pair = mean(q_prev[i] < q_next[i]),
                       top1 = mean(max.col(-vp[i, , drop = FALSE], "first") == val$prev[i] + 1L),
                       mae_prev = mean(abs(q_prev[i] - (d[i] - 1))),
                       mae_next = mean(abs(q_next[i] - (d[i] + 1))))
  if (!by_depth) return(one(rows))
  ds <- sort(unique(d))
  data.frame(depth = ds, n = as.vector(table(d)[as.character(ds)]),
             do.call(rbind, lapply(ds, function(k) one(which(d == k)))))
}
