# Why does the Pendulum (Box) PPO update not use ggmlR's graph backward?
#
#   Rscript inst/examples/diag_box_backward.R
#
# profile_ppo_stages.R pendulum: "graph backward: none recorded", 2160
# downloads (43.8 MB) per iteration in the backward stage -- the closure path.
# Tic-tac-toe on the same build does run as a graph. Here: one PPO iteration on
# the GPU in rl_train() modes, then
#   1. ggmlR's own count of fallback reasons and the path of the last backward;
#   2. the same update again in strict mode (an error inside the graph path is
#      raised instead of silently falling back), with its message and call.

suppressPackageStartupMessages({ library(rltoolsR); library(ggmlR) })
ns <- asNamespace("ggmlR")

set.seed(1)
env <- rl_vec_env("pendulum", n = 16, seed = 1)
algo <- rl_ppo(n_steps = 128, epochs = 1, minibatch = 256, lr = 1e-3,
               gamma = 0.9, lambda = 0.95, ent_coef = 0, hidden = 64)
ag_device("gpu")
ag_graph_mode(TRUE)
ag_matmul_precision("f32")
state <- algo$init(env, "gpu")
col <- algo$collect(state, env)

cat("graph backward enabled:", ns$.ag_bwd_is_enabled(), "\n")
ns$ag_backward_fallbacks(reset = TRUE)
upd <- algo$update(col$state)
cat("\n-- 1. fallback reasons (backward() calls) --\n")
print(ns$ag_backward_fallbacks())
cat("last path:", format(ns$ag_backward_path()), "\n")

cat("\n-- 2. strict mode --\n")
old <- options(ggmlR.ag_backward_strict = TRUE)
r <- tryCatch({ algo$update(col$state); "no error" },
              error = function(e) {
                cat("error:", conditionMessage(e), "\n")
                cat("call: ", deparse(conditionCall(e))[1], "\n")
                "error"
              })
options(old)
cat("result:", r, "\n")
