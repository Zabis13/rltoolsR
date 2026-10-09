# Repro for the ggmlR Vulkan bug: concurrent sets from graph_optimize + ggml-alloc
# memory reuse. Expected done: 0 0 0 0.
#   REW=1 Rscript tools/ggmlr_vk_alloc_race_repro.R            -> 1 1 1 1 (bug)
#   REW=1 ALLOUT=1 Rscript tools/ggmlr_vk_alloc_race_repro.R   -> 0 0 0 0 (no memory reuse)
#   REW=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 Rscript ...           -> 0 0 0 0
# Uses internal helpers from R/gpu_tictactoe.R (run from the package root).
suppressMessages(library(rltoolsR))
source("R/gpu_tictactoe.R")
REW <- Sys.getenv("REW") == "1"; CPY <- Sys.getenv("CPY") == "1"
be <- ggml_vulkan_init(0L); n <- 4
cs <- ggml_init(1e6, no_alloc = TRUE)
board <- ggml_new_tensor_2d(cs, GGML_TYPE_F32, 9, n); player <- ggml_new_tensor_2d(cs, GGML_TYPE_F32, 1, n)
b2 <- ggml_new_tensor_2d(cs, GGML_TYPE_F32, 9, n); p2 <- ggml_new_tensor_2d(cs, GGML_TYPE_F32, 1, n)
L <- ggml_new_tensor_2d(cs, GGML_TYPE_F32, 9, 8); I9 <- ggml_new_tensor_2d(cs, GGML_TYPE_F32, 9, 9)
act <- ggml_new_tensor_1d(cs, GGML_TYPE_I32, n)
buf <- ggml_backend_alloc_ctx_tensors(cs, be)
ggml_backend_tensor_set_data(board, numeric(9*n)); ggml_backend_tensor_set_data(player, rep(1, n))
ggml_backend_tensor_set_data(L, as.numeric(ttt_lines())); ggml_backend_tensor_set_data(I9, as.numeric(diag(9)))
ggml_backend_tensor_set_data(act, c(0L, 4L, 8L, 2L))
g <- ggml_init(1e7, no_alloc = TRUE)
ones <- ggml_sqr(g, player); onehot <- ggml_get_rows(g, I9, act)
m1 <- ttt_move(g, board, onehot, player, ones, L)
done <- ttt_ind(g, ggml_add(g, ggml_add(g, m1$illegal, m1$win), ttt_full(g, m1$board)))
reward <- ggml_sub(g, m1$win, m1$illegal)
np <- ggml_add(g, ggml_mul(g, ggml_neg(g, player), ggml_scale_bias(g, done, -1, 1)), done)
nb <- ggml_mul(g, m1$board, ggml_scale_bias(g, done, -1, 1))
outs <- list(obs = ggml_mul(g, nb, np), legal = ggml_scale_bias(g, ggml_sqr(g, nb), -1, 1))
if (REW) outs$reward <- reward
outs$done <- done; outs$np <- np
for (t in outs) ggml_set_output(t)
gr <- ggml_build_forward_expand(g, outs[[1]]); for (t in outs[-1]) ggml_graph_expand(gr, t)
if (CPY) { ggml_graph_expand(gr, ggml_cpy(g, nb, b2)); ggml_graph_expand(gr, ggml_cpy(g, np, p2)) }
if (Sys.getenv("ALLOUT") == "1") for (i in seq_len(ggml_graph_n_nodes(gr))) ggml_set_output(ggml_graph_node(gr, i - 1))
s <- ggml_backend_sched_new(list(be), parallel = FALSE)
ggml_backend_sched_alloc_graph(s, gr); ggml_backend_sched_graph_compute(s, gr)
cat(sprintf("REW=%d CPY=%d  done: %s\n", REW, CPY, paste(ggml_backend_tensor_get_data(outs$done), collapse = " ")))
