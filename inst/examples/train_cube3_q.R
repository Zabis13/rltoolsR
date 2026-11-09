# Q-model for the 3x3x3 trained online on fresh walks (rl_cube_q).
#   Rscript inst/examples/train_cube3_q.R [key=value ...]
#   keys: device=gpu batch=1024 updates=20000 depth=20 k_min=2 h1=1024 h2=512
#         blocks=2 lr=1e-3 lr_min=1e-5 warmup=500 val=4096 eval=500 seed=1
# Writes /tmp/cube3_q.rds (weights + meta + history) and /tmp/cube3_q*.csv.
devtools::load_all(quiet = TRUE)

opt <- list(device = "gpu", batch = 1024L, updates = 20000L, depth = 20L, k_min = 2L,
            h1 = 1024L, h2 = 512L, blocks = 2L, lr = 1e-3, lr_min = 1e-5,
            warmup = 500L, val = 4096L, eval = 500L, seed = 1L)
for (a in commandArgs(TRUE)) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (!kv[1] %in% names(opt)) stop("unknown key: ", kv[1])
  opt[[kv[1]]] <- if (is.character(opt[[kv[1]]])) kv[2] else as(kv[2], class(opt[[kv[1]]]))
}
str(opt)
set.seed(opt$seed)

# held-out set: drawn once, scored at every evaluation
env_val <- rl_gpu_env("cube", n = opt$val, size = 3, device = opt$device)
val <- rl_cube_walks(env_val, n_moves = opt$depth, k_min = opt$k_min)
rm(env_val); invisible(gc())

env <- rl_gpu_env("cube", n = opt$batch, size = 3, device = opt$device)
algo <- rl_cube_q(n_moves = opt$depth, k_min = opt$k_min, hidden = c(opt$h1, opt$h2),
                  blocks = opt$blocks, lr = opt$lr, lr_min = opt$lr_min,
                  warmup = opt$warmup, updates = opt$updates)
m <- rl_train(env, algo, total_steps = opt$batch * opt$updates, device = opt$device,
              eval_every_steps = opt$batch * opt$eval,
              eval_fn = function(model) rl_cube_q_eval(model, val),
              log = "/tmp/cube3_q.csv", overwrite = TRUE)

cat("\nby depth (held-out):\n")
print(rl_cube_q_eval(m, val, by_depth = TRUE), digits = 3, row.names = FALSE)
saveRDS(list(weights = rltoolsR:::save_net(m$net), meta = m$meta,
             history = m$history, eval = m$eval, opt = opt), "/tmp/cube3_q.rds")
cat("saved /tmp/cube3_q.rds\n")
