suppressMessages(library(ggmlR))

net_fn <- function(obs_dim = 9L, n_actions = 9L, hidden = 16L) {
  l1 <- ag_linear(obs_dim, hidden, activation = "relu")
  l2 <- ag_linear(hidden, hidden, activation = "relu")
  pi <- ag_linear(hidden, n_actions)
  v  <- ag_linear(hidden, 1L, activation = "tanh")
  forward <- function(x) {
    h <- l2$forward(l1$forward(x))
    list(logits = pi$forward(h), value = v$forward(h))
  }
  parameters <- function() list(
    l1_W = l1$W, l1_b = l1$b, l2_W = l2$W, l2_b = l2$b,
    pi_W = pi$W, pi_b = pi$b, v_W = v$W, v_b = v$b)
  structure(list(forward = forward, parameters = parameters,
                 params = function() unname(parameters())),
            class = c("rl_ppo_net", "ag_layer"))
}

gpu_ok <- ggml_vulkan_available() && ggml_vulkan_device_count() > 0L
cat("gpu available:", gpu_ok, "\n")

set.seed(1)
x <- matrix(runif(9 * 5), 9, 5)
fwd <- function(net) { o <- net$forward(ag_tensor(x)); list(as.matrix(o$logits), as.matrix(o$value)) }
wts <- function(net) lapply(net$parameters(), as.matrix)

check <- function(save_dev, load_dev, dtype = NULL) {
  ag_device(save_dev)
  net <- net_fn()
  cat(sprintf("\n[%s -> %s, dtype=%s]\n", save_dev, load_dev, format(dtype)))
  cat("class:", class(net), "| names:", names(net$parameters()), "\n")
  f <- tempfile(fileext = ".rds")
  ag_save_model(net, f, model_fn = function() net_fn())
  w0 <- wts(net); y0 <- fwd(net)
  net2 <- ag_load_model(f, device = load_dev, dtype = dtype)
  ag_device(load_dev)
  w1 <- wts(net2); y1 <- fwd(net2)
  cat("weights identical:", identical(w0, w1), "\n")
  cat("forward max diff: logits", max(abs(y0[[1]] - y1[[1]])),
      "value", max(abs(y0[[2]] - y1[[2]])), "\n")
}

check("cpu", "cpu")
if (gpu_ok) {
  check("cpu", "gpu")
  check("gpu", "cpu")
  check("gpu", "gpu", dtype = "f16")
}
