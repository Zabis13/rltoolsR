# Device selection shared by networks and GPU environments.

gpu_available <- function() {
  isTRUE(ggml_vulkan_available()) && ggml_vulkan_device_count() > 0L
}

# Resolve a `device` argument to "gpu" or "cpu".
# "auto" picks the GPU when a Vulkan device is present and falls back to the
# CPU with a message; an explicit "gpu" without a device is an error. Pass the
# device explicitly when results must be reproducible (tests, benchmarks).
rl_resolve_device <- function(device = c("auto", "gpu", "cpu")) {
  device <- match.arg(device)
  if (device == "auto") {
    if (gpu_available()) return("gpu")
    message("rltoolsR: no Vulkan device available, using CPU")
    return("cpu")
  }
  if (device == "gpu" && !gpu_available()) {
    stop("device = \"gpu\" requested but no Vulkan device is available")
  }
  device
}
