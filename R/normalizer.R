# Running mean/variance for observation and reward normalization.
#
# Statistics start at mean 0, var 1, count 0: the first rollout is seen
# unnormalized and the first update replaces the prior by the rollout's own
# statistics (count 0 gives the prior no weight), so a degenerate variance
# from a handful of reset observations never occurs.
#
# Normalized observation, the form stored in model metadata:
#   clip((x - mean) / sqrt(var + epsilon), -clip_obs, clip_obs)
# Scaled reward (training only):
#   clip(r / sqrt(var_ret + epsilon), -clip_reward, clip_reward)
# where var_ret is the variance of the running discounted return.

OBS_NORM_FORM <- "clip((x - mean) / sqrt(var + epsilon), -clip_obs, clip_obs)"

running_stats <- function(dim) {
  list(mean = numeric(dim), var = rep(1, dim), count = 0)
}

# Parallel (Chan et al.) update with a features x samples matrix.
running_stats_update <- function(rs, x) {
  n <- ncol(x)
  if (n == 0L) return(rs)
  bm <- rowMeans(x)
  bv <- rowMeans((x - bm)^2)
  tot <- rs$count + n
  d <- bm - rs$mean
  rs$mean <- rs$mean + d * n / tot
  rs$var <- (rs$var * rs$count + bv * n + d^2 * rs$count * n / tot) / tot
  rs$count <- tot
  rs
}

# Frozen observation statistics as stored in model metadata.
obs_norm_meta <- function(rs, epsilon, clip_obs) {
  list(mean = rs$mean, var = rs$var, count = rs$count, epsilon = epsilon,
       clip_obs = clip_obs, form = OBS_NORM_FORM)
}

# features x batch matrix -> network input tensor, normalized in the graph
# when `norm` (from obs_norm_meta()) is given.
obs_input <- function(x, norm = NULL) {
  t <- ag_tensor(x)
  if (is.null(norm)) return(t)
  obs_normalize(t, ag_tensor(matrix(norm$mean)), ag_tensor(obs_scale(norm)), norm$clip_obs)
}

# 1 / sqrt(var + epsilon) as an [obs_dim, 1] matrix.
obs_scale <- function(norm) matrix(1 / sqrt(norm$var + norm$epsilon))

# The normalization on tensors: x [obs_dim, B], mean and scale [obs_dim, 1].
# Shared by obs_input() and the captured forward (policy_forward()), where
# mean and scale are graph inputs because they move between iterations.
obs_normalize <- function(t, m, s, clip) ag_clamp(ag_mul(ag_sub(t, m), s), -clip, clip)

scale_reward <- function(r, rs, epsilon, clip_reward) {
  pmin(pmax(r / sqrt(rs$var + epsilon), -clip_reward), clip_reward)
}
