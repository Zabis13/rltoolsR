# rltoolsR 0.1.1

* New `rl_evaluate()` — mean return and length over N episodes on a vectorized environment, deterministic policy by default, one episode per seeded slot (independent of the number of copies), truncated episodes flagged; per-episode results in the `episodes` attribute. Usable as `eval_fn` in `rl_train()`.
* Replay buffer for off-policy algorithms (internal): transitions packed in one ggmlR device buffer, ring writes, batches gathered and split into fields on the device, saved and loaded in parts.
* Requires ggmlR (>= 0.8.7).

# rltoolsR 0.1.0

* Initial release. Bundles a subset of the RLtools headers; native code via `.Call`, no Rcpp.
* Environments: RLtools continuous control (`pendulum`, `acrobot`, `acrobot_swingup`, `car`, `car_track`) through `rl_env()` and the vectorized `rl_vec_env()`, plus batched tic-tac-toe on the GPU, `rl_gpu_env()`. Gymnasium-style spaces.
* PPO with `rl_ppo_train()` and `rl_act()` on ggmlR autograd, GPU or CPU, with legal-action masking and self-play.
* `rl_load_model()` loads on the CPU by default.
