# rltoolsR 0.1.3

* PPO picks actions on the device: forward, legal-action mask, action choice, its log-probability and the value are one graph launch during collection and in `rl_act()`; the logits are no longer downloaded.
* Action sampling now draws its noise on the device, the same on the CPU and on the GPU: a given `set.seed()` reproduces a run, but gives a different run than 0.1.2.
* GPU environments (`tictactoe`, `cube` with `device = "gpu"`) return observations on the device (`rl_device_obs()`); discrete PPO collects from them without downloading observations (cube collection about 3× faster).
* Requires ggmlR (>= 0.8.9.1).

# rltoolsR 0.1.2

* New `rl_gpu_env("cube", n, size)` — N×N×N Rubik's cube on GPU/CPU, and `rl_cube_walks()` for labelled random walks.
* New CartPole-v1 environment, on the CPU and on the GPU.

# rltoolsR 0.1.1

* New `rl_evaluate()` — mean return and length over N episodes on a vectorized environment, deterministic policy by default, one episode per seeded slot (independent of the number of copies), truncated episodes flagged; per-episode results in the `episodes` attribute. Usable as `eval_fn` in `rl_train()`.
* Replay buffer for off-policy algorithms (internal): transitions packed in one ggmlR device buffer, ring writes, batches gathered and split into fields on the device, saved and loaded in parts.
* Requires ggmlR (>= 0.8.7).
* The RLtools headers are no longer bundled (`inst/include/rl_tools` removed): `pendulum`, `acrobot`, `acrobot_swingup`, `car` and `car_track` are ported to standalone C++ in `src/env_control.cpp`. Same dynamics and rewards; `car_track` lidar no longer reads outside the track map. Trajectories for a given seed may differ in the last float digits.
* Removed `rltools_version()`.

# rltoolsR 0.1.0

* Initial release. Bundles a subset of the RLtools headers; native code via `.Call`, no Rcpp.
* Environments: RLtools continuous control (`pendulum`, `acrobot`, `acrobot_swingup`, `car`, `car_track`) through `rl_env()` and the vectorized `rl_vec_env()`, plus batched tic-tac-toe on the GPU, `rl_gpu_env()`. Gymnasium-style spaces.
* PPO with `rl_ppo_train()` and `rl_act()` on ggmlR autograd, GPU or CPU, with legal-action masking and self-play.
* `rl_load_model()` loads on the CPU by default.
