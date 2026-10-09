# rltoolsR

Reinforcement learning toolkit for R: environments, algorithms and training on top of [ggmlR](https://github.com/Zabis13/ggmlR) (Vulkan GPU with automatic CPU fallback).

> Not to be confused with [Rtools](https://cran.r-project.org/bin/windows/Rtools/), the R build toolchain for Windows.

> Status: early development (v0.1.1). The API is not stable yet.

## Features

- Neural networks run on GPU through ggmlR (Vulkan): forward and backward passes and Adam during training, and the forward pass in `rl_act()`. Action selection and masking run in R. GPU environments run as ggml graphs. `device = "auto"` falls back to the CPU when no Vulkan device is found.
- Environments:
  - `rl_env(name)`: [RLtools](https://github.com/rl-tools/rl-tools) continuous control in C++: `pendulum`, `acrobot`, `acrobot_swingup`, `car`, `car_track`
  - `rl_vec_env(name, n)`: `n` copies stepped in one call, auto-reset with `final_obs`
  - `rl_gpu_env("tictactoe", n)`: batched tic-tac-toe as a ggml graph on GPU/CPU, self-play or random opponent
  - Common API: `rl_reset()`, `rl_step()`, `rl_legal_actions()`, `rl_save_state()` / `rl_load_state()`
  - Gymnasium-style spaces: `space_box()`, `space_discrete()`, `space_multi_discrete()`, `space_multi_binary()`, `space_dict()`
- PPO (`rl_ppo()`, `rl_ppo_train()`): discrete actions with legal-action masking and self-play, continuous actions with a Gaussian policy; observation and reward normalization
- Training loop `rl_train()` for any `rl_algo()`: step budget, separate RNG streams for training and evaluation, CSV logging, checkpoints and resume
- Models: `rl_save_model()` / `rl_load_model()` (train on GPU, run on CPU), `rl_act()`, `rl_save_checkpoint()`
- Evaluation `rl_evaluate()`: mean return and episode length over N episodes, deterministic policy, reproducible seeded slots, per-episode results; usable as `eval_fn` in `rl_train()`
- Replay buffer for off-policy algorithms (internal): transitions in one device buffer, batches gathered and split into fields on the GPU
- Play tic-tac-toe against a trained model: `rl_play_tictactoe()`

## Quick start

```r
library(rltoolsR)

env <- rl_gpu_env("tictactoe", n = 256, opponent = "self")
model <- rl_ppo_train(env, iterations = 100, minibatch = 2048)
rl_save_model(model, "/tmp/ttt.rds")

m <- rl_load_model("/tmp/ttt.rds", device = "cpu")
rl_ppo_eval_tictactoe(m, n = 1000, device = "cpu")
rl_play_tictactoe(m, human = "O")
```

Scripts in `inst/examples`:

- `train_and_run_tictactoe.R`: train PPO on tic-tac-toe, save, load, evaluate
- `train_and_run_pendulum.R`: PPO with a Gaussian policy on Pendulum
- `play_tictactoe.R`: play against a saved model, `Rscript play_tictactoe.R model.rds O`

## Architecture

| Layer | Implementation |
|---|---|
| Environments | C++ (RLtools headers) via `.Call`; ggml graphs for GPU environments |
| Neural networks | ggmlR (Vulkan / CPU) |
| Algorithms, training loop | R |

## Installation

```r
# install.packages("remotes")
remotes::install_github("Zabis13/ggmlR")
remotes::install_github("Zabis13/rltoolsR")
```

## License

MIT © Yuri Baramykov.

The package uses [RLtools](https://github.com/rl-tools/rl-tools) (MIT, Copyright (c) 2023 Jonas Eschmann); a subset of its headers is bundled in `inst/include/rl_tools`. See `inst/COPYRIGHTS`.

## Author

Yuri Baramykov

## References

- Eschmann J., Albani D., Loianno G. (2024). *RLtools: A Fast, Portable Deep Reinforcement Learning Library for Continuous Control*. JMLR. <https://arxiv.org/abs/2306.03530>
