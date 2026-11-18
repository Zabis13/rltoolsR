# rltoolsR

Reinforcement learning toolkit for R: environments, algorithms and training on top of [ggmlR](https://github.com/Zabis13/ggmlR) (Vulkan GPU with automatic CPU fallback).

> Not to be confused with [Rtools](https://cran.r-project.org/bin/windows/Rtools/), the R build toolchain for Windows.

> Status: early development (v0.1.5). The API is not stable yet.

## Features

- Neural networks run on GPU through ggmlR (Vulkan): forward and backward passes and Adam during training, each update step recorded once as a captured graph and replayed; during PPO collection and in `rl_act()` the forward pass, legal-action mask, action choice (argmax or sampling), its log-probability and the value are one recorded graph launch -- only the actions, log-probabilities and values come back to R. Sampling noise is drawn on the device from a key taken from R's RNG, so `set.seed()` reproduces a run and the CPU and the GPU draw the same noise. GPU environments run as ggml graphs. `device = "auto"` falls back to the CPU when no Vulkan device is found.
- Environments:
  - `rl_env(name)`: continuous control in C++, ported from [RLtools](https://github.com/rl-tools/rl-tools): `pendulum`, `acrobot`, `acrobot_swingup`, `car`, `car_track`
  - `rl_env(name)`, ported from [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) (no Python):
    - classic_control: `cartpole`, `mountain_car`, `mountain_car_continuous`, `pendulum_gym` (action in [-1, 1]), `pendulum_gym_raw` (torque in [-2, 2], as in Gymnasium)
    - toy_text, as exact transition tables: `frozen_lake` (`_8x8`, `_det` for no slipping), `cliff_walking`, `taxi`, `blackjack`; each also as `_onehot` (one-hot observation)
    - box2d, on the bundled Box2D v3: `lunar_lander`, `lunar_lander_continuous`
    - the differences from Gymnasium's API are listed in `?rl_step` ("Differences from Gymnasium")
  - `rl_vec_env(name, n)`: `n` copies stepped in one call, auto-reset with `final_obs`
  - `rl_gpu_env("tictactoe", n)`: batched tic-tac-toe as a ggml graph on GPU/CPU, self-play or random opponent
  - `rl_gpu_env("cartpole" / "mountain_car" / "mountain_car_continuous" / "pendulum_gym" / "pendulum_gym_raw", n)`: the same physics as `rl_env()` as a ggml graph on GPU/CPU
  - `rl_gpu_env("cube", n, size = 3)`: batched N x N x N Rubik's cube as a ggml graph on GPU/CPU, one-hot sticker colours, `6 N` quarter turns, auto-reset to new scrambles; `rl_cube_walks()` for labelled random walks (prev/next moves, `no_undo` / `no_cancel`)
  - Common API: `rl_reset()`, `rl_step()`, `rl_legal_actions()`, `rl_save_state()` / `rl_load_state()`
  - Gymnasium-style spaces: `space_box()`, `space_discrete()`, `space_multi_discrete()`, `space_multi_binary()`, `space_dict()`
- PPO (`rl_ppo()`, `rl_ppo_train()`): discrete actions with legal-action masking and self-play, continuous actions with a Gaussian policy; observation and reward normalization; fictitious self-play against a pool of past policies (`pool`); resident rollout on the GPU (`rollout = "resident"`: policy and environment in one recorded graph per step) for tic-tac-toe self-play and the toy_text environments
- Cube Q-model (`rl_cube_q()`): online training on fresh walks, one value per move (predicted distance to solved), MLP with residual blocks; `rl_cube_q_values()`, `rl_cube_q_eval()` (pair / top1 accuracy by depth)
- Training loop `rl_train()` for any `rl_algo()` (write your own): step budget, evaluation during training on its own RNG stream, CSV logs written while training runs, atomic checkpoints with the full training state (weights, normalizer statistics, Adam moments, RNG streams) and resume
- Models: `rl_save_model()` / `rl_load_model()` (train on GPU, run on CPU), `rl_act()`, `rl_save_checkpoint()`
- Evaluation `rl_evaluate()`: mean return and episode length over N episodes, reproducible seeded episodes, per-episode results; greedy or sampled actions (each episode samples from its own RNG stream, so results do not depend on the batch size); usable as `eval_fn` in `rl_train()`
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
- `check_collect_cpu_gpu.R`: PPO rollouts on the GPU against the CPU, same weights and seed
- `check_cube.R`: cube move tables, environment and walk generator checks
- `train_cube3_q.R`: train a 3x3x3 Q-model online, `Rscript train_cube3_q.R depth=10 updates=2000`
- `diag_cube3_q_solve.R`: solve scrambles of growing depth with beam search over a Q-model, summary by depth

## Vignettes

- *Getting started*: environments and spaces, PPO on Pendulum, evaluation, saving and loading models, tic-tac-toe self-play
- *Training loop*: evaluation during training, CSV logs, checkpoints and resume, the GPU, writing your own algorithm

```r
browseVignettes("rltoolsR")
```

## Architecture

| Layer | Implementation |
|---|---|
| Environments | C++ via `.Call`; ggml graphs for GPU environments |
| Neural networks | ggmlR (Vulkan / CPU) |
| Algorithms, training loop | R |

## Mortal Kombat 3 (Genesis) RAM map

Work RAM of the Sega Genesis cartridge under MAME (`genesis`, 68000,
big-endian, `0xFF0000`-`0xFFFFFF`). Found with
`inst/examples/mame_ram_search.R` and `mame_ram_fighters.R`; HP and rounds
won match openai/retro.

| Field | P1 | P2 | Notes |
|---|---|---|---|
| HP | `0xFFF865` | `0xFFF9D7` | u8, full = 166 |
| Rounds won | `0xFFF86B` | `0xFFF9DD` | u8 |
| X | `0xFFCB1C` | `0xFFCB68` | u16; P1 starts at 500, P2 at 708 |
| Y | `0xFFCB20` | `0xFFCB6C` | s16; 103 on the ground of the first level, smaller in the air, negative above the screen; another level after an uppercut through the floor (e.g. 303) |
| Vertical speed | `0xFFCB18` | | s16, P1 only checked |
| In the air | `0xFFCB12` | `0xFFCB5E` | bit 7; not set while flying after a hit or a throw |
| Fighter (select screen) | `0xFF0127` | `0xFF0129` | u8, see below |
| Fighter id, current form | `0xFFCB4B` | `0xFFCB97` | u8, changes while Shang Tsung morphs |
| Round timer | `0xFFF5A3` (tens), `0xFFF5A5` (ones) | | decimal digits, 99 down; -1 every 40 frames |

Fighter ids follow the arcade MK3 order: 0 Kano, 1 Sonya, 2 Jax,
3 Nightwolf, 4 Sub-Zero, 5 Stryker, 6 Sindel, 7 Sektor, 8 Cyrax,
9 Kung Lao, 10 Kabal, 11 Sheeva, 12 Shang Tsung, 13 Liu Kang, 14 Smoke
(hidden). Checked: 0–13.

On the select screen `0xFF0127` / `0xFF0129` hold the cursor cell, not the
id: a 5 x 3 grid, cell 7 (centre) skipped. Cells 0–14: Shang Tsung, Sindel,
Jax, Kano, Liu Kang / Sonya, Stryker, –, Sub-Zero, Cyrax / Sektor,
Nightwolf, Sheeva, Kung Lao, Kabal.

Snapshots for `rl_mame_env()` are made from power-on by
`inst/examples/mame_make_snapshot.R` (Lua `inst/mame/mk3_snapshot.lua`):
`mode=vs` for any pair (both fighters controlled by the caller, for
`opponent = "policy"`), `mode=1p` for P1 against the CPU in tower 1 (the
game picks the opponent; for `opponent = "cpu"`). The state is saved 20
frames after the round starts (both HP 166, timer 99).

The fighter objects are `0x4C` bytes apart (P1 at `0xFFCB00`), the fighter
records holding HP `0x172` apart. The round ends on a KO (an HP reaches 0)
or when the timer reaches 0. Input is locked for about 80 frames after the
round starts ("Fight!").

Block is pad B (held; down + B is the low block). Run is pad Y, held
together with the direction towards the opponent; it lasts while both are
held. Both are held for the whole step; the other buttons (attacks) are
pressed for one frame.

Resetting from a snapshot: release all inputs and run a few frames
(4 here) before `load`. The six-button pad keeps the last pressed buttons
outside the snapshot; loading in the same frame as the release lets the
game see them after the load, and the episode then depends on how the
previous one ended.

## Installation

```r
install.packages("ggmlR")          # from CRAN
# install.packages("remotes")
remotes::install_github("Zabis13/rltoolsR", build_vignettes = TRUE)
```

## License

MIT © Yuri Baramykov.

The package uses [RLtools](https://github.com/rl-tools/rl-tools) (MIT, Copyright (c) 2023 Jonas Eschmann); the continuous-control environments in `src/env_control.cpp` are ported from it (the acrobot from Gymnasium).

It bundles the [Box2D](https://box2d.org) physics engine by Erin Catto (v3, [github.com/erincatto/box2d](https://github.com/erincatto/box2d), MIT, Copyright (c) 2022 Erin Catto) in `src/box2d`, used by the Box2D environments (`lunar_lander`).

The classic_control, toy_text and box2d environments follow [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) (MIT, Copyright (c) 2016 OpenAI, (c) Farama Foundation). See `inst/COPYRIGHTS`.

## Author

Yuri Baramykov

## References

- Eschmann J., Albani D., Loianno G. (2024). *RLtools: A Fast, Portable Deep Reinforcement Learning Library for Continuous Control*. JMLR. <https://arxiv.org/abs/2306.03530>
