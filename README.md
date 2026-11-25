# rltoolsR

Reinforcement learning toolkit for R: environments, algorithms and training on top of [ggmlR](https://github.com/Zabis13/ggmlR) (Vulkan GPU with automatic CPU fallback).

> Not to be confused with [Rtools](https://cran.r-project.org/bin/windows/Rtools/), the R build toolchain for Windows.

> Status: early development (v0.1.5). The API is not stable yet.

## Contents

- [Installation](#installation)
- [Overview](#overview)
- [Quick start](#quick-start)
- [Environments](#environments)
- [PPO](#ppo)
- [GPU rollouts: host, device, resident](#gpu-rollouts-host-device-resident)
- [Recurrent PPO](#recurrent-ppo)
- [Rubik's cube Q-model](#rubiks-cube-q-model)
- [Training loop](#training-loop)
- [Models and evaluation](#models-and-evaluation)
- [Off-policy building blocks](#off-policy-building-blocks)
- [Architecture](#architecture)
- [Example scripts](#example-scripts)
- [Vignettes](#vignettes)
- [Appendix: Mortal Kombat 3 (Genesis) RAM map](#appendix-mortal-kombat-3-genesis-ram-map)

## Installation

```r
install.packages("ggmlR")          # from CRAN
# install.packages("remotes")
remotes::install_github("Zabis13/rltoolsR", build_vignettes = TRUE)
```

A Vulkan GPU is optional: `device = "auto"` falls back to the CPU when no
Vulkan device is found. `rl_mame_env()` needs a MAME executable and a
cartridge you own.

## Overview

- Environments: continuous control ported from RLtools, Gymnasium ports
  (classic_control, toy_text, box2d) without Python, batched GPU
  environments (tic-tac-toe, cartpole, mountain car, pendulum, Rubik's cube)
  and Mortal Kombat 3 under MAME
- PPO for discrete and continuous actions, legal-action masking,
  self-play and fictitious self-play, observation and reward normalization
- Resident rollouts: the policy and the environment as one recorded GPU
  graph per step, nothing crosses the bus until the rollout ends
- Recurrent PPO with an LSTM, the update recorded once and replayed
- Rubik's cube Q-model trained online on random walks
- Training loop with evaluation, CSV logs, atomic checkpoints and resume;
  your own algorithms through `rl_algo()`
- Models trained on the GPU run on the CPU; reproducible evaluation

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

## Environments

All environments share one API: `rl_reset()`, `rl_step()`,
`rl_legal_actions()`, `rl_save_state()` / `rl_load_state()`, and
`rl_close()` for environments backed by external processes.
`rl_env_names()` lists the names `rl_env()` and `rl_vec_env()` accept.

```r
rl_env_names()

env <- rl_env("pendulum", seed = 1)
obs <- rl_reset(env)
r <- rl_step(env, 0.5)        # list(obs, reward, terminated, truncated, ...)

venv <- rl_vec_env("cartpole", n = 8, seed = 1)
obs <- rl_reset(venv)         # 8 x obs_dim matrix
r <- rl_step(venv, sample(0:1, 8, replace = TRUE))
r$final_obs                   # last observation of copies that were auto-reset
```

### Continuous control (RLtools)

Ported in C++ from [RLtools](https://github.com/rl-tools/rl-tools):
`pendulum`, `acrobot`, `acrobot_swingup`, `car`, `car_track`.

### Gymnasium

Ported from [Gymnasium](https://github.com/Farama-Foundation/Gymnasium),
no Python involved:

- classic_control: `cartpole`, `mountain_car`, `mountain_car_continuous`,
  `pendulum_gym` (action in [-1, 1]), `pendulum_gym_raw` (torque in
  [-2, 2], as in Gymnasium)
- toy_text, as exact transition tables: `frozen_lake` (`_8x8`, `_det` for
  no slipping), `cliff_walking`, `taxi`, `blackjack`; each also as
  `_onehot` (one-hot observation)
- box2d, on the bundled Box2D v3: `lunar_lander`, `lunar_lander_continuous`

```r
env <- rl_vec_env("frozen_lake_8x8_onehot", n = 16, seed = 1)
env <- rl_vec_env("lunar_lander", n = 8)
```

The differences from Gymnasium's API (for example, `rl_reset()` returns the
observation only, not `(obs, info)`) are listed in `?rl_step`, section
"Differences from Gymnasium".

### GPU environments

`rl_gpu_env(name, n)` runs `n` copies as a ggml graph on the GPU
(`device = "gpu"`) or the CPU:

- `tictactoe`: self-play (`opponent = "self"`) or a random opponent
- `cartpole`, `mountain_car`, `mountain_car_continuous`, `pendulum_gym`,
  `pendulum_gym_raw`: the same physics as `rl_env()`
- `cube`: see [Rubik's cube](#rubiks-cube-q-model)

```r
env <- rl_gpu_env("cartpole", n = 1024, device = "gpu")
obs <- rl_reset(env)          # rl_device_obs: stays in device memory
as.matrix(obs)                # explicit download to the host
```

With `device = "gpu"` observations come back as an `rl_device_obs`, a
handle to the device tensor; nothing is downloaded implicitly (see
`?rl_device_obs`).

### Spaces

Gymnasium-style spaces describe observations and actions:

```r
space_box(low = c(-1, -1), high = c(1, 1))
space_discrete(4)
space_multi_discrete(c(3, 3))
space_multi_binary(5)
s <- space_dict(pos = space_box(-1, 1), act = space_discrete(2))
space_size(s)                 # flattened size
```

### Mortal Kombat 3 under MAME

`rl_mame_env()` runs `n` MAME processes with the Sega Genesis cartridge of
Mortal Kombat 3, each driven by a Lua script over two FIFOs (Linux / macOS
only). One step is `frameskip` emulated frames; actions are
`Discrete(63)` (direction relative to the opponent x pad button);
the observation is 9 numbers from the fighter's own side (HP, distance,
heights, timer, "in the air" flags); the reward is damage dealt minus damage
taken, plus or minus 1 for the round. An episode is one round.

Two modes:

- snapshot mode (`start` given): every round starts from a `.sta`
  snapshot, against the game's AI (`opponent = "cpu"`) or with both
  fighters controlled by the caller (`opponent = "policy"`)
- tower mode (`start = NULL`, `fighter` given): the game is played on
  through tower 1 against the CPU; after a lost match either the fight is
  retried from a checkpoint or the game is continued (`retry`)

```r
env <- rl_mame_env("mk3", n = 4, cart = "Mortal Kombat 3 (USA).zip",
                   start = "sonya_sindel_cpu.sta", mame = "/usr/games/mame")
model <- rl_train(env, rl_ppo(n_steps = 128, minibatch = 256, lr = 3e-4,
                              hidden = 128, normalize_reward = TRUE),
                  total_steps = 5e5, device = "gpu")
rl_close(env)                 # stops the MAME processes
```

Snapshots are made by `inst/examples/mame_make_snapshot.R`; the full
training script is `inst/examples/train_mk3.R`. The RAM addresses the
environment reads are in the
[appendix](#appendix-mortal-kombat-3-genesis-ram-map).

## PPO

`rl_ppo()` returns an algorithm for `rl_train()`; `rl_ppo_train()` is a
shortcut that trains for a number of iterations. Discrete actions use
legal-action masking (`rl_legal_actions()`), continuous actions a Gaussian
policy. Observations and rewards can be normalized with running statistics
(`normalize_obs`, `normalize_reward`).

Continuous actions, Pendulum:

```r
env <- rl_vec_env("pendulum", n = 16, seed = 1)
algo <- rl_ppo(n_steps = 128, epochs = 10, minibatch = 256, lr = 1e-3,
               gamma = 0.9, ent_coef = 0, hidden = 64)
model <- rl_train(env, algo, total_steps = 2e5, device = "auto")
```

Discrete actions, cartpole on the GPU:

```r
env <- rl_gpu_env("cartpole", n = 8, device = "gpu")
algo <- rl_ppo(n_steps = 32, epochs = 20, minibatch = 256, gamma = 0.98,
               lambda = 0.8, ent_coef = 0)
model <- rl_train(env, algo, total_steps = 1e5, device = "gpu")
```

### Self-play and fictitious self-play

Two-player environments (tic-tac-toe) train by self-play: one network plays
both sides, the value being from the side to move. With `pool` the learner
also plays against a pool of its own past policies:

```r
env <- rl_gpu_env("tictactoe", n = 256, opponent = "self", device = "gpu")
algo <- rl_ppo(minibatch = 2048,
               pool = list(p_self = 0.8, every = 10, size = 20))
model <- rl_train(env, algo, total_steps = 256 * 9 * 300, device = "gpu")
```

Every `every` iterations the weights are copied into the pool (at most
`size` snapshots); every game plays self-play with probability `p_self`,
otherwise against one snapshot drawn for the rollout. The history then has
the columns `pool_games`, `pool_win`, `pool_loss` and others (see
`?rl_ppo`).

## GPU rollouts: host, device, resident

PPO collects a rollout in one of three ways; the update is the same
captured graph in all of them.

**Host** (`rollout = "host"`): a loop in R, one environment step per
iteration. With a GPU network each step is still one recorded graph launch
for the forward pass, the legal-action mask, the action choice (argmax or
sampling), its log-probability and the value; only actions,
log-probabilities and values come back to R. This is the path for the C++
environments and for MAME.

**Device**: a GPU environment (`rl_gpu_env(..., device = "gpu")`) on the
host path. The loop is still in R, but observations stay on the device
(`rl_device_obs`) and PPO copies them into its rollout buffer without
going through the host. Chosen automatically for discrete actions with a
GPU network, `capture = TRUE` and without `normalize_obs` (the running
statistics need the observations on the host).

**Resident** (`rollout = "resident"`): the policy and the environment are
one recorded device graph. The `n_steps` replays of it are queued together
and R waits once; each step reads a step counter on the device and writes
observations, legal masks, actions, log-probabilities, values, rewards and
done/truncated flags into device rollout buffers. Only after the rollout
the buffers come down, and GAE runs on the host. Truncated episodes are
bootstrapped with the value of the final observation, as on the host path;
`normalize_obs` works here too (moments taken on the device, in f32).

Supported for resident rollouts (with a GPU network and `capture = TRUE`):

| Environment | Notes |
|---|---|
| `rl_gpu_env("tictactoe", opponent = "self", device = "gpu")` | also with `pool`: the opponent snapshot runs in the same recording |
| `rl_gpu_env("cartpole" / "pendulum_gym" / "pendulum_gym_raw", device = "gpu")` | |
| `rl_env` / `rl_vec_env("cartpole" / "pendulum_gym" / "pendulum_gym_raw")` | f32 physics on the device, only with an explicit `rollout = "resident"` |
| toy_text: `rl_vec_env("frozen_lake..." / "cliff_walking..." / "taxi..." / "blackjack...")` | exact transition tables on the device |

`rollout = "auto"` (the default) picks resident when it is possible, except
for the C++ cartpole and pendulum, which stay on the host unless asked.
In resident mode the games live in the collection's own device buffers and
`env` itself is not stepped (it only gives the specification).

```r
env <- rl_vec_env("taxi", n = 512)
model <- rl_train(env, rl_ppo(n_steps = 64, minibatch = 8192,
                              rollout = "resident"),
                  total_steps = 512 * 64 * 200, device = "gpu")
```

Sampling noise is drawn on the device from a key taken from R's RNG, so
`set.seed()` reproduces a run, and the host and the device draw the same
noise for the same step.

## Recurrent PPO

`rl_ppo_recurrent()` puts an LSTM between an encoder and the policy and value
heads (one LSTM shared by both, as CleanRL's `ppo_atari_lstm`; SB3's
`RecurrentPPO` uses separate ones by default, so learning curves are not
directly comparable). Minibatches are subsets of the environment copies with
all their steps (`env$n` must be a multiple of `n_minibatches`); the update
re-runs the LSTM over the rollout from the state each copy had at its start,
resetting it where an episode begins. A truncated episode is bootstrapped
from the state it reached.

```r
env <- rl_vec_env("cartpole", n = 8)
m <- rl_train(env, rl_ppo_recurrent(n_steps = 128, n_minibatches = 4),
              total_steps = 8 * 128 * 50)
obs <- rl_reset(env)
r <- rl_act(m, obs)                          # list(action, state)
r <- rl_act(m, obs, state = r$state, starts = c(TRUE, rep(FALSE, 7)))
```

The minibatch update is one recording (`capture = TRUE`, needs ggmlR >=
0.8.9.4) replayed for every minibatch. One minibatch update, hidden 64,
cartpole, Radeon RX 9070 (RADV), median of 5
(`inst/examples/bench_ppo_recurrent_update.R`):

| n_envs | T (`n_steps`) | tape every minibatch | captured | speedup |
|---|---|---|---|---|
| 8 | 32 | 252 ms | 15 ms | 17x |
| 8 | 128 | 1182 ms | 43 ms | 27x |
| 32 | 32 | 265 ms | 17 ms | 16x |
| 32 | 128 | 1306 ms | 62 ms | 21x |

Recording happens once per minibatch shape and settings; its cost grows
slightly faster than T (`inst/examples/bench_ppo_recurrent_record.R`, n_envs
8, pipelines compiled beforehand):

| T | record | per step | replay |
|---|---|---|---|
| 32 | 268 ms | 8.4 ms | 12 ms |
| 64 | 628 ms | 9.8 ms | 22 ms |
| 128 | 1418 ms | 11.1 ms | 43 ms |
| 256 | 3588 ms | 14.0 ms | 85 ms |

Recording grows as about T^1.24 (the replay is linear), so it is a one-off of
a few seconds even at T = 256. Not supported yet: continuous actions, the
device/resident collection paths, `pool`.

## Rubik's cube Q-model

`rl_gpu_env("cube", n, size = 3)` is a batched N x N x N Rubik's cube as a
ggml graph: one-hot sticker colours, `6 N` quarter turns, auto-reset to new
scrambles. `rl_cube_walks()` generates labelled random walks (`obs`,
`prev` / `next` moves, `depth`; walks never undo the previous move, and
`no_cancel = TRUE` also refuses moves that cancel against earlier ones, so
the depth label is a tighter upper bound on the distance).

`rl_cube_q()` trains online on fresh walks: an MLP with residual blocks
predicting, for every move, the distance to solved.
`rl_cube_q_values()` gives these values, `rl_cube_q_eval()` the pair and
top-1 accuracy (optionally by depth).

```r
val <- rl_cube_walks(rl_gpu_env("cube", n = 4096, size = 3, device = "gpu"),
                     n_moves = 20, k_min = 2)
env <- rl_gpu_env("cube", n = 8192, size = 3, device = "gpu")
algo <- rl_cube_q(n_moves = 20, hidden = c(1024, 512), blocks = 2,
                  updates = 2000)
m <- rl_train(env, algo, total_steps = 8192 * 2000, device = "gpu",
              eval_every_steps = 8192 * 200,
              eval_fn = function(model) rl_cube_q_eval(model, val))
rl_cube_q_eval(m, val, by_depth = TRUE)
```

`inst/examples/diag_cube3_q_solve.R` solves scrambles of growing depth with
beam search over a trained Q-model.

## Training loop

`rl_train(env, algo, total_steps)` runs any algorithm built with `rl_algo()`
(PPO, recurrent PPO, the cube Q-model, or your own): a step budget,
evaluation during training on its own RNG stream, CSV logs written while
training runs, atomic checkpoints with the full training state (weights,
normalizer statistics, Adam moments, RNG streams) and resume.

```r
m <- rl_train(env, algo, total_steps = 1e6, device = "gpu",
              eval_every_steps = 1e4,
              eval_fn = function(m) rl_evaluate(m, "cartpole", n_episodes = 100),
              log = "/tmp/run.csv",
              checkpoint = "/tmp/run_ckpt.rds", checkpoint_every_steps = 5e4)

# continue after an interruption
m <- rl_train(env, algo, total_steps = 2e6, device = "gpu",
              resume = "/tmp/run_ckpt.rds", log = "/tmp/run.csv", append = TRUE)
```

Writing your own algorithm with `rl_algo(name, init, collect, update,
model, get_train_state, set_train_state)` is shown in the *Training loop*
vignette.

## Models and evaluation

```r
rl_save_model(model, "/tmp/pendulum.rds")          # weights + metadata, for running
m <- rl_load_model("/tmp/pendulum.rds", device = "cpu")   # train on GPU, run on CPU
rl_save_checkpoint(model, "/tmp/pendulum_ckpt.rds")       # full training state

env <- rl_vec_env("pendulum", n = 8, seed = 100)
obs <- rl_reset(env)
a <- rl_act(m, obs, greedy = TRUE)

rl_evaluate(m, "pendulum", n_episodes = 100, seed = 1e6)
```

`rl_act()` picks actions (argmax or sampled; with legal masks for discrete
environments; with an LSTM state for recurrent models). `rl_evaluate()`
returns the mean and standard deviation of the return and the episode
length over N episodes, with per-episode results in
`attr(, "episodes")`. Episodes are seeded and reproducible; sampled actions
use one RNG stream per episode, so results do not depend on the batch
size. It can be passed as `eval_fn` to `rl_train()`.

Tic-tac-toe has its own tools: `rl_ppo_eval_tictactoe()` (results against
a random player as X or O) and `rl_play_tictactoe()` (play against a model
in the console).

## Off-policy building blocks

Replay buffers come from ggmlR: `ag_replay_buffer()`, prioritized
`ag_per_sampler()` and `ag_soft_update()` for target networks. Transitions
live in one device buffer; batches are gathered and split into fields on
the GPU.

## Architecture

| Layer | Implementation |
|---|---|
| CPU environments | C++ via `.Call` (RLtools ports, Gymnasium ports, Box2D v3) |
| GPU environments | ggml graphs (tic-tac-toe, classic control, cube) |
| MAME environment | MAME processes driven through FIFOs and Lua scripts |
| Neural networks | ggmlR autograd (Vulkan / CPU) |
| Algorithms, training loop | R |

How the GPU is used:

- **Captured update.** Forward and backward passes and Adam of one update
  step are recorded once (`ag_capture()`) and replayed for every
  minibatch; a new recording is made only when the shapes or settings
  change.
- **Policy step as one launch.** During collection the forward pass,
  the legal-action mask, the action choice, its log-probability and the
  value are one recorded graph launch; only the results come back to R.
- **Observations on the device.** GPU environments keep their observations
  in device memory; PPO reads them there.
- **Resident rollout.** The environment step joins the policy step in one
  recording; a whole rollout is a queue of replays with a device step
  counter, and data crosses the bus once per rollout.
- **Noise on the device.** Sampling noise comes from a counter-based key
  derived from R's RNG, so CPU and GPU runs draw the same noise and
  `set.seed()` reproduces a run.

Data flow per PPO iteration:

```
host path:      R loop ── obs ──▶ GPU policy ── action ──▶ R ──▶ env (C++)      × n_steps
device path:    R loop ──▶ GPU policy ◀── obs (device) ── GPU env               × n_steps
resident path:  [GPU policy + GPU env] recorded once, replayed n_steps, R waits once
then:           rollout ──▶ host GAE ──▶ captured update replayed per minibatch
```

## Example scripts

In `inst/examples`, run with `Rscript <script> key=value ...`:

- Training: `train_and_run_tictactoe.R`, `train_and_run_pendulum.R`,
  `train_cartpole.R`, `train_frozen_lake.R`, `train_cube3_q.R`,
  `train_mk3.R`
- Play and watch: `play_tictactoe.R` (`Rscript play_tictactoe.R model.rds
  O`), `watch_mk3.R`
- Learning checks: `check_learning_cartpole_resident.R`,
  `check_learning_pendulum_resident.R`, `check_learning_tictactoe.R`,
  `check_learning_tictactoe_fsp.R`
- CPU against GPU: `check_collect_cpu_gpu.R`, `check_cartpole_cpu_gpu.R`,
  `check_cube.R`
- Cube: `diag_cube3_q_solve.R` (beam search over a Q-model)
- Benchmarks: `bench_cpu_gpu.R`, `bench_resident_parts.R`,
  `bench_capture_replay.R`, `bench_ppo_recurrent_update.R`,
  `bench_ppo_recurrent_record.R`, `bench_cube_walks.R`
- MAME: `mame_make_snapshot.R`, `mame_ram_search.R`,
  `mame_ram_fighters.R`, `mame_env_smoke.R`

## Vignettes

- *Getting started*: environments and spaces, PPO on Pendulum, evaluation, saving and loading models, tic-tac-toe self-play
- *Training loop*: evaluation during training, CSV logs, checkpoints and resume, the GPU, writing your own algorithm

```r
browseVignettes("rltoolsR")
```

## Appendix: Mortal Kombat 3 (Genesis) RAM map

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

## License

MIT © Yuri Baramykov.

The package uses [RLtools](https://github.com/rl-tools/rl-tools) (MIT, Copyright (c) 2023 Jonas Eschmann); the continuous-control environments in `src/env_control.cpp` are ported from it (the acrobot from Gymnasium).

It bundles the [Box2D](https://box2d.org) physics engine by Erin Catto (v3, [github.com/erincatto/box2d](https://github.com/erincatto/box2d), MIT, Copyright (c) 2022 Erin Catto) in `src/box2d`, used by the Box2D environments (`lunar_lander`).

The classic_control, toy_text and box2d environments follow [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) (MIT, Copyright (c) 2016 OpenAI, (c) Farama Foundation). See `inst/COPYRIGHTS`.

## Author

Yuri Baramykov

## References

- Eschmann J., Albani D., Loianno G. (2024). *RLtools: A Fast, Portable Deep Reinforcement Learning Library for Continuous Control*. JMLR. <https://arxiv.org/abs/2306.03530>
