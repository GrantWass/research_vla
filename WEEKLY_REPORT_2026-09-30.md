# Weekly Report — Multi-Robot Serving and Fine-Tuning

**Date:** 2026-09-30 · **Author:** Grant Wasserman

This week was about serving many robots from one pi0.5 policy on one 16 GB GPU
(RTX 4070 Ti SUPER). We built batched inference, server-side batching for
robots on separate connections, and a measurement stack with statistics that
can say when a result is real. Short version: **batching is correct and makes
serving up to ~30% cheaper, but in RoboDojo the brain is ~2% busy, so it does
not change wall-clock or task success.** Section 4 is a step-by-step
walkthrough for fine-tuning, using our TurboVLA run as the worked example.

Full log and every raw number: `MULTI_ROBOT_SERVING.md`.

---

## 1. What we built

| Piece | What it does | Where |
|---|---|---|
| Batched forward (S1) | All robots in one eval process answered in one model call instead of N | `patches/xpolicylab_pi05_batched_infer.patch` (`PI05_BATCHED_INFER=1`) |
| Atomic observe+answer (S2) | Fixes a real cross-talk bug: two clients could get answers computed from each other's cameras | same patch (`PI05_ATOMIC=1`) |
| Server-side batching (S3, new) | Robots on **separate connections** share one model call | `patches/xpolicylab_server_batch_window.patch` (`XPL_SERVER_BATCH=1`) |
| Compile warmup + VRAM auto-cap (new) | Compiles each batch size at startup instead of stalling robots mid-run; caps batches at the largest size that fits | `PI05_WARMUP_BATCHES=1,...,7` |
| Measurement stack | Fleet simulator, live replay client, sim and live battery drivers, comparison tables with confidence intervals and significance tests | `scripts/serve_bench/`, 106 tests (`make test`) |

All of it is off by default. With no flags set, serving behaves exactly like upstream.

## 2. Multi-robot serving: results

### 2.1 Is batching correct?

Yes. We fixed the sampler's start noise and compared each robot's answer from a
mixed batch against its answer when served alone, for the same observation.

| Comparison | Max difference in actions |
|---|---:|
| Same robot, solo call repeated | 0.000 |
| Same robot, batched vs solo | 0.012–0.044 |
| Same robot, noise seed 7 vs 8 (equally valid answers) | 0.28–0.76 |
| Different robots | 14.5 |

Batching moves an answer by **2–16% of what an ordinary resample moves it**.
The cause is bf16 GPU kernels summing in a different order for each batch
shape. That is far too small to change behavior.

Caveat: these were synthetic observations (random pixels), not sim frames.

### 2.2 What does batching buy?

The steady-state cost of one forward, by number of robots served:

| Robots per call | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| ms per call | 124 | 211 | 291 | 366 | 430 | 513 |
| ms per robot | 124 | 106 | 97 | 92 | 86 | 86 |

Each call costs about **47 ms of fixed overhead plus 77 ms per robot**. The GPU
is already compute-bound with a single robot, so batching only amortizes the
fixed part. The best possible saving on this card is ~38% per robot.

### 2.3 Separate clients: does server-side batching help?

We ran 1–8 independent clients against a live server with no simulator. Each
client was a closed loop: ask, then "execute" for a think time. Results at
saturation (think time 0):

| Clients | FIFO (upstream) p50 | Continuous batching p50 | FIFO req/s | Batched req/s |
|---:|---:|---:|---:|---:|
| 1 | 122 ms | 120 ms | 8.2 | 8.3 |
| 2 | 236 | 232 | 8.4 | 8.6 |
| 4 | 473 | 388 | 8.4 | 10.3 |
| 6 | 710 | 518 | 8.4 | 11.5 |
| 8 | 947 | 663 | 8.4 | **12.0** |

- **FIFO hits a hard ceiling at 8.4 requests/s**, and latency grows linearly
  with fleet size. Continuous batching raises the ceiling by 43% and cuts p50
  latency by 30% at 8 clients.
- **Two clients never batch on their own.** In a closed loop they settle into
  alternating turns, so they never arrive together. A 10 ms batching window
  breaks the alternation (2 clients: 216 ms vs 232 ms; 4 clients: 363 vs 388).
  But the window adds ~11 ms whenever load is light, and it never wins once
  robots spend ≥0.5 s executing each chunk. **Recommended default: batching
  on, window 0.**
- **At realistic load (2 s per chunk), every mode ties at ~123 ms.** Batching
  is insurance for when the brain becomes the bottleneck; it is not a speedup
  today.

### 2.4 Bugs the new tests caught

1. **Batch-of-8 runs out of GPU memory, and the fallback hid it.** Under the
   low-VRAM profile (7.2 GB for the model), every 8-robot batch failed. It
   quietly re-ran as 8 sequential calls, and latency rose to **11 seconds**.
   The warmup now probes memory and caps batches at 7. The rerun gave 678 ms
   with zero fallbacks. Comparison tables now count fallbacks.
2. **Mid-run compile stalls.** The first time any new batch size appears, JAX
   compiles it, which takes 9–13 s. Every robot waiting on that call stalls,
   and one action chunk lasts only ~2 s. Warmup moves the compiles to startup.
3. **Comparison script read from the wrong place.** The documented regenerate
   command read timing logs from `/tmp`, so it only worked on a machine that
   happened to have them there.
4. **"Identical" test observations weren't identical.** Pixels were seeded
   with Python's `hash()`, which is randomized for every process.

### 2.5 Can our evals tell strategies apart? Mostly no, and now we can say so

Every comparison table now carries a 95% interval, a Fisher test, and a paired
McNemar test. Runs that share a seed replay the same scenes, so they can be
compared scene by scene. Applied to last week's 12-episode runs:

| Comparison (stack_bowls) | Success | Baseline | Fisher p | Scenes only baseline won / only batched won |
|---|---:|---:|---:|---:|
| S1 vs S0, N=2 | 5/12 | 8/12 | 0.41 | 5 / 2 |
| S2 vs S0, N=2 | 7/12 | 8/12 | 1.00 | 2 / 1 |
| S1 vs S0, N=3 | 8/12 | 8/12 | 1.00 | 1 / 1 |
| S2 vs S0, N=3 | 6/12 | 8/12 | 0.68 | 2 / 0 |

Twelve episodes per arm can only detect a gap of **~57 points**. The 8/12 vs
5/12 that looked like a regression is noise (p=0.41). Pooled across the four
pairs, the baseline won 10 disputed scenes to batched's 4 (p=0.18). That was
suggestive enough to test properly, which is the next section.

### 2.6 Large paired run: does batching change success?

Six robots on one server, `stack_bowls`. Sequential (S0) was compared with
atomic+batched+warmup (S2W) on the same 75 scenes: 3 layout seeds x 25
episodes, each scene run under both strategies. That is 150 episodes, about
1.6 h of GPU time.

| Seed | S0 sequential | S2W batched | Scenes only S0 won / only S2W won | McNemar p |
|---|---:|---:|---:|---:|
| 0 | 20/25 | 20/25 | 3 / 3 | 1.00 |
| 1 | 18/25 | 12/25 | 9 / 3 | 0.15 |
| 2 | 10/25 | 14/25 | 3 / 7 | 0.34 |
| **Pooled** | **48/75 (64%)** | **46/75 (61%)** | **15 / 13** | **0.85** |

**Batching does not change task success.** Seed 1 leans one way and seed 2
leans the other, and pooled they cancel. Each seed on its own would have
"shown" a 16–24 point effect.

Two side findings matter for every future eval:

- **Scene difficulty swamps strategy.** The same policy scores 80% on seed 0
  layouts and 40–56% on seed 2. Compare strategies only on matched seeds.
- **pi0.5 is very stochastic per scene.** 28 of 75 scenes (37%) flipped
  between success and failure from one run to the next under near-identical
  serving. That variance is the floor any serving claim has to clear.

What batching did change: **76% fewer model calls** (S0 905, S2W 233), 31%
less GPU time per robot chunk (124 ms down to ~90 ms), no compile stalls
(warmup), no fallbacks, the same peak VRAM (14.2 GB), and the same wall
clock (942–1114 s per arm). It saves cost but has no effect on success.

Bug caught here: RoboDojo runs at most **25 episodes per layout seed** but
writes the *requested* count (54) into `_result.json`. The tables now count
the per-episode rows instead.

## 3. Corrections to last week's report (2026-09-23)

Checked against `~/ft_run.log` and `templates/turbovla_finetune/train.sh`:

- The TurboVLA warmup was **200 steps**, not 1,000, so the run went 1,800
  steps past warmup, not 1,000.
- The recipe's default effective batch is **192** (4 GPUs x 48), not 48.
  Ours was 16, which is **12x smaller** than the recipe, not 3x.

Neither changes the conclusion (undertrained, not miswired). Both change how
far off the recipe we were.

---

## 4. Walkthrough: fine-tuning a VLA for a RoboDojo task

This is the route we took with TurboVLA (0.2B) on `stack_bowls`, written as a
procedure to repeat for another task or another model. Each step lists the
command, what it should produce, and the failure we actually hit there. The
same seven stages apply to OpenVLA (LoRA) and pi0.5; section 4.8 says what
changes.

### 4.1 Decide what you are training, and the bar it has to clear

Before touching data, write down two numbers:

- **The baseline to beat.** For `stack_bowls`, pi0.5 (generalist, official
  checkpoint) scores 10/20. A specialist that cannot reach that is not worth
  serving.
- **The "do nothing" floor.** Holding the arm still scores MAE 0.212 against
  the demonstrations. Any checkpoint above that floor has not learned the task,
  however low its training loss looks. This floor is the cheapest early-warning
  signal we have (section 4.6).

Also decide the action space. RoboDojo ships two datasets: `lerobot_v3.0`
(14-D joint targets) and `lerobot_v3.0_ee` (16-D xyz + quaternion + gripper).
Our eval runs `action_type=joint`, so training must use the joint set.

### 4.2 Carve out one task's data (117 GB -> 3.1 GB)

The RoboDojo dataset is one combined LeRobot v3.0 repo: 3,500 episodes, 35
tasks, 1.86 M frames, 117 GB of LFS objects. You only need one task's files.

```bash
python templates/turbovla_finetune/make_task_dataset.py \
  --source RoboDojo/.cache/robodojo_assets_repo/data/RoboDojo_lerobot_v30_video \
  --out data/robodojo_tasks_joint --task stack_bowls
```

What it does: symlinks only the parquet and video files that task's episodes
reference, rewrites the metadata, and writes the GR00T `modality.json` the
trainer needs. Result for `stack_bowls`: **100 episodes, 44,774 frames,
3.1 GB**.

Check: `data/robodojo_tasks_joint/stack_bowls/meta/info.json` exists, state
and action are 14-D, three cameras (`cam_high`, `cam_left_wrist`,
`cam_right_wrist`).

Pitfall we hit: the `_ee` dataset looks identical at a glance but is 16-D.
The script now refuses it instead of mislabeling it as joints.

### 4.3 Normalization statistics

```bash
python templates/turbovla_finetune/compute_stats.py ...   # writes robodojo_stats.json
```

Proprio mean/std and action min/max per dimension, pyarrow only. These stats
ship with the checkpoint: the deploy config must point at the **same** file,
or every action is de-normalized wrongly at eval time.

### 4.4 Build the training environment (once per box)

```bash
bash ~/mk_train_env.sh     # conda env turbovla-train: py3.10, torch 2.6 cu124,
                           # pip install -e "./turbovla[robotwin]", flash-attn 2.7.4
bash ~/fix_train_env.sh    # DeepSpeed needs nvcc: conda cuda-nvcc 12.4, CUDA_HOME=$CONDA_PREFIX
bash scripts/install_turbovla_training.sh   # registers the robodojo_arx_x5 data mix
```

Keep `PIP_USER=0 PYTHONNOUSERSITE=1` set: a stray `~/.local` torch silently
shadows the env's build.

### 4.5 Smoke-train, then train

Smoke first (20 steps, batch 4) so the data path, init, and checkpoint save
are proven before a long run: `bash ~/ft_smoke.sh`. Then the real run:

```bash
conda activate turbovla-train
export CUDA_HOME=$CONDA_PREFIX PIP_USER=0 PYTHONNOUSERSITE=1
SHARED=RoboDojo/XPolicyLab/policy/TurboVLA/checkpoints/shared
ROBODOJO_DATA_ROOT=$PWD/data/robodojo_tasks_joint \
BERT_MODEL_PATH=$SHARED/bert-base-uncased DINOV3_MODEL_PATH=$SHARED/dinov3-vitl16 \
TURBOVLA_INIT_CKPT=$SHARED/checkpoints/robotwin/steps_55000_ema_model.safetensors \
TURBOVLA_INIT_FULL=1 \
NUM_PROCESSES=1 PER_DEVICE_BATCH_SIZE=4 GRAD_ACCUM=4 \
MAX_TRAIN_STEPS=4000 WARMUP_STEPS=200 SAVE_INTERVAL=1000 \
RUN_ID=turbovla_robodojo_stack_bowls_ft WANDB_MODE=disabled \
bash templates/turbovla_finetune/train.sh
```

What each choice means:

| Setting | Ours | Recipe default | Why ours |
|---|---:|---:|---|
| GPUs x per-device batch x accum | 1 x 4 x 4 = **16** | 4 x 48 x 1 = **192** | batch 8 runs out of 16 GB VRAM |
| Steps | 4,000 planned, 2,000 reached | 55,000 | first-pass budget |
| Warmup | 200 | 1,000 | scaled to the shorter run |
| LR (every param group) | 5e-5 | 5e-5 | unchanged |
| EMA decay | 0.999 | 0.999 | unchanged |
| Init | released RoboTwin ckpt (`TURBOVLA_INIT_FULL=1`) | GroundingDINO backbone | start from a model that already acts |

This is **full fine-tuning** (all parameter groups), L1 action loss,
DeepSpeed ZeRO-2, action horizon 50, three 224 px views.

Checks in the first minute of the log: `global_batch=16`, `warmup=200`, and
`full-checkpoint init: loaded 878/878 tensors`. The init patch aborts below
80% tensor match: a silent partial load looks like slow learning, not an
error.

Pitfalls we hit here:

1. The GR00T LeRobot loader built video paths from the parquet file index and
   read **the wrong episode's frames** (`patches/turbovla_lerobot_video_index.patch`).
   Nothing errors; the model just learns from mismatched images.
2. The run was killed by the **host** OOM killer (32 GB RAM) while saving the
   step-2000 checkpoint (1.7 GB weights + ZeRO-2 optimizer state), after
   27.5 minutes. Unfixed. Next run: save less often, keep only EMA weights,
   or offload optimizer state.
3. `train.sh` refuses an existing `RUN_ID`; bump it per run.
4. `ROBODOJO_DATA_ROOT` must directly contain `<task>/meta/info.json`; one
   level off and the trainer finds no datasets.

### 4.6 Score checkpoints offline before spending sim time

A 20-episode sim eval costs ~32 minutes. The offline diagnostic costs
seconds and answers the question that matters first: has the model learned
anything?

```bash
python scripts/diag_turbovla_actions.py      # one checkpoint, three variants
python scripts/diag_turbovla_step_sweep.py   # every saved step
```

| Checkpoint | MAE vs demos |
|---|---:|
| step 1000, EMA | 0.348 |
| step 2000, EMA | 0.339 |
| step 2000, raw weights | 0.415 |
| step 2000, cameras blacked out | 0.434 |
| **hold still** | **0.212** |

How to read it:

- **Above the hold-still floor = not learned yet.** Do not run the sim; 0/20
  is already predicted (and is what we got).
- **Real cameras beat blacked-out cameras** (0.339 < 0.434), so the vision
  pipeline is wired; the problem is training length, not plumbing. These two
  causes look identical from a 0/20 alone.
- **EMA improves ~0.008 MAE per 1k steps while raw weights get worse**,
  normal early in training. At that rate, crossing the floor needs on the
  order of 10^4 more steps, consistent with the recipe's 55k.
- We saw 32,000 samples = **0.71 epochs**, ~1.2% of the recipe's 2.64 M.

Gate for the sim: EMA MAE clearly below 0.212.

### 4.7 Install and evaluate in RoboDojo

```bash
bash scripts/install_adapter.sh turbovla                    # adapter: 3 views, dual-arm denorm
bash scripts/install_turbovla_deploy.sh \
  --run turbovla_robodojo_stack_bowls_ft --step 2000        # renders deploy.yml (EMA by default)
bash scripts/run_eval.sh --policy turbovla --task stack_bowls --eval-num 20 --seed 0
```

Render `deploy.yml` with the script, never by hand: `install_adapter.sh
--force` overwrites hand edits. Eval under the same render settings as
every other policy (antialiasing on, `num_envs: 1`), or the comparison is
invalid (both were confounds in earlier tables).

Read the score column, not just success: RoboDojo awards partial credit, and
our 0/20 was 0.0 on **every** episode, meaning the arms never reached the
bowls.

### 4.8 The same procedure for other models

| Stage | TurboVLA (done) | OpenVLA (template ready) | pi0.5 (upstream scripts) |
|---|---|---|---|
| Data format | LeRobot v3 (carved) | RLDS; register in `prismatic/vla/datasets/rlds/oxe/{configs,transforms,mixtures}.py` | `process_data.sh RoboDojo cotrain arx_x5 joint` |
| Method | full FT, ZeRO-2 | **LoRA** rank 32 (~1.4% of params) | full FT (upstream `train.sh`) |
| Command | `templates/turbovla_finetune/train.sh` | `cd openvla && bash ../templates/openvla_finetune/finetune_lora.sh` | `RoboDojo/XPolicyLab/policy/Pi_05/train.sh` |
| Key knobs | batch 4 x accum 4 on 16 GB | `BATCH_SIZE=16 LR=5e-4 LORA_RANK=32`; >= 27 GB GPU: `BATCH_SIZE=8 GRAD_ACCUM=2` | config `pi05_aloha`, ckpt at `checkpoints/RoboDojo-cotrain-arx_x5-joint-0/` |
| Fits our 16 GB box? | yes | no (7B; needs >= 27 GB even for LoRA) | inference yes, training no |
| Offline gate | `diag_turbovla_*` | open-loop replay of demos, then offline inference vs training | same idea; not built yet |

`make finetune-help` prints these entry points.

### 4.9 Next fine-tuning run

1. Fix the host-RAM ceiling at checkpoint save (EMA-only saves, or
   optimizer offload), then train **20k steps** (~4.5 h, ~7 epochs) on the
   same data and seed.
2. Sweep every saved step offline (4.6); only checkpoints under the 0.212
   floor go to the sim.
3. Sim-eval the best one at **>= 60 episodes**: at 20 episodes a success gap
   under ~45 points is indistinguishable from noise (section 2.5).

---

## 5. Next week

1. **Fine-tune TurboVLA for 20k steps** after fixing the host-RAM crash at
   checkpoint save (4.5, 4.9). This is the highest-value item: the only
   policy that works today is the 3B generalist.
2. **Make the brain the bottleneck on purpose**, so scheduling can be
   measured end to end. Options: a faster control rate, a heavier policy, or
   many replay clients mixed with one sim. At ~2% GPU busy, no serving
   strategy can show a wall-clock win in RoboDojo.
3. **Async execution + PAINT** (robot keeps moving while the brain thinks).
   This is the change that would actually cut robot idle time, which
   batching can't.
4. Keep every success claim at **>= 50 paired episodes**; below that, report
   the interval, not the winner.

## Reproduce

```bash
make test                                                        # 106 tests, Mac-safe
python3 scripts/serve_bench/compare_runs.py --pairwise           # sim tables (2.2, 2.5)
python3 scripts/serve_bench/fleet_report.py \
  scripts/serve_bench/data/fleet_capped_2026_09_30/fleet_summary.jsonl   # 2.3
python3 scripts/serve_bench/equivalence_report.py \
  scripts/serve_bench/data/equivalence_2026_09_30/equivalence_seed{7,8}.jsonl  # 2.1
# GPU box:
bash scripts/serve_bench/live_battery.sh equivalence|fleet OUT_DIR
bash scripts/serve_bench/sim_battery.sh NAME N TASK EPISODES SEED S0|S1|S2|S2W
```
