# OpenVLA + RoboDojo (Isaac Sim) — Setup & Usage

Date: 2026-09-06. Host at setup time: Apple M5 MacBook Pro, macOS 26.3.1 (arm64), Python 3.14.6, no NVIDIA GPU.

## 1. What was pulled

All repos live at the workspace root (the directory containing this file):

| Directory | Source | Commit at setup | Contents |
|---|---|---|---|
| `openvla/` | `https://github.com/openvla/openvla.git` | `c8f03f4` | OpenVLA-7B training / fine-tune / deploy code (`vla-scripts/`, `prismatic/`, `experiments/`) |
| `turbovla/` | `https://github.com/H-EmbodVis/TurboVLA.git` | `b29ab14` | TurboVLA-0.2B training / eval code (`turbovla/`, `experiments/`, `scripts/`); served in RoboDojo via `adapters/turbovla_robodojo/` (see `POLICIES_ROBODOJO.md`) |
| `RoboDojo/` | `https://github.com/RoboDojo-Benchmark/RoboDojo.git` | `ee67a14` | Sim benchmark (eval-only): `env/`, `env_cfg/`, `task/RoboDojo/`, `scripts/robodojo.sh`, `src/eval_client/` |
| `RoboDojo/XPolicyLab/` | submodule `https://github.com/XPolicyLab/XPolicyLab.git` | `432f82b` | Policy servers/adapters, incl. `policy/OpenVLA_OFT/` + `policy/demo_policy/`. Initialized with `git submodule update --init --depth 1 XPolicyLab`. |


`third_party/IsaacLab` and `third_party/curobo` submodules were deliberately **not** initialized on this Mac (Linux-only, NVIDIA-only build).

## 2. Simulator choice: RoboDojo (which *includes* Isaac Sim/Lab)

**Chosen: RoboDojo.** It is not an alternative to Isaac — it is built on **Isaac Sim 5.1 + Isaac Lab 2.3** and adds the exact layer OpenVLA needs.

| Option | What it is | Fit for OpenVLA |
|---|---|---|
| **RoboDojo (chosen)** | Unified sim-and-real benchmark: 42 sim tasks (+12 `_random` generalization variants = 54 runnable configs) + 18 real tasks, 5 capability dimensions (generalization, memory, precision, long-horizon, open), heterogeneous parallel sim, seed-controlled layouts, `summarize` leaderboard table. Policy side owned by XPolicyLab (41+ policies, incl. `OpenVLA_OFT`). | Best. Purpose-built VLA eval harness; `OpenVLA_OFT` adapter already exists; "integrate once, evaluate everywhere" (sim + RealEval). |
| Bare Isaac Sim + Isaac Lab | Generic NVIDIA sim framework (physically-based rendering, PhysX). | Worst for this goal. No VLA tasks, no eval protocol, no policy bridge — you would rebuild what RoboDojo already provides. |
| LIBERO (inside `openvla/`) | Lightweight CPU-friendly sim used in the OpenVLA paper (v2, App. E). | Good fallback **on this Mac** for OpenVLA-only testing, but narrow vs. RoboDojo. |

References: RoboDojo docs `https://robodojo-benchmark.com/doc/`, install `.../doc/usage/install-and-download/`, eval `.../doc/usage/quick-evaluation/`; OpenVLA paper `arXiv:2406.09246`; RoboDojo paper `arXiv:2607.04434`; XPolicyLab paper `arXiv:2608.09892`.

## 3. Important host limitation (read before running sim)

Full RoboDojo simulation **cannot run on this Mac**:

- Isaac Sim 5.1 is **Linux-only** (Ubuntu 22.04 x64 recommended), requires an **RTX GPU with RT cores** (A100/H100 explicitly unsupported), **32 GB+ RAM, 16 GB+ VRAM**, NVIDIA driver **570/580** (Linux 580.65.06 tested), CUDA 12.8, Vulkan.
- This machine is arm64 Apple Silicon with no NVIDIA GPU, so `scripts/install.sh` (apt/conda/Isaac Sim/Isaac Lab/CuRobo) is not runnable here.

What **was** verified on this Mac (no GPU needed):

```bash
cd RoboDojo
bash -n scripts/robodojo.sh scripts/eval_policy.sh          # syntax OK
python3 scripts/internal/task_inventory.py --format json --check  # runnable: 54, missing: 0
bash scripts/robodojo.sh doctor --skip-isaac --skip-conda --skip-policy  # 7 pass; expected FAILs: Assets/* missing, env_cfg refs (no pyyaml)
bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/OpenVLA_OFT \
  --task stack_bowls --ckpt demo --policy-env openvla-oft --dry-run  # exit 0

cd ../openvla
python3 -m py_compile vla-scripts/deploy.py vla-scripts/finetune.py \
  experiments/robot/libero/run_libero_eval.py  # OK
```

Expected `doctor` FAILs on a fresh clone: `Assets/Robots`, `Assets/Object/RoboDojo`, `Assets/Eval_Layout/RoboDojo`, `Assets/Material` (download step not run — large assets), plus `env_cfg references` (missing `yaml` module on system Python 3.14). These clear after the Linux-GPU setup below.

## 4. Full setup (Linux GPU machine, Ubuntu 22.04 x64 + RTX)

Run on the GPU box, not the Mac. Sync this folder over (`openvla/`, `RoboDojo/` incl. initialized `XPolicyLab/`), or re-clone there.

```bash
# 0. Pre-reqs
nvidia-smi  # driver 570/580, CUDA 12.8
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools git-lfs
vulkaninfo | head
git lfs install

# 1. RoboDojo native install (creates `RoboDojo` conda env:
#    Isaac Sim 5.1, Isaac Lab, CuRobo + Python deps)
cd RoboDojo
bash scripts/install.sh -i
# resume after a failure at a named step, e.g.:
# bash scripts/install.sh --from isaacsim   # steps: system conda base_deps submodules isaacsim isaaclab curobo
conda activate RoboDojo
ffmpeg -version  # required for some data workflows; install via apt if missing

# 2. Assets + embodiment paths
bash scripts/init_assets.sh
python utils/update_embodiment_config_path.py
# re-run the python line if you move the repo or re-download assets elsewhere

# 3. (Optional) datasets + checkpoints — large; pick only what you need
bash scripts/RoboDojo/download_data.sh                                   # lists formats/sizes
bash scripts/RoboDojo/download_data.sh huggingface lerobot_v3.0         # 120 GB -> data/RoboDojo_lerobot_v30_video
bash scripts/RoboDojo/download_ckpt.sh                                   # prints usage
bash scripts/RoboDojo/download_ckpt.sh huggingface Pi_0                  # -> XPolicyLab/policy/<POLICY>/checkpoints

# 4. OpenVLA policy env (example: OpenVLA-OFT adapter)
cd XPolicyLab/policy/OpenVLA_OFT
bash install.sh
conda activate openvla-oft  # env name per adapter README; see its install.sh
cd ../../..  # back to RoboDojo root

# 5. Verify
bash scripts/robodojo.sh doctor   # no --skip flags on the GPU box
bash scripts/robodojo.sh dimensions
```

Docker alternative (simulator-side container; policy server stays on host, assets mounted, not baked in):

```bash
cd RoboDojo
bash scripts/init_assets.sh && python utils/update_embodiment_config_path.py
sudo bash docker/install_docker_nvidia.sh; newgrp docker
docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 nvidia-smi
docker build -t robodojo:cuda12.8 .
bash docker/smoke_docker.sh run   # expect success_rate 0.0 from demo_policy (zero actions); checks sim+render+ws+result path
```

## 5. Usage

### 5a. OpenVLA standalone inference (minimal deps, GPU box)

```bash
pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
```

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True).to("cuda:0")
inputs = processor("In: What action should the robot take to put the corn on the plate?\nOut:", image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)  # 7-DoF
```

Serve over REST for an existing control stack: `python vla-scripts/deploy.py --help` (see `openvla/vla-scripts/deploy.py`).
Fine-tune with LoRA: `torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path openvla/openvla-7b --dataset_name bridge_orig ...` (full flags in `openvla/README.md`).

### 5b. OpenVLA on LIBERO sim (Mac-feasible fallback, GPU box recommended)

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && cd LIBERO && pip install -e .
cd ../openvla && pip install -r experiments/robot/libero/libero_requirements.txt
python experiments/robot/libero/run_libero_eval.py --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial --center_crop True   # keep center_crop True; 500 trials default (10x50)
# suites: libero_spatial | libero_object | libero_goal | libero_10 (checkpoints: ...-libero-{spatial,object,goal,10})
```

### 5c. RoboDojo eval (GPU box; `robodojo.sh` is the only entry point)

Prefer the model-agnostic wrapper (see `POLICIES_ROBODOJO.md`) — it resolves
`--policy-dir` / `--policy-env` / `--ckpt` from `policies/<name>.conf`, so
OpenVLA ↔ TurboVLA is one flag:

```bash
bash scripts/run_eval.sh --policy openvla  --task stack_bowls --dry-run
bash scripts/run_eval.sh --policy turbovla --task stack_bowls --dry-run
```

Raw `robodojo.sh` equivalents (what the wrapper forwards to):

```bash
cd RoboDojo
conda activate RoboDojo

# health + inventory (no sim)
bash scripts/robodojo.sh doctor
bash scripts/robodojo.sh tasks            # or: tasks --dimension memory
bash scripts/robodojo.sh dimensions

# dry-run first (no sim, no server) — works on Mac too
bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/OpenVLA_OFT \
  --task stack_bowls --ckpt <CKPT> --policy-env openvla-oft --eval-num 1 --dry-run

# single task, same machine (server + sim client on localhost)
bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/OpenVLA_OFT \
  --task stack_bowls --ckpt <CKPT> --policy-env openvla-oft --eval-num 1
# common opts: --env-cfg arx_x5 (default) --seed 0 --action-type ee --expert-num 100

# smoke (install/policy validation; EVAL_NUM=1 default; require exit 0 + _result.json with eval_time>=1)
bash scripts/robodojo.sh smoke --policy-dir XPolicyLab/policy/OpenVLA_OFT \
  --ckpt <CKPT> --policy-env openvla-oft --fail-fast
bash scripts/robodojo.sh smoke --policy-dir XPolicyLab/policy/OpenVLA_OFT \
  --ckpt <CKPT> --policy-env openvla-oft --only stack_bowls,push_T
bash scripts/robodojo.sh smoke --policy-dir XPolicyLab/policy/OpenVLA_OFT \
  --ckpt <CKPT> --policy-env openvla-oft --dimension memory

# full benchmark: 54 tasks x 3 seeds x native counts (non-general 50, general base/_random 25)
bash scripts/robodojo.sh benchmark --policy-dir XPolicyLab/policy/OpenVLA_OFT \
  --ckpt <CKPT> --policy-env openvla-oft --eval-num native --seed 0   # repeat seeds 1, 2
bash scripts/robodojo.sh benchmark --policy-dir XPolicyLab/policy/OpenVLA_OFT \
  --ckpt <CKPT> --policy-env openvla-oft --eval-num native --dimension generalization
# balanced multi-GPU instead of manual shards:
bash scripts/robodojo.sh benchmark --policy-dir XPolicyLab/policy/OpenVLA_OFT \
  --ckpt <CKPT> --policy-env openvla-oft --eval-num native --gpu-ids 0,1,3
bash scripts/robodojo.sh summarize   # -> eval_result/RoboDojo/_summary.md

# split machines / container boundary (server binds 0.0.0.0, client dials it)
bash scripts/robodojo.sh server --policy-dir XPolicyLab/policy/OpenVLA_OFT \
  --task stack_bowls --ckpt <CKPT> --policy-env openvla-oft --policy-port 9999 --bind-host 0.0.0.0
bash scripts/robodojo.sh client --policy-dir XPolicyLab/policy/OpenVLA_OFT \
  --task stack_bowls --policy-host <POLICY_IP> --policy-port 9999 \
  --ckpt <CKPT> --action-type ee --eval-num 1
```

Results: per-task `eval_result/RoboDojo/<task>/<policy>/<env_cfg>/<seed>_<info>/<run_id>/_result.json` (+ `episode_*.mp4`); smoke summaries in `smoke_results/<run_id>.{json,md}`.

### 5d. OpenVLA-OFT adapter directly (train/eval without robodojo.sh)

```bash
cd RoboDojo/XPolicyLab/policy/OpenVLA_OFT
bash install.sh && conda activate openvla-oft
python scripts/download_openvla.py          # base openvla-7b -> checkpoints/shared/openvla-7b
bash train.sh RoboDojo cotrain arx_x5 joint 0 0            # <bench> <ckpt> <env> <action> <seed> <gpu>
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_env> <sim_env>
# EVAL_ENV_TYPE=debug for offline wiring check (no sim); unset/sim for real sim
```

Key `deploy.yml` knobs: `base_model_path`, `use_film`, `use_l1_regression`, `use_proprio`, `num_images_in_input: 3`, `center_crop`, `lora_rank: 32`, `num_open_loop_steps: 25`, `unnorm_key`/`tfds_dataset_name` (null = auto), `port: 6000`. Full train-data recipe (ALOHA TFDS conversion) is in `XPolicyLab/policy/OpenVLA_OFT/README.md`.

## 6. Repo map (where things live)

- `openvla/vla-scripts/{deploy,finetune,train}.py` — inference server, LoRA FT, full FSDP training.
- `turbovla/turbovla/{models,evaluation}/` — TurboVLA architecture + LIBERO/RoboTwin eval adapters.
- `policies/{openvla,turbovla,demo}.conf` + `scripts/run_eval.sh` — model registry and the one-flag eval entry (`POLICIES_ROBODOJO.md`).
- `adapters/turbovla_robodojo/` — this-repo XPolicyLab adapter for TurboVLA (installed via `scripts/install_adapter.sh`).
- `openvla/experiments/robot/{libero,bridge}/` — LIBERO + WidowX eval scripts.
- `RoboDojo/scripts/robodojo.sh` — all eval commands (`doctor eval server client smoke benchmark summarize tasks dimensions`).
- `RoboDojo/task/RoboDojo/{config,tasks}/` — 54 runnable task YAML + logic; registry `task_registry.py`.
- `RoboDojo/env_cfg/` — `arx_x5.yml` (default), scene/sim/camera configs.
- `RoboDojo/XPolicyLab/policy/OpenVLA_OFT/{eval.sh,deploy.yml,train.sh,install.sh}` — OpenVLA policy wiring.
- `RoboDojo/Dockerfile`, `RoboDojo/docker/` — container sim client path.
- `tests/` — CPU-only smoke suite (stdlib unittest, no pytest/GPU/Isaac needed):
  `python3 -m unittest discover -s tests -v`. Covers workspace layout, task
  inventory (54 runnable, names match, real `run_reward`), eval dry-run wiring,
  policy contract, OpenVLA script compile, and template self-checks.
- `templates/robodojo_task/` — new-task skeleton (`my_task.yml` + `my_task.py` + checklist).
- `templates/xpolicylab_policy/my_policy/` — new-policy skeleton mirroring
  `XPolicyLab/policy/demo_policy/` (`eval.sh`, `deploy.yml`, server/client launchers,
  zero-action `model.py`, eval loop, `install.sh`).
- `templates/openvla_finetune/` — LoRA fine-tune launcher (`finetune_lora.sh`, all
  settings via env vars) + checklist (routes, dataset registration, data-collection
  rules, sanity checks). Run on the GPU box; `make finetune-help` for the shortcut.
- `Makefile` — `make test|setup|check|doctor|inventory|dry-run TASK=<t>|finetune-help|clean`.
- `setup.sh` (+ `--check`), `.gitignore`, `CONTRIBUTING.md`, `.pre-commit-config.yaml`,
  `.github/` (CI smoke workflow + PR template).

## 7. Troubleshooting

- `robodojo.sh install` on macOS: not supported — needs Ubuntu + `apt`, conda, NVIDIA. Do it on the GPU box (§4).
- `doctor` FAIL `Assets/*`: run `bash scripts/init_assets.sh` (+ `git lfs install` first), then `python utils/update_embodiment_config_path.py`.
- `doctor` FAIL `env_cfg references` / `No module named 'yaml'`: system Python lacks deps — use the `RoboDojo` conda env on the GPU box.
- Isaac Sim crash / driver errors: use tested driver **580.65.06** (not 595.x), CUDA 12.8, `vulkaninfo` must work.
- `ValueError: ... X5A.urdf is not a file` in Docker: you skipped the dual Assets mount — follow §8.6 of the install doc (mount `$PWD/Assets` at both container and absolute-host paths) plus cache mounts.
- OpenVLA near-100% `action_accuracy` when FT on BridgeData V2 with `--image_aug False`: expected (pretrained on a superset incl. Bridge V2) — not a bug.
- OpenVLA demo fails on a new robot out-of-the-box: expected — it needs ~100-demo FT on the target domain at 5–10 Hz with continuous motion (see `openvla/README.md` "VLA Performance Troubleshooting").
- `moviepy.editor` import error in Bridge WidowX eval: pin `moviepy==1.0.3` in `widowx_envs/requirements.txt` and rebuild the container.
- `robodojo.sh smoke` fails on macOS with `mapfile: command not found`: stock macOS bash is 3.2;
  `smoke_all_tasks.sh` needs bash 4+. Either `brew install bash` and re-run with the new bash,
  or validate via `eval --dry-run` on the Mac and run `smoke` on the Linux GPU box.
