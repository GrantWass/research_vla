# GPU box setup: fresh Linux to every policy running in Isaac Sim

End-to-end bring-up of a Linux GPU machine for this repo: driver, simulator,
all four policies, and a smoke test that runs each one through Isaac Sim.
Every command here was run on the reference box below (2026-09-22). Where the
upstream installers break, the fix is already in `scripts/setup_policy.sh`,
`scripts/lowvram.sh`, or `patches/`. This guide explains what each step does
and what to do when one fails.

**Reference box:** Ubuntu 24.04.5 LTS (kernel 7.0 HWE) · RTX 4070 Ti SUPER 16 GB ·
Ryzen 9 7950X · 32 GB RAM · NVIDIA driver 580.178.04 · 1 TB NVMe.

## Verified status

`make smoke-all` result on the reference box (task `stack_bowls`, 1 episode, low-VRAM profile on):

| Policy | Checkpoint | Result | Peak VRAM (policy + sim) | Wall time |
|---|---|---|---:|---:|
| `demo` | zero-action stub | PASS (wiring only, success 0) | 5.7 GB | ~2.5 min |
| `openvla` | official `RoboDojo-sim-arx_x5-joint-1`, 4-bit | PASS (arm grasps and moves a bowl, success 0) | 14.9 GB | ~2.5 min |
| `pi05` | official `RoboDojo-sim-arx_x5-joint-0` (params only) | PASS (**success 1.0**, bowls stacked) | 14.4 GB | ~2 min |
| `turbovla` | released **RoboTwin** ckpt (`--robotwin-smoke`) | PASS (both arms move and push bowls; wiring only) | 7.5 GB | ~2 min |

These single episodes prove that each model is configured correctly for Isaac Sim.
They are **not** benchmark numbers: the low-VRAM profile degrades rendering, OpenVLA
runs 4-bit, and the TurboVLA ckpt was trained on a different robot.

## Model comparison (same robot, same task, same renderer)

`stack_bowls`, `env_cfg=arx_x5`, joint actions, seed 0, **20 episodes** each,
run sequentially on the reference box. Two confounds found in the first pass
were removed before these numbers were taken (see below), so earlier 5-episode
results in this file's history are superseded.

| Policy | Checkpoint | Trained on | Success | Score |
|---|---|---|---:|---:|
| pi0.5 (3B) | official `RoboDojo-sim-arx_x5-joint-0`, bf16 | RoboDojo, all tasks | **10/20 (50%)** | 56.8 |
| OpenVLA-OFT (7B) | official `RoboDojo-sim-arx_x5-joint-1`, 4-bit | RoboDojo, all tasks | 0/20 | 0.0 |
| TurboVLA (0.2B) | released RoboTwin ckpt | RoboTwin, different robot | 0/20 | 0.0 |
| TurboVLA (0.2B) | ours, `steps_2000_ema` | **`stack_bowls` only** | 0/20 | 0.0 |

Read the last row as a **specialist**, not a peer: it is fine-tuned on this one
task, while the first three are **generalist** checkpoints evaluated on one of
the many tasks they cover. It answers "can this 0.2B model learn this task
here", not "is it better than pi0.5".

### Why the fine-tuned TurboVLA also scores zero: undertrained, not miswired

The fine-tune ran 2,000 steps at global batch 16, so it saw 32,000 samples
against a 44,774-frame dataset -- **0.71 epochs, never one full pass**, and only
1,000 steps past the 1,000-step warmup. The recipe's own default is 55,000 steps
at batch 48 (2.64 M samples), so this is **~1.2% of the intended training**. The
run was also cut short by a host-RAM OOM at a checkpoint save.

`scripts/diag_turbovla_actions.py` confirms that reading rather than assuming
it, by scoring predicted vs ground-truth actions offline on the task's own data:

| Variant | MAE vs ground truth |
|---|---:|
| real cameras | 0.339 |
| cameras blacked out | 0.434 |
| **hold still (do nothing)** | **0.212** |

The model is *worse than doing nothing*, so the weights have not learned the
task. But it does respond to the cameras (0.339 < 0.434), so the observation
path is wired correctly -- this is a training-budget result, not a deployment
bug. Compare OpenVLA-OFT, which beats its own hold-still baseline
(0.079 vs 0.128) and still scores 0 in the sim.

Run this check before reading anything into a 0/N sim result.

### Confounds removed before this run

Both were in `patches/robodojo_lowvram_sim.patch` and both made earlier numbers
meaningless:

1. **`antialiasing_mode: "Off"`** left every camera frame heavily speckled. That
   is a distribution shift for a vision policy, not a cosmetic setting, and
   every policy was seeing it. DLAA restored, at +0.37 GB VRAM.
2. **pi0.5 was running 10 parallel sims.** Its adapter sets `eval_batch: true`,
   which the shared `num_envs: 10` honored, while the other adapters ran a
   single env -- unequal compute, and the shrunk PhysX buffers were sized for
   1 env regardless. `num_envs` is now pinned to 1.

### Why OpenVLA-OFT scores zero

This matches the RoboDojo paper, which reports **0.21 score / 0.02% success**
for OpenVLA-OFT on the sim benchmark versus **11.41 / 6.91%** for pi0.5 (best
policies cluster under 15%; human teleop is 76%). It is not a setup fault, and
we verified that separately rather than assuming it:

- **Input mapping verified statically.** The checkpoint's own
  `dataset_statistics.json` expects a 14-D proprio vector whose grippers sit at
  indices 6 and 13 -- exactly RoboDojo's packed
  `[arm_0(6), ee_0(1), arm_1(6), ee_1(1)]`. Element-by-element the dataset we
  feed matches that distribution, so there is no arm swap or permutation.
  Images go in as ALOHA expects: `cam_high` primary, then left and right wrist.
- **The cameras barely matter to this checkpoint.** `scripts/diag_openvla_obs.py`
  scores predicted vs ground-truth actions under observation variants
  (5 episodes, 10 windows, 25-step chunks):

  | Variant | MAE (4-bit) | MAE (8-bit) |
  |---|---:|---:|
  | as-wired | 0.0913 | 0.0906 |
  | wrists swapped | 0.0856 | 0.0850 |
  | head only | 0.0861 | 0.0856 |
  | all cameras black | 0.0882 | 0.0875 |
  | hold still (baseline) | 0.1349 | 0.1349 |

  It beats hold-still, so it predicts real motion -- but blacking out every
  camera or swapping the wrists does **not** make it worse. The policy is
  largely vision-independent, predicting from proprioception and the
  instruction. That is enough for a low action MAE and not nearly enough to
  locate a bowl, which is why it scores 0 while pi0.5 reaches 50% through the
  *same* camera pipeline.
- **Not a quantization artifact:** 4-bit and 8-bit agree to ~0.001 MAE. bf16
  needs ~15.4 GB and does not fit alongside the display on a 16 GB card, so the
  unquantized control is still open.
- Note the limit of this test: because the checkpoint is camera-insensitive,
  MAE cannot detect a camera mis-mapping. The proprio check above is what
  establishes the input mapping, not the MAE column.
- The same diagnostic caught a real bug: in **8-bit** the vision
  backbone was being quantized and came out effectively blind -- blacking out
  every camera moved the actions by at most 0.02. Fixed by extending
  `llm_int8_skip_modules` to the 8-bit path (commit db9a2f2).

## 1. OS and NVIDIA driver

Use **driver 580**. Isaac Sim 5.1 segfaults in its RTX renderer
(`librtx.scenedb.plugin.so`) right after "app ready" on driver 595, which
Ubuntu's installer picks by default.

```bash
sudo apt update
sudo apt install -y nvidia-driver-580 linux-modules-nvidia-580-generic-hwe-24.04
sudo apt purge -y 'nvidia-driver-595*' 'linux-modules-nvidia-595*' || true
sudo reboot
nvidia-smi                       # Driver Version: 580.x
```

- The `linux-modules-nvidia-*` packages are Canonical-signed, so Secure Boot
  can stay on and no MOK re-enrollment is needed. With DKMS drivers you get a
  blue MOK screen on the next boot: choose Enroll MOK and enter the password
  you set.
- Don't accept a newer NVIDIA driver from Software Updater later.
- Ubuntu 22.04 (RoboDojo's documented target) should also work; 24.04 is what
  was verified.

System packages for the simulator:

```bash
sudo apt install -y build-essential cmake ffmpeg git git-lfs \
  libvulkan1 mesa-vulkan-drivers vulkan-tools libglu1-mesa
git lfs install
vulkaninfo --summary | grep deviceName   # must list the NVIDIA GPU
```

## 2. Repo and pinned upstream checkouts

```bash
git clone https://github.com/GrantWass/research_vla.git ~/code/research_vla
cd ~/code/research_vla
bash setup.sh        # openvla/, turbovla/, RoboDojo/ + XPolicyLab at pinned commits
make check && make test
```

## 3. RoboDojo simulator (Isaac Sim 5.1 + Isaac Lab + cuRobo)

Don't run `bash RoboDojo/scripts/install.sh -i` as-is. Its `submodules` step
runs `git submodule update --remote`, which moves XPolicyLab off the pin
`setup.sh` set. Instead, pin the other submodules, build the env by hand, then
resume the installer from `isaacsim`:

```bash
cd RoboDojo
git submodule update --init --depth 50 third_party/IsaacLab third_party/curobo

# conda + env + base deps (what install.sh's `conda` and `base_deps` steps do)
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
bash /tmp/mc.sh -b -p ~/miniconda3 && ~/miniconda3/bin/conda init bash && exec bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n RoboDojo python=3.11 -y && conda activate RoboDojo
export PIP_USER=0 PYTHONNOUSERSITE=1
pip install -r scripts/requirements.txt
pip install opencv-python-headless==4.11.0.86 pillow matplotlib scipy==1.15.3 scikit-learn
pip install numpy==1.26.0

OMNI_KIT_ACCEPT_EULA=YES bash scripts/install.sh --from isaacsim   # ~15 GB download
```

The `ERROR: pip's dependency resolver ...` lines during install are expected.
RoboDojo pins some packages (e.g. `starlette`) below what Isaac Lab declares.

Assets (~39 GB) and the embodiment config paths:

```bash
bash scripts/init_assets.sh
python utils/update_embodiment_config_path.py
bash scripts/robodojo.sh doctor --skip-policy    # expect pass=14 fail=0
cd ..
```

## 4. Policies

One command per policy. Each is idempotent, so re-run it after a failure.

```bash
bash scripts/setup_policy.sh demo
bash scripts/setup_policy.sh openvla            # env openvla_oft + ckpt (~19 GB)
bash scripts/setup_policy.sh pi05               # uv env + seed-0 params (~12 GB)
bash scripts/setup_policy.sh turbovla --robotwin-smoke
```

**TurboVLA needs gated Hugging Face access first.** It uses DINOv3 ViT-L:

1. Accept the license at https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m.
2. Create a read token at https://huggingface.co/settings/tokens.
3. Save it without echoing it to the terminal:
   `read -rsp "HF token: " t; mkdir -p ~/.cache/huggingface; printf %s "$t" > ~/.cache/huggingface/token; chmod 600 ~/.cache/huggingface/token`

What each target fixes compared with the upstream `install.sh`:

| Policy | Upstream problem | Fix |
|---|---|---|
| `openvla` | `flash-attn==2.5.5` build dies (`CUDA_HOME` not set) | Install the prebuilt cu122/torch2.2 wheel |
| `openvla` | Resolves numpy 2, opencv 5, protobuf 5, accelerate 1.x (import errors, broken 8/4-bit loading) | Pin `numpy==1.26.4 opencv-python-headless==4.11.0.86 protobuf==3.20.3 tensorflow-metadata==1.14.0 wandb==0.17.9 accelerate==0.30.1 bitsandbytes==0.43.3` |
| `pi05` | Server launcher reads `deploy.yml` with the **conda base** python | `conda install -n base pyyaml` |
| `pi05` | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.3` hardcoded (sized for 80 GB GPUs) | `patches/xpolicylab_pi05_mem_fraction.patch` makes it overridable |
| `pi05` | Official ckpt is ~45 GB per seed | Pull only `params/` + `assets/` (~12 GB); `train_state/` holds optimizer state |
| `turbovla` | The released ckpts predate upstream's module rename; RoboTwin is 3-view, dual-arm, `learned_patch` | The adapter remaps legacy keys and takes `num_views`, dual-arm layout and `action_layout` from its config |

## 5. 16 GB GPUs: low-VRAM profile

Isaac Sim needs ~7.7 GB with default settings. That's too much next to a
3–7B policy on 16 GB. With 24 GB or more, skip this step.

```bash
bash scripts/lowvram.sh apply     # status | revert
```

- **`patches/robodojo_lowvram_sim.patch`** changes rendering to performance mode
  (no DLAA, reflections, GI, translucency or denoiser; low texture budget) and
  sizes the PhysX GPU buffers for the 1-env eval. Isaac Sim drops from 7.7 to 5.7 GB.
- **`patches/xpolicylab_openvla_oft_lowvram.patch`** loads OpenVLA 4-bit
  (LLM only; vision backbone and projector stay bf16), plus upstream loader bugs
  that make quantized FiLM checkpoints fail. Quantizing the vision backbone is
  fatal in 4-bit and *silent* in 8-bit: the bf16 FiLM weights land in quantized
  layers and the policy goes effectively blind (verify with
  `scripts/diag_openvla_obs.py`). Measured policy memory: bf16 15.2 GB
  (OOM at load), 8-bit 11.7 GB (sim OOM), 4-bit 9.2 GB (fits).
- **`.lowvram.env`** (sourced by `scripts/run_eval.sh`) sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45` for pi0.5. At 0.55, Isaac Sim hit
  Vulkan OOM on textures.

The profile turns off reflections, global illumination and translucency and caps
the texture budget, but **keeps antialiasing (DLAA) on**: with it off the frames
are heavily speckled, which is a distribution shift for a vision policy rather
than a cosmetic change. It also pins `num_envs: 1` so batched adapters
(pi0.5 sets `eval_batch: true`) cannot quietly spawn 10 parallel sims.

4-bit still changes OpenVLA's outputs, so report headline numbers from a 24 GB+
GPU, or split the policy server onto a bigger machine
(`robodojo.sh server --bind-host 0.0.0.0` / `client --policy-host`).

## 6. Verify

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate RoboDojo
export OMNI_KIT_ACCEPT_EULA=YES PATH=$HOME/.local/bin:$PATH
make smoke-all              # demo, openvla, pi05, turbovla through Isaac Sim
```

- Outputs: `RoboDojo/eval_result/RoboDojo/<task>/<policy>/arx_x5/<seed>_.../`
  holds `_result.json` plus one mp4 per camera. Smoke summaries go to
  `RoboDojo/smoke_results/`.
- Check the videos, not only PASS/FAIL. A PASS means the episode ran; the head
  camera mp4 shows whether the arms moved.
- To check a policy without the simulator (model load, server, observation and
  action shapes only), run its adapter's `eval.sh` with `EVAL_ENV_TYPE=debug`
  and the `RoboDojo` env active.
- GPU memory: `nvidia-smi --query-gpu=memory.used --format=csv -lms 1000` in a
  second shell.

## 7. Fine-tuning TurboVLA on one task

`templates/turbovla_finetune/` fine-tunes the 0.2B TurboVLA on a single RoboDojo
task, so a small model can be compared as a specialist against the generalist
checkpoints. The run on the reference box:

```bash
# 1. carve the task out of the combined download (~3 GB, not 120 GB)
python templates/turbovla_finetune/make_task_dataset.py \
  --source RoboDojo/.cache/robodojo_assets_repo/data/RoboDojo_lerobot_v30_video \
  --out data/robodojo_tasks_joint --task stack_bowls
# 2. normalization stats
python templates/turbovla_finetune/compute_stats.py \
  --data data/robodojo_tasks_joint/stack_bowls --out <run>/robodojo_stats.json
# 3. train (init from the released RoboTwin ckpt, not GroundingDINO)
TURBOVLA_INIT_FULL=1 TURBOVLA_INIT_CKPT=<released_turbovla.pth> \
  bash templates/turbovla_finetune/train.sh
# 4. point the adapter at the result and evaluate
bash scripts/install_turbovla_deploy.sh --run <run> --step 2000
bash scripts/run_eval.sh --policy turbovla --task stack_bowls --eval-num 20 --seed 0
```

Settings that mattered on a 16 GB / 32 GB box: batch 4 x grad-accum 4 (global
16; batch 8 OOMs the GPU), ~1.25 steps/s, DeepSpeed ZeRO-2, EMA on. Init from
the released RoboTwin checkpoint loaded 878/878 tensors.

Apply `patches/turbovla_full_ckpt_init.patch` and
`patches/turbovla_lerobot_video_index.patch` to the `turbovla` checkout first:

- **full-checkpoint init** lets `TURBOVLA_INIT_FULL=1` start from a trained VLA
  instead of GroundingDINO, remapping the released checkpoint's legacy key names
  and refusing to proceed on under 80% tensor match (a silent mismatch here
  looks like slow convergence, not an error).
- **video index** -- the GR00T LeRobot loader built video paths from the *data*
  file index, so any dataset whose videos are not numbered like its parquet
  shards silently read the wrong episode's frames.

Known rough edge: the trainer was killed by the kernel OOM killer at step 2000
of 4000 while saving a checkpoint (32 GB RAM, checkpoint ~1.7 GB x2 plus the
ZeRO state). The step-1000 and step-2000 checkpoints are intact and usable;
budget RAM headroom or save less often for a longer run.

## 8. Offline checks without the simulator

Per-task demo data (for diagnostics and TurboVLA training) can be carved out of
RoboDojo's combined LeRobot download instead of pulling all 120 GB:

```bash
# meta + the data/video files one task needs (stack_bowls: 100 episodes, ~3 GB)
python templates/turbovla_finetune/make_task_dataset.py \
  --source RoboDojo/.cache/robodojo_assets_repo/data/RoboDojo_lerobot_v30_video \
  --out data/robodojo_tasks_joint --task stack_bowls
python scripts/diag_openvla_obs.py 3 4bit     # predicted vs ground-truth actions
```

Use `lerobot_v3.0` (14-D joint state/action, what `arx_x5` joint eval and the
TurboVLA recipe expect). `lerobot_v3.0_ee` is 16-D end-effector poses
(position + quaternion + gripper per arm), a different convention.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Isaac Sim segfault in `librtx.scenedb.plugin.so` right after "app ready" | Driver 595 | Driver 580 (§1) |
| `EnvironmentNameNotFound: openvla-oft` | Old registry name | Env is `openvla_oft` (`policies/openvla.conf`) |
| `setup_env_client.sh: python: command not found` | Client launcher calls bare `python`, activates no env | Fixed in `scripts/run_eval.sh` (activates `$ROBODOJO_ENV`); otherwise run evals with the `RoboDojo` env active |
| `conda: command not found` from the policy server, over ssh/nohup | Non-interactive shell has no conda on PATH | Fixed in `scripts/run_eval.sh` (sources conda) |
| `Unable to bootstrap inner kit kernel: EOF when reading a line` | Isaac Sim's EULA prompt with no tty | `export OMNI_KIT_ACCEPT_EULA=YES` (`run_eval.sh` now does) |
| `CUDA out of memory` (policy) or `VkResult: ERROR_OUT_OF_DEVICE_MEMORY` (sim) | Policy + sim > VRAM | `bash scripts/lowvram.sh apply` (§5) |
| Sim hangs after an OOM, zenity "not responding" popups | Isaac Sim can't recover from Vulkan OOM | Kill the run (`pkill -f smoke_all_tasks.sh`, then the python PIDs from `nvidia-smi`), fix memory, rerun |
| pi0.5 `No module named 'yaml'` | Base python lacks PyYAML | `conda install -n base pyyaml` |
| pi0.5 `RESOURCE_EXHAUSTED` while restoring params | JAX memory fraction too small | Pi_05 patch + `XLA_PYTHON_CLIENT_MEM_FRACTION` (0.45 on 16 GB) |
| TurboVLA `401` / gated repo | DINOv3 license not accepted or no token | §4 token steps |
| TurboVLA `Weights only load failed` / `unexpected keys` | Old adapter | Pull latest; the adapter reads safetensors and remaps legacy keys |
| `uv` download timeout (`fonttools`) | Slow mirror | Re-run; `setup_policy.sh` sets `UV_HTTP_TIMEOUT=120` |
| Eval silently uses the released RoboTwin weights after you pointed it at your own | `install_adapter.sh --force` / `setup_policy.sh --robotwin-smoke` reinstall the stock `deploy.yml` | Re-run `bash scripts/install_turbovla_deploy.sh --run <run> --step N`; check `checkpoint_path` and `num_views: 3` |
| TurboVLA ckpt loads but actions are nonsense | RoboDojo-trained ckpts carry a `model.` key prefix and pack actions `[arm_0, ee_0, arm_1, ee_1]` | Pull latest adapter (strips the prefix, `action_layout: packed`) |
| Trainer killed at a checkpoint save, no traceback | Host RAM OOM killer | Free RAM or save less often; `dmesg -T | grep -i oom` confirms |
