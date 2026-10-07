# Simulation: VLA policies in RoboDojo (Isaac Sim)

Secondary now that the real arms are running. Kept for comparing policies and
for the serving experiments (`docs/serving.md`).

RoboDojo is the benchmark: 54 runnable tasks on Isaac Sim 5.1 + Isaac Lab, a
dual ARX X5 robot (`env_cfg=arx_x5`), and XPolicyLab adapters that serve each
policy over a websocket. This repo adds a model registry on top, so swapping
models is one flag:

```bash
bash scripts/run_eval.sh --policy pi05 --task stack_bowls --dry-run   # openvla | turbovla | pi05 | demo
```

Needs Ubuntu 22.04/24.04, an RTX GPU, NVIDIA driver **580**, 32 GB RAM. These
steps were verified on Ubuntu 24.04. The Alienware (22.04, RTX 3090 24 GB,
driver 580) hasn't run the sim yet, so `setup.sh` hasn't been run there and
`make test` skips the sim tests.

## Policies

| `--policy` | Model | Adapter | Env | Checkpoint |
|---|---|---|---|---|
| `pi05` (default) | pi0.5 3B | upstream `XPolicyLab/policy/Pi_05` | `uv` | `RoboDojo-sim-arx_x5-joint-0` (params only) |
| `openvla` | OpenVLA-OFT 7B | upstream `XPolicyLab/policy/OpenVLA_OFT` | conda `openvla_oft` | `RoboDojo-sim-arx_x5-joint-1` |
| `turbovla` | TurboVLA 0.2B | ours, `adapters/turbovla_robodojo/` | conda `turbovla-robodojo` | released RoboTwin ckpt, or one we fine-tuned |
| `demo` | zero-action stub | upstream `demo_policy` | conda `RoboDojo` | none (wiring test) |

Each policy is one file in `policies/<name>.conf` (fields in
`policies/README.md`). `scripts/run_eval.sh` reads it and calls
`RoboDojo/scripts/robodojo.sh`, the only thing that touches the sim. CLI flags
override the conf, and unknown flags pass through to `robodojo.sh`.

**Adding a model:** copy `policies/demo.conf`. If XPolicyLab already ships an
adapter, that is the whole job. Otherwise copy `adapters/turbovla_robodojo/`
(or `templates/xpolicylab_policy/`), implement `model.py`, and install it with
`scripts/install_adapter.sh <name>`. `make test` checks that every conf
dry-runs.

## Setup

**1. Driver.** Use 580. Isaac Sim segfaults in `librtx.scenedb.plugin.so` on
driver 595, which Ubuntu installs by default. Don't accept newer drivers from
Software Updater.

```bash
sudo apt install -y nvidia-driver-580 build-essential cmake ffmpeg git git-lfs \
  libvulkan1 mesa-vulkan-drivers vulkan-tools libglu1-mesa
git lfs install && vulkaninfo --summary | grep deviceName   # must list the NVIDIA GPU
```

**2. Upstream checkouts** at pinned commits (`turbovla/`, `RoboDojo/` + XPolicyLab):

```bash
bash setup.sh && make check && make test
```

**3. Simulator.** Don't run `RoboDojo/scripts/install.sh -i` as-is: its
`submodules` step moves XPolicyLab off the pin. Build the env by hand, then
resume the installer at `isaacsim`:

```bash
cd RoboDojo
git submodule update --init --depth 50 third_party/IsaacLab third_party/curobo
conda create -n RoboDojo python=3.11 -y && conda activate RoboDojo
export PIP_USER=0 PYTHONNOUSERSITE=1
pip install -r scripts/requirements.txt
pip install opencv-python-headless==4.11.0.86 pillow matplotlib scipy==1.15.3 scikit-learn numpy==1.26.0
OMNI_KIT_ACCEPT_EULA=YES bash scripts/install.sh --from isaacsim   # ~15 GB; resolver warnings are expected
bash scripts/init_assets.sh && python utils/update_embodiment_config_path.py   # ~39 GB
bash scripts/robodojo.sh doctor --skip-policy                     # expect fail=0
```

**4. Policies.** Each command is idempotent and carries the fixes the upstream installers need:

```bash
bash scripts/setup_policy.sh demo
bash scripts/setup_policy.sh pi05                       # ~12 GB
bash scripts/setup_policy.sh openvla                    # ~19 GB
bash scripts/setup_policy.sh turbovla --robotwin-smoke  # needs HF access to facebook/dinov3-vitl16-pretrain-lvd1689m
make smoke-all                                          # every policy through Isaac Sim once
```

| Fix in `setup_policy.sh` / `patches/` | Why |
|---|---|
| OpenVLA: prebuilt `flash-attn` 2.5.5 wheel, pinned numpy/opencv/protobuf/accelerate/bitsandbytes | upstream build fails without `CUDA_HOME` and resolves incompatible versions |
| pi0.5: `conda install -n base pyyaml` | the server launcher reads `deploy.yml` with base python |
| pi0.5: `xpolicylab_pi05_mem_fraction.patch` | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.3` was hardcoded for 80 GB GPUs |
| pi0.5: pull only `params/` + `assets/` | the full ckpt is ~45 GB, mostly optimizer state |
| TurboVLA: adapter remaps legacy keys, reads views and arm layout from config | released ckpts predate upstream's renames |

**Memory.** Isaac Sim takes ~7.7 GB. pi0.5 next to it needs
`XLA_PYTHON_CLIENT_MEM_FRACTION` of about 0.45 on a 24 GB card. OpenVLA-OFT 7B
in bf16 (15.2 GB) doesn't fit next to the sim on 24 GB.

## Running evals

```bash
conda activate RoboDojo && export OMNI_KIT_ACCEPT_EULA=YES
bash scripts/run_eval.sh --policy pi05 --task stack_bowls --mode smoke --fail-fast
bash scripts/run_eval.sh --policy pi05 --task stack_bowls --eval-num 20 --seed 0
bash scripts/run_eval.sh --policy pi05 --mode benchmark --gpu-ids 0   # all tasks; then: cd RoboDojo && bash scripts/robodojo.sh summarize
```

- Results land in `RoboDojo/eval_result/RoboDojo/<task>/<policy>/arx_x5/<seed>_.../`:
  `_result.json` plus one mp4 per camera. Watch the head-camera video. A
  PASS only means the episode ran.
- To check wiring without the sim, run an adapter's `eval.sh` with `EVAL_ENV_TYPE=debug`.
- Split server and client: `--mode server --bind-host 0.0.0.0` on one side,
  `--mode client --policy-host <ip>` on the other.
- For fair comparisons keep `num_envs: 1` in `RoboDojo/env_cfg/sim/sim_config.yml`
  and leave antialiasing on (`docs/findings.md`).

## Fine-tuning

In sim, pi0.5 trains through its upstream adapter:
`cd RoboDojo/XPolicyLab/policy/Pi_05 && bash process_data.sh RoboDojo cotrain arx_x5 joint && bash train.sh RoboDojo cotrain arx_x5 joint 0 0`.
The OpenVLA LoRA and TurboVLA fine-tune recipes were removed from this repo
(they're in git history). Fine-tuning for the arms is in `docs/robot.md`.

Before reading anything into a 0/N sim result, score the checkpoint offline:
compare its predicted actions with ground truth, against a "hold still"
baseline, with the cameras real and blacked out (see `docs/findings.md`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Segfault in `librtx.scenedb.plugin.so` after "app ready" | Driver 580, not 595 |
| `Unable to bootstrap inner kit kernel: EOF` | `export OMNI_KIT_ACCEPT_EULA=YES` |
| `CUDA out of memory` / `VkResult: ERROR_OUT_OF_DEVICE_MEMORY` | policy + sim exceed VRAM: lower the JAX fraction or use a smaller policy; after a Vulkan OOM the sim hangs, so kill it and rerun |
| pi0.5 `RESOURCE_EXHAUSTED` restoring params | raise `XLA_PYTHON_CLIENT_MEM_FRACTION` |
| pi0.5 `No module named 'yaml'` | `conda install -n base pyyaml` |
| TurboVLA 401 / gated repo | accept the DINOv3 license and save an HF token to `~/.cache/huggingface/token` |
| Eval runs the RoboTwin weights after you pointed it at your own | `install_adapter.sh --force` reinstalls the stock `deploy.yml`; point `checkpoint_path` at your run again |
| `uv` download timeout | rerun (`UV_HTTP_TIMEOUT=120` is set) |
