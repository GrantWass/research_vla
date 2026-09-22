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

## Three-model comparison (same robot, same task)

`stack_bowls`, `env_cfg=arx_x5`, joint actions, seed 0, 5 episodes each,
sequential on the reference box with the low-VRAM profile:

| Policy | Checkpoint | Success | Score | Wall time |
|---|---|---:|---:|---:|
| pi0.5 | official `RoboDojo-sim-arx_x5-joint-0`, bf16 | **3/5** | 66.0 | 4.5 min |
| OpenVLA-OFT | official `RoboDojo-sim-arx_x5-joint-1`, 4-bit | 0/5 | 0.0 | 10 min |
| TurboVLA | released RoboTwin ckpt (different robot) | 0/5 | 0.0 | 8.2 min |

OpenVLA-OFT scoring zero matches the RoboDojo paper, which reports **0.21 score
/ 0.02% success** for it on the sim benchmark versus **11.41 / 6.91%** for pi0.5
(best policies cluster under 15%; human teleop is 76%). It is not a setup fault.
`scripts/diag_openvla_obs.py` confirms the observation path: in 4-bit the
predicted actions respond to the cameras (max change 0.33 when they are blacked
out) and beat a hold-still baseline (MAE 0.079 vs 0.128).

TurboVLA needs a RoboDojo-trained ckpt before it belongs in this table
(`templates/turbovla_finetune/`).

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

With the profile on, rendered frames show speckle noise (denoiser off), and
4-bit changes OpenVLA's outputs. Use it for wiring checks and development.
Report benchmark numbers from a 24 GB+ GPU, or split the policy server onto a
bigger machine (`robodojo.sh server --bind-host 0.0.0.0` / `client --policy-host`).

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

## 7. Offline checks without the simulator

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
| `setup_env_client.sh: python: command not found` | Client launcher calls bare `python` | Run evals with the `RoboDojo` env active |
| `CUDA out of memory` (policy) or `VkResult: ERROR_OUT_OF_DEVICE_MEMORY` (sim) | Policy + sim > VRAM | `bash scripts/lowvram.sh apply` (§5) |
| Sim hangs after an OOM, zenity "not responding" popups | Isaac Sim can't recover from Vulkan OOM | Kill the run (`pkill -f smoke_all_tasks.sh`, then the python PIDs from `nvidia-smi`), fix memory, rerun |
| pi0.5 `No module named 'yaml'` | Base python lacks PyYAML | `conda install -n base pyyaml` |
| pi0.5 `RESOURCE_EXHAUSTED` while restoring params | JAX memory fraction too small | Pi_05 patch + `XLA_PYTHON_CLIENT_MEM_FRACTION` (0.45 on 16 GB) |
| TurboVLA `401` / gated repo | DINOv3 license not accepted or no token | §4 token steps |
| TurboVLA `Weights only load failed` / `unexpected keys` | Old adapter | Pull latest; the adapter reads safetensors and remaps legacy keys |
| `uv` download timeout (`fonttools`) | Slow mirror | Re-run; `setup_policy.sh` sets `UV_HTTP_TIMEOUT=120` |
