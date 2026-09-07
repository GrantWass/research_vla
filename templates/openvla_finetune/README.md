# OpenVLA fine-tune template

Three routes — pick one before touching data:

| Route | When | Entry point |
|---|---|---|
| **LoRA FT (this template)** | New robot/tasks, 1 GPU, fastest path. Matches full-FT quality while training ~1.4% of params. | `finetune_lora.sh` |
| **OFT recipe** | Recommended by the OpenVLA authors (2025-03): 25–50× faster inference, higher success, multi-image, bimanual. | `https://openvla-oft.github.io/` (+ `XPolicyLab/policy/OpenVLA_OFT/`) |
| **Full FT (FSDP)** | Only if LoRA fails (e.g. target domain far from pretraining). Needs ~8×A100 node. | `openvla/vla-scripts/train.py` + Prismatic checkpoint `openvla-7b-prismatic` |

## 0. One-time setup (GPU box)

```bash
conda create -n openvla python=3.10 -y && conda activate openvla
# install torch for your CUDA first: https://pytorch.org/get-started/locally/
cd openvla && pip install -e . && pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
# HuggingFace token for gated checkpoints (NEVER commit it; root .gitignore covers .hf_token)
echo hf_... > ~/.hf_token   # or: huggingface-cli login
```

## 1. Dataset setup

- **OXE datasets**: download per [this script](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh).
  BridgeData V2: download from the [official site](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/)
  and rename to `bridge_orig/` (the OXE copy is stale; skipping the rename breaks the loader).
- **Custom dataset**: convert to RLDS ([builder instructions](https://github.com/kpertsch/rlds_dataset_builder)),
  then register it in three places in `openvla/`:
  - `prismatic/vla/datasets/rlds/oxe/configs.py` (~line 54) — observation/action spaces
  - `prismatic/vla/datasets/rlds/oxe/transforms.py` (~line 828) — standardization transform
  - dataset mixture (only for full-FT runs): `prismatic/vla/datasets/rlds/oxe/mixtures.py`

## 2. Data-collection rules (read before recording demos)

OpenVLA has no action chunking, so it is picky about demo quality:

- Control frequency **5–10 Hz** (downsample 50 Hz controllers; verify the task still solves at 5 Hz).
- **Continuous, slow motion** — no pauses/idle steps; the model gets stuck on near-zero actions.
- Cover the test distribution (varied object poses you plan to evaluate).
- Consistent strategies (same approach side, same sub-step order) — less multimodality, easier fit.
- Target **~100 demos** for a new domain.

## 3. Run

```bash
cd openvla
DATASET_NAME=bridge_orig DATA_ROOT=/path/to/datasets RUN_ROOT=/path/to/runs \
  bash ../templates/openvla_finetune/finetune_lora.sh
# small GPU (>=27 GB): BATCH_SIZE=8 GRAD_ACCUM=2
```

Notes:

- `--image_aug False` on BridgeData V2 shows ~100% action accuracy — expected
  (the base model already saw it in pretraining), not a bug.
- LoRA adapters save under `RUN_ROOT`; merge/load via HF AutoClasses.

## 4. Sanity checks (when success rate disappoints)

1. **Data pipeline**: replay demo actions open-loop on the robot — must succeed.
2. **Inference pipeline**: feed dataset images through the fine-tuned model offline —
   must reproduce training token accuracy / L1. If not, the deployment preprocessing
   (crop, normalize, `unnorm_key`) differs from training.
3. Only then blame the model/data (see §2 rules).

## 5. Evaluating the fine-tuned policy in RoboDojo

Either serve it behind an XPolicyLab adapter (copy `templates/xpolicylab_policy/my_policy/`,
point `checkpoint_path` at the run dir) or, for the OFT route, use
`RoboDojo/XPolicyLab/policy/OpenVLA_OFT/` (`train.sh`/`eval.sh`, TFDS `aloha_<run_id>`
dataset). Then: `bash scripts/robodojo.sh smoke --policy-dir ... --only <task> --dry-run`
first, runtime `smoke` on the GPU box second.
