#!/bin/bash
# OpenVLA LoRA fine-tune launcher. Run on the GPU box from openvla/ root:
#   cd openvla && bash ../templates/openvla_finetune/finetune_lora.sh
#
# All settings come from env vars (defaults = BridgeData V2 single-A100 recipe
# from the OpenVLA README). Override per run, e.g.:
#   DATASET_NAME=my_dataset DATA_ROOT=/data/oxe BATCH_SIZE=8 GRAD_ACCUM=2 bash ../templates/openvla_finetune/finetune_lora.sh
set -euo pipefail

# --- Model / data ---
VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets}"        # parent dir of the RLDS datasets
DATASET_NAME="${DATASET_NAME:-bridge_orig}"     # must be registered (see README)
RUN_ROOT="${RUN_ROOT:-$HOME/openvla-runs}"      # logs + checkpoints land here
ADAPTER_TMP="${ADAPTER_TMP:-/tmp/openvla-adapter}"

# --- Training hyperparams ---
# --batch_size 16 + --grad_accumulation_steps 1 needs ~72 GB VRAM.
# Smaller GPU (>= ~27 GB): lower BATCH_SIZE, raise GRAD_ACCUM to keep
# effective batch size stable, e.g. BATCH_SIZE=8 GRAD_ACCUM=2.
LORA_RANK="${LORA_RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-5e-4}"
IMAGE_AUG="${IMAGE_AUG:-True}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
GPUS="${GPUS:-1}"

# --- Logging ---
WANDB_PROJECT="${WANDB_PROJECT:-openvla-ft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"                # empty = default entity

if [[ ! -d "${DATA_ROOT}/${DATASET_NAME}" ]]; then
  echo "[ERROR] dataset not found: ${DATA_ROOT}/${DATASET_NAME}" >&2
  echo "        See templates/openvla_finetune/README.md (dataset setup)." >&2
  exit 1
fi

WANDB_ARGS=()
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
fi

torchrun --standalone --nnodes 1 --nproc-per-node "${GPUS}" vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --run_root_dir "${RUN_ROOT}" \
  --adapter_tmp_dir "${ADAPTER_TMP}" \
  --lora_rank "${LORA_RANK}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --image_aug "${IMAGE_AUG}" \
  --wandb_project "${WANDB_PROJECT}" \
  "${WANDB_ARGS[@]}" \
  --save_steps "${SAVE_STEPS}"

echo "Done. Adapters + logs under ${RUN_ROOT}. Load the checkpoint with"
echo "AutoModelForVision2Seq.from_pretrained(<run_dir>, trust_remote_code=True)."
