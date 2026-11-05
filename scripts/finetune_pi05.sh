#!/usr/bin/env bash
# Fine-tune the pi0.5 Piper checkpoint on a recorded demo dataset.
#
#   bash scripts/finetune_pi05.sh                       # demos/rubiks_bowl_v5, defaults below
#   EPOCHS=1 make finetune                              # one pass over the data
#   make finetune                                       # same
#   DATA=demos/other STEPS=5000 bash scripts/finetune_pi05.sh
#   EPISODES=0,1,2,5 EPOCHS=4 make finetune               # train on these episodes only (hold the rest out)
#   RELATIVE=1 make finetune                            # joints relative to the observed ones (real_robot/relative_actions.py)
#   RELATIVE=1 WRIST_OVERSAMPLE=3 JOINT_LOSS_WEIGHTS=1,1,1,2,1.5,2,1 make finetune   # push the rarely moving wrist (real_robot/train_relative.py)
#
# 1. Converts the raw recording (cam_high/cam_wrist, 7-D state) to the pi0.5
#    Piper format (real_robot/convert_dataset.py) unless <DATA>_pi05 exists.
# 2. Runs lerobot-train starting from the banana checkpoint, so the result
#    runs with the same `make policy` flags (CKPT_HF=<output>/checkpoints/last/pretrained_model).
#
# A full fine-tune (AdamW over ~3.6B params) does not fit a 24 GB RTX 3090, so
# the default trains the action expert and projections only (VLM frozen,
# bf16, gradient checkpointing). TRAIN_EXPERT_ONLY=false tries everything and
# will likely run out of memory here.
set -euo pipefail
cd "$(dirname "$0")/.."

LEROBOT_PY=${LEROBOT_PY:-$HOME/lerobot_robot_piper/.venv/bin/python}
LEROBOT_TRAIN=${LEROBOT_TRAIN:-$(dirname "$LEROBOT_PY")/lerobot-train}
DATA=${DATA:-demos/rubiks_bowl_v5}
CONVERTED=${CONVERTED:-${DATA%/}_pi05}
BASE=${BASE:-richardshkim/pi05_piper_banana_v2_full}
OUT=${OUT:-outputs/$(basename "${DATA%/}")_pi05_$(date +%Y%m%d-%H%M)}
EPOCHS=${EPOCHS:-}   # if set, STEPS = EPOCHS * frames / BATCH
STEPS=${STEPS:-3000}
DEVICE=${DEVICE:-cuda}
BATCH=${BATCH:-8}
LR=${LR:-2.5e-5}
WARMUP=${WARMUP:-100}
SAVE_FREQ=${SAVE_FREQ:-500}
WORKERS=${WORKERS:-4}
TRAIN_EXPERT_ONLY=${TRAIN_EXPERT_ONLY:-true}
EPISODES=${EPISODES:-}   # comma-separated episode indices to train on (default: all)
RELATIVE=${RELATIVE:-}   # 1: chunk-relative joint actions (real_robot/train_relative.py)
# LORA=<rank>: LoRA adapters on the vision encoder, language model and action
# expert (attention + MLP), base weights frozen. LeRobot's pi0.5 default targets
# the expert only, which is what expert-only training already does; the point
# here is to adapt the vision side. Fits 24 GB. Checkpoints hold adapters only:
# real_robot/merge_lora.py turns one into a full checkpoint for make policy.
LORA=${LORA:-}
LORA_TARGETS='model\.paligemma_with_expert\..*\.(self_attn\.(q|k|v|o|out)_proj|mlp\.(fc1|fc2|gate_proj|up_proj|down_proj))|model\.paligemma_with_expert\.paligemma\.model\.multi_modal_projector\.linear'
PEFT_ARGS=()
if [ -n "$LORA" ]; then
  PEFT_ARGS=(--peft.method_type=LORA --peft.r="$LORA" --peft.target_modules="$LORA_TARGETS")
  TRAIN_EXPERT_ONLY=false  # wrap_with_peft freezes the base; expert-only would keep the VLM in eval mode
fi

# .convert_done marks a finished conversion (LeRobot creates meta/ at the
# start, so meta/ alone can't tell a half-written dataset from a complete one).
if [ ! -f "$CONVERTED/.convert_done" ]; then
  echo ">> converting $DATA -> $CONVERTED"
  rm -rf "$CONVERTED"
  "$LEROBOT_PY" -m real_robot.convert_dataset "$DATA" "$CONVERTED"
  touch "$CONVERTED/.convert_done"
fi

if [ -n "$EPOCHS" ]; then
  FRAMES=$("$LEROBOT_PY" -c "
import glob, pandas as pd
e = pd.concat(pd.read_parquet(f) for f in glob.glob('$CONVERTED/meta/episodes/*/*.parquet'))
keep = '$EPISODES'
if keep: e = e[e.episode_index.isin([int(i) for i in keep.split(',')])]
print(int(e.length.sum()))")
  STEPS=$(python3 -c "import math;print(math.ceil($EPOCHS*$FRAMES/$BATCH))")
  echo ">> $EPOCHS epoch(s) of $CONVERTED${EPISODES:+ (episodes $EPISODES)} at batch $BATCH = $STEPS steps"
  [ "$SAVE_FREQ" -gt "$STEPS" ] && SAVE_FREQ=$STEPS
fi

EP_ARGS=()
[ -n "$EPISODES" ] && EP_ARGS=(--dataset.episodes="[$EPISODES]")
TRAIN_CMD=("$LEROBOT_TRAIN")
[ -n "$RELATIVE" ] && TRAIN_CMD=("$LEROBOT_PY" -m real_robot.train_relative)
echo ">> fine-tuning $BASE on $CONVERTED -> $OUT ($STEPS steps, batch $BATCH${RELATIVE:+, relative actions})"
exec "${TRAIN_CMD[@]}" \
  --dataset.repo_id="local/$(basename "$CONVERTED")" \
  --dataset.root="$CONVERTED" \
  "${EP_ARGS[@]}" "${PEFT_ARGS[@]}" \
  --policy.path="$BASE" \
  --policy.device="$DEVICE" \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model="${COMPILE:-false}" \
  --policy.train_expert_only="$TRAIN_EXPERT_ONLY" \
  --policy.push_to_hub=false \
  --policy.optimizer_lr="$LR" \
  --policy.scheduler_warmup_steps="$WARMUP" \
  --policy.scheduler_decay_steps="$STEPS" \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --num_workers="$WORKERS" \
  --wandb.enable=false \
  --output_dir="$OUT" \
  --job_name="$(basename "$OUT")"
