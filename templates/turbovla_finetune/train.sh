#!/usr/bin/env bash
# TurboVLA-on-RoboDojo training launcher. Run on the GPU box from turbovla/ root:
#   cd turbovla && bash ../templates/turbovla_finetune/train.sh
#
# All settings come from env vars. Minimal run (4x RTX 4090 paper recipe):
#   ROBODOJO_DATA_ROOT=/data/RoboDojo_ee_lerobot_v30_video \
#   BERT_MODEL_PATH=/models/bert-base-uncased DINOV3_MODEL_PATH=/models/dinov3-vitl \
#   TURBOVLA_INIT_CKPT=/models/groundingdino_swint_ogc.pth \
#   RUN_ID=turbovla_robodojo_arx_x5_55k \
#   bash ../templates/turbovla_finetune/train.sh
#
# Smaller GPU (>= ~24 GB): lower PER_DEVICE_BATCH_SIZE, raise GRAD_ACCUM to keep
# the global batch stable, e.g. PER_DEVICE_BATCH_SIZE=12 GRAD_ACCUM=4.
set -euo pipefail

TEMPLATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${TEMPLATE_DIR}/../.." && pwd)"
# Absolute, because we cd into the turbovla checkout below (this script is
# archived into the run dir for provenance).
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
TURBOVLA_REPO="${TURBOVLA_REPO:-${WORKSPACE_ROOT}/turbovla}"
cd "${TURBOVLA_REPO}"

export PYTHONPATH="${TURBOVLA_REPO}:${TURBOVLA_REPO}/third_party/starvla_runtime:${PYTHONPATH:-}"

: "${ROBODOJO_DATA_ROOT:?Set ROBODOJO_DATA_ROOT to the dir of per-task LeRobot datasets (lerobot_v3.0_ee).}"
: "${BERT_MODEL_PATH:?Set BERT_MODEL_PATH to a local bert-base-uncased directory.}"
: "${TURBOVLA_INIT_CKPT:?Set TURBOVLA_INIT_CKPT to groundingdino_swint_ogc.pth, or a released TurboVLA ckpt with TURBOVLA_INIT_FULL=1.}"
export TURBOVLA_INIT_FULL="${TURBOVLA_INIT_FULL:-false}"
: "${DINOV3_MODEL_PATH:?Set DINOV3_MODEL_PATH to a local DINOv3 model directory.}"

OVERLAY="${TURBOVLA_REPO}/experiments/robodojo"
if [[ ! -f "${OVERLAY}/data_registry/data_config.py" ]]; then
  echo "[ERROR] RoboDojo training overlay not installed at ${OVERLAY}." >&2
  echo "        Run: bash scripts/install_turbovla_training.sh  (from workspace root)" >&2
  exit 1
fi

# Discover per-task LeRobot datasets (a dir counts when it has meta/info.json).
if [[ -z "${ROBODOJO_TASKS:-}" ]]; then
  discovered="$(find "${ROBODOJO_DATA_ROOT}" -maxdepth 1 -mindepth 1 -type d \
    | while read -r d; do [[ -f "$d/meta/info.json" ]] && basename "$d"; done \
    | sort | paste -sd, -)"
  if [[ -z "${discovered}" ]]; then
    echo "[ERROR] No LeRobot datasets under ${ROBODOJO_DATA_ROOT} (want <task>/meta/info.json)." >&2
    echo "        Download: cd RoboDojo && bash scripts/RoboDojo/download_data.sh huggingface lerobot_v3.0_ee" >&2
    exit 1
  fi
  export ROBODOJO_TASKS="${discovered}"
fi
echo "[INFO] tasks: ${ROBODOJO_TASKS}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0,virbr0,veth}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"

config_yaml="${CONFIG_YAML:-${OVERLAY}/configs/robodojo.yaml}"
run_root_dir="${RUN_ROOT_DIR:-results/Checkpoints}"
run_id="${RUN_ID:-turbovla_robodojo_arx_x5_55k}"
launcher_python="${STARVLA_PYTHON:-python}"
num_processes="${NUM_PROCESSES:-4}"
main_process_port="${MAIN_PROCESS_PORT:-29630}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-48}"
gradient_accumulation_steps="${GRAD_ACCUM:-1}"
max_train_steps="${MAX_TRAIN_STEPS:-55000}"
warmup_steps="${WARMUP_STEPS:-1000}"
save_interval="${SAVE_INTERVAL:-5000}"
logging_frequency="${LOGGING_FREQUENCY:-50}"
learning_rate="${LEARNING_RATE:-5.0e-05}"
ema_decay="${EMA_DECAY:-0.999}"
ema_device="${EMA_DEVICE:-cuda}"
output_dir="${run_root_dir}/${run_id}"

for required_file in \
    "${DINOV3_MODEL_PATH}/config.json" \
    "${BERT_MODEL_PATH}/config.json" \
    "${TURBOVLA_INIT_CKPT}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[ERROR] Required model file not found: ${required_file}" >&2
    exit 1
  fi
done

if [[ -e "${output_dir}" ]]; then
  echo "[ERROR] Output directory already exists: ${output_dir}" >&2
  echo "        Set RUN_ID to a new name to start another experiment." >&2
  exit 1
fi

echo "[INFO] TurboVLA | RoboDojo arx_x5"
echo "[INFO] data_root=${ROBODOJO_DATA_ROOT}"
echo "[INFO] output_dir=${output_dir}"
echo "[INFO] processes=${num_processes} per_device_bs=${per_device_batch_size} grad_accum=${gradient_accumulation_steps}"
echo "[INFO] global_batch=$((num_processes * per_device_batch_size * gradient_accumulation_steps))"
echo "[INFO] lr=${learning_rate} warmup=${warmup_steps} ema_decay=${ema_decay}"

mkdir -p "${output_dir}"
cp "${SCRIPT_PATH}" "${output_dir}/"
cp "${TURBOVLA_REPO}/third_party/starvla_runtime/starVLA/training/train_robotwin_clean_act_pi05_recipe.py" "${output_dir}/"

"${launcher_python}" -m accelerate.commands.launch \
  --config_file "${TURBOVLA_REPO}/experiments/robotwin/configs/deepspeed_zero2.yaml" \
  --num_processes "${num_processes}" \
  --main_process_port "${main_process_port}" \
  "${TURBOVLA_REPO}/third_party/starvla_runtime/starVLA/training/train_robotwin_clean_act_pi05_recipe.py" \
  --config_yaml "${config_yaml}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --trainer.learning_rate.base "${learning_rate}" \
  --trainer.learning_rate.text_encoder "${learning_rate}" \
  --trainer.learning_rate.vision_encoder "${learning_rate}" \
  --trainer.learning_rate.vision_language_interaction "${learning_rate}" \
  --trainer.learning_rate.vision_projection "${learning_rate}" \
  --trainer.learning_rate.action_head "${learning_rate}" \
  --trainer.gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --trainer.ema_decay "${ema_decay}" \
  --trainer.ema_device "${ema_device}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.num_warmup_steps "${warmup_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.logging_frequency "${logging_frequency}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${WANDB_PROJECT:-turbovla_robodojo}" \
  --wandb_entity "${WANDB_ENTITY:-your_wandb_entity}" \
  "$@"

cat <<EOF

[INFO] Done. Next:
  1. Stats:  python templates/turbovla_finetune/compute_stats.py --data-root "${ROBODOJO_DATA_ROOT}" --out <run>/robodojo_stats.json
  2. Deploy: copy templates/turbovla_finetune/deploy.robodojo.yml over the adapter deploy.yml,
             point checkpoint_path at <run>/checkpoints/*_ema_pytorch_model.pt and stats_path at the stats JSON.
  3. Eval:   bash scripts/run_eval.sh --policy turbovla --task <task> --ckpt <run> --mode smoke --fail-fast
EOF
