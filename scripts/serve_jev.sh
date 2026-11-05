#!/usr/bin/env bash
# JEV-9B (vision) System 1 server for real_robot.jev, sized to share the robot
# PC's 24 GB RTX 3090 with the pi0.5 policy server (~8 GB).
#
# Upstream serve.sh (autotrust/JEV-9B vl/serve.sh) loads Qwen3.5-9B in bf16
# (~19 GB): does not fit next to pi0.5. Here the weights are FP8 (Marlin,
# weight-only on Ampere) and vLLM gets GPU_UTIL of the card.
#   bash scripts/serve_jev.sh            # setup on first run, then serve on :8000
#   ADAPTER=outputs/jev_lora_dir/adapter_vllm   # our fine-tuned direction adapter (real_robot.jev_lora_train)
#   JEV_DIR=... BASE_DIR=... PORT=... GPU_UTIL=0.64 QUANT=none bash scripts/serve_jev.sh
set -euo pipefail
ROOT=${JEV_ROOT:-$HOME/jev}
JEV_DIR=${JEV_DIR:-$ROOT/JEV-9B}
BASE_DIR=${BASE_DIR:-$ROOT/Qwen3.5-9B}
VENV=${JEV_VENV:-$ROOT/.venv}
QUANT=${QUANT:-fp8}
export PATH=$HOME/.local/bin:$PATH
mkdir -p "$ROOT"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv -p 3.12 "$VENV"
  VIRTUAL_ENV="$VENV" uv pip install vllm requests pillow
fi
HF="uvx --from huggingface_hub[cli] hf"
[ -f "$JEV_DIR/vl/serve_decide.py" ] || $HF download autotrust/JEV-9B --include "vl/*" --local-dir "$JEV_DIR"
[ -f "$JEV_DIR/vl/calibration.json" ] || $HF download autotrust/JEV-9B vl/calibration.json --local-dir "$JEV_DIR"
[ -f "$BASE_DIR/config.json" ] || $HF download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir "$BASE_DIR"
Q=(); [ "$QUANT" = none ] || Q=(--quantization "$QUANT")
exec "$VENV/bin/python" "$JEV_DIR/vl/serve_decide.py" --model "$BASE_DIR" --served-model-name autotrust/JEV-9B \
  --enable-lora --max-lora-rank 32 --lora-modules jev-decision="${ADAPTER:-$JEV_DIR/vl/adapter_vllm}" \
  --logprobs-mode processed_logprobs --max-model-len "${MAX_MODEL_LEN:-8192}" --enable-prefix-caching \
  --mamba-cache-mode align --limit-mm-per-prompt '{"image": 4}' --max-num-seqs 8 --max-logprobs 256 \
  --trust-request-chat-template --gpu-memory-utilization "${GPU_UTIL:-0.64}" "${Q[@]}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --max-num-batched-tokens 4096 --mm-processor-kwargs '{"max_pixels": 401408}' ${EXTRA_ARGS:-}
