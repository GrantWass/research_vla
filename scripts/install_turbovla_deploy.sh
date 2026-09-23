#!/bin/bash
# Point the TurboVLA adapter at a RoboDojo-FINE-TUNED checkpoint.
#
#   bash scripts/install_turbovla_deploy.sh --run turbovla_robodojo_stack_bowls_ft --step 2000
#
# `scripts/install_adapter.sh turbovla` installs the adapter's stock deploy.yml,
# which points at the released RoboTwin weights (2 views). A model we fine-tuned
# here needs different paths and 3 views, and hand-editing the installed file is
# fragile: install_adapter.sh --force and setup_policy.sh --robotwin-smoke both
# overwrite it, silently reverting an eval to the wrong checkpoint. This script
# renders templates/turbovla_finetune/deploy.robodojo.yml instead, so the
# deployed config is reproducible from the repo.
#
# Options:
#   --run NAME     run dir under <turbovla>/results/Checkpoints (required)
#   --step N       checkpoint step to deploy (default: highest available)
#   --no-ema       use steps_N_pytorch_model.pt instead of the EMA weights
#   --print        write the rendered config to stdout, install nothing
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="${ROOT}/templates/turbovla_finetune/deploy.robodojo.yml"
TURBOVLA_REPO="${TURBOVLA_REPO:-${ROOT}/turbovla}"
SHARED="${ROOT}/RoboDojo/XPolicyLab/policy/TurboVLA/checkpoints/shared"
DEST="${ROOT}/RoboDojo/XPolicyLab/policy/TurboVLA/deploy.yml"

RUN=""
STEP=""
EMA="_ema"
PRINT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --no-ema) EMA=""; shift ;;
    --print) PRINT=1; shift ;;
    -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "[turbovla_deploy] unknown flag: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${RUN}" ]] || { echo "[turbovla_deploy] --run is required (see --help)" >&2; exit 2; }
[[ -f "${TEMPLATE}" ]] || { echo "[turbovla_deploy] missing ${TEMPLATE}" >&2; exit 1; }

RUN_DIR="${TURBOVLA_REPO}/results/Checkpoints/${RUN}"
CKPT_DIR="${RUN_DIR}/checkpoints"
[[ -d "${CKPT_DIR}" ]] || { echo "[turbovla_deploy] no checkpoints dir: ${CKPT_DIR}" >&2; exit 1; }

if [[ -z "${STEP}" ]]; then
  STEP="$(ls "${CKPT_DIR}" \
    | sed -n "s/^steps_\([0-9]\+\)${EMA:+_ema}_pytorch_model\.pt$/\1/p" \
    | sort -n | tail -1)"
  [[ -n "${STEP}" ]] || { echo "[turbovla_deploy] no steps_*${EMA}_pytorch_model.pt in ${CKPT_DIR}" >&2; exit 1; }
  echo "[turbovla_deploy] latest step: ${STEP}" >&2
fi

CKPT="${CKPT_DIR}/steps_${STEP}${EMA}_pytorch_model.pt"
STATS="${RUN_DIR}/robodojo_stats.json"
for f in "${CKPT}" "${STATS}"; do
  [[ -f "${f}" ]] || { echo "[turbovla_deploy] missing: ${f}" >&2; exit 1; }
done
for d in "${SHARED}/dinov3-vitl16" "${SHARED}/bert-base-uncased"; do
  [[ -d "${d}" ]] || { echo "[turbovla_deploy] missing ${d} (run scripts/setup_policy.sh turbovla)" >&2; exit 1; }
done

render() {
  sed \
    -e "s|^turbovla_repo: null.*|turbovla_repo: ${TURBOVLA_REPO}|" \
    -e "s|^dinov3_path: null.*|dinov3_path: ${SHARED}/dinov3-vitl16|" \
    -e "s|^bert_path: null.*|bert_path: ${SHARED}/bert-base-uncased|" \
    -e "s|^checkpoint_path: null.*|checkpoint_path: ${CKPT}|" \
    -e "s|^stats_path: null.*|stats_path: ${STATS}|" \
    "${TEMPLATE}"
}

if [[ "${PRINT}" -eq 1 ]]; then
  render
  exit 0
fi

render > "${DEST}"
echo "[turbovla_deploy] installed ${DEST}"
echo "[turbovla_deploy]   ckpt  = ${CKPT}"
echo "[turbovla_deploy]   stats = ${STATS}"
echo "[turbovla_deploy] eval: bash scripts/run_eval.sh --policy turbovla --task <task> --ckpt ${RUN}_${STEP} --eval-num 20 --seed 0"
