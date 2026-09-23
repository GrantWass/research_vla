#!/bin/bash
# Model-agnostic RoboDojo eval entry point.
#
# Swapping VLA models is exactly one flag:
#
#   bash scripts/run_eval.sh --policy openvla  --task stack_bowls --dry-run
#   bash scripts/run_eval.sh --policy turbovla --task stack_bowls --dry-run
#   bash scripts/run_eval.sh --policy pi05     --task stack_bowls --dry-run
#
# The --policy name selects policies/<name>.conf, which resolves the
# XPolicyLab adapter dir, policy env, default ckpt, action type, and env cfg.
# Everything is forwarded to RoboDojo/scripts/robodojo.sh (the only supported
# eval entry point); this script only fills in model-specific defaults.
#
# Usage:
#   bash scripts/run_eval.sh [--policy NAME] --task TASK [--ckpt CKPT]
#       [--mode eval|dry-run|smoke|benchmark|server|client]
#       [--env-cfg CFG] [--action-type joint|ee] [--seed N] [--eval-num N]
#       [--policy-gpu ID] [--env-gpu ID] [--list] [EXTRA... -> robodojo.sh]
#
#   --policy defaults to $POLICY, then "openvla".
#   --mode defaults to "eval" ("dry-run" only resolves/prints, Mac-safe).
#   Unknown flags are passed through to robodojo.sh verbatim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICIES_DIR="${ROOT}/policies"
# Small-GPU memory settings from `scripts/lowvram.sh apply` (absent by default).
if [[ -f "${ROOT}/.lowvram.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.lowvram.env"
  echo "[run_eval] low-VRAM profile active (.lowvram.env)" >&2
fi
# RoboDojo's per-policy server scripts call `conda activate`, so conda must be on
# PATH. An interactive shell has it via ~/.bashrc, but `ssh host "bash ..."` and
# cron/nohup launches do not -- there the server dies with "conda: command not
# found" before it opens its port. Put conda on PATH here so every launch works.
if ! command -v conda >/dev/null 2>&1; then
  for _conda_sh in "${CONDA_ROOT:-}/etc/profile.d/conda.sh" \
                   "${HOME}/miniconda3/etc/profile.d/conda.sh" \
                   "${HOME}/anaconda3/etc/profile.d/conda.sh" \
                   "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -n "${_conda_sh}" && -f "${_conda_sh}" ]]; then
      # shellcheck disable=SC1090
      source "${_conda_sh}"
      echo "[run_eval] sourced conda from ${_conda_sh}" >&2
      break
    fi
  done
fi

# Overridable so tests can stub the harness and inspect the forwarded args.
ROBODOJO_SH="${ROBODOJO_SH:-${ROOT}/RoboDojo/scripts/robodojo.sh}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_eval.sh [--policy NAME] --task TASK [options]

Options:
  --policy NAME        Model in policies/<NAME>.conf (default: $POLICY or openvla)
  --task TASK          RoboDojo task (e.g. stack_bowls). Required except with --list
  --ckpt CKPT          Checkpoint run-dir (default: DEFAULT_CKPT from the conf)
  --mode MODE          eval|dry-run|smoke|benchmark|server|client (default: eval)
  --env-cfg CFG        Override conf ENV_CFG
  --action-type T      Override conf ACTION_TYPE (joint|ee)
  --seed N             Seed (default: 0)
  --eval-num N         Eval episodes (eval/client only)
  --policy-gpu ID      GPU for the policy server (default: 0)
  --env-gpu ID         GPU for the sim client (default: 0)
  --list               List registered policies and exit
  -h, --help           This message

Any other flags are forwarded to RoboDojo/scripts/robodojo.sh verbatim.
Examples:
  bash scripts/run_eval.sh --list
  bash scripts/run_eval.sh --policy pi05 --task stack_bowls --dry-run
  bash scripts/run_eval.sh --policy turbovla --task stack_bowls --dry-run
  bash scripts/run_eval.sh --policy openvla --task stack_bowls --mode smoke --fail-fast
EOF
}

list_policies() {
  echo "Registered policies (policies/*.conf):"
  for conf in "${POLICIES_DIR}"/*.conf; do
    [ -e "${conf}" ] || continue
    name="$(basename "${conf}" .conf)"
    desc="$(grep -E '^DESCRIPTION=' "${conf}" | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//')"
    echo "  ${name}  -  ${desc}"
  done
}

POLICY="${POLICY:-openvla}"
TASK=""
CKPT=""
MODE="eval"
ENV_CFG=""
ACTION_TYPE=""
SEED="0"
EVAL_NUM=""
POLICY_GPU="0"
ENV_GPU="0"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy) POLICY="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --env-cfg) ENV_CFG="$2"; shift 2 ;;
    --action-type) ACTION_TYPE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --eval-num) EVAL_NUM="$2"; shift 2 ;;
    --policy-gpu) POLICY_GPU="$2"; shift 2 ;;
    --env-gpu) ENV_GPU="$2"; shift 2 ;;
    --list) list_policies; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    --dry-run) MODE="dry-run"; shift ;;  # convenience alias for --mode dry-run
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

CONF="${POLICIES_DIR}/${POLICY}.conf"
if [[ ! -f "${CONF}" ]]; then
  echo "[run_eval] Unknown policy '${POLICY}'. Available:" >&2
  list_policies >&2
  exit 2
fi
# Registry files are repo-owned KEY=value lines; source them. The conf sets
# model defaults, so stash CLI-passed overrides first — sourcing would
# otherwise clobber them (ENV_CFG/ACTION_TYPE exist in both places).
CLI_ENV_CFG="${ENV_CFG}"
CLI_ACTION_TYPE="${ACTION_TYPE}"
# shellcheck disable=SC1090
source "${CONF}"

for var in POLICY_NAME XPOLICYLAB_POLICY_DIR POLICY_ENV; do
  if [[ -z "${!var:-}" ]]; then
    echo "[run_eval] ${CONF} is missing required field ${var}" >&2
    exit 2
  fi
done
if [[ "${POLICY_NAME}" != "${POLICY}" ]]; then
  echo "[run_eval] ${CONF}: POLICY_NAME='${POLICY_NAME}' != '${POLICY}'" >&2
  exit 2
fi

CKPT="${CKPT:-${DEFAULT_CKPT:-}}"
# Precedence: CLI flag > conf value > builtin default.
ENV_CFG="${CLI_ENV_CFG:-${ENV_CFG:-arx_x5}}"
ACTION_TYPE="${CLI_ACTION_TYPE:-${ACTION_TYPE:-ee}}"
POLICY_DIR="XPolicyLab/policy/$(basename "${XPOLICYLAB_POLICY_DIR}")"
# Guard against a conf whose policy dir disagrees with the value used.
if [[ "${XPOLICYLAB_POLICY_DIR}" != "${POLICY_DIR}" ]]; then
  echo "[run_eval] ${CONF}: XPOLICYLAB_POLICY_DIR must look like XPolicyLab/policy/<name>" >&2
  exit 2
fi

if [[ -z "${TASK}" ]]; then
  case "${MODE}" in
    smoke|benchmark) ;;  # task selection via --only/--dimension passthrough
    *) echo "[run_eval] --task is required for mode '${MODE}'" >&2; exit 2 ;;
  esac
fi
if [[ -z "${CKPT}" ]]; then
  echo "[run_eval] No --ckpt given and ${CONF} sets no DEFAULT_CKPT" >&2
  exit 2
fi

ROBODOJO_DIR="${ROOT}/RoboDojo"
if [[ ! -x "${ROBODOJO_SH}" && ! -f "${ROBODOJO_SH}" ]]; then
  echo "[run_eval] Missing ${ROBODOJO_SH} (run setup.sh first)" >&2
  exit 1
fi
if [[ ! -d "${ROBODOJO_DIR}/${POLICY_DIR}" ]]; then
  echo "[run_eval] Policy adapter not found: RoboDojo/${POLICY_DIR}" >&2
  echo "[run_eval] For this-repo adapters run: bash scripts/install_adapter.sh ${POLICY}" >&2
  exit 1
fi

common=(--policy-dir "${POLICY_DIR}" --ckpt "${CKPT}" --policy-env "${POLICY_ENV}")

case "${MODE}" in
  dry-run)
    args=(eval "${common[@]}")
    [[ -n "${TASK}" ]] && args+=(--task "${TASK}")
    args+=(--env-cfg "${ENV_CFG}" --action-type "${ACTION_TYPE}" --seed "${SEED}")
    [[ -n "${EVAL_NUM}" ]] && args+=(--eval-num "${EVAL_NUM}")
    args+=(--dry-run)
    exec bash "${ROBODOJO_SH}" "${args[@]}" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
    ;;
  eval)
    args=(eval "${common[@]}" --task "${TASK}"
      --env-cfg "${ENV_CFG}" --action-type "${ACTION_TYPE}" --seed "${SEED}"
      --policy-gpu "${POLICY_GPU}" --env-gpu "${ENV_GPU}")
    [[ -n "${EVAL_NUM}" ]] && args+=(--eval-num "${EVAL_NUM}")
    exec bash "${ROBODOJO_SH}" "${args[@]}" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
    ;;
  smoke|benchmark)
    args=("${MODE}" "${common[@]}"
      --env-cfg "${ENV_CFG}" --action-type "${ACTION_TYPE}" --seed "${SEED}")
    [[ -n "${TASK}" ]] && args+=(--only "${TASK}")
    exec bash "${ROBODOJO_SH}" "${args[@]}" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
    ;;
  server)
    exec bash "${ROBODOJO_SH}" server "${common[@]}" \
      --task "${TASK}" \
      --env-cfg "${ENV_CFG}" --action-type "${ACTION_TYPE}" --seed "${SEED}" \
      --policy-gpu "${POLICY_GPU}" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
    ;;
  client)
    args=(client "${common[@]}" --task "${TASK}"
      --env-cfg "${ENV_CFG}" --action-type "${ACTION_TYPE}" --seed "${SEED}"
      --env-gpu "${ENV_GPU}")
    [[ -n "${EVAL_NUM}" ]] && args+=(--eval-num "${EVAL_NUM}")
    exec bash "${ROBODOJO_SH}" "${args[@]}" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
    ;;
  *)
    echo "[run_eval] Unknown --mode '${MODE}' (want eval|dry-run|smoke|benchmark|server|client)" >&2
    exit 2
    ;;
esac
