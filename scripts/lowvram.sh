#!/bin/bash
# 16 GB GPU profile: fit Isaac Sim + a 3-7B policy server on one card.
#
#   bash scripts/lowvram.sh apply    # patch upstream checkouts + write .lowvram.env
#   bash scripts/lowvram.sh revert   # undo both
#   bash scripts/lowvram.sh status
#
# What it changes (measured on an RTX 4070 Ti SUPER, see GPU_BOX_SETUP.md):
#   - RoboDojo env_cfg/sim/sim_config.yml: performance render mode, no DLAA /
#     reflections / GI / translucency / denoiser, low texture budget, PhysX GPU
#     buffers sized for the num_envs=1 eval (Isaac Sim 7.7 GB -> 5.7 GB).
#   - XPolicyLab OpenVLA_OFT: load_in_4bit (LLM only) + two upstream loader
#     fixes needed for quantized FiLM checkpoints (bf16 7B does not fit).
#   - .lowvram.env (sourced by scripts/run_eval.sh): PyTorch expandable
#     segments and a JAX memory fraction that fits pi0.5 next to the sim.
#
# This is for wiring checks on small GPUs. Rendering and 4-bit weights change
# what the policy sees/outputs, so do not report benchmark numbers from it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBODOJO="${ROOT}/RoboDojo"
XPL="${ROBODOJO}/XPolicyLab"
ENV_FILE="${ROOT}/.lowvram.env"
# repo dir : patch file
PATCHES=(
  "${ROBODOJO}:${ROOT}/patches/robodojo_lowvram_sim.patch"
  "${XPL}:${ROOT}/patches/xpolicylab_openvla_oft_lowvram.patch"
)

usage() { sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

patch_state() {  # prints applied|clean|conflict
  local repo="$1" patch="$2"
  if git -C "${repo}" apply --check --reverse "${patch}" 2>/dev/null; then
    echo applied
  elif git -C "${repo}" apply --check "${patch}" 2>/dev/null; then
    echo clean
  else
    echo conflict
  fi
}

write_env() {
  cat > "${ENV_FILE}" <<'EOF'
# Written by scripts/lowvram.sh apply; sourced by scripts/run_eval.sh.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
EOF
}

[[ -d "${XPL}" ]] || { echo "[lowvram] RoboDojo/XPolicyLab missing (run setup.sh first)." >&2; exit 1; }

case "${1:-}" in
  apply)
    for entry in "${PATCHES[@]}"; do
      repo="${entry%%:*}"; patch="${entry#*:}"
      case "$(patch_state "${repo}" "${patch}")" in
        applied) echo "[lowvram] already applied: $(basename "${patch}")" ;;
        clean) git -C "${repo}" apply "${patch}"; echo "[lowvram] applied: $(basename "${patch}")" ;;
        *) echo "[lowvram] CONFLICT: $(basename "${patch}") (upstream moved or local edits)" >&2; exit 1 ;;
      esac
    done
    write_env
    echo "[lowvram] wrote ${ENV_FILE#"${ROOT}"/}"
    ;;
  revert)
    for entry in "${PATCHES[@]}"; do
      repo="${entry%%:*}"; patch="${entry#*:}"
      if [[ "$(patch_state "${repo}" "${patch}")" == applied ]]; then
        git -C "${repo}" apply --reverse "${patch}"
        echo "[lowvram] reverted: $(basename "${patch}")"
      else
        echo "[lowvram] not applied: $(basename "${patch}")"
      fi
    done
    rm -f "${ENV_FILE}"
    ;;
  status)
    for entry in "${PATCHES[@]}"; do
      repo="${entry%%:*}"; patch="${entry#*:}"
      echo "$(basename "${patch}"): $(patch_state "${repo}" "${patch}")"
    done
    [[ -f "${ENV_FILE}" ]] && echo ".lowvram.env: present" || echo ".lowvram.env: absent"
    ;;
  -h|--help|"") usage ;;
  *) echo "[lowvram] Unknown command: $1" >&2; usage >&2; exit 2 ;;
esac
