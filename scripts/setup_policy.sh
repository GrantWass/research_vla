#!/bin/bash
# One-shot policy env + checkpoint setup on the Linux GPU box (after setup.sh
# and RoboDojo's own install; see GPU_BOX_SETUP.md).
#
#   bash scripts/setup_policy.sh demo
#   bash scripts/setup_policy.sh openvla
#   bash scripts/setup_policy.sh pi05
#   bash scripts/setup_policy.sh turbovla [--robotwin-smoke]
#
# Each target is idempotent and encodes the fixes the upstream installers need
# (verified 2026-09 on Ubuntu 24.04 + driver 580 + RTX 4070 Ti SUPER):
#   openvla  conda env `openvla_oft`: prebuilt flash-attn wheel (the adapter's
#            install.sh dies on CUDA_HOME), dependency pins (`pip check` clean),
#            official RoboDojo ckpt RoboDojo-sim-arx_x5-joint-1 (~19 GB).
#   pi05     PyYAML in conda base (the server launcher reads deploy.yml with the
#            base python), uv env, overridable JAX memory fraction, and the
#            official seed-0 ckpt: params/ + assets/ only (~12 GB, skips the
#            ~30 GB optimizer train_state/).
#   turbovla this-repo adapter + `turbovla-robodojo` env + released ckpts,
#            DINOv3 ViT-L (gated: accept the license on Hugging Face and put a
#            read token in ~/.cache/huggingface/token first) + BERT.
#            --robotwin-smoke points the installed deploy.yml at the released
#            RoboTwin ckpt: proves sim wiring, motions are NOT meaningful.
#   demo     XPolicyLab into the RoboDojo env (zero-action wiring stub).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBODOJO="${ROOT}/RoboDojo"
XPL="${ROBODOJO}/XPolicyLab"
POLICY="${1:-}"
shift || true
ROBOTWIN_SMOKE=0
for arg in "$@"; do
  case "${arg}" in
    --robotwin-smoke) ROBOTWIN_SMOKE=1 ;;
    *) echo "[setup_policy] Unknown arg: ${arg}" >&2; exit 2 ;;
  esac
done

case "${POLICY}" in
  -h|--help|"") sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  demo|openvla|pi05|turbovla) ;;
  *) echo "[setup_policy] Unknown policy: ${POLICY} (demo|openvla|pi05|turbovla)" >&2; exit 2 ;;
esac

log() { echo "[setup_policy:${POLICY}] $*"; }
die() { echo "[setup_policy:${POLICY}] ERROR: $*" >&2; exit 1; }

[[ -d "${XPL}/policy" ]] || die "RoboDojo/XPolicyLab missing (run setup.sh first)."
command -v conda >/dev/null 2>&1 || [[ -x "${HOME}/miniconda3/bin/conda" ]] \
  || die "conda not found (run RoboDojo/scripts/install.sh first; see GPU_BOX_SETUP.md)."
# shellcheck disable=SC1091
source "$( (command -v conda >/dev/null && conda info --base) || echo "${HOME}/miniconda3")/etc/profile.d/conda.sh"
export PIP_USER=0 PYTHONNOUSERSITE=1

# Pull one sub-folder of the RoboDojo HF checkpoint repo (same cache layout as
# RoboDojo/scripts/RoboDojo/download_ckpt.sh) and link it into the adapter.
hf_ckpt_subset() {  # <local policy dir> <remote policy dir> <lfs include globs...>
  local local_policy="$1" remote_policy="$2"; shift 2
  local cache="${ROBODOJO}/.cache/robodojo_ckpt_huggingface_repo"
  local remote_dir="ckpt/RoboDojo/${remote_policy}"
  if [[ ! -d "${cache}/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --sparse \
      https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo "${cache}"
  fi
  GIT_LFS_SKIP_SMUDGE=1 git -C "${cache}" sparse-checkout add "$@"
  local include; include="$(IFS=,; echo "$*" | sed 's|\([^,]*\)|\1/**|g')"
  git -C "${cache}" lfs pull --include="${include}" --exclude=""
  ln -sfn "${cache}/${remote_dir}" "${XPL}/policy/${local_policy}/checkpoints"
}

setup_demo() {
  conda activate RoboDojo
  bash "${XPL}/policy/demo_policy/install.sh"
}

setup_openvla() {
  local dir="${XPL}/policy/OpenVLA_OFT"
  conda env list | awk '{print $1}' | grep -qx openvla_oft \
    || conda create -n openvla_oft python=3.10.6 -y
  conda activate openvla_oft
  pip install -e "${dir}/openvla_oft"   # pins torch 2.2.0 / transformers fork
  pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
  pip install -e "${XPL}"
  # XPolicyLab pulls numpy 2 / opencv 5; TF 2.15 needs protobuf 3.20; accelerate
  # 1.x breaks 8/4-bit loading on transformers 4.40.
  pip install "numpy==1.26.4" "opencv-python-headless==4.11.0.86" \
    "protobuf==3.20.3" "tensorflow-metadata==1.14.0" "wandb==0.17.9" \
    "accelerate==0.30.1" "bitsandbytes==0.43.3"
  pip check
  if [[ ! -e "${dir}/checkpoints/RoboDojo-sim-arx_x5-joint-1" ]]; then
    (cd "${ROBODOJO}" && bash scripts/RoboDojo/download_ckpt.sh huggingface OpenVLA_OFT)
  fi
  log "done. 16 GB GPU? run: bash scripts/lowvram.sh apply"
}

setup_pi05() {
  local dir="${XPL}/policy/Pi_05"
  conda install -n base -y pyyaml
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  local patch="${ROOT}/patches/xpolicylab_pi05_mem_fraction.patch"
  if git -C "${XPL}" apply --check "${patch}" 2>/dev/null; then
    git -C "${XPL}" apply "${patch}" && log "applied $(basename "${patch}")"
  fi
  (cd "${dir}" && UV_HTTP_TIMEOUT=120 bash install.sh)
  local run="ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999"
  if [[ ! -d "${dir}/checkpoints/RoboDojo-sim-arx_x5-joint-0/59999/params/d" ]]; then
    hf_ckpt_subset Pi_05 Pi_05 "${run}/params" "${run}/assets"
    # orbax needs the metadata file next to params/ (not a dir, so pulled by name)
    git -C "${ROBODOJO}/.cache/robodojo_ckpt_huggingface_repo" sparse-checkout add "${run}"
    git -C "${ROBODOJO}/.cache/robodojo_ckpt_huggingface_repo" lfs pull \
      --include="${run}/_CHECKPOINT_METADATA" --exclude=""
  fi
  log "done. 16 GB GPU? run: bash scripts/lowvram.sh apply (sets JAX mem fraction 0.45)"
}

setup_turbovla() {
  local dir="${XPL}/policy/TurboVLA"
  bash "${ROOT}/scripts/install_adapter.sh" turbovla || true
  bash "${dir}/install.sh"
  conda activate turbovla-robodojo
  local shared="${dir}/checkpoints/shared"
  [[ -s "${HOME}/.cache/huggingface/token" || -n "${HF_TOKEN:-}" ]] \
    || die "DINOv3 is gated: accept the license at https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m and save a read token to ~/.cache/huggingface/token"
  hf download facebook/dinov3-vitl16-pretrain-lvd1689m --local-dir "${shared}/dinov3-vitl16"
  hf download bert-base-uncased --local-dir "${shared}/bert-base-uncased" \
    --include "*.json" "*.txt" "model.safetensors"
  if [[ "${ROBOTWIN_SMOKE}" -eq 1 ]]; then
    sed "s|@SHARED@|${shared}|g" "${ROOT}/adapters/turbovla_robodojo/deploy.robotwin_smoke.yml" \
      > "${dir}/deploy.yml"
    log "installed deploy.yml -> released RoboTwin ckpt (smoke only; not RoboDojo-trained)"
  fi
  log "done. Eval: bash scripts/run_eval.sh --policy turbovla --task stack_bowls --ckpt turbovla_robotwin_55k --mode smoke"
}

case "${POLICY}" in
  demo) setup_demo ;;
  openvla) setup_openvla ;;
  pi05) setup_pi05 ;;
  turbovla) setup_turbovla ;;
esac
