#!/usr/bin/env bash
# TurboVLA policy env for RoboDojo eval (GPU box, Ubuntu 22.04 + CUDA).
#
#   cd RoboDojo/XPolicyLab/policy/TurboVLA   # after scripts/install_adapter.sh turbovla
#   bash install.sh
#   conda activate turbovla-robodojo
#
# What it does:
#   1. Creates/uses conda env $TURBOVLA_CONDA_ENV (default turbovla-robodojo, python 3.10).
#   2. Installs CUDA PyTorch ($TURBOVLA_TORCH_INDEX, default cu121 wheels).
#   3. pip-installs the turbovla checkout ($TURBOVLA_REPO or <workspace>/turbovla) + XPolicyLab.
#   4. Downloads the released TurboVLA ckpts ($TURBOVLA_CKPT_DIR, default ./checkpoints/shared).
#
# Model weights resolved at runtime (deploy.yml, explicit wins):
#   dinov3_path <- $DINOV3_PATH  (DINOv3 ViT-B dir, http://github.com/facebookresearch/dinov3)
#   bert_path   <- $BERT_PATH    (bert-base-uncased HF dir)
#   ckpt        <- --ckpt run dir under ./checkpoints/ or $TURBOVLA_CKPT file/dir
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
# Installed at RoboDojo/XPolicyLab/policy/TurboVLA -> workspace root is 4 levels up.
WORKSPACE_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)"

CONDA_ENV="${TURBOVLA_CONDA_ENV:-turbovla-robodojo}"
PYTHON_VERSION="${TURBOVLA_PYTHON_VERSION:-3.10}"
TURBOVLA_REPO="${TURBOVLA_REPO:-${WORKSPACE_ROOT}/turbovla}"
TORCH_INDEX="${TURBOVLA_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
CKPT_DIR="${TURBOVLA_CKPT_DIR:-${POLICY_DIR}/checkpoints/shared}"

echo "[TurboVLA] POLICY_DIR=${POLICY_DIR}"
echo "[TurboVLA] repo=${TURBOVLA_REPO}"
echo "[TurboVLA] conda env=${CONDA_ENV} (python=${PYTHON_VERSION})"

if [[ ! -d "${TURBOVLA_REPO}/turbovla/models" ]]; then
  echo "[TurboVLA] ERROR: turbovla checkout not found at ${TURBOVLA_REPO}." >&2
  echo "[TurboVLA] Run setup.sh at the workspace root first." >&2
  exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "[TurboVLA] ERROR: conda not found. Install Miniconda/Miniforge first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "[TurboVLA] Creating conda env: ${CONDA_ENV}"
  conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${CONDA_ENV}"

python -m pip install --upgrade pip setuptools wheel
echo "[TurboVLA] Installing PyTorch from ${TORCH_INDEX}"
pip install torch torchvision --index-url "${TORCH_INDEX}"

echo "[TurboVLA] Installing turbovla (editable) ..."
pip install -e "${TURBOVLA_REPO}"

echo "[TurboVLA] Installing XPolicyLab (editable) ..."
cd "${XPOLICYLAB_ROOT}"
pip install -e .
pip install h5py pillow safetensors transformers timm

python -c "import XPolicyLab; print('[TurboVLA] XPolicyLab ok')"
python -c "import turbovla; print('[TurboVLA] turbovla', turbovla.__version__)"

if command -v hf >/dev/null 2>&1; then
  echo "[TurboVLA] Downloading released checkpoints -> ${CKPT_DIR}"
  mkdir -p "${CKPT_DIR}"
  hf download H-EmbodVis/TurboVLA --local-dir "${CKPT_DIR}"
else
  cat <<EOF
[TurboVLA] 'hf' CLI not found; download checkpoints manually:
  pip install -U huggingface_hub
  hf download H-EmbodVis/TurboVLA --local-dir "${CKPT_DIR}"
EOF
fi

cat <<EOF

[TurboVLA] Installation finished.
  conda activate ${CONDA_ENV}
  Still needed before eval (see README.md):
    export DINOV3_PATH=<dinov3-vitb dir> BERT_PATH=<bert-base-uncased dir>
    --ckpt <your RoboDojo-finetuned run dir, or a released ckpt for smoke>
EOF
