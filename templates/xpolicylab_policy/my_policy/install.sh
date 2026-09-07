#!/bin/bash
# TODO: create the isolated policy conda env and install model deps here.
# Keep policy deps OUT of the RoboDojo sim env; the server and sim client
# activate different envs (see setup_eval_policy_server.sh).
set -euo pipefail

POLICY_ENV="${1:-my-policy}"   # TODO: rename default to your env name

if ! command -v conda &>/dev/null; then
  echo "[install] conda not found; install Miniconda, then re-run: bash install.sh ${POLICY_ENV}"
  exit 1
fi

conda create -n "${POLICY_ENV}" python=3.11 -y
conda activate "${POLICY_ENV}"

# TODO: install your deps, e.g.
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# pip install -r requirements.txt

echo "[install] policy env '${POLICY_ENV}' ready"
