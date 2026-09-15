#!/bin/bash
# Install the TurboVLA-on-RoboDojo training overlay into the upstream checkout.
#
#   bash scripts/install_turbovla_training.sh [--force]
#
# Copies templates/turbovla_finetune/{data_registry,configs} ->
#   turbovla/experiments/robodojo/{data_registry,configs}
# The StarVLA registry auto-discovers experiments/*/data_registry/, which is
# how the `robodojo_arx_x5` mix becomes visible to the training recipe.
# Re-run after template edits; without --force, differing files are reported
# and left untouched.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORCE=0
for arg in "$@"; do
  case "${arg}" in
    --force) FORCE=1 ;;
    -h|--help) echo "Usage: bash scripts/install_turbovla_training.sh [--force]"; exit 0 ;;
    *) echo "[install_training] Unknown arg: ${arg}" >&2; exit 2 ;;
  esac
done

SRC="${ROOT}/templates/turbovla_finetune"
DEST="${ROOT}/turbovla/experiments/robodojo"
if [[ ! -d "${ROOT}/turbovla/.git" ]]; then
  echo "[install_training] turbovla/ checkout missing (run setup.sh first)." >&2
  exit 1
fi

changed=0
for sub in data_registry configs; do
  mkdir -p "${DEST}/${sub}"
  for src_file in "${SRC}/${sub}"/*; do
    [[ -f "${src_file}" ]] || continue
    base="$(basename "${src_file}")"
    dest_file="${DEST}/${sub}/${base}"
    if [[ -f "${dest_file}" ]] && cmp -s "${src_file}" "${dest_file}"; then
      echo "[install_training] unchanged: ${sub}/${base}"
    elif [[ -f "${dest_file}" && "${FORCE}" -ne 1 ]]; then
      echo "[install_training] DIFFERS (use --force to overwrite): ${sub}/${base}"
      changed=1
    else
      cp "${src_file}" "${dest_file}"
      echo "[install_training] installed: ${sub}/${base}"
    fi
  done
done

if [[ "${changed}" -ne 0 ]]; then
  echo "[install_training] Done with warnings (re-run with --force)." >&2
  exit 1
fi
echo "[install_training] overlay -> turbovla/experiments/robodojo/"
