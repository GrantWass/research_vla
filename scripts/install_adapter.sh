#!/bin/bash
# Install a this-repo policy adapter into the upstream XPolicyLab checkout.
#
#   bash scripts/install_adapter.sh turbovla [--force]
#
# Copies adapters/<name>_robodojo/ -> RoboDojo/XPolicyLab/policy/<DirFromConf>/,
# where <DirFromConf> is the basename of XPOLICYLAB_POLICY_DIR in
# policies/<name>.conf (TurboVLA for turbovla). Upstream files are never edited
# by hand; re-run to pick up adapter changes. Without --force, files that
# differ from the source are left untouched and reported.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORCE=0
FORCE_CONFIG=0
POLICY=""

# deploy.yml is machine-local CONFIG, not code: it points at whichever checkpoint
# this box should serve, which for a model we fine-tuned here is NOT the stock
# released one the adapter ships. --force must not clobber it, or `make test`
# (which force-syncs the adapter) silently reverts a deployed fine-tuned config
# -- and an eval launched afterwards reports numbers for the wrong checkpoint.
# Use --force-config to deliberately reset it to the adapter's default.
CONFIG_FILES=("deploy.yml")

for arg in "$@"; do
  case "${arg}" in
    --force) FORCE=1 ;;
    --force-config) FORCE=1; FORCE_CONFIG=1 ;;
    -h|--help)
      echo "Usage: bash scripts/install_adapter.sh <policy> [--force] [--force-config]"
      echo "  --force         overwrite adapter CODE that differs"
      echo "  --force-config  also overwrite deploy.yml (resets the served checkpoint)"
      exit 0
      ;;
    *) POLICY="${arg}" ;;
  esac
done

if [[ -z "${POLICY}" ]]; then
  echo "[install_adapter] Usage: bash scripts/install_adapter.sh <policy> [--force]" >&2
  exit 2
fi

CONF="${ROOT}/policies/${POLICY}.conf"
if [[ ! -f "${CONF}" ]]; then
  echo "[install_adapter] Unknown policy '${POLICY}' (no ${CONF})" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${CONF}"

SRC="${ROOT}/adapters/${POLICY}_robodojo"
if [[ ! -d "${SRC}" ]]; then
  echo "[install_adapter] '${POLICY}' uses an upstream adapter (${XPOLICYLAB_POLICY_DIR:-unknown}); nothing to install."
  exit 0
fi

DEST_DIR_NAME="$(basename "${XPOLICYLAB_POLICY_DIR}")"
DEST="${ROOT}/RoboDojo/XPolicyLab/policy/${DEST_DIR_NAME}"
mkdir -p "${DEST}"

changed=0
for src_file in "${SRC}"/*; do
  [[ -f "${src_file}" ]] || continue
  base="$(basename "${src_file}")"
  dest_file="${DEST}/${base}"
  is_config=0
  for cfg in "${CONFIG_FILES[@]}"; do
    [[ "${base}" == "${cfg}" ]] && is_config=1
  done
  if [[ "${is_config}" -eq 1 && "${FORCE_CONFIG}" -ne 1 && -f "${dest_file}" ]] \
     && ! cmp -s "${src_file}" "${dest_file}"; then
    echo "[install_adapter] kept local config: ${base} (--force-config to reset)"
    continue
  fi
  if [[ -f "${dest_file}" ]] && cmp -s "${src_file}" "${dest_file}"; then
    echo "[install_adapter] unchanged: ${base}"
  elif [[ -f "${dest_file}" && "${FORCE}" -ne 1 ]]; then
    echo "[install_adapter] DIFFERS (use --force to overwrite): ${base}"
    changed=1
  else
    cp "${src_file}" "${dest_file}"
    echo "[install_adapter] installed: ${base}"
  fi
done

# The policy server imports <policy_dir>.model as a package module.
touch "${DEST}/__init__.py"

if [[ "${changed}" -ne 0 ]]; then
  echo "[install_adapter] Done with warnings (some files differ; re-run with --force)." >&2
  exit 1
fi
echo "[install_adapter] ${POLICY} -> RoboDojo/${XPOLICYLAB_POLICY_DIR}/"
