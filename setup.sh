#!/bin/bash
# Reproduce the upstream checkouts used by this workspace at pinned commits.
# Clones are git-ignored (see .gitignore); run this on any fresh machine.
#
#   bash setup.sh          # clone missing repos, checkout pins, init XPolicyLab
#   bash setup.sh --check  # verify existing checkouts match pins (no network writes)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OPENVLA_PIN="c8f03f4"
TURBOVLA_PIN="b29ab14"
ROBODOJO_PIN="ee67a14"
XPOLICYLAB_PIN="432f82b1758c5b1202e42a3dfe014546dbc50871"

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
fi

ensure_clone() {
  local dir="$1" url="$2" pin="$3" label="$4"
  if [[ ! -d "${ROOT}/${dir}/.git" ]]; then
    if [[ "${CHECK_ONLY}" -eq 1 ]]; then
      echo "[FAIL] ${label}: missing ${dir}/ (run without --check to clone)"
      return 1
    fi
    echo "[setup] cloning ${label} -> ${dir}/"
    git clone --depth 50 "${url}" "${ROOT}/${dir}"
  fi
  if [[ -n "${pin}" ]]; then
    local actual
    actual="$(git -C "${ROOT}/${dir}" rev-parse HEAD)"
    if [[ "${actual}" != "${pin}"* ]]; then
      if [[ "${CHECK_ONLY}" -eq 1 ]]; then
        echo "[FAIL] ${label}: at ${actual}, want ${pin}"
        return 1
      fi
      echo "[setup] ${label}: checking out ${pin} (was ${actual})"
      git -C "${ROOT}/${dir}" fetch --depth 50 origin "${pin}" 2>/dev/null || \
        git -C "${ROOT}/${dir}" fetch origin
      git -C "${ROOT}/${dir}" checkout "${pin}"
    else
      echo "[ok] ${label}: ${dir}/ at ${pin}"
    fi
  else
    echo "[ok] ${label}: ${dir}/ present (unpinned)"
  fi
}

fail=0
ensure_clone "openvla" "https://github.com/openvla/openvla.git" "${OPENVLA_PIN}" "OpenVLA" || fail=1
ensure_clone "turbovla" "https://github.com/H-EmbodVis/TurboVLA.git" "${TURBOVLA_PIN}" "TurboVLA" || fail=1
ensure_clone "RoboDojo" "https://github.com/RoboDojo-Benchmark/RoboDojo.git" "${ROBODOJO_PIN}" "RoboDojo" || fail=1

# XPolicyLab submodule (policy adapters incl. OpenVLA_OFT + demo_policy).
if [[ ! -f "${ROOT}/RoboDojo/XPolicyLab/policy/demo_policy/eval.sh" ]]; then
  if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    echo "[FAIL] XPolicyLab submodule not initialized"
    fail=1
  else
    echo "[setup] initializing XPolicyLab submodule"
    git -C "${ROOT}/RoboDojo" submodule update --init --depth 1 XPolicyLab
  fi
fi
if [[ -d "${ROOT}/RoboDojo/XPolicyLab/.git" || -f "${ROOT}/RoboDojo/XPolicyLab/.git" ]]; then
  actual="$(git -C "${ROOT}/RoboDojo/XPolicyLab" rev-parse HEAD)"
  if [[ "${actual}" != "${XPOLICYLAB_PIN}" ]]; then
    echo "[WARN] XPolicyLab at ${actual}, workspace used ${XPOLICYLAB_PIN}"
  else
    echo "[ok] XPolicyLab at ${XPOLICYLAB_PIN}"
  fi
fi

if [[ "${fail}" -ne 0 ]]; then
  echo "setup --check FAILED" >&2
  exit 1
fi
if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "setup --check OK"
else
  echo "setup done. Next: make test"
fi
