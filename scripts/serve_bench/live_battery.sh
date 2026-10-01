#!/bin/bash
# Live-server serving battery (GPU box). For each server mode, starts one
# standalone pi0.5 policy server, runs replay_client scenarios against it, and
# shuts it down. No simulator: isolates serving behavior from sim speed.
#
#   bash scripts/serve_bench/live_battery.sh equivalence
#   bash scripts/serve_bench/live_battery.sh fleet [OUT_DIR]
#   MODES="cont win10" bash ... fleet OUT_DIR    # append a subset of modes
#
# Server modes (env at server start):
#   fifo    XPL_SERVER_BATCH=0                      upstream: one call at a time
#   cont    XPL_SERVER_BATCH=1 window 0 ms          batch whatever queued meanwhile
#   win10   XPL_SERVER_BATCH=1 XPL_BATCH_WINDOW_MS=10
#   win25   XPL_SERVER_BATCH=1 XPL_BATCH_WINDOW_MS=25
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PI05="${ROOT}/RoboDojo/XPolicyLab/policy/Pi_05"
PY="${PI05}/openpi/.venv/bin/python"
PORT="${PORT:-19732}"
URL="ws://localhost:${PORT}"
OUT="${2:-/tmp/live_battery}"
mkdir -p "${OUT}"
# shellcheck disable=SC1091
[[ -f "${ROOT}/.lowvram.env" ]] && source "${ROOT}/.lowvram.env"

# Non-interactive ssh shells lack conda on PATH; the server script needs it.
export PATH="${HOME}/miniconda3/bin:${HOME}/miniconda3/condabin:${PATH}"
SERVER_PID=""
start_server() { # label, then KEY=VAL env pairs
  local label="$1"; shift
  # Compile every batch size a fleet of <= 8 can produce before measuring.
  env "$@" PI05_BATCHED_INFER=1 PI05_WARMUP_BATCHES="${PI05_WARMUP_BATCHES:-1,2,3,4,5,6,7,8}" \
    PI05_TIMING_LOG="${OUT}/timing_${label}.jsonl" \
    bash "${PI05}/setup_eval_policy_server.sh" RoboDojo stack_bowls \
    RoboDojo-sim-arx_x5-joint-0 arx_x5 joint 0 0 uv "${PORT}" localhost \
    > "${OUT}/server_${label}.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 180); do
    if ss -ltn | grep -q ":${PORT} "; then return 0; fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "server ${label} died; see ${OUT}/server_${label}.log" >&2; return 1
    fi
    sleep 2
  done
  echo "server ${label} never listened" >&2; return 1
}
stop_server() {
  [[ -n "${SERVER_PID}" ]] || return 0
  pkill -TERM -P "${SERVER_PID}" 2>/dev/null; kill "${SERVER_PID}" 2>/dev/null
  wait "${SERVER_PID}" 2>/dev/null
  for _ in $(seq 1 30); do ss -ltn | grep -q ":${PORT} " || break; sleep 1; done
  SERVER_PID=""
}
trap stop_server EXIT

client() {
  (cd "${ROOT}/RoboDojo/XPolicyLab" && PYTHONPATH="${ROOT}/RoboDojo:${ROOT}/RoboDojo/XPolicyLab" \
    "${PY}" "${ROOT}/scripts/serve_bench/replay_client.py" --url "${URL}" "$@")
}

case "${1:-}" in
  equivalence)
    # Two noise seeds: batch-vs-solo drift is judged against the spread a
    # different (equally valid) start noise produces for the same robot.
    for seed in 7 8; do
      start_server "equiv${seed}" PI05_NOISE_SEED="${seed}" XPL_SERVER_BATCH=0 || exit 1
      client --scenario equivalence --iters 3 \
        --out "${OUT}/equivalence_seed${seed}.jsonl" | cut -c1-120
      stop_server
    done
    python3 "${ROOT}/scripts/serve_bench/equivalence_report.py" \
      "${OUT}/equivalence_seed7.jsonl" "${OUT}/equivalence_seed8.jsonl"
    ;;
  fleet)
    [[ -n "${MODES:-}" ]] || : > "${OUT}/fleet_summary.jsonl"
    for mode in ${MODES:-fifo cont win10 win25}; do
      case "${mode}" in
        fifo) envs=(XPL_SERVER_BATCH=0) ;;
        cont) envs=(XPL_SERVER_BATCH=1 XPL_BATCH_WINDOW_MS=0) ;;
        win10) envs=(XPL_SERVER_BATCH=1 XPL_BATCH_WINDOW_MS=10) ;;
        win25) envs=(XPL_SERVER_BATCH=1 XPL_BATCH_WINDOW_MS=25) ;;
      esac
      start_server "${mode}" "${envs[@]}" || continue
      # Warm every batch size once so JIT compiles stay out of the numbers.
      for k in 1 2 4 6 8; do
        client --scenario fleet --clients "${k}" --iters 3 --think-s 0 \
          --label "warm" > /dev/null 2>&1
      done
      for think in 0 0.5 2.0; do
        for k in 1 2 4 6 8; do
          client --scenario fleet --clients "${k}" --iters "${ITERS:-30}" \
            --think-s "${think}" --label "${mode}" \
            --out "${OUT}/fleet_${mode}_k${k}_t${think}.jsonl" \
            | tail -n 1 | tee -a "${OUT}/fleet_summary.jsonl"
        done
      done
      stop_server
    done
    ;;
  *) echo "usage: $0 equivalence|fleet [OUT_DIR]" >&2; exit 2 ;;
esac
