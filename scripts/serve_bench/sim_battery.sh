#!/bin/bash
# Sim eval battery for serving strategies (GPU box). Replaces the throwaway
# /tmp/sweep*.sh drivers. Each run temporarily sets num_envs, runs one pi0.5
# eval, records wall time + peak VRAM + the exact result directory it created
# (never "newest directory": that misread two runs on 2026-09-30), and
# restores num_envs to 1 on exit.
#
#   bash scripts/serve_bench/sim_battery.sh NAME N TASK EPISODES SEED STRATEGY [MAX_BATCH]
#   STRATEGY: S0 (sequential) | S1 (batched) | S2 (atomic+batched) | S2W (S2 + warmup)
#
# Appends one JSON line per run to $OUT/manifest.jsonl in the
# compare_runs.py manifest format.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-/tmp/sim_battery}"
CFG="${ROOT}/RoboDojo/env_cfg/sim/sim_config.yml"
mkdir -p "${OUT}"
cd "${ROOT}" || exit 1

name="$1" n="$2" task="$3" eps="$4" seed="$5" strat="$6" maxb="${7:-}"
restore() { sed -i "s/^  num_envs:.*/  num_envs: 1/" "${CFG}"; }
trap restore EXIT

case "${strat}" in
  S0) batched=0 atomic=0 warm="" ;;
  S1) batched=1 atomic=0 warm="" ;;
  S2) batched=1 atomic=1 warm="" ;;
  S2W) batched=1 atomic=1 warm="$(seq -s, 1 "${n}")" ;;
  *) echo "unknown strategy ${strat}" >&2; exit 2 ;;
esac

sed -i "s/^  num_envs:.*/  num_envs: ${n}/" "${CFG}"
results="${ROOT}/RoboDojo/eval_result/RoboDojo/${task}/Pi_05"
before="$(find "${results}" -name _result.json 2>/dev/null | sort)"
log="${OUT}/timing_${name}.jsonl"
rm -f "${log}"
nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader -lms 2000 \
  > "${OUT}/vram_${name}.log" 2>&1 &
sampler=$!
start=$(date +%s)
PI05_BATCHED_INFER="${batched}" PI05_ATOMIC="${atomic}" PI05_MAX_BATCH="${maxb}" \
  PI05_WARMUP_BATCHES="${warm}" PI05_TIMING_LOG="${log}" \
  bash scripts/run_eval.sh --policy pi05 --task "${task}" --eval-num "${eps}" \
  --seed "${seed}" > "${OUT}/eval_${name}.log" 2>&1
rc=$?
wall=$(( $(date +%s) - start ))
kill "${sampler}" 2>/dev/null
peak="$(sort -n "${OUT}/vram_${name}.log" | tail -n 1)"
new="$(comm -13 <(echo "${before}") <(find "${results}" -name _result.json 2>/dev/null | sort))"
if [[ "$(echo "${new}" | grep -c .)" != 1 ]]; then
  echo "${name}: expected exactly one new _result.json, got: ${new:-none}" >&2
  exit 1
fi
python3 - "${new}" <<PY | tee -a "${OUT}/manifest.jsonl"
import json, sys
res = json.load(open(sys.argv[1]))
print(json.dumps({"name": "${name}", "strategy": "${strat}", "task": "${task}",
    "seed": ${seed}, "n_envs": ${n}, "eval_time": len(res.get("details", {})) or ${eps},
    "eval_requested": ${eps},
    "success_rate": res.get("success_rate"), "score": res.get("score"),
    "wall_s": ${wall}, "peak_vram_mib": int("${peak}" or 0), "rc": ${rc},
    "timing_log": "timing_${name}.jsonl", "result_json": sys.argv[1],
    "episodes": [[d.get("layout_id"), bool(d.get("success")), d.get("score")]
                 for _, d in sorted(res.get("details", {}).items(),
                                    key=lambda kv: int(kv[0]))]}))
PY
