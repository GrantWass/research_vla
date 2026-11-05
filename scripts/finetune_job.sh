#!/usr/bin/env bash
# Launch a fine-tune as a detached job that survives this terminal closing.
#
#   bash scripts/finetune_job.sh                     # or: make finetune-job
#   EPOCHS=1 BATCH=8 DATA=demos/x bash scripts/finetune_job.sh
#
# Takes the same env vars as finetune_pi05.sh. Writes:
#   outputs/<name>.log           everything (conversion, training progress)
#   outputs/<name>_report.md     status report, refreshed every minute and once at the end
#   outputs/finetune_report.md   symlink to the latest job's report
# The model goes to outputs/<name>/ (name = <dataset>_pi05_<timestamp>).
set -uo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-demos/rubiks_bowl_v5}
export OUT=${OUT:-outputs/$(basename "${DATA%/}")_pi05_$(date +%Y%m%d-%H%M)}
export DATA EPOCHS=${EPOCHS-1}
NAME=$(basename "$OUT")
LOG=outputs/$NAME.log
REPORT=outputs/${NAME}_report.md
mkdir -p outputs
[ -e "$OUT" ] && { echo "$OUT already exists"; exit 1; }
if pgrep -f "[b]in/lerobot-train|[r]eal_robot.train_relative" >/dev/null; then echo "a training run is already using the GPU"; exit 1; fi

ln -sfn "${NAME}_report.md" outputs/finetune_report.md
echo "job $NAME: log $LOG, report $REPORT (also outputs/finetune_report.md)"

setsid nohup bash -c '
  echo "JOB START $(date +%FT%T) pid $$" > "$1"
  ( while sleep 60; do bash scripts/finetune_report.sh "$1" "$2" >/dev/null 2>&1; done ) &
  ticker=$!
  bash scripts/finetune_pi05.sh >> "$1" 2>&1
  code=$?
  kill $ticker 2>/dev/null
  echo "JOB EXIT $code $(date +%FT%T)" >> "$1"
  bash scripts/finetune_report.sh "$1" "$2" >/dev/null 2>&1
' _ "$LOG" "$REPORT" > /dev/null 2>&1 < /dev/null &
sleep 1
bash scripts/finetune_report.sh "$LOG" "$REPORT" 2>/dev/null || true
