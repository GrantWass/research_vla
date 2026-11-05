#!/usr/bin/env bash
# Write a markdown status report for a finetune_job.sh run from its log.
#   bash scripts/finetune_report.sh LOG [REPORT.md]
# Status comes from the log's own markers ("JOB START", "JOB EXIT <code>",
# "End of training"), not from process matching or the word "Error".
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=${1:?usage: finetune_report.sh LOG [REPORT.md]}
REPORT=${2:-${LOG%.log}_report.md}
[ -f "$LOG" ] || { echo "no log $LOG"; exit 1; }

CLEAN=$(mktemp); trap 'rm -f "$CLEAN"' EXIT
tr '\r' '\n' < "$LOG" | grep -vE "Svt|mp4 @|Map:|^\s*$" > "$CLEAN"

exit_line=$(grep -E "^JOB EXIT" "$CLEAN" | tail -1)
fine=$(grep -E "^>> fine-tuning" "$CLEAN" | tail -1)
out=$(sed -n 's/.* -> \(outputs\/[^ ]*\) (.*/\1/p' <<<"$fine")
last_step=$(grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:?]+, +[0-9.?]+s/step" "$CLEAN" | tail -1)
if [ -n "$exit_line" ]; then
  code=$(awk '{print $3}' <<<"$exit_line")
  if [ "$code" = 0 ] && grep -q "End of training" "$CLEAN"; then state="FINISHED"; else state="FAILED (exit code $code)"; fi
elif [ -n "$fine" ]; then state="TRAINING (${last_step:-loading model})"
elif grep -q "^>> converting" "$CLEAN"; then state="CONVERTING dataset"
else state="STARTING"; fi
ckpt=$(ls -d "${out:-/nonexistent}"/checkpoints/last/pretrained_model 2>/dev/null)

{
  echo "# Fine-tune report: $(basename "${LOG%.log}")"
  echo
  echo "Updated: $(date '+%F %T')  "
  echo "**Status: $state**"
  echo
  echo "- Started: $(sed -n 's/^JOB START \([^ ]*\).*/\1/p' "$CLEAN" | head -1)"
  [ -n "$exit_line" ] && echo "- Ended: $(awk '{print $4}' <<<"$exit_line")"
  echo "- Log: \`$LOG\`"
  echo
  echo "## Encoding"
  grep -E "^>> converting|^wrote " "$CLEAN" | sed 's/^/- /' | tail -2
  grep -q "^>> converting" "$CLEAN" || echo "- reused an already converted dataset"
  echo
  echo "## Training"
  [ -n "$fine" ] && echo "- ${fine#>> }"
  grep -E "^>> [0-9.]+ epoch" "$CLEAN" | tail -1 | sed 's/^>> /- /'
  [ -n "$out" ] && echo "- Output dir: \`$out\`"
  if [ -n "$ckpt" ]; then
    echo "- Final model: \`$ckpt\`"
    echo "- Run with: \`make policy CKPT_HF=$ckpt INSTR=\"<task text used in recording>\"\`"
  fi
  [ -z "$exit_line" ] && echo "- GPU now: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null)"
  echo
  echo "### Loss (every 5th logged point)"
  echo '```'
  grep -oE "step:[0-9.]+K? .*loss:[0-9.]+" "$CLEAN" | sed -E 's/ smpl:[^ ]+//; s/ ep:[^ ]+//; s/ grdn:[^ ]+//' | awk 'NR%5==1' | tail -30
  echo '```'
  echo
  echo "### Errors"
  echo '```'
  grep -nE "Traceback|^[A-Za-z.]*(Error|Exception)\b|out of memory" "$CLEAN" | tail -8 | cut -c1-200
  echo '```'
  echo
  echo "### Last log lines"
  echo '```'
  tail -n 8 "$CLEAN" | cut -c1-200
  echo '```'
} > "$REPORT.tmp" && mv "$REPORT.tmp" "$REPORT"
