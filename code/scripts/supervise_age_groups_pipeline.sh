#!/bin/bash
set -euo pipefail

ROOT="/Users/siyu/Documents/GitHub/VAM-studying"
PID_FILE="$ROOT/age_groups_efficient.pid"
LOG_FILE="$ROOT/age_groups_supervisor.log"
POST_LOG="$ROOT/age_groups_post_analysis.log"

cd "$ROOT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Supervisor started" >> "$LOG_FILE"

if [ ! -f "$PID_FILE" ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: PID file missing: $PID_FILE" >> "$LOG_FILE"
  exit 1
fi

PID=$(cat "$PID_FILE")
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitoring PID $PID" >> "$LOG_FILE"

while kill -0 "$PID" >/dev/null 2>&1; do
  sleep 60
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training process exited" >> "$LOG_FILE"

REQ1="$ROOT/checkpoints_age_groups/20-29/stage2/best_config.json"
REQ2="$ROOT/checkpoints_age_groups/20-29/stage2/best_model_params.npz"
REQ3="$ROOT/checkpoints_age_groups/80-89/stage2/best_config.json"
REQ4="$ROOT/checkpoints_age_groups/80-89/stage2/best_model_params.npz"

if [ -f "$REQ1" ] && [ -f "$REQ2" ] && [ -f "$REQ3" ] && [ -f "$REQ4" ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage-2 artifacts found for both age groups; starting post analysis" >> "$LOG_FILE"
  python run_age_group_post_analysis.py >> "$POST_LOG" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Post analysis completed" >> "$LOG_FILE"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Required Stage-2 artifacts not found; post analysis skipped" >> "$LOG_FILE"
  for f in "$REQ1" "$REQ2" "$REQ3" "$REQ4"; do
    if [ ! -f "$f" ]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Missing: $f" >> "$LOG_FILE"
    fi
  done
  exit 2
fi
