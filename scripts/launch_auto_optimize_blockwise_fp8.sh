#!/usr/bin/env bash
###############################################################################
# launch_auto_optimize_blockwise_fp8.sh — start the autonomous round loop in
# the background. Detaches via nohup + setsid so the daemon survives this
# shell exiting (and survives this Claude session ending).
#
# Each "round" =
#   1. build prompt (task .md + state)
#   2. spawn `claude --print` (claude edits files, makes ONE commit, exits)
#   3. daemon runs the metric → score
#   4. update best/streak, write summary.json
#   5. early-stop if no-improvement streak ≥ --patience
#
# Usage:
#   bash scripts/launch_auto_optimize_blockwise_fp8.sh
#   ROUNDS=200 PATIENCE=50 bash scripts/launch_auto_optimize_blockwise_fp8.sh
#   bash scripts/launch_auto_optimize_blockwise_fp8.sh --resume-state \\
#       auto_optimize_logs/blockwise_fp8_<TS>/summary.json
#
# Env overrides:
#   ROUNDS        default 200
#   PATIENCE      default 50
#   MODEL         default claude-opus-4-7[1m]
#   EFFORT        default max
#
# Tail liveness:
#   tail -f auto_optimize_logs/blockwise_fp8_<TS>.stdout.log
#
# Stop the loop:
#   kill -- -$(cat auto_optimize_logs/blockwise_fp8_<TS>.pid)   # process group
#   # or just `kill <pid>` (the daemon handles SIGTERM via finally clause and
#   # writes a final summary.json)
###############################################################################
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

ROUNDS="${ROUNDS:-200}"
PATIENCE="${PATIENCE:-50}"
MODEL="${MODEL:-claude-opus-4-7[1m]}"
EFFORT="${EFFORT:-max}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_NAME="blockwise_fp8_${TS}"
LOG_ROOT="$REPO/auto_optimize_logs"
mkdir -p "$LOG_ROOT"
LOG_DIR="$LOG_ROOT/$LOG_NAME"
STDOUT_LOG="$LOG_ROOT/${LOG_NAME}.stdout.log"
PID_FILE="$LOG_ROOT/${LOG_NAME}.pid"

CMD=(
    python3 -u scripts/auto_optimize_blockwise_fp8.py
    --rounds   "$ROUNDS"
    --patience "$PATIENCE"
    --model    "$MODEL"
    --effort   "$EFFORT"
    --log-dir  "$LOG_DIR"
    "$@"
)

echo "[launch] track=blockwise_fp8 (local MI300X, single repo)"
echo "[launch] rounds=$ROUNDS patience=$PATIENCE model=$MODEL effort=$EFFORT"
echo "[launch] log-dir=$LOG_DIR"
echo "[launch] stdout=$STDOUT_LOG"
echo "[launch] pid-file=$PID_FILE"
echo "[launch] cmd: ${CMD[*]}"

nohup setsid "${CMD[@]}" >"$STDOUT_LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"
disown "$PID" 2>/dev/null || true

sleep 2
if kill -0 "$PID" 2>/dev/null; then
    echo "[launch] OK pid=$PID (recorded in $PID_FILE)"
    echo "[launch] tail: tail -f $STDOUT_LOG"
    echo "[launch] stop: kill -- -\$(cat $PID_FILE)"
else
    echo "[launch] FAILED — process exited within 2 s. Inspect $STDOUT_LOG" >&2
    tail -40 "$STDOUT_LOG" >&2 || true
    exit 1
fi
