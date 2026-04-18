#!/bin/bash
# R36 Dev C — Phase 2 advisory firing check.
#
# Targets: 2 R36-C RCR predicates fire (1 advisory each on the matching shape).
# Regression: 5 R34/R35 RRR predicates still fire on their respective shapes.
# Negatives: c_default (8192^3) emits 0 advisories.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PYEXT="$(python3-config --extension-suffix)"
LOG="$HERE/r36c_advisory_check.log"
: > "$LOG"

probe() {
  local cell=$1 m=$2 n=$3 k=$4 expected=$5
  local mod="tk_mxfp8_r36c_${cell}"
  local so="$HERE/${mod}${PYEXT}"
  local stderrf="$HERE/r36c_advisory_${cell}.stderr"
  HIP_VISIBLE_DEVICES=0 python3 r35a_advisory_probe.py "$so" "$m" "$n" "$k" \
    > /dev/null 2> "$stderrf"
  local count
  count=$(grep -c '\[tk_mxfp8_layouts\]' "$stderrf" 2>/dev/null)
  count=${count:-0}
  local verdict
  if [ "$count" = "$expected" ]; then verdict="PASS"; else verdict="FAIL"; fi
  printf '%-18s shape=(%4d,%5d,%5d) expected=%d got=%d %s\n' "$cell" "$m" "$n" "$k" "$expected" "$count" "$verdict" | tee -a "$LOG"
}

echo "=== R36 Dev C advisory firing check (Phase 2) ===" | tee -a "$LOG"
echo "=== R36-C RCR predicates: must fire (expected=1) ===" | tee -a "$LOG"
probe c_qo_8b      4096  4096  4096 1     # NEW R36-C RCR predicate (8B Q/O)
probe c_qo_70b     4096  8192  8192 1     # NEW R36-C RCR predicate (70B Q/O)

echo "=== R34/R35 RRR predicates regression: must still fire (expected=1) ===" | tee -a "$LOG"
probe c0_70b_down   4096  8192 28672 1     # R33 Dev A
probe c1_70b_gateup 4096 28672  8192 1     # R34 Dev A (covers Gate+Up)
probe c4_70b_kv     4096  1024  8192 1     # R34 Dev A
probe c5_8b_gateup  4096 14336  4096 1     # R35 Dev A
probe c8_8b_kv      4096  1024  4096 1     # R34 Dev A

echo "=== Negative: default 8192^3 must NOT fire (expected=0) ===" | tee -a "$LOG"
probe c_default     8192  8192  8192 0

n_fail=$(grep -c FAIL "$LOG")
echo "=== SUMMARY: $n_fail failures (should be 0) ===" | tee -a "$LOG"
exit $n_fail
