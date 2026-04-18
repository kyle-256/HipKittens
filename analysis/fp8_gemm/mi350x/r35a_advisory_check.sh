#!/bin/bash
# R35 Dev A — Phase 2 advisory firing check.
#
# Expected: 5 target shapes (c0/c1/c2/c4/c5/c6/c8 — but c1+c2 share shape and
# c5+c6 share shape, so 5 distinct shapes) emit exactly 1 advisory each;
# 3 neighbor shapes (c3, c7, default 8192^3) emit 0 advisories.
#
# Note: c2 (70B Up) shares shape with c1 — would fire same predicate; c6 (8B Up)
# shares shape with c5. We test all 5 distinct shape predicates that should fire,
# plus the 3 negative shapes.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PYEXT="$(python3-config --extension-suffix)"
LOG="$HERE/r35a_advisory_check.log"
: > "$LOG"

probe() {
  local cell=$1 m=$2 n=$3 k=$4 expected=$5
  local mod="tk_mxfp8_r35a_${cell}"
  local so="$HERE/${mod}${PYEXT}"
  local stderrf="$HERE/r35a_advisory_${cell}.stderr"
  HIP_VISIBLE_DEVICES=0 python3 r35a_advisory_probe.py "$so" "$m" "$n" "$k" \
    > /dev/null 2> "$stderrf"
  local count
  count=$(grep -c '\[tk_mxfp8_layouts\]' "$stderrf" 2>/dev/null)
  count=${count:-0}
  local verdict
  if [ "$count" = "$expected" ]; then verdict="PASS"; else verdict="FAIL"; fi
  printf '%-16s shape=(%4d,%5d,%5d) expected=%d got=%d %s\n' "$cell" "$m" "$n" "$k" "$expected" "$count" "$verdict" | tee -a "$LOG"
}

echo "=== R35 Dev A advisory firing check (Phase 2) ==="            | tee -a "$LOG"
echo "=== Targets: predicate must fire exactly once (expected=1) ===" | tee -a "$LOG"
probe c5_8b_gate    4096 14336  4096 1     # NEW R35 predicate
probe c0_70b_down   4096  8192 28672 1     # R33 Dev A
probe c1_70b_gate   4096 28672  8192 1     # R34 Dev A (covers c1+c2)
probe c4_70b_kv     4096  1024  8192 1     # R34 Dev A
probe c8_8b_kv      4096  1024  4096 1     # R34 Dev A

echo "=== Negatives: predicate must NOT fire (expected=0) ===" | tee -a "$LOG"
probe c3_70b_qo    4096  8192  8192 0
probe c7_8b_qo     4096  4096  4096 0
probe default      8192  8192  8192 0

# Summary
n_fail=$(grep -c FAIL "$LOG")
echo "=== SUMMARY: $n_fail failures (should be 0) ===" | tee -a "$LOG"
exit $n_fail
