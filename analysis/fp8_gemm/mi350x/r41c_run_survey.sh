#!/bin/bash
# R41 Dev C — decode-shape coverage survey driver.
# 6 shapes × {MXFP8, FP8} = 12 measurements, RCR layout (LLaMA Q/K/V/O proj).
set -uo pipefail

cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT=/tmp/wt-r41-c
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-2}
export WARMUP=${WARMUP:-50}
export ITERS=${ITERS:-100}
export PREHEAT_S=${PREHEAT_S:-3.0}
export MXFP8_DISPATCH_TRACE=1

OUT="r41c_survey_results.txt"
: > "$OUT"

run_one() {
    local mod=$1 kind=$2 layout=$3 M=$4 N=$5 K=$6
    local err=$(mktemp)
    echo "--- $kind $layout M=$M N=$N K=$K (mod=$mod) ---" | tee -a "$OUT"
    python3 r41c_decode_bench.py "$mod" "$kind" "$layout" "$M" "$N" "$K" 2>"$err" | tee -a "$OUT"
    grep -E '\[mxfp8_dispatch\]|R41C_RESULT' "$err" | tee -a "$OUT"
    echo >> "$OUT"
    rm -f "$err"
}

# 8B decode (N=4096, K=4096) — uses 8B 4kx4kx4k .so
for M in 1 32 128; do
    run_one tk_mxfp8_8b_4kx4kx4k mxfp8 rcr "$M" 4096 4096
    run_one tk_fp8_8b_4kx4kx4k   fp8   rcr "$M" 4096 4096
done

# 70B decode (N=8192, K=8192) — uses 70B 4kx8kx8k .so
for M in 1 32 128; do
    run_one tk_mxfp8_70b_4kx8kx8k mxfp8 rcr "$M" 8192 8192
    run_one tk_fp8_70b_4kx8kx8k   fp8   rcr "$M" 8192 8192
done

echo "=== Done. Results in $OUT ==="
