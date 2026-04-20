#!/bin/bash
# R56 Dev A Phase 2: 3-variant bench (V0/V1/V2) across 3 RRR cells.
# Per cell × variant: 5 runs, MXFP8_WARMUP=100 MXFP8_ITERS=200.
# 30s cooldown between runs, 60s rebuild_cool between (re)builds.
# GPU 0 isolated. Median (rank 3 of 5) is the score.
set -euo pipefail
cd "$(dirname "$0")/r56a_workspace"
export THUNDERKITTENS_ROOT="$(cd ../../../.. && pwd)"
OUTDIR="../r56a_results/bench"
mkdir -p "$OUTDIR"

export HIP_VISIBLE_DEVICES=0
export MXFP8_WARMUP=100
export MXFP8_ITERS=200
export MXFP8_LAYOUTS=rrr
export MXFP8_PRESHUFFLE_QUANT=1
export MXFP8_RRR_PRESHUFFLE_V2_RUNTIME=1
export MXFP8_CHECK=0

# Cells (label, M, N, K)
# 8B Gate/Up RRR: PRIMARY HEADROOM cell from R55F (-1.1pp gap, SALU-DOMINANT signature).
# 70B Gate/Up RRR: cross-shape consistency, K=4096, larger N.
# 70B Q/O RRR: cross-shape consistency, K=8192, where R55F PMC showed SALU
#              expansion drops to +5.4% vs FP8 (vs +50.6% on 8B GU).
CELLS=(
  "8B_GateUp_RRR:4096:14336:4096"
  "70B_GateUp_RRR:4096:28672:8192"
  "70B_QO_RRR:4096:8192:8192"
)

build_for_cell() {
    local M=$1 N=$2 K=$3 V=$4
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DMXFP8_RRR_SALU_SETPRIO_R56A=$V" \
        > "$OUTDIR/build_${M}x${N}x${K}_v${V}.log" 2>&1
}

run_5x() {
    local LABEL=$1 M=$2 N=$3 K=$4 V=$5
    local LOG="$OUTDIR/${LABEL}_v${V}.log"
    : > "$LOG"
    for run in 1 2 3 4 5; do
        echo "  run $run/5 ..."
        timeout 600 python3 test_mxfp8_python.py $M $N $K >> "$LOG" 2>&1 || echo "  (timeout/fail)" >> "$LOG"
        if [ $run -lt 5 ]; then sleep 30; fi
    done
}

# Extract MXFP8 RRR TFLOPS values; median (rank 3 of 5).
extract_tflops_median() {
    local LOG=$1
    # Pull MXFP8 RRR TFLOPS lines (one per run).
    awk '/--- RRR Layout/{rrr=1} rrr && /MXFP8.*TFLOPS|TFLOPS.*MXFP8|achieved_TFLOPS/{print}' "$LOG" \
      | grep -oE "[0-9]+\.[0-9]+" | sort -n | head -100 > /tmp/_tflops_$$.txt
    # The bench prints multiple TFLOPS values per run; we want the last one
    # (the MXFP8 score). Use a different strategy: parse line-by-line per run.
    rm -f /tmp/_tflops_$$.txt
    python3 -c "
import re,sys
with open('$LOG') as f: lines=f.readlines()
# Per-run extraction: find each '--- RRR Layout' section and grab the
# 'mxfp8' MXFP8 throughput. Print one TFLOPS per run, then median.
runs=[]
in_run=False; in_rrr=False; cur=None
for ln in lines:
    if 'MXFP8 GEMM Layout Benchmark' in ln:
        in_run=True; in_rrr=False; cur=None
    if in_run and '--- RRR Layout' in ln:
        in_rrr=True
    if in_rrr:
        m=re.search(r'mxfp8.*?([0-9]+\.[0-9]+)\s*TFLOPS', ln)
        if m:
            cur=float(m.group(1))
            runs.append(cur)
            in_rrr=False
            in_run=False
runs.sort()
if len(runs)>=3:
    median=runs[len(runs)//2]
    print(f'{median:.2f}')
elif len(runs)>0:
    print(f'{runs[len(runs)//2]:.2f}')
else:
    print('?')
"
}

echo "=========== R56 Dev A Phase 2 SETPRIO bench ==========="
date
for CELL in "${CELLS[@]}"; do
    LABEL="${CELL%%:*}"
    REST="${CELL#*:}"
    M="${REST%%:*}"; REST="${REST#*:}"
    N="${REST%%:*}"; K="${REST#*:}"
    echo "=== $LABEL  M=$M N=$N K=$K ==="

    for V in 0 1 2; do
        echo "  -- V=$V build --"
        build_for_cell $M $N $K $V
        sleep 60   # rebuild_cool
        echo "  -- V=$V 5-run bench --"
        run_5x $LABEL $M $N $K $V
    done
done

echo ""
echo "=========== SUMMARY (median TFLOPS, MXFP8 RRR) ==========="
date
printf "%-24s %-10s %-10s %-10s %-12s %-12s\n" "CELL" "V0" "V1" "V2" "V1_DELTA%" "V2_DELTA%"
for CELL in "${CELLS[@]}"; do
    LABEL="${CELL%%:*}"
    v0=$(extract_tflops_median "$OUTDIR/${LABEL}_v0.log" 2>/dev/null || echo "?")
    v1=$(extract_tflops_median "$OUTDIR/${LABEL}_v1.log" 2>/dev/null || echo "?")
    v2=$(extract_tflops_median "$OUTDIR/${LABEL}_v2.log" 2>/dev/null || echo "?")
    if [[ "$v0" != "?" && "$v1" != "?" ]]; then
        d1=$(python3 -c "print(f'{($v1-$v0)/$v0*100:+.2f}')")
    else
        d1="?"
    fi
    if [[ "$v0" != "?" && "$v2" != "?" ]]; then
        d2=$(python3 -c "print(f'{($v2-$v0)/$v0*100:+.2f}')")
    else
        d2="?"
    fi
    printf "%-24s %-10s %-10s %-10s %-12s %-12s\n" "$LABEL" "$v0" "$v1" "$v2" "$d1" "$d2"
done
