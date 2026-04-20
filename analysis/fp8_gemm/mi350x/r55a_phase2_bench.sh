#!/bin/bash
# R55 Dev A Phase 2: 4-variant bench across 3 cells.
# 5 runs/cell, MXFP8_WARMUP=100 MXFP8_ITERS=200, 30s cooldown, 60s rebuild_cool.
# Median of 5 runs (rank 3) is the score.
# GPU 0 isolated.
set -euo pipefail
cd "$(dirname "$0")/r55a_workspace"
export THUNDERKITTENS_ROOT="$(cd ../../../.. && pwd)"
OUTDIR="../r55a_results/bench"
mkdir -p "$OUTDIR"

export HIP_VISIBLE_DEVICES=0
# Note: 70B Down has K=28672 — heavier than CRR cells. Using WARMUP=50 ITERS=100
# (still strict-SCLK regime). Correctness check disabled in main bench loop;
# done separately at end on V0 baseline only. Det check separate.
export MXFP8_WARMUP=50
export MXFP8_ITERS=100
export MXFP8_LAYOUTS=rcr
export MXFP8_PRESHUFFLE_QUANT=1
export MXFP8_CHECK=0

# Cells (label, M, N, K)
CELLS=(
  "70B_Down_RCR:4096:8192:28672"
  "8B_Down_RCR:4096:4096:14336"
  "70B_QO_RCR:4096:8192:8192"
)

build_for_cell() {
    local M=$1 N=$2 K=$3 V=$4
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DMXFP8_RCR_SCALE_INTERLEAVE=$V" \
        > /tmp/r55a_build_${V}_${M}x${N}x${K}.log 2>&1
}

run_5x() {
    local LABEL=$1 M=$2 N=$3 K=$4 V=$5
    local LOG="$OUTDIR/${LABEL}_v${V}.log"
    : > "$LOG"
    for run in 1 2 3 4 5; do
        echo "  run $run/5 ..."
        timeout 600 python3 test_mxfp8_python.py $M $N $K >> "$LOG" 2>&1 || echo "  (timeout)" >> "$LOG"
        if [ $run -lt 5 ]; then sleep 30; fi
    done
}

extract_tflops() {
    local LOG=$1
    # Pull all TFLOPS values from RCR layout sections (one per run).
    # Median = rank 3 of 5.
    grep -oE "TFLOPS:[ ]*[0-9]+\.[0-9]+" "$LOG" \
        | grep -oE "[0-9]+\.[0-9]+$" \
        | sort -n | awk 'NR==3 {print}'
}

echo "=========== R55 Dev A Phase 2 INTERLEAVE bench ==========="
for CELL in "${CELLS[@]}"; do
    LABEL="${CELL%%:*}"
    REST="${CELL#*:}"
    M="${REST%%:*}"; REST="${REST#*:}"
    N="${REST%%:*}"; K="${REST#*:}"
    echo "=== $LABEL  M=$M N=$N K=$K ==="

    for V in 0 1 2 3; do
        echo "  -- V=$V build --"
        build_for_cell $M $N $K $V
        sleep 60   # rebuild_cool
        echo "  -- V=$V 5-run bench --"
        run_5x $LABEL $M $N $K $V
    done
done

echo ""
echo "=========== SUMMARY ==========="
printf "%-22s %-10s %-10s %-10s %-10s %-12s\n" "CELL" "V0" "V1" "V2" "V3" "BEST_DELTA%"
for CELL in "${CELLS[@]}"; do
    LABEL="${CELL%%:*}"
    v0=$(extract_tflops "$OUTDIR/${LABEL}_v0.log" 2>/dev/null || echo "")
    v1=$(extract_tflops "$OUTDIR/${LABEL}_v1.log" 2>/dev/null || echo "")
    v2=$(extract_tflops "$OUTDIR/${LABEL}_v2.log" 2>/dev/null || echo "")
    v3=$(extract_tflops "$OUTDIR/${LABEL}_v3.log" 2>/dev/null || echo "")
    if [[ -n "$v0" ]]; then
        best=$(python3 -c "
v0=$v0
vs=[x for x in [${v1:-None}, ${v2:-None}, ${v3:-None}] if x is not None]
if vs:
    bv=max(vs); print(f'{(bv-v0)/v0*100:+.2f}')
else:
    print('?')
")
    else
        best="?"
    fi
    printf "%-22s %-10s %-10s %-10s %-10s %-12s\n" "$LABEL" "${v0:-?}" "${v1:-?}" "${v2:-?}" "${v3:-?}" "$best"
done
