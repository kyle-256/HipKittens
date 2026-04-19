#!/bin/bash
# R54 Dev I Phase 2 QUICK: single-cell A/B sanity bench on primary cell
# (70B Gate/Up CRR, M=4096 N=28672 K=8192). 5 runs each, full WARMUP/ITERS,
# 30s cooldown between runs, 60s rebuild_cool. Median (rank 3) is the score.
set -euo pipefail
cd "$(dirname "$0")/r54i_workspace"
export THUNDERKITTENS_ROOT="$(cd ../../../.. && pwd)"
OUTDIR="../r54i_results/bench"
mkdir -p "$OUTDIR"

export HIP_VISIBLE_DEVICES=0
export MXFP8_WARMUP=100
export MXFP8_ITERS=200
export MXFP8_LAYOUTS=crr
export MXFP8_PRESHUFFLE_QUANT=1
export MXFP8_DETERMINISM_RUNS=3
export MXFP8_SNR_THRESHOLD_DB=48.0

LABEL=70B_GateUp_CRR
M=4096; N=28672; K=8192

build_for() {
    local GATE=$1
    local FLAGS=""
    if [[ "$GATE" == "1" ]]; then
        FLAGS="-DMXFP8_CRR_BRANCHLESS_SHIFT=1"
    fi
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $FLAGS" \
        > /tmp/r54i_quick_build_${GATE}.log 2>&1
    if ! ls tk_mxfp8_layouts*.so >/dev/null 2>&1; then
        echo "BUILD FAILED gate=$GATE"
        tail -20 /tmp/r54i_quick_build_${GATE}.log
        exit 1
    fi
}

run_5x() {
    local GATE=$1
    local LOG="$OUTDIR/${LABEL}_gate${GATE}.log"
    : > "$LOG"
    for run in 1 2 3 4 5; do
        echo "  run $run/5 (gate=$GATE) ..."
        timeout 240 python3 test_mxfp8_python.py $M $N $K >> "$LOG" 2>&1 || echo "  (timeout)" >> "$LOG"
        echo "----- end run $run -----" >> "$LOG"
        if [[ $run -lt 5 ]]; then sleep 30; fi
    done
}

for GATE in 0 1; do
    echo "============================="
    echo "GATE=$GATE  building $LABEL ..."
    build_for $GATE
    echo "rebuild_cool 60s ..."
    sleep 60
    echo "GATE=$GATE  running 5x ..."
    run_5x $GATE
done

echo ""
echo "=========== EXTRACTED TFLOPS (5 runs) ==========="
for GATE in 0 1; do
    LOG="$OUTDIR/${LABEL}_gate${GATE}.log"
    echo "--- GATE=$GATE ---"
    grep -E "TFLOPS:" "$LOG" | head -10
    echo "--- GATE=$GATE SNR/Det ---"
    grep -E "SNR:|Determinism|PASS|FAIL" "$LOG" | head -10
done

echo ""
echo "=========== MEDIAN (rank 3 of 5) ==========="
for GATE in 0 1; do
    LOG="$OUTDIR/${LABEL}_gate${GATE}.log"
    median=$(grep -oE "TFLOPS: [0-9]+\.[0-9]+" "$LOG" | awk '{print $2}' | sort -n | awk 'NR==3 {print}')
    echo "  GATE=$GATE  median TFLOPS = $median"
done
