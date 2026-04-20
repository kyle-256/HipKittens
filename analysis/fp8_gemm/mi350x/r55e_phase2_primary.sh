#!/bin/bash
# R55 Dev E Phase 2 primary-cell A/B (70B Gate/Up CRR) — fast turnaround
set -euo pipefail
cd "$(dirname "$0")/r55e_workspace"
export THUNDERKITTENS_ROOT="$(cd ../../../.. && pwd)"
OUTDIR="../r55e_results/bench"
mkdir -p "$OUTDIR"

export HIP_VISIBLE_DEVICES=4
export MXFP8_WARMUP=100
export MXFP8_ITERS=200
export MXFP8_LAYOUTS=crr
export MXFP8_PRESHUFFLE_QUANT=1
export MXFP8_DETERMINISM_RUNS=3
export MXFP8_SNR_THRESHOLD_DB=48.0

M=4096; N=28672; K=8192
LABEL="70B_GateUp_CRR"

build_for() {
    local GATE=$1
    local FLAGS=""
    if [[ "$GATE" == "1" ]]; then
        FLAGS="-DMXFP8_CRR_SCALE_LDS_V2=1"
    fi
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $FLAGS" \
        > /tmp/r55e_pri_build_${GATE}.log 2>&1
}

run_5x() {
    local GATE=$1
    local LOG="$OUTDIR/${LABEL}_gate${GATE}.log"
    : > "$LOG"
    for run in 1 2 3 4 5; do
        echo "  run $run/5 ..."
        echo "----- run $run -----" >> "$LOG"
        timeout 240 python3 test_mxfp8_python.py $M $N $K >> "$LOG" 2>&1 || echo "  (timeout)" >> "$LOG"
        sleep 30
    done
}

extract_tflops() {
    grep -E "TFLOPS:" "$1" | grep -oE "TFLOPS: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | sort -n | awk 'NR==3 {print}'
}

for GATE in 0 1; do
    echo "  -- GATE=$GATE build --"
    build_for $GATE
    sleep 60
    echo "  -- GATE=$GATE 5-run bench --"
    run_5x $GATE
done

base=$(extract_tflops "$OUTDIR/${LABEL}_gate0.log")
treat=$(extract_tflops "$OUTDIR/${LABEL}_gate1.log")
delta=$(python3 -c "print(f'{(($treat - $base)/$base*100):+.2f}')" 2>/dev/null || echo "?")
echo "===== PRIMARY CELL RESULT ====="
echo "  baseline median = $base TFLOPS"
echo "  treatment median = $treat TFLOPS"
echo "  delta = $delta %"
