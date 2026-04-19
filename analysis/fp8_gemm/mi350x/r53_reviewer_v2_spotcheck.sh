#!/bin/bash
# R53 Reviewer: spot-check R52 Dev O V2-RRR claim on 2 cells (8B Gate/Up + 70B Q/O).
# Strict-SCLK A/B: 5 runs each side, 30s cooldown, 60s rebuild cooldown,
# WARMUP=100, ITERS=200. GPU 3.
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"
export ROCM_PATH=/opt/rocm
export PYTHONPATH="$THUNDERKITTENS_ROOT/src/common/pyutils"

GPU=3
WARMUP=100
ITERS=200
RUNS=5
COOL=30
REBUILD_COOL=60
OUTDIR="r53_reviewer_results/v2_spotcheck"
mkdir -p "$OUTDIR"

build_shape() {
    local M=$1 N=$2 K=$3 LABEL=$4
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" \
        > "$OUTDIR/${LABEL}_build.log" 2>&1
}

run_cell() {
    local LABEL=$1 M=$2 N=$3 K=$4 TAG=$5 V2_FLAG=$6
    for r in $(seq 1 $RUNS); do
        HIP_VISIBLE_DEVICES=$GPU \
        MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=rrr MXFP8_PRESHUFFLE_QUANT=1 \
        MXFP8_RRR_PRESHUFFLE_V2_RUNTIME=$V2_FLAG \
        python3 test_mxfp8_python.py $M $N $K \
            > "$OUTDIR/${LABEL}_${TAG}_run${r}.log" 2>&1
        if [ $r -lt $RUNS ]; then sleep $COOL; fi
    done
}

run_check() {
    local LABEL=$1 M=$2 N=$3 K=$4 TAG=$5 V2_FLAG=$6
    HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=20 MXFP8_ITERS=20 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 \
    MXFP8_SNR_THRESHOLD_DB=45 MXFP8_LAYOUTS=rrr MXFP8_PRESHUFFLE_QUANT=1 \
    MXFP8_RRR_PRESHUFFLE_V2_RUNTIME=$V2_FLAG \
    python3 test_mxfp8_python.py $M $N $K \
        > "$OUTDIR/${LABEL}_${TAG}_check.log" 2>&1 || true
}

run_pair() {
    local LABEL=$1 M=$2 N=$3 K=$4
    echo "=== $LABEL ($M x $N x $K) RRR ==="
    build_shape "$M" "$N" "$K" "$LABEL"
    sleep 5
    echo "  --> [V1]"
    run_check "$LABEL" "$M" "$N" "$K" "v1" "0"
    sleep $COOL
    run_cell "$LABEL" "$M" "$N" "$K" "v1" "0"
    sleep $REBUILD_COOL
    echo "  --> [V2]"
    run_check "$LABEL" "$M" "$N" "$K" "v2" "1"
    sleep $COOL
    run_cell "$LABEL" "$M" "$N" "$K" "v2" "1"
    sleep $REBUILD_COOL
}

echo "=== R53 Reviewer V2 spot-check (GPU $GPU, $RUNS runs/arm) ==="
run_pair "8B_GateUp"  4096 14336 4096
run_pair "70B_QO"     4096  8192 8192
echo "=== DONE ==="
