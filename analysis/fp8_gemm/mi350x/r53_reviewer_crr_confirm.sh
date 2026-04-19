#!/bin/bash
# R53 Reviewer: re-validate R52G CRR XCD swizzle un-ship at 70B Gate/Up CRR.
# 5-run strict-SCLK A/B: ON (HEAD, swizzle=1) vs OFF (per-shape un-ship sim, swizzle=0).
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
OUTDIR="r53_reviewer_results/crr_confirm"
mkdir -p "$OUTDIR"

M=4096; N=28672; K=8192
LABEL=70B_GateUp_CRR

run_arm() {
    local TAG=$1 SWIZ=$2
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DMXFP8_CRR_BLOCK_SWIZZLE=$SWIZ" \
        > "$OUTDIR/${LABEL}_${TAG}_build.log" 2>&1

    HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=20 MXFP8_ITERS=20 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 \
    MXFP8_SNR_THRESHOLD_DB=45 MXFP8_LAYOUTS=crr MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py $M $N $K \
        > "$OUTDIR/${LABEL}_${TAG}_check.log" 2>&1 || true

    sleep $COOL
    for r in $(seq 1 $RUNS); do
        HIP_VISIBLE_DEVICES=$GPU \
        MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=crr MXFP8_PRESHUFFLE_QUANT=1 \
        python3 test_mxfp8_python.py $M $N $K \
            > "$OUTDIR/${LABEL}_${TAG}_run${r}.log" 2>&1
        if [ $r -lt $RUNS ]; then sleep $COOL; fi
    done
}

echo "=== R53 Reviewer CRR confirmation: 70B Gate/Up CRR (GPU $GPU) ==="
echo "  --> [ON] swizzle=1 (HEAD)"
run_arm "on"  1
sleep $REBUILD_COOL
echo "  --> [OFF] swizzle=0 (un-ship sim)"
run_arm "off" 0
echo "=== DONE ==="
