#!/bin/bash
# R54 Reviewer 9-cell baseline matrix.
# Strict-SCLK: 5 runs/cell, 30s cooldown, 60s rebuild, WARMUP=100, ITERS=200.
# GPU 7.
#
# 3 layouts (RRR, CRR, RCR) x 3 LLaMA shapes:
#   8B Gate/Up:   M=4096 N=14336 K=4096
#   70B Q/O:      M=4096 N=8192  K=8192
#   70B Gate/Up:  M=4096 N=28672 K=8192
#
# Cells = 9.  Default macros only (no -DMXFP8_CRR_BRANCHLESS_SHIFT or any R54 dev macro).
set -uo pipefail
cd "$(dirname "$0")/../"
SRC_DIR="$(pwd)"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"
export ROCM_PATH=/opt/rocm
export PYTHONPATH="$THUNDERKITTENS_ROOT/src/common/pyutils"

GPU=${HIP_VISIBLE_DEVICES:-7}
WARMUP=${WARMUP:-100}
ITERS=${ITERS:-200}
RUNS=${RUNS:-5}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}

OUTDIR="r54_reviewer_results/baseline_9cell"
mkdir -p "$OUTDIR"

declare -A SHAPE_LAYOUTS
SHAPE_LAYOUTS["8B_GateUp|4096|14336|4096"]="rcr rrr crr"
SHAPE_LAYOUTS["70B_QO|4096|8192|8192"]="rcr rrr crr"
SHAPE_LAYOUTS["70B_GateUp|4096|28672|8192"]="rcr rrr crr"

run_one() {
    local kind=$1 M=$2 N=$3 K=$4 LABEL=$5 LAY=$6
    local OUTBASE="$OUTDIR/${LABEL}_${kind}_${LAY}"
    for r in $(seq 1 $RUNS); do
        if [ -f "${OUTBASE}_run${r}.log" ] && grep -q "TFLOPS:" "${OUTBASE}_run${r}.log" 2>/dev/null; then
            continue
        fi
        set +e
        if [ "$kind" = "fp8" ]; then
            HIP_VISIBLE_DEVICES=$GPU \
            FP8_BUILD_M=$M FP8_BUILD_N=$N FP8_BUILD_K=$K \
            FP8_WARMUP=$WARMUP FP8_ITERS=$ITERS FP8_CHECK=0 FP8_LAYOUTS=$LAY \
            python3 test_python.py $M $N $K > "${OUTBASE}_run${r}.log" 2>&1
        else
            HIP_VISIBLE_DEVICES=$GPU \
            MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
            MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
            MXFP8_LAYOUTS=$LAY MXFP8_PRESHUFFLE_QUANT=1 \
            python3 test_mxfp8_python.py $M $N $K > "${OUTBASE}_run${r}.log" 2>&1
        fi
        set -e
        if [ $r -lt $RUNS ]; then sleep $COOL; fi
    done
}

run_check() {
    local M=$1 N=$2 K=$3 LABEL=$4 LAY=$5
    local OUTBASE="$OUTDIR/${LABEL}_check_${LAY}"
    if [ -f "${OUTBASE}.log" ] && grep -q "SNR" "${OUTBASE}.log" 2>/dev/null; then return; fi
    set +e
    HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=20 MXFP8_ITERS=20 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 \
    MXFP8_SNR_THRESHOLD_DB=45 MXFP8_LAYOUTS=$LAY MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py $M $N $K > "${OUTBASE}.log" 2>&1
    set -e
}

for KEY in "${!SHAPE_LAYOUTS[@]}"; do
    IFS='|' read -r LABEL M N K <<< "$KEY"
    LAYS="${SHAPE_LAYOUTS[$KEY]}"
    echo "=== $LABEL ${M}x${N}x${K} layouts=[$LAYS] ==="

    rm -f tk_fp8_layouts*.so
    BUILD_LOG="$OUTDIR/${LABEL}_fp8_build.log"
    make TARGET=tk_fp8_layouts SRC=kernel_fp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" > "$BUILD_LOG" 2>&1
    for LAY in $LAYS; do
        run_one fp8 $M $N $K $LABEL $LAY
        sleep $COOL
    done
    sleep $REBUILD_COOL

    rm -f tk_mxfp8_layouts*.so
    BUILD_LOG="$OUTDIR/${LABEL}_mxfp8_build.log"
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" > "$BUILD_LOG" 2>&1
    for LAY in $LAYS; do
        run_check $M $N $K $LABEL $LAY
        run_one mxfp8 $M $N $K $LABEL $LAY
        sleep $COOL
    done
    sleep $REBUILD_COOL
done
echo "=== DONE ==="
