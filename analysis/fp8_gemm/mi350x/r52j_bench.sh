#!/bin/bash
# R52 Dev J: CRR_EXACT_B1_LDS_INSERT_AFTER per-shape sweep at 8B Gate/Up CRR
# Phase 1: INSERT_AFTER ∈ {0, 2, 3, 4(baseline), 5, 6, 8} at 4096x14336x4096 CRR
#  - 5 runs per cell, 30s cooldown between runs, 60s rebuild cooldown
#  - GPU 4 isolated, MXFP8_PRESHUFFLE_QUANT=1, MXFP8_LAYOUTS=crr
#  - Capture build remarks (VGPRs, AGPRs, ScratchSize, Spill, Occupancy)
# Phase 2 (only if Phase 1 finds candidate ≥+1.5%): on 70B Gate/Up (4096 28672 8192)
set -euo pipefail
cd "$(dirname "$0")"
# Worktree-local THUNDERKITTENS_ROOT so includes resolve to THIS tree
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"
export ROCM_PATH=/opt/rocm

GPU=${HIP_VISIBLE_DEVICES:-4}
WARMUP=${WARMUP:-100}
ITERS=${ITERS:-200}
RUNS=${RUNS:-5}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}

OUTDIR="r52j_results"
mkdir -p "$OUTDIR"

# Shape list. First entry = Phase 1 (8B Gate/Up CRR — known-stable canary).
declare -a SHAPES=(
    "4096 14336 4096|8B_GateUp"
    "4096 28672 8192|70B_GateUp"
)

# INSERT_AFTER ∈ [0, 8]. Sweep brackets the default (4) plus extremes.
declare -a IA_PHASE1=(0 2 3 4 5 6 8)

run_bench() {
    local M=$1 N=$2 K=$3 LABEL=$4 IA=$5
    local OUTBASE="$OUTDIR/${LABEL}_IA${IA}"
    for r in $(seq 1 $RUNS); do
        HIP_VISIBLE_DEVICES=$GPU \
        MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=crr MXFP8_PRESHUFFLE_QUANT=1 \
        python3 test_mxfp8_python.py $M $N $K > "${OUTBASE}_run${r}.log" 2>&1
        if [ $r -lt $RUNS ]; then sleep $COOL; fi
    done
}

run_check() {
    local M=$1 N=$2 K=$3 LABEL=$4 IA=$5
    local OUTBASE="$OUTDIR/${LABEL}_IA${IA}_check"
    HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=20 MXFP8_ITERS=20 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 \
    MXFP8_SNR_THRESHOLD_DB=45 MXFP8_LAYOUTS=crr MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py $M $N $K > "${OUTBASE}.log" 2>&1
}

build_and_capture() {
    local M=$1 N=$2 K=$3 LABEL=$4 IA=$5
    local BUILDLOG="$OUTDIR/${LABEL}_IA${IA}_build.log"
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DCRR_EXACT_B1_LDS_INSERT_AFTER=$IA" \
        > "$BUILDLOG" 2>&1
}

PHASE=${PHASE:-1}

if [ "$PHASE" = "1" ]; then
    echo "######################################################"
    echo "## R52 Dev J PHASE 1: 8B Gate/Up CRR INSERT_AFTER sweep ##"
    echo "######################################################"
    read -r SHAPE_M SHAPE_N SHAPE_K SHAPE_LABEL <<< "$(echo ${SHAPES[0]} | tr '|' ' ')"
    M=$SHAPE_M; N=$SHAPE_N; K=$SHAPE_K; L=$SHAPE_LABEL
    echo "Target: $L (${M}x${N}x${K})"
    for i in "${!IA_PHASE1[@]}"; do
        IA="${IA_PHASE1[$i]}"
        echo "=== [IA=$IA] ==="
        build_and_capture $M $N $K $L $IA
        echo "  built (IA=$IA)"
        run_bench $M $N $K $L $IA
        echo "  benched ($RUNS runs)"
        sleep $COOL
        run_check $M $N $K $L $IA
        echo "  corr/det checked"
        if [ $i -lt $((${#IA_PHASE1[@]} - 1)) ]; then sleep $REBUILD_COOL; fi
    done
    echo ""
    echo "Phase 1 complete. Inspect $OUTDIR for results."
fi

if [ "$PHASE" = "2" ]; then
    CAND_IA=${CAND_IA:-4}
    echo "######################################################"
    echo "## R52 Dev J PHASE 2: candidate IA=${CAND_IA} on 70B Gate/Up ##"
    echo "######################################################"
    read -r SHAPE_M SHAPE_N SHAPE_K SHAPE_LABEL <<< "$(echo ${SHAPES[1]} | tr '|' ' ')"
    M=$SHAPE_M; N=$SHAPE_N; K=$SHAPE_K; L=$SHAPE_LABEL
    echo "=== [$L] ${M}x${N}x${K} IA=${CAND_IA} ==="
    build_and_capture $M $N $K $L $CAND_IA
    echo "  built"
    run_bench $M $N $K $L $CAND_IA
    echo "  benched"
    sleep $COOL
    run_check $M $N $K $L $CAND_IA
    echo "  corr/det checked"
    sleep $REBUILD_COOL
    # Baseline IA=4 for direct comparison
    if [ "$CAND_IA" != "4" ]; then
        build_and_capture $M $N $K $L 4
        run_bench $M $N $K $L 4
        echo "  baseline IA=4 benched"
    fi
fi
