#!/bin/bash
# R51 Dev F: CRR_MAIN_UNROLL per-shape sweep at 8B Gate/Up CRR
# Phase 1: U ∈ {1(baseline), 2, 4, 8, 16} at 4096x14336x4096 CRR
#  - 5 runs per cell, 30s cooldown between runs, 60s rebuild cooldown
#  - GPU 2 isolated, MXFP8_PRESHUFFLE_QUANT=1, MXFP8_LAYOUTS=crr
#  - Capture build remarks (VGPRs, AGPRs, ScratchSize, Spill, Occupancy)
# Phase 2 (only if Phase 1 finds candidate): U on other 6 CRR shapes
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2/.claude/worktrees/agent-ad61c7d9
export ROCM_PATH=/opt/rocm

GPU=${HIP_VISIBLE_DEVICES:-2}
WARMUP=${WARMUP:-100}
ITERS=${ITERS:-200}
RUNS=${RUNS:-5}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}

OUTDIR="r51f_results"
mkdir -p "$OUTDIR"

# Shape list. First entry is the Phase 1 target (8B Gate/Up CRR).
declare -a SHAPES=(
    "4096 14336 4096|8B_GateUp"
    "8192 8192 8192|8192cube"
    "4096 4096 4096|8B_QO"
    "4096 4096 14336|8B_Down"
    "4096 8192 8192|70B_QO"
    "4096 28672 8192|70B_GateUp"
    "4096 8192 28672|70B_Down"
)

# U values to sweep in Phase 1. Note CRR's production default = 1 (not 0 like RRR).
# U=1 IS the baseline. Sweep U ∈ {1, 2, 4, 8, 16}.
declare -a US_PHASE1=(1 2 4 8 16)

run_bench() {
    local M=$1 N=$2 K=$3 LABEL=$4 U=$5
    local OUTBASE="$OUTDIR/${LABEL}_U${U}"
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
    local M=$1 N=$2 K=$3 LABEL=$4 U=$5
    local OUTBASE="$OUTDIR/${LABEL}_U${U}_check"
    HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=20 MXFP8_ITERS=20 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 \
    MXFP8_SNR_THRESHOLD_DB=45 MXFP8_LAYOUTS=crr MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py $M $N $K > "${OUTBASE}.log" 2>&1
}

build_and_capture() {
    local M=$1 N=$2 K=$3 LABEL=$4 U=$5
    local BUILDLOG="$OUTDIR/${LABEL}_U${U}_build.log"
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DCRR_MAIN_UNROLL=$U" \
        > "$BUILDLOG" 2>&1
}

PHASE=${PHASE:-1}

if [ "$PHASE" = "1" ]; then
    echo "######################################################"
    echo "## R51 Dev F PHASE 1: 8B Gate/Up CRR U-sweep        ##"
    echo "######################################################"
    read -r SHAPE_M SHAPE_N SHAPE_K SHAPE_LABEL <<< "$(echo ${SHAPES[0]} | tr '|' ' ')"
    M=$SHAPE_M; N=$SHAPE_N; K=$SHAPE_K; L=$SHAPE_LABEL
    echo "Target: $L (${M}x${N}x${K})"
    for i in "${!US_PHASE1[@]}"; do
        U="${US_PHASE1[$i]}"
        echo "=== [U=$U] ==="
        build_and_capture $M $N $K $L $U
        echo "  built (U=$U)"
        run_bench $M $N $K $L $U
        echo "  benched ($RUNS runs)"
        sleep $COOL
        run_check $M $N $K $L $U
        echo "  corr/det checked"
        if [ $i -lt $((${#US_PHASE1[@]} - 1)) ]; then sleep $REBUILD_COOL; fi
    done
    echo ""
    echo "Phase 1 complete. Inspect $OUTDIR for results."
fi

if [ "$PHASE" = "2" ]; then
    # Sweep candidate U on the other 6 shapes
    CAND_U=${CAND_U:-2}
    echo "######################################################"
    echo "## R51 Dev F PHASE 2: candidate U=${CAND_U} on other 6 CRR shapes ##"
    echo "######################################################"
    for i in "${!SHAPES[@]}"; do
        if [ $i -eq 0 ]; then continue; fi   # skip 8B Gate/Up (already done)
        read -r SHAPE_M SHAPE_N SHAPE_K SHAPE_LABEL <<< "$(echo ${SHAPES[$i]} | tr '|' ' ')"
        M=$SHAPE_M; N=$SHAPE_N; K=$SHAPE_K; L=$SHAPE_LABEL
        echo "=== [$L] ${M}x${N}x${K} U=${CAND_U} ==="
        build_and_capture $M $N $K $L $CAND_U
        echo "  built"
        run_bench $M $N $K $L $CAND_U
        echo "  benched"
        sleep $COOL
        run_check $M $N $K $L $CAND_U
        echo "  corr/det checked"
        sleep $REBUILD_COOL
        # Also rebuild + bench at U=1 baseline for direct comparison
        build_and_capture $M $N $K $L 1
        run_bench $M $N $K $L 1
        echo "  baseline U=1 benched"
        if [ $i -lt $((${#SHAPES[@]} - 1)) ]; then sleep $REBUILD_COOL; fi
    done
fi
