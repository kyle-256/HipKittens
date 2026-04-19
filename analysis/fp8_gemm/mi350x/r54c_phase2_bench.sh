#!/usr/bin/env bash
# R54 Dev C Phase 2: A/B bench surviving flags vs baseline.
# All Phase 1B-surviving flags hold V2 RRR at 254 VGPR / 0 spill / 0 scratch / occ=2.
# Bench each candidate flag against baseline on the primary cell (8B Gate/Up RRR).
# Then run promising ones across the full 5-cell baseline.
#
# Bench protocol: 5 runs/cell, MXFP8_WARMUP=100, MXFP8_ITERS=200, GPU 5,
# 30s cooldown, 60s rebuild_cool.
set -uo pipefail
cd "$(dirname "$0")"

export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"
export ROCM_PATH=/opt/rocm
GPU=${HIP_VISIBLE_DEVICES:-5}
WARMUP=${WARMUP:-100}
ITERS=${ITERS:-200}
RUNS=${RUNS:-5}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}

OUTDIR="r54c_results/phase2"
mkdir -p "$OUTDIR"

# Phase 2A: primary cell screening (8B Gate/Up RRR, 4096x14336x4096)
# Bench every flag whose ISA differed from baseline (we sampled 5 already; assume
# all others differ too since LLVM consumed them). Pick a curated subset of
# 10 most-promising to keep runtime tractable.
declare -a P2A_FLAGS=(
    "baseline|"
    "sched_max_occ|-mllvm -amdgpu-sched-strategy=max-occupancy"
    "sched_max_ilp|-mllvm -amdgpu-sched-strategy=max-ilp"
    "sched_iter_max_occ|-mllvm -amdgpu-sched-strategy=iterative-maximum-occupancy"
    "sched_metric_bias_100|-mllvm -amdgpu-schedule-metric-bias=100"
    "sched_relaxed_occ|-mllvm -amdgpu-schedule-relaxed-occupancy"
    "amdgpu_trackers|-mllvm -amdgpu-use-amdgpu-trackers"
    "loop_align_off|-mllvm -amdgpu-disable-loop-alignment"
    "loop_prefetch|-mllvm -amdgpu-loop-prefetch"
    "wave_priority|-mllvm -amdgpu-set-wave-priority"
    "max_occ_plus_trackers|-mllvm -amdgpu-sched-strategy=max-occupancy -mllvm -amdgpu-use-amdgpu-trackers"
)

# Phase 2B targets: full 5-cell baseline (run only for finalists)
# 8B Gate/Up RRR + 70B QO RRR + 8B Down RRR + 70B Down RRR + 70B Down CRR (cross-check)
declare -a P2B_CELLS=(
    "8B_GateUp_RRR|4096|14336|4096|rrr"
    "70B_QO_RRR|4096|8192|8192|rrr"
    "8B_Down_RRR|4096|4096|14336|rrr"
    "70B_Down_RRR|4096|8192|28672|rrr"
    "70B_Down_CRR|4096|8192|28672|crr"
)

build() {
    local M=$1 N=$2 K=$3 NAME=$4 FLAGSTR=$5
    rm -f tk_mxfp8_layouts*.so
    local LOG="$OUTDIR/build_${NAME}_${M}x${N}x${K}.log"
    set +e
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K ${FLAGSTR}" \
        > "$LOG" 2>&1
    local rc=$?
    set -e
    return $rc
}

bench_cell() {
    local M=$1 N=$2 K=$3 LABEL=$4 LAY=$5 NAME=$6
    local OUTBASE="$OUTDIR/${LABEL}_${LAY}_${NAME}"
    for r in $(seq 1 $RUNS); do
        if [ -f "${OUTBASE}_run${r}.log" ] && grep -q "TFLOPS:" "${OUTBASE}_run${r}.log" 2>/dev/null; then
            continue
        fi
        set +e
        HIP_VISIBLE_DEVICES=$GPU \
        MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=$LAY MXFP8_PRESHUFFLE_QUANT=1 \
        python3 test_mxfp8_python.py $M $N $K > "${OUTBASE}_run${r}.log" 2>&1
        set -e
        if [ $r -lt $RUNS ]; then sleep $COOL; fi
    done
}

PHASE=${PHASE:-2A}

if [ "$PHASE" = "2A" ]; then
    echo "===================================="
    echo "## R54 Dev C Phase 2A: 8B Gate/Up RRR flag screen"
    echo "===================================="
    M=4096; N=14336; K=4096; LABEL="8B_GateUp"; LAY="rrr"
    for entry in "${P2A_FLAGS[@]}"; do
        IFS='|' read -r NAME FLAGSTR <<< "$entry"
        echo "=== [$NAME] ==="
        if ! build $M $N $K $NAME "$FLAGSTR"; then
            echo "  BUILD FAILED — skipping"
            continue
        fi
        bench_cell $M $N $K $LABEL $LAY $NAME
        echo "  benched ($RUNS runs)"
        sleep $REBUILD_COOL
    done
fi

if [ "$PHASE" = "2B" ]; then
    # Pass winning flag in WIN_NAME / WIN_FLAGS env vars
    : "${WIN_NAME:?must set WIN_NAME}"
    : "${WIN_FLAGS:=}"
    echo "===================================="
    echo "## R54 Dev C Phase 2B: 5-cell sweep with [$WIN_NAME] = '${WIN_FLAGS}'"
    echo "===================================="
    for entry in "${P2B_CELLS[@]}"; do
        IFS='|' read -r LABEL M N K LAY <<< "$entry"
        echo "=== [$LABEL ${M}x${N}x${K} layout=$LAY] ==="
        # Baseline
        if ! build $M $N $K "${LABEL}_baseline" ""; then
            echo "  baseline build FAILED — skipping"; continue
        fi
        bench_cell $M $N $K $LABEL $LAY "baseline"
        sleep $REBUILD_COOL
        # Treatment
        if ! build $M $N $K "${LABEL}_${WIN_NAME}" "$WIN_FLAGS"; then
            echo "  treatment build FAILED — skipping"; continue
        fi
        bench_cell $M $N $K $LABEL $LAY "$WIN_NAME"
        sleep $REBUILD_COOL
    done
fi

echo ""
echo "=== Phase $PHASE complete. Inspect $OUTDIR ==="
