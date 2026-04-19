#!/bin/bash
# R52 Dev O: V1 vs V2 RRR strict-SCLK A/B over 7 RRR cells.
#
# Premise correction: R52N's claim that V2-RRR is "NOT wired to dispatch by
# default" is FALSE in the production benchmark path. Both gemm_rrr_pq (V1)
# and gemm_rrr_pq_v2 (V2) are pybind-bound; test_mxfp8_python.py picks V2
# when MXFP8_RRR_PRESHUFFLE_V2_RUNTIME != "0" (default = 1, ON since R22-A
# 5x A/B benchmark passed +5.94%). The orchestrator's mandate to "wire V2"
# was based on a misread; V2 has been the default since R22-A (commit
# dabeffa0).
#
# This bench RECONFIRMS V2 vs V1 across all 7 RRR shapes under strict SCLK
# to detect any silent regression and to quantify the V2 win/loss per cell.
#
# Strict-SCLK protocol:
#   - 5 runs/cell, 30s cooldown between runs
#   - 60s rebuild cooldown between V1 and V2 sides per cell
#   - MXFP8_WARMUP=100, MXFP8_ITERS=200
#   - MXFP8_PRESHUFFLE_QUANT=1 (production path)
#   - Side A = V1 (MXFP8_RRR_PRESHUFFLE_V2_RUNTIME=0)
#   - Side B = V2 (MXFP8_RRR_PRESHUFFLE_V2_RUNTIME=1, default)
#   - HIP_VISIBLE_DEVICES=2

set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"
export ROCM_PATH=/opt/rocm
export PYTHONPATH="$THUNDERKITTENS_ROOT/src/common/pyutils"

GPU=${HIP_VISIBLE_DEVICES:-2}
WARMUP=${MXFP8_WARMUP:-100}
ITERS=${MXFP8_ITERS:-200}
RUNS=${RUNS:-5}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}
OUTDIR="r52o_results"
mkdir -p "$OUTDIR"

# Build is identical for V1 and V2 — both kernels live in the same TU; only
# the runtime gate changes which symbol is launched. We rebuild ONCE per
# shape to ensure a fresh M/N/K-specialized .so, then loop over V1/V2 sides
# without rebuilding.
build_shape() {
    local M=$1 N=$2 K=$3 LABEL=$4
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" \
        > "$OUTDIR/${LABEL}_build.log" 2>&1
}

# Args: $1=label  $2=M  $3=N  $4=K  $5=tag(v1|v2)  $6=v2_flag(0|1)
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

# Args: $1=label  $2=M  $3=N  $4=K  $5=tag(v1|v2)  $6=v2_flag(0|1)
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

# Args: $1=label  $2=M  $3=N  $4=K
run_pair() {
    local LABEL=$1 M=$2 N=$3 K=$4
    echo "=== $LABEL ($M x $N x $K) RRR ==="
    echo "  build for shape ($M,$N,$K)..."
    build_shape "$M" "$N" "$K" "$LABEL"
    sleep 5
    echo "  --> [A=V1] (RRR_PRESHUFFLE_V2_RUNTIME=0)"
    run_check "$LABEL" "$M" "$N" "$K" "v1" "0"
    sleep $COOL
    run_cell "$LABEL" "$M" "$N" "$K" "v1" "0"
    echo "      sleep ${REBUILD_COOL}s (no rebuild needed; cool only)"
    sleep $REBUILD_COOL
    echo "  --> [B=V2] (RRR_PRESHUFFLE_V2_RUNTIME=1)"
    run_check "$LABEL" "$M" "$N" "$K" "v2" "1"
    sleep $COOL
    run_cell "$LABEL" "$M" "$N" "$K" "v2" "1"
    sleep $REBUILD_COOL
}

echo "=== R52 Dev O: V1 vs V2 RRR strict-SCLK A/B over 7 cells ==="
echo "    GPU: HIP_VISIBLE_DEVICES=$GPU"
echo "    WARMUP=$WARMUP  ITERS=$ITERS  RUNS=$RUNS  COOL=${COOL}s  REBUILD_COOL=${REBUILD_COOL}s"

# 7 RRR cells
run_pair "8B_GateUp"   4096 14336 4096
run_pair "8B_QO"       4096  4096 4096
run_pair "8B_Down"     4096  4096 14336
run_pair "70B_GateUp"  4096 28672 8192
run_pair "70B_QO"      4096  8192 8192
run_pair "70B_Down"    4096  8192 28672
run_pair "Cube_8192"   8192  8192 8192

echo "=== R52O strict-SCLK A/B complete ==="
