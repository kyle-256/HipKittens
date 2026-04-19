#!/bin/bash
# R54 Dev D Phase 1: confirm 8B Q/O wave-tail gap exists.
# Bench 4096^3 FP8 + MXFP8 V2 across RCR/RRR/CRR.
# 8B Q proj == 8B O proj (same shape M=N=K=4096).
# Strict-SCLK: 5 runs/cell, 30s cooldown, 60s rebuild_cool, WARMUP=100 ITERS=200.
# GPU 6.

set -uo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"
export ROCM_PATH=/opt/rocm

GPU=${HIP_VISIBLE_DEVICES:-6}
WARMUP=${WARMUP:-100}
ITERS=${ITERS:-200}
RUNS=${RUNS:-5}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}

OUTDIR="r54d_results/phase1_baseline_8B_QO"
mkdir -p "$OUTDIR"

LABEL="8B_QO"
M=4096; N=4096; K=4096
LAYS="rcr rrr crr"

run_one() {
    local kind=$1 LAY=$2
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
    local LAY=$1
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

echo "=== $LABEL ${M}x${N}x${K} layouts=[$LAYS] GPU=$GPU ==="
echo "=== sclk pre: $(rocm-smi --showclocks -d $GPU 2>&1 | grep sclk | head -1) ==="

# --- FP8 ---
rm -f tk_fp8_layouts*.so
BUILD_LOG="$OUTDIR/${LABEL}_fp8_build.log"
make TARGET=tk_fp8_layouts SRC=kernel_fp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" > "$BUILD_LOG" 2>&1
for LAY in $LAYS; do
    echo "  FP8 $LAY runs..."
    run_one fp8 $LAY
    sleep $COOL
done
sleep $REBUILD_COOL

# --- MXFP8 ---
rm -f tk_mxfp8_layouts*.so
BUILD_LOG="$OUTDIR/${LABEL}_mxfp8_build.log"
make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" > "$BUILD_LOG" 2>&1
for LAY in $LAYS; do
    echo "  MXFP8 check + runs $LAY..."
    run_check $LAY
    run_one mxfp8 $LAY
    sleep $COOL
done
echo "=== sclk post: $(rocm-smi --showclocks -d $GPU 2>&1 | grep sclk | head -1) ==="
echo "=== DONE ==="
