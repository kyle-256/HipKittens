#!/usr/bin/env bash
# R53 Dev A bench driver: quick A/B between baseline (LAYOUT_V2=0) and
# treatment (LAYOUT_V2=1) on the target cell + 2 cross-validation cells.
# 3 runs/arm/cell (instead of 5) since we already have build-time bail signal
# (70 VGPR spill); this bench is to MEASURE the regression magnitude for the
# REFUTED writeup, not to validate ship.
set -euo pipefail

cd "$(dirname "$0")"

export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0
export MXFP8_LAYOUTS=crr
export MXFP8_PRESHUFFLE_QUANT=1
export MXFP8_WARMUP=100
export MXFP8_ITERS=200

mkdir -p r53a_results

SHAPES=(
    "4096 28672 8192 70B_GateUp"
    "4096 8192 8192 70B_QO"
    "8192 8192 8192 8192cube"
)

for entry in "${SHAPES[@]}"; do
    read M N K LABEL <<< "$entry"
    for ARM in baseline v2; do
        if [[ "$ARM" == "v2" ]]; then
            EXTRA="-DMXFP8_CRR_SCALE_LAYOUT_V2=1"
        else
            EXTRA=""
        fi
        echo ""
        echo "=== Build cell ${LABEL} (${M}x${N}x${K}) arm=${ARM} ==="
        rm -f tk_mxfp8_layouts*.so
        make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
            CXXFLAGS="-w -DM_DIM=${M} -DN_DIM=${N} -DK_DIM=${K} ${EXTRA}" \
            > r53a_results/build_${LABEL}_${ARM}_full.log 2>&1
        sleep 60
        for RUN in 1 2 3; do
            echo "=== Run cell=${LABEL} arm=${ARM} run=${RUN} ==="
            MXFP8_BUILD_M=${M} MXFP8_BUILD_N=${N} MXFP8_BUILD_K=${K} \
                python3 test_mxfp8_python.py ${M} ${N} ${K} \
                > r53a_results/run_${LABEL}_${ARM}_run${RUN}.log 2>&1 || true
            sleep 30
        done
    done
done

echo ""
echo "=== SUMMARY ==="
for entry in "${SHAPES[@]}"; do
    read M N K LABEL <<< "$entry"
    echo ""
    echo "--- ${LABEL} (${M}x${N}x${K}) ---"
    for ARM in baseline v2; do
        echo "  ${ARM}:"
        for RUN in 1 2 3; do
            grep -E "CRR.*TFLOPS|CRR.*Layout|TFLOPs|tflops" r53a_results/run_${LABEL}_${ARM}_run${RUN}.log 2>/dev/null | head -3 | sed "s/^/    run${RUN}: /"
        done
    done
done
