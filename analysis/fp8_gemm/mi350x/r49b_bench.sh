#!/bin/bash
# R49 Dev B benchmark driver: MXFP8 RRR phase-split lever
# Tests baseline (PHASE_SPLIT=0) vs treatment (PHASE_SPLIT=1) across all 7 RRR shapes.
# Strict SCLK protocol per R48 Dev A: 5 runs/cell, 30s cooldown, 60s rebuild cooldown.
set -e
cd /shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2
export ROCM_PATH=/opt/rocm
export PYTHONPATH=$THUNDERKITTENS_ROOT/src/common/pyutils:$PYTHONPATH

mkdir -p r49b_results

# Shapes: M N K LABEL  -- all 7 RRR cells
SHAPES=(
  "8192 8192 8192 8192cube"
  "4096 4096 4096 8B_QO"
  "4096 14336 4096 8B_GateUp"
  "4096 4096 14336 8B_Down"
  "4096 8192 8192 70B_QO"
  "4096 28672 8192 70B_GateUp"
  "4096 8192 28672 70B_Down"
)

TAGS=(baseline phasesplit)

for tag in "${TAGS[@]}"; do
  EXTRA=""
  if [ "$tag" = "phasesplit" ]; then
    EXTRA="-DMXFP8_RRR_PHASE_SPLIT=1"
  fi
  for line in "${SHAPES[@]}"; do
    read M N K LABEL <<< "$line"
    echo "=========="
    echo ">>> tag=$tag shape=${M}x${N}x${K} ($LABEL) $(date)"
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
      CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $EXTRA" \
      > r49b_results/build_${tag}_${LABEL}.log 2>&1
    # 60s rebuild cooldown
    sleep 60
    for i in 1 2 3 4 5; do
      HIP_VISIBLE_DEVICES=4 MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=50 MXFP8_ITERS=100 MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=rrr MXFP8_PRESHUFFLE_QUANT=1 \
        python3 test_mxfp8_python.py $M $N $K 2>&1 | grep TFLOPS \
        | tee -a r49b_results/r49b_${tag}_${LABEL}.log
      sleep 30
    done
  done
done

echo "ALL DONE $(date)"
