#!/bin/bash
# R50 Dev D Phase 1: MXFP8_RRR_SCHED_BARRIER mask sweep on 8B Gate/Up RRR
# Strict SCLK protocol: 5 runs/cell, 30s cooldown, 60s rebuild cooldown.
set -e
cd /shared_nfs/kyle/test/Hipkittens2/.claude/worktrees/agent-ac10186c/analysis/fp8_gemm/mi350x
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2
export ROCM_PATH=/opt/rocm
export PYTHONPATH=$THUNDERKITTENS_ROOT/src/common/pyutils:$PYTHONPATH

mkdir -p r50d_results

# 8B Gate/Up: M=4096 N=14336 K=4096
M=4096
N=14336
K=4096
LABEL="8B_GateUp"

# Mask values to sweep
MASKS=(0 1 2 3 4)

for mask in "${MASKS[@]}"; do
  echo "=========="
  echo ">>> mask=$mask shape=${M}x${N}x${K} ($LABEL) $(date)"
  rm -f tk_mxfp8_layouts*.so
  make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DMXFP8_RRR_SCHED_BARRIER=$mask" \
    > r50d_results/build_mask${mask}_${LABEL}.log 2>&1
  echo ">>> Build done, sleeping 60s rebuild cooldown..."
  sleep 60
  for i in 1 2 3 4 5; do
    echo ">>> mask=$mask run=$i $(date)"
    HIP_VISIBLE_DEVICES=4 MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
      MXFP8_WARMUP=100 MXFP8_ITERS=200 MXFP8_CHECK=0 \
      MXFP8_LAYOUTS=rrr MXFP8_PRESHUFFLE_QUANT=1 \
      python3 test_mxfp8_python.py $M $N $K 2>&1 | grep TFLOPS \
      | tee -a r50d_results/r50d_mask${mask}_${LABEL}.log
    sleep 30
  done
done

echo "ALL DONE $(date)"
