#!/usr/bin/env bash
# R48 Dev F step 3: A-scale-first issue-order sweep on 70B Down RCR.
# Tests MXFP8_RCR_ASCALE_FIRST=0 (baseline) vs =1 (sched_barrier pins
# A-scale b128 issue strictly before B-scale b64).
set -euo pipefail
WT=/shared_nfs/kyle/test/Hipkittens2/.claude/worktrees/agent-a26c1935
cd $WT
source env.src
export THUNDERKITTENS_ROOT=$WT
cd $WT/analysis/fp8_gemm/mi350x

M=4096; N=8192; K=28672

for af in 0 1; do
  echo "=== BUILD ascale_first=$af ==="
  rm -f tk_mxfp8_layouts*.so
  make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=${M} -DN_DIM=${N} -DK_DIM=${K} -DMXFP8_RCR_ASCALE_FIRST=${af}" \
    > r48f_build_af${af}.log 2>&1 || { echo "BUILD FAILED af=$af"; tail -40 r48f_build_af${af}.log; exit 1; }
  for i in 1 2 3; do
    echo "--- bench af=$af run=$i ---"
    HIP_VISIBLE_DEVICES=4 MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
      MXFP8_WARMUP=50 MXFP8_ITERS=100 MXFP8_CHECK=0 \
      MXFP8_LAYOUTS=rcr MXFP8_PRESHUFFLE_QUANT=1 \
      python3 test_mxfp8_python.py $M $N $K 2>&1 | grep TFLOPS | tee -a r48f_70bdown_af${af}_run${i}.log
    sleep 20
  done
done
echo "DONE"
