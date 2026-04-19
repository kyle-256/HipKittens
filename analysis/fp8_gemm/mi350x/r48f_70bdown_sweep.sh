#!/usr/bin/env bash
# R48 Dev F: A-scale cachepolicy sweep on 70B Down RCR.
# Sweeps MXFP8_RCR_ASCALE_CACHEPOLICY = 0/1/2/3 (sentinel >=0 overrides
# the unified MXFP8_RCR_V2_SCALE_CACHEPOLICY for A-scale b128 loads only).
set -euo pipefail
WT=/shared_nfs/kyle/test/Hipkittens2/.claude/worktrees/agent-a26c1935
cd $WT
source env.src
export THUNDERKITTENS_ROOT=$WT
cd $WT/analysis/fp8_gemm/mi350x

M=4096; N=8192; K=28672

for p in 0 1 2 3; do
  echo "=== BUILD policy=$p ==="
  rm -f tk_mxfp8_layouts*.so
  make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=${M} -DN_DIM=${N} -DK_DIM=${K} -DMXFP8_RCR_ASCALE_CACHEPOLICY=${p}" \
    > r48f_build_p${p}.log 2>&1 || { echo "BUILD FAILED p=$p" ; cat r48f_build_p${p}.log | tail -40; exit 1; }
  for i in 1 2 3; do
    echo "--- bench p=$p run=$i ---"
    HIP_VISIBLE_DEVICES=4 MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
      MXFP8_WARMUP=50 MXFP8_ITERS=100 MXFP8_CHECK=0 \
      MXFP8_LAYOUTS=rcr MXFP8_PRESHUFFLE_QUANT=1 \
      python3 test_mxfp8_python.py $M $N $K 2>&1 | grep TFLOPS | tee -a r48f_70bdown_p${p}_run${i}.log
    sleep 20
  done
done
echo "DONE"
