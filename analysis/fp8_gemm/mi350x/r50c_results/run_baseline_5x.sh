#!/bin/bash
# R50 Dev C empirical anchor: 5-run baseline at 8B QO 4096^3 RCR (and 8B QO 4096^3 RRR/CRR)
# to confirm baselines for the REFUTATION findings doc.
# Strict SCLK: 5 runs, 30s cooldown, GPU 3.
set -euo pipefail
cd "$(dirname "$0")/.."
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2/.claude/worktrees/agent-a8573652
export ROCM_PATH=/opt/rocm
GPU=3
WARMUP=100
ITERS=200
COOL=30
OUT="r50c_results"

# Build once at 4096^3
echo "=== Build baseline 4096^3 ==="
rm -f tk_mxfp8_layouts*.so
make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
  CXXFLAGS="-w -DM_DIM=4096 -DN_DIM=4096 -DK_DIM=4096" \
  > "$OUT/build_baseline_4096cube.log" 2>&1

for LAY in rcr rrr crr; do
  for r in 1 2 3 4 5; do
    echo "  baseline 8B_QO $LAY run $r $(date)"
    HIP_VISIBLE_DEVICES=$GPU \
      MXFP8_BUILD_M=4096 MXFP8_BUILD_N=4096 MXFP8_BUILD_K=4096 \
      MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
      MXFP8_LAYOUTS=$LAY MXFP8_PRESHUFFLE_QUANT=1 \
      python3 test_mxfp8_python.py 4096 4096 4096 \
      > "$OUT/baseline_8B_QO_${LAY}_run${r}.log" 2>&1
    [ $r -lt 5 ] && sleep $COOL
  done
  sleep $COOL
done

# Correctness check 1x per layout (SNR + det)
echo "=== Correctness check ==="
for LAY in rcr rrr crr; do
  HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=4096 MXFP8_BUILD_N=4096 MXFP8_BUILD_K=4096 \
    MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=1 MXFP8_DET_RUNS=3 \
    MXFP8_LAYOUTS=$LAY MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py 4096 4096 4096 \
    > "$OUT/correctness_8B_QO_${LAY}.log" 2>&1
  sleep 5
done

echo "=== Done baseline anchor $(date) ==="
