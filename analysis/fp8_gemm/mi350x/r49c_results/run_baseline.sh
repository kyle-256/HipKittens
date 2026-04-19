#!/bin/bash
# R49 Dev C: empirical baseline (PERSISTENT=0) for 3 target shapes.
# 5 runs per shape per layout, 30s cooldown. GPU 5 only.
set -euo pipefail
cd "$(dirname "$0")/.."
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2
GPU=5
WARMUP=100
ITERS=200
RUNS=5
COOL=30
OUT="r49c_results"

declare -a SHAPES=("4096 4096 4096" "4096 14336 4096" "4096 4096 14336")
declare -a LABELS=("8B_QO" "8B_GateUp" "8B_Down")

for i in 0 1 2; do
  read -r M N K <<< "${SHAPES[$i]}"
  L="${LABELS[$i]}"
  echo "=== Build baseline $L ${M}x${N}x${K} ==="
  rm -f tk_mxfp8_layouts*.so
  make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" >/dev/null 2>&1
  for LAY in rcr rrr crr; do
    for r in $(seq 1 $RUNS); do
      echo "  baseline $L $LAY run $r"
      HIP_VISIBLE_DEVICES=$GPU \
        MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=$LAY MXFP8_PRESHUFFLE_QUANT=1 \
        python3 test_mxfp8_python.py $M $N $K \
        > "$OUT/baseline_${L}_${LAY}_run${r}.log" 2>&1
      [ $r -lt $RUNS ] && sleep $COOL
    done
    sleep $COOL
  done
done
echo "=== Done baseline ==="
