#!/bin/bash
# R49 Dev C: A/B test of MXFP8_RCR_V2_PERSISTENT (R31D's early-exit prologue +
# grid=512 = num_CUs * occ=2) at the only target shape with a structural
# wave-tail: 4096×14336×4096 (8B Gate/Up). RCR only.
#
# Hypothesis under test: at this shape with 896 tiles vs 256 CUs (3.5 waves),
# SPI redistribution at occ=2 may reduce the 14% tail. R31D demonstrated NULL
# at 4096^3 with grid=608 (using their 304-CU model); this re-tests at the
# correct CU count (256) using grid=num_CUs*2=512 and the only shape that has
# a real tail.
set -euo pipefail
cd "$(dirname "$0")/.."
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2
GPU=5
WARMUP=100
ITERS=200
RUNS=5
COOL=30
OUT="r49c_results"

M=4096; N=14336; K=4096
L="8B_GateUp"

# Baseline: PERSISTENT=0 already covered by run_baseline.sh; but rebuild
# fresh here to keep matched conditions on consecutive build/bench.
echo "=== A: baseline build (PERSISTENT=0) ==="
rm -f tk_mxfp8_layouts*.so
make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
  CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" >/dev/null 2>&1
for r in $(seq 1 $RUNS); do
  echo "  A baseline run $r"
  HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
    MXFP8_LAYOUTS=rcr MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py $M $N $K \
    > "$OUT/persist_A_baseline_${L}_rcr_run${r}.log" 2>&1
  [ $r -lt $RUNS ] && sleep $COOL
done
sleep $COOL

# B: persistent grid=512 (= 256 CUs * occ=2) early-exit
echo "=== B: persistent build (PERSISTENT=1, grid=512) ==="
rm -f tk_mxfp8_layouts*.so
make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
  CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DMXFP8_RCR_V2_PERSISTENT=1 -DMXFP8_RCR_V2_PERSISTENT_GRID=512" >/dev/null 2>&1
for r in $(seq 1 $RUNS); do
  echo "  B persistent run $r"
  HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
    MXFP8_LAYOUTS=rcr MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py $M $N $K \
    > "$OUT/persist_B_p512_${L}_rcr_run${r}.log" 2>&1
  [ $r -lt $RUNS ] && sleep $COOL
done
sleep $COOL

# B2: persistent grid=896 (= total tiles, no early-exit) — tests pure SPI
# rescheduling with no waste
echo "=== B2: persistent build (PERSISTENT=1, grid=896 = total tiles) ==="
rm -f tk_mxfp8_layouts*.so
make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
  CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DMXFP8_RCR_V2_PERSISTENT=1 -DMXFP8_RCR_V2_PERSISTENT_GRID=896" >/dev/null 2>&1
for r in $(seq 1 $RUNS); do
  echo "  B2 persistent grid=896 run $r"
  HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
    MXFP8_LAYOUTS=rcr MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py $M $N $K \
    > "$OUT/persist_B2_p896_${L}_rcr_run${r}.log" 2>&1
  [ $r -lt $RUNS ] && sleep $COOL
done

echo "=== Done persistent A/B ==="
