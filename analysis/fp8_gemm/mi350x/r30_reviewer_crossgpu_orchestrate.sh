#!/bin/bash
# R30 Reviewer: cross-GPU reverify of the 3 R29 negative-t cells.
# Reuses r29_reviewer_bench5x.py harness (5x preheat-then-bench, identical to R29).
# Adds R29 Dev C build-cache hygiene rule: rm -f tk_*.so + per-build md5 log.
#
# Cells flagged by R29 Reviewer (Welch t < -3, MXFP8 absolute drop, no source change since R28):
#   1. 8b_gate_crr   (M=4096 N=14336 K=4096 ): -0.63%, t=-4.11
#   2. 8b_down_crr   (M=4096 N=4096  K=14336): -2.42%, t=-3.66
#   3. 70b_kv_crr    (M=4096 N=1024  K=8192 ): -2.68%, t=-8.51 (largest)
#
# Hypothesis: GPU4 DPM/state effect; not a real source-driven regression.
# Methodology: re-run on a DIFFERENT physical GPU (try GPU5 first; fall back to GPU6).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

OUTDIR="$HERE/r30_crossgpu_runs"
mkdir -p "$OUTDIR"

# Physical GPU exposed via ROCR_VISIBLE_DEVICES outside this script.
# Inside, tk always sees device 0 via HIP_VISIBLE_DEVICES=0.
GPU=0
N_RUNS=${N_RUNS:-5}
WARMUP=${WARMUP:-50}
ITERS=${ITERS:-100}

# physical GPU label for the output (just metadata; doesn't change runtime)
PHYS_GPU=${PHYS_GPU:-unknown}

# cell_name layout M N K
CELLS=(
  "8b_gate_crr   crr 4096 14336 4096"
  "8b_down_crr   crr 4096 4096  14336"
  "70b_kv_crr    crr 4096 1024  8192"
)

md5sum_file() {
  if [ -f "$1" ]; then
    md5sum "$1" | awk '{print $1}'
  else
    echo "MISSING"
  fi
}

build_fp8() {
  local M=$1 N=$2 K=$3
  # build-cache hygiene: explicit .so removal + make clean
  rm -f tk_fp8_layouts*.so tk_mxfp8_layouts*.so
  make clean >/dev/null 2>&1
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_fp8_layouts SRC=kernel_fp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" \
    > "$OUTDIR/build_fp8_${M}x${N}x${K}.log" 2>&1
  local rc=$?
  local h=$(md5sum_file tk_fp8_layouts*.so)
  echo "[build_fp8 ${M}x${N}x${K} rc=$rc md5=$h]" | tee -a "$OUTDIR/build_md5.log"
  return $rc
}

build_mxfp8() {
  local M=$1 N=$2 K=$3
  rm -f tk_fp8_layouts*.so tk_mxfp8_layouts*.so
  make clean >/dev/null 2>&1
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" \
    > "$OUTDIR/build_mxfp8_${M}x${N}x${K}.log" 2>&1
  local rc=$?
  local h=$(md5sum_file tk_mxfp8_layouts*.so)
  echo "[build_mxfp8 ${M}x${N}x${K} rc=$rc md5=$h]" | tee -a "$OUTDIR/build_md5.log"
  return $rc
}

run_cell() {
  local kind=$1 cell=$2 layout=$3 M=$4 N=$5 K=$6
  local out="$OUTDIR/${cell}_${kind}.txt"
  echo "===== cell=$cell kind=$kind layout=$layout M=$M N=$N K=$K phys_gpu=$PHYS_GPU =====" | tee "$out"
  local rc
  if [ "$kind" = "fp8" ]; then
    build_fp8 $M $N $K; rc=$?
  else
    build_mxfp8 $M $N $K; rc=$?
  fi
  if [ $rc -ne 0 ]; then
    echo "BUILD FAIL rc=$rc see $OUTDIR/build_${kind}_${M}x${N}x${K}.log" | tee -a "$out"
    return 1
  fi
  HIP_VISIBLE_DEVICES=$GPU N_RUNS=$N_RUNS WARMUP=$WARMUP ITERS=$ITERS \
    python3 r29_reviewer_bench5x.py $kind $layout $M $N $K 2>&1 | tee -a "$out"
}

echo "[orchestrate] phys_gpu=$PHYS_GPU N_RUNS=$N_RUNS WARMUP=$WARMUP ITERS=$ITERS" | tee "$OUTDIR/orchestrate.log"
date | tee -a "$OUTDIR/orchestrate.log"

for line in "${CELLS[@]}"; do
  read -r cell layout M N K <<< "$line"
  run_cell fp8   $cell $layout $M $N $K
  run_cell mxfp8 $cell $layout $M $N $K
done

echo "ALL CELLS DONE" | tee -a "$OUTDIR/orchestrate.log"
date | tee -a "$OUTDIR/orchestrate.log"
