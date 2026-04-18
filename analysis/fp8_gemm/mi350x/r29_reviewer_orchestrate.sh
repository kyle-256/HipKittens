#!/bin/bash
# R29 Reviewer: per-shape build + 5x preheat-bench for both FP8 and MXFP8 V2.
# Cells exactly mirror R27 Reviewer baseline schema (10 cells) for delta-vs-R27.
# GPU4 (ROCR_VISIBLE_DEVICES=4 -> tk sees device 0 via HIP_VISIBLE_DEVICES=0).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

OUTDIR="$HERE/r29_runs"
mkdir -p "$OUTDIR"

GPU=${GPU:-0}   # which device tk sees (we use ROCR_VISIBLE_DEVICES=4 outside to expose only GPU4)
N_RUNS=${N_RUNS:-5}
WARMUP=${WARMUP:-50}
ITERS=${ITERS:-100}

# cell_name layout M N K
CELLS=(
  "8k_rcr        rcr 8192 8192 8192"
  "8k_rrr        rrr 8192 8192 8192"
  "8k_crr        crr 8192 8192 8192"
  "4k_rcr        rcr 4096 4096 4096"
  "8b_gate_crr   crr 4096 14336 4096"
  "8b_down_crr   crr 4096 4096 14336"
  "70b_qo_rcr    rcr 4096 8192 8192"
  "70b_kv_crr    crr 4096 1024 8192"
  "70b_gate_crr  crr 4096 28672 8192"
  "70b_down_crr  crr 4096 8192 28672"
)

build_fp8() {
  local M=$1 N=$2 K=$3
  make clean >/dev/null 2>&1
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_fp8_layouts SRC=kernel_fp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" \
    > "$OUTDIR/build_fp8_${M}x${N}x${K}.log" 2>&1
}

build_mxfp8() {
  local M=$1 N=$2 K=$3
  make clean >/dev/null 2>&1
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" \
    > "$OUTDIR/build_mxfp8_${M}x${N}x${K}.log" 2>&1
}

run_cell() {
  local kind=$1 cell=$2 layout=$3 M=$4 N=$5 K=$6
  local out="$OUTDIR/${cell}_${kind}.txt"
  echo "===== cell=$cell kind=$kind layout=$layout M=$M N=$N K=$K =====" | tee "$out"
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

for line in "${CELLS[@]}"; do
  read -r cell layout M N K <<< "$line"
  run_cell fp8   $cell $layout $M $N $K
  run_cell mxfp8 $cell $layout $M $N $K
done

echo "ALL CELLS DONE"
date
