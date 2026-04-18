#!/bin/bash
# R31 Reviewer Phase 1: 4-GPU triangulation for 70B KV V2-CRR (M=4096 N=1024 K=8192)
# Goal: establish cross-GPU dispersion + new live baseline.
# Methodology: identical 5x preheat-then-bench (mirror R30 reviewer), build-cache hygiene
# (rm + md5 log), single shape only.
#
# Usage:
#   PHYS_GPU=4 ./r31_reviewer_4gpu_orchestrate.sh
#   PHYS_GPU=5 ./r31_reviewer_4gpu_orchestrate.sh
#   PHYS_GPU=6 ./r31_reviewer_4gpu_orchestrate.sh
#   PHYS_GPU=0 ./r31_reviewer_4gpu_orchestrate.sh   # last to deconflict with Dev A
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

OUTDIR="$HERE/r31_4gpu_runs"
mkdir -p "$OUTDIR"

PHYS_GPU=${PHYS_GPU:-4}
N_RUNS=${N_RUNS:-5}
WARMUP=${WARMUP:-50}
ITERS=${ITERS:-100}

# Single cell: 70B KV V2-CRR
M=4096; N=1024; K=8192; LAYOUT=crr; CELL=70b_kv_crr

md5sum_file() {
  if [ -f "$1" ]; then
    md5sum "$1" | awk '{print $1}'
  else
    echo "MISSING"
  fi
}

build_mxfp8() {
  rm -f tk_fp8_layouts*.so tk_mxfp8_layouts*.so
  make clean >/dev/null 2>&1
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" \
    > "$OUTDIR/build_mxfp8_gpu${PHYS_GPU}_${M}x${N}x${K}.log" 2>&1
  local rc=$?
  local h=$(md5sum_file tk_mxfp8_layouts*.so)
  echo "[build_mxfp8 gpu=$PHYS_GPU ${M}x${N}x${K} rc=$rc md5=$h]" | tee -a "$OUTDIR/build_md5.log"
  return $rc
}

OUT="$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}.txt"
echo "===== cell=$CELL kind=mxfp8 layout=$LAYOUT M=$M N=$N K=$K phys_gpu=$PHYS_GPU =====" | tee "$OUT"
echo "[orchestrate] phys_gpu=$PHYS_GPU N_RUNS=$N_RUNS WARMUP=$WARMUP ITERS=$ITERS" | tee -a "$OUT"
date | tee -a "$OUT"

build_mxfp8
rc=$?
if [ $rc -ne 0 ]; then
  echo "BUILD FAIL rc=$rc see $OUTDIR/build_mxfp8_gpu${PHYS_GPU}_${M}x${N}x${K}.log" | tee -a "$OUT"
  exit 1
fi

# Use ROCR_VISIBLE_DEVICES so the python sees physical GPU as device 0.
ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 \
  N_RUNS=$N_RUNS WARMUP=$WARMUP ITERS=$ITERS \
  python3 r29_reviewer_bench5x.py mxfp8 $LAYOUT $M $N $K 2>&1 | tee -a "$OUT"

echo "DONE gpu=$PHYS_GPU" | tee -a "$OUT"
date | tee -a "$OUT"
