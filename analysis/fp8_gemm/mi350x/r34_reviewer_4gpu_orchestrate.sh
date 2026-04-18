#!/bin/bash
# R34 Reviewer Phase 1: 4-GPU triangulation for 70B KV V2-CRR (M=4096 N=1024 K=8192)
# Goal: 4th cross-cycle data point for stochastic per-(GPU x cycle) high-regime hypothesis.
#       R31 GPU0 high (786.38), R32 GPU5 high (786.59), R33 NONE (768 cluster), R34 ???
#
# R31 paradigm correction APPLIED in this orchestrate:
#   - r33_reviewer_bench5x.py reads PHYS_GPU env var for `rocm-smi -d $PHYS_GPU`
#   - we export PHYS_GPU separately from ROCR_VISIBLE_DEVICES so the smi reading
#     reflects the working GPU's actual DPM state, not GPU0 idle.
#
# Methodology rules (R31/R32 closures, MUST follow):
#   1. rm -f tk_mxfp8_layouts*.so + log per-build md5 (R29 Dev C rule)
#   2. rocm-smi -d $PHYS_GPU not -d 0 (R31 Reviewer rule)
#   3. SHIP claim normalization: cross-GPU triangulation on >=2 GPUs (R32 Reviewer rule;
#      R31 GPU0 2.6% discount rule deprecated)
#   4. All in-process A/B: BABA pattern + 30s preheat (R31 Dev D rule)
#   5. MXFP8_*_PERSISTENT_GRID macros: build-time assert grid >= total_tiles (R31 Dev D rule)
#
# Usage:
#   PHYS_GPU=4 ./r34_reviewer_4gpu_orchestrate.sh
#   PHYS_GPU=5 ./r34_reviewer_4gpu_orchestrate.sh
#   PHYS_GPU=6 ./r34_reviewer_4gpu_orchestrate.sh
#   PHYS_GPU=0 ./r34_reviewer_4gpu_orchestrate.sh   # last to deconflict with R34 Dev A
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

OUTDIR="$HERE/r34_reviewer_4gpu_runs"
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
echo "[orchestrate] sclk readings now use phys_gpu=$PHYS_GPU (R31 Reviewer fix applied)" | tee -a "$OUT"
date | tee -a "$OUT"

# Confirm rocm-smi reads working GPU before starting
echo "[pre-orchestrate sclk check]" | tee -a "$OUT"
rocm-smi --showclocks -d $PHYS_GPU 2>&1 | grep -i sclk | head -3 | tee -a "$OUT"

build_mxfp8
rc=$?
if [ $rc -ne 0 ]; then
  echo "BUILD FAIL rc=$rc see $OUTDIR/build_mxfp8_gpu${PHYS_GPU}_${M}x${N}x${K}.log" | tee -a "$OUT"
  exit 1
fi

# ROCR_VISIBLE_DEVICES = $PHYS_GPU so HIP sees the physical GPU as logical 0.
# PHYS_GPU passed through to bench so its rocm-smi read targets the working GPU.
ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
  N_RUNS=$N_RUNS WARMUP=$WARMUP ITERS=$ITERS \
  python3 r34_reviewer_bench5x.py mxfp8 $LAYOUT $M $N $K 2>&1 | tee -a "$OUT"

echo "DONE gpu=$PHYS_GPU" | tee -a "$OUT"
date | tee -a "$OUT"
