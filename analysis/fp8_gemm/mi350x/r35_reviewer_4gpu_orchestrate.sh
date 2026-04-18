#!/bin/bash
# R35 Reviewer Phase 1: 4-GPU triangulation for 70B KV V2-CRR (M=4096 N=1024 K=8192)
# Goal: 5th cross-cycle data point for stochastic per-(GPU x cycle) high-regime hypothesis.
#       R31 GPU0 high (786.38), R32 GPU5 high (786.59), R33 NONE (768 cluster),
#       R34 GPU6 high (787.13), R35 ???
#
# Methodology rules (R29-R34 closures, MUST follow):
#   1. rm -f tk_mxfp8_layouts*.so + log per-build md5 (R29 Dev C rule)
#   2. rocm-smi -d $PHYS_GPU not -d 0 (R31 Reviewer rule)
#   3. SHIP claim normalization: cross-GPU triangulation on >=2 GPUs (R32 Reviewer rule)
#   4. R33 sub-rule: any GPU >+1.5% above others ⇒ use min-of-GPUs not mean
#   5. R34 NEW: auto-retry up to 3x on `sclk-post-preheat < 2200 MHz` (same-node DPM contention)
#
# Usage:
#   PHYS_GPU=4 ./r35_reviewer_4gpu_orchestrate.sh
#   PHYS_GPU=5 ./r35_reviewer_4gpu_orchestrate.sh
#   PHYS_GPU=6 ./r35_reviewer_4gpu_orchestrate.sh
#   PHYS_GPU=0 ./r35_reviewer_4gpu_orchestrate.sh   # last to deconflict with R35 Dev A
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

OUTDIR="$HERE/r35_reviewer_4gpu_runs"
mkdir -p "$OUTDIR"

PHYS_GPU=${PHYS_GPU:-4}
N_RUNS=${N_RUNS:-5}
WARMUP=${WARMUP:-50}
ITERS=${ITERS:-100}
SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
MAX_RETRIES=${MAX_RETRIES:-3}

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

# Returns numeric MHz from line like "GPU[4]: sclk clock level: 1: (2354Mhz)"
read_sclk_mhz() {
  local out
  out=$(rocm-smi --showclocks -d $PHYS_GPU 2>/dev/null | grep -i "sclk clock level" | head -1)
  # Extract digits before "Mhz"
  echo "$out" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p'
}

OUT="$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}.txt"
echo "===== cell=$CELL kind=mxfp8 layout=$LAYOUT M=$M N=$N K=$K phys_gpu=$PHYS_GPU =====" | tee "$OUT"
echo "[orchestrate] phys_gpu=$PHYS_GPU N_RUNS=$N_RUNS WARMUP=$WARMUP ITERS=$ITERS sclk_gate=${SCLK_GATE_MHZ}MHz max_retries=$MAX_RETRIES" | tee -a "$OUT"
echo "[orchestrate] R34 auto-retry rule active: bench retries up to ${MAX_RETRIES}x if sclk-post-preheat < ${SCLK_GATE_MHZ} MHz" | tee -a "$OUT"
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

# Bench-with-retry: run bench; capture stderr; check sclk-post-preheat MHz; retry up to MAX_RETRIES.
attempt=1
while [ $attempt -le $MAX_RETRIES ]; do
  echo "[bench attempt $attempt/$MAX_RETRIES]" | tee -a "$OUT"
  BENCH_OUT="$OUTDIR/bench_gpu${PHYS_GPU}_attempt${attempt}.txt"
  BENCH_ERR="$OUTDIR/bench_gpu${PHYS_GPU}_attempt${attempt}.err"
  ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
    N_RUNS=$N_RUNS WARMUP=$WARMUP ITERS=$ITERS \
    python3 r35_reviewer_bench5x.py mxfp8 $LAYOUT $M $N $K \
    > "$BENCH_OUT" 2> "$BENCH_ERR"
  brc=$?
  cat "$BENCH_OUT" | tee -a "$OUT"
  cat "$BENCH_ERR" | tee -a "$OUT"
  # Extract sclk-post-preheat MHz
  POST_LINE=$(grep "sclk-post-preheat" "$BENCH_ERR" | head -1)
  POST_MHZ=$(echo "$POST_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')
  echo "[orchestrate] attempt=$attempt rc=$brc sclk-post-preheat=${POST_MHZ:-unknown}MHz gate=${SCLK_GATE_MHZ}MHz" | tee -a "$OUT"
  if [ $brc -eq 0 ] && [ -n "$POST_MHZ" ] && [ "$POST_MHZ" -ge "$SCLK_GATE_MHZ" ]; then
    echo "[orchestrate] PASS sclk gate at attempt=$attempt; using this run." | tee -a "$OUT"
    cp "$BENCH_OUT" "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_clean.txt"
    cp "$BENCH_ERR" "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_clean.err"
    break
  fi
  echo "[orchestrate] attempt=$attempt FAILED gate; retrying after 5s wait..." | tee -a "$OUT"
  sleep 5
  attempt=$((attempt+1))
done

if [ $attempt -gt $MAX_RETRIES ]; then
  echo "[orchestrate] EXHAUSTED $MAX_RETRIES retries; using last attempt." | tee -a "$OUT"
  cp "$BENCH_OUT" "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_lastattempt.txt"
  cp "$BENCH_ERR" "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_lastattempt.err"
fi

echo "DONE gpu=$PHYS_GPU attempts=$attempt" | tee -a "$OUT"
date | tee -a "$OUT"
