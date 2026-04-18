#!/bin/bash
# R38 Reviewer Phase 1: 4-GPU triangulation for 70B KV V2-CRR (M=4096 N=1024 K=8192)
# Goal: 8th cross-cycle data point + verify R37 baseline of 786.64 TF holds.
#       R31 GPU0 high (786.38), R32 GPU5 high (786.59), R33 NONE (768 cluster),
#       R34 GPU6 high (787.13), R35 GPU6 high (789.28), R36 GPU0 high (789.66),
#       R37 GPU4 (794.88; +1.05% above other-4 median; first "no-outlier" cycle since R33).
#       R38 ???
#
# Methodology rules (R29-R35 closures, MUST follow):
#   1. rm -f tk_mxfp8_layouts*.so + log per-build md5 (R29 Dev C rule)
#   2. rocm-smi -d $PHYS_GPU not -d 0 (R31 Reviewer rule)
#   3. SHIP claim normalization: cross-GPU triangulation on >=2 GPUs (R32 Reviewer rule)
#   4. R33 sub-rule: any GPU >+1.5% above others ⇒ use min-of-GPUs not mean
#   5. R34 NEW: auto-retry up to 3x on `sclk-post-preheat < 2200 MHz` (same-node DPM contention)
#   6. R35 NEW (mandatory): SECOND sclk gate either
#        (a) sclk-post-bench >= SCLK_POSTBENCH_GATE_MHZ (default 2200), OR
#        (b) per-run stdev/mean <= STDEV_MEAN_GATE (default 0.01)
#      Catches the GPU0-attempt-1 class of mid-bench contention regressions
#      (R35 finding: post-preheat passed at 2277 MHz but mid-bench dropped to 2091/2076,
#       tflops_median = 364.73 with stdev 23.77 -- post-preheat-only gate did NOT catch it).
#
# Usage:
#   PHYS_GPU=0 ./r36_reviewer_4gpu_orchestrate.sh
#   PHYS_GPU=4 ./r36_reviewer_4gpu_orchestrate.sh
#   PHYS_GPU=5 ./r36_reviewer_4gpu_orchestrate.sh
#   PHYS_GPU=6 ./r36_reviewer_4gpu_orchestrate.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

OUTDIR="$HERE/r38_reviewer_4gpu_runs"
mkdir -p "$OUTDIR"

PHYS_GPU=${PHYS_GPU:-0}
N_RUNS=${N_RUNS:-5}
WARMUP=${WARMUP:-50}
ITERS=${ITERS:-100}
SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
SCLK_POSTBENCH_GATE_MHZ=${SCLK_POSTBENCH_GATE_MHZ:-2200}
STDEV_MEAN_GATE=${STDEV_MEAN_GATE:-0.01}
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

OUT="$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}.txt"
echo "===== cell=$CELL kind=mxfp8 layout=$LAYOUT M=$M N=$N K=$K phys_gpu=$PHYS_GPU =====" | tee "$OUT"
echo "[orchestrate] phys_gpu=$PHYS_GPU N_RUNS=$N_RUNS WARMUP=$WARMUP ITERS=$ITERS sclk_gate=${SCLK_GATE_MHZ}MHz post_bench_gate=${SCLK_POSTBENCH_GATE_MHZ}MHz stdev_mean_gate=${STDEV_MEAN_GATE} max_retries=$MAX_RETRIES" | tee -a "$OUT"
echo "[orchestrate] R34 gate: bench retries up to ${MAX_RETRIES}x if sclk-post-preheat < ${SCLK_GATE_MHZ} MHz" | tee -a "$OUT"
echo "[orchestrate] R35 NEW gate (mandatory): also retry if sclk-post-bench < ${SCLK_POSTBENCH_GATE_MHZ} MHz OR per-run stdev/mean > ${STDEV_MEAN_GATE}" | tee -a "$OUT"
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
  # R34 gate: extract sclk-post-preheat MHz
  POST_LINE=$(grep "sclk-post-preheat" "$BENCH_ERR" | head -1)
  POST_MHZ=$(echo "$POST_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')
  # R35 gate (a): extract sclk-post-bench MHz
  POSTB_LINE=$(grep "sclk-post-bench" "$BENCH_ERR" | head -1)
  POSTB_MHZ=$(echo "$POSTB_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')
  # R35 gate (b): extract stdev/mean ratio from SUMMARY line
  SUM_LINE=$(grep "^SUMMARY " "$BENCH_OUT" | head -1)
  MEAN_VAL=$(echo "$SUM_LINE" | sed -nE 's/.*mean=([0-9.]+).*/\1/p')
  STDEV_VAL=$(echo "$SUM_LINE" | sed -nE 's/.*stdev=([0-9.]+).*/\1/p')
  if [ -n "$MEAN_VAL" ] && [ -n "$STDEV_VAL" ] && [ "$MEAN_VAL" != "0" ]; then
    RATIO=$(python3 -c "print(f'{(${STDEV_VAL}/${MEAN_VAL}):.6f}')" 2>/dev/null || echo "NaN")
  else
    RATIO="NaN"
  fi
  # Check ratio against gate using python (avoid bash float)
  RATIO_PASS=0
  if [ "$RATIO" != "NaN" ]; then
    RATIO_PASS=$(python3 -c "print(1 if ${RATIO} <= ${STDEV_MEAN_GATE} else 0)" 2>/dev/null || echo 0)
  fi
  echo "[orchestrate] attempt=$attempt rc=$brc sclk-post-preheat=${POST_MHZ:-unknown}MHz (gate=${SCLK_GATE_MHZ}) sclk-post-bench=${POSTB_MHZ:-unknown}MHz (gate=${SCLK_POSTBENCH_GATE_MHZ}) stdev/mean=${RATIO} (gate=${STDEV_MEAN_GATE} pass=${RATIO_PASS})" | tee -a "$OUT"
  # Pass criteria:
  #   bench rc == 0
  #   AND sclk-post-preheat >= SCLK_GATE_MHZ (R34 gate)
  #   AND (sclk-post-bench >= SCLK_POSTBENCH_GATE_MHZ OR stdev/mean <= STDEV_MEAN_GATE)  (R35 NEW gate)
  PASS=1
  if [ $brc -ne 0 ]; then PASS=0; fi
  if [ -z "$POST_MHZ" ] || [ "$POST_MHZ" -lt "$SCLK_GATE_MHZ" ]; then PASS=0; fi
  GATE2_PASS=0
  if [ -n "$POSTB_MHZ" ] && [ "$POSTB_MHZ" -ge "$SCLK_POSTBENCH_GATE_MHZ" ]; then GATE2_PASS=1; fi
  if [ "$RATIO_PASS" = "1" ]; then GATE2_PASS=1; fi
  if [ $GATE2_PASS -eq 0 ]; then PASS=0; fi
  if [ $PASS -eq 1 ]; then
    echo "[orchestrate] PASS R34+R35 gates at attempt=$attempt; using this run." | tee -a "$OUT"
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
