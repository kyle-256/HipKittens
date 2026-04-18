#!/bin/bash
# R36 Reviewer / R36 Dev D — Methodology hardening: SECOND sclk gate
# Successor to r35_reviewer_4gpu_orchestrate.sh.
#
# Implements BOTH defense-in-depth gates surfaced by R35 Reviewer:
#   GATE 1 (R34 carry-forward): sclk-post-preheat >= SCLK_GATE_MHZ (default 2200)
#   GATE 2a (R36 NEW):          sclk-post-bench   >= SCLK_GATE_MHZ (default 2200)
#   GATE 2b (R36 NEW):          per-run stdev/mean ratio (CV) <= STDEV_RATIO_MAX (default 0.01 = 1%)
#
# A run is accepted iff (gate1 AND gate2a AND gate2b) all pass.
# Otherwise auto-retry, up to MAX_RETRIES (default 3).
#
# Why both gates: R35 Reviewer GPU0 attempt-1 PASSED gate1 (post-preheat=2277 MHz)
# but mid-bench DPM dropped to 2091/2076 MHz, giving tflops_median=364.73, stdev=23.77
# (CV=6.5%). Adding gate2a alone would have caught it via post-bench MHz; adding
# gate2b alone catches the high-variance signature even without an MHz read. We
# implement both — defense in depth — and require both to pass.
#
# Methodology rules (R29-R35 closures, MUST follow):
#   1. rm -f tk_mxfp8_layouts*.so + log per-build md5 (R29 Dev C rule)
#   2. rocm-smi -d $PHYS_GPU not -d 0 (R31 Reviewer rule)
#   3. SHIP claim normalization: cross-GPU triangulation on >=2 GPUs (R32 Reviewer)
#   4. R33 sub-rule: any GPU >+1.5% above others ⇒ use min-of-GPUs not mean
#   5. R34: auto-retry up to 3x on `sclk-post-preheat < 2200 MHz` (gate 1)
#   6. R36 NEW: auto-retry on `sclk-post-bench < 2200 MHz` OR `stdev/mean > 1%`
#
# Usage:
#   PHYS_GPU=6 ./r36_reviewer_orchestrate.sh
#   PHYS_GPU=4 ./r36_reviewer_orchestrate.sh
#   ...
#
# Optional env:
#   N_RUNS              (default 5)
#   WARMUP              (default 50)
#   ITERS               (default 100)
#   SCLK_GATE_MHZ       (default 2200)  - both gate1 and gate2a use this
#   STDEV_RATIO_MAX     (default 0.01)  - gate2b CV ceiling
#   MAX_RETRIES         (default 3)
#   M / N / K           (default 4096 / 1024 / 8192 — 70B-KV V2-CRR fastpath)
#   LAYOUT              (default crr)
#   CELL                (default 70b_kv_crr)
#   OUTSUBDIR           (default r36_reviewer_runs)
#   SKIP_BUILD          (default 0; set 1 to reuse existing .so)

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

PHYS_GPU=${PHYS_GPU:-6}
N_RUNS=${N_RUNS:-5}
WARMUP=${WARMUP:-50}
ITERS=${ITERS:-100}
SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
STDEV_RATIO_MAX=${STDEV_RATIO_MAX:-0.01}
MAX_RETRIES=${MAX_RETRIES:-3}
M=${M:-4096}; N=${N:-1024}; K=${K:-8192}
LAYOUT=${LAYOUT:-crr}
CELL=${CELL:-70b_kv_crr}
OUTSUBDIR=${OUTSUBDIR:-r36_reviewer_runs}
SKIP_BUILD=${SKIP_BUILD:-0}

OUTDIR="$HERE/$OUTSUBDIR"
mkdir -p "$OUTDIR"

md5sum_file() {
  if ls $1 1>/dev/null 2>&1 ; then
    md5sum $1 2>/dev/null | head -1 | awk '{print $1}'
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
  local h=$(md5sum_file "tk_mxfp8_layouts*.so")
  echo "[build_mxfp8 gpu=$PHYS_GPU ${M}x${N}x${K} rc=$rc md5=$h]" | tee -a "$OUTDIR/build_md5.log"
  return $rc
}

OUT="$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}.txt"
echo "===== R36 cell=$CELL kind=mxfp8 layout=$LAYOUT M=$M N=$N K=$K phys_gpu=$PHYS_GPU =====" | tee "$OUT"
echo "[orchestrate] phys_gpu=$PHYS_GPU N_RUNS=$N_RUNS WARMUP=$WARMUP ITERS=$ITERS sclk_gate=${SCLK_GATE_MHZ}MHz stdev_ratio_max=${STDEV_RATIO_MAX} max_retries=$MAX_RETRIES" | tee -a "$OUT"
echo "[orchestrate] R36 GATES active: (1) sclk-post-preheat>=${SCLK_GATE_MHZ}MHz  AND  (2a) sclk-post-bench>=${SCLK_GATE_MHZ}MHz  AND  (2b) stdev/mean<=${STDEV_RATIO_MAX}" | tee -a "$OUT"
date | tee -a "$OUT"

# Confirm rocm-smi reads working GPU before starting
echo "[pre-orchestrate sclk check]" | tee -a "$OUT"
rocm-smi --showclocks -d $PHYS_GPU 2>&1 | grep -i sclk | head -3 | tee -a "$OUT"

if [ "$SKIP_BUILD" = "0" ]; then
  build_mxfp8
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "BUILD FAIL rc=$rc see $OUTDIR/build_mxfp8_gpu${PHYS_GPU}_${M}x${N}x${K}.log" | tee -a "$OUT"
    exit 1
  fi
else
  echo "[orchestrate] SKIP_BUILD=1; reusing existing .so. md5=$(md5sum_file tk_mxfp8_layouts*.so)" | tee -a "$OUT"
fi

# Helper: extract numeric MHz from a "sclk clock level: N: (XXXXMhz)" line
extract_mhz() {
  echo "$1" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p'
}

attempt=1
final_attempt=0
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

  # Gate 1: sclk-post-preheat
  POST_PRE_LINE=$(grep "sclk-post-preheat" "$BENCH_ERR" | head -1)
  POST_PRE_MHZ=$(extract_mhz "$POST_PRE_LINE")
  # Gate 2a: sclk-post-bench
  POST_BENCH_LINE=$(grep "sclk-post-bench" "$BENCH_ERR" | head -1)
  POST_BENCH_MHZ=$(extract_mhz "$POST_BENCH_LINE")
  # Gate 2b: stdev/mean ratio from SUMMARY line (mean=X stdev=Y)
  SUMMARY_LINE=$(grep "^SUMMARY" "$BENCH_OUT" | head -1)
  MEAN_VAL=$(echo "$SUMMARY_LINE" | sed -nE 's/.*mean=([0-9.]+).*/\1/p')
  STDEV_VAL=$(echo "$SUMMARY_LINE" | sed -nE 's/.*stdev=([0-9.]+).*/\1/p')

  # Compute stdev/mean ratio (CV)
  if [ -n "$MEAN_VAL" ] && [ -n "$STDEV_VAL" ] && [ "$MEAN_VAL" != "0" ]; then
    RATIO=$(python3 -c "print(f'{$STDEV_VAL/$MEAN_VAL:.6f}')")
  else
    RATIO=""
  fi

  # Decide pass/fail per gate
  G1_PASS=0; G2A_PASS=0; G2B_PASS=0
  if [ -n "$POST_PRE_MHZ" ] && [ "$POST_PRE_MHZ" -ge "$SCLK_GATE_MHZ" ]; then G1_PASS=1; fi
  if [ -n "$POST_BENCH_MHZ" ] && [ "$POST_BENCH_MHZ" -ge "$SCLK_GATE_MHZ" ]; then G2A_PASS=1; fi
  if [ -n "$RATIO" ]; then
    G2B_PASS=$(python3 -c "print(1 if $RATIO <= $STDEV_RATIO_MAX else 0)")
  fi

  echo "[orchestrate] attempt=$attempt rc=$brc | G1 sclk-post-preheat=${POST_PRE_MHZ:-NA}MHz pass=$G1_PASS | G2a sclk-post-bench=${POST_BENCH_MHZ:-NA}MHz pass=$G2A_PASS | G2b stdev/mean=${RATIO:-NA} pass=$G2B_PASS" | tee -a "$OUT"

  if [ $brc -eq 0 ] && [ "$G1_PASS" -eq 1 ] && [ "$G2A_PASS" -eq 1 ] && [ "$G2B_PASS" -eq 1 ]; then
    echo "[orchestrate] ALL GATES PASS at attempt=$attempt; using this run." | tee -a "$OUT"
    cp "$BENCH_OUT" "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_clean.txt"
    cp "$BENCH_ERR" "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_clean.err"
    final_attempt=$attempt
    break
  fi
  echo "[orchestrate] attempt=$attempt FAILED gate(s); retrying after 5s wait..." | tee -a "$OUT"
  sleep 5
  attempt=$((attempt+1))
done

if [ $attempt -gt $MAX_RETRIES ]; then
  echo "[orchestrate] EXHAUSTED $MAX_RETRIES retries; using last attempt (gates not all met)." | tee -a "$OUT"
  cp "$BENCH_OUT" "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_lastattempt.txt"
  cp "$BENCH_ERR" "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_lastattempt.err"
  final_attempt=$((attempt-1))
fi

echo "DONE gpu=$PHYS_GPU attempts=$final_attempt" | tee -a "$OUT"
date | tee -a "$OUT"
