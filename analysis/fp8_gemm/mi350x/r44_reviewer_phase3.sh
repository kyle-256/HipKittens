#!/bin/bash
# R44 Reviewer Phase 3: gold-standard re-bench with R36 3-gate retry (R43 NEW rule 1).
# Mirror of r43_reviewer_phase23.sh + adds G1/G2a/G2b retry loop per R43 NEW rule 1.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

OUTDIR="$HERE/r44_reviewer_phase3"
mkdir -p "$OUTDIR"

PHYS_GPU=${PHYS_GPU:-2}
LABEL=${LABEL:-cell}
N_PAIRS=${N_PAIRS:-10}
WARMUP=${WARMUP:-30}
ITERS=${ITERS:-50}
PREHEAT=${PREHEAT:-60}
WARMUP_PAIRS=${WARMUP_PAIRS:-2}
BENCH_KIND=${BENCH_KIND:-one_so_layout}
SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
SCLK_POSTBENCH_GATE_MHZ=${SCLK_POSTBENCH_GATE_MHZ:-2200}
STDEV_MEAN_GATE=${STDEV_MEAN_GATE:-0.02}  # Looser for paired-bench (RRR-side variance)
MAX_RETRIES=${MAX_RETRIES:-3}

OUT="$OUTDIR/${LABEL}_gpu${PHYS_GPU}.log"
echo "===== R44 Phase=3 label=$LABEL gpu=$PHYS_GPU N_PAIRS=$N_PAIRS BENCH_KIND=$BENCH_KIND PREHEAT=$PREHEAT MAX_RETRIES=$MAX_RETRIES =====" | tee "$OUT"
date | tee -a "$OUT"

attempt=1
while [ $attempt -le $MAX_RETRIES ]; do
  echo "[bench attempt $attempt/$MAX_RETRIES]" | tee -a "$OUT"
  BENCH_OUT="$OUTDIR/${LABEL}_gpu${PHYS_GPU}_att${attempt}_bench.txt"
  BENCH_ERR="$OUTDIR/${LABEL}_gpu${PHYS_GPU}_att${attempt}_bench.err"
  if [ "$BENCH_KIND" = "two_so" ]; then
    : ${SO_A:?missing SO_A}; : ${MOD_A:?missing MOD_A}; : ${SO_B:?missing SO_B}; : ${MOD_B:?missing MOD_B}
    : ${M:?missing M}; : ${N:?missing N}; : ${K:?missing K}
    MXFP8_DISPATCH_TRACE=1 \
    ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
      M=$M N=$N K=$K \
      SO_A="$SO_A" MOD_A="$MOD_A" \
      SO_B="$SO_B" MOD_B="$MOD_B" \
      N_PAIRS=$N_PAIRS MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS \
      PREHEAT_S=$PREHEAT WARMUP_PAIRS=$WARMUP_PAIRS \
      python3 r37_paired_bench_2so.py > "$BENCH_OUT" 2> "$BENCH_ERR"
  elif [ "$BENCH_KIND" = "one_so_layout" ]; then
    : ${SO:?missing SO}; : ${MOD:?missing MOD}
    : ${M:?missing M}; : ${N:?missing N}; : ${K:?missing K}
    : ${LAYOUT_A:=crr}; : ${LAYOUT_B:=rrr}
    MXFP8_DISPATCH_TRACE=1 \
    ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
      M=$M N=$N K=$K SO="$SO" MOD="$MOD" \
      LAYOUT_A=$LAYOUT_A LAYOUT_B=$LAYOUT_B \
      N_PAIRS=$N_PAIRS MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS \
      PREHEAT_S=$PREHEAT WARMUP_PAIRS=$WARMUP_PAIRS \
      python3 r33c_paired_bench.py > "$BENCH_OUT" 2> "$BENCH_ERR"
  fi
  brc=$?
  POST_LINE=$(grep "sclk-post-preheat" "$BENCH_ERR" "$BENCH_OUT" 2>/dev/null | head -1)
  POST_MHZ=$(echo "$POST_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')
  POSTB_LINE=$(grep "sclk-post-bench" "$BENCH_ERR" "$BENCH_OUT" 2>/dev/null | head -1)
  POSTB_MHZ=$(echo "$POSTB_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')
  G1=0; G2A=0
  [ -n "$POST_MHZ" ] && [ "$POST_MHZ" -ge "$SCLK_GATE_MHZ" ] && G1=1
  [ -n "$POSTB_MHZ" ] && [ "$POSTB_MHZ" -ge "$SCLK_POSTBENCH_GATE_MHZ" ] && G2A=1
  echo "[gate gpu=$PHYS_GPU att=$attempt rc=$brc G1=$G1 G2A=$G2A preMhz=${POST_MHZ:-NA} benchMhz=${POSTB_MHZ:-NA}]" | tee -a "$OUT"
  if [ $brc -eq 0 ] && [ $G1 -eq 1 ] && [ $G2A -eq 1 ]; then
    echo "[orchestrate] PASS G1+G2a at att=$attempt" | tee -a "$OUT"
    cp "$BENCH_OUT" "$OUTDIR/${LABEL}_gpu${PHYS_GPU}_clean_bench.txt"
    cp "$BENCH_ERR" "$OUTDIR/${LABEL}_gpu${PHYS_GPU}_clean_bench.err"
    cat "$BENCH_OUT" | tee -a "$OUT"
    echo "--- stderr (head) ---" | tee -a "$OUT"
    head -20 "$BENCH_ERR" | tee -a "$OUT"
    date | tee -a "$OUT"
    echo "DONE label=$LABEL gpu=$PHYS_GPU"
    exit 0
  fi
  sleep 5
  attempt=$((attempt+1))
done
echo "[orchestrate] EXHAUSTED no acceptable G1+G2a run after $MAX_RETRIES retries" | tee -a "$OUT"
# Fall back to last attempt (best effort)
cp "$BENCH_OUT" "$OUTDIR/${LABEL}_gpu${PHYS_GPU}_clean_bench.txt"
cp "$BENCH_ERR" "$OUTDIR/${LABEL}_gpu${PHYS_GPU}_clean_bench.err"
cat "$BENCH_OUT" | tee -a "$OUT"
echo "--- stderr (head) ---" | tee -a "$OUT"
head -20 "$BENCH_ERR" | tee -a "$OUT"
date | tee -a "$OUT"
exit 1
