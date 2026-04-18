#!/bin/bash
# R45 Dev D — within-GPU variance baseline (R44 NEW rule 1 enforcement, R45+ #7).
#
# Replays each of the 4 R44 Reviewer Phase 3 gold-standard cells N>=5 reps on a
# single GPU using the IDENTICAL .so artifacts the R44 Reviewer used (md5
# verified: d6319f / 625e56 / 08e7bf / 8193e3 / 2a1ec5 / 9159c5).
#
# Each rep is one r37_paired_bench_2so.py (two_so) or r33c_paired_bench.py
# (one_so_layout) invocation wrapped in R36 NEW 3-gate retry harness (G1
# sclk-post-preheat >= 2200 MHz, G2a sclk-post-bench >= 2200 MHz, MAX_RETRIES=3
# per attempt). Sleep 5s between reps.
#
# Outputs per cell: r45d_variance_runs/${LABEL}_gpu${GPU}_rep{1..N}_clean_bench.txt
# Aggregator: r45d_variance_aggregate.py reads the .txt files, reports
# mean/stdev/min/max of DELTA_MEDIAN_PCT.

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

OUTDIR="$HERE/r45d_variance_runs"
mkdir -p "$OUTDIR"

PHYS_GPU=${PHYS_GPU:-4}
LABEL=${LABEL:-cell}
N_REPS=${N_REPS:-5}
N_PAIRS=${N_PAIRS:-10}
WARMUP=${WARMUP:-30}
ITERS=${ITERS:-50}
PREHEAT=${PREHEAT:-60}
WARMUP_PAIRS=${WARMUP_PAIRS:-2}
BENCH_KIND=${BENCH_KIND:-two_so}
SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
SCLK_POSTBENCH_GATE_MHZ=${SCLK_POSTBENCH_GATE_MHZ:-2200}
MAX_RETRIES=${MAX_RETRIES:-3}

OUT="$OUTDIR/${LABEL}_gpu${PHYS_GPU}.log"
echo "===== R45D variance label=$LABEL gpu=$PHYS_GPU N_REPS=$N_REPS BENCH_KIND=$BENCH_KIND PREHEAT=$PREHEAT MAX_RETRIES=$MAX_RETRIES =====" | tee "$OUT"
date | tee -a "$OUT"

run_one_attempt() {
  local rep=$1 attempt=$2
  BENCH_OUT="$OUTDIR/${LABEL}_gpu${PHYS_GPU}_rep${rep}_att${attempt}_bench.txt"
  BENCH_ERR="$OUTDIR/${LABEL}_gpu${PHYS_GPU}_rep${rep}_att${attempt}_bench.err"
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
  echo "[gate rep=$rep att=$attempt rc=$brc G1=$G1 G2A=$G2A preMhz=${POST_MHZ:-NA} benchMhz=${POSTB_MHZ:-NA}]" | tee -a "$OUT"
  if [ $brc -eq 0 ] && [ $G1 -eq 1 ] && [ $G2A -eq 1 ]; then
    cp "$BENCH_OUT" "$OUTDIR/${LABEL}_gpu${PHYS_GPU}_rep${rep}_clean_bench.txt"
    cp "$BENCH_ERR" "$OUTDIR/${LABEL}_gpu${PHYS_GPU}_rep${rep}_clean_bench.err"
    DELTA_LINE=$(grep "DELTA_MEDIAN_PCT" "$BENCH_OUT" | head -1)
    echo "[clean rep=$rep] $DELTA_LINE" | tee -a "$OUT"
    return 0
  fi
  return 1
}

for rep in $(seq 1 $N_REPS); do
  echo "----- rep $rep / $N_REPS -----" | tee -a "$OUT"
  attempt=1
  ok=1
  while [ $attempt -le $MAX_RETRIES ]; do
    if run_one_attempt $rep $attempt; then ok=0; break; fi
    sleep 5
    attempt=$((attempt+1))
  done
  if [ $ok -ne 0 ]; then
    echo "[orchestrate] rep $rep EXHAUSTED no acceptable G1+G2a after $MAX_RETRIES retries" | tee -a "$OUT"
  fi
  sleep 5
done

date | tee -a "$OUT"
echo "DONE label=$LABEL gpu=$PHYS_GPU"
