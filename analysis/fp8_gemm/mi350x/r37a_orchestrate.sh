#!/bin/bash
# R37 Dev A: HB shrink Stage B1 production predicate triangulation orchestrate
#
# Methodology:
#   - Bench 70B-KV (M=4096 N=1024 K=8192) under TWO .so:
#       (A) baseline default        — md5 b912c51d878e7621d0d92f5de239db29
#       (B) HB shrink Stage B1 prod — md5 ac1b12be9c499444ebbe988ca0bb1676
#   - Paired BABA layout: B,A,B,A,B (5 reps each, alternating)
#   - R36 NEW 3-gate logic (G1 sclk-post-preheat, G2a sclk-post-bench, G2b CV ≤ 1%)
#     gates each individual run; auto-retry up to 3x.
#
# Usage:
#   PHYS_GPU=4 ./r37a_orchestrate.sh
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

PHYS_GPU=${PHYS_GPU:-4}
N_RUNS=${N_RUNS:-5}
WARMUP=${WARMUP:-50}
ITERS=${ITERS:-100}
SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
STDEV_RATIO_MAX=${STDEV_RATIO_MAX:-0.01}
MAX_RETRIES=${MAX_RETRIES:-3}
M=4096; N=1024; K=8192; LAYOUT=crr

OUTSUBDIR=${OUTSUBDIR:-r37a_runs}
OUTDIR="$HERE/$OUTSUBDIR"
mkdir -p "$OUTDIR"

OUT="$OUTDIR/orchestrate_gpu${PHYS_GPU}.txt"
echo "===== R37 Dev A HB shrink predicate triangulation gpu=${PHYS_GPU} =====" | tee "$OUT"
echo "[orchestrate] R36 NEW 3-gate logic; N=5 reps per .so; alternating prod/base" | tee -a "$OUT"
date | tee -a "$OUT"

extract_mhz() { echo "$1" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p'; }

run_bench() {
  local so_path=$1 tag=$2 rep=$3
  cp -f "$so_path" "$HERE/tk_mxfp8_layouts.cpython-310-x86_64-linux-gnu.so"
  local md5=$(md5sum "$HERE/tk_mxfp8_layouts.cpython-310-x86_64-linux-gnu.so" | awk '{print $1}')
  local attempt=1
  while [ $attempt -le $MAX_RETRIES ]; do
    BENCH_OUT="$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_attempt${attempt}.txt"
    BENCH_ERR="$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_attempt${attempt}.err"
    ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
      N_RUNS=$N_RUNS WARMUP=$WARMUP ITERS=$ITERS \
      python3 r35_reviewer_bench5x.py mxfp8 $LAYOUT $M $N $K \
      > "$BENCH_OUT" 2> "$BENCH_ERR"
    local brc=$?
    POST_PRE_MHZ=$(extract_mhz "$(grep "sclk-post-preheat" "$BENCH_ERR" | head -1)")
    POST_BENCH_MHZ=$(extract_mhz "$(grep "sclk-post-bench" "$BENCH_ERR" | head -1)")
    SUMMARY_LINE=$(grep "^SUMMARY" "$BENCH_OUT" | head -1)
    MEAN_VAL=$(echo "$SUMMARY_LINE" | sed -nE 's/.*mean=([0-9.]+).*/\1/p')
    STDEV_VAL=$(echo "$SUMMARY_LINE" | sed -nE 's/.*stdev=([0-9.]+).*/\1/p')
    MED_VAL=$(echo "$SUMMARY_LINE" | sed -nE 's/.*median=([0-9.]+).*/\1/p')
    RATIO=""
    if [ -n "$MEAN_VAL" ] && [ -n "$STDEV_VAL" ] && [ "$MEAN_VAL" != "0" ]; then
      RATIO=$(python3 -c "print(f'{$STDEV_VAL/$MEAN_VAL:.6f}')")
    fi
    G1=0; G2A=0; G2B=0
    [ -n "$POST_PRE_MHZ" ] && [ "$POST_PRE_MHZ" -ge "$SCLK_GATE_MHZ" ] && G1=1
    [ -n "$POST_BENCH_MHZ" ] && [ "$POST_BENCH_MHZ" -ge "$SCLK_GATE_MHZ" ] && G2A=1
    [ -n "$RATIO" ] && G2B=$(python3 -c "print(1 if $RATIO <= $STDEV_RATIO_MAX else 0)")
    echo "[$tag rep=$rep att=$attempt md5=${md5:0:8} rc=$brc med=${MED_VAL:-NA} mean=${MEAN_VAL:-NA} stdev=${STDEV_VAL:-NA} cv=${RATIO:-NA} G1=$G1 G2A=$G2A G2B=$G2B preMhz=${POST_PRE_MHZ:-NA} benchMhz=${POST_BENCH_MHZ:-NA}]" | tee -a "$OUT"
    if [ $brc -eq 0 ] && [ $G1 -eq 1 ] && [ $G2A -eq 1 ] && [ $G2B -eq 1 ]; then
      echo "$MED_VAL" > "$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_clean.med"
      cp "$BENCH_OUT" "$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_clean.txt"
      cp "$BENCH_ERR" "$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_clean.err"
      return 0
    fi
    sleep 5
    attempt=$((attempt+1))
  done
  echo "[$tag rep=$rep gpu=$PHYS_GPU EXHAUSTED retries; using last attempt med=${MED_VAL:-NA}]" | tee -a "$OUT"
  echo "$MED_VAL" > "$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_lastattempt.med"
  cp "$BENCH_OUT" "$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_lastattempt.txt"
  cp "$BENCH_ERR" "$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_lastattempt.err"
  return 1
}

# BABA pattern: prod, base, prod, base, prod, base (3 prod + 3 base = 6 paired)
SEQ=("prod" "base" "prod" "base" "prod" "base")
for i in "${!SEQ[@]}"; do
  case "${SEQ[$i]}" in
    prod) run_bench /tmp/r37a_prod_70bkv.so prod $i ;;
    base) run_bench /tmp/r37a_baseline_70bkv.so base $i ;;
  esac
done

echo "[done gpu=$PHYS_GPU] $(date)" | tee -a "$OUT"
