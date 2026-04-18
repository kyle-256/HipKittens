#!/bin/bash
# R37 Dev A: per-GPU isolated runner.
# Usage: PHYS_GPU=N WD=/tmp/r37a_wd_gpuN ./r37a_run_gpu.sh
set -u

PHYS_GPU=${PHYS_GPU:-1}
WD=${WD:-/tmp/r37a_wd_gpu$PHYS_GPU}
HERE="$(cd "$(dirname "$0")" && pwd)"
N_RUNS=${N_RUNS:-5}
WARMUP=${WARMUP:-50}
ITERS=${ITERS:-100}
SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
STDEV_RATIO_MAX=${STDEV_RATIO_MAX:-0.01}
MAX_RETRIES=${MAX_RETRIES:-3}
M=4096; N=1024; K=8192; LAYOUT=crr

mkdir -p "$WD"
cp -f "$HERE/r35_reviewer_bench5x.py" "$WD/"
OUT="$WD/orchestrate.log"
echo "===== R37 Dev A HB shrink predicate triangulation gpu=$PHYS_GPU =====" | tee "$OUT"
date | tee -a "$OUT"

extract_mhz() { echo "$1" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p'; }

run_bench() {
  local so_path=$1 tag=$2 rep=$3
  cp -f "$so_path" "$WD/tk_mxfp8_layouts.cpython-310-x86_64-linux-gnu.so"
  local md5=$(md5sum "$WD/tk_mxfp8_layouts.cpython-310-x86_64-linux-gnu.so" | awk '{print $1}')
  local attempt=1
  while [ $attempt -le $MAX_RETRIES ]; do
    BENCH_OUT="$WD/${tag}_rep${rep}_attempt${attempt}.txt"
    BENCH_ERR="$WD/${tag}_rep${rep}_attempt${attempt}.err"
    cd "$WD"
    ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
      N_RUNS=$N_RUNS WARMUP=$WARMUP ITERS=$ITERS \
      python3 r35_reviewer_bench5x.py mxfp8 $LAYOUT $M $N $K \
      > "$BENCH_OUT" 2> "$BENCH_ERR"
    local brc=$?
    POST_PRE_MHZ=$(extract_mhz "$(grep 'sclk-post-preheat' "$BENCH_ERR" | head -1)")
    POST_BENCH_MHZ=$(extract_mhz "$(grep 'sclk-post-bench' "$BENCH_ERR" | head -1)")
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
      echo "$MED_VAL" > "$WD/${tag}_rep${rep}_clean.med"
      cp "$BENCH_OUT" "$WD/${tag}_rep${rep}_clean.txt"
      cp "$BENCH_ERR" "$WD/${tag}_rep${rep}_clean.err"
      return 0
    fi
    sleep 5
    attempt=$((attempt+1))
  done
  echo "[$tag rep=$rep gpu=$PHYS_GPU EXHAUSTED retries; using last attempt med=${MED_VAL:-NA}]" | tee -a "$OUT"
  echo "$MED_VAL" > "$WD/${tag}_rep${rep}_lastattempt.med"
  return 1
}

# BABA: prod, base, prod, base, prod, base
SEQ=("prod" "base" "prod" "base" "prod" "base")
for i in "${!SEQ[@]}"; do
  case "${SEQ[$i]}" in
    prod) run_bench /tmp/r37a_prod_70bkv.so prod $i ;;
    base) run_bench /tmp/r37a_baseline_70bkv.so base $i ;;
  esac
done

echo "[done gpu=$PHYS_GPU] $(date)" | tee -a "$OUT"
