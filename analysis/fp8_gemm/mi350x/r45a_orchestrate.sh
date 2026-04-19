#!/bin/bash
# R45 Dev A — orchestrator for paired BABA bench of R44A M=2..16 fastpath
# vs V1-LEGACY-FALLBACK across 4 GPUs (GPU2/3/6/7).
#
# Inherits R36 NEW 3-gate retry pattern (R44 NEW rule 2 mandatory):
#   G1  sclk-post-preheat   >= SCLK_GATE_MHZ        (default 2200)
#   G2a sclk-post-bench     >= SCLK_POSTBENCH_GATE  (default 2200)
#   G2b per-run stdev/mean  <= STDEV_MEAN_GATE      (default 0.01)
#
# Paired BABA: DECODE (R44A fastpath) vs BASELINE (V1-LEGACY-FALLBACK).
# Same input data, two .so loaded simultaneously per r37_paired_bench_2so.py
# pattern. Reports DELTA_MEDIAN_PCT, Welch t, abs TFLOPS, sclk gates.
#
# Usage:
#   PHYS_GPU=2 LABEL=4x4kx4k_rcr SHAPE=4x4kx4k LAYOUT=rcr ./r45a_orchestrate.sh
#   PHYS_GPU=3 LABEL=4x4kx4k_rcr SHAPE=4x4kx4k LAYOUT=rcr ./r45a_orchestrate.sh
#   ... (parallel-fan across GPU2/3/6/7)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

OUTDIR="$HERE/r45a_runs"
mkdir -p "$OUTDIR"

PHYS_GPU=${PHYS_GPU:-2}
LABEL=${LABEL:-cell}
SHAPE=${SHAPE:?missing SHAPE (e.g. 4x4kx4k)}
LAYOUT=${LAYOUT:-rcr}
N_PAIRS=${N_PAIRS:-5}
WARMUP=${WARMUP:-30}
ITERS=${ITERS:-50}
PREHEAT=${PREHEAT:-30}
WARMUP_PAIRS=${WARMUP_PAIRS:-2}

SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
SCLK_POSTBENCH_GATE_MHZ=${SCLK_POSTBENCH_GATE_MHZ:-2200}
STDEV_MEAN_GATE=${STDEV_MEAN_GATE:-0.01}
MAX_RETRIES=${MAX_RETRIES:-3}

# Decode inferred shape from SHAPE (MxNxK strings like 4x4kx4k or 16x8kx8k)
case "$SHAPE" in
  4x4kx4k)  M=4;  N=4096; K=4096 ;;
  4x8kx8k)  M=4;  N=8192; K=8192 ;;
  8x4kx4k)  M=8;  N=4096; K=4096 ;;
  8x8kx8k)  M=8;  N=8192; K=8192 ;;
  16x4kx4k) M=16; N=4096; K=4096 ;;
  16x8kx8k) M=16; N=8192; K=8192 ;;
  *) echo "unknown SHAPE=$SHAPE" >&2; exit 1 ;;
esac

# .so files (built by r44a_build.sh)
MOD_DECODE="tk_mxfp8_r44a_decode_${SHAPE}"
MOD_BASELINE="tk_mxfp8_r44a_baseline_${SHAPE}"
SO_DECODE="$HERE/${MOD_DECODE}.cpython-310-x86_64-linux-gnu.so"
SO_BASELINE="$HERE/${MOD_BASELINE}.cpython-310-x86_64-linux-gnu.so"

if [ ! -f "$SO_DECODE" ];   then echo "missing $SO_DECODE   — run r44a_build.sh first" >&2; exit 1; fi
if [ ! -f "$SO_BASELINE" ]; then echo "missing $SO_BASELINE — run r44a_build.sh first" >&2; exit 1; fi

OUT="$OUTDIR/${LABEL}_gpu${PHYS_GPU}.log"
echo "===== R45A label=$LABEL gpu=$PHYS_GPU SHAPE=$SHAPE LAYOUT=$LAYOUT M=$M N=$N K=$K =====" | tee "$OUT"
echo "[orchestrate] N_PAIRS=$N_PAIRS WARMUP=$WARMUP ITERS=$ITERS PREHEAT=${PREHEAT}s WARMUP_PAIRS=$WARMUP_PAIRS" | tee -a "$OUT"
echo "[orchestrate] R36 3-gate: G1>=${SCLK_GATE_MHZ} G2a>=${SCLK_POSTBENCH_GATE_MHZ} G2b<=${STDEV_MEAN_GATE} MAX_RETRIES=$MAX_RETRIES" | tee -a "$OUT"
date | tee -a "$OUT"

run_attempt() {
  local attempt=$1
  BENCH_OUT="$OUTDIR/${LABEL}_gpu${PHYS_GPU}_attempt${attempt}.txt"
  BENCH_ERR="$OUTDIR/${LABEL}_gpu${PHYS_GPU}_attempt${attempt}.err"
  MXFP8_DISPATCH_TRACE=1 \
  ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
    M=$M N=$N K=$K LAYOUT=$LAYOUT \
    SO_DECODE="$SO_DECODE" MOD_DECODE="$MOD_DECODE" \
    SO_BASELINE="$SO_BASELINE" MOD_BASELINE="$MOD_BASELINE" \
    N_PAIRS=$N_PAIRS MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS \
    PREHEAT_S=$PREHEAT WARMUP_PAIRS=$WARMUP_PAIRS \
    python3 r45a_paired_bench.py > "$BENCH_OUT" 2> "$BENCH_ERR"
  brc=$?
  echo "[bench attempt=$attempt rc=$brc]" | tee -a "$OUT"
  cat "$BENCH_OUT" | tee -a "$OUT"
  echo "--- stderr (head) ---" | tee -a "$OUT"
  head -20 "$BENCH_ERR" | tee -a "$OUT"

  POST_LINE=$(grep "sclk-post-preheat" "$BENCH_OUT" | head -1)
  POST_MHZ=$(echo "$POST_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')
  POSTB_LINE=$(grep "sclk-post-bench" "$BENCH_OUT" | head -1)
  POSTB_MHZ=$(echo "$POSTB_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')

  RATIO_DEC=$(grep -E "^DECODE " "$BENCH_OUT" | sed -nE 's/.*median=([0-9.]+).*stdev=([0-9.]+).*/\1 \2/p' | python3 -c "import sys
try:
  m,s=map(float,sys.stdin.read().split())
  print(f'{(s/m):.6f}' if m>0 else 'NaN')
except: print('NaN')" 2>/dev/null)
  RATIO_BASE=$(grep -E "^BASELINE " "$BENCH_OUT" | sed -nE 's/.*median=([0-9.]+).*stdev=([0-9.]+).*/\1 \2/p' | python3 -c "import sys
try:
  m,s=map(float,sys.stdin.read().split())
  print(f'{(s/m):.6f}' if m>0 else 'NaN')
except: print('NaN')" 2>/dev/null)
  if [ -z "$RATIO_DEC" ];  then RATIO_DEC="NaN"; fi
  if [ -z "$RATIO_BASE" ]; then RATIO_BASE="NaN"; fi

  RATIO_PASS=0
  if [ "$RATIO_DEC" != "NaN" ] && [ "$RATIO_BASE" != "NaN" ]; then
    RATIO_PASS=$(python3 -c "print(1 if max(${RATIO_DEC}, ${RATIO_BASE}) <= ${STDEV_MEAN_GATE} else 0)" 2>/dev/null || echo 0)
  fi

  echo "[orchestrate] attempt=$attempt rc=$brc G1=${POST_MHZ:-unknown}MHz G2a=${POSTB_MHZ:-unknown}MHz G2b DEC=${RATIO_DEC} BASE=${RATIO_BASE} (gate=${STDEV_MEAN_GATE} pass=${RATIO_PASS})" | tee -a "$OUT"

  PASS=1
  if [ $brc -ne 0 ]; then PASS=0; fi
  if [ -z "$POST_MHZ" ] || [ "$POST_MHZ" -lt "$SCLK_GATE_MHZ" ]; then PASS=0; fi
  GATE2_PASS=0
  if [ -n "$POSTB_MHZ" ] && [ "$POSTB_MHZ" -ge "$SCLK_POSTBENCH_GATE_MHZ" ]; then GATE2_PASS=1; fi
  if [ "$RATIO_PASS" = "1" ]; then GATE2_PASS=1; fi
  if [ $GATE2_PASS -eq 0 ]; then PASS=0; fi
}

attempt=1
PASS=0
while [ $attempt -le $MAX_RETRIES ]; do
  echo "[bench attempt $attempt/$MAX_RETRIES]" | tee -a "$OUT"
  run_attempt $attempt
  if [ $PASS -eq 1 ]; then
    echo "[orchestrate] PASS R36 3-gate at attempt=$attempt; using as final." | tee -a "$OUT"
    cp "$BENCH_OUT" "$OUTDIR/${LABEL}_gpu${PHYS_GPU}_clean.txt"
    cp "$BENCH_ERR" "$OUTDIR/${LABEL}_gpu${PHYS_GPU}_clean.err"
    break
  fi
  echo "[orchestrate] attempt=$attempt FAILED gates; sleep 5..." | tee -a "$OUT"
  sleep 5
  attempt=$((attempt+1))
done

if [ $PASS -ne 1 ]; then
  echo "[orchestrate] FAIL: 3-gate retry exhausted MAX_RETRIES=$MAX_RETRIES" | tee -a "$OUT"
  exit 2
fi

date | tee -a "$OUT"
echo "DONE label=$LABEL gpu=$PHYS_GPU"
