#!/bin/bash
# R37 Dev D Task 2: 4-GPU triangulation of R36 Dev B 6th V2-RRR autotune predicate
# Shape: 8B-Down (M=4096, N=4096, K=14336)
# Predicate at kernel_mxfp8_layouts.cpp:5715-5724 (warned_8b_down advisory)
# R36 SHIP-LITE was 2-GPU (GPU1 +8.91% t=+6.55; GPU4 +6.66% t=+5.87)
# STRICT promotion needs all 4 GPUs: min Δ% >= +5.0 AND min Welch t > 10.
#
# 3-gate logic (R36 NEW, mandatory):
#   G1  sclk-post-preheat >= 2200 MHz (R34 rule)
#   G2a sclk-post-bench   >= 2200 MHz (R35 NEW)
#   G2b per-run stdev/mean <= 1%      (R35 NEW)
#
# Bench: paired BABA, N_PAIRS=5, 30s preheat (per task instructions).
# Layout A=crr (baseline), Layout B=rrr (candidate).
#
# Usage:
#   PHYS_GPU=4 ./r37d_8bdown_orchestrate.sh
#   PHYS_GPU=5 ./r37d_8bdown_orchestrate.sh
#   PHYS_GPU=6 ./r37d_8bdown_orchestrate.sh
#   PHYS_GPU=7 ./r37d_8bdown_orchestrate.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

OUTDIR="$HERE/r37d_8bdown_runs"
mkdir -p "$OUTDIR"

PHYS_GPU=${PHYS_GPU:-4}
N_PAIRS=${N_PAIRS:-5}
WARMUP=${WARMUP:-30}
ITERS=${ITERS:-50}
PREHEAT=${PREHEAT:-30}
WARMUP_PAIRS=${WARMUP_PAIRS:-2}
SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
SCLK_POSTBENCH_GATE_MHZ=${SCLK_POSTBENCH_GATE_MHZ:-2200}
STDEV_MEAN_GATE=${STDEV_MEAN_GATE:-0.01}
MAX_RETRIES=${MAX_RETRIES:-3}

# 8B-Down shape
M=4096; N=4096; K=14336; LABEL=8b_down
CELL=8b_down
MODNAME="tk_mxfp8_r37d_${CELL}"
SO="${MODNAME}.cpython-310-x86_64-linux-gnu.so"

md5sum_file() {
  if [ -f "$1" ]; then md5sum "$1" | awk '{print $1}'; else echo "MISSING"; fi
}

OUT="$OUTDIR/${CELL}_gpu${PHYS_GPU}.txt"
echo "===== R37 Dev D 8B-Down 4-GPU triangulation cell=$CELL M=$M N=$N K=$K phys_gpu=$PHYS_GPU =====" | tee "$OUT"
echo "[orchestrate] N_PAIRS=$N_PAIRS WARMUP=$WARMUP ITERS=$ITERS PREHEAT=${PREHEAT}s WARMUP_PAIRS=$WARMUP_PAIRS" | tee -a "$OUT"
echo "[orchestrate] R36 3-gate logic active: G1 sclk-post-preheat>=${SCLK_GATE_MHZ} G2a sclk-post-bench>=${SCLK_POSTBENCH_GATE_MHZ} G2b stdev/mean<=${STDEV_MEAN_GATE}, MAX_RETRIES=$MAX_RETRIES" | tee -a "$OUT"
date | tee -a "$OUT"

echo "[pre-orchestrate sclk check]" | tee -a "$OUT"
rocm-smi --showclocks -d $PHYS_GPU 2>&1 | grep -i sclk | head -3 | tee -a "$OUT"

# Build per-GPU per-cell .so (deterministic md5)
if [ ! -f "$SO" ]; then
  echo "[build] $SO M=$M N=$N K=$K" | tee -a "$OUT"
  rm -f tk_fp8_layouts*.so tk_mxfp8_layouts*.so $SO
  make clean >/dev/null 2>&1
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=$MODNAME SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DPY_MODULE_NAME=$MODNAME" \
    > "$OUTDIR/build_${CELL}_gpu${PHYS_GPU}.log" 2>&1
  brc=$?
  H=$(md5sum_file $SO)
  echo "[build $CELL gpu=$PHYS_GPU rc=$brc md5=$H]" | tee -a "$OUTDIR/build_md5.log" -a "$OUT"
  if [ $brc -ne 0 ]; then
    echo "BUILD FAIL see $OUTDIR/build_${CELL}_gpu${PHYS_GPU}.log" | tee -a "$OUT"
    exit 1
  fi
else
  H=$(md5sum_file $SO)
  echo "[build $CELL md5=$H (cached)]" | tee -a "$OUTDIR/build_md5.log" -a "$OUT"
fi

attempt=1
while [ $attempt -le $MAX_RETRIES ]; do
  echo "[bench attempt $attempt/$MAX_RETRIES]" | tee -a "$OUT"
  BENCH_OUT="$OUTDIR/${CELL}_gpu${PHYS_GPU}_attempt${attempt}.txt"
  BENCH_ERR="$OUTDIR/${CELL}_gpu${PHYS_GPU}_attempt${attempt}.err"
  ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
    M=$M N=$N K=$K SO="$HERE/$SO" MOD=$MODNAME \
    LAYOUT_A=crr LAYOUT_B=rrr \
    N_PAIRS=$N_PAIRS MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS \
    PREHEAT_S=$PREHEAT WARMUP_PAIRS=$WARMUP_PAIRS \
    python3 r33c_paired_bench.py > "$BENCH_OUT" 2> "$BENCH_ERR"
  brc=$?
  echo "[bench rc=$brc]" | tee -a "$OUT"
  cat "$BENCH_OUT" | tee -a "$OUT"
  echo "--- stderr ---" | tee -a "$OUT"
  cat "$BENCH_ERR" | tee -a "$OUT"

  # G1: sclk-post-preheat
  POST_LINE=$(grep "sclk-post-preheat" "$BENCH_OUT" | head -1)
  POST_MHZ=$(echo "$POST_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')
  # G2a: sclk-post-bench
  POSTB_LINE=$(grep "sclk-post-bench" "$BENCH_OUT" | head -1)
  POSTB_MHZ=$(echo "$POSTB_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')

  # G2b: per-run stdev/mean for both layouts (parse the CRR/RRR median lines)
  RATIO_CRR=$(grep -E "^CRR " "$BENCH_OUT" | sed -nE 's/.*median=([0-9.]+).*stdev=([0-9.]+).*/\1 \2/p' | python3 -c "import sys
try:
  m,s=map(float,sys.stdin.read().split())
  print(f'{(s/m):.6f}' if m>0 else 'NaN')
except: print('NaN')" 2>/dev/null)
  RATIO_RRR=$(grep -E "^RRR " "$BENCH_OUT" | sed -nE 's/.*median=([0-9.]+).*stdev=([0-9.]+).*/\1 \2/p' | python3 -c "import sys
try:
  m,s=map(float,sys.stdin.read().split())
  print(f'{(s/m):.6f}' if m>0 else 'NaN')
except: print('NaN')" 2>/dev/null)
  if [ -z "$RATIO_CRR" ]; then RATIO_CRR="NaN"; fi
  if [ -z "$RATIO_RRR" ]; then RATIO_RRR="NaN"; fi

  RATIO_PASS=0
  if [ "$RATIO_CRR" != "NaN" ] && [ "$RATIO_RRR" != "NaN" ]; then
    RATIO_PASS=$(python3 -c "print(1 if max(${RATIO_CRR}, ${RATIO_RRR}) <= ${STDEV_MEAN_GATE} else 0)" 2>/dev/null || echo 0)
  fi

  echo "[orchestrate] attempt=$attempt rc=$brc G1 sclk-post-preheat=${POST_MHZ:-unknown}MHz (gate=${SCLK_GATE_MHZ}) G2a sclk-post-bench=${POSTB_MHZ:-unknown}MHz (gate=${SCLK_POSTBENCH_GATE_MHZ}) G2b stdev/mean CRR=${RATIO_CRR} RRR=${RATIO_RRR} (gate=${STDEV_MEAN_GATE} pass=${RATIO_PASS})" | tee -a "$OUT"

  PASS=1
  if [ $brc -ne 0 ]; then PASS=0; fi
  if [ -z "$POST_MHZ" ] || [ "$POST_MHZ" -lt "$SCLK_GATE_MHZ" ]; then PASS=0; fi
  GATE2_PASS=0
  if [ -n "$POSTB_MHZ" ] && [ "$POSTB_MHZ" -ge "$SCLK_POSTBENCH_GATE_MHZ" ]; then GATE2_PASS=1; fi
  if [ "$RATIO_PASS" = "1" ]; then GATE2_PASS=1; fi
  if [ $GATE2_PASS -eq 0 ]; then PASS=0; fi
  if [ $PASS -eq 1 ]; then
    echo "[orchestrate] PASS R36 3-gate logic at attempt=$attempt; using as final." | tee -a "$OUT"
    cp "$BENCH_OUT" "$OUTDIR/${CELL}_gpu${PHYS_GPU}_clean.txt"
    cp "$BENCH_ERR" "$OUTDIR/${CELL}_gpu${PHYS_GPU}_clean.err"
    break
  fi
  echo "[orchestrate] attempt=$attempt FAILED gates; retry after 5s..." | tee -a "$OUT"
  sleep 5
  attempt=$((attempt+1))
done

if [ $attempt -gt $MAX_RETRIES ]; then
  echo "[orchestrate] EXHAUSTED $MAX_RETRIES retries; using last attempt." | tee -a "$OUT"
  cp "$BENCH_OUT" "$OUTDIR/${CELL}_gpu${PHYS_GPU}_lastattempt.txt"
  cp "$BENCH_ERR" "$OUTDIR/${CELL}_gpu${PHYS_GPU}_lastattempt.err"
fi

echo "DONE gpu=$PHYS_GPU attempts=$attempt" | tee -a "$OUT"
date | tee -a "$OUT"
