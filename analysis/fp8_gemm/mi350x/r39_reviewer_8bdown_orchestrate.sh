#!/bin/bash
# R39 Reviewer Phase 3b: RECONFIRM R38 Dev C STRICT promote 8B-Down V2-RRR (commit e466e582)
# Shape: 8B-Down (M=4096, N=4096, K=14336)
# R36/R37/R38 cross-cycle: +6.66% / +7.25% / +7.42%
# Use GPUs DIFFERENT from Dev C's GPU2/3/6/7 — suggested GPU0 + GPU4.
#
# Single .so, dual-layout BABA via r33c_paired_bench.py:
#   LAYOUT_A=crr (baseline)
#   LAYOUT_B=rrr (candidate — V2-RRR autotune predicate)
#
# Methodology rules (R29-R38 closures, MUST follow):
#   R34 G1 sclk-post-preheat >= 2200 MHz
#   R35 G2a sclk-post-bench >= 2200 MHz OR per-run stdev/mean <= 1%
#   R37/R38 G1' fallback: bench-MHz >= 2200 AND median-sane (post-hoc filter)
#
# Usage:
#   PHYS_GPU=0 ./r39_reviewer_8bdown_orchestrate.sh
#   PHYS_GPU=4 ./r39_reviewer_8bdown_orchestrate.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

OUTDIR="$HERE/r39_reviewer_phase3"
mkdir -p "$OUTDIR"

PHYS_GPU=${PHYS_GPU:-0}
N_PAIRS=${N_PAIRS:-5}
WARMUP=${WARMUP:-30}
ITERS=${ITERS:-50}
PREHEAT=${PREHEAT:-30}
WARMUP_PAIRS=${WARMUP_PAIRS:-2}
SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
SCLK_POSTBENCH_GATE_MHZ=${SCLK_POSTBENCH_GATE_MHZ:-2200}
STDEV_MEAN_GATE=${STDEV_MEAN_GATE:-0.01}
MAX_RETRIES=${MAX_RETRIES:-3}

M=4096; N=4096; K=14336; LABEL=8b_down
CELL=8b_down
MODNAME="tk_mxfp8_r39rev_${CELL}"
SO="${MODNAME}.cpython-310-x86_64-linux-gnu.so"

md5sum_file() { [ -f "$1" ] && md5sum "$1" | awk '{print $1}' || echo MISSING; }

OUT="$OUTDIR/${CELL}_gpu${PHYS_GPU}.txt"
echo "===== R39 Reviewer 8B-Down RECONFIRM cell=$CELL M=$M N=$N K=$K phys_gpu=$PHYS_GPU =====" | tee "$OUT"
echo "[orchestrate] N_PAIRS=$N_PAIRS WARMUP=$WARMUP ITERS=$ITERS PREHEAT=${PREHEAT}s WARMUP_PAIRS=$WARMUP_PAIRS" | tee -a "$OUT"
date | tee -a "$OUT"

echo "[pre-orchestrate sclk check]" | tee -a "$OUT"
rocm-smi --showclocks -d $PHYS_GPU 2>&1 | grep -i sclk | head -3 | tee -a "$OUT"

if [ ! -f "$SO" ]; then
  echo "[build] $SO M=$M N=$N K=$K" | tee -a "$OUT"
  rm -f tk_mxfp8_r39rev_${CELL}*.so
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=$MODNAME SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DPY_MODULE_NAME=$MODNAME" \
    > "$OUTDIR/build_${CELL}.log" 2>&1
  brc=$?
  H=$(md5sum_file $SO)
  echo "[build $CELL rc=$brc md5=$H]" | tee -a "$OUTDIR/build_md5.log" -a "$OUT"
  [ $brc -ne 0 ] && { echo BUILD FAIL; exit 1; }
else
  H=$(md5sum_file $SO)
  echo "[build $CELL md5=$H (cached)]" | tee -a "$OUT"
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
  cat "$BENCH_OUT" | tee -a "$OUT"

  POST_LINE=$(grep "sclk-post-preheat" "$BENCH_OUT" | head -1)
  POST_MHZ=$(echo "$POST_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')
  POSTB_LINE=$(grep "sclk-post-bench" "$BENCH_OUT" | head -1)
  POSTB_MHZ=$(echo "$POSTB_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')

  PASS=1
  [ $brc -ne 0 ] && PASS=0
  [ -z "$POST_MHZ" ] || [ "$POST_MHZ" -lt "$SCLK_GATE_MHZ" ] && PASS=0
  [ -z "$POSTB_MHZ" ] || [ "$POSTB_MHZ" -lt "$SCLK_POSTBENCH_GATE_MHZ" ] && PASS=0
  echo "[orchestrate] attempt=$attempt rc=$brc preheat=${POST_MHZ:-?} postbench=${POSTB_MHZ:-?} PASS=$PASS" | tee -a "$OUT"
  if [ $PASS -eq 1 ]; then
    echo "[orchestrate] PASS gates at attempt=$attempt" | tee -a "$OUT"
    cp "$BENCH_OUT" "$OUTDIR/${CELL}_gpu${PHYS_GPU}_clean.txt"
    cp "$BENCH_ERR" "$OUTDIR/${CELL}_gpu${PHYS_GPU}_clean.err"
    break
  fi
  echo "[orchestrate] retrying after 5s..." | tee -a "$OUT"
  sleep 5
  attempt=$((attempt+1))
done

if [ $attempt -gt $MAX_RETRIES ]; then
  echo "[orchestrate] EXHAUSTED $MAX_RETRIES retries" | tee -a "$OUT"
  cp "$BENCH_OUT" "$OUTDIR/${CELL}_gpu${PHYS_GPU}_lastattempt.txt"
fi

echo DONE gpu=$PHYS_GPU attempts=$attempt | tee -a "$OUT"
date | tee -a "$OUT"
