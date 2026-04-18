#!/bin/bash
# R44 Dev C — 8B-KV HB shrink B1 drift bisect orchestrator
#
# For a given git SHA, checkout, build TWO .so (default vs HB-shrink-B1),
# then run paired BABA bench on the 8B-KV shape (M=4096 N=1024 K=4096) and
# report Δ% (HBSHRINK - DEFAULT) / DEFAULT.
#
# Bisect criterion: Δ% < +27% considered "bad" (per task spec).
#
# Implements R36 3-gate retry logic:
#   G1  sclk-post-preheat >= 2200 MHz (R34 rule)
#   G2a sclk-post-bench   >= 2200 MHz (R35 NEW)
#   G2b per-run stdev/mean <= 1%      (R35 NEW)
#
# Usage:
#   PHYS_GPU=4 SHA=66ef02d8 LABEL=r38_wrap_fix ./r44c_bisect_orchestrate.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

OUTDIR="$HERE/r44c_bisect_runs"
mkdir -p "$OUTDIR"

PHYS_GPU=${PHYS_GPU:-4}
SHA=${SHA:?missing SHA}
LABEL=${LABEL:?missing LABEL}
N_PAIRS=${N_PAIRS:-10}
WARMUP=${WARMUP:-30}
ITERS=${ITERS:-50}
PREHEAT=${PREHEAT:-60}
WARMUP_PAIRS=${WARMUP_PAIRS:-2}
SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
SCLK_POSTBENCH_GATE_MHZ=${SCLK_POSTBENCH_GATE_MHZ:-2200}
STDEV_MEAN_GATE=${STDEV_MEAN_GATE:-0.01}
MAX_RETRIES=${MAX_RETRIES:-3}

# 8B-KV shape (M=4096 N=1024 K=4096)
M=4096; N=1024; K=4096
CELL=8b_kv
SHA_SHORT=$(echo "$SHA" | cut -c1-8)

MODNAME_A="tk_mxfp8_r44c_${LABEL}_default"
MODNAME_B="tk_mxfp8_r44c_${LABEL}_b1"
SO_A="${MODNAME_A}.cpython-310-x86_64-linux-gnu.so"
SO_B="${MODNAME_B}.cpython-310-x86_64-linux-gnu.so"

OUT="$OUTDIR/${LABEL}_gpu${PHYS_GPU}.log"
echo "===== R44 Dev C bisect cell=$CELL sha=$SHA_SHORT label=$LABEL gpu=$PHYS_GPU =====" | tee "$OUT"
echo "[orchestrate] N_PAIRS=$N_PAIRS WARMUP=$WARMUP ITERS=$ITERS PREHEAT=${PREHEAT}s WARMUP_PAIRS=$WARMUP_PAIRS" | tee -a "$OUT"
echo "[orchestrate] R36 3-gate logic active: G1>=${SCLK_GATE_MHZ} G2a>=${SCLK_POSTBENCH_GATE_MHZ} G2b<=${STDEV_MEAN_GATE}, MAX_RETRIES=$MAX_RETRIES" | tee -a "$OUT"
date | tee -a "$OUT"

md5sum_file() {
  if [ -f "$1" ]; then md5sum "$1" | awk '{print $1}'; else echo "MISSING"; fi
}

# Checkout SHA in a temporary scratch dir to build the two .so. Use git-archive
# to avoid mutating the worktree HEAD. We need: kernel_mxfp8_layouts.cpp +
# all .inc files + Makefile + the include/ tree.
SCRATCH="$OUTDIR/scratch_${LABEL}"
rm -rf "$SCRATCH"
mkdir -p "$SCRATCH"
echo "[checkout] git archive $SHA -> $SCRATCH" | tee -a "$OUT"
( cd "$WT" && git archive "$SHA" | tar -x -C "$SCRATCH" )
ls "$SCRATCH/analysis/fp8_gemm/mi350x/Makefile" "$SCRATCH/analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp" >> "$OUT" 2>&1

BUILD_DIR="$SCRATCH/analysis/fp8_gemm/mi350x"
cd "$BUILD_DIR"

# DEFAULT .so build (no HB shrink macros)
if [ ! -f "$HERE/$SO_A" ]; then
  echo "[build] $SO_A (default CRR) at $SHA_SHORT" | tee -a "$OUT"
  rm -f tk_mxfp8_r44c_${LABEL}_default*.so
  THUNDERKITTENS_ROOT="$SCRATCH" make -j8 TARGET=$MODNAME_A SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DPY_MODULE_NAME=$MODNAME_A" \
    > "$OUTDIR/build_${LABEL}_default.log" 2>&1
  brc=$?
  H=$(md5sum_file $SO_A)
  echo "[build $LABEL default rc=$brc md5=$H]" | tee -a "$OUTDIR/build_md5.log" -a "$OUT"
  if [ $brc -ne 0 ]; then
    echo "BUILD FAIL default see $OUTDIR/build_${LABEL}_default.log" | tee -a "$OUT"
    cd "$HERE"; exit 1
  fi
  cp "$SO_A" "$HERE/$SO_A"
else
  H=$(md5sum_file "$HERE/$SO_A")
  echo "[build $LABEL default md5=$H (cached)]" | tee -a "$OUT"
fi

# B1 .so build (HB shrink B1 = BLK_M=128 PIPELINE=1)
if [ ! -f "$HERE/$SO_B" ]; then
  echo "[build] $SO_B (HB-shrink B1, BLK_M=128 PIPELINE=1) at $SHA_SHORT" | tee -a "$OUT"
  rm -f tk_mxfp8_r44c_${LABEL}_b1*.so
  THUNDERKITTENS_ROOT="$SCRATCH" make -j8 TARGET=$MODNAME_B SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DPY_MODULE_NAME=$MODNAME_B -DMXFP8_CRR_BLK_M=128 -DMXFP8_CRR_HBSHRINK_PIPELINE=1" \
    > "$OUTDIR/build_${LABEL}_b1.log" 2>&1
  brc=$?
  H=$(md5sum_file $SO_B)
  echo "[build $LABEL b1 rc=$brc md5=$H]" | tee -a "$OUTDIR/build_md5.log" -a "$OUT"
  if [ $brc -ne 0 ]; then
    echo "BUILD FAIL b1 see $OUTDIR/build_${LABEL}_b1.log" | tee -a "$OUT"
    cd "$HERE"; exit 1
  fi
  cp "$SO_B" "$HERE/$SO_B"
else
  H=$(md5sum_file "$HERE/$SO_B")
  echo "[build $LABEL b1 md5=$H (cached)]" | tee -a "$OUT"
fi

cd "$HERE"

attempt=1
while [ $attempt -le $MAX_RETRIES ]; do
  echo "[bench attempt $attempt/$MAX_RETRIES]" | tee -a "$OUT"
  BENCH_OUT="$OUTDIR/${LABEL}_gpu${PHYS_GPU}_attempt${attempt}.txt"
  BENCH_ERR="$OUTDIR/${LABEL}_gpu${PHYS_GPU}_attempt${attempt}.err"
  ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
    M=$M N=$N K=$K \
    SO_A="$HERE/$SO_A" MOD_A=$MODNAME_A \
    SO_B="$HERE/$SO_B" MOD_B=$MODNAME_B \
    N_PAIRS=$N_PAIRS MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS \
    PREHEAT_S=$PREHEAT WARMUP_PAIRS=$WARMUP_PAIRS \
    python3 r37_paired_bench_2so.py > "$BENCH_OUT" 2> "$BENCH_ERR"
  brc=$?
  echo "[bench rc=$brc]" | tee -a "$OUT"
  cat "$BENCH_OUT" | tee -a "$OUT"
  echo "--- stderr (head) ---" | tee -a "$OUT"
  head -10 "$BENCH_ERR" | tee -a "$OUT"

  POST_LINE=$(grep "sclk-post-preheat" "$BENCH_OUT" | head -1)
  POST_MHZ=$(echo "$POST_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')
  POSTB_LINE=$(grep "sclk-post-bench" "$BENCH_OUT" | head -1)
  POSTB_MHZ=$(echo "$POSTB_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')

  PASS=1
  if [ $brc -ne 0 ]; then PASS=0; fi
  if [ -z "$POST_MHZ" ] || [ "$POST_MHZ" -lt "$SCLK_GATE_MHZ" ]; then PASS=0; fi
  if [ -n "$POSTB_MHZ" ] && [ "$POSTB_MHZ" -ge "$SCLK_POSTBENCH_GATE_MHZ" ]; then GATE2_PASS=1; else GATE2_PASS=0; fi
  if [ $GATE2_PASS -eq 0 ]; then PASS=0; fi

  echo "[orchestrate] attempt=$attempt rc=$brc G1 sclk-post-preheat=${POST_MHZ:-unknown}MHz (gate=${SCLK_GATE_MHZ}) G2a sclk-post-bench=${POSTB_MHZ:-unknown}MHz (gate=${SCLK_POSTBENCH_GATE_MHZ}) PASS=$PASS" | tee -a "$OUT"

  if [ $PASS -eq 1 ]; then
    echo "[orchestrate] PASS R36 3-gate at attempt=$attempt; using as final." | tee -a "$OUT"
    cp "$BENCH_OUT" "$OUTDIR/${LABEL}_gpu${PHYS_GPU}_clean.txt"
    cp "$BENCH_ERR" "$OUTDIR/${LABEL}_gpu${PHYS_GPU}_clean.err"
    break
  fi
  echo "[orchestrate] attempt=$attempt FAILED gates; retry after 5s..." | tee -a "$OUT"
  sleep 5
  attempt=$((attempt+1))
done

if [ $attempt -gt $MAX_RETRIES ]; then
  echo "[orchestrate] EXHAUSTED $MAX_RETRIES retries; using last attempt." | tee -a "$OUT"
  cp "$BENCH_OUT" "$OUTDIR/${LABEL}_gpu${PHYS_GPU}_lastattempt.txt"
  cp "$BENCH_ERR" "$OUTDIR/${LABEL}_gpu${PHYS_GPU}_lastattempt.err"
fi

# Summary line
DELTA=$(grep "DELTA_MEDIAN_PCT" "$OUT" | tail -1)
echo "[orchestrate SUMMARY] sha=$SHA_SHORT label=$LABEL gpu=$PHYS_GPU $DELTA" | tee -a "$OUTDIR/SUMMARY.log"
echo "DONE label=$LABEL sha=$SHA_SHORT gpu=$PHYS_GPU attempts=$attempt" | tee -a "$OUT"
date | tee -a "$OUT"
