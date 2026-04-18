#!/bin/bash
# R40 Reviewer Phase 2: paired bench runner for one (CELL, PHYS_GPU) combo.
# Required env:
#   CELL — one of: 8b_gateup, 8b_down, 8b_kv_b1, 70b_kv_b1
#   PHYS_GPU — physical GPU
#   N_PAIRS — paired BABA reps (default 10)
#   SO_A, MOD_A, SO_B, MOD_B for two_so paired bench
#   BENCH_KIND — one_so_layout (CRR vs RRR via single .so), or two_so (default vs B1)
#   LABEL — label for output files
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

OUTDIR="$HERE/r40_reviewer_phase2"
mkdir -p "$OUTDIR"

PHYS_GPU=${PHYS_GPU:-2}
CELL=${CELL:-8b_gateup}
N_PAIRS=${N_PAIRS:-10}
WARMUP=${WARMUP:-30}
ITERS=${ITERS:-50}
PREHEAT=${PREHEAT:-45}
WARMUP_PAIRS=${WARMUP_PAIRS:-2}
BENCH_KIND=${BENCH_KIND:-two_so}
LABEL=${LABEL:-${CELL}}

OUT="$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}.log"
echo "===== R40 Reviewer Phase 2 cell=$CELL label=$LABEL gpu=$PHYS_GPU N_PAIRS=$N_PAIRS BENCH_KIND=$BENCH_KIND =====" | tee "$OUT"
date | tee -a "$OUT"

if [ "$BENCH_KIND" = "two_so" ]; then
  : ${SO_A:?missing SO_A}; : ${MOD_A:?missing MOD_A}; : ${SO_B:?missing SO_B}; : ${MOD_B:?missing MOD_B}
  : ${M:?missing M}; : ${N:?missing N}; : ${K:?missing K}
  echo "[two_so] SO_A=$SO_A MOD_A=$MOD_A SO_B=$SO_B MOD_B=$MOD_B" | tee -a "$OUT"
  BENCH_OUT="$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}_bench.txt"
  BENCH_ERR="$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}_bench.err"
  MXFP8_DISPATCH_TRACE=1 \
  ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
    M=$M N=$N K=$K \
    SO_A="$SO_A" MOD_A="$MOD_A" \
    SO_B="$SO_B" MOD_B="$MOD_B" \
    N_PAIRS=$N_PAIRS MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS \
    PREHEAT_S=$PREHEAT WARMUP_PAIRS=$WARMUP_PAIRS \
    python3 r37_paired_bench_2so.py > "$BENCH_OUT" 2> "$BENCH_ERR"
  brc=$?
  echo "[bench rc=$brc]" | tee -a "$OUT"
  cat "$BENCH_OUT" | tee -a "$OUT"
  echo "--- stderr ---" | tee -a "$OUT"
  cat "$BENCH_ERR" | tee -a "$OUT"

elif [ "$BENCH_KIND" = "one_so_layout" ]; then
  : ${SO:?missing SO}; : ${MOD:?missing MOD}
  : ${M:?missing M}; : ${N:?missing N}; : ${K:?missing K}
  : ${LAYOUT_A:=crr}; : ${LAYOUT_B:=rrr}
  echo "[one_so_layout] SO=$SO MOD=$MOD LAYOUT_A=$LAYOUT_A LAYOUT_B=$LAYOUT_B" | tee -a "$OUT"
  BENCH_OUT="$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}_bench.txt"
  BENCH_ERR="$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}_bench.err"
  MXFP8_DISPATCH_TRACE=1 \
  ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
    M=$M N=$N K=$K SO="$SO" MOD="$MOD" \
    LAYOUT_A=$LAYOUT_A LAYOUT_B=$LAYOUT_B \
    N_PAIRS=$N_PAIRS MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS \
    PREHEAT_S=$PREHEAT WARMUP_PAIRS=$WARMUP_PAIRS \
    python3 r33c_paired_bench.py > "$BENCH_OUT" 2> "$BENCH_ERR"
  brc=$?
  echo "[bench rc=$brc]" | tee -a "$OUT"
  cat "$BENCH_OUT" | tee -a "$OUT"
  echo "--- stderr ---" | tee -a "$OUT"
  cat "$BENCH_ERR" | tee -a "$OUT"
fi

date | tee -a "$OUT"
echo "DONE cell=$CELL gpu=$PHYS_GPU"
