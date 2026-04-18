#!/bin/bash
# R42 Reviewer Phase 2/3: paired bench dispatcher (one_so_layout or two_so).
# Mirrors r40_reviewer_phase2_pair.sh interface.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

OUTDIR_P2="$HERE/r42_reviewer_phase2"
OUTDIR_P3="$HERE/r42_reviewer_phase3"
mkdir -p "$OUTDIR_P2" "$OUTDIR_P3"

PHYS_GPU=${PHYS_GPU:-2}
LABEL=${LABEL:-cell}
N_PAIRS=${N_PAIRS:-10}
WARMUP=${WARMUP:-30}
ITERS=${ITERS:-50}
PREHEAT=${PREHEAT:-60}
WARMUP_PAIRS=${WARMUP_PAIRS:-2}
BENCH_KIND=${BENCH_KIND:-one_so_layout}
PHASE=${PHASE:-2}

if [ "$PHASE" = "2" ]; then OUTDIR="$OUTDIR_P2"; else OUTDIR="$OUTDIR_P3"; fi

OUT="$OUTDIR/${LABEL}_gpu${PHYS_GPU}.log"
echo "===== R42 Phase=$PHASE label=$LABEL gpu=$PHYS_GPU N_PAIRS=$N_PAIRS BENCH_KIND=$BENCH_KIND PREHEAT=$PREHEAT =====" | tee "$OUT"
date | tee -a "$OUT"

if [ "$BENCH_KIND" = "two_so" ]; then
  : ${SO_A:?missing SO_A}; : ${MOD_A:?missing MOD_A}; : ${SO_B:?missing SO_B}; : ${MOD_B:?missing MOD_B}
  : ${M:?missing M}; : ${N:?missing N}; : ${K:?missing K}
  echo "[two_so] SO_A=$SO_A MOD_A=$MOD_A SO_B=$SO_B MOD_B=$MOD_B M=$M N=$N K=$K" | tee -a "$OUT"
  BENCH_OUT="$OUTDIR/${LABEL}_gpu${PHYS_GPU}_bench.txt"
  BENCH_ERR="$OUTDIR/${LABEL}_gpu${PHYS_GPU}_bench.err"
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
  echo "--- stderr (head) ---" | tee -a "$OUT"
  head -20 "$BENCH_ERR" | tee -a "$OUT"
elif [ "$BENCH_KIND" = "one_so_layout" ]; then
  : ${SO:?missing SO}; : ${MOD:?missing MOD}
  : ${M:?missing M}; : ${N:?missing N}; : ${K:?missing K}
  : ${LAYOUT_A:=crr}; : ${LAYOUT_B:=rrr}
  echo "[one_so_layout] SO=$SO MOD=$MOD LAYOUT_A=$LAYOUT_A LAYOUT_B=$LAYOUT_B M=$M N=$N K=$K" | tee -a "$OUT"
  BENCH_OUT="$OUTDIR/${LABEL}_gpu${PHYS_GPU}_bench.txt"
  BENCH_ERR="$OUTDIR/${LABEL}_gpu${PHYS_GPU}_bench.err"
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
  echo "--- stderr (head) ---" | tee -a "$OUT"
  head -20 "$BENCH_ERR" | tee -a "$OUT"
fi

date | tee -a "$OUT"
echo "DONE label=$LABEL gpu=$PHYS_GPU"
