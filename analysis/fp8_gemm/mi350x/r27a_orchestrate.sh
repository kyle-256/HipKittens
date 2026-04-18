#!/bin/bash
# R27 Dev A H1 orchestrator: build cp=0,1,2,3 then bench 5x for one cell.
# Usage: ./r27a_orchestrate.sh <layout> <M> <N> <K>
set -u
LAYOUT=$1; M=$2; N=$3; K=$4
LAYOUT_UP=$(echo "$LAYOUT" | tr '[:lower:]' '[:upper:]')
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"
OUT="$HERE/r27a_${LAYOUT}_${M}x${N}x${K}.txt"
: > "$OUT"
for CP in 0 1 2 3; do
  echo "===== cp=$CP $LAYOUT $M $N $K =====" | tee -a "$OUT"
  make clean >/dev/null 2>&1
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DMXFP8_${LAYOUT_UP}_V2_SCALE_CACHEPOLICY=$CP" \
    > "$HERE/build_cp${CP}_${LAYOUT}_${M}x${N}x${K}.log" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "BUILD FAIL rc=$RC" | tee -a "$OUT"
    continue
  fi
  HIP_VISIBLE_DEVICES=0 N_RUNS=5 MXFP8_WARMUP=50 MXFP8_ITERS=100 \
    python3 r27a_bench5x.py $LAYOUT $M $N $K 2>&1 | tee -a "$OUT"
done
echo "Saved $OUT"
