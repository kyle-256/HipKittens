#!/bin/bash
# R37 Reviewer Phase 2: SHIP verification of R36 still-pending items
#
# Cells:
#   c_b1   = R36 Dev A HB shrink Stage B1 SHIP @ 70B-KV (M=4096 N=1024 K=8192)
#            -- compares HB-shrink-B1 CRR vs DEFAULT CRR (paired BABA, two .so)
#   c_qo8  = R36 Dev C V2-RCR predicate @ 8B Q/O (M=4096 N=4096 K=4096)
#            -- compares CRR vs RCR (paired BABA, single .so)
#   c_qo70 = R36 Dev C V2-RCR predicate @ 70B Q/O (M=4096 N=8192 K=8192)
#            -- compares CRR vs RCR (paired BABA, single .so)
#
# Methodology rules: same R29-R36 closures (R34 sclk-post-preheat + R35 NEW
# sclk-post-bench + per-run stdev/mean -- enforced by r33c_paired_bench.py
# pre/post sclk reads + R35 post-hoc gate logged below).
#
# Usage:
#   PHYS_GPU=4 CELL=c_b1   ./r37_reviewer_ship_verify.sh   # 70B KV HB shrink
#   PHYS_GPU=5 CELL=c_qo8  ./r37_reviewer_ship_verify.sh   # 8B Q/O RCR
#   PHYS_GPU=6 CELL=c_qo70 ./r37_reviewer_ship_verify.sh   # 70B Q/O RCR
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

OUTDIR="$HERE/r37_reviewer_devXXX_verify"
mkdir -p "$OUTDIR"

PHYS_GPU=${PHYS_GPU:-4}
CELL=${CELL:-c_b1}
N_PAIRS=${N_PAIRS:-5}
WARMUP=${WARMUP:-30}
ITERS=${ITERS:-50}
PREHEAT=${PREHEAT:-45}
WARMUP_PAIRS=${WARMUP_PAIRS:-2}

case "$CELL" in
  c_b1)
    M=4096; N=1024; K=8192; LABEL=70b_kv_b1
    LAYOUT_A=crr; LAYOUT_B=crr  # both crr -- candidate is HB-shrink build
    MODE=two_so
    ;;
  c_qo8)
    M=4096; N=4096; K=4096; LABEL=8b_qo
    LAYOUT_A=crr; LAYOUT_B=rcr
    MODE=single_so
    ;;
  c_qo70)
    M=4096; N=8192; K=8192; LABEL=70b_qo
    LAYOUT_A=crr; LAYOUT_B=rcr
    MODE=single_so
    ;;
  *)
    echo "Unknown CELL=$CELL; use c_b1, c_qo8 or c_qo70" >&2
    exit 1
    ;;
esac

md5sum_file() { [ -f "$1" ] && md5sum "$1" | awk '{print $1}' || echo "MISSING"; }

OUTLOG="$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}.txt"
echo "===== R37 Reviewer SHIP verify cell=$CELL label=$LABEL M=$M N=$N K=$K phys_gpu=$PHYS_GPU mode=$MODE =====" | tee "$OUTLOG"
date | tee -a "$OUTLOG"

if [ "$MODE" = "single_so" ]; then
  MODNAME="tk_mxfp8_r37rev_${CELL}"
  SO="${MODNAME}.cpython-310-x86_64-linux-gnu.so"
  if [ ! -f "$SO" ]; then
    echo "[build] $SO" | tee -a "$OUTLOG"
    rm -f tk_mxfp8_r37rev_${CELL}*.so
    THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=$MODNAME SRC=kernel_mxfp8_layouts.cpp \
      CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DPY_MODULE_NAME=$MODNAME" \
      > "$OUTDIR/build_${CELL}.log" 2>&1
    brc=$?
    H=$(md5sum_file $SO)
    echo "[build $CELL rc=$brc md5=$H]" | tee -a "$OUTDIR/build_md5.log" -a "$OUTLOG"
    [ $brc -ne 0 ] && { echo "BUILD FAIL see $OUTDIR/build_${CELL}.log" >&2 ; exit 1; }
  else
    H=$(md5sum_file $SO)
    echo "[build $CELL md5=$H (cached)]" | tee -a "$OUTLOG"
  fi

  BENCH_OUT="$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}_bench.txt"
  BENCH_ERR="$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}_bench.err"
  ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
    M=$M N=$N K=$K SO="$HERE/$SO" MOD=$MODNAME \
    LAYOUT_A=$LAYOUT_A LAYOUT_B=$LAYOUT_B \
    N_PAIRS=$N_PAIRS MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS \
    PREHEAT_S=$PREHEAT WARMUP_PAIRS=$WARMUP_PAIRS \
    python3 r33c_paired_bench.py > "$BENCH_OUT" 2> "$BENCH_ERR"
  brc=$?
  echo "[bench rc=$brc]" | tee -a "$OUTLOG"
  cat "$BENCH_OUT" | tee -a "$OUTLOG"
  echo "--- stderr ---" | tee -a "$OUTLOG"
  cat "$BENCH_ERR" | tee -a "$OUTLOG"

elif [ "$MODE" = "two_so" ]; then
  # Build TWO .so: default CRR vs HB-shrink-B1 CRR (PIPELINE=1)
  MODNAME_A="tk_mxfp8_r37rev_${CELL}_default"
  MODNAME_B="tk_mxfp8_r37rev_${CELL}_b1"
  SO_A="${MODNAME_A}.cpython-310-x86_64-linux-gnu.so"
  SO_B="${MODNAME_B}.cpython-310-x86_64-linux-gnu.so"

  if [ ! -f "$SO_A" ]; then
    echo "[build] $SO_A (default CRR)" | tee -a "$OUTLOG"
    rm -f tk_mxfp8_r37rev_${CELL}_default*.so
    THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=$MODNAME_A SRC=kernel_mxfp8_layouts.cpp \
      CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DPY_MODULE_NAME=$MODNAME_A" \
      > "$OUTDIR/build_${CELL}_default.log" 2>&1
    brc=$?
    H=$(md5sum_file $SO_A)
    echo "[build $CELL default rc=$brc md5=$H]" | tee -a "$OUTDIR/build_md5.log" -a "$OUTLOG"
    [ $brc -ne 0 ] && { echo "BUILD FAIL default see $OUTDIR/build_${CELL}_default.log" >&2 ; exit 1; }
  else
    H=$(md5sum_file $SO_A)
    echo "[build $CELL default md5=$H (cached)]" | tee -a "$OUTLOG"
  fi

  if [ ! -f "$SO_B" ]; then
    echo "[build] $SO_B (HB-shrink B1, BLK_M=128 PIPELINE=1)" | tee -a "$OUTLOG"
    rm -f tk_mxfp8_r37rev_${CELL}_b1*.so
    THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=$MODNAME_B SRC=kernel_mxfp8_layouts.cpp \
      CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DPY_MODULE_NAME=$MODNAME_B -DMXFP8_CRR_BLK_M=128 -DMXFP8_CRR_HBSHRINK_PIPELINE=1" \
      > "$OUTDIR/build_${CELL}_b1.log" 2>&1
    brc=$?
    H=$(md5sum_file $SO_B)
    echo "[build $CELL b1 rc=$brc md5=$H]" | tee -a "$OUTDIR/build_md5.log" -a "$OUTLOG"
    [ $brc -ne 0 ] && { echo "BUILD FAIL b1 see $OUTDIR/build_${CELL}_b1.log" >&2 ; exit 1; }
  else
    H=$(md5sum_file $SO_B)
    echo "[build $CELL b1 md5=$H (cached)]" | tee -a "$OUTLOG"
  fi

  # R44 Dev D: this historical R37-cycle script lacks R36 3-gate retry
  # (R43 NEW methodology rule 1, MANDATORY for new harnesses).  DEPRECATED:
  # use r43_reviewer_phase23.sh (BENCH_KIND=two_so) for any new dual-.so
  # paired-BABA work; it implements G1+G2a+G2b retry per r38c reference.
  # Use r37_paired_bench_2so.py: dual-.so paired BABA on the SAME entrypoint (gemm_crr_pq_v2)
  BENCH_OUT="$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}_bench.txt"
  BENCH_ERR="$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}_bench.err"
  ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
    M=$M N=$N K=$K \
    SO_A="$HERE/$SO_A" MOD_A=$MODNAME_A \
    SO_B="$HERE/$SO_B" MOD_B=$MODNAME_B \
    N_PAIRS=$N_PAIRS MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS \
    PREHEAT_S=$PREHEAT WARMUP_PAIRS=$WARMUP_PAIRS \
    python3 r37_paired_bench_2so.py > "$BENCH_OUT" 2> "$BENCH_ERR"
  brc=$?
  echo "[bench rc=$brc]" | tee -a "$OUTLOG"
  cat "$BENCH_OUT" | tee -a "$OUTLOG"
  echo "--- stderr ---" | tee -a "$OUTLOG"
  cat "$BENCH_ERR" | tee -a "$OUTLOG"
fi

date | tee -a "$OUTLOG"
echo "DONE cell=$CELL gpu=$PHYS_GPU"
