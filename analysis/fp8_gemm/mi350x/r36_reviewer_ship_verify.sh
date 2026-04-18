#!/bin/bash
# R36 Reviewer Phase 2: SHIP verification on shape-specific builds
# Re-tests cells:
#   c4 = R33 Dev C 70B KV @ M=4096 N=1024 K=8192  (RRR vs CRR; expect RRR +10%+)
#   c5 = R34 Dev B / R35 Dev A 8B Gate @ M=4096 N=14336 K=4096 (RRR vs CRR; expect RRR +5%+)
#
# Uses the same paired BABA bench harness (r33c_paired_bench.py) as R33-R35 reviewers.
# Single .so per cell (M_DIM/N_DIM/K_DIM matched), separate PY_MODULE_NAME.
#
# Methodology rules: same R29-R35 closures (incl. R35 NEW second sclk gate).
# We rely on the long PREHEAT (60s) + the bench's own sclk reads + post-bench
# stdev/mean check after the fact (re-bench if not clean).
#
# Usage:
#   PHYS_GPU=4 CELL=c4 ./r36_reviewer_ship_verify.sh   # 70B KV
#   PHYS_GPU=5 CELL=c5 ./r36_reviewer_ship_verify.sh   # 8B Gate
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

OUTDIR="$HERE/r36_reviewer_devXXX_verify"
mkdir -p "$OUTDIR"

PHYS_GPU=${PHYS_GPU:-4}
CELL=${CELL:-c4}
N_PAIRS=${N_PAIRS:-5}
WARMUP=${WARMUP:-30}
ITERS=${ITERS:-50}
PREHEAT=${PREHEAT:-45}
WARMUP_PAIRS=${WARMUP_PAIRS:-2}

case "$CELL" in
  c4)
    M=4096; N=1024; K=8192; LABEL=70b_kv
    ;;
  c5)
    M=4096; N=14336; K=4096; LABEL=8b_gate
    ;;
  *)
    echo "Unknown CELL=$CELL; use c4 or c5" >&2
    exit 1
    ;;
esac

MODNAME="tk_mxfp8_r36rev_${CELL}"
SO="${MODNAME}.cpython-310-x86_64-linux-gnu.so"

md5sum_file() { [ -f "$1" ] && md5sum "$1" | awk '{print $1}' || echo "MISSING"; }

echo "===== R36 Reviewer SHIP verify cell=$CELL label=$LABEL M=$M N=$N K=$K phys_gpu=$PHYS_GPU =====" | tee "$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}.txt"
date | tee -a "$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}.txt"

# Build only if SO missing for this cell (md5 deterministic)
if [ ! -f "$SO" ]; then
  echo "[build] $SO" | tee -a "$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}.txt"
  rm -f tk_fp8_layouts*.so tk_mxfp8_layouts*.so tk_mxfp8_r36rev*.so
  make clean >/dev/null 2>&1
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=$MODNAME SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DPY_MODULE_NAME=$MODNAME" \
    > "$OUTDIR/build_${CELL}.log" 2>&1
  brc=$?
  H=$(md5sum_file $SO)
  echo "[build $CELL rc=$brc md5=$H]" | tee -a "$OUTDIR/build_md5.log" -a "$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}.txt"
  if [ $brc -ne 0 ]; then
    echo "BUILD FAIL see $OUTDIR/build_${CELL}.log" >&2
    exit 1
  fi
else
  H=$(md5sum_file $SO)
  echo "[build $CELL md5=$H (cached)]" | tee -a "$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}.txt"
fi

# Bench BABA paired: layout_a=crr, layout_b=rrr
BENCH_OUT="$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}_bench.txt"
BENCH_ERR="$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}_bench.err"

ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
  M=$M N=$N K=$K SO="$HERE/$SO" MOD=$MODNAME \
  LAYOUT_A=crr LAYOUT_B=rrr \
  N_PAIRS=$N_PAIRS MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS \
  PREHEAT_S=$PREHEAT WARMUP_PAIRS=$WARMUP_PAIRS \
  python3 r33c_paired_bench.py > "$BENCH_OUT" 2> "$BENCH_ERR"
brc=$?
echo "[bench rc=$brc]" | tee -a "$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}.txt"
cat "$BENCH_OUT" | tee -a "$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}.txt"
echo "--- stderr ---" | tee -a "$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}.txt"
cat "$BENCH_ERR" | tee -a "$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}.txt"

# Apply R35 NEW gate post-hoc: stdev/mean ratio of CRR + RRR lists
RATIO_CRR=$(grep -E "^CRR[ \t]+median" "$BENCH_OUT" | sed -nE 's/.*median=([0-9.]+).*stdev=([0-9.]+).*/\1 \2/p' | python3 -c "import sys; m,s=map(float,sys.stdin.read().split()); print(f'{s/m:.6f}' if m>0 else 'NaN')" 2>/dev/null || echo "NaN")
RATIO_RRR=$(grep -E "^RRR[ \t]+median" "$BENCH_OUT" | sed -nE 's/.*median=([0-9.]+).*stdev=([0-9.]+).*/\1 \2/p' | python3 -c "import sys; m,s=map(float,sys.stdin.read().split()); print(f'{s/m:.6f}' if m>0 else 'NaN')" 2>/dev/null || echo "NaN")
echo "[R35 gate post-hoc] stdev/mean CRR=$RATIO_CRR RRR=$RATIO_RRR (gate <=0.02 for paired BABA tolerable; <=0.01 for single-layout)" | tee -a "$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}.txt"

date | tee -a "$OUTDIR/${CELL}_${LABEL}_gpu${PHYS_GPU}.txt"
echo "DONE cell=$CELL gpu=$PHYS_GPU"
