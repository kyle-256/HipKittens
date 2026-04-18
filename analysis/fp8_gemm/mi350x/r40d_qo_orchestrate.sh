#!/bin/bash
# R40 Dev D: 4-GPU STRICT-promotion of R36 Dev C V2-RCR autotune predicates.
# Two shapes covered (separate per-cell .so):
#   CELL=qo_8b   M=4096 N=4096  K=4096   (8B Q/O   ADVISE-V2-RCR-8B-QO)
#   CELL=qo_70b  M=4096 N=8192  K=8192   (70B Q/O  ADVISE-V2-RCR-70B-QO)
#
# These predicates are advisory-only in the dispatcher — they recommend
# the caller switch from gemm_crr_pq_v2 to gemm_rcr_pq_v2. The actual
# performance gain is measured by paired BABA bench of CRR vs RCR on the
# same .so (R33C harness), exactly the same protocol R36 Dev C used at
# N_PAIRS=5. R40 Dev D bumps N_PAIRS=20 + PREHEAT=120 to attempt STRICT
# promotion (min Welch t > 10) per R39 Dev B's protocol.
#
# 3-gate logic (R36 NEW, mandatory R38+):
#   G1  sclk-post-preheat >= 2200 MHz
#   G2a sclk-post-bench   >= 2200 MHz
#   G2b per-run stdev/mean <= 1%
# Plus G1' fallback (R37/R38 NEW): bench_mhz >= 2200 AND median > 500 TF
#
# Usage:
#   CELL=qo_8b  PHYS_GPU=0 ./r40d_qo_orchestrate.sh
#   CELL=qo_70b PHYS_GPU=4 ./r40d_qo_orchestrate.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

OUTDIR="$HERE/r40d_qo_runs"
mkdir -p "$OUTDIR"

PHYS_GPU=${PHYS_GPU:-0}
CELL=${CELL:-qo_8b}
N_PAIRS=${N_PAIRS:-20}
WARMUP=${WARMUP:-30}
ITERS=${ITERS:-50}
PREHEAT=${PREHEAT:-120}
WARMUP_PAIRS=${WARMUP_PAIRS:-2}
SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
SCLK_POSTBENCH_GATE_MHZ=${SCLK_POSTBENCH_GATE_MHZ:-2200}
STDEV_MEAN_GATE=${STDEV_MEAN_GATE:-0.01}
MAX_RETRIES=${MAX_RETRIES:-3}
G1P_MIN_MEDIAN_TF=${G1P_MIN_MEDIAN_TF:-500}

case "$CELL" in
  qo_8b)  M=4096; N=4096; K=4096 ;;
  qo_70b) M=4096; N=8192; K=8192 ;;
  *) echo "ERROR: unknown CELL=$CELL"; exit 2 ;;
esac

MODNAME="tk_mxfp8_r40d_${CELL}"
SO="${MODNAME}.cpython-310-x86_64-linux-gnu.so"

md5sum_file() {
  if [ -f "$1" ]; then md5sum "$1" | awk '{print $1}'; else echo "MISSING"; fi
}

OUT="$OUTDIR/${CELL}_gpu${PHYS_GPU}.txt"
echo "===== R40 Dev D V2-RCR QO 4-GPU STRICT-promote cell=$CELL M=$M N=$N K=$K phys_gpu=$PHYS_GPU =====" | tee "$OUT"
echo "[orchestrate] N_PAIRS=$N_PAIRS WARMUP=$WARMUP ITERS=$ITERS PREHEAT=${PREHEAT}s WARMUP_PAIRS=$WARMUP_PAIRS" | tee -a "$OUT"
echo "[orchestrate] R36/R38 3-gate: G1 sclk-post-preheat>=${SCLK_GATE_MHZ} G2a sclk-post-bench>=${SCLK_POSTBENCH_GATE_MHZ} G2b stdev/mean<=${STDEV_MEAN_GATE}, MAX_RETRIES=$MAX_RETRIES, G1P_MIN_MEDIAN_TF=${G1P_MIN_MEDIAN_TF}" | tee -a "$OUT"
date | tee -a "$OUT"

echo "[pre-orchestrate sclk check]" | tee -a "$OUT"
rocm-smi --showclocks -d $PHYS_GPU 2>&1 | grep -i sclk | head -3 | tee -a "$OUT"

# Build per-cell .so (single .so used by all GPUs of this cell)
if [ ! -f "$SO" ]; then
  echo "[build] $SO M=$M N=$N K=$K" | tee -a "$OUT"
  rm -f $SO
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=$MODNAME SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DPY_MODULE_NAME=$MODNAME" \
    > "$OUTDIR/build_${CELL}.log" 2>&1
  brc=$?
  H=$(md5sum_file $SO)
  echo "[build $CELL rc=$brc md5=$H]" | tee -a "$OUTDIR/build_md5.log" -a "$OUT"
  if [ $brc -ne 0 ]; then
    echo "BUILD FAIL see $OUTDIR/build_${CELL}.log" | tee -a "$OUT"
    exit 1
  fi
else
  H=$(md5sum_file $SO)
  echo "[build $CELL md5=$H (cached)]" | tee -a "$OUTDIR/build_md5.log" -a "$OUT"
fi

best_g1p_path=""
best_g1p_err=""

attempt=1
while [ $attempt -le $MAX_RETRIES ]; do
  echo "[bench attempt $attempt/$MAX_RETRIES]" | tee -a "$OUT"
  BENCH_OUT="$OUTDIR/${CELL}_gpu${PHYS_GPU}_attempt${attempt}.txt"
  BENCH_ERR="$OUTDIR/${CELL}_gpu${PHYS_GPU}_attempt${attempt}.err"
  ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
    M=$M N=$N K=$K SO="$HERE/$SO" MOD=$MODNAME \
    LAYOUT_A=crr LAYOUT_B=rcr \
    N_PAIRS=$N_PAIRS MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS \
    PREHEAT_S=$PREHEAT WARMUP_PAIRS=$WARMUP_PAIRS \
    MXFP8_DISPATCH_TRACE=1 \
    python3 r33c_paired_bench.py > "$BENCH_OUT" 2> "$BENCH_ERR"
  brc=$?
  echo "[bench rc=$brc]" | tee -a "$OUT"
  cat "$BENCH_OUT" | tee -a "$OUT"
  echo "--- stderr ---" | tee -a "$OUT"
  cat "$BENCH_ERR" | tee -a "$OUT"

  POST_LINE=$(grep "sclk-post-preheat" "$BENCH_OUT" | head -1)
  POST_MHZ=$(echo "$POST_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')
  POSTB_LINE=$(grep "sclk-post-bench" "$BENCH_OUT" | head -1)
  POSTB_MHZ=$(echo "$POSTB_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')

  RATIO_CRR=$(grep -E "^CRR " "$BENCH_OUT" | sed -nE 's/.*median=([0-9.]+).*stdev=([0-9.]+).*/\1 \2/p' | python3 -c "import sys
try:
  m,s=map(float,sys.stdin.read().split())
  print(f'{(s/m):.6f}' if m>0 else 'NaN')
except: print('NaN')" 2>/dev/null)
  RATIO_RCR=$(grep -E "^RCR " "$BENCH_OUT" | sed -nE 's/.*median=([0-9.]+).*stdev=([0-9.]+).*/\1 \2/p' | python3 -c "import sys
try:
  m,s=map(float,sys.stdin.read().split())
  print(f'{(s/m):.6f}' if m>0 else 'NaN')
except: print('NaN')" 2>/dev/null)
  if [ -z "$RATIO_CRR" ]; then RATIO_CRR="NaN"; fi
  if [ -z "$RATIO_RCR" ]; then RATIO_RCR="NaN"; fi

  RATIO_PASS=0
  if [ "$RATIO_CRR" != "NaN" ] && [ "$RATIO_RCR" != "NaN" ]; then
    RATIO_PASS=$(python3 -c "print(1 if max(${RATIO_CRR}, ${RATIO_RCR}) <= ${STDEV_MEAN_GATE} else 0)" 2>/dev/null || echo 0)
  fi

  echo "[orchestrate] attempt=$attempt rc=$brc G1 sclk-post-preheat=${POST_MHZ:-unknown}MHz (gate=${SCLK_GATE_MHZ}) G2a sclk-post-bench=${POSTB_MHZ:-unknown}MHz (gate=${SCLK_POSTBENCH_GATE_MHZ}) G2b stdev/mean CRR=${RATIO_CRR} RCR=${RATIO_RCR} (gate=${STDEV_MEAN_GATE} pass=${RATIO_PASS})" | tee -a "$OUT"

  PASS=1
  if [ $brc -ne 0 ]; then PASS=0; fi
  if [ -z "$POST_MHZ" ] || [ "$POST_MHZ" -lt "$SCLK_GATE_MHZ" ]; then PASS=0; fi
  GATE2_PASS=0
  if [ -n "$POSTB_MHZ" ] && [ "$POSTB_MHZ" -ge "$SCLK_POSTBENCH_GATE_MHZ" ]; then GATE2_PASS=1; fi
  if [ "$RATIO_PASS" = "1" ]; then GATE2_PASS=1; fi
  if [ $GATE2_PASS -eq 0 ]; then PASS=0; fi
  if [ $PASS -eq 1 ]; then
    echo "[orchestrate] PASS G1+G2a+G2b at attempt=$attempt; using as final." | tee -a "$OUT"
    cp "$BENCH_OUT" "$OUTDIR/${CELL}_gpu${PHYS_GPU}_clean.txt"
    cp "$BENCH_ERR" "$OUTDIR/${CELL}_gpu${PHYS_GPU}_clean.err"
    echo "[orchestrate] gate_path=G1+G2a+G2b" | tee -a "$OUT"
    break
  fi
  CRR_MED=$(grep -E "^CRR " "$BENCH_OUT" | sed -nE 's/.*median=([0-9.]+).*/\1/p' | head -1)
  RCR_MED=$(grep -E "^RCR " "$BENCH_OUT" | sed -nE 's/.*median=([0-9.]+).*/\1/p' | head -1)
  if [ -n "$POSTB_MHZ" ] && [ "$POSTB_MHZ" -ge "$SCLK_POSTBENCH_GATE_MHZ" ] && [ "$RATIO_PASS" = "1" ] && [ -n "$CRR_MED" ] && [ -n "$RCR_MED" ]; then
    G1P_OK=$(python3 -c "print(1 if min(${CRR_MED}, ${RCR_MED}) > ${G1P_MIN_MEDIAN_TF} else 0)" 2>/dev/null || echo 0)
    if [ "$G1P_OK" = "1" ] && [ -z "$best_g1p_path" ]; then
      best_g1p_path="$BENCH_OUT"
      best_g1p_err="$BENCH_ERR"
    fi
  fi
  echo "[orchestrate] attempt=$attempt FAILED gates; retry after 5s..." | tee -a "$OUT"
  sleep 5
  attempt=$((attempt+1))
done

if [ $attempt -gt $MAX_RETRIES ]; then
  if [ -n "$best_g1p_path" ]; then
    echo "[orchestrate] EXHAUSTED $MAX_RETRIES; using G1' fallback from $best_g1p_path." | tee -a "$OUT"
    cp "$best_g1p_path" "$OUTDIR/${CELL}_gpu${PHYS_GPU}_clean.txt"
    cp "$best_g1p_err" "$OUTDIR/${CELL}_gpu${PHYS_GPU}_clean.err"
    echo "[orchestrate] gate_path=G1'_fallback" | tee -a "$OUT"
  else
    echo "[orchestrate] EXHAUSTED $MAX_RETRIES retries; using last attempt." | tee -a "$OUT"
    cp "$BENCH_OUT" "$OUTDIR/${CELL}_gpu${PHYS_GPU}_lastattempt.txt"
    cp "$BENCH_ERR" "$OUTDIR/${CELL}_gpu${PHYS_GPU}_lastattempt.err"
    echo "[orchestrate] gate_path=EXHAUSTED" | tee -a "$OUT"
  fi
fi

echo "DONE gpu=$PHYS_GPU cell=$CELL attempts=$attempt" | tee -a "$OUT"
date | tee -a "$OUT"
