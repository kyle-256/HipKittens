#!/bin/bash
# R41 Reviewer Phase 1: 4-GPU baseline 70B-KV V2-CRR (M=4096 N=1024 K=8192)
# Locks to GPU 2/3/6/7 per R40 Reviewer recommendation (like-for-like cross-cycle).
# Uses per-GPU staged .so to allow parallel execution.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

OUTDIR="$HERE/r41_reviewer_4gpu_runs"
mkdir -p "$OUTDIR"

PHYS_GPU=${PHYS_GPU:-2}
N_RUNS=${N_RUNS:-5}
WARMUP=${WARMUP:-50}
ITERS=${ITERS:-100}
SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
SCLK_POSTBENCH_GATE_MHZ=${SCLK_POSTBENCH_GATE_MHZ:-2200}
STDEV_MEAN_GATE=${STDEV_MEAN_GATE:-0.01}
G1P_MIN_MEDIAN_TF=${G1P_MIN_MEDIAN_TF:-500}
MAX_RETRIES=${MAX_RETRIES:-3}

M=4096; N=1024; K=8192; LAYOUT=crr; CELL=70b_kv_crr

SO_SRC=${SO_SRC:-/tmp/r41_70bkv_baseline.so}

# Per-GPU staging dir (avoid concurrent overwrites)
STAGE_DIR="/tmp/r41_stage_gpu${PHYS_GPU}"
mkdir -p "$STAGE_DIR"
STAGED_SO="$STAGE_DIR/tk_mxfp8_layouts.cpython-310-x86_64-linux-gnu.so"
cp -f "$SO_SRC" "$STAGED_SO"
H=$(md5sum "$STAGED_SO" | awk '{print $1}')
echo "[r41_reviewer_baseline gpu=$PHYS_GPU md5=$H so=$SO_SRC stage=$STAGE_DIR]"

# Symlink test scripts into stage dir so python can import the local .so
for f in r35_reviewer_bench5x.py test_mxfp8_python.py; do
  [ -e "$STAGE_DIR/$f" ] || ln -sf "$HERE/$f" "$STAGE_DIR/$f"
done

OUT="$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}.txt"
echo "===== R41 Reviewer Phase 1 baseline cell=$CELL phys_gpu=$PHYS_GPU =====" | tee "$OUT"
date | tee -a "$OUT"
echo "[pre-orchestrate sclk]" | tee -a "$OUT"
rocm-smi --showclocks -d $PHYS_GPU 2>&1 | grep -i sclk | head -3 | tee -a "$OUT"

attempt=1
best_g1p_attempt=0
while [ $attempt -le $MAX_RETRIES ]; do
  echo "[bench attempt $attempt/$MAX_RETRIES]" | tee -a "$OUT"
  BENCH_OUT="$OUTDIR/bench_gpu${PHYS_GPU}_attempt${attempt}.txt"
  BENCH_ERR="$OUTDIR/bench_gpu${PHYS_GPU}_attempt${attempt}.err"
  ( cd "$STAGE_DIR" && \
    PYTHONPATH="$STAGE_DIR:${PYTHONPATH:-}" \
    ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
      N_RUNS=$N_RUNS WARMUP=$WARMUP ITERS=$ITERS \
      python3 "$HERE/r35_reviewer_bench5x.py" mxfp8 $LAYOUT $M $N $K \
      > "$BENCH_OUT" 2> "$BENCH_ERR" )
  brc=$?
  POST_LINE=$(grep "sclk-post-preheat" "$BENCH_ERR" | head -1)
  POST_MHZ=$(echo "$POST_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')
  POSTB_LINE=$(grep "sclk-post-bench" "$BENCH_ERR" | head -1)
  POSTB_MHZ=$(echo "$POSTB_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')
  SUM_LINE=$(grep "^SUMMARY " "$BENCH_OUT" | head -1)
  MEAN_VAL=$(echo "$SUM_LINE" | sed -nE 's/.*mean=([0-9.]+).*/\1/p')
  STDEV_VAL=$(echo "$SUM_LINE" | sed -nE 's/.*stdev=([0-9.]+).*/\1/p')
  MED_VAL=$(echo "$SUM_LINE" | sed -nE 's/.*median=([0-9.]+).*/\1/p')
  if [ -n "$MEAN_VAL" ] && [ -n "$STDEV_VAL" ] && [ "$MEAN_VAL" != "0" ]; then
    RATIO=$(python3 -c "print(f'{(${STDEV_VAL}/${MEAN_VAL}):.6f}')" 2>/dev/null || echo "NaN")
  else
    RATIO="NaN"
  fi
  G1=0; G2A=0; G2B=0; G1P=0
  [ -n "$POST_MHZ" ] && [ "$POST_MHZ" -ge "$SCLK_GATE_MHZ" ] && G1=1
  [ -n "$POSTB_MHZ" ] && [ "$POSTB_MHZ" -ge "$SCLK_POSTBENCH_GATE_MHZ" ] && G2A=1
  if [ "$RATIO" != "NaN" ]; then
    G2B=$(python3 -c "print(1 if ${RATIO} <= ${STDEV_MEAN_GATE} else 0)" 2>/dev/null || echo 0)
  fi
  if [ -n "$POSTB_MHZ" ] && [ "$POSTB_MHZ" -ge "$SCLK_POSTBENCH_GATE_MHZ" ] && [ -n "$MED_VAL" ]; then
    G1P=$(python3 -c "print(1 if ${MED_VAL} > ${G1P_MIN_MEDIAN_TF} else 0)" 2>/dev/null || echo 0)
  fi
  echo "[gate gpu=$PHYS_GPU att=$attempt rc=$brc med=${MED_VAL:-NA} cv=${RATIO} G1=$G1 G2A=$G2A G2B=$G2B G1P=$G1P preMhz=${POST_MHZ:-NA} benchMhz=${POSTB_MHZ:-NA}]" | tee -a "$OUT"
  if [ $brc -eq 0 ] && [ $G1 -eq 1 ] && [ $G2A -eq 1 ] && [ $G2B -eq 1 ]; then
    echo "[orchestrate] PASS G1+G2a+G2b at att=$attempt" | tee -a "$OUT"
    cp "$BENCH_OUT" "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_clean.txt"
    cp "$BENCH_ERR" "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_clean.err"
    echo "$MED_VAL" > "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_clean.med"
    echo "G1+G2a+G2b" > "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_clean.gate"
    exit 0
  fi
  if [ $brc -eq 0 ] && [ $G2A -eq 1 ] && [ $G2B -eq 1 ] && [ $G1P -eq 1 ]; then
    best_g1p_attempt=$attempt
    cp "$BENCH_OUT" "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_g1pcandidate.txt"
    cp "$BENCH_ERR" "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_g1pcandidate.err"
    echo "$MED_VAL" > "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_g1pcandidate.med"
  fi
  sleep 5
  attempt=$((attempt+1))
done
if [ $best_g1p_attempt -gt 0 ]; then
  echo "[orchestrate] ACCEPT G1' fallback att=$best_g1p_attempt" | tee -a "$OUT"
  cp "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_g1pcandidate.txt" "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_clean.txt"
  cp "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_g1pcandidate.err" "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_clean.err"
  cp "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_g1pcandidate.med" "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_clean.med"
  echo "G1'_fallback" > "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_clean.gate"
  exit 0
fi
echo "[orchestrate] EXHAUSTED no acceptable run" | tee -a "$OUT"
echo "EXHAUSTED" > "$OUTDIR/${CELL}_mxfp8_gpu${PHYS_GPU}_clean.gate"
exit 1
