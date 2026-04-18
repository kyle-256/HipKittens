#!/bin/bash
# R35 Dev A — auto-retry single cell on a target GPU until
# sclk-post-preheat >= 2200 MHz (R34 NEW rule). Up to 3 attempts.
#
# Usage:  PHYS_GPU=0 HIP_IDX=0 CELL=c8_8b_kv M=4096 N=1024 K=4096 ./r35a_retry_cells.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PYEXT="$(python3-config --extension-suffix)"
PHYS_GPU=${PHYS_GPU:-0}
HIP_IDX=${HIP_IDX:-0}
CELL=${CELL:?}
M=${M:?}
N=${N:?}
K=${K:?}
SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
MAX_RETRIES=${MAX_RETRIES:-3}
mod_name="tk_mxfp8_r35a_${CELL}"
so="$HERE/${mod_name}${PYEXT}"

best_attempt=0
best_sclk=0
for attempt in $(seq 1 $MAX_RETRIES); do
  echo "===== ${CELL} GPU${PHYS_GPU} attempt $attempt / $MAX_RETRIES ====="
  out="$HERE/r35a_${CELL}_crr_vs_rrr_gpu${PHYS_GPU}_retry${attempt}.txt"
  HIP_VISIBLE_DEVICES=$HIP_IDX \
    M=$M N=$N K=$K \
    SO=$so MOD=$mod_name LAYOUT_A=crr LAYOUT_B=rrr \
    PHYS_GPU=$PHYS_GPU N_PAIRS=5 MXFP8_WARMUP=30 MXFP8_ITERS=50 \
    PREHEAT_S=60 WARMUP_PAIRS=3 \
    python3 r33c_paired_bench.py 2>&1 | tee "$out"
  sclk=$(grep 'sclk-post-preheat' "$out" | tail -1 | grep -oE '[0-9]+Mhz' | grep -oE '[0-9]+')
  sclk=${sclk:-0}
  echo "[attempt $attempt] sclk-post-preheat = ${sclk} MHz"
  if [ "$sclk" -gt "$best_sclk" ]; then best_sclk=$sclk; best_attempt=$attempt; fi
  if [ "$sclk" -ge "$SCLK_GATE_MHZ" ]; then
    echo "[attempt $attempt] OK sclk >= $SCLK_GATE_MHZ — keeping result"
    cp "$out" "$HERE/r35a_${CELL}_crr_vs_rrr_gpu${PHYS_GPU}.txt"
    echo "ACCEPTED  attempt=$attempt  sclk=$sclk"
    exit 0
  fi
  echo "[attempt $attempt] sclk too low, will retry after 30s cooldown"
  sleep 30
done

echo "WARNING: could not get sclk >= $SCLK_GATE_MHZ in $MAX_RETRIES attempts"
echo "Best attempt: $best_attempt sclk=$best_sclk"
cp "$HERE/r35a_${CELL}_crr_vs_rrr_gpu${PHYS_GPU}_retry${best_attempt}.txt" \
   "$HERE/r35a_${CELL}_crr_vs_rrr_gpu${PHYS_GPU}.txt"
exit 1
