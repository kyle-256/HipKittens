#!/bin/bash
# R35 Dev A — auto-retry c5 (4096x14336x4096) on a target GPU until
# sclk-post-preheat >= 2200 MHz (R34 NEW rule). Up to 3 attempts.
#
# Usage:  PHYS_GPU=4 HIP_IDX=4 ./r35a_retry_c5.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PYEXT="$(python3-config --extension-suffix)"
PHYS_GPU=${PHYS_GPU:-0}
HIP_IDX=${HIP_IDX:-0}
SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
MAX_RETRIES=${MAX_RETRIES:-3}

cell=c5_8b_gate
M=4096; N=14336; K=4096
mod_name="tk_mxfp8_r35a_${cell}"
so="$HERE/${mod_name}${PYEXT}"

for attempt in $(seq 1 $MAX_RETRIES); do
  echo "===== c5 GPU${PHYS_GPU} attempt $attempt / $MAX_RETRIES ====="
  out="$HERE/r35a_${cell}_crr_vs_rrr_gpu${PHYS_GPU}_retry${attempt}.txt"
  HIP_VISIBLE_DEVICES=$HIP_IDX \
    M=$M N=$N K=$K \
    SO=$so MOD=$mod_name LAYOUT_A=crr LAYOUT_B=rrr \
    PHYS_GPU=$PHYS_GPU N_PAIRS=5 MXFP8_WARMUP=30 MXFP8_ITERS=50 \
    PREHEAT_S=60 WARMUP_PAIRS=3 \
    python3 r33c_paired_bench.py 2>&1 | tee "$out"
  sclk=$(grep 'sclk-post-preheat' "$out" | tail -1 | grep -oE '[0-9]+Mhz' | grep -oE '[0-9]+')
  echo "[attempt $attempt] sclk-post-preheat = ${sclk:-unk} MHz"
  if [ -n "$sclk" ] && [ "$sclk" -ge "$SCLK_GATE_MHZ" ]; then
    echo "[attempt $attempt] OK sclk >= $SCLK_GATE_MHZ — keeping result"
    cp "$out" "$HERE/r35a_${cell}_crr_vs_rrr_gpu${PHYS_GPU}.txt"
    echo "ACCEPTED  attempt=$attempt  sclk=$sclk"
    exit 0
  fi
  echo "[attempt $attempt] sclk too low, will retry after 30s cooldown"
  sleep 30
done

echo "WARNING: could not get sclk >= $SCLK_GATE_MHZ in $MAX_RETRIES attempts"
echo "Best-effort attempt $MAX_RETRIES kept as the final result"
cp "$out" "$HERE/r35a_${cell}_crr_vs_rrr_gpu${PHYS_GPU}.txt"
exit 1
