#!/bin/bash
# R34 Dev B: re-run with longer preheat (60s) and more warmup pairs (3) to
# eliminate initial-throttle artifacts seen on GPU 5 and GPU 6.
set -u
PHYS=$1
HIDX=$2
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PYEXT="$(python3-config --extension-suffix)"

declare -a CELLS=(
  "c5_8b_gate     4096 14336  4096 crr rrr"
  "c6_8b_up       4096 14336  4096 crr rrr"
)

echo "===== PREFLIGHT rocm-smi -d $PHYS ====="
rocm-smi --showclocks -d $PHYS 2>&1 | head -20
echo "========================================"

for row in "${CELLS[@]}"; do
  read -r tag M N K LA LB <<< "$row"
  mod_name="tk_mxfp8_r34b_${tag}"
  so="$HERE/${mod_name}${PYEXT}"
  out="$HERE/r34b_${tag}_${LA}_vs_${LB}_gpu${PHYS}_clean.txt"
  echo "===== BENCH cell=$tag GPU=$PHYS (HIP=$HIDX) ${M}x${N}x${K} $LA vs $LB ====="
  HIP_VISIBLE_DEVICES=$HIDX \
    M=$M N=$N K=$K \
    SO=$so MOD=$mod_name LAYOUT_A=$LA LAYOUT_B=$LB \
    PHYS_GPU=$PHYS N_PAIRS=5 MXFP8_WARMUP=50 MXFP8_ITERS=100 \
    PREHEAT_S=60 WARMUP_PAIRS=3 \
    python3 r33c_paired_bench.py 2>&1 | tee "$out"
  echo "Saved $out"
done
echo "ALL CELLS DONE on PHYS_GPU=$PHYS (clean run)"
