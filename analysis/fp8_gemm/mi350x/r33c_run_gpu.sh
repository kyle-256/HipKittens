#!/bin/bash
# R33 Dev C: Run all 8 cell benches sequentially on a single GPU.
# Args: <phys_gpu> <hip_idx>
set -u
PHYS=$1
HIDX=$2
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PYEXT="$(python3-config --extension-suffix)"

declare -a CELLS=(
  "c1_70b_gate    4096 28672  8192 crr rrr"
  "c2_70b_up      4096 28672  8192 crr rrr"
  "c3_70b_qo      4096  8192  8192 rcr rrr"
  "c4_70b_kv      4096  1024  8192 crr rrr"
  "c5_8b_gate     4096 14336  4096 crr rrr"
  "c6_8b_up       4096 14336  4096 crr rrr"
  "c7_8b_qo       4096  4096  4096 rcr rrr"
  "c8_8b_kv       4096  1024  4096 crr rrr"
)

for row in "${CELLS[@]}"; do
  read -r tag M N K LA LB <<< "$row"
  mod_name="tk_mxfp8_r33c_${tag}"
  so="$HERE/${mod_name}${PYEXT}"
  out="$HERE/r33c_${tag}_${LA}_vs_${LB}_gpu${PHYS}.txt"
  echo "===== BENCH cell=$tag GPU=$PHYS ${M}x${N}x${K} $LA vs $LB ====="
  HIP_VISIBLE_DEVICES=$HIDX \
    M=$M N=$N K=$K \
    SO=$so MOD=$mod_name LAYOUT_A=$LA LAYOUT_B=$LB \
    PHYS_GPU=$PHYS N_PAIRS=5 MXFP8_WARMUP=30 MXFP8_ITERS=50 \
    PREHEAT_S=45 WARMUP_PAIRS=2 \
    python3 r33c_paired_bench.py 2>&1 | tee "$out"
  echo "Saved $out"
done
echo "ALL CELLS DONE on GPU=$PHYS"
