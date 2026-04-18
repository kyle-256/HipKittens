#!/bin/bash
# R34 Dev B: Run c5 + c6 paired benches sequentially on a single physical GPU.
# Args: <phys_gpu> <hip_idx>
#
# Note about HIP_VISIBLE_DEVICES: when you set HIP_VISIBLE_DEVICES=N, that
# physical GPU appears to the runtime as device 0. So inside the python harness
# torch.cuda.device(0) is the chosen physical GPU. We pass PHYS_GPU=<phys_gpu>
# to the harness so rocm-smi monitoring queries the correct physical device.
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

# Preflight rocm-smi (R31 paradigm correction — query physical GPU)
echo "===== PREFLIGHT rocm-smi -d $PHYS ====="
rocm-smi --showclocks -d $PHYS 2>&1 | head -20
echo "========================================"

for row in "${CELLS[@]}"; do
  read -r tag M N K LA LB <<< "$row"
  mod_name="tk_mxfp8_r34b_${tag}"
  so="$HERE/${mod_name}${PYEXT}"
  out="$HERE/r34b_${tag}_${LA}_vs_${LB}_gpu${PHYS}.txt"
  echo "===== BENCH cell=$tag GPU=$PHYS (HIP=$HIDX) ${M}x${N}x${K} $LA vs $LB ====="
  HIP_VISIBLE_DEVICES=$HIDX \
    M=$M N=$N K=$K \
    SO=$so MOD=$mod_name LAYOUT_A=$LA LAYOUT_B=$LB \
    PHYS_GPU=$PHYS N_PAIRS=5 MXFP8_WARMUP=50 MXFP8_ITERS=100 \
    PREHEAT_S=30 WARMUP_PAIRS=2 \
    python3 r33c_paired_bench.py 2>&1 | tee "$out"
  echo "Saved $out"
done
echo "ALL CELLS DONE on PHYS_GPU=$PHYS"
