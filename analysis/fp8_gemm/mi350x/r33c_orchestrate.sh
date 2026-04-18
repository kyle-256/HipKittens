#!/bin/bash
# R33 Dev C: Per-shape RRR exploration on remaining 8 LLaMA cells.
#
# For each cell: build single .so with M/N/K, run BABA paired bench RRR vs
# baseline (CRR or RCR depending on what the existing autotune favors).
#
# Cells (priority order):
#   1. 70B Gate 4096x28672x8192 — RRR vs CRR (R28 Dev A SHIP shape)
#   2. 70B Up   4096x28672x8192 — RRR vs CRR (same shape as Gate)
#   3. 70B Q/O  4096x8192x8192  — RRR vs RCR
#   4. 70B KV   4096x1024x8192  — RRR vs CRR (R31 Reviewer's small-N shape)
#   5. 8B Gate  4096x14336x4096 — RRR vs CRR
#   6. 8B Up    4096x14336x4096 — RRR vs CRR
#   7. 8B Q/O   4096x4096x4096  — RRR vs RCR
#   8. 8B KV    4096x1024x4096  — RRR vs CRR
#
# Per R31/R32 closures:
#  - rm -f specific .so before each build
#  - log per-build md5
#  - rocm-smi -d $PHYS_GPU
#  - BABA + 30s+ preheat
#  - cross-GPU triangulation: GPU2 + GPU3
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"
PYEXT="$(python3-config --extension-suffix)"

build_so() {
  # Args: tag M N K
  local tag=$1 m=$2 n=$3 k=$4
  local mod_name="tk_mxfp8_r33c_${tag}"
  local out_so="${mod_name}${PYEXT}"
  local log="$HERE/r33c_build_${tag}.log"
  echo "===== BUILD tag=$tag ${m}x${n}x${k} ====="
  rm -f "${out_so}"
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET="${mod_name}" SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$m -DN_DIM=$n -DK_DIM=$k -DPY_MODULE_NAME=${mod_name}" \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "BUILD FAIL rc=$rc see $log"; return 1; fi
  local md5=$(md5sum "$out_so" | awk '{print $1}')
  echo "BUILD OK $out_so md5=$md5"
}

paired_bench() {
  # Args: cell_id phys_gpu hip_idx M N K layout_a layout_b
  local cell=$1 phys=$2 hidx=$3 m=$4 n=$5 k=$6 la=$7 lb=$8
  local mod_name="tk_mxfp8_r33c_${cell}"
  local so="$HERE/${mod_name}${PYEXT}"
  local out="$HERE/r33c_${cell}_${la}_vs_${lb}_gpu${phys}.txt"
  echo "===== BENCH cell=$cell GPU=$phys (HIP=$hidx) ${m}x${n}x${k} $la vs $lb ====="
  HIP_VISIBLE_DEVICES=$hidx \
    M=$m N=$n K=$k \
    SO=$so MOD=$mod_name LAYOUT_A=$la LAYOUT_B=$lb \
    PHYS_GPU=$phys N_PAIRS=5 MXFP8_WARMUP=30 MXFP8_ITERS=50 \
    PREHEAT_S=45 WARMUP_PAIRS=2 \
    python3 r33c_paired_bench.py 2>&1 | tee "$out"
  echo "Saved $out"
}

# ---- Cell list ----
# Each row: tag M N K layout_a(baseline) layout_b(candidate)
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

# Build all .so first (cells 1+2 share shape; cells 5+6 share shape — but build
# distinct .so for clarity and to prevent any module-name reuse race).
echo "########## BUILD PHASE ##########"
for row in "${CELLS[@]}"; do
  read -r tag M N K LA LB <<< "$row"
  build_so "$tag" $M $N $K
done

# Bench phase: GPU2 + GPU3 for each cell
echo "########## BENCH PHASE — GPU2 ##########"
for row in "${CELLS[@]}"; do
  read -r tag M N K LA LB <<< "$row"
  paired_bench "$tag" 2 2 $M $N $K $LA $LB
done

echo "########## BENCH PHASE — GPU3 ##########"
for row in "${CELLS[@]}"; do
  read -r tag M N K LA LB <<< "$row"
  paired_bench "$tag" 3 3 $M $N $K $LA $LB
done

echo "ALL R33C BENCHES DONE"
