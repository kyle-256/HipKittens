#!/bin/bash
# R34 Dev A: V2-RRR autotune fan-out wire-in + LLaMA matrix regression check.
#
# 4 target cells (R33 STRICT SHIPs being newly wired):
#   c1 70B Gate 4096x28672x8192 — CRR vs RRR (advisory expected)
#   c2 70B Up   4096x28672x8192 — CRR vs RRR (advisory expected — same shape as c1)
#   c4 70B KV   4096x1024x8192  — CRR vs RRR (advisory expected)
#   c8 8B  KV   4096x1024x4096  — CRR vs RRR (advisory expected)
# Plus the existing wired cell (regression sanity):
#   c0 70B Down 4096x8192x28672 — CRR vs RRR (advisory expected)
#
# Neighboring shapes (advisory must NOT fire — regression-negative):
#   c3 70B Q/O    4096x8192x8192  — RCR is prod baseline
#   c5 8B Gate    4096x14336x4096 — SHIP-LITE (deferred to R34 reviewer)
#   c7 8B Q/O     4096x4096x4096  — RCR is prod baseline
#
# Per R31/R32/R33 closures:
#  - rm -f specific .so before each build
#  - log per-build md5
#  - rocm-smi -d $PHYS_GPU
#  - BABA + 30s+ preheat
#  - cross-GPU triangulation: GPU0 + GPU4 (R33 sub-rule: min-of-GPUs)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"
PYEXT="$(python3-config --extension-suffix)"

build_so() {
  # Args: tag M N K
  local tag=$1 m=$2 n=$3 k=$4
  local mod_name="tk_mxfp8_r34a_${tag}"
  local out_so="${mod_name}${PYEXT}"
  local log="$HERE/r34a_build_${tag}.log"
  echo "===== BUILD tag=$tag ${m}x${n}x${k} ====="
  rm -f "${out_so}"
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET="${mod_name}" SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$m -DN_DIM=$n -DK_DIM=$k -DPY_MODULE_NAME=${mod_name}" \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "BUILD FAIL rc=$rc see $log"; return 1; fi
  local md5=$(md5sum "$out_so" | awk '{print $1}')
  echo "BUILD OK $out_so md5=$md5"
  echo "$md5  $out_so" >> "$HERE/r34a_build_md5.log"
}

paired_bench() {
  # Args: cell_id phys_gpu hip_idx M N K layout_a layout_b
  local cell=$1 phys=$2 hidx=$3 m=$4 n=$5 k=$6 la=$7 lb=$8
  local mod_name="tk_mxfp8_r34a_${cell}"
  local so="$HERE/${mod_name}${PYEXT}"
  local out="$HERE/r34a_${cell}_${la}_vs_${lb}_gpu${phys}.txt"
  echo "===== BENCH cell=$cell GPU=$phys (HIP=$hidx) ${m}x${n}x${k} $la vs $lb ====="
  HIP_VISIBLE_DEVICES=$hidx \
    M=$m N=$n K=$k \
    SO=$so MOD=$mod_name LAYOUT_A=$la LAYOUT_B=$lb \
    PHYS_GPU=$phys N_PAIRS=5 MXFP8_WARMUP=30 MXFP8_ITERS=50 \
    PREHEAT_S=45 WARMUP_PAIRS=2 \
    python3 r33c_paired_bench.py 2>&1 | tee "$out"
  echo "Saved $out"
}

# Target cells (autotune-wired); all CRR vs RRR
declare -a TARGET_CELLS=(
  "c0_70b_down    4096  8192 28672 crr rrr"
  "c1_70b_gate    4096 28672  8192 crr rrr"
  "c2_70b_up      4096 28672  8192 crr rrr"
  "c4_70b_kv      4096  1024  8192 crr rrr"
  "c8_8b_kv       4096  1024  4096 crr rrr"
)

# Neighboring/regression-negative cells
declare -a NEIGHBOR_CELLS=(
  "c3_70b_qo      4096  8192  8192 crr rrr"
  "c5_8b_gate     4096 14336  4096 crr rrr"
  "c7_8b_qo       4096  4096  4096 crr rrr"
)

: > "$HERE/r34a_build_md5.log"

echo "########## BUILD PHASE — TARGET CELLS ##########"
for row in "${TARGET_CELLS[@]}"; do
  read -r tag M N K LA LB <<< "$row"
  build_so "$tag" $M $N $K
done

echo "########## BUILD PHASE — NEIGHBOR CELLS ##########"
for row in "${NEIGHBOR_CELLS[@]}"; do
  read -r tag M N K LA LB <<< "$row"
  build_so "$tag" $M $N $K
done

PHYS_GPU=${PHYS_GPU_ARG:-0}
HIP_IDX=${HIP_IDX_ARG:-0}

echo "########## BENCH PHASE — TARGET CELLS GPU${PHYS_GPU} ##########"
for row in "${TARGET_CELLS[@]}"; do
  read -r tag M N K LA LB <<< "$row"
  paired_bench "$tag" $PHYS_GPU $HIP_IDX $M $N $K $LA $LB
done

echo "########## BENCH PHASE — NEIGHBOR CELLS GPU${PHYS_GPU} ##########"
for row in "${NEIGHBOR_CELLS[@]}"; do
  read -r tag M N K LA LB <<< "$row"
  paired_bench "$tag" $PHYS_GPU $HIP_IDX $M $N $K $LA $LB
done

echo "ALL R34A BENCHES DONE FOR GPU${PHYS_GPU}"
