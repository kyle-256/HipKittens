#!/bin/bash
# R36 Dev C: V2-RCR autotune fan-out for square Q/O cells (8B and 70B).
#
# 4 cells (2 distinct shapes):
#   8B  Q + 8B  O  shape (M=4096, N= 4096, K= 4096) — RCR vs CRR +5.83-+7.05% (R35 Dev D)
#   70B Q + 70B O  shape (M=4096, N= 8192, K= 8192) — RCR vs CRR +8.20-+8.32% (R35 Dev D)
#
# Phases:
#  1. BUILD: per-shape .so + default 8192^3 .so (byte-identical sanity).
#  2. ADVISORY check: 2 RCR predicates fire; 5 RRR predicates do NOT fire on RCR cells; default 8192^3 fires nothing.
#  3. PERF (CRR vs RCR): BABA-paired bench n=10/kernel on HIP=2 + 1 cross-GPU.
#  4. REGRESSION (CRR vs RRR on the 5 V2-RRR predicate cells): re-bench HIP=2.
#
# Per R29-R35 closures:
#  - rm -f specific .so before each build
#  - log per-build md5 (sanity)
#  - rocm-smi -d $PHYS_GPU; sclk-pre/post-preheat AND sclk-post-bench (R35 NEW)
#  - BABA pattern + 30s+ preheat
#  - cross-GPU triangulation: HIP=2 + (HIP=4 or 5)
#  - sclk-post-preheat < 2200 MHz auto-retry up to 3x
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"
PYEXT="$(python3-config --extension-suffix)"

build_so() {
  # Args: tag M N K
  local tag=$1 m=$2 n=$3 k=$4
  local mod_name="tk_mxfp8_r36c_${tag}"
  local out_so="${mod_name}${PYEXT}"
  local log="$HERE/r36c_build_${tag}.log"
  echo "===== BUILD tag=$tag ${m}x${n}x${k} ====="
  rm -f "${out_so}"
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET="${mod_name}" SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$m -DN_DIM=$n -DK_DIM=$k -DPY_MODULE_NAME=${mod_name}" \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "BUILD FAIL rc=$rc see $log"; return 1; fi
  local md5
  md5=$(md5sum "$out_so" | awk '{print $1}')
  echo "BUILD OK $out_so md5=$md5"
  echo "$md5  $out_so" >> "$HERE/r36c_build_md5.log"
}

paired_bench() {
  # Args: cell_id phys_gpu hip_idx M N K layout_a layout_b
  local cell=$1 phys=$2 hidx=$3 m=$4 n=$5 k=$6 la=$7 lb=$8
  local mod_name="tk_mxfp8_r36c_${cell}"
  local so="$HERE/${mod_name}${PYEXT}"
  local out="$HERE/r36c_${cell}_${la}_vs_${lb}_gpu${phys}.txt"
  echo "===== BENCH cell=$cell GPU=$phys (HIP=$hidx) ${m}x${n}x${k} $la vs $lb ====="
  HIP_VISIBLE_DEVICES=$hidx \
    M=$m N=$n K=$k \
    SO=$so MOD=$mod_name LAYOUT_A=$la LAYOUT_B=$lb \
    PHYS_GPU=$phys N_PAIRS=5 MXFP8_WARMUP=30 MXFP8_ITERS=50 \
    PREHEAT_S=45 WARMUP_PAIRS=2 \
    python3 r33c_paired_bench.py 2>&1 | tee "$out"
  echo "Saved $out"
}

# Target cells (2 distinct shapes covering 4 LLaMA cells):
#   c_qo_8b  4096x 4096x 4096   (8B Q + 8B O)
#   c_qo_70b 4096x 8192x 8192   (70B Q + 70B O)
declare -a TARGET_CELLS=(
  "c_qo_8b      4096  4096  4096 crr rcr"
  "c_qo_70b     4096  8192  8192 crr rcr"
)
# Regression cells (5 V2-RRR predicates from R34/R35):
declare -a REGRESSION_CELLS=(
  "c0_70b_down    4096  8192 28672 crr rrr"
  "c1_70b_gateup  4096 28672  8192 crr rrr"
  "c4_70b_kv      4096  1024  8192 crr rrr"
  "c5_8b_gateup   4096 14336  4096 crr rrr"
  "c8_8b_kv       4096  1024  4096 crr rrr"
)
# Negative cells (must NOT fire any advisory):
declare -a NEGATIVE_CELLS=(
  "c_default      8192  8192  8192 crr rrr"
)

: > "$HERE/r36c_build_md5.log"

phase=${1:-all}

if [[ "$phase" == "build" || "$phase" == "all" ]]; then
  echo "########## BUILD PHASE — TARGET CELLS ##########"
  for row in "${TARGET_CELLS[@]}"; do
    read -r tag M N K LA LB <<< "$row"
    build_so "$tag" $M $N $K
  done
  echo "########## BUILD PHASE — REGRESSION CELLS ##########"
  for row in "${REGRESSION_CELLS[@]}"; do
    read -r tag M N K LA LB <<< "$row"
    build_so "$tag" $M $N $K
  done
  echo "########## BUILD PHASE — NEGATIVE / DEFAULT 8192^3 ##########"
  for row in "${NEGATIVE_CELLS[@]}"; do
    read -r tag M N K LA LB <<< "$row"
    build_so "$tag" $M $N $K
  done
fi

PHYS_GPU=${PHYS_GPU_ARG:-2}
HIP_IDX=${HIP_IDX_ARG:-0}

if [[ "$phase" == "bench-target" || "$phase" == "all" ]]; then
  echo "########## BENCH PHASE — TARGET CELLS GPU${PHYS_GPU} ##########"
  for row in "${TARGET_CELLS[@]}"; do
    read -r tag M N K LA LB <<< "$row"
    paired_bench "$tag" $PHYS_GPU $HIP_IDX $M $N $K $LA $LB
  done
fi

if [[ "$phase" == "bench-regression" || "$phase" == "all" ]]; then
  echo "########## BENCH PHASE — REGRESSION CELLS GPU${PHYS_GPU} ##########"
  for row in "${REGRESSION_CELLS[@]}"; do
    read -r tag M N K LA LB <<< "$row"
    paired_bench "$tag" $PHYS_GPU $HIP_IDX $M $N $K $LA $LB
  done
fi

echo "ALL R36C TASKS DONE FOR PHASE=$phase GPU${PHYS_GPU}"
