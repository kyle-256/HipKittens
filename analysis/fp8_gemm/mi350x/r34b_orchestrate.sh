#!/bin/bash
# R34 Dev B: 4-GPU triangulation on R33 Dev C SHIP-LITE cells.
#   c5  8B Gate 4096x14336x4096 V2-CRR vs V2-RRR
#   c6  8B Up   4096x14336x4096 V2-CRR vs V2-RRR
#
# GPUs: 1, 2, 5, 6 (orthogonal to Dev C's 2, 3 — but GPU 2 is included as a
# cross-cycle sanity check vs r33c).
#
# Per R28/R31/R32/R33 closures:
#   - rm -f specific .so + make clean before each build
#   - log per-build md5
#   - rocm-smi -d $PHYS_GPU
#   - BABA + 30s+ preheat (we use 30s preheat per R31 Dev D / task spec)
#   - per task spec: warmup=50, iters=100, n_pairs=5
#   - cross-GPU triangulation: GPUs 1, 2, 5, 6 (4 GPUs)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"
PYEXT="$(python3-config --extension-suffix)"

build_so() {
  # Args: tag M N K
  local tag=$1 m=$2 n=$3 k=$4
  local mod_name="tk_mxfp8_r34b_${tag}"
  local out_so="${mod_name}${PYEXT}"
  local log="$HERE/r34b_build_${tag}.log"
  echo "===== BUILD tag=$tag ${m}x${n}x${k} ====="
  rm -f "${out_so}"
  THUNDERKITTENS_ROOT="$WT" make clean >/dev/null 2>&1
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET="${mod_name}" SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$m -DN_DIM=$n -DK_DIM=$k -DPY_MODULE_NAME=${mod_name}" \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "BUILD FAIL rc=$rc see $log"; return 1; fi
  local md5=$(md5sum "$out_so" | awk '{print $1}')
  echo "BUILD OK $out_so md5=$md5"
  echo "$md5  $out_so" >> "$HERE/r34b_md5.log"
}

paired_bench() {
  # Args: cell_id phys_gpu hip_idx M N K layout_a layout_b
  local cell=$1 phys=$2 hidx=$3 m=$4 n=$5 k=$6 la=$7 lb=$8
  local mod_name="tk_mxfp8_r34b_${cell}"
  local so="$HERE/${mod_name}${PYEXT}"
  local out="$HERE/r34b_${cell}_${la}_vs_${lb}_gpu${phys}.txt"
  echo "===== BENCH cell=$cell GPU=$phys (HIP=$hidx) ${m}x${n}x${k} $la vs $lb ====="
  HIP_VISIBLE_DEVICES=$hidx \
    M=$m N=$n K=$k \
    SO=$so MOD=$mod_name LAYOUT_A=$la LAYOUT_B=$lb \
    PHYS_GPU=$phys N_PAIRS=5 MXFP8_WARMUP=50 MXFP8_ITERS=100 \
    PREHEAT_S=30 WARMUP_PAIRS=2 \
    python3 r33c_paired_bench.py 2>&1 | tee "$out"
  echo "Saved $out"
}

# ---- Cell list ----
declare -a CELLS=(
  "c5_8b_gate     4096 14336  4096 crr rrr"
  "c6_8b_up       4096 14336  4096 crr rrr"
)

> "$HERE/r34b_md5.log"
echo "########## BUILD PHASE ##########"
for row in "${CELLS[@]}"; do
  read -r tag M N K LA LB <<< "$row"
  build_so "$tag" $M $N $K
done

echo "########## BUILD MD5 SUMMARY ##########"
cat "$HERE/r34b_md5.log"
