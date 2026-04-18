#!/bin/bash
# R36 Dev B: 6th V2-RRR autotune predicate (8B-Down shape 4096x4096x14336).
#
# Phases:
#  1. Source patch (already applied to kernel_mxfp8_layouts.cpp).
#  2. Build sanity (default 8192^3 + per-shape advisory verification).
#  3. Per-shape numerics + perf re-verify (c_8b_down).
#  4. Cross-GPU verify (HIP_VISIBLE_DEVICES=1 primary; HIP_VISIBLE_DEVICES=4 secondary).
#  5. Regression smoke test (re-bench 5 existing predicate cells + 70B-Down).
#
# Per R31/R32/R33/R34/R35 closures:
#  - rm -f specific .so before each build
#  - log per-build md5
#  - rocm-smi -d $PHYS_GPU
#  - BABA + 30s+ preheat
#  - cross-GPU triangulation: GPU1 (primary) + GPU4 (secondary)
#  - R34 sclk-post-preheat < 2200 MHz auto-retry up to 3x
#  - R35 NEW rule: sclk-post-bench >= 2200 MHz second gate, retry up to 3x

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"
PYEXT="$(python3-config --extension-suffix)"

build_so() {
  # Args: tag M N K
  local tag=$1 m=$2 n=$3 k=$4
  local mod_name="tk_mxfp8_r36b_${tag}"
  local out_so="${mod_name}${PYEXT}"
  local log="$HERE/r36b_build_${tag}.log"
  echo "===== BUILD tag=$tag ${m}x${n}x${k} ====="
  rm -f "${out_so}"
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET="${mod_name}" SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$m -DN_DIM=$n -DK_DIM=$k -DPY_MODULE_NAME=${mod_name}" \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "BUILD FAIL rc=$rc see $log"; return 1; fi
  local md5=$(md5sum "$out_so" | awk '{print $1}')
  echo "BUILD OK $out_so md5=$md5"
  echo "$md5  $out_so" >> "$HERE/r36b_build_md5.log"
}

paired_bench() {
  # Args: cell_id phys_gpu hip_idx M N K layout_a layout_b
  local cell=$1 phys=$2 hidx=$3 m=$4 n=$5 k=$6 la=$7 lb=$8
  local mod_name="tk_mxfp8_r36b_${cell}"
  local so="$HERE/${mod_name}${PYEXT}"
  local out="$HERE/r36b_${cell}_${la}_vs_${lb}_gpu${phys}.txt"
  echo "===== BENCH cell=$cell GPU=$phys (HIP=$hidx) ${m}x${n}x${k} $la vs $lb ====="
  HIP_VISIBLE_DEVICES=$hidx \
    M=$m N=$n K=$k \
    SO=$so MOD=$mod_name LAYOUT_A=$la LAYOUT_B=$lb \
    PHYS_GPU=$phys N_PAIRS=5 MXFP8_WARMUP=30 MXFP8_ITERS=50 \
    PREHEAT_S=45 WARMUP_PAIRS=2 \
    python3 r33c_paired_bench.py 2>&1 | tee "$out"
  echo "Saved $out"
}

# Target cells:
#   c_8b_down is the new predicate target (4096x4096x14336)
# Regression smoke test cells (previously wired predicates):
#   c0 c1 c4 c8 c5 — must still emit 1 advisory each, Δ% must remain ≥+5%
# Neighbor cells (must NOT fire):
#   c3 70B Q/O, c7 8B Q/O, default (8192^3)

declare -a TARGET_CELLS=(
  "c_8b_down       4096  4096 14336 crr rrr"
)
declare -a REGRESSION_CELLS=(
  "c0_70b_down    4096  8192 28672 crr rrr"
  "c1_70b_gate    4096 28672  8192 crr rrr"
  "c4_70b_kv      4096  1024  8192 crr rrr"
  "c8_8b_kv       4096  1024  4096 crr rrr"
  "c5_8b_gate     4096 14336  4096 crr rrr"
)
declare -a NEIGHBOR_CELLS=(
  "c3_70b_qo      4096  8192  8192 crr rrr"
  "c7_8b_qo       4096  4096  4096 crr rrr"
  "default        8192  8192  8192 crr rrr"
)

phase=${1:-all}

if [[ "$phase" == "build" || "$phase" == "all" ]]; then
  : > "$HERE/r36b_build_md5.log"
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

  echo "########## BUILD PHASE — NEIGHBOR CELLS ##########"
  for row in "${NEIGHBOR_CELLS[@]}"; do
    read -r tag M N K LA LB <<< "$row"
    build_so "$tag" $M $N $K
  done
fi

PHYS_GPU=${PHYS_GPU_ARG:-1}
HIP_IDX=${HIP_IDX_ARG:-1}

if [[ "$phase" == "bench_target" || "$phase" == "bench_all" || "$phase" == "all" ]]; then
  echo "########## BENCH PHASE — TARGET CELL c_8b_down GPU${PHYS_GPU} ##########"
  for row in "${TARGET_CELLS[@]}"; do
    read -r tag M N K LA LB <<< "$row"
    paired_bench "$tag" $PHYS_GPU $HIP_IDX $M $N $K $LA $LB
  done
fi

if [[ "$phase" == "bench_regression" || "$phase" == "bench_all" ]]; then
  echo "########## BENCH PHASE — REGRESSION CELLS GPU${PHYS_GPU} ##########"
  for row in "${REGRESSION_CELLS[@]}"; do
    read -r tag M N K LA LB <<< "$row"
    paired_bench "$tag" $PHYS_GPU $HIP_IDX $M $N $K $LA $LB
  done
  echo "ALL R36B BENCHES DONE FOR GPU${PHYS_GPU}"
fi
