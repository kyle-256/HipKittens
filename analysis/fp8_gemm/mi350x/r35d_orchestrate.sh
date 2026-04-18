#!/bin/bash
# R35 Dev D — LLaMA full matrix re-bench at R34 head.
# Verifies 4 wired autotune predicates fire on target shapes (5 cells out of 14)
# and Δ% holds end-to-end. r35-a 5th predicate (8B Gate/Up) NOT yet landed at
# R34 head — c5/c6 will report Δ% but no advisory will fire.
#
# Matrix:
#   LLaMA 8B (hidden=4096, intermediate=14336):
#     8B-Q     4096x4096x4096        (RCR baseline; advisory MUST NOT fire)
#     8B-K     4096x1024x4096        (advisory expected)
#     8B-V     4096x1024x4096        (advisory expected)
#     8B-O     4096x4096x4096        (RCR baseline; advisory MUST NOT fire)
#     8B-Gate  4096x14336x4096       (R34 head: NO advisory; r35-a wires this)
#     8B-Up    4096x14336x4096       (R34 head: NO advisory; r35-a wires this)
#     8B-Down  4096x4096x14336       (no advisory wired; reference)
#   LLaMA 70B (hidden=8192, intermediate=28672):
#     70B-Q    4096x8192x8192        (RCR baseline; advisory MUST NOT fire)
#     70B-K    4096x1024x8192        (advisory expected)
#     70B-V    4096x1024x8192        (advisory expected)
#     70B-O    4096x8192x8192        (RCR baseline; advisory MUST NOT fire)
#     70B-Gate 4096x28672x8192       (advisory expected)
#     70B-Up   4096x28672x8192       (advisory expected)
#     70B-Down 4096x8192x28672       (advisory expected)
#
# K/V cells share shape with same-tag-different-module per cell so we get
# distinct .so per cell (R33 Dev C convention).
#
# Methodology (R34+):
#  - rm -f specific .so before each build; per-build md5 logged
#  - rocm-smi -d $PHYS_GPU
#  - 45s preheat + BABA + 2 warmup pairs + 5 recorded pairs (n=10/kernel)
#  - auto-retry up to 3x on sclk-post-preheat < 2200 MHz (R34 NEW)
#  - cross-GPU triangulation: GPU6 (primary) + GPU7 (verify)
#  - min-of-GPUs Δ% is SHIP gate (R33 sub-rule)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"
PYEXT="$(python3-config --extension-suffix)"

MODE="${1:-build}"  # build | bench-gpu6 | bench-gpu7 | all
PHYS_GPU_ARG="${PHYS_GPU_ARG:-6}"
HIP_IDX_ARG="${HIP_IDX_ARG:-6}"

build_so() {
  local tag=$1 m=$2 n=$3 k=$4
  local mod_name="tk_mxfp8_r35d_${tag}"
  local out_so="${mod_name}${PYEXT}"
  local log="$HERE/r35d_build_${tag}.log"
  echo "===== BUILD tag=$tag ${m}x${n}x${k} ====="
  rm -f "${out_so}"
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET="${mod_name}" SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$m -DN_DIM=$n -DK_DIM=$k -DPY_MODULE_NAME=${mod_name}" \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "BUILD FAIL rc=$rc see $log"; return 1; fi
  local md5=$(md5sum "$out_so" | awk '{print $1}')
  echo "BUILD OK $out_so md5=$md5"
  echo "$md5  $out_so" >> "$HERE/r35d_build_md5.log"
}

paired_bench_one() {
  # Args: cell_id phys_gpu hip_idx M N K layout_a layout_b suffix
  local cell=$1 phys=$2 hidx=$3 m=$4 n=$5 k=$6 la=$7 lb=$8 sfx=${9:-}
  local mod_name="tk_mxfp8_r35d_${cell}"
  local so="$HERE/${mod_name}${PYEXT}"
  local out="$HERE/r35d_${cell}_${la}_vs_${lb}_gpu${phys}${sfx}.txt"
  for attempt in 1 2 3; do
    echo "===== BENCH cell=$cell GPU=$phys (HIP=$hidx) ${m}x${n}x${k} $la vs $lb attempt=$attempt ====="
    HIP_VISIBLE_DEVICES=$hidx \
      M=$m N=$n K=$k \
      SO=$so MOD=$mod_name LAYOUT_A=$la LAYOUT_B=$lb \
      PHYS_GPU=$phys N_PAIRS=5 MXFP8_WARMUP=30 MXFP8_ITERS=50 \
      PREHEAT_S=45 WARMUP_PAIRS=2 \
      python3 r33c_paired_bench.py 2>&1 | tee "$out"
    # Auto-retry on low sclk-post-preheat (R34 NEW rule)
    local sclk_line=$(grep "sclk-post-preheat" "$out" | tail -1)
    local sclk_mhz=$(echo "$sclk_line" | grep -oE "[0-9]+Mhz" | head -1 | sed 's/Mhz//')
    if [ -n "$sclk_mhz" ] && [ "$sclk_mhz" -lt 2200 ] && [ $attempt -lt 3 ]; then
      echo "AUTO-RETRY: sclk-post-preheat=${sclk_mhz}MHz < 2200, attempt=$attempt"
      sleep 10
      continue
    fi
    break
  done
  echo "Saved $out"
}

# 14-shape matrix (cell_tag M N K layout_a layout_b).
# Note: K/V share shape but get distinct module names → distinct .so.
declare -a ALL_CELLS=(
  "8b_q     4096  4096  4096 crr rrr"
  "8b_k     4096  1024  4096 crr rrr"
  "8b_v     4096  1024  4096 crr rrr"
  "8b_o     4096  4096  4096 crr rrr"
  "8b_gate  4096 14336  4096 crr rrr"
  "8b_up    4096 14336  4096 crr rrr"
  "8b_down  4096  4096 14336 crr rrr"
  "70b_q    4096  8192  8192 crr rrr"
  "70b_k    4096  1024  8192 crr rrr"
  "70b_v    4096  1024  8192 crr rrr"
  "70b_o    4096  8192  8192 crr rrr"
  "70b_gate 4096 28672  8192 crr rrr"
  "70b_up   4096 28672  8192 crr rrr"
  "70b_down 4096  8192 28672 crr rrr"
)

# Square cells get RCR (3-way coverage)
declare -a RCR_CELLS=(
  "8b_q     4096  4096  4096 crr rcr"
  "8b_o     4096  4096  4096 crr rcr"
  "70b_q    4096  8192  8192 crr rcr"
  "70b_o    4096  8192  8192 crr rcr"
)

if [ "$MODE" = "build" ] || [ "$MODE" = "all" ]; then
  : > "$HERE/r35d_build_md5.log"
  echo "########## BUILD PHASE — 14 cells ##########"
  for row in "${ALL_CELLS[@]}"; do
    read -r tag M N K LA LB <<< "$row"
    build_so "$tag" $M $N $K
  done
  echo "########## BUILD MD5 SUMMARY ##########"
  cat "$HERE/r35d_build_md5.log"
fi

if [ "$MODE" = "bench-gpu6" ] || [ "$MODE" = "all" ]; then
  PHYS_GPU=6; HIP_IDX=6
  echo "########## BENCH PHASE — CRR vs RRR GPU${PHYS_GPU} ##########"
  for row in "${ALL_CELLS[@]}"; do
    read -r tag M N K LA LB <<< "$row"
    paired_bench_one "$tag" $PHYS_GPU $HIP_IDX $M $N $K $LA $LB
  done
  echo "########## BENCH PHASE — CRR vs RCR (square only) GPU${PHYS_GPU} ##########"
  for row in "${RCR_CELLS[@]}"; do
    read -r tag M N K LA LB <<< "$row"
    paired_bench_one "$tag" $PHYS_GPU $HIP_IDX $M $N $K $LA $LB
  done
fi

if [ "$MODE" = "bench-gpu7" ] || [ "$MODE" = "all" ]; then
  PHYS_GPU=7; HIP_IDX=7
  echo "########## BENCH PHASE — CRR vs RRR GPU${PHYS_GPU} ##########"
  for row in "${ALL_CELLS[@]}"; do
    read -r tag M N K LA LB <<< "$row"
    paired_bench_one "$tag" $PHYS_GPU $HIP_IDX $M $N $K $LA $LB
  done
  echo "########## BENCH PHASE — CRR vs RCR (square only) GPU${PHYS_GPU} ##########"
  for row in "${RCR_CELLS[@]}"; do
    read -r tag M N K LA LB <<< "$row"
    paired_bench_one "$tag" $PHYS_GPU $HIP_IDX $M $N $K $LA $LB
  done
fi

echo "DONE r35d MODE=$MODE PHYS_GPU_ARG=$PHYS_GPU_ARG"
