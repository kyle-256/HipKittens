#!/bin/bash
# R33 Dev A — Task 1: V2-RRR autotune SHIP verification cross-GPU.
# Builds CRR + RRR for shape (M=4096, N=8192, K=28672) and runs the Dev C
# C4 paired bench on TWO GPUs to cross-GPU triangulate the +12.14% SHIP claim.
# Per R32 Reviewer rule (R31 GPU0 discount DEPRECATED, ≥2-GPU triangulation).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"
PYEXT="$(python3-config --extension-suffix)"
M=4096; N=8192; K=28672

build_so() {
  local tag=$1 m=$2 n=$3 k=$4 extra="${5:-}"
  local mod_name="tk_mxfp8_r33a_${tag}"
  local out_so="${mod_name}${PYEXT}"
  local log="$HERE/r33a_build_${tag}.log"
  echo "===== BUILD $tag ${m}x${n}x${k} extra=[$extra] ====="
  rm -f "${out_so}"
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET="${mod_name}" SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$m -DN_DIM=$n -DK_DIM=$k -DPY_MODULE_NAME=${mod_name} $extra" \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "BUILD FAIL rc=$rc see $log"; tail -30 "$log"; return 1
  fi
  local md5=$(md5sum "$out_so" | awk '{print $1}')
  echo "BUILD OK $out_so md5=$md5"
}

# Task 1 SHIP target: (M=4096, N=8192, K=28672)
build_so "task1_crr" $M $N $K "" || exit 1
build_so "task1_rrr" $M $N $K "" || exit 1

run_paired_on_gpu() {
  local phys_gpu=$1
  local out="$HERE/r33a_task1_C4_gpu${phys_gpu}.txt"
  echo "===== BENCH GPU=${phys_gpu} ${M}x${N}x${K} CRR vs RRR ====="
  HIP_VISIBLE_DEVICES=${phys_gpu} \
    M=$M N=$N K=$K \
    SO_CRR="$HERE/tk_mxfp8_r33a_task1_crr${PYEXT}" MOD_CRR="tk_mxfp8_r33a_task1_crr" \
    SO_RRR="$HERE/tk_mxfp8_r33a_task1_rrr${PYEXT}" MOD_RRR="tk_mxfp8_r33a_task1_rrr" \
    PHYS_GPU=${phys_gpu} N_PAIRS=5 MXFP8_WARMUP=30 MXFP8_ITERS=50 \
    PREHEAT_S=30 WARMUP_PAIRS=2 \
    python3 r32c_c4_paired_bench.py 2>&1 | tee "$out"
  echo "Saved $out"
}

# Cross-GPU triangulate on GPU0 + at least one other GPU per task brief.
run_paired_on_gpu 0
run_paired_on_gpu 4

echo "TASK 1 BENCH DONE — see r33a_task1_C4_gpu0.txt and r33a_task1_C4_gpu4.txt"
