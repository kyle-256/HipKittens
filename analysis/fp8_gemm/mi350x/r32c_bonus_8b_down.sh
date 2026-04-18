#!/bin/bash
# R32 Dev C bonus: V2-RRR vs V2-RCR for 8B Down (4096x4096x14336)
# Tests whether C4's K-large RRR advantage (70B Down: +12%) extends to 8B Down.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"
PYEXT="$(python3-config --extension-suffix)"
PHYS_GPU=2
M=4096; N=4096; K=14336

build_so() {
  local tag=$1 m=$2 n=$3 k=$4
  local mod_name="tk_mxfp8_r32c_bonus_${tag}"
  local out_so="${mod_name}${PYEXT}"
  local log="$HERE/r32c_bonus_build_${tag}.log"
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

# Both layouts at this shape — same compile-time gate (no cp override).
build_so "rcr" $M $N $K
build_so "rrr" $M $N $K

HIP_VISIBLE_DEVICES=2 \
  M=$M N=$N K=$K \
  SO_RCR="$HERE/tk_mxfp8_r32c_bonus_rcr${PYEXT}" MOD_RCR="tk_mxfp8_r32c_bonus_rcr" \
  SO_RRR="$HERE/tk_mxfp8_r32c_bonus_rrr${PYEXT}" MOD_RRR="tk_mxfp8_r32c_bonus_rrr" \
  PHYS_GPU=$PHYS_GPU N_PAIRS=5 MXFP8_WARMUP=30 MXFP8_ITERS=50 \
  PREHEAT_S=60 WARMUP_PAIRS=2 \
  python3 r32c_bonus_paired_bench.py 2>&1 | tee "$HERE/r32c_BONUS_8b_down_rrr_vs_rcr.txt"
echo "BONUS DONE"
