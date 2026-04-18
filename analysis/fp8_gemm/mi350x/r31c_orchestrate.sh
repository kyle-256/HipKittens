#!/bin/bash
# R31 Dev C: V2-RCR PIPELINE_SCALE second-buffer (VGPR-prefetch pivot) bench.
# Build cache hygiene: rm -f the actual .so before each build (R29 Dev C rule).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

build_and_bench() {
  local cell=$1 m=$2 n=$3 k=$4 extra="${5:-}"
  local out="$HERE/r31c_${cell}_${m}x${n}x${k}_v2rcr.txt"
  local log="$HERE/r31c_${cell}_${m}x${n}x${k}_build.log"
  echo "===== cell=$cell rcr $m $n $k extra=[$extra] =====" | tee "$out"
  rm -f "$HERE"/tk_mxfp8_layouts*.so "$HERE"/tk_mxfp8_layouts
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$m -DN_DIM=$n -DK_DIM=$k $extra" \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "BUILD FAIL rc=$rc see $log" | tee -a "$out"
    return 1
  fi
  echo "MD5 $(md5sum tk_mxfp8_layouts*.so | awk '{print $1}')" | tee -a "$out"
  HIP_VISIBLE_DEVICES=2 N_RUNS=5 MXFP8_WARMUP=50 MXFP8_ITERS=100 \
    python3 r28a_bench5x.py rcr $m $n $k 2>&1 | tee -a "$out"
  echo "Saved $out"
}

# Baselines (default macro state — byte-identical to head, md5 4ae07863... for 4k)
build_and_bench baseline 4096 4096 4096 ""
build_and_bench baseline 8192 8192 8192 ""

# Variant: VGPR-prefetch second-buffer (R31 Dev C lever)
build_and_bench prefetch 4096 4096 4096 "-DMXFP8_RCR_V2_SCALE_PREFETCH=1"
build_and_bench prefetch 8192 8192 8192 "-DMXFP8_RCR_V2_SCALE_PREFETCH=1"
