#!/bin/bash
# R29 Dev C: build + bench V2-RCR setprio sweep on 4096^3 and 8192^3.
# Crucially: rm -f the actual .so file before each build because
# `make clean` in this Makefile only removes $(TARGET) (no extension)
# and leaves the .cpython.so intact, which can mask rebuilds.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

build_and_bench() {
  local cell=$1 m=$2 n=$3 k=$4 extra="${5:-}"
  local out="$HERE/r29c_cell${cell}_rcr_${m}x${n}x${k}.txt"
  local log="$HERE/r29c_build_cell${cell}_rcr_${m}x${n}x${k}.log"
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
    python3 r29c_bench.py rcr $m $n $k 2>&1 | tee -a "$out"
  echo "Saved $out"
}

# Baselines (default macros, byte-identical to head)
build_and_bench A_base 4096 4096 4096 ""
build_and_bench B_base 8192 8192 8192 ""

# Variants — V2-RCR MMA setprio sweep
build_and_bench A_p2 4096 4096 4096 "-DMXFP8_RCR_V2_MMA_SETPRIO=2"
build_and_bench A_p3 4096 4096 4096 "-DMXFP8_RCR_V2_MMA_SETPRIO=3"
build_and_bench B_p2 8192 8192 8192 "-DMXFP8_RCR_V2_MMA_SETPRIO=2"
build_and_bench B_p3 8192 8192 8192 "-DMXFP8_RCR_V2_MMA_SETPRIO=3"

echo "ALL CELLS DONE"
