#!/bin/bash
# R30 Dev D: V2-RCR B-tile load reorder (H7) — variant sweep on 4096^3, plus
# 8192^3 no-regression check on the surviving variants.
#
# Build hygiene: rm -f $(TARGET)*.so before each compile (Makefile clean
# rule omits the extension and leaves the cpython.so in place).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

PHYS_GPU=${PHYS_GPU:-3}
N_RUNS=${N_RUNS:-5}

build_and_bench() {
  local cell=$1 m=$2 n=$3 k=$4 extra="${5:-}"
  local out="$HERE/r30d_cell${cell}_rcr_${m}x${n}x${k}.txt"
  local log="$HERE/r30d_build_cell${cell}_rcr_${m}x${n}x${k}.log"
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
  HIP_VISIBLE_DEVICES=$PHYS_GPU R30D_PHYS_GPU=$PHYS_GPU \
    N_RUNS=$N_RUNS MXFP8_WARMUP=50 MXFP8_ITERS=100 \
    python3 r30d_bench.py rcr $m $n $k 2>&1 | tee -a "$out"
  echo "Saved $out"
}

# Phase 1: 4096^3 baseline + 4 H7 variants
build_and_bench A0_base 4096 4096 4096 ""
build_and_bench A1_v1   4096 4096 4096 "-DMXFP8_RCR_V2_BLOAD_REORDER=1"
build_and_bench A2_v2   4096 4096 4096 "-DMXFP8_RCR_V2_BLOAD_REORDER=2"
build_and_bench A3_v3   4096 4096 4096 "-DMXFP8_RCR_V2_BLOAD_REORDER=3"
build_and_bench A4_v4   4096 4096 4096 "-DMXFP8_RCR_V2_BLOAD_REORDER=4"

# Phase 2: 8192^3 baseline (no-regression floor)
build_and_bench B0_base 8192 8192 8192 ""

echo "ALL CELLS DONE"
