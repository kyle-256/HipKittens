#!/bin/bash
# R31 Dev D: A/B/A test of persistent-CU dispatch on V2-RCR 4096^3.
# Interleaved baseline -> persist304 -> baseline -> persist304 -> baseline
# in same session to detect & control for sclk/thermal drift.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

PHYS_GPU=${PHYS_GPU:-3}
N_RUNS=${N_RUNS:-5}

build_and_bench() {
  local cell=$1 m=$2 n=$3 k=$4 extra="${5:-}"
  local out="$HERE/r31d_ab_cell${cell}_rcr_${m}x${n}x${k}.txt"
  local log="$HERE/r31d_ab_build_cell${cell}_rcr_${m}x${n}x${k}.log"
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

# A/B/A pattern: 5 cells alternating
build_and_bench R0_base_rep1 4096 4096 4096 ""
build_and_bench R1_p304_rep1 4096 4096 4096 "-DMXFP8_RCR_V2_PERSISTENT=1 -DMXFP8_RCR_V2_PERSISTENT_GRID=304"
build_and_bench R2_base_rep2 4096 4096 4096 ""
build_and_bench R3_p304_rep2 4096 4096 4096 "-DMXFP8_RCR_V2_PERSISTENT=1 -DMXFP8_RCR_V2_PERSISTENT_GRID=304"
build_and_bench R4_base_rep3 4096 4096 4096 ""
echo "ALL CELLS DONE"
