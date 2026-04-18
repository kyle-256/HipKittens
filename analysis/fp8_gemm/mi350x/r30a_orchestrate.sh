#!/bin/bash
# R30 Dev A: rect-V2 cycle — Path A reassessment + baseline reverify.
# Cell 1: V2-CRR baseline at 70B KV target shape (4096x1024x8192) — confirms baseline
# Cell 2: V2-CRR regression check at 8192^3 — confirms scaffolding adds zero overhead
# Cell 3: rect-build compile sanity (MXFP8_RECT_BLK_N=64, V1 fallback path verified)
#
# R30 outcome: NO SHIP. Path A (true rect-V2 fastpath) requires ~2.5 days per
# R28D estimate; Path B (V1 fallback) gives 2.66 TFLOPS = 0.34% of square baseline.
# See r30a_findings.md for refined Path A breakdown.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

build_and_bench() {
  local cell=$1 stage=$2 m=$3 n=$4 k=$5 extra="${6:-}"
  local out="$HERE/r30a_cell${cell}_${stage}_${m}x${n}x${k}.txt"
  local log="$HERE/r30a_build_cell${cell}_${stage}_${m}x${n}x${k}.log"
  echo "===== R30 cell=$cell stage=$stage $m $n $k extra=[$extra] =====" | tee "$out"
  rm -f "$HERE"/tk_mxfp8_layouts*.so
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$m -DN_DIM=$n -DK_DIM=$k $extra" \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "BUILD FAIL rc=$rc see $log" | tee -a "$out"
    return 1
  fi
  echo "md5(.so) = $(md5sum tk_mxfp8_layouts*.so | awk '{print $1}')" | tee -a "$out"
  HIP_VISIBLE_DEVICES=0 N_RUNS=5 MXFP8_WARMUP=50 MXFP8_ITERS=100 \
    python3 r29a_bench.py $stage crr $m $n $k 2>&1 | tee -a "$out"
}

# Cell 1: V2-CRR baseline at 70B KV target shape (default build)
build_and_bench 1 v2 4096 1024 8192 ""

# Cell 2: 8192^3 V2-CRR regression check (default build)
build_and_bench 2 v2 8192 8192 8192 ""

# Cell 3: rect-build compile sanity — V1 fallback path proven by R29 Cell 2
echo "===== R30 cell=3 rect-build compile sanity =====" | tee r30a_cell3_rect_build_compile.txt
rm -f tk_mxfp8_layouts*.so
THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
  CXXFLAGS="-w -DM_DIM=4096 -DN_DIM=1024 -DK_DIM=8192 -DMXFP8_RECT_BLK_N=64" \
  > r30a_build_cell3_rect.log 2>&1
echo "rect build rc=$?" | tee -a r30a_cell3_rect_build_compile.txt
md5sum tk_mxfp8_layouts*.so | tee -a r30a_cell3_rect_build_compile.txt

echo "ALL CELLS DONE"
