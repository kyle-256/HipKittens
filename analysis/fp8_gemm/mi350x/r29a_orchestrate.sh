#!/bin/bash
# R29 Dev A: build + bench cells for rect-V2 cycle (Path B-prime).
# Cell 1: default build, V2-CRR on 4096x1024x8192 (rect target shape, V2 baseline)
# Cell 2: rect build (-DMXFP8_RECT_BLK_N=64), V1-CRR on 4096x1024x8192 (Path B fallback)
# Cell 3: default build, V2-CRR on 8192^3 (regression check vs R28 baseline 2844)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

build_and_bench() {
  local cell=$1 stage=$2 m=$3 n=$4 k=$5 extra="${6:-}"
  local out="$HERE/r29a_cell${cell}_${stage}_${m}x${n}x${k}.txt"
  local log="$HERE/r29a_build_cell${cell}_${stage}_${m}x${n}x${k}.log"
  echo "===== cell=$cell stage=$stage $m $n $k extra=[$extra] =====" | tee "$out"
  make clean >/dev/null 2>&1
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$m -DN_DIM=$n -DK_DIM=$k $extra" \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "BUILD FAIL rc=$rc see $log" | tee -a "$out"
    return 1
  fi
  HIP_VISIBLE_DEVICES=0 N_RUNS=5 MXFP8_WARMUP=50 MXFP8_ITERS=100 \
    python3 r29a_bench.py $stage crr $m $n $k 2>&1 | tee -a "$out"
  echo "Saved $out"
}

# Cell 1: V2-CRR baseline at rect target shape (default build, N=1024)
build_and_bench 1 v2 4096 1024 8192 ""

# Cell 2: Path B fallback — V1-CRR at rect target shape with rect build
build_and_bench 2 v1 4096 1024 8192 "-DMXFP8_RECT_BLK_N=64"

# Cell 3: 8192^3 V2-CRR regression check (default build)
build_and_bench 3 v2 8192 8192 8192 ""

echo "ALL CELLS DONE"
