#!/bin/bash
# R28 Dev A: build + bench each cell with the auto-default macro gate.
# For cell B (70B Gate), we ALSO build an explicit cp=0 baseline to compute
# a fresh-same-day Welch t-stat for the auto vs baseline comparison.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

# Args: cell_id layout M N K [extra_cxxflags]
build_and_bench() {
  local cell=$1 layout=$2 m=$3 n=$4 k=$5 extra="${6:-}"
  local layout_up=$(echo "$layout" | tr '[:lower:]' '[:upper:]')
  local out="$HERE/r28a_cell${cell}_${layout}_${m}x${n}x${k}.txt"
  local log="$HERE/r28a_build_cell${cell}_${layout}_${m}x${n}x${k}.log"
  echo "===== cell=$cell $layout $m $n $k extra=[$extra] =====" | tee "$out"
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
    python3 r28a_bench5x.py $layout $m $n $k 2>&1 | tee -a "$out"
  echo "Saved $out"
}

# Cell A: 8192^3 V2-CRR with NO M/N/K override -> defaults stay 8192,8192,8192
# (M_DIM/N_DIM/K_DIM already default to 8192). Auto-gate: N_DIM=8192 < 28672 -> cp=0.
build_and_bench A crr 8192 8192 8192 ""

# Cell B: 70B Gate V2-CRR -> auto cp=2
build_and_bench B crr 4096 28672 8192 ""

# Cell B-baseline: 70B Gate V2-CRR with explicit override -> cp=0 (fresh-same-day baseline)
build_and_bench B0 crr 4096 28672 8192 "-DMXFP8_CRR_V2_SCALE_CACHEPOLICY=0"

# Cell C: 70B KV V2-CRR -> auto cp=0 (N=1024 < 28672)
build_and_bench C crr 4096 1024 8192 ""

# Cell D: 8B Gate V2-CRR -> auto cp=0 (K=4096 < 8192)
build_and_bench D crr 4096 14336 4096 ""

# Cell E: 4096^3 V2-RCR sanity -> RCR macro untouched, stays 0
build_and_bench E rcr 4096 4096 4096 ""

echo "ALL CELLS DONE"
