#!/bin/bash
# R32 Dev C: K-large MLP shapes investigation.
#
# C1: 70B Down 4096x8192x28672 V2-CRR cp 0/1/2/3 sweep
# C2: V2-CRR resource report audit (K=28672 vs K=8192) — handled in r32c_audit.sh
# C3: 8B Down  4096x4096x14336 V2-RCR cp 0/1/2/3 sweep
# C4: V2-RRR for 70B Down (K=28672, large K)
#
# Per R31 closures:
#  - rm -f tk_mxfp8_layouts*.so before each build
#  - log per-build md5
#  - rocm-smi -d $PHYS_GPU (=2) not -d 0
#  - BABA + 30s preheat
#
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"
PYEXT="$(python3-config --extension-suffix)"
PHYS_GPU=2

build_so() {
  # Args: tag layout M N K extra_flags
  # Builds .so as `tk_mxfp8_r32c_<tag>` (PY_MODULE_NAME matches filename so
  # importlib.util.spec_from_file_location can load it under that name).
  local tag=$1 layout=$2 m=$3 n=$4 k=$5 extra="${6:-}"
  local mod_name="tk_mxfp8_r32c_${tag}"
  local out_so="${mod_name}${PYEXT}"
  local log="$HERE/r32c_build_${tag}.log"
  echo "===== BUILD tag=$tag $layout ${m}x${n}x${k} extra=[$extra] ====="
  rm -f "${out_so}"
  # Note: TARGET= controls Make's filename; PY_MODULE_NAME= controls pybind init symbol.
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET="${mod_name}" SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$m -DN_DIM=$n -DK_DIM=$k -DPY_MODULE_NAME=${mod_name} $extra" \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "BUILD FAIL rc=$rc see $log"; return 1
  fi
  local md5=$(md5sum "$out_so" | awk '{print $1}')
  echo "BUILD OK $out_so md5=$md5"
}

paired_bench() {
  # Args: cell_id layout M N K mod_a label_a mod_b label_b
  # mod_a/mod_b are module names (matching the .so basename minus PYEXT).
  local cell=$1 layout=$2 m=$3 n=$4 k=$5 mod_a=$6 la=$7 mod_b=$8 lb=$9
  local so_a="$HERE/${mod_a}${PYEXT}"
  local so_b="$HERE/${mod_b}${PYEXT}"
  local out="$HERE/r32c_${cell}_${la}_vs_${lb}.txt"
  echo "===== BENCH cell=$cell $layout ${m}x${n}x${k} $la vs $lb ====="
  HIP_VISIBLE_DEVICES=2 \
    M=$m N=$n K=$k LAYOUT=$layout \
    SO_A=$so_a SO_B=$so_b MOD_A=$mod_a MOD_B=$mod_b LABEL_A=$la LABEL_B=$lb \
    PHYS_GPU=$PHYS_GPU N_PAIRS=5 MXFP8_WARMUP=30 MXFP8_ITERS=50 \
    PREHEAT_S=60 WARMUP_PAIRS=2 \
    python3 r32c_paired_bench.py 2>&1 | tee "$out"
  echo "Saved $out"
}

# ============== C1: 70B Down V2-CRR cp sweep ==============
echo "########## C1: 70B Down 4096x8192x28672 V2-CRR cp sweep ##########"
M=4096; N=8192; K=28672
for cp in 0 1 2 3; do
  build_so "c1_cp${cp}" crr $M $N $K "-DMXFP8_CRR_V2_SCALE_CACHEPOLICY=${cp}"
done
# Pair each cp against cp=0 baseline
for cp in 1 2 3; do
  paired_bench "C1" crr $M $N $K \
    "tk_mxfp8_r32c_c1_cp0" "cp0" \
    "tk_mxfp8_r32c_c1_cp${cp}" "cp${cp}"
done

# ============== C3: 8B Down V2-RCR cp sweep ==============
echo "########## C3: 8B Down 4096x4096x14336 V2-RCR cp sweep ##########"
M=4096; N=4096; K=14336
for cp in 0 1 2 3; do
  build_so "c3_cp${cp}" rcr $M $N $K "-DMXFP8_RCR_V2_SCALE_CACHEPOLICY=${cp}"
done
for cp in 1 2 3; do
  paired_bench "C3" rcr $M $N $K \
    "tk_mxfp8_r32c_c3_cp0" "cp0" \
    "tk_mxfp8_r32c_c3_cp${cp}" "cp${cp}"
done

# ============== C4: V2-RRR alternative for 70B Down ==============
echo "########## C4: 70B Down 4096x8192x28672 V2-RRR vs V2-CRR ##########"
# Build V2-RRR for 70B Down. CRR baseline already built above (c1_cp0).
M=4096; N=8192; K=28672
build_so "c4_rrr" rrr $M $N $K ""
# Use cross-layout BABA paired bench (separate harness)
HIP_VISIBLE_DEVICES=2 \
  M=$M N=$N K=$K \
  SO_CRR="$HERE/tk_mxfp8_r32c_c1_cp0${PYEXT}" MOD_CRR="tk_mxfp8_r32c_c1_cp0" \
  SO_RRR="$HERE/tk_mxfp8_r32c_c4_rrr${PYEXT}" MOD_RRR="tk_mxfp8_r32c_c4_rrr" \
  PHYS_GPU=$PHYS_GPU N_PAIRS=5 MXFP8_WARMUP=30 MXFP8_ITERS=50 \
  PREHEAT_S=60 WARMUP_PAIRS=2 \
  python3 r32c_c4_paired_bench.py 2>&1 | tee "$HERE/r32c_C4_rrr_vs_crr.txt"

echo "ALL C1/C3/C4 BENCHES DONE"
