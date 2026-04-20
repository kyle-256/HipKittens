#!/bin/bash
# R57 Dev A: Phase 2 bench — CRR_STEADY_VMCNT sweep
# Uses CPPFLAGS (not CXXFLAGS_EXTRA) for compile-time defines
set -euo pipefail

export HIP_VISIBLE_DEVICES=0
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2

WD=/shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x
RESULTS=$WD/r57a_results
cd "$WD"

VMCNT_VALUES="4 6 8 12"
RUNS=5
WARMUP=100
ITERS=200

echo "===== Phase 2: 5-run SCLK bench ====="
echo "Start: $(date)"

# ==== FP8 BASELINE for ratio calculations ====
echo ""
echo "===== FP8 Baseline: 70B Down CRR (4096x8192x28672) ====="
make -B TARGET=tk_fp8_layouts SRC=kernel_fp8_layouts.cpp \
  CPPFLAGS="-DM_DIM=4096 -DN_DIM=8192 -DK_DIM=28672" 2>&1 > /dev/null
sleep 60
> "$RESULTS/fp8_baseline_down.txt"
for RUN in $(seq 1 $RUNS); do
  echo "  FP8 baseline Down run $RUN/$RUNS"
  FP8_WARMUP=$WARMUP FP8_ITERS=$ITERS FP8_LAYOUTS=crr FP8_CHECK=0 \
    python3 test_python.py 4096 8192 28672 2>&1 | tee -a "$RESULTS/fp8_baseline_down.txt" | tail -3
  sleep 30
done

echo ""
echo "===== FP8 Baseline: 70B GateUp CRR (4096x28672x8192) ====="
make -B TARGET=tk_fp8_layouts SRC=kernel_fp8_layouts.cpp \
  CPPFLAGS="-DM_DIM=4096 -DN_DIM=28672 -DK_DIM=8192" 2>&1 > /dev/null
sleep 60
> "$RESULTS/fp8_baseline_gateup.txt"
for RUN in $(seq 1 $RUNS); do
  echo "  FP8 baseline GateUp run $RUN/$RUNS"
  FP8_WARMUP=$WARMUP FP8_ITERS=$ITERS FP8_LAYOUTS=crr FP8_CHECK=0 \
    python3 test_python.py 4096 28672 8192 2>&1 | tee -a "$RESULTS/fp8_baseline_gateup.txt" | tail -3
  sleep 30
done

# ==== MXFP8 sweep per vmcnt value ====
for V in $VMCNT_VALUES; do
  echo ""
  echo "========================================="
  echo "  MXFP8 CRR vmcnt=$V SWEEP"
  echo "========================================="

  # ---- 70B Down CRR ----
  echo "  Building mxfp8 vmcnt=$V for 70B Down CRR (4096x8192x28672)..."
  make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CPPFLAGS="-DM_DIM=4096 -DN_DIM=8192 -DK_DIM=28672 -DCRR_STEADY_VMCNT=$V" 2>&1 > /dev/null
  sleep 60
  > "$RESULTS/mxfp8_vmcnt${V}_down.txt"
  for RUN in $(seq 1 $RUNS); do
    echo "  vmcnt=$V Down run $RUN/$RUNS"
    MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_LAYOUTS=crr MXFP8_CHECK=0 \
      python3 test_mxfp8_python.py 4096 8192 28672 2>&1 | tee -a "$RESULTS/mxfp8_vmcnt${V}_down.txt" | tail -3
    sleep 30
  done

  # ---- 70B GateUp CRR ----
  echo "  Building mxfp8 vmcnt=$V for 70B GateUp CRR (4096x28672x8192)..."
  make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CPPFLAGS="-DM_DIM=4096 -DN_DIM=28672 -DK_DIM=8192 -DCRR_STEADY_VMCNT=$V" 2>&1 > /dev/null
  sleep 60
  > "$RESULTS/mxfp8_vmcnt${V}_gateup.txt"
  for RUN in $(seq 1 $RUNS); do
    echo "  vmcnt=$V GateUp run $RUN/$RUNS"
    MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_LAYOUTS=crr MXFP8_CHECK=0 \
      python3 test_mxfp8_python.py 4096 28672 8192 2>&1 | tee -a "$RESULTS/mxfp8_vmcnt${V}_gateup.txt" | tail -3
    sleep 30
  done
done

echo ""
echo "===== Phase 2 COMPLETE ====="
echo "End: $(date)"
echo "Results in: $RESULTS/"
ls -la "$RESULTS/"
