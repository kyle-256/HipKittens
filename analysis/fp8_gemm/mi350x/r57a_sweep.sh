#!/bin/bash
# R57 Dev A: CRR_STEADY_VMCNT sweep for WAIT-DOMINANT bottleneck
# GPU 0 only, 5 runs per variant, 30s cooldown between runs, 60s between rebuilds
set -euo pipefail

export HIP_VISIBLE_DEVICES=0
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2

WD=/shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x
RESULTS=$WD/r57a_results
mkdir -p "$RESULTS"

VMCNT_VALUES="4 6 8 12"

# ==== PHASE 0: Build check (VGPR/spill) for each vmcnt value ====
echo "===== PHASE 0: Build & VGPR check ====="
for V in $VMCNT_VALUES; do
  echo ""
  echo "--- Building vmcnt=$V (70B Down CRR: M=4096 N=8192 K=28672) ---"
  cd "$WD"
  make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS_EXTRA="-DM_DIM=4096 -DN_DIM=8192 -DK_DIM=28672 -DCRR_STEADY_VMCNT=$V" \
    2>&1 | tee "$RESULTS/build_vmcnt${V}.log" | grep -E 'remark.*VGPRs|remark.*Spill|error' || true
  echo ""

  # Extract VGPR and spill info
  VGPR=$(grep -oP 'VGPRs:\s+\K\d+' "$RESULTS/build_vmcnt${V}.log" | tail -1 || echo "N/A")
  SPILL=$(grep -oP 'VGPRs Spill:\s+\K\d+' "$RESULTS/build_vmcnt${V}.log" | tail -1 || echo "0")
  echo "vmcnt=$V => VGPRs=$VGPR, Spill=$SPILL"
  echo "vmcnt=$V VGPRs=$VGPR Spill=$SPILL" >> "$RESULTS/phase0_vgpr.txt"

  if [[ "$SPILL" != "0" && "$SPILL" != "N/A" ]]; then
    echo "WARNING: vmcnt=$V causes VGPR spill=$SPILL, will skip bench"
  fi
done

echo ""
echo "===== PHASE 0 SUMMARY ====="
cat "$RESULTS/phase0_vgpr.txt"

echo ""
echo "===== PHASE 1: ISA inspection ====="
# Build each variant with -save-temps and verify s_waitcnt vmcnt(N) differs
for V in $VMCNT_VALUES; do
  echo ""
  echo "--- ISA check vmcnt=$V ---"
  cd "$WD"
  make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS_EXTRA="-DM_DIM=4096 -DN_DIM=8192 -DK_DIM=28672 -DCRR_STEADY_VMCNT=$V -save-temps" \
    2>&1 > /dev/null

  # Find the ISA file
  ISA_FILE=$(ls -t kernel_mxfp8_layouts-*.s 2>/dev/null | head -1)
  if [[ -z "$ISA_FILE" ]]; then
    ISA_FILE=$(ls -t *.s 2>/dev/null | head -1)
  fi

  if [[ -n "$ISA_FILE" ]]; then
    echo "ISA file: $ISA_FILE"
    # Count s_waitcnt vmcnt occurrences
    echo "s_waitcnt vmcnt patterns in ISA for vmcnt=$V:"
    grep -oP 's_waitcnt\s+vmcnt\(\d+\)' "$ISA_FILE" | sort | uniq -c | sort -rn | head -10 > "$RESULTS/isa_vmcnt${V}.txt"
    cat "$RESULTS/isa_vmcnt${V}.txt"
    cp "$ISA_FILE" "$RESULTS/isa_vmcnt${V}.s"
  else
    echo "WARNING: No ISA file found for vmcnt=$V"
  fi
done

echo ""
echo "===== PHASE 2: 5-run SCLK bench ====="

# First, build and bench FP8 baseline for ratio calculations
echo ""
echo "--- FP8 Baseline (70B Down CRR: M=4096 N=8192 K=28672) ---"
cd "$WD"
make -B TARGET=tk_fp8_layouts SRC=kernel_fp8_layouts.cpp \
  CXXFLAGS_EXTRA="-DM_DIM=4096 -DN_DIM=8192 -DK_DIM=28672" \
  2>&1 > /dev/null

for RUN in 1 2 3 4 5; do
  echo "FP8 baseline run $RUN/5 (70B Down CRR)"
  FP8_WARMUP=100 FP8_ITERS=200 FP8_LAYOUTS=crr FP8_CHECK=0 \
    python3 test_python.py 4096 8192 28672 2>&1 | tee -a "$RESULTS/fp8_baseline_down.txt" | tail -3
  sleep 30
done

echo ""
echo "--- FP8 Baseline (70B GateUp CRR: M=4096 N=28672 K=8192) ---"
cd "$WD"
make -B TARGET=tk_fp8_layouts SRC=kernel_fp8_layouts.cpp \
  CXXFLAGS_EXTRA="-DM_DIM=4096 -DN_DIM=28672 -DK_DIM=8192" \
  2>&1 > /dev/null
sleep 60

for RUN in 1 2 3 4 5; do
  echo "FP8 baseline run $RUN/5 (70B GateUp CRR)"
  FP8_WARMUP=100 FP8_ITERS=200 FP8_LAYOUTS=crr FP8_CHECK=0 \
    python3 test_python.py 4096 28672 8192 2>&1 | tee -a "$RESULTS/fp8_baseline_gateup.txt" | tail -3
  sleep 30
done

# Now sweep each vmcnt variant
for V in $VMCNT_VALUES; do
  # Check for spill - skip if nonzero
  SPILL=$(grep "vmcnt=$V " "$RESULTS/phase0_vgpr.txt" | grep -oP 'Spill=\K\d+' || echo "0")
  if [[ "$SPILL" != "0" ]]; then
    echo ""
    echo "SKIPPING vmcnt=$V (VGPR spill=$SPILL)"
    continue
  fi

  echo ""
  echo "========================================="
  echo "  MXFP8 CRR vmcnt=$V SWEEP"
  echo "========================================="

  # Build for 70B Down CRR
  echo "Building mxfp8 vmcnt=$V for 70B Down CRR..."
  cd "$WD"
  make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS_EXTRA="-DM_DIM=4096 -DN_DIM=8192 -DK_DIM=28672 -DCRR_STEADY_VMCNT=$V" \
    2>&1 > /dev/null
  sleep 60

  # 5 runs for 70B Down CRR
  echo "" > "$RESULTS/mxfp8_vmcnt${V}_down.txt"
  for RUN in 1 2 3 4 5; do
    echo "vmcnt=$V - 70B Down CRR run $RUN/5"
    MXFP8_WARMUP=100 MXFP8_ITERS=200 MXFP8_LAYOUTS=crr MXFP8_CHECK=0 \
      python3 test_mxfp8_python.py 4096 8192 28672 2>&1 | tee -a "$RESULTS/mxfp8_vmcnt${V}_down.txt" | tail -3
    sleep 30
  done

  # Build for 70B GateUp CRR (different shape)
  echo "Building mxfp8 vmcnt=$V for 70B GateUp CRR..."
  cd "$WD"
  make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS_EXTRA="-DM_DIM=4096 -DN_DIM=28672 -DK_DIM=8192 -DCRR_STEADY_VMCNT=$V" \
    2>&1 > /dev/null
  sleep 60

  # 5 runs for 70B GateUp CRR
  echo "" > "$RESULTS/mxfp8_vmcnt${V}_gateup.txt"
  for RUN in 1 2 3 4 5; do
    echo "vmcnt=$V - 70B GateUp CRR run $RUN/5"
    MXFP8_WARMUP=100 MXFP8_ITERS=200 MXFP8_LAYOUTS=crr MXFP8_CHECK=0 \
      python3 test_mxfp8_python.py 4096 28672 8192 2>&1 | tee -a "$RESULTS/mxfp8_vmcnt${V}_gateup.txt" | tail -3
    sleep 30
  done
done

echo ""
echo "===== ALL SWEEPS COMPLETE ====="
echo "Results in: $RESULTS/"
ls -la "$RESULTS/"
