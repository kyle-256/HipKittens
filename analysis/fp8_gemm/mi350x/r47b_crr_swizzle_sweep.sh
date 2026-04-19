#!/bin/bash
# R47 Dev B: Sweep CRR XCD-aware block swizzle across all 7 compute-bound shapes.
# Compare baseline vs +SWIZZLE; report MX/FP8 ratio improvement.
# Mirrors r46_swizzle_sweep.sh but for CRR layout.
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"

GPU=${HIP_VISIBLE_DEVICES:-1}
WARMUP=${MXFP8_WARMUP:-50}
ITERS=${MXFP8_ITERS:-100}
OUTDIR="r47b_crr_swizzle_results"
mkdir -p "$OUTDIR"

declare -a SHAPES=("8192 8192 8192" "4096 4096 4096" "4096 14336 4096" "4096 4096 14336" "4096 8192 8192" "4096 28672 8192" "4096 8192 28672")
declare -a LABELS=("8192cube" "8B_QO" "8B_GateUp" "8B_Down" "70B_QO" "70B_GateUp" "70B_Down")

run_crr() {
    local M=$1 N=$2 K=$3 LABEL=$4 EXTRA_FLAGS=$5 TAG=$6
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $EXTRA_FLAGS" > "$OUTDIR/${LABEL}_${TAG}_build.log" 2>&1
    HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
    MXFP8_LAYOUTS=crr MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py $M $N $K > "$OUTDIR/${LABEL}_${TAG}.log" 2>&1
}

for i in "${!SHAPES[@]}"; do
    read -r M N K <<< "${SHAPES[$i]}"
    L="${LABELS[$i]}"
    echo "=== [$((i+1))/${#SHAPES[@]}] $L (${M}x${N}x${K}) ==="
    run_crr $M $N $K $L "" baseline
    run_crr $M $N $K $L "-DMXFP8_CRR_BLOCK_SWIZZLE=1" swizzle
    BASE_TF=$(grep TFLOPS "$OUTDIR/${L}_baseline.log" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    SWIZ_TF=$(grep TFLOPS "$OUTDIR/${L}_swizzle.log" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    echo "  CRR baseline: $BASE_TF TF | +SWIZZLE: $SWIZ_TF TF"
done

echo ""
echo "========== R47 Dev B CRR SWIZZLE SUMMARY =========="
python3 - "$OUTDIR" <<'PYEOF'
import sys, re, os
D = sys.argv[1]
labels = ["8192cube", "8B_QO", "8B_GateUp", "8B_Down", "70B_QO", "70B_GateUp", "70B_Down"]
names = {"8192cube":"8192³", "8B_QO":"8B Q/O", "8B_GateUp":"8B Gate/Up",
         "8B_Down":"8B Down", "70B_QO":"70B Q/O", "70B_GateUp":"70B Gate/Up",
         "70B_Down":"70B Down"}
def get_tf(p):
    try:
        t = open(p).read()
        m = re.search(r'TFLOPS:\s+([0-9.]+)', t)
        return float(m.group(1)) if m else None
    except: return None
print(f"{'Shape':<15} | {'Baseline':>9} | {'+Swizzle':>9} | {'Δ%':>7}")
print("-"*50)
positives = 0
worst = 999.0
for L in labels:
    b = get_tf(f"{D}/{L}_baseline.log")
    s = get_tf(f"{D}/{L}_swizzle.log")
    if b and s:
        d = 100*(s-b)/b
        if d >= 1.0: positives += 1
        worst = min(worst, d)
        flag = " ★" if d >= 2.0 else (" XX" if d <= -2.0 else "")
        print(f"{names[L]:<15} | {b:9.1f} | {s:9.1f} | {d:+6.2f}%{flag}")
print(f"\nDecision criteria: ≥2 shapes with ≥+1%, worst ≥-1%")
print(f"  shapes ≥+1%: {positives}, worst Δ%: {worst:+.2f}%")
verdict = "SHIP" if (positives >= 2 and worst >= -1.0) else "REFUTED"
print(f"  Verdict: {verdict}")
PYEOF
