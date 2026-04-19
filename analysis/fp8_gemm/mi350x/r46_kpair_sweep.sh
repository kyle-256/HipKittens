#!/bin/bash
# R46: Sweep CRR k-pair loop (Dev C) across all 7 compute-bound shapes.
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2

GPU=${HIP_VISIBLE_DEVICES:-2}
WARMUP=50
ITERS=100
OUTDIR="r46_kpair_results"
mkdir -p "$OUTDIR"

declare -a SHAPES=("8192 8192 8192" "4096 4096 4096" "4096 14336 4096" "4096 4096 14336" "4096 8192 8192" "4096 28672 8192" "4096 8192 28672")
declare -a LABELS=("8192cube" "8B_QO" "8B_GateUp" "8B_Down" "70B_QO" "70B_GateUp" "70B_Down")

run_crr() {
    local M=$1 N=$2 K=$3 LABEL=$4 EXTRA_FLAGS=$5 TAG=$6
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $EXTRA_FLAGS" > /dev/null 2>&1
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
    run_crr $M $N $K $L "-DMXFP8_CRR_KPAIR_LOOP=1" kpair
done

python3 - "$OUTDIR" <<'PYEOF'
import sys, re, os
D = sys.argv[1]
labels = ["8192cube", "8B_QO", "8B_GateUp", "8B_Down", "70B_QO", "70B_GateUp", "70B_Down"]
names = {"8192cube":"8192³", "8B_QO":"8B Q/O", "8B_GateUp":"8B Gate/Up",
         "8B_Down":"8B Down", "70B_QO":"70B Q/O", "70B_GateUp":"70B Gate/Up",
         "70B_Down":"70B Down"}
def get_tf(p):
    try:
        return float(re.search(r'TFLOPS:\s+([0-9.]+)', open(p).read()).group(1))
    except: return None
print("\n========== CRR K-PAIR SUMMARY ==========")
print(f"{'Shape':<15} | {'Baseline':>9} | {'+KPAIR':>9} | {'Δ%':>7}")
print("-"*50)
for L in labels:
    b = get_tf(f"{D}/{L}_baseline.log")
    k = get_tf(f"{D}/{L}_kpair.log")
    if b and k:
        d = 100*(k-b)/b
        flag = " ★" if d >= 2.0 else (" XX" if d <= -2.0 else "")
        print(f"{names[L]:<15} | {b:9.1f} | {k:9.1f} | {d:+6.2f}%{flag}")
PYEOF
