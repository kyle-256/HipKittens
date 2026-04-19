#!/bin/bash
# R47 Dev A: Sweep RCR XCD swizzle (port from R46 Dev D RRR) across all 7
# compute-bound shapes. Compare baseline vs +SWIZZLE; report TFLOPS Δ%.
# Required: SNR ≥ 45 dB + det 3/3 PASS on at least 70B Gate/Up RCR + 8192³ RCR
# before treating wins as real (validated separately by re-running with
# MXFP8_CHECK=1 on the gating shapes after the perf sweep).
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2

GPU=${HIP_VISIBLE_DEVICES:-0}
WARMUP=50
ITERS=100
OUTDIR="r47a_swizzle_results"
mkdir -p "$OUTDIR"

declare -a SHAPES=("8192 8192 8192" "4096 4096 4096" "4096 14336 4096" "4096 4096 14336" "4096 8192 8192" "4096 28672 8192" "4096 8192 28672")
declare -a LABELS=("8192cube" "8B_QO" "8B_GateUp" "8B_Down" "70B_QO" "70B_GateUp" "70B_Down")

run_rcr() {
    local M=$1 N=$2 K=$3 LABEL=$4 EXTRA_FLAGS=$5 TAG=$6
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $EXTRA_FLAGS" > "$OUTDIR/${LABEL}_${TAG}_build.log" 2>&1
    HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
    MXFP8_LAYOUTS=rcr MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py $M $N $K > "$OUTDIR/${LABEL}_${TAG}.log" 2>&1
}

for i in "${!SHAPES[@]}"; do
    read -r M N K <<< "${SHAPES[$i]}"
    L="${LABELS[$i]}"
    echo "=== [$((i+1))/${#SHAPES[@]}] $L (${M}x${N}x${K}) ==="
    run_rcr $M $N $K $L "" baseline
    run_rcr $M $N $K $L "-DMXFP8_RCR_BLOCK_SWIZZLE=1" swizzle
    BASE_TF=$(grep TFLOPS "$OUTDIR/${L}_baseline.log" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    SWIZ_TF=$(grep TFLOPS "$OUTDIR/${L}_swizzle.log" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    echo "  RCR baseline: $BASE_TF TF | +SWIZZLE: $SWIZ_TF TF"
done

echo ""
echo "========== R47A RCR SWIZZLE SUMMARY =========="
python3 - "$OUTDIR" <<'PYEOF'
import sys, re, os
D = sys.argv[1]
labels = ["8192cube", "8B_QO", "8B_GateUp", "8B_Down", "70B_QO", "70B_GateUp", "70B_Down"]
names = {"8192cube":"8192 cube", "8B_QO":"8B Q/O", "8B_GateUp":"8B Gate/Up",
         "8B_Down":"8B Down", "70B_QO":"70B Q/O", "70B_GateUp":"70B Gate/Up",
         "70B_Down":"70B Down"}
def get_tf(p):
    try:
        t = open(p).read()
        m = re.search(r'TFLOPS:\s+([0-9.]+)', t)
        return float(m.group(1)) if m else None
    except: return None
print(f"{'Shape':<15} | {'Baseline':>9} | {'+Swizzle':>9} | {'D%':>7}")
print("-"*50)
deltas = []
for L in labels:
    b = get_tf(f"{D}/{L}_baseline.log")
    s = get_tf(f"{D}/{L}_swizzle.log")
    if b and s:
        d = 100*(s-b)/b
        deltas.append((L, d))
        flag = " WIN" if d >= 2.0 else (" REG" if d <= -2.0 else "")
        print(f"{names[L]:<15} | {b:9.1f} | {s:9.1f} | {d:+6.2f}%{flag}")
if deltas:
    worst = min(d for _,d in deltas)
    n_pos = sum(1 for _,d in deltas if d >= 1.0)
    n_neg1 = sum(1 for _,d in deltas if d <= -1.0)
    print()
    print(f"Worst regression: {worst:+.2f}%  | shapes >= +1%: {n_pos}  | shapes <= -1%: {n_neg1}")
    if worst >= -1.0 and n_pos >= 2:
        print("DECISION: net-positive (rule met) -> default ON candidate")
    else:
        print("DECISION: not net-positive -> leave default OFF")
PYEOF
