#!/bin/bash
# R48 Dev C Step 3: Sweep GROUP_M=8 across all 7 CRR shapes vs default GROUP_M=4.
# (GROUP_M=8 was the only positive on 70B Gate/Up at +0.66% — below SHIP gate.
# Confirm whether net positive or negative across the suite.)
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"

GPU=${HIP_VISIBLE_DEVICES:-5}
WARMUP=${MXFP8_WARMUP:-50}
ITERS=${MXFP8_ITERS:-100}
RUNS=${RUNS:-3}
COOL=${COOL:-20}
REBUILD_COOL=${REBUILD_COOL:-60}
OUTDIR="r48c_groupm_full_results"
mkdir -p "$OUTDIR"

declare -a SHAPES=("8192 8192 8192" "4096 4096 4096" "4096 14336 4096" "4096 4096 14336" "4096 8192 8192" "4096 28672 8192" "4096 8192 28672")
declare -a LABELS=("8192cube" "8B_QO" "8B_GateUp" "8B_Down" "70B_QO" "70B_GateUp" "70B_Down")

run_crr() {
    local M=$1 N=$2 K=$3 LABEL=$4 GM=$5
    local TAG="gm${GM}"
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DMXFP8_CRR_BLOCK_SWIZZLE=1 -DMXFP8_CRR_BLOCK_SWIZZLE_GROUP_M=$GM" \
        > "$OUTDIR/${LABEL}_${TAG}_build.log" 2>&1
    for r in $(seq 1 $RUNS); do
        HIP_VISIBLE_DEVICES=$GPU \
        MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=crr MXFP8_PRESHUFFLE_QUANT=1 \
        python3 test_mxfp8_python.py $M $N $K > "$OUTDIR/${LABEL}_${TAG}_run${r}.log" 2>&1
        if [ $r -lt $RUNS ]; then sleep $COOL; fi
    done
}

for i in "${!SHAPES[@]}"; do
    read -r M N K <<< "${SHAPES[$i]}"
    L="${LABELS[$i]}"
    echo "=== [$((i+1))/${#SHAPES[@]}] $L (${M}x${N}x${K}) ==="
    run_crr $M $N $K $L 4
    sleep $REBUILD_COOL
    run_crr $M $N $K $L 8
    if [ $i -lt $((${#SHAPES[@]} - 1)) ]; then sleep $REBUILD_COOL; fi
done

echo ""
echo "========== R48 Dev C GROUP_M=4 vs GROUP_M=8 (3-run avg) =========="
python3 - "$OUTDIR" <<'PYEOF'
import sys, re
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
def avg(xs):
    xs=[x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None
print(f"{'Shape':<15} | {'GM=4 avg':>9} | {'GM=8 avg':>9} | {'Δ%':>7}")
print("-"*55)
worst=999; positives=0
for L in labels:
    gm4 = avg([get_tf(f"{D}/{L}_gm4_run{r}.log") for r in (1,2,3)])
    gm8 = avg([get_tf(f"{D}/{L}_gm8_run{r}.log") for r in (1,2,3)])
    if gm4 and gm8:
        d = 100*(gm8-gm4)/gm4
        worst = min(worst, d)
        if d >= 1.0: positives += 1
        flag = " ★" if d >= 2.0 else (" XX" if d <= -2.0 else "")
        print(f"{names[L]:<15} | {gm4:9.1f} | {gm8:9.1f} | {d:+6.2f}%{flag}")
print(f"\nshapes ≥+1%: {positives}, worst Δ%: {worst:+.2f}%")
verdict = "SHIP" if (positives >= 1 and worst >= -2.0) else "REFUTED"
print(f"Verdict: {verdict}")
PYEOF
