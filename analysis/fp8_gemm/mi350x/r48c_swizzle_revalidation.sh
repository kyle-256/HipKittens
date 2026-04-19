#!/bin/bash
# R48 Dev C Step 1: CRR swizzle ON vs OFF re-validation across 7 shapes.
# 3 runs each with 20s cooldown between runs, 60s between shape rebuilds.
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"

GPU=${HIP_VISIBLE_DEVICES:-5}
WARMUP=${MXFP8_WARMUP:-50}
ITERS=${MXFP8_ITERS:-100}
RUNS=${RUNS:-3}
COOL=${COOL:-20}
REBUILD_COOL=${REBUILD_COOL:-60}
OUTDIR="r48c_swizzle_revalidation_results"
mkdir -p "$OUTDIR"

declare -a SHAPES=("8192 8192 8192" "4096 4096 4096" "4096 14336 4096" "4096 4096 14336" "4096 8192 8192" "4096 28672 8192" "4096 8192 28672")
declare -a LABELS=("8192cube" "8B_QO" "8B_GateUp" "8B_Down" "70B_QO" "70B_GateUp" "70B_Down")

run_crr() {
    local M=$1 N=$2 K=$3 LABEL=$4 SWIZ=$5
    local TAG="swiz${SWIZ}"
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DMXFP8_CRR_BLOCK_SWIZZLE=$SWIZ" \
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
    run_crr $M $N $K $L 0
    sleep $REBUILD_COOL
    run_crr $M $N $K $L 1
    if [ $i -lt $((${#SHAPES[@]} - 1)) ]; then sleep $REBUILD_COOL; fi
done

echo ""
echo "========== R48 Dev C swizzle ON vs OFF (3-run avg) =========="
python3 - "$OUTDIR" <<'PYEOF'
import sys, re, os, glob
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
print(f"{'Shape':<15} | {'OFF avg':>9} | {'ON avg':>9} | {'Δ% (ON−OFF)':>12} | runs(off,on)")
print("-"*85)
for L in labels:
    off = [get_tf(f"{D}/{L}_swiz0_run{r}.log") for r in (1,2,3)]
    on  = [get_tf(f"{D}/{L}_swiz1_run{r}.log") for r in (1,2,3)]
    ao = avg(off); an = avg(on)
    if ao and an:
        d = 100*(an-ao)/ao
        flag = " ★" if d >= 2.0 else (" XX" if d <= -2.0 else "")
        runs_off = ",".join(f"{x:.1f}" if x else "N" for x in off)
        runs_on = ",".join(f"{x:.1f}" if x else "N" for x in on)
        print(f"{names[L]:<15} | {ao:9.1f} | {an:9.1f} | {d:+11.2f}%{flag} | off=[{runs_off}] on=[{runs_on}]")
PYEOF
