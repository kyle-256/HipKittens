#!/bin/bash
# R48 Dev C Step 2: Sweep MXFP8_CRR_BLOCK_SWIZZLE_GROUP_M on 70B Gate/Up CRR.
# Hypothesis: wide-N shapes benefit from larger M-group (more B-tile reuse before
# eviction). Test GROUP_M ∈ {2, 4 (default), 8, 16}.
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"

GPU=${HIP_VISIBLE_DEVICES:-5}
WARMUP=${MXFP8_WARMUP:-50}
ITERS=${MXFP8_ITERS:-100}
RUNS=${RUNS:-3}
COOL=${COOL:-20}
REBUILD_COOL=${REBUILD_COOL:-60}
OUTDIR="r48c_groupm_sweep_results"
mkdir -p "$OUTDIR"

# 70B Gate/Up shape
M=4096
N=28672
K=8192

run_crr() {
    local GM=$1
    local TAG="gm${GM}"
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DMXFP8_CRR_BLOCK_SWIZZLE=1 -DMXFP8_CRR_BLOCK_SWIZZLE_GROUP_M=$GM" \
        > "$OUTDIR/${TAG}_build.log" 2>&1
    for r in $(seq 1 $RUNS); do
        HIP_VISIBLE_DEVICES=$GPU \
        MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=crr MXFP8_PRESHUFFLE_QUANT=1 \
        python3 test_mxfp8_python.py $M $N $K > "$OUTDIR/${TAG}_run${r}.log" 2>&1
        if [ $r -lt $RUNS ]; then sleep $COOL; fi
    done
}

GMS="${GMS:-2 4 8 16}"
first=1
for GM in $GMS; do
    if [ $first -eq 0 ]; then sleep $REBUILD_COOL; fi
    first=0
    echo "=== GROUP_M=$GM ==="
    run_crr $GM
done

echo ""
echo "========== R48 Dev C 70B Gate/Up CRR GROUP_M sweep (3-run avg) =========="
python3 - "$OUTDIR" "$GMS" <<'PYEOF'
import sys, re
D = sys.argv[1]
GMS = sys.argv[2].split()
def get_tf(p):
    try:
        t=open(p).read()
        m=re.search(r'TFLOPS:\s+([0-9.]+)', t)
        return float(m.group(1)) if m else None
    except: return None
def avg(xs):
    xs=[x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None
print(f"{'GROUP_M':>8} | {'avg TFLOPS':>10} | runs")
print("-"*60)
results={}
for GM in GMS:
    runs = [get_tf(f"{D}/gm{GM}_run{r}.log") for r in (1,2,3)]
    a = avg(runs)
    results[GM] = a
    runs_s = ",".join(f"{x:.1f}" if x else "N" for x in runs)
    print(f"{GM:>8} | {a if a else 0:10.1f} | [{runs_s}]")
if "4" in results and results["4"]:
    base = results["4"]
    print(f"\nΔ% vs GROUP_M=4 (default):")
    for GM in GMS:
        if GM != "4" and results[GM]:
            d = 100*(results[GM]-base)/base
            print(f"  GROUP_M={GM}: {d:+.2f}%")
PYEOF
