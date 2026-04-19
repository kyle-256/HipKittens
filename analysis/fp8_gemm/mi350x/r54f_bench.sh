#!/bin/bash
# R54 Dev F: XCD swizzle granularity sweep at 70B Gate/Up CRR (Phase 1).
# Strict-SCLK: 5 runs/cell, MXFP8_WARMUP=100, MXFP8_ITERS=200,
# 30 s cool between runs, 60 s rebuild_cool between tags.
# GPU 1 isolation (verify --showpids empty before start).

set -euo pipefail
cd "$(dirname "$0")/r54f_workspace"
export THUNDERKITTENS_ROOT="$(cd ../../../.. && pwd)"

GPU=${HIP_VISIBLE_DEVICES:-1}
WARMUP=${MXFP8_WARMUP:-100}
ITERS=${MXFP8_ITERS:-200}
RUNS=${RUNS:-5}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}
OUTDIR="../r54f_results"
mkdir -p "$OUTDIR"

M=4096; N=28672; K=8192   # 70B Gate/Up CRR

# tag, extra cxx flags
run_tag() {
    local TAG=$1; local FLAGS=$2
    echo "  --> TAG=$TAG  FLAGS=\"$FLAGS\""
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $FLAGS" \
        > "$OUTDIR/${TAG}_build.log" 2>&1
    for r in $(seq 1 $RUNS); do
        HIP_VISIBLE_DEVICES=$GPU \
        MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=crr MXFP8_PRESHUFFLE_QUANT=1 \
        python3 test_mxfp8_python.py $M $N $K \
            > "$OUTDIR/${TAG}_run${r}.log" 2>&1
        if [ $r -lt $RUNS ]; then sleep $COOL; fi
    done
    sleep $REBUILD_COOL
}

echo "=== Phase 1: 70B Gate/Up CRR XCD-swizzle granularity sweep ==="
# Anchor: default (= -DMXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS=8 + GROUP_M=4)
run_tag "default"      ""

# NUM_XCDS sweep, GROUP_M=4 fixed (default).
run_tag "xcds04_gm04"  "-DMXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS=4 -DMXFP8_CRR_BLOCK_SWIZZLE_GROUP_M=4"
# xcds08_gm04 == default (skip)
run_tag "xcds16_gm04"  "-DMXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS=16 -DMXFP8_CRR_BLOCK_SWIZZLE_GROUP_M=4"
run_tag "xcds32_gm04"  "-DMXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS=32 -DMXFP8_CRR_BLOCK_SWIZZLE_GROUP_M=4"

# GROUP_M sweep, NUM_XCDS=8 fixed (default).
run_tag "xcds08_gm01"  "-DMXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS=8 -DMXFP8_CRR_BLOCK_SWIZZLE_GROUP_M=1"
run_tag "xcds08_gm02"  "-DMXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS=8 -DMXFP8_CRR_BLOCK_SWIZZLE_GROUP_M=2"
run_tag "xcds08_gm08"  "-DMXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS=8 -DMXFP8_CRR_BLOCK_SWIZZLE_GROUP_M=8"
run_tag "xcds08_gm16"  "-DMXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS=8 -DMXFP8_CRR_BLOCK_SWIZZLE_GROUP_M=16"

echo ""
echo "============== R54 Dev F Phase 1 SUMMARY (70B Gate/Up CRR) =============="
python3 - "$OUTDIR" <<'PYEOF'
import sys, re, glob, statistics
D = sys.argv[1]
TAGS = ["default",
        "xcds04_gm04","xcds16_gm04","xcds32_gm04",
        "xcds08_gm01","xcds08_gm02","xcds08_gm08","xcds08_gm16"]
def get_tf(p):
    try:
        t = open(p).read()
        m = re.search(r'TFLOPS:\s+([0-9.]+)', t)
        return float(m.group(1)) if m else None
    except Exception:
        return None
def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return (None,None,None,None)
    return (statistics.median(xs), min(xs), max(xs), statistics.mean(xs))

base = stats([get_tf(f"{D}/default_run{r}.log") for r in range(1,6)])
mb = base[0]
FP8_REF = 2807.1   # R53 Reviewer 70B Gate/Up CRR FP8 baseline

print(f"{'TAG':<14} | {'med':>8} {'min':>8} {'max':>8} {'spread':>7} | {'MX/FP8':>7} | {'vs default':>11}")
print("-"*90)
for tag in TAGS:
    runs = [get_tf(f"{D}/{tag}_run{r}.log") for r in range(1,6)]
    m, lo, hi, mn = stats(runs)
    if m is None:
        print(f"{tag:<14} | MISSING")
        continue
    sp = 100*(hi-lo)/m
    mxratio = 100*m/FP8_REF
    delt = 100*(m-mb)/mb if mb else 0.0
    flag = ""
    if delt >= 1.0: flag = " <-- WIN"
    if delt <= -1.0: flag = " <-- REGR"
    print(f"{tag:<14} | {m:8.1f} {lo:8.1f} {hi:8.1f} {sp:6.2f}% | {mxratio:6.2f}% | {delt:+10.2f}%{flag}")
PYEOF
