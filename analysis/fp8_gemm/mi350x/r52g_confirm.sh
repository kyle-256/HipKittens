#!/bin/bash
# R52 Dev G confirmation re-run: just the 70B Gate/Up CRR pair, more runs,
# to break the verdict tie introduced by transient GPU contention in the
# main bench (run 3 of POST and runs 1+3 of PRE were heavily slowed).
# 8 runs per arm, same SCLK protocol.

set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"

GPU=${HIP_VISIBLE_DEVICES:-5}
WARMUP=${MXFP8_WARMUP:-100}
ITERS=${MXFP8_ITERS:-200}
RUNS=${RUNS:-8}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}
OUTDIR="r52g_results"
mkdir -p "$OUTDIR"

run_cell() {
    local LABEL=$1 M=$2 N=$3 K=$4 LAYOUT=$5 TAG=$6 EXTRA=$7
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $EXTRA" \
        > "$OUTDIR/${LABEL}_${TAG}_build.log" 2>&1
    for r in $(seq 1 $RUNS); do
        HIP_VISIBLE_DEVICES=$GPU \
        MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=$LAYOUT MXFP8_PRESHUFFLE_QUANT=1 \
        python3 test_mxfp8_python.py $M $N $K \
            > "$OUTDIR/${LABEL}_${TAG}_run${r}.log" 2>&1
        if [ $r -lt $RUNS ]; then sleep $COOL; fi
    done
}

echo "=== R52 Dev G confirmation: 70B Gate/Up CRR, 8 runs/arm ==="
echo "  --> 70B_GateUp_CRR  POST  (swizzle OFF — patch in effect)"
run_cell "CONFIRM_70B_GateUp_CRR" 4096 28672 8192 crr "post" "-DMXFP8_CRR_BLOCK_SWIZZLE=0"
echo "      sleep ${REBUILD_COOL}s"
sleep $REBUILD_COOL
echo "  --> 70B_GateUp_CRR  PRE   (swizzle ON  — current HEAD)"
run_cell "CONFIRM_70B_GateUp_CRR" 4096 28672 8192 crr "pre" "-DMXFP8_CRR_BLOCK_SWIZZLE=1"

python3 - "$OUTDIR" <<'PYEOF'
import sys, re, statistics
D = sys.argv[1]
def get_tf(p):
    try:
        m = re.search(r'TFLOPS:\s+([0-9.]+)', open(p).read())
        return float(m.group(1)) if m else None
    except: return None
def report(label, n):
    runs = [get_tf(f"{D}/{label}_{n}_run{r}.log") for r in range(1, 9)]
    runs = [r for r in runs if r is not None]
    s = sorted(runs)
    cleaned = [r for r in runs if r > max(runs) * 0.8]
    cs = sorted(cleaned)
    print(f"{label} {n}:")
    print(f"  raw med={statistics.median(s):.1f} count={len(s)} all={s}")
    print(f"  cleaned med={statistics.median(cs):.1f} count={len(cs)} all={cs}")

report("CONFIRM_70B_GateUp_CRR", "post")
report("CONFIRM_70B_GateUp_CRR", "pre")
PYEOF
