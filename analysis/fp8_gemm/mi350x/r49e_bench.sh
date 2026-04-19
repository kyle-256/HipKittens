#!/bin/bash
# R49 Dev E: Re-evaluate dormant CRR variants under strict SCLK protocol.
# - hbnshrink (MXFP8_CRR_BLK_N=128) on 8B Gate/Up + 70B Gate/Up + 70B Down
# - rect      (MXFP8_RECT_BLK_N=64)  on 8B Gate/Up + 70B Gate/Up + 70B Down
# - baseline  (no variant macro) on each of the 3 cells
# 5 runs per cell, 30s cooldown between runs, 60s rebuild cooldown.
# warpsm4 is SKIPPED — its dispatcher wiring is not in place ("Stage A2-A4
# deliverable" per the variant header) and the prompt says do not introduce
# dispatch changes.
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"

GPU=${HIP_VISIBLE_DEVICES:-0}  # R49 Dev E note: orchestrator said GPU 6 but GPU 6 was contested by r48_full_bench_resume; switched to GPU 0 (verified idle).
WARMUP=${MXFP8_WARMUP:-50}
ITERS=${MXFP8_ITERS:-100}
RUNS=${RUNS:-5}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}
OUTDIR="r49e_results"
mkdir -p "$OUTDIR"

# Cells (label : M N K)
CELL_LABELS=(8B_GateUp 70B_GateUp 70B_Down)
CELL_MNKS=("4096 14336 4096" "4096 28672 8192" "4096 4096 14336")

# Variants:
#   "label : extra_cxxflags"
VARIANT_LABELS=(baseline hbnshrink rect)
VARIANT_FLAGS=(
    ""
    "-DMXFP8_CRR_BLK_N=128"
    "-DMXFP8_RECT_BLK_N=64"
)

run_cell() {
    local LABEL=$1
    local M=$2
    local N=$3
    local K=$4
    local VLABEL=$5
    local VFLAGS=$6
    local TAG="${VLABEL}_${LABEL}"

    rm -f tk_mxfp8_layouts*.so
    echo ">> Build $TAG  M=$M N=$N K=$K  flags='$VFLAGS'"
    if ! make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $VFLAGS" \
        > "$OUTDIR/${TAG}_build.log" 2>&1; then
        echo "BUILD FAILED for $TAG — see $OUTDIR/${TAG}_build.log"
        return 1
    fi
    for r in $(seq 1 $RUNS); do
        HIP_VISIBLE_DEVICES=$GPU \
        MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=crr MXFP8_PRESHUFFLE_QUANT=1 \
        python3 test_mxfp8_python.py $M $N $K \
            > "$OUTDIR/${TAG}_run${r}.log" 2>&1 || echo "run $r exited non-zero"
        if [ $r -lt $RUNS ]; then sleep $COOL; fi
    done
}

for vi in 0 1 2; do
    VLABEL=${VARIANT_LABELS[$vi]}
    VFLAGS=${VARIANT_FLAGS[$vi]}
    for ci in 0 1 2; do
        CLABEL=${CELL_LABELS[$ci]}
        read -r M N K <<<"${CELL_MNKS[$ci]}"
        echo ""
        echo "===== Variant=$VLABEL Cell=$CLABEL ($M x $N x $K) ====="
        run_cell "$CLABEL" "$M" "$N" "$K" "$VLABEL" "$VFLAGS"
        echo "Sleeping ${REBUILD_COOL}s for rebuild cooldown..."
        sleep $REBUILD_COOL
    done
done

echo ""
echo "========== R49 Dev E summary =========="
python3 - "$OUTDIR" <<'PYEOF'
import sys, re, statistics, os
D = sys.argv[1]
def get_tf(p):
    try:
        t = open(p).read()
        m = re.search(r'TFLOPS:\s+([0-9.]+)', t)
        return float(m.group(1)) if m else None
    except: return None

cells = ["8B_GateUp", "70B_GateUp", "70B_Down"]
variants = ["baseline", "hbnshrink", "rect"]

print(f"{'Cell':<12} {'Variant':<10} {'med':>7} {'min':>7} {'max':>7} {'mean':>7} {'spread%':>8} {'runs':<40}")
print("-" * 110)
results = {}
for c in cells:
    for v in variants:
        runs = [get_tf(f"{D}/{v}_{c}_run{r}.log") for r in range(1,6)]
        runs = [r for r in runs if r is not None]
        if not runs:
            print(f"{c:<12} {v:<10} {'N/A':>7}")
            continue
        med = statistics.median(runs)
        lo, hi = min(runs), max(runs)
        mn = statistics.mean(runs)
        spread = (hi-lo)/med*100 if med else 0
        results[(c,v)] = med
        runs_s = " ".join(f"{r:.1f}" for r in runs)
        print(f"{c:<12} {v:<10} {med:>7.2f} {lo:>7.2f} {hi:>7.2f} {mn:>7.2f} {spread:>7.2f}% {runs_s}")

print()
print("Delta vs baseline (median):")
for c in cells:
    base = results.get((c,"baseline"))
    if not base: continue
    for v in ("hbnshrink","rect"):
        treat = results.get((c,v))
        if not treat:
            print(f"  {c:<12} {v:<10} N/A")
            continue
        d = 100*(treat-base)/base
        verdict = "SHIP" if d >= 2.0 else ("MARGINAL" if d >= 0.5 else ("NEUTRAL" if d > -1.0 else "REGRESSION"))
        print(f"  {c:<12} {v:<10} base={base:.2f} treat={treat:.2f}  Δ={d:+.2f}% [{verdict}]")
PYEOF
