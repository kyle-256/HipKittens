#!/bin/bash
# R49 Dev D Phase 1: Strict-SCLK re-validation of R48 Dev C +5.47% on
# 8B Gate/Up CRR (M=4096, N=14336, K=4096). 5 runs/cell, 30s cooldown,
# 60s rebuild cooldown. Single-shape only (GM=4 baseline vs GM=8 treatment).
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"

GPU=${HIP_VISIBLE_DEVICES:-7}
WARMUP=${MXFP8_WARMUP:-50}
ITERS=${MXFP8_ITERS:-100}
RUNS=${RUNS:-5}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}
OUTDIR="r49d_phase1_results"
mkdir -p "$OUTDIR"

# 8B Gate/Up shape
M=4096
N=14336
K=4096
LABEL="8B_GateUp"

run_crr() {
    local GM=$1
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

echo "=== Phase 1 baseline GM=4 ==="
run_crr 4
echo "Sleeping ${REBUILD_COOL}s for rebuild cooldown..."
sleep $REBUILD_COOL
echo "=== Phase 1 treatment GM=8 ==="
run_crr 8

echo ""
echo "========== R49 Dev D Phase 1: 8B Gate/Up CRR (5-run) =========="
python3 - "$OUTDIR" "$LABEL" <<'PYEOF'
import sys, re, statistics
D, L = sys.argv[1], sys.argv[2]
def get_tf(p):
    try:
        t = open(p).read()
        m = re.search(r'TFLOPS:\s+([0-9.]+)', t)
        return float(m.group(1)) if m else None
    except: return None
def stats(xs):
    xs=[x for x in xs if x is not None]
    if not xs: return (None, None, None, None)
    return (statistics.median(xs), min(xs), max(xs), statistics.mean(xs))
runs4 = [get_tf(f"{D}/{L}_gm4_run{r}.log") for r in range(1,6)]
runs8 = [get_tf(f"{D}/{L}_gm8_run{r}.log") for r in range(1,6)]
m4, lo4, hi4, mn4 = stats(runs4)
m8, lo8, hi8, mn8 = stats(runs8)
print(f"GM=4 runs : {[f'{x:.1f}' if x else 'N' for x in runs4]}")
print(f"  median={m4:.2f} min={lo4:.2f} max={hi4:.2f} mean={mn4:.2f} spread={(hi4-lo4)/m4*100:.2f}%")
print(f"GM=8 runs : {[f'{x:.1f}' if x else 'N' for x in runs8]}")
print(f"  median={m8:.2f} min={lo8:.2f} max={hi8:.2f} mean={mn8:.2f} spread={(hi8-lo8)/m8*100:.2f}%")
d_med = 100*(m8-m4)/m4
d_mean = 100*(mn8-mn4)/mn4
print(f"\nΔ median GM=8 vs GM=4: {d_med:+.2f}%")
print(f"Δ mean   GM=8 vs GM=4: {d_mean:+.2f}%")
verdict = "PROCEED" if d_med >= 3.0 else ("MARGINAL" if d_med >= 1.5 else "REFUTED")
print(f"Phase 1 verdict: {verdict}")
PYEOF
