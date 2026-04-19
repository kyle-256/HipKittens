#!/bin/bash
# R52 Reviewer Phase 2 (focused): re-audit 3 highest-spread R49 cells where
# the verdict could flip due to noise. Strict-SCLK A/B.
set -uo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"

GPU=${HIP_VISIBLE_DEVICES:-7}
WARMUP=${MXFP8_WARMUP:-100}
ITERS=${MXFP8_ITERS:-200}
RUNS=${RUNS:-5}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}
OUTDIR="r52_reviewer_phase2_results"
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

run_pair() {
    local LABEL=$1 M=$2 N=$3 K=$4 LAYOUT=$5 OFF_FLAG=$6 ON_FLAG=$7
    echo "  --> $LABEL ($M x $N x $K) $LAYOUT  WITHOUT"
    run_cell "$LABEL" "$M" "$N" "$K" "$LAYOUT" "off" "$OFF_FLAG"
    echo "      sleep ${REBUILD_COOL}s"
    sleep $REBUILD_COOL
    echo "  --> $LABEL ($M x $N x $K) $LAYOUT  WITH"
    run_cell "$LABEL" "$M" "$N" "$K" "$LAYOUT" "on" "$ON_FLAG"
    echo "      sleep ${REBUILD_COOL}s"
    sleep $REBUILD_COOL
}

# Top 3 highest-spread R49 cells where the verdict could flip
echo "=== B_70B_Down_CRR (R49: CONFIRMED, OFF spread 70.94%) ==="
run_pair "B_70B_Down_CRR"   4096 8192 28672 crr "-DMXFP8_CRR_BLOCK_SWIZZLE=0" "-DMXFP8_CRR_BLOCK_SWIZZLE=1"
echo "=== B_8B_Down_CRR (R49: NOISE, OFF spread 94.63%) ==="
run_pair "B_8B_Down_CRR"    4096 4096 14336 crr "-DMXFP8_CRR_BLOCK_SWIZZLE=0" "-DMXFP8_CRR_BLOCK_SWIZZLE=1"
echo "=== C_70B_GateUp_CRR (R49: CONFIRMED, OFF spread 14.77%) ==="
run_pair "C_70B_GateUp_CRR" 4096 28672 8192 crr "-DMXFP8_CRR_V2_SCALE_CACHEPOLICY=2" "-DMXFP8_CRR_V2_SCALE_CACHEPOLICY=0"

echo ""
echo "========== R52 REVIEWER PHASE 2 (focused) — strict-SCLK A/B summary =========="
python3 - "$OUTDIR" $RUNS <<'PYEOF'
import sys, re, statistics
D = sys.argv[1]; RUNS = int(sys.argv[2])
def get_tf(p):
    try:
        t = open(p).read()
        m = re.search(r'TFLOPS:\s+([0-9.]+)', t)
        return float(m.group(1)) if m else None
    except: return None
def stats(xs):
    xs=[x for x in xs if x is not None]
    if not xs: return (None, None, None, None)
    return (statistics.median(xs), min(xs), max(xs), xs)

# R49 prior (delta_pct, claim_pct, verdict)
R49 = {
    "B_70B_Down_CRR":   ( +4.84, 3.35, "CONFIRMED"),
    "B_8B_Down_CRR":    ( +0.37, 1.13, "NOISE"),
    "C_70B_GateUp_CRR": ( +9.74, 1.52, "CONFIRMED"),
}

print(f"{'Cell':<24} | {'OFF (med, sp%)':>22} | {'ON (med, sp%)':>22} | {'real %':>7} | {'claim':>6} | {'R52':>11} | R49")
print("-"*150)
for label, (d49, claim, v49) in R49.items():
    off = [get_tf(f"{D}/{label}_off_run{r}.log") for r in range(1, RUNS+1)]
    on  = [get_tf(f"{D}/{label}_on_run{r}.log")  for r in range(1, RUNS+1)]
    mo, lo_o, hi_o, off_xs = stats(off)
    mn, lo_n, hi_n, on_xs  = stats(on)
    if mo is None or mn is None:
        print(f"{label:<24} | MISSING DATA")
        continue
    d = 100*(mn-mo)/mo
    sp_o = (hi_o-lo_o)/mo*100 if mo else 0
    sp_n = (hi_n-lo_n)/mn*100 if mn else 0
    ratio = d/claim if claim != 0 else float('nan')
    if d < -0.5 and claim > 0:
        v52 = "REGRESSION"
    elif abs(d) < max(sp_o, sp_n) and abs(d) < 0.5:
        v52 = "NOISE"
    elif ratio < 0.25:
        v52 = "NOISE"
    elif ratio < 0.75:
        v52 = "INFLATED"
    else:
        v52 = "CONFIRMED"
    flip = "" if v52 == v49 else f"  <-- FLIP from {v49}"
    print(f"{label:<24} | {mo:7.1f} sp={sp_o:5.2f}% {str(off_xs):>0} | {mn:7.1f} sp={sp_n:5.2f}% | {d:+6.2f}% | {claim:+5.2f}% | {v52:>11} | R49={v49} (was {d49:+.2f}%){flip}")
PYEOF
