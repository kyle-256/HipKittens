#!/bin/bash
# R49 Reviewer audit: strict-SCLK A/B re-bench of R47 SHIPs.
# Phase A: R47 Dev A — RCR XCD swizzle (MXFP8_RCR_BLOCK_SWIZZLE)
# Phase B: R47 Dev B — CRR XCD swizzle (MXFP8_CRR_BLOCK_SWIZZLE)
# Phase C: R47 Dev C — CRR SLC removal (MXFP8_CRR_V2_SCALE_CACHEPOLICY)
# Protocol: 5 runs/cell, 30s cooldown, 60s rebuild cooldown, isolated GPU0.

set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"

GPU=${HIP_VISIBLE_DEVICES:-0}
WARMUP=${MXFP8_WARMUP:-50}
ITERS=${MXFP8_ITERS:-100}
RUNS=${RUNS:-5}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}
OUTDIR="r49_reviewer_audit_results"
mkdir -p "$OUTDIR"

# A cell runs N=5 of a single layout/shape with given build flags.
# Args: $1=label  $2=M  $3=N  $4=K  $5=layout  $6=tag  $7=extra_cxxflags
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

# A pair of cells: WITHOUT vs WITH.
# Args: $1=label  $2=M  $3=N  $4=K  $5=layout  $6=without_flag  $7=with_flag
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

# ============================================================
# Phase A: R47 Dev A — RCR XCD swizzle
# ============================================================
echo "=== Phase A: R47 Dev A — RCR XCD swizzle ==="
# 70B Gate/Up RCR — claimed +8.46%
run_pair "A_70B_GateUp_RCR" 4096 28672 8192 rcr "-DMXFP8_RCR_BLOCK_SWIZZLE=0" "-DMXFP8_RCR_BLOCK_SWIZZLE=1"
# 70B Down RCR — claimed +2.90%
run_pair "A_70B_Down_RCR"   4096 8192 28672 rcr "-DMXFP8_RCR_BLOCK_SWIZZLE=0" "-DMXFP8_RCR_BLOCK_SWIZZLE=1"

# ============================================================
# Phase B: R47 Dev B — CRR XCD swizzle
# ============================================================
echo "=== Phase B: R47 Dev B — CRR XCD swizzle ==="
# 70B Down CRR — claimed +3.35%
run_pair "B_70B_Down_CRR"   4096 8192 28672 crr "-DMXFP8_CRR_BLOCK_SWIZZLE=0" "-DMXFP8_CRR_BLOCK_SWIZZLE=1"
# 8B Q/O CRR — claimed +2.70%
run_pair "B_8B_QO_CRR"      4096 4096 4096  crr "-DMXFP8_CRR_BLOCK_SWIZZLE=0" "-DMXFP8_CRR_BLOCK_SWIZZLE=1"
# 70B Gate/Up CRR — claimed +2.21%
run_pair "B_70B_GateUp_CRR" 4096 28672 8192 crr "-DMXFP8_CRR_BLOCK_SWIZZLE=0" "-DMXFP8_CRR_BLOCK_SWIZZLE=1"
# 8B Down CRR — claimed +1.13%
run_pair "B_8B_Down_CRR"    4096 4096 14336 crr "-DMXFP8_CRR_BLOCK_SWIZZLE=0" "-DMXFP8_CRR_BLOCK_SWIZZLE=1"

# ============================================================
# Phase C: R47 Dev C — CRR scale cachepolicy SLC removal
# OFF (re-add SLC) = -DMXFP8_CRR_V2_SCALE_CACHEPOLICY=2 (mimic R28 conditional)
# ON  (current default = no SLC) = -DMXFP8_CRR_V2_SCALE_CACHEPOLICY=0
# Note: 8B QO (4096^3) and 8192^3 don't trigger the conditional gate
# (which required N>=28672 AND K>=8192) so this is a "what if" comparison.
# ============================================================
echo "=== Phase C: R47 Dev C — CRR scale cachepolicy ==="
# 70B Gate/Up CRR — only shape that hit the gate, cell where claim was made
run_pair "C_70B_GateUp_CRR" 4096 28672 8192 crr "-DMXFP8_CRR_V2_SCALE_CACHEPOLICY=2" "-DMXFP8_CRR_V2_SCALE_CACHEPOLICY=0"
# 8192^3 CRR
run_pair "C_8192cube_CRR"   8192 8192 8192  crr "-DMXFP8_CRR_V2_SCALE_CACHEPOLICY=2" "-DMXFP8_CRR_V2_SCALE_CACHEPOLICY=0"
# 8B Q/O CRR
run_pair "C_8B_QO_CRR"      4096 4096 4096  crr "-DMXFP8_CRR_V2_SCALE_CACHEPOLICY=2" "-DMXFP8_CRR_V2_SCALE_CACHEPOLICY=0"

echo ""
echo "========== R49 REVIEWER AUDIT — strict-SCLK A/B summary =========="
python3 - "$OUTDIR" <<'PYEOF'
import sys, re, os, glob, statistics
D = sys.argv[1]
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
def report(label, claim_pct):
    off = [get_tf(f"{D}/{label}_off_run{r}.log") for r in (1,2,3,4,5)]
    on  = [get_tf(f"{D}/{label}_on_run{r}.log")  for r in (1,2,3,4,5)]
    mo, lo_o, hi_o, mn_o = stats(off)
    mn, lo_n, hi_n, mn_n = stats(on)
    if mo is None or mn is None:
        print(f"{label:<24} | MISSING DATA")
        return
    d = 100*(mn-mo)/mo
    spread_o = (hi_o-lo_o)/mo*100
    spread_n = (hi_n-lo_n)/mn*100
    ratio = d/claim_pct if claim_pct != 0 else float('nan')
    if d < 0.0 and claim_pct > 0:
        verdict = "REGRESSION"
    elif abs(d) < max(spread_o, spread_n) and abs(d) < 0.5:
        verdict = "NOISE(within-spread)"
    elif ratio < 0.25:
        verdict = "NOISE"
    elif ratio < 0.75:
        verdict = "INFLATED"
    else:
        verdict = "CONFIRMED"
    print(f"{label:<24} | OFF med={mo:7.1f} sp={spread_o:5.2f}% | ON med={mn:7.1f} sp={spread_n:5.2f}% | dlt={d:+6.2f}% | claim={claim_pct:+5.2f}% | ratio={ratio:+5.2f} | {verdict}")

print(f"{'Cell':<24} | {'OFF (no swizzle/SLC=2)':>26} | {'ON (default HEAD)':>20} | {'real':>7} | {'claim':>6} | {'ratio':>6} | verdict")
print("-"*150)

# Phase A
print("--- Phase A: R47 Dev A RCR swizzle ---")
report("A_70B_GateUp_RCR", 8.46)
report("A_70B_Down_RCR",   2.90)

# Phase B
print("--- Phase B: R47 Dev B CRR swizzle ---")
report("B_70B_Down_CRR",   3.35)
report("B_8B_QO_CRR",      2.70)
report("B_70B_GateUp_CRR", 2.21)
report("B_8B_Down_CRR",    1.13)

# Phase C: R47 Dev C — SLC removal. In Phase C, "OFF" means SLC=2 (the
# pre-R47C behavior on shapes that hit the conditional). "ON" means SLC=0
# (post-R47C unconditional default). R47C reported -1.52% for SLC vs no-SLC
# on 70B Gate/Up, i.e. removal was a +1.52% gain. So claim = +1.52%.
print("--- Phase C: R47 Dev C CRR SLC removal (claim = removal gain) ---")
report("C_70B_GateUp_CRR", 1.52)
report("C_8192cube_CRR",   0.00)
report("C_8B_QO_CRR",      0.00)
PYEOF
