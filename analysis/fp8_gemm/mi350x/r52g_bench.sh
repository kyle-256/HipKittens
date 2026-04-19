#!/bin/bash
# R52 Dev G: per-shape un-ship of CRR XCD swizzle for 70B Gate/Up.
# Strict-SCLK A/B: 5 runs/cell, 30 s cooldown between runs, 60 s rebuild
# cooldown between A and B sides, MXFP8_WARMUP=100, MXFP8_ITERS=200.
#
# A side ("off") = simulate the patch by passing -DMXFP8_CRR_BLOCK_SWIZZLE=0
#                  for the 70B Gate/Up cell, =1 for all others. This is the
#                  POST-PATCH state.
# B side ("on")  = -DMXFP8_CRR_BLOCK_SWIZZLE=1 for ALL cells (current HEAD,
#                  PRE-PATCH state where the swizzle is forced ON globally).
#
# We use explicit -D on both sides so the source-level auto-gate is bypassed
# (the user-provided -D wins by design). This isolates the per-shape effect.
# We then verify in a final pass that the auto-gate (no -D) lands on the same
# value as the "off" leg for 70B Gate/Up and same as "on" for everything else.

set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"

GPU=${HIP_VISIBLE_DEVICES:-5}
WARMUP=${MXFP8_WARMUP:-100}
ITERS=${MXFP8_ITERS:-200}
RUNS=${RUNS:-5}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}
OUTDIR="r52g_results"
mkdir -p "$OUTDIR"

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

# Args: $1=label  $2=M  $3=N  $4=K  $5=layout  $6=off_flag  $7=on_flag
run_pair() {
    local LABEL=$1 M=$2 N=$3 K=$4 LAYOUT=$5 OFF_FLAG=$6 ON_FLAG=$7
    echo "  --> $LABEL ($M x $N x $K) $LAYOUT  [post-patch / off-or-on]"
    run_cell "$LABEL" "$M" "$N" "$K" "$LAYOUT" "post" "$OFF_FLAG"
    echo "      sleep ${REBUILD_COOL}s"
    sleep $REBUILD_COOL
    echo "  --> $LABEL ($M x $N x $K) $LAYOUT  [pre-patch / always-on]"
    run_cell "$LABEL" "$M" "$N" "$K" "$LAYOUT" "pre" "$ON_FLAG"
    echo "      sleep ${REBUILD_COOL}s"
    sleep $REBUILD_COOL
}

# Cells touched by R49 reviewer Phase B (CRR XCD swizzle), plus 70B Q/O which
# the reviewer noted as a candidate for the gate-on side.
echo "=== R52 Dev G: CRR XCD swizzle per-shape gate strict-SCLK A/B ==="

# 70B Gate/Up CRR — primary target (regression cell). post=OFF, pre=ON.
run_pair "70B_GateUp_CRR" 4096 28672 8192 crr "-DMXFP8_CRR_BLOCK_SWIZZLE=0" "-DMXFP8_CRR_BLOCK_SWIZZLE=1"

# All other cells: post=ON (no change), pre=ON. Same flag both sides — this
# confirms the patch is a no-op for these cells and gives a fresh baseline.
run_pair "70B_Down_CRR"   4096 8192 28672 crr "-DMXFP8_CRR_BLOCK_SWIZZLE=1" "-DMXFP8_CRR_BLOCK_SWIZZLE=1"
run_pair "70B_QO_CRR"     4096 8192 8192  crr "-DMXFP8_CRR_BLOCK_SWIZZLE=1" "-DMXFP8_CRR_BLOCK_SWIZZLE=1"
run_pair "8B_GateUp_CRR"  4096 14336 4096 crr "-DMXFP8_CRR_BLOCK_SWIZZLE=1" "-DMXFP8_CRR_BLOCK_SWIZZLE=1"
run_pair "8B_QO_CRR"      4096 4096 4096  crr "-DMXFP8_CRR_BLOCK_SWIZZLE=1" "-DMXFP8_CRR_BLOCK_SWIZZLE=1"
run_pair "8B_Down_CRR"    4096 4096 14336 crr "-DMXFP8_CRR_BLOCK_SWIZZLE=1" "-DMXFP8_CRR_BLOCK_SWIZZLE=1"
run_pair "8192cube_CRR"   8192 8192 8192  crr "-DMXFP8_CRR_BLOCK_SWIZZLE=1" "-DMXFP8_CRR_BLOCK_SWIZZLE=1"

# Auto-gate verification — build with NO -D for swizzle, let the source choose.
# 70B Gate/Up should auto-OFF; one other cell (8B Gate/Up) sanity-checked auto-ON.
echo "=== Auto-gate verification (no -DMXFP8_CRR_BLOCK_SWIZZLE flag) ==="
run_cell "AUTOGATE_70B_GateUp_CRR" 4096 28672 8192 crr "auto" ""
sleep $REBUILD_COOL
run_cell "AUTOGATE_8B_GateUp_CRR"  4096 14336 4096 crr "auto" ""
sleep $REBUILD_COOL

echo ""
echo "========== R52 Dev G — strict-SCLK A/B summary =========="
python3 - "$OUTDIR" <<'PYEOF'
import sys, re, glob, statistics
D = sys.argv[1]
def get_tf(p):
    try:
        t = open(p).read()
        m = re.search(r'TFLOPS:\s+([0-9.]+)', t)
        return float(m.group(1)) if m else None
    except Exception:
        return None
def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (None, None, None, None)
    return (statistics.median(xs), min(xs), max(xs), statistics.mean(xs))
def report(label, expected_sign):
    post = [get_tf(f"{D}/{label}_post_run{r}.log") for r in (1,2,3,4,5)]
    pre  = [get_tf(f"{D}/{label}_pre_run{r}.log")  for r in (1,2,3,4,5)]
    mp, lo_p, hi_p, mn_p = stats(post)
    mr, lo_r, hi_r, mn_r = stats(pre)
    if mp is None or mr is None:
        print(f"{label:<24} | MISSING DATA")
        return
    d = 100 * (mp - mr) / mr
    sp_p = (hi_p - lo_p) / mp * 100
    sp_r = (hi_r - lo_r) / mr * 100
    print(f"{label:<24} | PRE(swizzle ON, current HEAD) med={mr:7.1f} sp={sp_r:5.2f}% | POST(per-shape gate) med={mp:7.1f} sp={sp_p:5.2f}% | dlt={d:+6.2f}% | expected={expected_sign}")

print(f"{'Cell':<24} | {'PRE (current HEAD)':>20} | {'POST (per-shape gate)':>22} | {'delta':>6} | expected")
print("-"*150)
report("70B_GateUp_CRR",  "+ (recover from R47B regression)")
report("70B_Down_CRR",    "= (no-op)")
report("70B_QO_CRR",      "= (no-op)")
report("8B_GateUp_CRR",   "= (no-op)")
report("8B_QO_CRR",       "= (no-op)")
report("8B_Down_CRR",     "= (no-op)")
report("8192cube_CRR",    "= (no-op)")

print()
print("--- Auto-gate verification ---")
def auto_report(label):
    runs = [get_tf(f"{D}/{label}_auto_run{r}.log") for r in (1,2,3,4,5)]
    m, lo, hi, mn = stats(runs)
    if m is None:
        print(f"{label:<28} | MISSING")
        return
    sp = (hi - lo) / m * 100
    print(f"{label:<28} | med={m:7.1f} sp={sp:5.2f}%")
auto_report("AUTOGATE_70B_GateUp_CRR")
auto_report("AUTOGATE_8B_GateUp_CRR")
PYEOF
