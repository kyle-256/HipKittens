#!/bin/bash
# R52 Dev H: RRR soft scheduler barrier (asm volatile("" ::: "memory"))
# at the cB->cC quartet boundary inside do_k_iter. Strict-SCLK A/B on
# 8B Gate/Up RRR (the only RRR HEADROOM cell per R48 Dev D), plus
# cross-shape spot-checks on 8B Q/O RRR and 70B Q/O RRR.
#
# Protocol:
#   - 5 runs/cell, 30s cooldown between runs
#   - 60s cooldown between rebuilds
#   - MXFP8_WARMUP=100, MXFP8_ITERS=200
#   - HIP_VISIBLE_DEVICES=1
#   - A side ("off") = baseline (no -DMXFP8_RRR_SOFT_BARRIER)
#   - B side ("on")  = -DMXFP8_RRR_SOFT_BARRIER=1
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"
export ROCM_PATH=/opt/rocm

GPU=${HIP_VISIBLE_DEVICES:-1}
WARMUP=${MXFP8_WARMUP:-100}
ITERS=${MXFP8_ITERS:-200}
RUNS=${RUNS:-5}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}
OUTDIR="r52h_results"
mkdir -p "$OUTDIR"

# Args: $1=label  $2=M  $3=N  $4=K  $5=tag  $6=extra_cxxflags
run_cell() {
    local LABEL=$1 M=$2 N=$3 K=$4 TAG=$5 EXTRA=$6
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $EXTRA" \
        > "$OUTDIR/${LABEL}_${TAG}_build.log" 2>&1
    for r in $(seq 1 $RUNS); do
        HIP_VISIBLE_DEVICES=$GPU \
        MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=rrr MXFP8_PRESHUFFLE_QUANT=1 \
        python3 test_mxfp8_python.py $M $N $K \
            > "$OUTDIR/${LABEL}_${TAG}_run${r}.log" 2>&1
        if [ $r -lt $RUNS ]; then sleep $COOL; fi
    done
}

run_check() {
    local LABEL=$1 M=$2 N=$3 K=$4 TAG=$5
    HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=20 MXFP8_ITERS=20 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 \
    MXFP8_SNR_THRESHOLD_DB=45 MXFP8_LAYOUTS=rrr MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py $M $N $K \
        > "$OUTDIR/${LABEL}_${TAG}_check.log" 2>&1 || true
}

run_pair() {
    local LABEL=$1 M=$2 N=$3 K=$4
    echo "  --> $LABEL ($M x $N x $K) RRR  [A=off]"
    run_cell "$LABEL" "$M" "$N" "$K" "off" ""
    echo "      sleep ${REBUILD_COOL}s"
    sleep $REBUILD_COOL
    echo "  --> $LABEL ($M x $N x $K) RRR  [B=on]"
    run_cell "$LABEL" "$M" "$N" "$K" "on" "-DMXFP8_RRR_SOFT_BARRIER=1"
    sleep $COOL
    run_check "$LABEL" "$M" "$N" "$K" "on"
    echo "      sleep ${REBUILD_COOL}s"
    sleep $REBUILD_COOL
}

echo "=== R52 Dev H: RRR soft scheduler barrier strict-SCLK A/B ==="

# Primary target: 8B Gate/Up RRR.
run_pair "8B_GateUp_RRR"  4096 14336 4096
# Cross-shape spot checks (cheap RRR cells; bench size moderate).
run_pair "8B_QO_RRR"      4096 4096  4096
run_pair "70B_QO_RRR"     4096 8192  8192

echo ""
echo "========== R52 Dev H — strict-SCLK A/B summary =========="
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
def report(label):
    off = [get_tf(f"{D}/{label}_off_run{r}.log") for r in (1,2,3,4,5)]
    on  = [get_tf(f"{D}/{label}_on_run{r}.log")  for r in (1,2,3,4,5)]
    mo, lo_o, hi_o, mn_o = stats(off)
    mn, lo_n, hi_n, mn_n = stats(on)
    if mo is None or mn is None:
        print(f"{label:<24} | MISSING DATA off={off} on={on}")
        return
    d = 100 * (mn - mo) / mo
    sp_o = (hi_o - lo_o) / mo * 100
    sp_n = (hi_n - lo_n) / mn * 100
    print(f"{label:<24} | OFF med={mo:7.1f} sp={sp_o:5.2f}% | ON med={mn:7.1f} sp={sp_n:5.2f}% | dlt={d:+6.2f}%")

print(f"{'Cell':<24} | {'OFF (baseline)':>20} | {'ON (soft barrier)':>22} | {'delta':>6}")
print("-"*100)
report("8B_GateUp_RRR")
report("8B_QO_RRR")
report("70B_QO_RRR")
PYEOF
