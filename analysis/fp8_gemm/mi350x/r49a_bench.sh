#!/bin/bash
# R49 Dev A: CRR opsel scale-pack fastpath validation
# - 7 compute-bound shapes × {OPSEL=0 baseline, OPSEL=1 opsel}
# - CRR layout only
# - Strict SCLK protocol: 5 runs/cell, 30s cooldown, 60s rebuild cooldown
# - GPU 3 isolated
# - One MXFP8_CHECK=1 run per (shape, OPSEL) for SNR + det (3 runs)
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2

GPU=${HIP_VISIBLE_DEVICES:-3}
WARMUP=${WARMUP:-100}
ITERS=${ITERS:-200}
RUNS=${RUNS:-5}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}

declare -a SHAPES=("8192 8192 8192" "4096 4096 4096" "4096 14336 4096" "4096 4096 14336" "4096 8192 8192" "4096 28672 8192" "4096 8192 28672")
declare -a LABELS=("8192cube" "8B_QO" "8B_GateUp" "8B_Down" "70B_QO" "70B_GateUp" "70B_Down")

OUTDIR="r49a_results"
mkdir -p "$OUTDIR"

run_one() {
    local M=$1 N=$2 K=$3 LABEL=$4 OPSEL=$5
    local TAG="opsel${OPSEL}"
    local OUTBASE="$OUTDIR/${LABEL}_mxfp8_crr_${TAG}"
    for r in $(seq 1 $RUNS); do
        HIP_VISIBLE_DEVICES=$GPU \
        MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=crr MXFP8_PRESHUFFLE_QUANT=1 \
        python3 test_mxfp8_python.py $M $N $K > "${OUTBASE}_run${r}.log" 2>&1
        if [ $r -lt $RUNS ]; then sleep $COOL; fi
    done
}

run_check() {
    local M=$1 N=$2 K=$3 LABEL=$4 OPSEL=$5
    local TAG="opsel${OPSEL}"
    local OUTBASE="$OUTDIR/${LABEL}_check_crr_${TAG}"
    HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=20 MXFP8_ITERS=20 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 \
    MXFP8_SNR_THRESHOLD_DB=45 MXFP8_LAYOUTS=crr MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py $M $N $K > "${OUTBASE}.log" 2>&1
}

for OPSEL in 0 1; do
    echo "############################################"
    echo "##  PASS: MXFP8_CRR_OPSEL=${OPSEL}  ##"
    echo "############################################"
    for i in "${!SHAPES[@]}"; do
        read -r M N K <<< "${SHAPES[$i]}"
        L="${LABELS[$i]}"
        echo "=== [opsel=$OPSEL] [$((i+1))/${#SHAPES[@]}] $L (${M}x${N}x${K}) ==="

        rm -f tk_mxfp8_layouts*.so
        make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
            CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DMXFP8_CRR_OPSEL=${OPSEL}" > /dev/null 2>&1
        echo "  built"

        run_one $M $N $K $L $OPSEL
        echo "  benched ($RUNS runs)"
        sleep $COOL

        run_check $M $N $K $L $OPSEL
        echo "  corr/det checked"

        if [ $i -lt $((${#SHAPES[@]} - 1)) ]; then sleep $REBUILD_COOL; fi
    done
    echo ""
    if [ $OPSEL -eq 0 ]; then sleep $REBUILD_COOL; fi
done

echo ""
echo "========== R49 Dev A SUMMARY =========="

python3 - "$OUTDIR" $RUNS <<'PYEOF'
import sys, re, statistics
D = sys.argv[1]; RUNS = int(sys.argv[2])

names = {
    "8192cube": "8192^3",
    "8B_QO": "8B Q/O (4096^3)",
    "8B_GateUp": "8B Gate/Up",
    "8B_Down": "8B Down",
    "70B_QO": "70B Q/O",
    "70B_GateUp": "70B Gate/Up",
    "70B_Down": "70B Down",
}
order = list(names.keys())

def get_tf_runs(label, opsel):
    vals = []
    for r in range(1, RUNS+1):
        p = f"{D}/{label}_mxfp8_crr_opsel{opsel}_run{r}.log"
        try:
            t = open(p).read()
            m = re.search(r'TFLOPS:\s+([0-9.]+)', t)
            if m: vals.append(float(m.group(1)))
        except: pass
    return vals

def get_check(label, opsel):
    p = f"{D}/{label}_check_crr_opsel{opsel}.log"
    try:
        t = open(p).read()
        snr = re.search(r'SNR:\s+([0-9.]+)\s+dB', t)
        det = re.search(r'Determinism.*?:\s+(PASS|FAIL)', t)
        return (float(snr.group(1)) if snr else None,
                "OK" if (det and "PASS" in det.group(1)) else "FAIL")
    except: return (None, "N/A")

print()
hdr = (f"{'Shape':<20} | {'OFF med':>8} {'ON med':>8} {'spread%':>7} | "
       f"{'delta%':>7} | {'SNR_OFF':>7} {'SNR_ON':>7} {'detOFF':>6} {'detON':>6}")
print(hdr); print("-"*len(hdr))
lines = [hdr, "-"*len(hdr)]

deltas = []
for label in order:
    off_runs = get_tf_runs(label, 0)
    on_runs  = get_tf_runs(label, 1)
    snr0, det0 = get_check(label, 0)
    snr1, det1 = get_check(label, 1)

    if not off_runs or not on_runs:
        line = f"{names[label]:<20} | DATA MISSING"
        print(line); lines.append(line); continue

    off_med = statistics.median(off_runs)
    on_med  = statistics.median(on_runs)
    on_spread = (max(on_runs) - min(on_runs)) / on_med * 100
    delta = (on_med - off_med) / off_med * 100
    deltas.append(delta)
    snr0_s = f"{snr0:.1f}" if snr0 else "N/A"
    snr1_s = f"{snr1:.1f}" if snr1 else "N/A"
    line = (f"{names[label]:<20} | {off_med:8.1f} {on_med:8.1f} {on_spread:6.2f}% | "
            f"{delta:+6.2f}% | {snr0_s:>7} {snr1_s:>7} {det0:>6} {det1:>6}")
    print(line); lines.append(line)

if deltas:
    import math
    geomean = math.exp(sum(math.log(1+d/100) for d in deltas)/len(deltas)) - 1
    geomean *= 100
    worst = min(deltas)
    best  = max(deltas)
    summary = f"\nGEOMEAN delta: {geomean:+.2f}%   worst: {worst:+.2f}%   best: {best:+.2f}%"
    print(summary); lines.append(summary)
    ship_gate_geomean = geomean >= 1.5
    ship_gate_worst   = worst >= -2.0
    verdict = "SHIP" if (ship_gate_geomean and ship_gate_worst) else "REFUTED"
    final = f"VERDICT: {verdict}  (gate: geomean>=+1.5%={ship_gate_geomean}, worst>=-2%={ship_gate_worst})"
    print(final); lines.append(final)

with open(f"{D}/SUMMARY.txt", "w") as f:
    for l in lines: f.write(l + "\n")
print(f"\nWritten to {D}/SUMMARY.txt")
PYEOF
