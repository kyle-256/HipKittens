#!/bin/bash
# R48 Full bench RESUME: continues from where r48_full_bench.sh died on shape 3
# Differences from original:
#   - Picks up from shape index 2 (8B_GateUp) — shapes 0,1 (8192cube, 8B_QO) already done
#   - set +e around per-run python invocations so transient HIP aborts don't kill the whole bench
#   - Skips 8B_GateUp_mxfp8_rcr_run1 (already complete, leaves run1 file intact)
#   - Each run that aborts logs the abort but bench continues
set -uo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2

GPU=${HIP_VISIBLE_DEVICES:-6}
WARMUP=${WARMUP:-100}
ITERS=${ITERS:-200}
RUNS=${RUNS:-5}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}

declare -a SHAPES=("8192 8192 8192" "4096 4096 4096" "4096 14336 4096" "4096 4096 14336" "4096 8192 8192" "4096 28672 8192" "4096 8192 28672")
declare -a LABELS=("8192cube" "8B_QO" "8B_GateUp" "8B_Down" "70B_QO" "70B_GateUp" "70B_Down")

OUTDIR="r48_full_results"
mkdir -p "$OUTDIR"

# Resume from this shape index (0-indexed). Set START_IDX=0 to restart entirely.
START_IDX=${START_IDX:-2}

run_one() {
    local kind=$1 M=$2 N=$3 K=$4 LABEL=$5 LAY=$6
    local OUTBASE="$OUTDIR/${LABEL}_${kind}_${LAY}"
    for r in $(seq 1 $RUNS); do
        # Skip if log already exists AND has TFLOPS line (don't reuse aborted runs)
        if [ -f "${OUTBASE}_run${r}.log" ] && grep -q "TFLOPS:" "${OUTBASE}_run${r}.log" 2>/dev/null; then
            echo "    skip ${LABEL} ${kind} ${LAY} run${r} (already valid)"
            continue
        fi
        set +e
        if [ "$kind" = "fp8" ]; then
            HIP_VISIBLE_DEVICES=$GPU \
            FP8_BUILD_M=$M FP8_BUILD_N=$N FP8_BUILD_K=$K \
            FP8_WARMUP=$WARMUP FP8_ITERS=$ITERS FP8_CHECK=0 FP8_LAYOUTS=$LAY \
            python3 test_python.py $M $N $K > "${OUTBASE}_run${r}.log" 2>&1
        else
            HIP_VISIBLE_DEVICES=$GPU \
            MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
            MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
            MXFP8_LAYOUTS=$LAY MXFP8_PRESHUFFLE_QUANT=1 \
            python3 test_mxfp8_python.py $M $N $K > "${OUTBASE}_run${r}.log" 2>&1
        fi
        rc=$?
        set -e
        if [ $rc -ne 0 ]; then
            echo "    ABORT ${LABEL} ${kind} ${LAY} run${r} rc=${rc} (continuing)"
        fi
        if [ $r -lt $RUNS ]; then sleep $COOL; fi
    done
}

run_check() {
    local M=$1 N=$2 K=$3 LABEL=$4 LAY=$5
    local OUTBASE="$OUTDIR/${LABEL}_check_${LAY}"
    if [ -f "${OUTBASE}.log" ] && grep -q "SNR" "${OUTBASE}.log" 2>/dev/null; then
        echo "    skip ${LABEL} check ${LAY} (already valid)"
        return
    fi
    set +e
    HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=20 MXFP8_ITERS=20 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 \
    MXFP8_SNR_THRESHOLD_DB=45 MXFP8_LAYOUTS=$LAY MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py $M $N $K > "${OUTBASE}.log" 2>&1
    rc=$?
    set -e
    if [ $rc -ne 0 ]; then
        echo "    ABORT ${LABEL} check ${LAY} rc=${rc} (continuing)"
    fi
}

for i in "${!SHAPES[@]}"; do
    if [ $i -lt $START_IDX ]; then continue; fi
    read -r M N K <<< "${SHAPES[$i]}"
    L="${LABELS[$i]}"
    echo "=== [$((i+1))/${#SHAPES[@]}] $L (${M}x${N}x${K}) ==="

    rm -f tk_fp8_layouts*.so
    make TARGET=tk_fp8_layouts SRC=kernel_fp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" > /dev/null 2>&1
    echo "  FP8 built"
    for LAY in rcr rrr crr; do
        run_one fp8 $M $N $K $L $LAY
        sleep $COOL
    done
    echo "  FP8 benched"
    sleep $REBUILD_COOL

    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" > /dev/null 2>&1
    echo "  MXFP8 built"
    for LAY in rcr rrr crr; do
        run_one mxfp8 $M $N $K $L $LAY
        sleep $COOL
    done
    echo "  MXFP8 benched"

    for LAY in rcr rrr crr; do
        run_check $M $N $K $L $LAY
        sleep 5
    done
    echo "  MXFP8 corr/det checked"

    if [ $i -lt $((${#SHAPES[@]} - 1)) ]; then sleep $REBUILD_COOL; fi
done

echo ""
echo "========== R48 SUMMARY =========="

python3 - "$OUTDIR" $RUNS <<'PYEOF'
import sys, re, os, statistics
D = sys.argv[1]; RUNS = int(sys.argv[2])

names = {
    "8192cube": "8192³",
    "8B_QO": "8B Q/O (4096³)",
    "8B_GateUp": "8B Gate/Up",
    "8B_Down": "8B Down",
    "70B_QO": "70B Q/O",
    "70B_GateUp": "70B Gate/Up",
    "70B_Down": "70B Down",
}
order = list(names.keys())

def get_tf_runs(label, kind, lay):
    vals = []
    for r in range(1, RUNS+1):
        p = f"{D}/{label}_{kind}_{lay}_run{r}.log"
        try:
            t = open(p).read()
            m = re.search(r'TFLOPS:\s+([0-9.]+)', t)
            if m: vals.append(float(m.group(1)))
        except: pass
    return vals

def get_check(label, lay):
    p = f"{D}/{label}_check_{lay}.log"
    try:
        t = open(p).read()
        snr = re.search(r'SNR:\s+([0-9.]+)\s+dB', t)
        det = re.search(r'Determinism.*?:\s+(PASS|FAIL)', t)
        return (float(snr.group(1)) if snr else None,
                "OK" if (det and "PASS" in det.group(1)) else "FAIL")
    except: return (None, "N/A")

print()
hdr = f"{'Shape':<20} | {'Layout':>6} | {'FP8 med':>8} {'MX med':>8} {'MX/FP8':>7} | {'MX spread':>9} | {'SNR':>6} {'Det':>4} | {'Status':>6} | {'N runs':>8}"
print(hdr); print("-"*len(hdr))

lines = []
for label in order:
    for lay in ["rcr", "rrr", "crr"]:
        fp_runs = get_tf_runs(label, "fp8", lay)
        mx_runs = get_tf_runs(label, "mxfp8", lay)
        snr, det = get_check(label, lay)

        if not fp_runs or not mx_runs:
            line = f"{names[label]:<20} | {lay.upper():>6} | DATA MISSING fp={len(fp_runs)} mx={len(mx_runs)}"
            print(line); lines.append(line); continue

        fp_med = statistics.median(fp_runs)
        mx_med = statistics.median(mx_runs)
        ratio = mx_med / fp_med
        mx_spread = (max(mx_runs) - min(mx_runs)) / mx_med * 100 if len(mx_runs) > 1 else 0.0

        snr_s = f"{snr:.1f}" if snr else "N/A"
        perf_ok = ratio >= 0.95
        snr_ok = snr and snr >= 45.0
        det_ok = det == "OK"
        all_ok = perf_ok and snr_ok and det_ok
        status = "PASS" if all_ok else "FAIL"

        nruns = f"f{len(fp_runs)}/m{len(mx_runs)}"
        line = (f"{names[label]:<20} | {lay.upper():>6} | "
                f"{fp_med:8.1f} {mx_med:8.1f} {100*ratio:6.1f}% | "
                f"{mx_spread:8.2f}% | {snr_s:>6} {det:>4} | {status:>6} | {nruns:>8}")
        print(line); lines.append(line)
    print()

with open(f"{D}/SUMMARY.txt", "w") as f:
    f.write(f"R48 strict-protocol baseline (target {RUNS} runs/cell, median)\n")
    f.write(hdr + "\n")
    f.write("-"*len(hdr) + "\n")
    for l in lines: f.write(l + "\n")
print(f"\nWritten to {D}/SUMMARY.txt")
PYEOF
