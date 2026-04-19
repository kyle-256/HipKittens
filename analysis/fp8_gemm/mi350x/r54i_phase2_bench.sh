#!/bin/bash
# R54 Dev I Phase 2: A/B bench across 4 CRR cells.
# 5 runs/cell, MXFP8_WARMUP=100 MXFP8_ITERS=200, 30s cooldown, 60s rebuild_cool.
# Median of 5 runs (rank 3) is the score.
set -euo pipefail
cd "$(dirname "$0")/r54i_workspace"
export THUNDERKITTENS_ROOT="$(cd ../../../.. && pwd)"
OUTDIR="../r54i_results/bench"
mkdir -p "$OUTDIR"

export HIP_VISIBLE_DEVICES=0
export MXFP8_WARMUP=100
export MXFP8_ITERS=200
export MXFP8_LAYOUTS=crr
export MXFP8_PRESHUFFLE_QUANT=1
export MXFP8_DETERMINISM_RUNS=3

# Cells (label, M, N, K)
CELLS=(
  "70B_GateUp_CRR:4096:28672:8192"
  "70B_QO_CRR:4096:8192:8192"
  "8B_Down_CRR:4096:4096:14336"
  "70B_Down_CRR:4096:8192:28672"
)

build_for_cell() {
    local M=$1 N=$2 K=$3 GATE=$4
    local FLAGS=""
    if [[ "$GATE" == "1" ]]; then
        FLAGS="-DMXFP8_CRR_BRANCHLESS_SHIFT=1"
    fi
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $FLAGS" \
        > /tmp/r54i_build_${GATE}.log 2>&1
}

run_5x() {
    local LABEL=$1 M=$2 N=$3 K=$4 GATE=$5
    local LOG="$OUTDIR/${LABEL}_gate${GATE}.log"
    : > "$LOG"
    for run in 1 2 3 4 5; do
        echo "  run $run/5 ..."
        timeout 240 python3 test_mxfp8_python.py $M $N $K >> "$LOG" 2>&1 || echo "  (timeout)" >> "$LOG"
        sleep 30
    done
}

extract_tflops_snr() {
    local LOG=$1
    # Extract TFLOPS lines for CRR layout, take median (rank 3 of 5).
    # Output format from test_mxfp8_python.py:
    #   "crr ... TFLOPS=NNN.NN ..."
    grep -oE "crr.*[Tt][Ff][Ll][Oo][Pp][Ss][:= ]+[0-9]+\.[0-9]+" "$LOG" \
        | grep -oE "[0-9]+\.[0-9]+$" \
        | sort -n | awk 'NR==3 {print}'
}

echo "=========== R54 Dev I Phase 2 A/B bench ==========="
for CELL in "${CELLS[@]}"; do
    LABEL="${CELL%%:*}"
    REST="${CELL#*:}"
    M="${REST%%:*}"; REST="${REST#*:}"
    N="${REST%%:*}"; K="${REST#*:}"
    echo "=== $LABEL  M=$M N=$N K=$K ==="

    for GATE in 0 1; do
        echo "  -- GATE=$GATE build --"
        build_for_cell $M $N $K $GATE
        sleep 60   # rebuild_cool
        echo "  -- GATE=$GATE 5-run bench --"
        run_5x $LABEL $M $N $K $GATE
    done
done

echo ""
echo "=========== SUMMARY ==========="
printf "%-22s %-12s %-12s %-12s\n" "CELL" "BASELINE" "BRANCHLESS" "DELTA%"
for CELL in "${CELLS[@]}"; do
    LABEL="${CELL%%:*}"
    base=$(extract_tflops_snr "$OUTDIR/${LABEL}_gate0.log" 2>/dev/null || echo "")
    brn=$(extract_tflops_snr "$OUTDIR/${LABEL}_gate1.log" 2>/dev/null || echo "")
    if [[ -n "$base" && -n "$brn" ]]; then
        delta=$(python3 -c "print(f'{(($brn - $base)/$base*100):+.2f}')")
    else
        delta="?"
    fi
    printf "%-22s %-12s %-12s %-12s\n" "$LABEL" "${base:-?}" "${brn:-?}" "$delta"
done
