#!/bin/bash
# R55 Dev C Phase 2: A/B bench across primary + co-cells.
# 5 runs/cell, MXFP8_WARMUP=100 MXFP8_ITERS=200, 30s cooldown, 60s rebuild_cool.
# Median of 5 runs (rank 3) is the score. GPU 2 isolated.
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../../.. && pwd)"
OUTDIR="../r55c_results/bench"
mkdir -p "$OUTDIR"

export HIP_VISIBLE_DEVICES=2
export MXFP8_WARMUP=100
export MXFP8_ITERS=200
export MXFP8_LAYOUTS=rcr
export MXFP8_PRESHUFFLE_QUANT=1
export MXFP8_DETERMINISM_RUNS=3

# Cells (label, M, N, K)
CELLS=(
  "70B_Down_RCR:4096:8192:28672"
  "70B_QO_RCR:4096:8192:8192"
  "8B_Down_RCR:4096:4096:14336"
)

build_for_cell() {
    local M=$1 N=$2 K=$3 GATE=$4
    local FLAGS=""
    if [[ "$GATE" == "1" ]]; then
        FLAGS="-DMXFP8_RCR_VALU_DEP_BREAK=1"
    fi
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $FLAGS" \
        > /tmp/r55c_build_${GATE}.log 2>&1
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

extract_tflops() {
    local LOG=$1
    # Output format: "  Avg time: 0.6646 ms, TFLOPS: 2895.10"
    grep -oE "TFLOPS: [0-9]+\.[0-9]+" "$LOG" \
        | awk '{print $2}' \
        | sort -n | awk 'NR==3 {print}'
}

extract_snr() {
    local LOG=$1
    grep -oE "SNR: [0-9]+\.[0-9]+ dB" "$LOG" \
        | awk '{print $2}' \
        | sort -n | awk 'NR==3 {print}'
}

extract_det() {
    local LOG=$1
    grep -c "Determinism (3 runs): PASS" "$LOG" || echo 0
}

echo "=========== R55 Dev C Phase 2 A/B bench (GPU 2) ==========="
date
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
date
printf "%-22s %-12s %-12s %-12s %-10s %-10s %-8s %-8s\n" "CELL" "BASELINE" "TREATMENT" "DELTA%" "B_SNR" "T_SNR" "B_DET" "T_DET"
for CELL in "${CELLS[@]}"; do
    LABEL="${CELL%%:*}"
    base=$(extract_tflops "$OUTDIR/${LABEL}_gate0.log" 2>/dev/null || echo "")
    trt=$(extract_tflops  "$OUTDIR/${LABEL}_gate1.log" 2>/dev/null || echo "")
    bsnr=$(extract_snr    "$OUTDIR/${LABEL}_gate0.log" 2>/dev/null || echo "")
    tsnr=$(extract_snr    "$OUTDIR/${LABEL}_gate1.log" 2>/dev/null || echo "")
    bdet=$(extract_det    "$OUTDIR/${LABEL}_gate0.log" 2>/dev/null || echo "")
    tdet=$(extract_det    "$OUTDIR/${LABEL}_gate1.log" 2>/dev/null || echo "")
    if [[ -n "$base" && -n "$trt" ]]; then
        delta=$(python3 -c "print(f'{(($trt - $base)/$base*100):+.2f}')")
    else
        delta="?"
    fi
    printf "%-22s %-12s %-12s %-12s %-10s %-10s %-8s %-8s\n" "$LABEL" "${base:-?}" "${trt:-?}" "$delta" "${bsnr:-?}" "${tsnr:-?}" "${bdet:-?}" "${tdet:-?}"
done
