#!/bin/bash
# R56 Dev C Phase 2: 5-arm bench (baseline / R55C-alone / Stack1-INT2 /
# Stack2-POST / Stack3-STEADY) across 3 cells.
# 5 runs/cell/arm, MXFP8_WARMUP=100 MXFP8_ITERS=200, 30s cooldown,
# 60s rebuild_cool. Median rank-3-of-5 is the score. GPU 2 isolated.
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../../.. && pwd)"
OUTDIR="../r56c_results/bench"
mkdir -p "$OUTDIR"

export HIP_VISIBLE_DEVICES=2
export MXFP8_WARMUP=100
export MXFP8_ITERS=200
export MXFP8_LAYOUTS=rcr
export MXFP8_PRESHUFFLE_QUANT=1
export MXFP8_DETERMINISM_RUNS=3
export MXFP8_RCR_PRESHUFFLE_V2_RUNTIME=1

# Cells (label, M, N, K)
CELLS=(
  "70B_Down_RCR:4096:8192:28672"
  "70B_QO_RCR:4096:8192:8192"
  "8B_Down_RCR:4096:4096:14336"
)

# Arm name -> macro flags
ARMS=(
  "baseline:"
  "R55C:-DMXFP8_RCR_VALU_DEP_BREAK=1"
  "Stack1_INT2:-DMXFP8_RCR_VALU_DEP_BREAK=1 -DMXFP8_RCR_SCALE_INTERLEAVE=2"
  "Stack2_POST:-DMXFP8_RCR_VALU_DEP_BREAK=1 -DMXFP8_RCR_VALU_DEP_BREAK_R56C_POST=1"
  "Stack3_STDY:-DMXFP8_RCR_VALU_DEP_BREAK=1 -DMXFP8_RCR_VALU_DEP_BREAK_R56C_STEADY_ONLY=1"
)

build_for_cell() {
    local M=$1 N=$2 K=$3 ARMNAME=$4 FLAGS="$5"
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $FLAGS" \
        > /tmp/r56c_build_${ARMNAME}.log 2>&1
}

run_5x() {
    local LABEL=$1 M=$2 N=$3 K=$4 ARMNAME=$5
    local LOG="$OUTDIR/${LABEL}_${ARMNAME}.log"
    : > "$LOG"
    for run in 1 2 3 4 5; do
        echo "    run $run/5 ..."
        timeout 240 python3 test_mxfp8_python.py $M $N $K >> "$LOG" 2>&1 || echo "  (timeout)" >> "$LOG"
        sleep 30
    done
}

extract_tflops() {
    local LOG=$1
    grep -oE "TFLOPS: [0-9]+\.[0-9]+" "$LOG" \
        | awk '{print $2}' \
        | sort -n | awk 'NR==3 {print}'
}
extract_tflops_all() {
    local LOG=$1
    grep -oE "TFLOPS: [0-9]+\.[0-9]+" "$LOG" \
        | awk '{print $2}' \
        | sort -n | tr '\n' ' '
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

echo "=========== R56 Dev C Phase 2 5-arm bench (GPU 2) ==========="
date
for CELL in "${CELLS[@]}"; do
    LABEL="${CELL%%:*}"
    REST="${CELL#*:}"
    M="${REST%%:*}"; REST="${REST#*:}"
    N="${REST%%:*}"; K="${REST#*:}"
    echo "=== $LABEL  M=$M N=$N K=$K ==="

    for ARM in "${ARMS[@]}"; do
        ARMNAME="${ARM%%:*}"
        FLAGS="${ARM#*:}"
        echo "  -- ARM=$ARMNAME (flags='$FLAGS') --"
        build_for_cell $M $N $K $ARMNAME "$FLAGS"
        sleep 60
        run_5x $LABEL $M $N $K $ARMNAME
    done
done

echo ""
echo "=========== SUMMARY ==========="
date
printf "%-22s %-14s %-12s %-9s %-10s %-7s\n" "CELL" "ARM" "MEDIAN" "DELTA%" "SNR_med" "DET_5x"
for CELL in "${CELLS[@]}"; do
    LABEL="${CELL%%:*}"
    base=""
    for ARM in "${ARMS[@]}"; do
        ARMNAME="${ARM%%:*}"
        LOG="$OUTDIR/${LABEL}_${ARMNAME}.log"
        med=$(extract_tflops "$LOG" 2>/dev/null || echo "")
        snr=$(extract_snr "$LOG" 2>/dev/null || echo "")
        det=$(extract_det "$LOG" 2>/dev/null || echo "")
        if [[ "$ARMNAME" == "baseline" ]]; then
            base="$med"
            delta="0.00"
        elif [[ -n "$base" && -n "$med" ]]; then
            delta=$(python3 -c "print(f'{(($med - $base)/$base*100):+.2f}')")
        else
            delta="?"
        fi
        printf "%-22s %-14s %-12s %-9s %-10s %-7s\n" "$LABEL" "$ARMNAME" "${med:-?}" "$delta" "${snr:-?}" "${det:-?}"
    done
done

echo ""
echo "=========== FULL DISTRIBUTIONS ==========="
for CELL in "${CELLS[@]}"; do
    LABEL="${CELL%%:*}"
    echo "--- $LABEL ---"
    for ARM in "${ARMS[@]}"; do
        ARMNAME="${ARM%%:*}"
        LOG="$OUTDIR/${LABEL}_${ARMNAME}.log"
        all=$(extract_tflops_all "$LOG" 2>/dev/null || echo "")
        printf "  %-14s %s\n" "$ARMNAME" "$all"
    done
done
