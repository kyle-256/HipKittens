#!/bin/bash
# R54 Dev B PMC profiling at 70B Down RCR (target), 8B Down RCR (control,
# K=14336 PASS at 96.3%), 70B Q/O RCR (control, K=8192 PASS).
# Profiles MXFP8 + FP8 baselines on each cell with set1 + set3.
# Mirrors R53C methodology, scoped to RCR layout only.
set -uo pipefail

cd "$(dirname "$0")/.."   # to mi350x dir
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"
export ROCM_PATH=/opt/rocm

GPU=${HIP_VISIBLE_DEVICES:-4}
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-5}

OUTROOT="r54b_pmc_results"
PMC1="$OUTROOT/pmc_set1.txt"
PMC3="$OUTROOT/pmc_set3.txt"

# Cells: LABEL|M|N|K
CELLS=(
    "70B_Down|4096|8192|28672"
    "8B_Down|4096|4096|14336"
    "70B_QO|4096|8192|8192"
)

run_pmc_mxfp8() {
    local label=$1 M=$2 N=$3 K=$4 pmcset=$5 pmcfile=$6
    local outdir="$OUTROOT/${label}_mxfp8_rcr_${pmcset}"
    mkdir -p "$outdir"
    if [ -f "$outdir/pmc_5/pmc_counter_collection.csv" ]; then
        echo "  skip $label mxfp8 $pmcset (already complete)"
        return
    fi
    echo "  running mxfp8 $label $pmcset -> $outdir"
    HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
    MXFP8_LAYOUTS=rcr MXFP8_PRESHUFFLE_QUANT=1 \
    rocprofv3 -i "$pmcfile" --output-format csv -d "$outdir" \
        -- python3 test_mxfp8_python.py $M $N $K \
        > "$outdir/run.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "    ABORT mxfp8 $label $pmcset rc=$rc"
        tail -20 "$outdir/run.log"
    fi
}

run_pmc_fp8() {
    local label=$1 M=$2 N=$3 K=$4 pmcset=$5 pmcfile=$6
    local outdir="$OUTROOT/${label}_fp8_rcr_${pmcset}"
    mkdir -p "$outdir"
    if [ -f "$outdir/pmc_5/pmc_counter_collection.csv" ]; then
        echo "  skip $label fp8 $pmcset (already complete)"
        return
    fi
    echo "  running fp8 $label $pmcset -> $outdir"
    HIP_VISIBLE_DEVICES=$GPU \
    FP8_BUILD_M=$M FP8_BUILD_N=$N FP8_BUILD_K=$K \
    FP8_WARMUP=$WARMUP FP8_ITERS=$ITERS FP8_CHECK=0 FP8_LAYOUTS=rcr \
    rocprofv3 -i "$pmcfile" --output-format csv -d "$outdir" \
        -- python3 test_python.py $M $N $K \
        > "$outdir/run.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "    ABORT fp8 $label $pmcset rc=$rc"
        tail -20 "$outdir/run.log"
    fi
}

build_kernels() {
    local M=$1 N=$2 K=$3 label=$4
    echo "=== building kernels for $label (M=$M N=$N K=$K) ==="
    rm -f tk_fp8_layouts.cpython-310-x86_64-linux-gnu.so
    make TARGET=tk_fp8_layouts SRC=kernel_fp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" \
        > "$OUTROOT/${label}_fp8_build.log" 2>&1 || \
        { echo "  FP8 build FAILED for $label"; return 1; }
    rm -f tk_mxfp8_layouts.cpython-310-x86_64-linux-gnu.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" \
        > "$OUTROOT/${label}_mxfp8_build.log" 2>&1 || \
        { echo "  MXFP8 build FAILED for $label"; return 1; }
}

for CELL in "${CELLS[@]}"; do
    IFS='|' read -r LABEL M N K <<< "$CELL"
    build_kernels $M $N $K $LABEL || continue
    run_pmc_fp8 $LABEL $M $N $K set1 "$PMC1"
    sleep 5
    run_pmc_fp8 $LABEL $M $N $K set3 "$PMC3"
    sleep 5
    run_pmc_mxfp8 $LABEL $M $N $K set1 "$PMC1"
    sleep 5
    run_pmc_mxfp8 $LABEL $M $N $K set3 "$PMC3"
    sleep 10
done

echo "DONE"
