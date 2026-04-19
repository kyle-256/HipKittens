#!/bin/bash
# R53 Dev C — PMC profiling at 70B Down (M=4096 N=8192 K=28672)
# Profile both RRR and CRR with set1 (utilization) and set3 (stall attribution).
# Mirrors R52P methodology with shape adjusted for K=28672.
set -uo pipefail

cd "$(dirname "$0")/.."   # to mi350x dir
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"
export ROCM_PATH=/opt/rocm

GPU=${HIP_VISIBLE_DEVICES:-2}
M=4096; N=8192; K=28672
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-5}

OUTROOT="r53c_pmc_results"
PMC1="$OUTROOT/pmc_set1.txt"
PMC3="$OUTROOT/pmc_set3.txt"

run_pmc() {
    local layout=$1   # rrr or crr
    local pmcset=$2   # set1 or set3
    local pmcfile=$3  # path to pmc spec
    local outdir="$OUTROOT/${layout}_${pmcset}"
    mkdir -p "$outdir"
    if [ -f "$outdir/pmc_5/pmc_counter_collection.csv" ]; then
        echo "  skip $layout $pmcset (already complete)"
        return
    fi

    echo "  running $layout $pmcset -> $outdir"
    HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
    MXFP8_LAYOUTS=$layout MXFP8_PRESHUFFLE_QUANT=1 \
    rocprofv3 -i "$pmcfile" --output-format csv -d "$outdir" \
        -- python3 test_mxfp8_python.py $M $N $K \
        > "$outdir/run.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "    ABORT $layout $pmcset rc=$rc"
        cat "$outdir/run.log" | tail -20
    fi
}

for LAY in rrr crr; do
    echo "=== layout=$LAY ==="
    run_pmc $LAY set1 "$PMC1"
    sleep 5
    run_pmc $LAY set3 "$PMC3"
    sleep 5
done

echo "DONE"
