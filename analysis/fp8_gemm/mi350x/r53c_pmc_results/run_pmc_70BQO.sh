#!/bin/bash
# R53 Dev C — Cross-check PMC at 70B Q/O (M=4096 N=8192 K=8192) for both layouts.
# Same K-discriminator hypothesis: profile CRR + RRR at K=8192 and compare with K=28672.
set -uo pipefail

cd "$(dirname "$0")/.."   # to mi350x dir
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"
export ROCM_PATH=/opt/rocm

GPU=${HIP_VISIBLE_DEVICES:-2}
M=4096; N=8192; K=8192
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-5}

OUTROOT="r53c_pmc_results"
PMC1="$OUTROOT/pmc_set1.txt"
PMC3="$OUTROOT/pmc_set3.txt"

run_pmc() {
    local layout=$1
    local pmcset=$2
    local pmcfile=$3
    local outdir="$OUTROOT/QO_${layout}_${pmcset}"
    mkdir -p "$outdir"
    if find "$outdir" -name "*counter_collection.csv" -size +5k 2>/dev/null | head -1 | grep -q . ; then
        echo "  skip QO $layout $pmcset (already has data)"
        return
    fi
    echo "  running QO $layout $pmcset -> $outdir"
    HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
    MXFP8_LAYOUTS=$layout MXFP8_PRESHUFFLE_QUANT=1 \
    rocprofv3 -i "$pmcfile" --output-format csv -d "$outdir" \
        -- python3 test_mxfp8_python.py $M $N $K \
        > "$outdir/run.log" 2>&1
}

for LAY in rrr crr; do
    echo "=== layout=$LAY ==="
    run_pmc $LAY set1 "$PMC1"; sleep 5
    run_pmc $LAY set3 "$PMC3"; sleep 5
done
echo "DONE"
