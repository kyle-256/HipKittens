#!/bin/bash
# R53 Dev C — FP8 baseline PMC at 70B Down (M=4096 N=8192 K=28672)
# Confirm whether FP8 CRR's strong 70B Down performance matches its K=8192 behavior.
set -uo pipefail

cd "$(dirname "$0")/.."   # to mi350x
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"
export ROCM_PATH=/opt/rocm

GPU=${HIP_VISIBLE_DEVICES:-2}
M=4096; N=8192; K=28672
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-5}

OUTROOT="r53c_pmc_results"
PMC1="$OUTROOT/pmc_set1.txt"

run_pmc_fp8() {
    local layout=$1
    local pmcset=$2
    local pmcfile=$3
    local outdir="$OUTROOT/fp8_Down_${layout}_${pmcset}"
    mkdir -p "$outdir"
    if find "$outdir" -name "*counter_collection.csv" -size +5k 2>/dev/null | head -1 | grep -q . ; then
        echo "  skip fp8 Down $layout $pmcset (already)"
        return
    fi
    echo "  running fp8 Down $layout $pmcset -> $outdir"
    HIP_VISIBLE_DEVICES=$GPU \
    FP8_BUILD_M=$M FP8_BUILD_N=$N FP8_BUILD_K=$K \
    FP8_WARMUP=$WARMUP FP8_ITERS=$ITERS FP8_CHECK=0 FP8_LAYOUTS=$layout \
    rocprofv3 -i "$pmcfile" --output-format csv -d "$outdir" \
        -- python3 test_python.py $M $N $K \
        > "$outdir/run.log" 2>&1
}

for LAY in rrr crr; do
    echo "=== layout=$LAY ==="
    run_pmc_fp8 $LAY set1 "$PMC1"; sleep 5
done
echo "DONE"
