#!/bin/bash
# R53 Dev C — FP8 baseline PMC at 70B Q/O (K=8192)
set -uo pipefail
cd "$(dirname "$0")/.."
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"; export ROCM_PATH=/opt/rocm
GPU=${HIP_VISIBLE_DEVICES:-2}
M=4096; N=8192; K=8192
PMC1="r53c_pmc_results/pmc_set1.txt"

run_pmc() {
    local layout=$1; local outdir="r53c_pmc_results/fp8_QO_${layout}_set1"
    mkdir -p "$outdir"
    if find "$outdir" -name "*counter_collection.csv" -size +5k 2>/dev/null | head -1 | grep -q .; then
        echo "  skip $layout"; return
    fi
    echo "  running fp8 QO $layout"
    HIP_VISIBLE_DEVICES=$GPU \
    FP8_BUILD_M=$M FP8_BUILD_N=$N FP8_BUILD_K=$K \
    FP8_WARMUP=3 FP8_ITERS=5 FP8_CHECK=0 FP8_LAYOUTS=$layout \
    rocprofv3 -i "$PMC1" --output-format csv -d "$outdir" \
        -- python3 test_python.py $M $N $K > "$outdir/run.log" 2>&1
}
for LAY in rrr crr; do run_pmc $LAY; sleep 5; done
echo DONE
