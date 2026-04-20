#!/bin/bash
# R55 Dev D — bench runner with strict SCLK protocol
# Usage: ./run_bench.sh <label> <M> <N> <K> [extra env...]
set -uo pipefail

WORKSPACE="/shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x/r55d_workspace"
RESULTS="/shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x/r55d_results"
GPU=${HIP_VISIBLE_DEVICES:-3}
WARMUP=${WARMUP:-100}
ITERS=${ITERS:-200}
RUNS=${RUNS:-5}
COOLDOWN=${COOLDOWN:-30}
LABEL=${1:?label required}
M=${2:?M required}
N=${3:?N required}
K=${4:?K required}
shift 4
EXTRA_ENV=("$@")

cd "$WORKSPACE"
mkdir -p "$RESULTS/${LABEL}"
OUT="$RESULTS/${LABEL}/runs.txt"
> "$OUT"

for i in $(seq 1 $RUNS); do
    echo "=== Run $i/$RUNS ($LABEL) ==="
    HIP_VISIBLE_DEVICES=$GPU \
        MXFP8_PRESHUFFLE_QUANT=1 MXFP8_LAYOUTS=rcr \
        MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS \
        MXFP8_OUTPUT="${RESULTS}/${LABEL}/run_${i}.json" \
        "${EXTRA_ENV[@]}" \
        python3 test_mxfp8_python.py $M $N $K 2>&1 | tee -a "$OUT" | tail -10 | head -3
    echo "Run $i done — cooldown ${COOLDOWN}s..."
    sleep $COOLDOWN
done

echo "=== Summary $LABEL ==="
python3 - <<EOF
import json, glob, statistics
runs = sorted(glob.glob("${RESULTS}/${LABEL}/run_*.json"))
ts = []
for r in runs:
    with open(r) as f:
        d = json.load(f)
    k = list(d.keys())[0]
    layout = list(d[k].keys())[0]
    ts.append(d[k][layout].get('tflops', 0.0))
print(f"  Runs ({len(ts)}): {[f'{t:.2f}' for t in ts]}")
print(f"  Median: {statistics.median(ts):.2f} TFLOPS")
print(f"  Mean:   {statistics.mean(ts):.2f} TFLOPS")
print(f"  Stdev:  {statistics.stdev(ts) if len(ts) > 1 else 0.0:.2f} TFLOPS")
EOF
