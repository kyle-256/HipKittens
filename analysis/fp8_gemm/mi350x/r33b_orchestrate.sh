#!/bin/bash
# R33 Dev B: Stage A2d BABA orchestration.
# Builds two variants: rect (MXFP8_RECT_BLK_N=64) and default (no rect macro,
# square fastpath). Runs BABA-pattern paired bench at 4096^3 with 30s preheat.
#
# Usage: bash r33b_orchestrate.sh [GPU_ID]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GPU_ID="${1:-1}"
PHYS_GPU="$GPU_ID"
export PHYS_GPU
export HIP_VISIBLE_DEVICES="$GPU_ID"

source /tmp/wt-r33-b/env.src
export THUNDERKITTENS_ROOT=/tmp/wt-r33-b
export ROCM_PATH=/opt/rocm

M=4096; N=4096; K=4096
N_RUNS_PER_VARIANT=3  # Use lower to allow ABAB pattern (3 cycles A,B,A,B,A,B)
export N_RUNS=1  # Inner: each python invocation = 1 measurement

LOG_PREFIX="r33b_baba_gpu${GPU_ID}"

build_default() {
    echo "=== Build DEFAULT (square fastpath, no MXFP8_RECT_BLK_N) ==="
    rm -f tk_mxfp8_layouts*.so
    CPPFLAGS="-DM_DIM=4096 -DN_DIM=4096 -DK_DIM=4096" \
        make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        > "${LOG_PREFIX}_build_default.log" 2>&1
    md5sum tk_mxfp8_layouts*.so | tee "${LOG_PREFIX}_md5_default.txt"
}

build_rect() {
    echo "=== Build RECT (MXFP8_RECT_BLK_N=64) ==="
    rm -f tk_mxfp8_layouts*.so
    CPPFLAGS="-DM_DIM=4096 -DN_DIM=4096 -DK_DIM=4096 -DMXFP8_RECT_BLK_N=64" \
        make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        > "${LOG_PREFIX}_build_rect.log" 2>&1
    md5sum tk_mxfp8_layouts*.so | tee "${LOG_PREFIX}_md5_rect.txt"
}

# BABA pattern across two-process builds. 3 cycles -> 6 measurements (3 per variant).
# Cycle: B=square, A=rect, B=square, A=rect, B=square, A=rect.
# Issue: switching builds requires rm + recompile each toggle. Each compile takes
# ~30s. Solution: do one full square then one full rect repeated OR pre-build
# both .so files and copy the right one in.
#
# Simpler approach: build square .so to a backup name, build rect .so to another,
# then copy back-and-forth between cycles WITHOUT recompiling.

build_default
cp tk_mxfp8_layouts*.so tk_mxfp8_layouts_default.so.bak
build_rect
cp tk_mxfp8_layouts*.so tk_mxfp8_layouts_rect.so.bak

SO_NAME=$(ls tk_mxfp8_layouts.cpython-*.so | head -1)

# BABA pattern: square (default), rect, square (default), rect, square (default), rect
RECT_LIST=()
SQUARE_LIST=()

for cycle in 1 2 3; do
    echo "=== Cycle $cycle: SQUARE ==="
    cp tk_mxfp8_layouts_default.so.bak "$SO_NAME"
    md5sum "$SO_NAME"
    OUT=$(python3 r33b_baba_bench.py square $M $N $K 2>&1)
    echo "$OUT" >> "${LOG_PREFIX}_run_square_c${cycle}.log"
    SQ_TF=$(echo "$OUT" | grep "^RUN 0" | awk '{for(i=1;i<=NF;i++)if($i~/^tflops=/){print substr($i,8); exit}}')
    echo "[orchestrate] cycle $cycle SQUARE tflops=$SQ_TF"
    SQUARE_LIST+=("$SQ_TF")

    echo "=== Cycle $cycle: RECT ==="
    cp tk_mxfp8_layouts_rect.so.bak "$SO_NAME"
    md5sum "$SO_NAME"
    OUT=$(python3 r33b_baba_bench.py rect $M $N $K 2>&1)
    echo "$OUT" >> "${LOG_PREFIX}_run_rect_c${cycle}.log"
    RC_TF=$(echo "$OUT" | grep "^RUN 0" | awk '{for(i=1;i<=NF;i++)if($i~/^tflops=/){print substr($i,8); exit}}')
    echo "[orchestrate] cycle $cycle RECT tflops=$RC_TF"
    RECT_LIST+=("$RC_TF")
done

echo ""
echo "=== Stage A2d Summary GPU $GPU_ID ==="
echo "SQUARE_LIST: ${SQUARE_LIST[*]}"
echo "RECT_LIST: ${RECT_LIST[*]}"

python3 -c "
import statistics, math
sq = [${SQUARE_LIST[0]}, ${SQUARE_LIST[1]}, ${SQUARE_LIST[2]}]
rc = [${RECT_LIST[0]}, ${RECT_LIST[1]}, ${RECT_LIST[2]}]
sq_med = statistics.median(sq)
rc_med = statistics.median(rc)
sq_mean = statistics.mean(sq)
rc_mean = statistics.mean(rc)
sq_sd = statistics.stdev(sq)
rc_sd = statistics.stdev(rc)
n = len(sq)
delta = (rc_mean - sq_mean) / sq_mean * 100
# Welch t (unequal variances)
se = math.sqrt(sq_sd**2 / n + rc_sd**2 / n)
welch_t = (rc_mean - sq_mean) / se if se > 0 else float('inf')
print(f'SQUARE: median={sq_med:.2f} mean={sq_mean:.2f} sd={sq_sd:.2f}')
print(f'RECT:   median={rc_med:.2f} mean={rc_mean:.2f} sd={rc_sd:.2f}')
print(f'DELTA: {delta:+.2f}% (rect vs square)')
print(f'WELCH_T: {welch_t:.2f}')
"
