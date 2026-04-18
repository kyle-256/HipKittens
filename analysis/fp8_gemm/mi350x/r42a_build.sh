#!/bin/bash
# R42 Dev A — build all .so artifacts for the M=1 decode bench.
# Builds 4 modules:
#   - tk_mxfp8_decode_m1_8b   (-DMXFP8_DECODE_M1_ENABLE=1, M=1, N=4096, K=4096)
#   - tk_mxfp8_decode_m1_70b  (-DMXFP8_DECODE_M1_ENABLE=1, M=1, N=8192, K=8192)
#   - tk_mxfp8_baseline_8b    (legacy V1, M=1, N=4096, K=4096; no decode flag)
#   - tk_mxfp8_baseline_70b   (legacy V1, M=1, N=8192, K=8192; no decode flag)
#   - tk_fp8_8b               (FP8 reference per-tensor, M=1, N=4096, K=4096)
#   - tk_fp8_70b              (FP8 reference per-tensor, M=1, N=8192, K=8192)
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(git rev-parse --show-toplevel)"
export ROCM_PATH=/opt/rocm

build_mxfp8() {
    local target=$1 mdef=$2 ndef=$3 kdef=$4 enable=$5
    local extra=""
    [ "$enable" = "1" ] && extra="-DMXFP8_DECODE_M1_ENABLE=1"
    echo "=== build mxfp8 $target M=$mdef N=$ndef K=$kdef ENABLE=$enable ==="
    CPPFLAGS="-DM_DIM=$mdef -DN_DIM=$ndef -DK_DIM=$kdef -DPY_MODULE_NAME=$target $extra" \
        make -B TARGET=$target SRC=kernel_mxfp8_layouts.cpp 2>&1 | grep -E '^remark.*Function Name|TotalSGPRs|VGPRs:|VGPRs Spill|Occupancy|LDS Size|error' | grep -E 'gemv_m1|tail|error' | head -20 || true
}

build_fp8() {
    local target=$1 mdef=$2 ndef=$3 kdef=$4
    echo "=== build fp8 $target M=$mdef N=$ndef K=$kdef ==="
    CPPFLAGS="-DM_DIM=$mdef -DN_DIM=$ndef -DK_DIM=$kdef -DPY_MODULE_NAME=$target" \
        make -B TARGET=$target SRC=kernel_fp8_layouts.cpp 2>&1 | grep -E 'error|^/' | head -10 || true
}

# MXFP8 with decode-m1 fastpath enabled (R42A)
build_mxfp8 tk_mxfp8_decode_m1_8b   1 4096 4096 1
build_mxfp8 tk_mxfp8_decode_m1_70b  1 8192 8192 1

# MXFP8 baseline (legacy V1 -> tail kernel; same as R41 Dev C measured)
build_mxfp8 tk_mxfp8_baseline_8b    1 4096 4096 0
build_mxfp8 tk_mxfp8_baseline_70b   1 8192 8192 0

# FP8 per-tensor reference
build_fp8 tk_fp8_8b   1 4096 4096
build_fp8 tk_fp8_70b  1 8192 8192

echo "=== nm gate: decode-m1 .so should have gemv_m1_decode_kernel symbols ==="
for so in tk_mxfp8_decode_m1_8b tk_mxfp8_decode_m1_70b; do
    n=$(nm -D "$so".cpython-*.so 2>&1 | grep -c gemv_m1_decode_kernel || true)
    echo "  $so: $n gemv_m1 symbols (expected >=2)"
done

echo "=== nm gate: baseline .so should have ZERO gemv_m1 symbols (default-build invariance) ==="
for so in tk_mxfp8_baseline_8b tk_mxfp8_baseline_70b; do
    n=$(nm -D "$so".cpython-*.so 2>&1 | grep -c gemv_m1_decode_kernel || true)
    echo "  $so: $n gemv_m1 symbols (expected 0)"
done

echo "=== ALL BUILDS DONE ==="
