#!/bin/bash
# R44 Dev A — build .so artifacts for the M=2..16 small-batch decode bench.
# Builds (per shape):
#   - tk_mxfp8_r44a_decode_<shape>      (-DMXFP8_DECODE_M2_16_ENABLE=1)
#   - tk_mxfp8_r44a_baseline_<shape>    (legacy V1, no flag)
#   - tk_fp8_r44a_<shape>               (FP8 reference per-tensor)
#
# Bench shapes (M ∈ {4, 8, 16}; RCR primary; RRR optional):
#   8B Q/K/V/O  : M x 4096 x 4096
#   70B Q/K/V/O : M x 8192 x 8192

set -uo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(git rev-parse --show-toplevel)"
export ROCM_PATH=/opt/rocm

build_mxfp8() {
    local target=$1 mdef=$2 ndef=$3 kdef=$4 enable=$5
    local extra=""
    [ "$enable" = "1" ] && extra="-DMXFP8_DECODE_M2_16_ENABLE=1"
    rm -f "${target}.cpython"-*.so
    echo "=== build mxfp8 $target M=$mdef N=$ndef K=$kdef ENABLE=$enable ==="
    CPPFLAGS="-DM_DIM=$mdef -DN_DIM=$ndef -DK_DIM=$kdef -DPY_MODULE_NAME=$target $extra" \
        make TARGET=$target SRC=kernel_mxfp8_layouts.cpp 2>&1 \
        | grep -E '(gemv_m2_16|VGPRs:|VGPRs Spill|Occupancy|error|^/.*: error)' \
        | grep -E '(gemv_m2_16|error)' | head -20 || true
}

build_fp8() {
    local target=$1 mdef=$2 ndef=$3 kdef=$4
    rm -f "${target}.cpython"-*.so
    echo "=== build fp8 $target M=$mdef N=$ndef K=$kdef ==="
    CPPFLAGS="-DM_DIM=$mdef -DN_DIM=$ndef -DK_DIM=$kdef -DPY_MODULE_NAME=$target" \
        make TARGET=$target SRC=kernel_fp8_layouts.cpp 2>&1 | grep -E 'error' | head -5 || true
}

# M=4 / M=8 / M=16 × {4096×4096 (8B), 8192×8192 (70B)}
#
# IMPORTANT: For "decode" .so we build with M_DIM=$M so the R44A predicate
# `can_use_decode_m2_16` (runtime g.m == M) fires and the dispatch trace
# names the cell.
# For "baseline" / "fp8" we build with M_DIM=BLK (256) so the V2 fastpath
# predicate `g.m == M_DIM` does NOT match runtime g.m=$M (which would set
# grid=0 and crash with hipErrorInvalidConfiguration). The runtime falls
# through to V1-LEGACY tail kernel — the same path production hits when
# small M reaches an M_DIM=8192 default-built .so.
BASELINE_MDIM=256
for M in 4 8 16; do
    # 8B shape
    build_mxfp8 tk_mxfp8_r44a_decode_${M}x4kx4k     $M             4096 4096 1
    build_mxfp8 tk_mxfp8_r44a_baseline_${M}x4kx4k   $BASELINE_MDIM 4096 4096 0
    build_fp8   tk_fp8_r44a_${M}x4kx4k              $BASELINE_MDIM 4096 4096
    # 70B shape
    build_mxfp8 tk_mxfp8_r44a_decode_${M}x8kx8k     $M             8192 8192 1
    build_mxfp8 tk_mxfp8_r44a_baseline_${M}x8kx8k   $BASELINE_MDIM 8192 8192 0
    build_fp8   tk_fp8_r44a_${M}x8kx8k              $BASELINE_MDIM 8192 8192
done

echo "=== nm gate: R44A .so should have gemv_m2_16_decode_kernel symbols (>=1 per layout) ==="
for M in 4 8 16; do
    for shape in 4kx4k 8kx8k; do
        for variant in decode baseline; do
            so="tk_mxfp8_r44a_${variant}_${M}x${shape}".cpython-*.so
            if ls $so >/dev/null 2>&1; then
                n=$(nm -D $so 2>&1 | grep -c gemv_m2_16_decode_kernel || true)
                expected=0
                [ "$variant" = "decode" ] && expected=">=1"
                echo "  $variant ${M}x${shape}: $n gemv_m2_16 symbols (expected $expected)"
            fi
        done
    done
done

echo "=== ALL BUILDS DONE ==="
