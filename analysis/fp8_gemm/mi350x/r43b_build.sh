#!/bin/bash
# R43 Dev B — build .so artifacts for the M=1 RRR/CRR decode bench.
# Builds (per layout/shape):
#   - tk_mxfp8_decode_m1rc_<llama>_<layout>  (-DMXFP8_DECODE_M1_RRR_CRR_ENABLE=1, M=1)
#   - tk_mxfp8_baseline_<llama>_<layout>     (legacy V1, M=1, no flag)
#   - tk_fp8_<llama>_<layout>                (FP8 reference per-tensor, M=1)
#
# LLaMA decode shapes covered (M=1):
#   8B Q/K/V/O proj      (RCR/RRR/CRR all candidates): 1 x 4096 x 4096
#   8B SwiGLU gate/up    (RRR):                        1 x 14336 x 4096
#   8B SwiGLU down       (RRR):                        1 x 4096 x 14336
#   8B KV (RRR for V):                                 1 x 1024 x 4096
#   70B Q/K/V/O proj     (RCR/RRR/CRR all candidates): 1 x 8192 x 8192

set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(git rev-parse --show-toplevel)"
export ROCM_PATH=/opt/rocm

build_mxfp8() {
    local target=$1 mdef=$2 ndef=$3 kdef=$4 enable=$5
    local extra=""
    [ "$enable" = "1" ] && extra="-DMXFP8_DECODE_M1_RRR_CRR_ENABLE=1"
    echo "=== build mxfp8 $target M=$mdef N=$ndef K=$kdef ENABLE=$enable ==="
    CPPFLAGS="-DM_DIM=$mdef -DN_DIM=$ndef -DK_DIM=$kdef -DPY_MODULE_NAME=$target $extra" \
        make -B TARGET=$target SRC=kernel_mxfp8_layouts.cpp 2>&1 | grep -E '^remark|TotalSGPRs|VGPRs:|VGPRs Spill|Occupancy|LDS Size|error|^/.*: error' | grep -E 'gemv_m1|tail|error' | head -20 || true
}

build_fp8() {
    local target=$1 mdef=$2 ndef=$3 kdef=$4
    echo "=== build fp8 $target M=$mdef N=$ndef K=$kdef ==="
    CPPFLAGS="-DM_DIM=$mdef -DN_DIM=$ndef -DK_DIM=$kdef -DPY_MODULE_NAME=$target" \
        make -B TARGET=$target SRC=kernel_fp8_layouts.cpp 2>&1 | grep -E 'error|^/' | head -10 || true
}

# 8B Q/K/V/O 1x4096x4096 (RRR, CRR) and 70B 1x8192x8192 (RRR, CRR)
build_mxfp8 tk_mxfp8_decode_m1rc_8b_4kx4k   1 4096 4096 1
build_mxfp8 tk_mxfp8_decode_m1rc_70b_8kx8k  1 8192 8192 1
# 8B SwiGLU shapes (only need RRR; we still build 1x14336x4096 / 1x4096x14336)
build_mxfp8 tk_mxfp8_decode_m1rc_8b_14kx4k  1 14336 4096 1
build_mxfp8 tk_mxfp8_decode_m1rc_8b_4kx14k  1 4096 14336 1

# MXFP8 baseline (legacy V1 -> tail kernel)
build_mxfp8 tk_mxfp8_baseline_8b_4kx4k      1 4096 4096 0
build_mxfp8 tk_mxfp8_baseline_70b_8kx8k     1 8192 8192 0
build_mxfp8 tk_mxfp8_baseline_8b_14kx4k     1 14336 4096 0
build_mxfp8 tk_mxfp8_baseline_8b_4kx14k     1 4096 14336 0

# FP8 per-tensor reference
build_fp8 tk_fp8_8b_4kx4k        1 4096 4096
build_fp8 tk_fp8_70b_8kx8k       1 8192 8192
build_fp8 tk_fp8_8b_14kx4k       1 14336 4096
build_fp8 tk_fp8_8b_4kx14k       1 4096 14336

echo "=== nm gate: decode-m1rc .so should have gemv_m1_decode_rrr_crr_kernel symbols ==="
for so in tk_mxfp8_decode_m1rc_8b_4kx4k tk_mxfp8_decode_m1rc_70b_8kx8k \
          tk_mxfp8_decode_m1rc_8b_14kx4k tk_mxfp8_decode_m1rc_8b_4kx14k; do
    n=$(nm -D "$so".cpython-*.so 2>&1 | grep -c gemv_m1_decode_rrr_crr_kernel || true)
    echo "  $so: $n gemv_m1_rrr_crr symbols (expected >=2 = RRR + CRR true)"
done

echo "=== nm gate: baseline .so should have ZERO gemv_m1_decode_rrr_crr symbols (default-build invariance) ==="
for so in tk_mxfp8_baseline_8b_4kx4k tk_mxfp8_baseline_70b_8kx8k \
          tk_mxfp8_baseline_8b_14kx4k tk_mxfp8_baseline_8b_4kx14k; do
    n=$(nm -D "$so".cpython-*.so 2>&1 | grep -c gemv_m1_decode_rrr_crr_kernel || true)
    echo "  $so: $n gemv_m1_rrr_crr symbols (expected 0)"
done

echo "=== ALL BUILDS DONE ==="
