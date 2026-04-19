#!/bin/bash
# R45 Dev C — build .so artifacts for B-side scalar-load vectorization
# (gemm_tail_kernel_smallm_b32_bvec, MXFP8_SMALLM_B32_BSIDE_VEC_ENABLE).
#
# Bench shapes (RCR primary; M=32/128, N=1024, K=4096/8192):
#   8B-K/V  M=32  N=1024 K=4096
#   70B-K/V M=32  N=1024 K=8192
#   8B-K/V  M=128 N=1024 K=4096
#   70B-K/V M=128 N=1024 K=8192
#
# Per shape, builds 3 .so:
#   - tk_mxfp8_r45c_bvec_<shape>      (-DMXFP8_SMALLM_B32_FASTPATH=1 -DMXFP8_SMALLM_B32_BSIDE_VEC_ENABLE=1)
#   - tk_mxfp8_r45c_r42b_<shape>      (-DMXFP8_SMALLM_B32_FASTPATH=1; R42B SHIP baseline)
#   - tk_fp8_r45c_<shape>             (FP8 reference per-tensor)
#
# Default 8192³ byte-identity check: separately built without ANY of these
# macros to ensure the new variant adds 0 symbols when default-OFF.

set -uo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(git rev-parse --show-toplevel)"
export ROCM_PATH=/opt/rocm

# M_DIM has to be a value where the V2 fastpath predicate (g.m == M_DIM) is
# either disabled or doesn't fire for these small-M shapes; we use M_DIM=256
# so that V2 predicate fails (g.m=32 or 128 != 256) and falls through to V1
# tail kernel where the SMALLM-B32 dispatch lives. Same approach as R42a/R44a.
BASELINE_MDIM=256

build_mxfp8_bvec() {
    local target=$1 ndef=$2 kdef=$3
    rm -f "${target}.cpython"-*.so
    echo "=== build mxfp8 BVEC $target M=$BASELINE_MDIM N=$ndef K=$kdef ==="
    CPPFLAGS="-DM_DIM=$BASELINE_MDIM -DN_DIM=$ndef -DK_DIM=$kdef -DPY_MODULE_NAME=$target -DMXFP8_SMALLM_B32_FASTPATH=1 -DMXFP8_SMALLM_B32_BSIDE_VEC_ENABLE=1" \
        make TARGET=$target SRC=kernel_mxfp8_layouts.cpp 2>&1 \
        | grep -E '(smallm_b32_bvec|VGPRs:|VGPRs Spill|Occupancy|error|^/.*: error)' \
        | grep -E '(smallm_b32_bvec|error)' | head -20 || true
}

build_mxfp8_r42b() {
    local target=$1 ndef=$2 kdef=$3
    rm -f "${target}.cpython"-*.so
    echo "=== build mxfp8 R42B baseline $target M=$BASELINE_MDIM N=$ndef K=$kdef ==="
    CPPFLAGS="-DM_DIM=$BASELINE_MDIM -DN_DIM=$ndef -DK_DIM=$kdef -DPY_MODULE_NAME=$target -DMXFP8_SMALLM_B32_FASTPATH=1" \
        make TARGET=$target SRC=kernel_mxfp8_layouts.cpp 2>&1 \
        | grep -E 'error|^/.*: error' | head -10 || true
}

build_fp8() {
    local target=$1 ndef=$2 kdef=$3
    rm -f "${target}.cpython"-*.so
    echo "=== build fp8 $target M=$BASELINE_MDIM N=$ndef K=$kdef ==="
    CPPFLAGS="-DM_DIM=$BASELINE_MDIM -DN_DIM=$ndef -DK_DIM=$kdef -DPY_MODULE_NAME=$target" \
        make TARGET=$target SRC=kernel_fp8_layouts.cpp 2>&1 | grep -E 'error' | head -5 || true
}

# 4 KV-decode shapes (RCR), parameterized by (N, K). M is runtime (32 or 128).
SHAPES=(
    "8b_n1024_k4096:1024:4096"
    "70b_n1024_k8192:1024:8192"
)
for entry in "${SHAPES[@]}"; do
    name="${entry%%:*}"; rest="${entry#*:}"
    N="${rest%%:*}"; K="${rest##*:}"
    build_mxfp8_bvec    tk_mxfp8_r45c_bvec_${name}    $N $K
    build_mxfp8_r42b    tk_mxfp8_r45c_r42b_${name}    $N $K
    build_fp8           tk_fp8_r45c_${name}           $N $K
done

# Default 8192³ byte-identity build (NO macros). Compare md5 against pre-R45C HEAD.
echo "=== build mxfp8 DEFAULT 8192³ (byte-identity check) ==="
rm -f tk_mxfp8_r45c_default.cpython-*.so
CPPFLAGS="-DPY_MODULE_NAME=tk_mxfp8_r45c_default" \
    make TARGET=tk_mxfp8_r45c_default SRC=kernel_mxfp8_layouts.cpp 2>&1 | grep -E 'error' | head -5 || true

echo "=== nm-gate: BVEC .so should have gemm_tail_kernel_smallm_b32_bvec symbols (expect 6 = 3 layouts × 2 PQ) ==="
for entry in "${SHAPES[@]}"; do
    name="${entry%%:*}"
    so=tk_mxfp8_r45c_bvec_${name}.cpython-*.so
    if ls $so >/dev/null 2>&1; then
        n=$(nm -D $so 2>&1 | grep -c gemm_tail_kernel_smallm_b32_bvec || true)
        echo "  bvec $name: $n bvec symbols (expected 6)"
    fi
done

echo "=== nm-gate: R42B baseline .so should have ZERO bvec symbols ==="
for entry in "${SHAPES[@]}"; do
    name="${entry%%:*}"
    so=tk_mxfp8_r45c_r42b_${name}.cpython-*.so
    if ls $so >/dev/null 2>&1; then
        n=$(nm -D $so 2>&1 | grep -c gemm_tail_kernel_smallm_b32_bvec || true)
        echo "  r42b $name: $n bvec symbols (expected 0)"
    fi
done

echo "=== nm-gate: DEFAULT 8192³ .so should have ZERO bvec AND ZERO r42b smallm symbols ==="
so=tk_mxfp8_r45c_default.cpython-*.so
if ls $so >/dev/null 2>&1; then
    nbvec=$(nm -D $so 2>&1 | grep -c gemm_tail_kernel_smallm_b32_bvec || true)
    nr42b=$(nm -D $so 2>&1 | grep -c gemm_tail_kernel_smallm_b32 || true)
    nr42bonly=$((nr42b - nbvec))
    echo "  default: $nbvec bvec symbols (expected 0); $nr42bonly r42b-only smallm symbols (expected 0)"
fi

echo "=== ALL BUILDS DONE ==="
