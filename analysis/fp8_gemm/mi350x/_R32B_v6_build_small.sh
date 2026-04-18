#!/bin/bash
# Build V6 split-K variants at SMALL shape (4096^3) for SNR validation.
set -eo pipefail

KERNEL=/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp_v6.cpp
OUT=/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_v6_small
mkdir -p "$OUT"

COMMON_FLAGS=(
  --offload-arch=gfx950
  -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS
  -ffast-math
  -I/opt/rocm/include/rocrand
  -I/shared_nfs/kyle/test/HipKittens/include
  -I/shared_nfs/kyle/test/HipKittens/prototype
  -I/opt/rocm/include/hip
  -I/usr/include/python3.10
  -I/opt/venv/lib/python3.10/site-packages/pybind11/include
  -shared -fPIC -std=c++20 -w
  -L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu
  -L/usr/lib/x86_64-linux-gnu -ldl -lm
  -DM_DIM=4096 -DN_DIM=4096 -DK_DIM=4096
  -DTAIL_SPLIT=1
  -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12
  -DBARRIER_TO_WAITCNT_ALL=1
  -mllvm -amdgpu-sched-strategy=max-memory-clause
  -DGROUP_SIZE_M=8
)

build_one() {
  local NAME=$1; shift
  local WRAP="$OUT/wrap_${NAME}.cpp"
  local OUT_SO="$OUT/tk_${NAME}.cpython-310-x86_64-linux-gnu.so"
  echo "[build] $NAME"
  # Make a wrap file with renamed module
  sed "s/PYBIND11_MODULE(tk_mxfp4_gluon_cpp,/PYBIND11_MODULE(tk_${NAME},/" "$KERNEL" > "$WRAP"
  /opt/rocm/bin/hipcc "$WRAP" \
    "${COMMON_FLAGS[@]}" \
    "$@" \
    -o "$OUT_SO" 2>&1 | tail -25
  echo "[ok]    $NAME"
}

build_one mxfp4_v6_small_split1   -DK_SPLIT=1
build_one mxfp4_v6_small_split2_s0 -DK_SPLIT=2 -DS_IDX=0
build_one mxfp4_v6_small_split2_s1 -DK_SPLIT=2 -DS_IDX=1

echo "DONE"
ls -la "$OUT"
