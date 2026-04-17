#!/bin/bash
# Round 13 OptC ASM-diff verification.
# For 4096x32768x28672 (DLA3 / R11 _v20_memc_r11_iterilp WIN), build:
#   (a) iterilp baseline (parent _v20_memc + iterilp)
#   (b) iterilp + iterminreg replacement (just iterminreg, no iterilp)
#   (c) iterilp + max-memory-clause stacked TWICE (test if compiler dedupes or stacks)
#   (d) iterilp + amdgpu-mfma-padding-ratio=10
#   (e) iterilp + amdgpu-mfma-padding-ratio=25
#   (f) iterilp + STEP12_BR_LGKMCNT=4
#   (g) iterminreg + max-memory-clause
#   (h) iterilp WITHOUT max-memory-clause (test if memc is load-bearing)
# Then size+md5 each .s. Equal size+md5 => silent no-op.
set -e
cd "$(dirname "$0")"
SCRIPT_DIR=$(pwd)
TK_ROOT=$(realpath "${SCRIPT_DIR}/../../..")
ASMDIR="${SCRIPT_DIR}/asm_round13_optC"
mkdir -p "${ASMDIR}"

# Use 4096x32768x28672 shape's _v20_memc parent flags
N=32768
K=28672
PARENT_FLAGS="-DSTEP3_BARRIER_VMCNT=20 -mllvm -amdgpu-sched-strategy=max-memory-clause"

BASE_CC="--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
  -I/opt/rocm/include/rocrand -I${TK_ROOT}/include -I${TK_ROOT}/prototype \
  -I/opt/rocm/include/hip \
  -I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include \
  -fPIC -std=c++20 -w \
  -DK_DIM=${K} -DN_DIM=${N} \
  -S -emit-llvm=false"

# Note: -S generates ASM. We want the GPU asm; use --cuda-device-only -x hip + -S
build_asm() {
  local label="$1"
  shift
  local extra="$*"
  local out="${ASMDIR}/${label}.s"
  if [ -f "${out}" ]; then
    echo "  cached ${label}"
    return 0
  fi
  echo "  building ${label} ..."
  /opt/rocm/bin/hipcc "${SCRIPT_DIR}/kernel_mxfp4_gluon_cpp.cpp" \
    --offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
    -I/opt/rocm/include/rocrand -I${TK_ROOT}/include -I${TK_ROOT}/prototype \
    -I/opt/rocm/include/hip \
    -I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include \
    -fPIC -std=c++20 -w \
    -DK_DIM=${K} -DN_DIM=${N} \
    ${PARENT_FLAGS} ${extra} \
    --cuda-device-only -S -o "${out}" 2>&1 | tail -5
  if [ ! -f "${out}" ]; then
    echo "  FAIL ${label}"
    return 1
  fi
}

# (a) iterilp baseline (current win flag)
build_asm "a_iterilp_base"      "-mllvm -amdgpu-sched-strategy=iterative-ilp"

# (b) iterminreg replacement (no iterilp)
build_asm "b_iterminreg_alone"  "-mllvm -amdgpu-sched-strategy=iterative-minreg"

# (c) iterilp + extra max-memory-clause (already in PARENT)
#    PARENT has max-memory-clause; adding iterilp last takes precedence?
build_asm "c_iterilp_mmc_dup"   "-mllvm -amdgpu-sched-strategy=iterative-ilp -mllvm -amdgpu-sched-strategy=max-memory-clause"

# (d) iterilp + mfma-padding-ratio=10
build_asm "d_iterilp_pad10"     "-mllvm -amdgpu-sched-strategy=iterative-ilp -mllvm -amdgpu-mfma-padding-ratio=10"

# (e) iterilp + mfma-padding-ratio=25
build_asm "e_iterilp_pad25"     "-mllvm -amdgpu-sched-strategy=iterative-ilp -mllvm -amdgpu-mfma-padding-ratio=25"

# (f) iterilp + STEP12_BR_LGKMCNT=4
build_asm "f_iterilp_brlgk4"    "-DSTEP12_BR_LGKMCNT=4 -mllvm -amdgpu-sched-strategy=iterative-ilp"

# (g) iterminreg + max-memory-clause
build_asm "g_iterminreg_mmc"    "-mllvm -amdgpu-sched-strategy=iterative-minreg -mllvm -amdgpu-sched-strategy=max-memory-clause"

# (h) iterilp WITHOUT memc (override PARENT's memc)
#    can't easily override; skip for now

# (i) iterative-max-occupancy-experimental
build_asm "i_iter_maxocc"       "-mllvm -amdgpu-sched-strategy=iterative-max-occupancy-experimental"

echo
echo "======== Size + MD5 comparison ========"
for f in "${ASMDIR}"/*.s; do
  sz=$(wc -c < "$f")
  md=$(md5sum "$f" | awk '{print $1}')
  ln=$(wc -l < "$f")
  echo "$(basename ${f} .s)  size=${sz}  lines=${ln}  md5=${md}"
done
