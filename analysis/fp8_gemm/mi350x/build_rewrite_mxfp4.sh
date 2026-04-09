#!/bin/bash
# Build pipeline: C++ → .s → Python rewrite → .so for MXFP4 KPAIR kernel
#
# Usage:
#   ./build_rewrite_mxfp4.sh                 # Full build with rewrite
#   ./build_rewrite_mxfp4.sh --no-rewrite    # Build baseline only
#   ./build_rewrite_mxfp4.sh --rewrite-only  # Rewrite + assemble (skip C++ compile)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

THUNDERKITTENS_ROOT="$(git rev-parse --show-toplevel)"
ROCM=/opt/rocm
HIPCC=$ROCM/bin/hipcc
CLANG=$ROCM/lib/llvm/bin/clang
LLD=$ROCM/lib/llvm/bin/ld.lld
BUNDLER=$ROCM/lib/llvm/bin/clang-offload-bundler
SRC=kernel_mxfp4_colfirst.cpp
REWRITER=rewrite_mxfp4_kpair.py

M=${M_DIM:-8192}
N=${N_DIM:-8192}
K=${K_DIM:-8192}
DIMS="-DM_DIM=$M -DN_DIM=$N -DK_DIM=$K"

PY_EXT="$(python3-config --extension-suffix)"
BASELINE="tk_mxfp4_colfirst${PY_EXT}"
OPTIMIZED="tk_mxfp4_colfirst_opt${PY_EXT}"
DEVICE_S="mxfp4_kpair_device.s"
OPT_S="mxfp4_kpair_opt.s"

# Fixed cuid shared between device .s generation and host compilation
CUID="mxfp4_kpair_rewrite_v1"

COMMON_FLAGS="-DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math"
INCLUDE_FLAGS="-I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype \
  $(python3 -m pybind11 --includes) -I$ROCM/include/hip -I$ROCM/include/rocrand"

MODE="${1:-}"

if [[ "$MODE" != "--rewrite-only" ]]; then
    echo "=== Step 1: hipcc baseline build ==="
    THUNDERKITTENS_ROOT="$THUNDERKITTENS_ROOT" ROCM_PATH=$ROCM \
      CPPFLAGS="$DIMS" make -B TARGET=tk_mxfp4_colfirst SRC=$SRC 2>&1 | grep -E 'remark:|error' || true

    echo "=== Step 2: Generate device .s using exact device cc1 flags ==="
    # Extract the exact device compilation command from hipcc -###
    # and change -emit-obj to -S for assembly output
    DEVICE_CMD=$($HIPCC -### $SRC $COMMON_FLAGS -std=c++20 -w $INCLUDE_FLAGS $DIMS \
      -shared -fPIC -cuid=$CUID -o /dev/null 2>&1 | grep fcuda-is-device)
    DEVICE_CMD=$(echo "$DEVICE_CMD" | sed 's/"-emit-obj"/"-S"/')
    DEVICE_CMD=$(echo "$DEVICE_CMD" | sed "s|-o\" \"[^\"]*|-o\" \"$DEVICE_S|")
    eval $DEVICE_CMD 2>&1 | grep -E 'remark:' || true
fi

if [[ "$MODE" == "--no-rewrite" ]]; then
    echo "=== Done (no rewrite) ==="
    exit 0
fi

echo "=== Step 3: Python rewrite ==="
python3 "$REWRITER" "$DEVICE_S" "$OPT_S"

echo "=== Step 4: Assemble → link → bundle → .so (cuid=$CUID) ==="
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

$CLANG -x assembler "$OPT_S" -target amdgcn-amd-amdhsa -mcpu=gfx950 -c -o "$TMP/device.o"

$LLD -m elf64_amdgpu --no-undefined -shared \
  -plugin-opt=-amdgpu-internalize-symbols --lto-partitions=8 \
  -plugin-opt=mcpu=gfx950 -plugin-opt=O3 --lto-CGO3 \
  --whole-archive -o "$TMP/device.out" "$TMP/device.o" --no-whole-archive

$BUNDLER -type=o -bundle-align=4096 \
  -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx950 \
  -input=/dev/null -input="$TMP/device.out" -output="$TMP/kernel.hipfb"

$CLANG -cc1 -triple x86_64-unknown-linux-gnu -aux-triple amdgcn-amd-amdhsa \
  -emit-obj -main-file-name $SRC \
  -mrelocation-model pic -pic-level 2 -mframe-pointer=none \
  -menable-no-infs -menable-no-nans -fapprox-func -funsafe-math-optimizations \
  -fno-signed-zeros -mreassociate -freciprocal-math -ffp-contract=fast \
  -fno-rounding-math -ffast-math -ffinite-math-only -complex-range=basic \
  -mconstructor-aliases -funwind-tables=2 -target-cpu x86-64 \
  -fcoverage-compilation-dir=. -resource-dir $ROCM/lib/llvm/lib/clang/20 \
  -internal-isystem $ROCM/lib/llvm/lib/clang/20/include/cuda_wrappers \
  -idirafter $ROCM/include \
  -include __clang_hip_runtime_wrapper.h \
  -D KITTENS_CDNA4 -D HIP_ENABLE_WARP_SYNC_BUILTINS \
  -I $ROCM/include/rocrand \
  -I ${THUNDERKITTENS_ROOT}/include -I ${THUNDERKITTENS_ROOT}/prototype \
  -I /usr/include/python3.10 -I /opt/venv/lib/python3.10/site-packages/pybind11/include \
  -I $ROCM/include/hip \
  -D M_DIM=$M -D N_DIM=$N -D K_DIM=$K \
  -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12 \
  -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/x86_64-linux-gnu/c++/12 \
  -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12/backward \
  -internal-isystem $ROCM/lib/llvm/lib/clang/20/include \
  -internal-isystem /usr/local/include \
  -internal-externc-isystem /usr/include/x86_64-linux-gnu \
  -internal-externc-isystem /usr/include \
  -O3 -w -std=c++20 -fdeprecated-macro \
  -fhip-new-launch-api -fgnuc-version=4.2.1 -fno-implicit-modules \
  -fskip-odr-check-in-gmf -fcxx-exceptions -fexceptions \
  -fcuda-include-gpubinary "$TMP/kernel.hipfb" \
  -cuid=$CUID -fgpu-approx-transcendentals -fcuda-allow-variadic-functions \
  -faddrsig -D__GCC_HAVE_DWARF2_CFI_ASM=1 \
  -o "$TMP/host.o" -x hip $SRC

$LLD -z relro --hash-style=gnu --eh-frame-hdr -m elf_x86_64 -shared \
  -o "$OPTIMIZED" \
  /lib/x86_64-linux-gnu/crti.o /usr/lib/gcc/x86_64-linux-gnu/12/crtbeginS.o \
  -L/usr/lib/gcc/x86_64-linux-gnu/12 -L/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu \
  --enable-new-dtags "$TMP/host.o" -L$ROCM/lib -rpath $ROCM/lib \
  -lamdhip64 -lstdc++ -lm -lgcc_s -lgcc -lc \
  /usr/lib/gcc/x86_64-linux-gnu/12/crtendS.o /lib/x86_64-linux-gnu/crtn.o

echo "=== Verify cuid match ==="
DEV_CUID=$(grep -oP '__hip_cuid_\K[0-9a-f]+' "$OPT_S" | head -1)
HOST_CUID=$(nm "$OPTIMIZED" | grep -oP '__hip_cuid_\K[0-9a-f]+' | head -1)
if [[ "$DEV_CUID" == "$HOST_CUID" ]]; then
    echo "  cuid MATCH: $DEV_CUID"
else
    echo "  cuid MISMATCH: device=$DEV_CUID host=$HOST_CUID"
    exit 1
fi

echo "=== Done ==="
echo "  Baseline: $BASELINE"
echo "  Optimized: $OPTIMIZED"
