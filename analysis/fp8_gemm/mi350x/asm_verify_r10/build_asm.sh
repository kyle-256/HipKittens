#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TK_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
KERNEL=$SCRIPT_DIR/../kernel_mxfp4_gluon_cpp.cpp

BASE="--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
-I/opt/rocm/include/rocrand -I$TK_ROOT/include -I$TK_ROOT/prototype -I/opt/rocm/include/hip \
-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include \
-fPIC -std=c++20 -w -DK_DIM=32768 -DN_DIM=4096 \
-DSTEP3_BARRIER_VMCNT=16 -DWAVES_PER_EU_2=1 \
--cuda-device-only -S"

LABEL="$1"
shift
EXTRA="$@"
OUT="$SCRIPT_DIR/asm_${LABEL}.s"
/opt/rocm/bin/hipcc $KERNEL $BASE $EXTRA -o "$OUT" 2>&1 | tail -3
md5sum "$OUT" | head
wc -l "$OUT" | head
