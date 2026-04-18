#!/bin/bash
# Build BOTH base and p304 .so files (different module names) for paired bench.
set -ue
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

M_DIM=${M_DIM:-4096}; N_DIM=${N_DIM:-4096}; K_DIM=${K_DIM:-4096}
build_one() {
  local tag=$1 extra=$2
  local target=tk_mxfp8_layouts_${tag}
  rm -f "${target}"*.so "${target}"
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=${target} SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=${M_DIM} -DN_DIM=${N_DIM} -DK_DIM=${K_DIM} -DPY_MODULE_NAME=${target} ${extra}" \
    > /tmp/r31d_build_${tag}.log 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "BUILD FAIL ${tag} rc=$rc"; tail -20 /tmp/r31d_build_${tag}.log; exit 1
  fi
  # Pybind module name is set by the SRC's PYBIND11_MODULE macro — need to
  # override. Instead patch via #define PYBIND_MODULE_NAME — fallback: rename
  # after build. Since the SRC defines tk_mxfp8_layouts as the python module
  # name, we cannot have two modules in same proc. Workaround: patch
  # PYBIND11_MODULE name via -D.
  echo "MD5 ${tag} $(md5sum ${target}*.so | awk '{print $1}')"
}

# Find the PYBIND11_MODULE name in source
grep -n "PYBIND11_MODULE" kernel_mxfp8_layouts.cpp | head -3

# We need different python module names. Inspect:
echo "---"
build_one base ""
build_one p304 "-DMXFP8_RCR_V2_PERSISTENT=1 -DMXFP8_RCR_V2_PERSISTENT_GRID=304"
