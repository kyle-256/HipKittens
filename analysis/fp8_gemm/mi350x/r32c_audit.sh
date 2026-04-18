#!/bin/bash
# R32 Dev C C2: V2-CRR resource report audit for K=28672 vs K=8192.
# Build with -Rpass-analysis=kernel-resource-usage and grep VGPR/LDS/spill.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

audit_build() {
  local tag=$1 m=$2 n=$3 k=$4
  local log="$HERE/r32c_audit_${tag}.log"
  echo "===== AUDIT BUILD tag=$tag ${m}x${n}x${k} ====="
  rm -f tk_mxfp8_layouts*.so
  make clean >/dev/null 2>&1
  THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$m -DN_DIM=$n -DK_DIM=$k" \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "BUILD FAIL rc=$rc"; return 1
  fi
  local md5=$(md5sum tk_mxfp8_layouts*.so | awk '{print $1}')
  echo "BUILD OK md5=$md5"
  echo "----- crr_kernel resource lines -----"
  grep -E 'crr.*scaled|VGPRs|SGPRs|VGPR spill|LDS|Memory|Occupancy' "$log" | grep -i -E 'crr|vgpr|sgpr|lds|spill|occupancy' | head -60
  echo
}

# K=8192 baseline (use 70B Gate shape: 4096x28672x8192) — to match prior R28 audits
audit_build "k8192_70b_gate" 4096 28672 8192

# K=28672 (70B Down: 4096x8192x28672)
audit_build "k28672_70b_down" 4096 8192 28672

# K=8192 70B Down-shaped baseline (4096x8192x8192) for direct K-only comparison
audit_build "k8192_70b_qo" 4096 8192 8192

echo "ALL AUDITS DONE"
