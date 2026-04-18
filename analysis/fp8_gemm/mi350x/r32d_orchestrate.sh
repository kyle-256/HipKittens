#!/bin/bash
# R32 Dev D: Stage A1 — rect-V2 RCR fastpath kernel scaffolding (mirrors
# R31 Dev A's rect-V2 CRR Stage A1).
# Cell 1: default build byte-identical regression check (Stage A1a baseline).
# Cell 2: rect build clean-compile + resource report (Stage A1a primary).
# Cell 3: rect build GPU-fault test on V2-RCR 4096^3 (Stage A1b).
#
# R32 outcome: SHIP Stage A1a + A1b. A1c (correct numerics) deferred to
# next cycle (requires Stage A2 host preshuffle + K_HALF audit work).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

echo "===== R32D cell=1 default build byte-identical regression =====" | tee r32d_cell1_default_build.txt
rm -f tk_mxfp8_layouts*.so
THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
  CXXFLAGS="-w -DM_DIM=4096 -DN_DIM=4096 -DK_DIM=4096" > r32d_default_postchange_build.log 2>&1
echo "default build rc=$?" | tee -a r32d_cell1_default_build.txt
md5sum tk_mxfp8_layouts*.so | tee -a r32d_cell1_default_build.txt
# Expected md5: 7d6c1ae78ee0001b45930835237673e6 (matches r32-d head pre-Stage-A1).

echo "===== R32D cell=2 rect build compile + resource report =====" | tee r32d_cell2_rect_build.txt
rm -f tk_mxfp8_layouts*.so
THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
  CXXFLAGS="-w -DM_DIM=4096 -DN_DIM=4096 -DK_DIM=4096 -DMXFP8_RECT_BLK_N=64" > r32d_rect_build.log 2>&1
echo "rect build rc=$?" | tee -a r32d_cell2_rect_build.txt
md5sum tk_mxfp8_layouts*.so | tee -a r32d_cell2_rect_build.txt
echo "--- rect RCR kernel resource report ---" | tee -a r32d_cell2_rect_build.txt
grep -A 11 "rcr_exact_8wave_scaled_rect_kernel" r32d_rect_build.log | tee -a r32d_cell2_rect_build.txt
echo "--- baseline V2-RCR kernel resource report (from cell 1 default build) ---" | tee -a r32d_cell2_rect_build.txt
grep -A 11 "_Z29rcr_exact_8wave_scaled_kernelILb1ELi2E" r32d_default_postchange_build.log | tee -a r32d_cell2_rect_build.txt

echo "===== R32D cell=3 rect build GPU-fault test (Stage A1b) =====" | tee r32d_cell3_a1b.txt
HIP_VISIBLE_DEVICES=3 timeout 120 python3 r32d_stage_a1b_gpufault_test.py 2>&1 | tee -a r32d_cell3_a1b.txt
echo "stage_a1b rc=${PIPESTATUS[0]}" | tee -a r32d_cell3_a1b.txt

echo "ALL CELLS DONE"
