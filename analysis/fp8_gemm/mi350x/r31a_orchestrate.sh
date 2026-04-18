#!/bin/bash
# R31 Dev A: Stage A1 — rect-V2-CRR fastpath kernel scaffolding.
# Cell 1: default build byte-identical regression check (Stage A1a baseline).
# Cell 2: rect build clean-compile + resource report (Stage A1a primary).
# Cell 3: rect build GPU-fault test (Stage A1b).
# Cell 4: rect build with MIN_BLOCKS_PER_CU=3 occupancy probe (bonus).
#
# R31 outcome: SHIP Stage A1a + A1b. A1c (correct numerics) deferred to
# next cycle (requires Stage A2 host preshuffle work).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

echo "===== R31 cell=1 default build byte-identical regression =====" | tee r31a_cell1_default_build.txt
rm -f tk_mxfp8_layouts*.so
THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
  CXXFLAGS="-w -DM_DIM=4096 -DN_DIM=1024 -DK_DIM=8192" > r31a_build_cell1.log 2>&1
echo "default build rc=$?" | tee -a r31a_cell1_default_build.txt
md5sum tk_mxfp8_layouts*.so | tee -a r31a_cell1_default_build.txt
# Expected md5: 79c2816c54a00cf0680d32a36af25c98 (matches r30-a head pre-Stage-A1).

echo "===== R31 cell=2 rect build compile + resource report =====" | tee r31a_cell2_rect_build.txt
rm -f tk_mxfp8_layouts*.so
THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
  CXXFLAGS="-w -DM_DIM=4096 -DN_DIM=1024 -DK_DIM=8192 -DMXFP8_RECT_BLK_N=64" > r31a_build_cell2.log 2>&1
echo "rect build rc=$?" | tee -a r31a_cell2_rect_build.txt
md5sum tk_mxfp8_layouts*.so | tee -a r31a_cell2_rect_build.txt
echo "--- rect kernel resource report ---" | tee -a r31a_cell2_rect_build.txt
grep -A 11 "_Z34crr_exact_8wave_scaled_rect_kernel" r31a_build_cell2.log | tee -a r31a_cell2_rect_build.txt
echo "--- square V2-CRR kernel resource report (from cell 1 default build) ---" | tee -a r31a_cell2_rect_build.txt
grep -A 11 "_Z29crr_exact_8wave_scaled_kernelILb1ELi2E" r31a_build_cell1.log | tee -a r31a_cell2_rect_build.txt

echo "===== R31 cell=3 rect build GPU-fault test (Stage A1b) =====" | tee r31a_cell3_a1b.txt
HIP_VISIBLE_DEVICES=0 timeout 120 python3 r31a_stage_a1b_gpufault_test.py 2>&1 | tee -a r31a_cell3_a1b.txt
echo "stage_a1b rc=$?" | tee -a r31a_cell3_a1b.txt

echo "===== R31 cell=4 rect MIN_BLOCKS_PER_CU=3 probe (bonus) =====" | tee r31a_cell4_occprobe.txt
rm -f tk_mxfp8_layouts*.so
THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
  CXXFLAGS="-w -DM_DIM=4096 -DN_DIM=1024 -DK_DIM=8192 -DMXFP8_RECT_BLK_N=64 -DGEMM_MIN_BLOCKS_PER_CU=3" > r31a_build_cell4.log 2>&1
echo "occ=3 build rc=$?" | tee -a r31a_cell4_occprobe.txt
grep -A 11 "_Z34crr_exact_8wave_scaled_rect_kernel" r31a_build_cell4.log | tee -a r31a_cell4_occprobe.txt

echo "ALL CELLS DONE"
