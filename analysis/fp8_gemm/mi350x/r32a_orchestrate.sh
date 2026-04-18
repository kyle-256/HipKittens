#!/bin/bash
# R32 Dev A: Stage A2 — rect-V2-CRR fastpath kernel correct numerics.
# Stage A2a: preshuffle_v2_b_rect host function (built into r32a_bench.py).
# Stage A2b: rewrite load_col_from_v2_st_rect to fetch both K_HALFs (in .inc/.cpp).
# Stage A2c: SNR >= 48 dB + det 3/3 PASS at 70B KV 4096x1024x8192 — SHIP gate.
# Stage A2d: paired BABA bench rect vs square at 70B KV (only if A2c PASS).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

PHYS_GPU="${PHYS_GPU:-0}"

echo "===== R32 cell=1 default build (square baseline, byte-identical regression) ====="  | tee r32a_cell1_default_build.txt
rm -f tk_mxfp8_layouts*.so
THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
  CXXFLAGS="-w -DM_DIM=4096 -DN_DIM=1024 -DK_DIM=8192" > r32a_build_cell1.log 2>&1
echo "default build rc=$?" | tee -a r32a_cell1_default_build.txt
md5sum tk_mxfp8_layouts*.so | tee -a r32a_cell1_default_build.txt

echo "===== R32 cell=2 rect build clean-compile + resource report ====="  | tee r32a_cell2_rect_build.txt
rm -f tk_mxfp8_layouts*.so
THUNDERKITTENS_ROOT="$WT" make -j8 TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
  CXXFLAGS="-w -DM_DIM=4096 -DN_DIM=1024 -DK_DIM=8192 -DMXFP8_RECT_BLK_N=64" > r32a_build_cell2.log 2>&1
echo "rect build rc=$?" | tee -a r32a_cell2_rect_build.txt
md5sum tk_mxfp8_layouts*.so | tee -a r32a_cell2_rect_build.txt
echo "--- rect kernel resource report ---" | tee -a r32a_cell2_rect_build.txt
grep -A 11 "_Z34crr_exact_8wave_scaled_rect_kernel" r32a_build_cell2.log | tee -a r32a_cell2_rect_build.txt

echo "===== R32 cell=3 Stage A2c: SNR + det at 70B KV (rect kernel still loaded) =====" | tee r32a_cell3_a2c.txt
HIP_VISIBLE_DEVICES=0 PHYS_GPU="$PHYS_GPU" N_RUNS=3 timeout 300 python3 r32a_bench.py 4096 1024 8192 2>&1 | tee -a r32a_cell3_a2c.txt
echo "stage_a2c rc=$?" | tee -a r32a_cell3_a2c.txt

echo "ALL CELLS DONE"
