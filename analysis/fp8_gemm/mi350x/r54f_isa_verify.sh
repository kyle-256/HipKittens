#!/bin/bash
# R54 Dev F: Phase 1 ISA verification — confirm byte-identical resources
# (VGPR/spill/scratch/LDS) across the XCD swizzle granularity sweep.
# Per-shape build at 70B Gate/Up (M=4096 N=28672 K=8192).
#
# Sweep axes:
#   - MXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS in {4, 8, 16, 32}  (default 8)
#   - MXFP8_CRR_BLOCK_SWIZZLE_GROUP_M  in {1, 2, 4, 8, 16}  (default 4)

set -euo pipefail
cd "$(dirname "$0")/r54f_workspace"
export THUNDERKITTENS_ROOT="$(cd ../../../.. && pwd)"

OUTDIR="../r54f_results"
mkdir -p "$OUTDIR/isa"

M=4096; N=28672; K=8192

build_one() {
    local TAG=$1; local FLAGS=$2
    rm -f tk_mxfp8_layouts*.so
    echo "=== build TAG=$TAG FLAGS=$FLAGS"
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $FLAGS" \
        > "$OUTDIR/isa/${TAG}_build.log" 2>&1
    grep -A1 "kernel_mxfp8_layouts.cpp:2415:1: remark:.*Function Name" "$OUTDIR/isa/${TAG}_build.log" || true
    echo "  resources at line 2415 (CRR kernel):"
    awk '/kernel_mxfp8_layouts.cpp:2415:1: remark:/ {print "    "$0}' "$OUTDIR/isa/${TAG}_build.log" | head -10
}

# default reference
build_one "default"       ""

# NUM_XCDS sweep, GROUP_M=4 (default)
build_one "xcds04_gm04"   "-DMXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS=4"
build_one "xcds08_gm04"   "-DMXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS=8"
build_one "xcds16_gm04"   "-DMXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS=16"
build_one "xcds32_gm04"   "-DMXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS=32"

# GROUP_M sweep, NUM_XCDS=8 (default)
build_one "xcds08_gm01"   "-DMXFP8_CRR_BLOCK_SWIZZLE_GROUP_M=1"
build_one "xcds08_gm02"   "-DMXFP8_CRR_BLOCK_SWIZZLE_GROUP_M=2"
build_one "xcds08_gm08"   "-DMXFP8_CRR_BLOCK_SWIZZLE_GROUP_M=8"
build_one "xcds08_gm16"   "-DMXFP8_CRR_BLOCK_SWIZZLE_GROUP_M=16"

echo ""
echo "=========== ISA RESOURCE SUMMARY (line 2415 CRR kernel) ==========="
printf "%-24s %-10s %-10s %-12s %-14s %-10s %-10s\n" "TAG" "VGPRs" "SGPRs" "LDS_bytes" "Scratch_lane" "VSpill" "SSpill"
for tag in default xcds04_gm04 xcds08_gm04 xcds16_gm04 xcds32_gm04 xcds08_gm01 xcds08_gm02 xcds08_gm08 xcds08_gm16; do
    log="$OUTDIR/isa/${tag}_build.log"
    block=$(awk '/kernel_mxfp8_layouts.cpp:2415:1: remark:/' "$log")
    vgpr=$(echo "$block" | grep -m1 "VGPRs:" | grep -v Spill | sed 's/.*VGPRs:[[:space:]]*\([0-9]*\).*/\1/')
    sgpr=$(echo "$block" | grep -m1 "TotalSGPRs:" | sed 's/.*TotalSGPRs:[[:space:]]*\([0-9]*\).*/\1/')
    lds=$(echo  "$block" | grep -m1 "LDS Size" | sed 's/.*\[bytes\/block\]:[[:space:]]*\([0-9]*\).*/\1/')
    scr=$(echo  "$block" | grep -m1 "ScratchSize" | sed 's/.*\[bytes\/lane\]:[[:space:]]*\([0-9]*\).*/\1/')
    vsp=$(echo  "$block" | grep -m1 "VGPRs Spill:" | sed 's/.*VGPRs Spill:[[:space:]]*\([0-9]*\).*/\1/')
    ssp=$(echo  "$block" | grep -m1 "SGPRs Spill:" | sed 's/.*SGPRs Spill:[[:space:]]*\([0-9]*\).*/\1/')
    printf "%-24s %-10s %-10s %-12s %-14s %-10s %-10s\n" "$tag" "${vgpr:-?}" "${sgpr:-?}" "${lds:-?}" "${scr:-?}" "${vsp:-?}" "${ssp:-?}"
done
