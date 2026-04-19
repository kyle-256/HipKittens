#!/bin/bash
# R54 Dev I Phase 1: ISA-only sanity scan for the branchless unconditional
# shift block lever. Build the 70B Gate/Up CRR shape (M=4096 N=28672 K=8192)
# with gate=0 (baseline) and gate=1 (branchless). Compare resources, MFMA
# per K-loop body iter, and verify the s_bitcmp0_b32/s_cbranch_scc1 conditional
# jump is GONE in the gate=1 build.
set -euo pipefail
cd "$(dirname "$0")/r54i_workspace"
export THUNDERKITTENS_ROOT="$(cd ../../../.. && pwd)"

OUTDIR="../r54i_results"
mkdir -p "$OUTDIR/isa"

# Primary cell: 70B Gate/Up CRR.
M=4096; N=28672; K=8192

build_one() {
    local TAG=$1; local FLAGS=$2
    rm -f tk_mxfp8_layouts*.so
    echo "=== build TAG=$TAG FLAGS=$FLAGS"
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $FLAGS" \
        > "$OUTDIR/isa/${TAG}_build.log" 2>&1
}

# Generate device .s for ISA inspection.
gen_device_s() {
    local TAG=$1; local FLAGS=$2
    echo "=== gen device .s TAG=$TAG"
    /opt/rocm/bin/hipcc --offload-device-only -S kernel_mxfp8_layouts.cpp \
        -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
        -ffast-math -std=c++20 -w \
        -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype \
        $(python3 -m pybind11 --includes) -I/opt/rocm/include/hip -I/opt/rocm/include/rocrand \
        -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $FLAGS \
        -Rpass-analysis=kernel-resource-usage \
        -o "$OUTDIR/isa/${TAG}_device.s" 2>&1 | tail -50 > "$OUTDIR/isa/${TAG}_device_remarks.log"
}

build_one "baseline"   ""
build_one "branchless" "-DMXFP8_CRR_BRANCHLESS_SHIFT=1"

gen_device_s "baseline"   ""
gen_device_s "branchless" "-DMXFP8_CRR_BRANCHLESS_SHIFT=1"

echo ""
echo "=========== ISA RESOURCE SUMMARY (CRR scaled kernel) ==========="
printf "%-14s %-10s %-10s %-12s %-14s %-10s %-10s\n" "TAG" "VGPRs" "SGPRs" "LDS_bytes" "Scratch_lane" "VSpill" "SSpill"
for tag in baseline branchless; do
    log="$OUTDIR/isa/${tag}_build.log"
    # Find the CRR scaled kernel resource usage block.
    block=$(awk '/crr_exact_8wave_scaled_kernel.*remark:/,/Occupancy.*waves/' "$log" | head -20)
    vgpr=$(echo "$block" | grep -m1 "VGPRs:" | grep -v Spill | sed 's/.*VGPRs:[[:space:]]*\([0-9]*\).*/\1/')
    sgpr=$(echo "$block" | grep -m1 "TotalSGPRs:" | sed 's/.*TotalSGPRs:[[:space:]]*\([0-9]*\).*/\1/')
    lds=$(echo  "$block" | grep -m1 "LDS Size" | sed 's/.*\[bytes\/block\]:[[:space:]]*\([0-9]*\).*/\1/')
    scr=$(echo  "$block" | grep -m1 "ScratchSize" | sed 's/.*\[bytes\/lane\]:[[:space:]]*\([0-9]*\).*/\1/')
    vsp=$(echo  "$block" | grep -m1 "VGPRs Spill:" | sed 's/.*VGPRs Spill:[[:space:]]*\([0-9]*\).*/\1/')
    ssp=$(echo  "$block" | grep -m1 "SGPRs Spill:" | sed 's/.*SGPRs Spill:[[:space:]]*\([0-9]*\).*/\1/')
    printf "%-14s %-10s %-10s %-12s %-14s %-10s %-10s\n" "$tag" "${vgpr:-?}" "${sgpr:-?}" "${lds:-?}" "${scr:-?}" "${vsp:-?}" "${ssp:-?}"
done

echo ""
echo "=========== CRR SCALED KERNEL ISA EXTRACT ==========="
for tag in baseline branchless; do
    s="$OUTDIR/isa/${tag}_device.s"
    out_kernel="$OUTDIR/isa/${tag}_crr_kernel.s"
    out_kloop="$OUTDIR/isa/${tag}_crr_kloop.s"
    if [[ -f "$s" ]]; then
        # Extract the CRR exact 8-wave scaled kernel function.
        awk '/^_ZN.*crr_exact_8wave_scaled_kernel/{found=1} found{print; if (/^.Lfunc_end/) exit}' "$s" > "$out_kernel"
        # Extract just the inner K-loop body (depth=1).
        awk '/=>This Inner Loop Header: Depth=1/{found=1} found{print; if (/^.LBB.*:[[:space:]]*$/ && !/Inner Loop/) {if (++blocks > 30) exit}}' "$out_kernel" > "$out_kloop"
        # Counts.
        mfma_count=$(grep -cE "^\s*v_mfma" "$out_kloop" || true)
        lshrrev_count=$(grep -cE "^\s*v_lshrrev_b32" "$out_kloop" || true)
        bitcmp_count=$(grep -cE "^\s*s_bitcmp" "$out_kloop" || true)
        cbranch_count=$(grep -cE "^\s*s_cbranch" "$out_kloop" || true)
        ds_read_count=$(grep -cE "^\s*ds_read" "$out_kloop" || true)
        echo "  $tag: mfma=$mfma_count v_lshrrev_b32=$lshrrev_count s_bitcmp=$bitcmp_count s_cbranch=$cbranch_count ds_read=$ds_read_count"
    fi
done

echo ""
echo "=========== BITCMP/CBRANCH EVIDENCE (gate verification) ==========="
for tag in baseline branchless; do
    s="$OUTDIR/isa/${tag}_crr_kernel.s"
    if [[ -f "$s" ]]; then
        echo "--- $tag: s_bitcmp0_b32 occurrences in CRR kernel ---"
        grep -nE "s_bitcmp0_b32|s_bitcmp1_b32" "$s" | head -20
    fi
done
