#!/bin/bash
# R55 Dev B Phase 0: ISA inspection of the RCR exact 8-wave scaled kernel
# (V2-RCR path, used for 70B Down M=4096 N=8192 K=28672). Identify scale-base
# address SALU instructions inside the K-loop body — confirm whether the
# compiler has already hoisted them or whether they re-execute every iteration.
set -euo pipefail
cd "$(dirname "$0")/r55b_workspace"
export THUNDERKITTENS_ROOT="$(cd ../../../.. && pwd)"

OUTDIR="../r55b_results/isa"
mkdir -p "$OUTDIR"

# 70B Down RCR cell.
M=4096; N=8192; K=28672

build_one() {
    local TAG=$1; local FLAGS=$2
    rm -f tk_mxfp8_layouts*.so
    echo "=== build TAG=$TAG FLAGS=$FLAGS"
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $FLAGS" \
        > "$OUTDIR/${TAG}_build.log" 2>&1
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
        -o "$OUTDIR/${TAG}_device.s" 2>&1 | tail -50 > "$OUTDIR/${TAG}_device_remarks.log"
}

build_one "baseline" ""
gen_device_s "baseline" ""

echo ""
echo "=========== ISA RESOURCE SUMMARY (RCR scaled kernel) ==========="
printf "%-14s %-10s %-10s %-12s %-14s %-10s %-10s\n" "TAG" "VGPRs" "SGPRs" "LDS_bytes" "Scratch_lane" "VSpill" "SSpill"
for tag in baseline; do
    log="$OUTDIR/${tag}_build.log"
    block=$(awk '/rcr_exact_8wave_scaled_kernel.*remark:/,/Occupancy.*waves/' "$log" | head -20)
    vgpr=$(echo "$block" | grep -m1 "VGPRs:" | grep -v Spill | sed 's/.*VGPRs:[[:space:]]*\([0-9]*\).*/\1/')
    sgpr=$(echo "$block" | grep -m1 "TotalSGPRs:" | sed 's/.*TotalSGPRs:[[:space:]]*\([0-9]*\).*/\1/')
    lds=$(echo  "$block" | grep -m1 "LDS Size" | sed 's/.*\[bytes\/block\]:[[:space:]]*\([0-9]*\).*/\1/')
    scr=$(echo  "$block" | grep -m1 "ScratchSize" | sed 's/.*\[bytes\/lane\]:[[:space:]]*\([0-9]*\).*/\1/')
    vsp=$(echo  "$block" | grep -m1 "VGPRs Spill:" | sed 's/.*VGPRs Spill:[[:space:]]*\([0-9]*\).*/\1/')
    ssp=$(echo  "$block" | grep -m1 "SGPRs Spill:" | sed 's/.*SGPRs Spill:[[:space:]]*\([0-9]*\).*/\1/')
    printf "%-14s %-10s %-10s %-12s %-14s %-10s %-10s\n" "$tag" "${vgpr:-?}" "${sgpr:-?}" "${lds:-?}" "${scr:-?}" "${vsp:-?}" "${ssp:-?}"
done

echo ""
echo "=========== RCR SCALED KERNEL ISA EXTRACT ==========="
for tag in baseline; do
    s="$OUTDIR/${tag}_device.s"
    out_kernel="$OUTDIR/${tag}_rcr_kernel.s"
    out_kloop="$OUTDIR/${tag}_rcr_kloop.s"
    if [[ -f "$s" ]]; then
        # The RCR scaled kernel: rcr_exact_8wave_scaled_kernel<true,2>.
        awk '/^_ZN.*rcr_exact_8wave_scaled_kernel/{found=1} found{print; if (/^.Lfunc_end/) exit}' "$s" > "$out_kernel"
        awk '/=>This Inner Loop Header: Depth=1/{found=1} found{print; if (/^.LBB.*:[[:space:]]*$/ && !/Inner Loop/) {if (++blocks > 30) exit}}' "$out_kernel" > "$out_kloop"
        mfma_count=$(grep -cE "^\s*v_mfma" "$out_kloop" || true)
        salu_add=$(grep -cE "^\s*s_add(_u32|_i32|_co_u32)?\s" "$out_kloop" || true)
        salu_addc=$(grep -cE "^\s*s_addc_u32" "$out_kloop" || true)
        salu_lshl=$(grep -cE "^\s*s_lshl" "$out_kloop" || true)
        salu_mul=$(grep -cE "^\s*s_mul" "$out_kloop" || true)
        s_load=$(grep -cE "^\s*s_load_dword" "$out_kloop" || true)
        ds_read=$(grep -cE "^\s*ds_read" "$out_kloop" || true)
        v_lshrrev=$(grep -cE "^\s*v_lshrrev_b32" "$out_kloop" || true)
        echo "  $tag (kloop): mfma=$mfma_count s_add=$salu_add s_addc=$salu_addc s_lshl=$salu_lshl s_mul=$salu_mul s_load_dword=$s_load ds_read=$ds_read v_lshrrev=$v_lshrrev"
        wc_kernel=$(wc -l < "$out_kernel")
        wc_kloop=$(wc -l < "$out_kloop")
        echo "  $tag: kernel=$wc_kernel lines, kloop=$wc_kloop lines"
    fi
done

echo ""
echo "=========== ALL SALU IN KLOOP (baseline) ==========="
grep -nE "^\s*s_(add|addc|sub|lshl|mul|and|or|xor|bfe|bfm|nor|cmp|cmpk)" "$OUTDIR/baseline_rcr_kloop.s" | head -100 > "$OUTDIR/baseline_kloop_salu.txt"
wc -l "$OUTDIR/baseline_kloop_salu.txt"
head -80 "$OUTDIR/baseline_kloop_salu.txt"
