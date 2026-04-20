#!/bin/bash
# R55 Dev C Phase 0: ISA disassembly of the SQUARE RCR MXFP8 scaled kernel
# at the 70B Down RCR cell (M=4096, N=8192, K=28672).
# Identify scale-broadcast → MFMA dependency chains.
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../../.. && pwd)"

OUTDIR="../r55c_results/isa"
mkdir -p "$OUTDIR"

M=4096; N=8192; K=28672

build_one() {
    local TAG=$1; local FLAGS=$2
    echo "=== gen device .s TAG=$TAG FLAGS=$FLAGS"
    /opt/rocm/bin/hipcc --offload-device-only -S kernel_mxfp8_layouts.cpp \
        -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
        -ffast-math -std=c++20 -w \
        -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype \
        $(python3 -m pybind11 --includes) -I/opt/rocm/include/hip -I/opt/rocm/include/rocrand \
        -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $FLAGS \
        -Rpass-analysis=kernel-resource-usage \
        -o "$OUTDIR/${TAG}_device.s" 2>&1 | tail -50 > "$OUTDIR/${TAG}_remarks.log" || true
}

build_one "baseline" ""

# Extract the SQUARE RCR scaled kernel (NOT rect).
for tag in baseline; do
    s="$OUTDIR/${tag}_device.s"
    out_kernel="$OUTDIR/${tag}_rcr_kernel.s"
    out_kloop="$OUTDIR/${tag}_rcr_kloop.s"
    if [[ -f "$s" ]]; then
        # Find the RCR scaled kernel (not rect, not RRR/CRR).
        awk '/^_Z.*rcr_exact_8wave_scaled_kernelILb1ELi2EEv/{found=1} found{print; if (/^.Lfunc_end/) exit}' "$s" > "$out_kernel"
        # Inner loop body.
        awk '/=>This Inner Loop Header: Depth=1/{found=1} found{print; if (/^.LBB.*:[[:space:]]*$/ && !/Inner Loop/) {if (++blocks > 30) exit}}' "$out_kernel" > "$out_kloop"

        # Counts.
        mfma_count=$(grep -cE "^\s*v_mfma_scale" "$out_kloop" 2>/dev/null || echo 0)
        v_mov_count=$(grep -cE "^\s*v_mov_b32" "$out_kloop" 2>/dev/null || echo 0)
        v_pk_mov_count=$(grep -cE "^\s*v_pk_mov_b32" "$out_kloop" 2>/dev/null || echo 0)
        lshrrev_count=$(grep -cE "^\s*v_lshrrev_b32" "$out_kloop" 2>/dev/null || echo 0)
        ds_read_count=$(grep -cE "^\s*ds_read" "$out_kloop" 2>/dev/null || echo 0)
        buffer_load_count=$(grep -cE "^\s*buffer_load" "$out_kloop" 2>/dev/null || echo 0)
        s_waitcnt_count=$(grep -cE "^\s*s_waitcnt" "$out_kloop" 2>/dev/null || echo 0)
        echo "  $tag: mfma_scale=$mfma_count v_mov=$v_mov_count v_pk_mov=$v_pk_mov_count v_lshrrev=$lshrrev_count ds_read=$ds_read_count buffer_load=$buffer_load_count s_waitcnt=$s_waitcnt_count"

        # Extract a windowed view around each MFMA.
        out_chains="$OUTDIR/${tag}_mfma_chains.s"
        : > "$out_chains"
        # Each MFMA followed by ~5 lines of context (and 5 lines preceding).
        grep -nE "^\s*v_mfma_scale" "$out_kloop" | head -16 | while IFS=: read -r lineno _; do
            start=$((lineno > 5 ? lineno - 5 : 1))
            end=$((lineno + 1))
            echo "===== MFMA #$lineno (context lines $start..$end) =====" >> "$out_chains"
            sed -n "${start},${end}p" "$out_kloop" >> "$out_chains"
            echo "" >> "$out_chains"
        done
    fi
done

# Resource summary.
echo ""
echo "=========== ISA RESOURCE SUMMARY ==========="
printf "%-14s %-10s %-10s %-12s %-14s %-10s %-10s\n" "TAG" "VGPRs" "SGPRs" "LDS_bytes" "Scratch_lane" "VSpill" "SSpill"
for tag in baseline; do
    log="$OUTDIR/${tag}_remarks.log"
    block=$(awk '/rcr_exact_8wave_scaled_kernelILb1ELi2EE.*remark:/,/Occupancy.*waves/' "$log" 2>/dev/null | head -25)
    vgpr=$(echo "$block" | grep -m1 "VGPRs:" | grep -v Spill | sed 's/.*VGPRs:[[:space:]]*\([0-9]*\).*/\1/')
    sgpr=$(echo "$block" | grep -m1 "TotalSGPRs:" | sed 's/.*TotalSGPRs:[[:space:]]*\([0-9]*\).*/\1/')
    lds=$(echo  "$block" | grep -m1 "LDS Size" | sed 's/.*\[bytes\/block\]:[[:space:]]*\([0-9]*\).*/\1/')
    scr=$(echo  "$block" | grep -m1 "ScratchSize" | sed 's/.*\[bytes\/lane\]:[[:space:]]*\([0-9]*\).*/\1/')
    vsp=$(echo  "$block" | grep -m1 "VGPRs Spill:" | sed 's/.*VGPRs Spill:[[:space:]]*\([0-9]*\).*/\1/')
    ssp=$(echo  "$block" | grep -m1 "SGPRs Spill:" | sed 's/.*SGPRs Spill:[[:space:]]*\([0-9]*\).*/\1/')
    printf "%-14s %-10s %-10s %-12s %-14s %-10s %-10s\n" "$tag" "${vgpr:-?}" "${sgpr:-?}" "${lds:-?}" "${scr:-?}" "${vsp:-?}" "${ssp:-?}"
done
