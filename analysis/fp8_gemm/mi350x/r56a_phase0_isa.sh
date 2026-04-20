#!/bin/bash
# R56 Dev A Phase 0: ISA inspection of the V2 RRR exact 8-wave scaled kernel
# at TWO shapes: 8B Gate/Up (M=4096 N=14336 K=4096) — primary HEADROOM cell
# with R55F's identified SALU-DOMINANT signature (+50.6% SALU vs FP8) — and
# 70B Q/O (M=4096 N=8192 K=8192) — control where R55F PMC showed SALU
# expansion drops to +5.4% (per-tile fixed cost amortized away).
#
# Goal: categorize ALL SALU instructions in the steady-state K-loop body of
# `_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EE` for both shapes, identify the
# dominant SALU subgroup in the 8B GU body, and quantify the per-iter delta
# vs 70B Q/O so any R56A lever targets the actual dominator.

set -euo pipefail
cd "$(dirname "$0")/r56a_workspace"
export THUNDERKITTENS_ROOT="$(cd ../../../.. && pwd)"

OUTDIR="../r56a_results/isa"
mkdir -p "$OUTDIR"

gen_device_s() {
    local TAG=$1
    local M=$2; local N=$3; local K=$4
    echo "=== gen device .s TAG=$TAG  M=$M N=$N K=$K"
    /opt/rocm/bin/hipcc --offload-device-only -S kernel_mxfp8_layouts.cpp \
        -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
        -ffast-math -std=c++20 -w \
        -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype \
        $(python3 -m pybind11 --includes) -I/opt/rocm/include/hip -I/opt/rocm/include/rocrand \
        -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K \
        -Rpass-analysis=kernel-resource-usage \
        -o "$OUTDIR/${TAG}_device.s" 2> "$OUTDIR/${TAG}_device_remarks.log"
}

extract_v2_rrr() {
    local TAG=$1
    local s="$OUTDIR/${TAG}_device.s"
    local out_kernel="$OUTDIR/${TAG}_v2rrr_kernel.s"
    local out_kloop="$OUTDIR/${TAG}_v2rrr_kloop.s"
    if [[ -f "$s" ]]; then
        # Extract the rrr_exact_8wave_scaled_kernel<true,2> symbol body.
        awk '/^_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EE/{found=1} found{print; if (/^.Lfunc_end/) exit}' "$s" > "$out_kernel"
        # Steady-state K-loop body: walk from the deepest loop header to the
        # first basic block that closes the loop. The K-loop body is the
        # largest single-block stretch starting at the inner-loop header.
        awk '/=>This Inner Loop Header: Depth=1/{found=1} found{print; if (/^.LBB.*:[[:space:]]*$/ && !/Inner Loop/) {if (++blocks > 30) exit}}' "$out_kernel" > "$out_kloop"
        local mfma=$(grep -cE "^\s*v_mfma" "$out_kloop" || true)
        local total_lines=$(wc -l < "$out_kloop")
        local kernel_lines=$(wc -l < "$out_kernel")
        echo "  $TAG kernel=$kernel_lines lines kloop=$total_lines lines mfma=$mfma"
    fi
}

categorize_salu() {
    local TAG=$1
    local kloop="$OUTDIR/${TAG}_v2rrr_kloop.s"
    local out="$OUTDIR/${TAG}_salu_breakdown.txt"
    if [[ ! -f "$kloop" ]]; then return; fi
    {
        echo "=== SALU instruction breakdown for $TAG (V2 RRR steady K-loop body) ==="
        echo ""
        echo "--- raw counts of SALU op-mnemonic prefix (top 30) ---"
        grep -oE "^\s*s_[a-z0-9_]+" "$kloop" | sort | uniq -c | sort -rn | head -30
        echo ""
        echo "--- specific SALU groups ---"
        printf "  %-32s %d\n" "s_add_*  (s_add_u32/i32/co_u32)"  $(grep -cE "^\s*s_add(_u32|_i32|_co_u32)?\s" "$kloop" || true)
        printf "  %-32s %d\n" "s_addc_u32"                       $(grep -cE "^\s*s_addc_u32" "$kloop" || true)
        printf "  %-32s %d\n" "s_addk_i32"                       $(grep -cE "^\s*s_addk_i32" "$kloop" || true)
        printf "  %-32s %d\n" "s_sub_*"                          $(grep -cE "^\s*s_sub" "$kloop" || true)
        printf "  %-32s %d\n" "s_lshl_*"                         $(grep -cE "^\s*s_lshl" "$kloop" || true)
        printf "  %-32s %d\n" "s_lshr_*"                         $(grep -cE "^\s*s_lshr" "$kloop" || true)
        printf "  %-32s %d\n" "s_mul_*"                          $(grep -cE "^\s*s_mul" "$kloop" || true)
        printf "  %-32s %d\n" "s_mov_b32"                        $(grep -cE "^\s*s_mov_b32" "$kloop" || true)
        printf "  %-32s %d\n" "s_mov_b64"                        $(grep -cE "^\s*s_mov_b64" "$kloop" || true)
        printf "  %-32s %d\n" "s_load_dword_*"                   $(grep -cE "^\s*s_load_dword" "$kloop" || true)
        printf "  %-32s %d\n" "s_load_b*"                        $(grep -cE "^\s*s_load_b" "$kloop" || true)
        printf "  %-32s %d\n" "s_and_*"                          $(grep -cE "^\s*s_and" "$kloop" || true)
        printf "  %-32s %d\n" "s_or_*"                           $(grep -cE "^\s*s_or" "$kloop" || true)
        printf "  %-32s %d\n" "s_xor_*"                          $(grep -cE "^\s*s_xor" "$kloop" || true)
        printf "  %-32s %d\n" "s_bfe_*"                          $(grep -cE "^\s*s_bfe" "$kloop" || true)
        printf "  %-32s %d\n" "s_bfm_*"                          $(grep -cE "^\s*s_bfm" "$kloop" || true)
        printf "  %-32s %d\n" "s_min/s_max"                      $(grep -cE "^\s*s_(min|max)" "$kloop" || true)
        printf "  %-32s %d\n" "s_cmp/s_cmpk"                     $(grep -cE "^\s*s_cmp" "$kloop" || true)
        printf "  %-32s %d\n" "s_cbranch/s_branch"               $(grep -cE "^\s*s_(cbranch|branch)" "$kloop" || true)
        printf "  %-32s %d\n" "s_setprio"                        $(grep -cE "^\s*s_setprio" "$kloop" || true)
        printf "  %-32s %d\n" "s_barrier"                        $(grep -cE "^\s*s_barrier" "$kloop" || true)
        printf "  %-32s %d\n" "s_waitcnt"                        $(grep -cE "^\s*s_waitcnt" "$kloop" || true)
        printf "  %-32s %d\n" "s_setpc/s_swappc"                 $(grep -cE "^\s*s_(setpc|swappc)" "$kloop" || true)
        printf "  %-32s %d\n" "TOTAL SALU (s_*)"                 $(grep -cE "^\s*s_" "$kloop" || true)
        printf "  %-32s %d\n" "v_mfma"                           $(grep -cE "^\s*v_mfma" "$kloop" || true)
        printf "  %-32s %d\n" "v_lshrrev_b32 (CRR scale shift)"  $(grep -cE "^\s*v_lshrrev_b32" "$kloop" || true)
        printf "  %-32s %d\n" "ds_read_*"                        $(grep -cE "^\s*ds_read" "$kloop" || true)
        printf "  %-32s %d\n" "ds_write_*"                       $(grep -cE "^\s*ds_write" "$kloop" || true)
        printf "  %-32s %d\n" "buffer_load_*"                    $(grep -cE "^\s*buffer_load" "$kloop" || true)
    } > "$out"
}

# Phase 0: capture both shapes.
gen_device_s "8B_GateUp"  4096 14336 4096
extract_v2_rrr "8B_GateUp"
categorize_salu "8B_GateUp"

gen_device_s "70B_QO"     4096  8192 8192
extract_v2_rrr "70B_QO"
categorize_salu "70B_QO"

# Pretty side-by-side table.
{
    echo "=== V2 RRR steady-K-loop SALU comparison (8B Gate/Up vs 70B Q/O) ==="
    echo ""
    paste <(cat "$OUTDIR/8B_GateUp_salu_breakdown.txt" | head -45) <(cat "$OUTDIR/70B_QO_salu_breakdown.txt" | head -45)
} > "$OUTDIR/salu_compare.txt"
echo ""
echo "=== Top output: $OUTDIR/salu_compare.txt ==="
cat "$OUTDIR/8B_GateUp_salu_breakdown.txt"
echo ""
echo "============= 70B Q/O ============="
cat "$OUTDIR/70B_QO_salu_breakdown.txt"
