#!/bin/bash
# R56 Dev A Phase 1: ISA diff for MXFP8_RRR_SALU_SETPRIO_R56A = {0,1,2}
# at 8B Gate/Up shape (M=4096 N=14336 K=4096). V0 must be byte-identical to
# baseline. V1/V2 should reduce s_setprio count in the steady K-loop body.

set -euo pipefail
cd "$(dirname "$0")/r56a_workspace"
export THUNDERKITTENS_ROOT="$(cd ../../../.. && pwd)"

OUTDIR="../r56a_results/isa"
mkdir -p "$OUTDIR"

M=4096; N=14336; K=4096

gen_one() {
    local TAG=$1; local FLAGS=$2
    echo "=== gen $TAG $FLAGS"
    /opt/rocm/bin/hipcc --offload-device-only -S kernel_mxfp8_layouts.cpp \
        -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
        -ffast-math -std=c++20 -w \
        -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype \
        $(python3 -m pybind11 --includes) -I/opt/rocm/include/hip -I/opt/rocm/include/rocrand \
        -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $FLAGS \
        -Rpass-analysis=kernel-resource-usage \
        -o "$OUTDIR/p1_${TAG}_device.s" 2> "$OUTDIR/p1_${TAG}_device_remarks.log"
}

extract_kloop() {
    local TAG=$1
    local s="$OUTDIR/p1_${TAG}_device.s"
    local out_kernel="$OUTDIR/p1_${TAG}_v2rrr_kernel.s"
    local out_kloop="$OUTDIR/p1_${TAG}_v2rrr_kloop.s"
    awk '/^_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EE/{found=1} found{print; if (/^.Lfunc_end/) exit}' "$s" > "$out_kernel"
    awk '/=>This Inner Loop Header: Depth=1/{found=1} found{print; if (/^.LBB.*:[[:space:]]*$/ && !/Inner Loop/) {if (++blocks > 30) exit}}' "$out_kernel" > "$out_kloop"
}

count_salu() {
    local TAG=$1
    local kloop="$OUTDIR/p1_${TAG}_v2rrr_kloop.s"
    if [[ ! -f "$kloop" ]]; then echo "  (no $TAG)"; return; fi
    local total=$(wc -l < "$kloop")
    local mfma=$(grep -cE "^\s*v_mfma" "$kloop" || true)
    local salu=$(grep -cE "^\s*s_" "$kloop" || true)
    local setprio=$(grep -cE "^\s*s_setprio" "$kloop" || true)
    local barrier=$(grep -cE "^\s*s_barrier" "$kloop" || true)
    local waitcnt=$(grep -cE "^\s*s_waitcnt" "$kloop" || true)
    local mov_b32=$(grep -cE "^\s*s_mov_b32" "$kloop" || true)
    local s_add=$(grep -cE "^\s*s_add(_u32|_i32|_co_u32)?\s" "$kloop" || true)
    local s_addc=$(grep -cE "^\s*s_addc_u32" "$kloop" || true)
    local s_addk=$(grep -cE "^\s*s_addk_i32" "$kloop" || true)
    local s_nop=$(grep -cE "^\s*s_nop" "$kloop" || true)
    printf "  %-12s lines=%-5d mfma=%-3d SALU_total=%-4d setprio=%-3d barrier=%-3d waitcnt=%-3d mov32=%-3d add=%-3d addc=%-3d addk=%-3d nop=%-3d\n" \
        "$TAG" "$total" "$mfma" "$salu" "$setprio" "$barrier" "$waitcnt" "$mov_b32" "$s_add" "$s_addc" "$s_addk" "$s_nop"
}

resource_summary() {
    local TAG=$1
    local log="$OUTDIR/p1_${TAG}_device_remarks.log"
    if [[ ! -f "$log" ]]; then return; fi
    local block=$(awk '/rrr_exact_8wave_scaled_kernel.*remark:/,/Occupancy.*waves/' "$log" | head -25)
    local vgpr=$(echo "$block" | grep -m1 "VGPRs:" | grep -v Spill | sed 's/.*VGPRs:[[:space:]]*\([0-9]*\).*/\1/')
    local sgpr=$(echo "$block" | grep -m1 "TotalSGPRs:" | sed 's/.*TotalSGPRs:[[:space:]]*\([0-9]*\).*/\1/')
    local lds=$(echo  "$block" | grep -m1 "LDS Size" | sed 's/.*\[bytes\/block\]:[[:space:]]*\([0-9]*\).*/\1/')
    local vsp=$(echo  "$block" | grep -m1 "VGPRs Spill:" | sed 's/.*VGPRs Spill:[[:space:]]*\([0-9]*\).*/\1/')
    local ssp=$(echo  "$block" | grep -m1 "SGPRs Spill:" | sed 's/.*SGPRs Spill:[[:space:]]*\([0-9]*\).*/\1/')
    local occ=$(echo  "$block" | grep -m1 "Occupancy" | sed 's/.*Occupancy:[[:space:]]*\([0-9]*\).*/\1/')
    printf "  %-12s VGPR=%-4s VSpill=%-3s SGPR=%-4s SSpill=%-3s LDS=%-7s Occ=%s\n" \
        "$TAG" "${vgpr:-?}" "${vsp:-?}" "${sgpr:-?}" "${ssp:-?}" "${lds:-?}" "${occ:-?}"
}

gen_one "v0" ""
extract_kloop "v0"
gen_one "v1" "-DMXFP8_RRR_SALU_SETPRIO_R56A=1"
extract_kloop "v1"
gen_one "v2" "-DMXFP8_RRR_SALU_SETPRIO_R56A=2"
extract_kloop "v2"

echo ""
echo "=========== K-LOOP SALU COUNTS ==========="
count_salu "v0"
count_salu "v1"
count_salu "v2"

echo ""
echo "=========== RESOURCE SUMMARY ==========="
resource_summary "v0"
resource_summary "v1"
resource_summary "v2"

echo ""
echo "=========== diff v0 vs v1 (kloop) [first 30 diff hunks] ==========="
diff "$OUTDIR/p1_v0_v2rrr_kloop.s" "$OUTDIR/p1_v1_v2rrr_kloop.s" | head -60 || true
echo ""
echo "=========== diff v0 vs v2 (kloop) [first 30 diff hunks] ==========="
diff "$OUTDIR/p1_v0_v2rrr_kloop.s" "$OUTDIR/p1_v2_v2rrr_kloop.s" | head -60 || true
echo ""
echo "=========== diff v0 vs base (R56A vs PRE-EDIT baseline) ==========="
# This catches any unintended drift from injecting macros at v0 (must be empty).
diff "$OUTDIR/8B_GateUp_v2rrr_kloop.s" "$OUTDIR/p1_v0_v2rrr_kloop.s" | head -60 || true
