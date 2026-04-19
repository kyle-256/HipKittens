#!/usr/bin/env bash
# R54 Dev C Phase 1: ISA-only sanity scan of allocator-side flags on V2 RRR.
# Target shape: 4096 14336 4096 (8B Gate/Up — primary RRR target).
# Build only; capture VGPR / scratch / spill / occupancy on V2 RRR kernel.
# Drop any flag that pushes V2 RRR past 254 VGPR or introduces scratch.
set -uo pipefail
cd "$(dirname "$0")"

OUTDIR="r54c_results/phase1"
ISADIR="r54c_isa"
mkdir -p "$OUTDIR" "$ISADIR"

M=4096; N=14336; K=4096
LABEL="8B_GateUp"

# Flag combinations to try (each isolated). Each entry: NAME|FLAGS
declare -a FLAGS=(
    "baseline|"
    "sched_max_occupancy|-mllvm -amdgpu-sched-strategy=max-occupancy"
    "sched_ilp|-mllvm -amdgpu-sched-strategy=max-ilp"
    "sched_iterative_ilp|-mllvm -amdgpu-sched-strategy=iterative-ilp"
    "sched_iterative_minreg|-mllvm -amdgpu-sched-strategy=iterative-minreg"
    "sched_iterative_max_occ|-mllvm -amdgpu-sched-strategy=iterative-maximum-occupancy"
    "power_sched_off|-mllvm -amdgpu-disable-power-sched"
    "igrouplp_off|-mllvm -amdgpu-igrouplp=0"
    "promote_alloca_16|-mllvm -amdgpu-promote-alloca-to-vector-limit=16"
    "promote_alloca_32|-mllvm -amdgpu-promote-alloca-to-vector-limit=32"
    "promote_alloca_64|-mllvm -amdgpu-promote-alloca-to-vector-limit=64"
    "vgpr_index_mode_dis|-mllvm -amdgpu-vgpr-index-mode=0"
    "loop_align_off|-mllvm -amdgpu-disable-loop-alignment"
    "num_vgpr_240|-mllvm -amdgpu-num-vgpr=240"
    "num_vgpr_224|-mllvm -amdgpu-num-vgpr=224"
    "num_vgpr_256|-mllvm -amdgpu-num-vgpr=256"
)

# Optional list of additional curated flags (kept in case GPU/LLVM accepts them)
declare -a EXTRA_FLAGS=(
    "early_ifcvt_off|-mllvm -amdgpu-early-ifcvt=0"
    "function_calls_off|-mllvm -amdgpu-function-calls=0"
    "atomic_optimizer_off|-mllvm -amdgpu-atomic-optimizer-strategy=None"
    "max_uniform_chain|-mllvm -amdgpu-max-uniform-kernel-args=0"
    "regalloc_basic|-mllvm -regalloc=basic"
    "regalloc_greedy|-mllvm -regalloc=greedy"
    "regalloc_fast|-mllvm -regalloc=fast"
    "regalloc_pbqp|-mllvm -regalloc=pbqp"
)

build_one() {
    local NAME=$1
    local FLAGSTR=$2
    local LOG="$OUTDIR/build_${NAME}.log"
    rm -f tk_mxfp8_layouts*.so
    echo "=== Building $NAME (flags: ${FLAGSTR}) ==="
    set +e
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K ${FLAGSTR}" \
        > "$LOG" 2>&1
    local rc=$?
    set -e
    if [ $rc -ne 0 ]; then
        echo "  BUILD FAILED rc=$rc — check $LOG"
    fi
    return $rc
}

extract_v2rrr_metrics() {
    # Extracts VGPRs/scratch/spill/occupancy for the V2 RRR kernel from a build log.
    # The V2 RRR kernel template is at line 262 of rrr_mxfp8_exact_8wave_fastpath.inc
    # but in remarks it shows up as "rrr_mxfp8_exact_8wave_fastpath.inc:223:1" historically,
    # plus we want the SCALE_VERSION=2 instantiation.
    # We grab all "rrr_mxfp8_exact_8wave_fastpath.inc" remarks (both V1 and V2 instantiations
    # appear); for our purposes both should have similar resources, so we just take any.
    local LOG=$1
    if [ ! -f "$LOG" ]; then echo "no_log,no_log,no_log,no_log,no_log"; return; fi
    local VGPR=$(grep "rrr_mxfp8_exact_8wave_fastpath.inc.*VGPRs:" "$LOG" | head -1 | grep -oE 'VGPRs: [0-9]+' | grep -oE '[0-9]+')
    local SPILL=$(grep "rrr_mxfp8_exact_8wave_fastpath.inc.*VGPRs Spill:" "$LOG" | head -1 | grep -oE 'Spill: [0-9]+' | grep -oE '[0-9]+')
    local SCRATCH=$(grep "rrr_mxfp8_exact_8wave_fastpath.inc.*ScratchSize" "$LOG" | head -1 | grep -oE 'lane\]: [0-9]+' | grep -oE '[0-9]+')
    local OCC=$(grep "rrr_mxfp8_exact_8wave_fastpath.inc.*Occupancy" "$LOG" | head -1 | grep -oE 'SIMD\]: [0-9]+' | grep -oE '[0-9]+')
    local SGPR_SPILL=$(grep "rrr_mxfp8_exact_8wave_fastpath.inc.*SGPRs Spill:" "$LOG" | head -1 | grep -oE 'Spill: [0-9]+' | grep -oE '[0-9]+')
    echo "${VGPR:-NA},${SPILL:-NA},${SCRATCH:-NA},${OCC:-NA},${SGPR_SPILL:-NA}"
}

dump_isa() {
    local NAME=$1
    local SO=$(ls tk_mxfp8_layouts*.so 2>/dev/null | head -1)
    if [ -z "$SO" ]; then return; fi
    local OUT="$ISADIR/${NAME}.s"
    /opt/rocm/lib/llvm/bin/llvm-objdump -d "$SO" 2>/dev/null \
        | sed -n '/_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EE/,/^$/p' \
        | head -200 > "$OUT" 2>&1 || true
}

PHASE_INPUT=( "${FLAGS[@]}" )
if [ "${WITH_EXTRA:-0}" = "1" ]; then
    PHASE_INPUT+=( "${EXTRA_FLAGS[@]}" )
fi

SUMMARY="$OUTDIR/SUMMARY.csv"
echo "name,flags,build_ok,vgpr,spill_v,scratch,occupancy,spill_s" > "$SUMMARY"

for entry in "${PHASE_INPUT[@]}"; do
    IFS='|' read -r NAME FLAGSTR <<< "$entry"
    if build_one "$NAME" "$FLAGSTR"; then
        BUILD_OK=1
        METRICS=$(extract_v2rrr_metrics "$OUTDIR/build_${NAME}.log")
        dump_isa "$NAME"
    else
        BUILD_OK=0
        METRICS="NA,NA,NA,NA,NA"
    fi
    echo "${NAME},\"${FLAGSTR}\",${BUILD_OK},${METRICS}" >> "$SUMMARY"
    echo "  [${NAME}] -> ${METRICS}"
done

echo ""
echo "=== Phase 1 ISA scan complete ==="
column -t -s, "$SUMMARY"
