#!/usr/bin/env bash
# R54 Dev C Phase 1B: Refined ISA-only sanity scan with VALID flags only.
# Drop flags that don't exist in /opt/rocm/lib/llvm. Add some additional ones found.
set -uo pipefail
cd "$(dirname "$0")"

OUTDIR="r54c_results/phase1b"
ISADIR="r54c_isa_phase1b"
mkdir -p "$OUTDIR" "$ISADIR"

M=4096; N=14336; K=4096
LABEL="8B_GateUp"

# All flags below are confirmed valid via `llc --help-hidden`.
declare -a FLAGS=(
    "baseline|"
    # Scheduler strategies (validated set)
    "sched_max_occ|-mllvm -amdgpu-sched-strategy=max-occupancy"
    "sched_max_ilp|-mllvm -amdgpu-sched-strategy=max-ilp"
    "sched_iter_ilp|-mllvm -amdgpu-sched-strategy=iterative-ilp"
    "sched_iter_max_occ|-mllvm -amdgpu-sched-strategy=iterative-maximum-occupancy"
    # Scheduling tuning knobs
    "sched_metric_bias_50|-mllvm -amdgpu-schedule-metric-bias=50"
    "sched_metric_bias_100|-mllvm -amdgpu-schedule-metric-bias=100"
    "sched_metric_bias_75|-mllvm -amdgpu-schedule-metric-bias=75"
    "sched_relaxed_occ|-mllvm -amdgpu-schedule-relaxed-occupancy"
    "amdgpu_trackers|-mllvm -amdgpu-use-amdgpu-trackers"
    "disable_clustered_low_occ|-mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"
    "disable_unclustered_high_rp|-mllvm -amdgpu-disable-unclustered-high-rp-reschedule"
    # Promote alloca
    "promote_alloca_16|-mllvm -amdgpu-promote-alloca-to-vector-limit=16"
    "promote_alloca_32|-mllvm -amdgpu-promote-alloca-to-vector-limit=32"
    "promote_alloca_64|-mllvm -amdgpu-promote-alloca-to-vector-limit=64"
    # Loop / alignment
    "loop_align_off|-mllvm -amdgpu-disable-loop-alignment"
    # VGPR liverange + opt-vgpr-liverange
    "opt_vgpr_liverange|-mllvm -amdgpu-opt-vgpr-liverange"
    # Loop prefetch
    "loop_prefetch|-mllvm -amdgpu-loop-prefetch"
    # Wave priority
    "wave_priority|-mllvm -amdgpu-set-wave-priority"
    # VOPD dual issue (gfx11+) — try
    "vopd|-mllvm -amdgpu-enable-vopd"
    # Merge M0
    "merge_m0|-mllvm -amdgpu-enable-merge-m0"
    # Post-RA sched options
    "post_ra_disable|-mllvm -disable-post-ra"
    "misched_postra|-mllvm -misched-postra"
    # Regalloc options
    "regalloc_basic|-mllvm -regalloc=basic"
    "regalloc_greedy|-mllvm -regalloc=greedy"
    "regalloc_pbqp|-mllvm -regalloc=pbqp"
    "vgpr_regalloc_basic|-mllvm -vgpr-regalloc=basic"
    "vgpr_regalloc_greedy|-mllvm -vgpr-regalloc=greedy"
    # Membound / limit-wave thresholds (often coupled with relaxed-occupancy)
    "membound_50|-mllvm -amdgpu-membound-threshold=50"
    "limit_wave_30|-mllvm -amdgpu-limit-wave-threshold=30"
    # Combos worth testing pre-bench
    "max_occ_plus_trackers|-mllvm -amdgpu-sched-strategy=max-occupancy -mllvm -amdgpu-use-amdgpu-trackers"
    "max_occ_plus_metric_50|-mllvm -amdgpu-sched-strategy=max-occupancy -mllvm -amdgpu-schedule-metric-bias=50"
)

build_one() {
    local NAME=$1
    local FLAGSTR=$2
    local LOG="$OUTDIR/build_${NAME}.log"
    rm -f tk_mxfp8_layouts*.so
    set +e
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K ${FLAGSTR}" \
        > "$LOG" 2>&1
    local rc=$?
    set -e
    return $rc
}

extract_v2rrr_metrics() {
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
        | sed -n '/_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EE/,/^[[:space:]]*$/p' \
        | head -300 > "$OUT" 2>&1 || true
    # Save full object size of the V2 RRR symbol
}

SUMMARY="$OUTDIR/SUMMARY.csv"
echo "name,flags,build_ok,vgpr,spill_v,scratch,occupancy,spill_s" > "$SUMMARY"

for entry in "${FLAGS[@]}"; do
    IFS='|' read -r NAME FLAGSTR <<< "$entry"
    echo "=== $NAME ==="
    if build_one "$NAME" "$FLAGSTR"; then
        BUILD_OK=1
        METRICS=$(extract_v2rrr_metrics "$OUTDIR/build_${NAME}.log")
        dump_isa "$NAME"
    else
        BUILD_OK=0
        METRICS="NA,NA,NA,NA,NA"
        # Extract reason from log
        ERR=$(grep -E "Unknown command|error:" "$OUTDIR/build_${NAME}.log" | head -1 | tr -d ',')
        echo "  FAILED: $ERR"
    fi
    echo "${NAME},\"${FLAGSTR}\",${BUILD_OK},${METRICS}" >> "$SUMMARY"
    echo "  -> ${METRICS}"
done

echo ""
echo "=== Phase 1B ISA scan SUMMARY ==="
cat "$SUMMARY" | awk -F, '{printf "%-32s %-7s %-6s %-7s %-9s %-5s %-7s\n", $1, $3, $4, $5, $6, $7, $8}'
