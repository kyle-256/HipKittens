#!/bin/bash
# R43 Dev B — full sweep across LLaMA decode M=1 shapes × {RRR, CRR}.
# Builds .so files were prepared by r43b_build.sh. Baselines (FP8 ref +
# MXFP8 V1 legacy) use M_DIM=2 trick to avoid V2 fastpath crash at M=1.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p r43b_runs
GPU=${GPU:-3}
export HIP_VISIBLE_DEVICES=$GPU
export PREHEAT_S=${PREHEAT_S:-30}
export WARMUP=${WARMUP:-30}
export ITERS=${ITERS:-100}
export MXFP8_DISPATCH_TRACE=1

OUT="r43b_runs/sweep_gpu${GPU}.log"
: > "$OUT"

run_one() {
    local label=$1; shift
    echo "--- $label ---" | tee -a "$OUT"
    python3 r43b_decode_m1_rrr_crr_bench.py "$@" 2>&1 | grep -E "R43B_RESULT|\[mxfp8_dispatch\]|Error|error" | tee -a "$OUT"
    echo "" >> "$OUT"
}

# Shape table: (TAG, N, K, MXFP8_FAST_MODULE, MXFP8_V1_MODULE, FP8_MODULE)
# 8B Q/K/V/O 1x4096x4096
run_one "8B_4kx4k_rrr_FAST"  tk_mxfp8_decode_m1rc_8b_4kx4k mxfp8 decode_m1_rrr_crr rrr 1 4096 4096
run_one "8B_4kx4k_crr_FAST"  tk_mxfp8_decode_m1rc_8b_4kx4k mxfp8 decode_m1_rrr_crr crr 1 4096 4096
run_one "8B_4kx4k_rrr_V1"    tk_mxfp8_legacy_8b_4kx4k       mxfp8 legacy_v1pq      rrr 1 4096 4096
run_one "8B_4kx4k_crr_V1"    tk_mxfp8_legacy_8b_4kx4k       mxfp8 legacy_v1pq      crr 1 4096 4096
run_one "8B_4kx4k_rrr_FP8"   tk_fp8_baseline_8b_4kx4k       fp8   fp8_pertensor    rrr 1 4096 4096
run_one "8B_4kx4k_crr_FP8"   tk_fp8_baseline_8b_4kx4k       fp8   fp8_pertensor    crr 1 4096 4096

# 70B Q/K/V/O 1x8192x8192
run_one "70B_8kx8k_rrr_FAST" tk_mxfp8_decode_m1rc_70b_8kx8k mxfp8 decode_m1_rrr_crr rrr 1 8192 8192
run_one "70B_8kx8k_crr_FAST" tk_mxfp8_decode_m1rc_70b_8kx8k mxfp8 decode_m1_rrr_crr crr 1 8192 8192
run_one "70B_8kx8k_rrr_V1"   tk_mxfp8_legacy_70b_8kx8k      mxfp8 legacy_v1pq      rrr 1 8192 8192
run_one "70B_8kx8k_crr_V1"   tk_mxfp8_legacy_70b_8kx8k      mxfp8 legacy_v1pq      crr 1 8192 8192
run_one "70B_8kx8k_rrr_FP8"  tk_fp8_baseline_70b_8kx8k      fp8   fp8_pertensor    rrr 1 8192 8192
run_one "70B_8kx8k_crr_FP8"  tk_fp8_baseline_70b_8kx8k      fp8   fp8_pertensor    crr 1 8192 8192

# 8B SwiGLU gate/up 1x14336x4096 (RRR primarily)
run_one "8B_14kx4k_rrr_FAST"  tk_mxfp8_decode_m1rc_8b_14kx4k mxfp8 decode_m1_rrr_crr rrr 1 14336 4096
run_one "8B_14kx4k_crr_FAST"  tk_mxfp8_decode_m1rc_8b_14kx4k mxfp8 decode_m1_rrr_crr crr 1 14336 4096
run_one "8B_14kx4k_rrr_V1"    tk_mxfp8_legacy_8b_14kx4k      mxfp8 legacy_v1pq      rrr 1 14336 4096
run_one "8B_14kx4k_crr_V1"    tk_mxfp8_legacy_8b_14kx4k      mxfp8 legacy_v1pq      crr 1 14336 4096
run_one "8B_14kx4k_rrr_FP8"   tk_fp8_baseline_8b_14kx4k      fp8   fp8_pertensor    rrr 1 14336 4096
run_one "8B_14kx4k_crr_FP8"   tk_fp8_baseline_8b_14kx4k      fp8   fp8_pertensor    crr 1 14336 4096

# 8B SwiGLU down 1x4096x14336 (RRR)
run_one "8B_4kx14k_rrr_FAST"  tk_mxfp8_decode_m1rc_8b_4kx14k mxfp8 decode_m1_rrr_crr rrr 1 4096 14336
run_one "8B_4kx14k_crr_FAST"  tk_mxfp8_decode_m1rc_8b_4kx14k mxfp8 decode_m1_rrr_crr crr 1 4096 14336
run_one "8B_4kx14k_rrr_V1"    tk_mxfp8_legacy_8b_4kx14k      mxfp8 legacy_v1pq      rrr 1 4096 14336
run_one "8B_4kx14k_crr_V1"    tk_mxfp8_legacy_8b_4kx14k      mxfp8 legacy_v1pq      crr 1 4096 14336
run_one "8B_4kx14k_rrr_FP8"   tk_fp8_baseline_8b_4kx14k      fp8   fp8_pertensor    rrr 1 4096 14336
run_one "8B_4kx14k_crr_FP8"   tk_fp8_baseline_8b_4kx14k      fp8   fp8_pertensor    crr 1 4096 14336

echo "=== sweep complete: $OUT ==="
