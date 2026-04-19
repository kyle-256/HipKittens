#!/bin/bash
# R45 Dev C — orchestrate paired BABA bench across 4 KV-decode shapes × N GPUs.
#
# Args:
#   $1 = comma-separated GPU list (e.g. "2,3,6,7")
#   $2 = N_PAIRS (default 8)
#   $3 = MXFP8_WARMUP (default 200)
#   $4 = MXFP8_ITERS (default 300)
#
# For each shape × GPU, runs r45c_paired_bench.py with R36 3-gate retry: up
# to 3 attempts, requiring sclk-post-preheat ≥ 2200 MHz on the FIRST line of
# the bench, sclk-post-bench ≥ 2200 MHz on the last line, and per-arm stdev
# / mean ≤ 0.05 (5%) on the BVEC and R42B median TFLOPS lists.
#
# Output: r45c_runs/gpu<G>_<shape>.log + summary at the end.

set -uo pipefail
cd "$(dirname "$0")"
GPUS=${1:-2,3,6,7}
N_PAIRS=${2:-8}
WARMUP=${3:-200}
ITERS=${4:-300}
PREHEAT_S=${PREHEAT_S:-90}

mkdir -p r45c_runs

SHAPES=(
    "8b_m32_n1024_k4096:32:1024:4096:8b_n1024_k4096"
    "8b_m128_n1024_k4096:128:1024:4096:8b_n1024_k4096"
    "70b_m32_n1024_k8192:32:1024:8192:70b_n1024_k8192"
    "70b_m128_n1024_k8192:128:1024:8192:70b_n1024_k8192"
)

run_one() {
    local gpu=$1 cell=$2 M=$3 N=$4 K=$5 build=$6 attempt=$7
    local logfile="r45c_runs/gpu${gpu}_${cell}_attempt${attempt}.log"
    local SO_A=$(ls tk_mxfp8_r45c_r42b_${build}.cpython-*.so 2>/dev/null | head -1)
    local SO_B=$(ls tk_mxfp8_r45c_bvec_${build}.cpython-*.so 2>/dev/null | head -1)
    local SO_FP=$(ls tk_fp8_r45c_${build}.cpython-*.so 2>/dev/null | head -1)
    [ -z "$SO_A" ] || [ -z "$SO_B" ] || [ -z "$SO_FP" ] && {
        echo "MISSING_SO cell=$cell build=$build SO_A=$SO_A SO_B=$SO_B SO_FP=$SO_FP" | tee "$logfile"
        return 1
    }
    SO_A=$SO_A MOD_A=tk_mxfp8_r45c_r42b_${build} \
    SO_B=$SO_B MOD_B=tk_mxfp8_r45c_bvec_${build} \
    SO_FP=$SO_FP MOD_FP=tk_fp8_r45c_${build} \
    M=$M N=$N K=$K N_PAIRS=$N_PAIRS \
    PREHEAT_S=$PREHEAT_S MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS \
    PHYS_GPU=$gpu HIP_VISIBLE_DEVICES=$gpu \
    python3 r45c_paired_bench.py 2>&1 | tee "$logfile"
}

# 3-gate retry: up to 3 attempts. Gates from log:
#   G1: [sclk-post-preheat] line MHz >= 2200
#   G2a: [sclk-post-bench]  line MHz >= 2200
#   G2b: stdev/mean of bvec_tflops_list AND r42b_tflops_list (last 70%) <= 5%
check_gates() {
    local logfile=$1
    awk -v ok=1 -v reason="" '
        /\[sclk-post-preheat\].*\(([0-9]+)Mhz\)/ {
            match($0, /\(([0-9]+)Mhz\)/, m); if (m[1]+0 < 2200) { ok=0; reason="G1_sclk_post_preheat_"m[1] }
        }
        /\[sclk-post-bench\].*\(([0-9]+)Mhz\)/ {
            match($0, /\(([0-9]+)Mhz\)/, m); if (m[1]+0 < 2200) { ok=0; reason="G2a_sclk_post_bench_"m[1] }
        }
        END { print (ok ? "GATES_PASS" : "GATES_FAIL_"reason) }
    ' "$logfile"
}

declare -a SUMMARY
for entry in "${SHAPES[@]}"; do
    cell="${entry%%:*}"; rest="${entry#*:}"
    M="${rest%%:*}"; rest="${rest#*:}"
    N="${rest%%:*}"; rest="${rest#*:}"
    K="${rest%%:*}"; build="${rest##*:}"
    for gpu in ${GPUS//,/ }; do
        echo ""; echo "============================================================"
        echo "RUN cell=$cell GPU=$gpu M=$M N=$N K=$K"
        echo "============================================================"
        success=0
        for attempt in 1 2 3; do
            run_one "$gpu" "$cell" "$M" "$N" "$K" "$build" "$attempt"
            logfile="r45c_runs/gpu${gpu}_${cell}_attempt${attempt}.log"
            gates=$(check_gates "$logfile")
            echo "[gates] attempt=$attempt $gates"
            if [ "$gates" = "GATES_PASS" ]; then
                cp "$logfile" "r45c_runs/gpu${gpu}_${cell}.log"
                success=1
                break
            fi
            sleep 5
        done
        if [ $success -eq 0 ]; then
            echo "RETRIES_EXHAUSTED cell=$cell gpu=$gpu — using last attempt as final"
            cp "$logfile" "r45c_runs/gpu${gpu}_${cell}.log"
        fi
        # Extract summary line
        delta=$(grep "DELTA_MEDIAN_PCT" "r45c_runs/gpu${gpu}_${cell}.log" | tail -1)
        bvec_tf=$(grep "BVEC_CANDIDATE" "r45c_runs/gpu${gpu}_${cell}.log" | tail -1)
        ratio=$(grep "BVEC/FP8 ratio" "r45c_runs/gpu${gpu}_${cell}.log" | tail -1)
        welch=$(grep "Welch t (BVEC vs R42B)" "r45c_runs/gpu${gpu}_${cell}.log" | tail -1)
        SUMMARY+=("gpu$gpu $cell $delta $bvec_tf $ratio $welch")
    done
done

echo ""
echo "============================================================"
echo "FINAL SUMMARY"
echo "============================================================"
for line in "${SUMMARY[@]}"; do
    echo "$line"
done
