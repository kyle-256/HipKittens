#!/bin/bash
# R45 Reviewer Phase 2a — STRICT RECONFIRM R44 Dev A M=2..16 fastpath
# 4 INCLUDED RCR cells × per-GPU; uses N_REPS=3 best-of for noise tolerance.
# R36 retry pattern adapted for short-running decode kernels (preheat dominates wall-time).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

OUTDIR="$HERE/r45_reviewer_p2a_devA"
mkdir -p "$OUTDIR"

PHYS_GPU=${PHYS_GPU:-2}
WARMUP=${WARMUP:-30}
ITERS=${ITERS:-200}
PREHEAT_S=${PREHEAT_S:-60.0}
N_REPS=${N_REPS:-3}

# Run mxfp8/fp8 back-to-back; keep best (max TF) of N_REPS
run_cell() {
    local cell=$1 M=$2 N=$3 K=$4
    local mxfp8_so="tk_mxfp8_r44a_decode_${cell}.cpython-310-x86_64-linux-gnu.so"
    local mxfp8_mod="tk_mxfp8_r44a_decode_${cell}"
    local fp8_so="tk_fp8_r44a_${cell}.cpython-310-x86_64-linux-gnu.so"
    local fp8_mod="tk_fp8_r44a_${cell}"

    if [ ! -f "$HERE/$mxfp8_so" ] || [ ! -f "$HERE/$fp8_so" ]; then
        echo "[r45_p2a cell=$cell] MISSING .so"
        return 1
    fi

    local STAGE="/tmp/r45_p2a_gpu${PHYS_GPU}"
    mkdir -p "$STAGE"
    cp -f "$HERE/$mxfp8_so" "$STAGE/$mxfp8_so"
    cp -f "$HERE/$fp8_so" "$STAGE/$fp8_so"
    for f in r44a_decode_m2_16_bench.py; do
        ln -sf "$HERE/$f" "$STAGE/$f"
    done

    local OUT="$OUTDIR/${cell}_gpu${PHYS_GPU}.log"
    echo "===== R45 Phase 2a cell=$cell M=$M N=$N K=$K phys_gpu=$PHYS_GPU =====" | tee "$OUT"

    local mxfp8_tfs="" fp8_tfs=""
    for rep in $(seq 1 $N_REPS); do
        # mxfp8
        ( cd "$STAGE" && \
          PYTHONPATH="$STAGE:${PYTHONPATH:-}" \
          ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 \
          WARMUP=$WARMUP ITERS=$ITERS PREHEAT_S=$PREHEAT_S CHECK_SNR=0 \
          python3 r44a_decode_m2_16_bench.py "$mxfp8_mod" mxfp8 decode_m2_16 rcr $M $N $K \
          > "$OUTDIR/${cell}_gpu${PHYS_GPU}_mxfp8_rep${rep}.out" 2> "$OUTDIR/${cell}_gpu${PHYS_GPU}_mxfp8_rep${rep}.err" )
        local mtf
        mtf=$(grep "R44A_RESULT" "$OUTDIR/${cell}_gpu${PHYS_GPU}_mxfp8_rep${rep}.out" | sed -nE 's/.*tflops=([0-9.]+).*/\1/p' | head -1)

        # fp8 (immediately, while clock still high)
        ( cd "$STAGE" && \
          PYTHONPATH="$STAGE:${PYTHONPATH:-}" \
          ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 \
          WARMUP=$WARMUP ITERS=$ITERS PREHEAT_S=$PREHEAT_S CHECK_SNR=0 \
          python3 r44a_decode_m2_16_bench.py "$fp8_mod" fp8 fp8_pertensor rcr $M $N $K \
          > "$OUTDIR/${cell}_gpu${PHYS_GPU}_fp8_rep${rep}.out" 2> "$OUTDIR/${cell}_gpu${PHYS_GPU}_fp8_rep${rep}.err" )
        local ftf
        ftf=$(grep "R44A_RESULT" "$OUTDIR/${cell}_gpu${PHYS_GPU}_fp8_rep${rep}.out" | sed -nE 's/.*tflops=([0-9.]+).*/\1/p' | head -1)

        echo "[rep=$rep mxfp8_tf=${mtf:-NA} fp8_tf=${ftf:-NA}]" | tee -a "$OUT"
        [ -n "$mtf" ] && mxfp8_tfs="$mxfp8_tfs $mtf"
        [ -n "$ftf" ] && fp8_tfs="$fp8_tfs $ftf"
    done

    # Use max-of-reps as best (least throttled)
    local best_mxfp8 best_fp8
    best_mxfp8=$(python3 -c "vals=[$(echo $mxfp8_tfs | tr ' ' ',' | sed 's/^,//;s/,$//')]; print(max(vals) if vals else 'NA')")
    best_fp8=$(python3 -c "vals=[$(echo $fp8_tfs | tr ' ' ',' | sed 's/^,//;s/,$//')]; print(max(vals) if vals else 'NA')")

    if [ "$best_mxfp8" != "NA" ] && [ "$best_fp8" != "NA" ]; then
        local ratio
        ratio=$(python3 -c "print(f'{(${best_mxfp8}/${best_fp8})*100:.2f}')")
        echo "[BEST cell=$cell mxfp8_tf=$best_mxfp8 fp8_tf=$best_fp8 ratio=${ratio}%]" | tee -a "$OUT"
        echo "$cell,$PHYS_GPU,$best_mxfp8,$best_fp8,$ratio,OK" >> "$OUTDIR/results.csv"
        return 0
    fi
    echo "[FAIL cell=$cell]" | tee -a "$OUT"
    echo "$cell,$PHYS_GPU,${best_mxfp8},${best_fp8},NA,FAIL" >> "$OUTDIR/results.csv"
    return 1
}

# Reset csv on first GPU only (PHYS_GPU==2 starts the cycle)
# Each parallel job appends; deduplication handled in aggregator
run_cell 4x4kx4k   4  4096 4096
run_cell 4x8kx8k   4  8192 8192
run_cell 8x8kx8k   8  8192 8192
run_cell 16x8kx8k  16 8192 8192
