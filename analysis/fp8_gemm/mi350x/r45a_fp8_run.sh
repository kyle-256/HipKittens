#!/bin/bash
# R45 Dev A — FP8 reference bench across 4 INCLUDED cells × 4 GPUs.
# Single-call FP8 per-tensor median TFLOPS (no paired BABA needed).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p r45a_runs

GPUS=${GPUS:-"2 3 6 7"}
SHAPES=${SHAPES:-"4x4kx4k 4x8kx8k 8x8kx8k 16x8kx8k"}
LAYOUTS=${LAYOUTS:-"rcr rrr crr"}
PREHEAT=${PREHEAT:-30}

run_gpu() {
  local gpu=$1
  for shape in $SHAPES; do
    case "$shape" in
      4x4kx4k)  M=4;  N=4096; K=4096 ;;
      4x8kx8k)  M=4;  N=8192; K=8192 ;;
      8x4kx4k)  M=8;  N=4096; K=4096 ;;
      8x8kx8k)  M=8;  N=8192; K=8192 ;;
      16x4kx4k) M=16; N=4096; K=4096 ;;
      16x8kx8k) M=16; N=8192; K=8192 ;;
    esac
    MOD="tk_fp8_r44a_${shape}"
    SO="$HERE/${MOD}.cpython-310-x86_64-linux-gnu.so"
    for layout in $LAYOUTS; do
      label="fp8_${shape}_${layout}_gpu${gpu}"
      OUT="r45a_runs/${label}.log"
      echo "[gpu$gpu] === $label ===" | tee "$OUT"
      ROCR_VISIBLE_DEVICES=$gpu HIP_VISIBLE_DEVICES=0 PHYS_GPU=$gpu \
        M=$M N=$N K=$K SO="$SO" MOD="$MOD" LAYOUT=$layout \
        PREHEAT_S=$PREHEAT MXFP8_WARMUP=30 MXFP8_ITERS=100 \
        python3 r45a_fp8_ref_bench.py >> "$OUT" 2>&1
      tail -3 "$OUT"
    done
  done
}

PIDS=()
for g in $GPUS; do
  run_gpu $g &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait $pid; done
echo "FP8 ALL DONE"
