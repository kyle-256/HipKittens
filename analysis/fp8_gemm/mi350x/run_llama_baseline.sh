#!/bin/bash
# R25 LLaMA baseline driver
# Usage: ./run_llama_baseline.sh <GPU_ID>
set -u
GPU=${1:-5}
WT=/tmp/wt-r25-llama
MIROOT=$WT/analysis/fp8_gemm/mi350x
OUTDIR=$MIROOT/llama_runs
mkdir -p "$OUTDIR"

# Shape table: name M N K
SHAPES=(
  "llama8b_qo  4096 4096 4096"
  "llama8b_kv  4096 1024 4096"
  "llama8b_gateup  4096 14336 4096"
  "llama8b_down  4096 4096 14336"
  "llama70b_qo  4096 8192 8192"
  "llama70b_kv  4096 1024 8192"
  "llama70b_gateup  4096 28672 8192"
  "llama70b_down  4096 8192 28672"
)

set_perf() {
  rocm-smi -d "$GPU" --setperflevel high >/dev/null 2>&1
}

build_one() {
  local kind=$1 M=$2 N=$3 K=$4
  local src target
  if [ "$kind" = "fp8" ]; then
    src=kernel_fp8_layouts.cpp; target=tk_fp8_layouts
  else
    src=kernel_mxfp8_layouts.cpp; target=tk_mxfp8_layouts
  fi
  cd "$MIROOT"
  make clean >/dev/null 2>&1
  THUNDERKITTENS_ROOT=$WT make -j8 TARGET=$target SRC=$src \
    CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" \
    > "$OUTDIR/build_${kind}_${M}x${N}x${K}.log" 2>&1
  return $?
}

run_one() {
  local kind=$1 name=$2 M=$3 N=$4 K=$5 run_idx=$6
  set_perf
  cd "$MIROOT"
  if [ "$kind" = "fp8" ]; then
    HIP_VISIBLE_DEVICES=$GPU FP8_WARMUP=200 FP8_ITERS=100 \
      timeout 600 python3 test_python.py $M $N $K \
      > "$OUTDIR/run_${kind}_${name}_r${run_idx}.log" 2>&1
  else
    HIP_VISIBLE_DEVICES=$GPU MXFP8_WARMUP=100 MXFP8_ITERS=100 MXFP8_PRESHUFFLE_QUANT=1 \
      timeout 600 python3 test_mxfp8_python.py $M $N $K \
      > "$OUTDIR/run_${kind}_${name}_r${run_idx}.log" 2>&1
  fi
  return $?
}

echo "=== R25 LLaMA baseline @ GPU$GPU ==="
date

for shape_line in "${SHAPES[@]}"; do
  read -r name M N K <<< "$shape_line"
  for kind in fp8 mxfp8; do
    echo "[$(date +%H:%M:%S)] BUILD $kind $name ${M}x${N}x${K}"
    if ! build_one $kind $M $N $K; then
      echo "  BUILD FAILED, see $OUTDIR/build_${kind}_${M}x${N}x${K}.log"
      continue
    fi
    for i in 1 2 3 4 5; do
      echo "  [$(date +%H:%M:%S)] RUN $i"
      run_one $kind $name $M $N $K $i
      tflops=$(grep -E "^.*TFLOPS:" "$OUTDIR/run_${kind}_${name}_r${i}.log" | head -3 | sed -E 's/.*TFLOPS: //')
      echo "    TFLOPS RCR/RRR/CRR: $(echo $tflops | tr '\n' ' ')"
    done
  done
done

echo "=== DONE @ $(date) ==="
