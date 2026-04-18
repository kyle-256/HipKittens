#!/bin/bash
# R26 Reviewer reverify driver
set -u
GPU=${GPU:-1}
WT=/tmp/wt-r26-rev
MIROOT=$WT/analysis/fp8_gemm/mi350x
OUTDIR=$WT/r26_runs
mkdir -p "$OUTDIR"

set_perf() { rocm-smi -d "$GPU" --setperflevel high >/dev/null 2>&1; }
get_sclk() {
  rocm-smi -d "$GPU" --showclocks 2>/dev/null \
    | awk -F'[():]' '/sclk clock/ {gsub(/[^0-9]/,"",$3); print $3; exit}'
}

# Args: kind name M N K layout
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
}

run_one() {
  local kind=$1 name=$2 M=$3 N=$4 K=$5 layout=$6 idx=$7
  set_perf
  cd "$MIROOT"
  if [ "$kind" = "fp8" ]; then
    HIP_VISIBLE_DEVICES=$GPU FP8_WARMUP=300 FP8_ITERS=300 \
      FP8_LAYOUTS=$layout MXFP8_CHECK=0 \
      timeout 600 python3 test_python.py $M $N $K \
      > "$OUTDIR/run_${kind}_${name}_${layout}_r${idx}.log" 2>&1
  else
    HIP_VISIBLE_DEVICES=$GPU MXFP8_WARMUP=200 MXFP8_ITERS=300 MXFP8_PRESHUFFLE_QUANT=1 \
      MXFP8_LAYOUTS=$layout MXFP8_CHECK=0 \
      timeout 600 python3 test_mxfp8_python.py $M $N $K \
      > "$OUTDIR/run_${kind}_${name}_${layout}_r${idx}.log" 2>&1
  fi
}

# Cells: name M N K layout
CELLS=(
  "70b_kv_crr        4096 1024 8192   crr"
  "70b_gateup_crr    4096 28672 8192  crr"
  "70b_down_crr      4096 8192 28672  crr"
  "8b_kv_rrr         4096 1024 4096   rrr"
  "70b_down_rrr      4096 8192 28672  rrr"
)

echo "=== R26 Reviewer reverify @ GPU$GPU $(date) ==="

declare -A FP8_BUILT
declare -A MXFP8_BUILT

for cell in "${CELLS[@]}"; do
  read -r name M N K layout <<< "$cell"
  shape="${M}x${N}x${K}"
  echo "[$(date +%H:%M:%S)] CELL $name $shape layout=$layout"

  # Build FP8 if not built for this shape
  if [ -z "${FP8_BUILT[$shape]:-}" ]; then
    echo "  BUILD fp8 $shape"
    build_one fp8 $M $N $K
    FP8_BUILT[$shape]=1
  fi
  # Run FP8 10 reps for this layout
  for i in $(seq 1 10); do
    sclk_pre=$(get_sclk)
    run_one fp8 $name $M $N $K $layout $i
    sclk_post=$(get_sclk)
    tflops=$(grep -E "TFLOPS:" "$OUTDIR/run_fp8_${name}_${layout}_r${i}.log" | head -1 | sed -E 's/.*TFLOPS: //')
    echo "  FP8 r$i sclk=${sclk_pre}->${sclk_post} TFLOPS=$tflops"
  done

  # Build MXFP8 if needed
  if [ -z "${MXFP8_BUILT[$shape]:-}" ]; then
    echo "  BUILD mxfp8 $shape"
    build_one mxfp8 $M $N $K
    MXFP8_BUILT[$shape]=1
  fi
  for i in $(seq 1 10); do
    sclk_pre=$(get_sclk)
    run_one mxfp8 $name $M $N $K $layout $i
    sclk_post=$(get_sclk)
    tflops=$(grep -E "TFLOPS:" "$OUTDIR/run_mxfp8_${name}_${layout}_r${i}.log" | head -1 | sed -E 's/.*TFLOPS: //')
    echo "  MXFP8 r$i sclk=${sclk_pre}->${sclk_post} TFLOPS=$tflops"
  done
done

echo "=== DONE @ $(date) ==="
