#!/bin/bash
# R48 Dev E benchmark driver: MXFP8 RRR full-unroll lever
# Tests baseline (no unroll) vs unroll=32 across all 7 RRR shapes.
# Cooldown discipline: 3 runs × 20s sleep.
set -e
cd /shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2
export ROCM_PATH=/opt/rocm
export PYTHONPATH=$THUNDERKITTENS_ROOT/src/common/pyutils:$PYTHONPATH

# Shapes: M N K LABEL  -- all 7 RRR cells from R47 baseline
SHAPES=(
  "4096 4096 4096 8B_QO"          # K=4096 target
  "4096 14336 4096 8B_GateUp"     # K=4096 target
  "4096 4096 14336 8B_Down"       # K=14336 no-regression
  "4096 8192 8192 70B_QO"         # K=8192 no-regression
  "4096 28672 8192 70B_GateUp"    # K=8192 no-regression
  "4096 8192 28672 70B_Down"      # K=28672 no-regression
  "8192 8192 8192 8192cube"       # K=8192 no-regression
)

UNROLL_VAL="${UNROLL_VAL:-32}"
TAGS=(baseline unroll)

for tag in "${TAGS[@]}"; do
  EXTRA=""
  if [ "$tag" = "unroll" ]; then
    EXTRA="-DMXFP8_RRR_MAIN_UNROLL=$UNROLL_VAL"
  fi
  for line in "${SHAPES[@]}"; do
    read M N K LABEL <<< "$line"
    echo ">>> tag=$tag shape=${M}x${N}x${K} ($LABEL)"
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
      CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K $EXTRA" > /dev/null 2>&1
    for i in 1 2 3; do
      HIP_VISIBLE_DEVICES=3 MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=50 MXFP8_ITERS=100 MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=rrr MXFP8_PRESHUFFLE_QUANT=1 \
        python3 test_mxfp8_python.py $M $N $K 2>&1 | grep TFLOPS \
        | tee -a r48e_${tag}_${LABEL}_run${i}.log
      sleep 20
    done
  done
done

echo "ALL DONE"
