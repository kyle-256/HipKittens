#!/bin/bash
# R48 Dev E unroll factor sweep for K=4096 RRR cells.
# After unroll=32 catastrophically regressed (~ -70% perf due to VGPR spill),
# sweep smaller factors {2, 4, 8} on the two K=4096 RRR target cells.
set -e
cd /shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2
export ROCM_PATH=/opt/rocm
export PYTHONPATH=$THUNDERKITTENS_ROOT/src/common/pyutils:$PYTHONPATH

SHAPES=(
  "4096 4096 4096 8B_QO"
  "4096 14336 4096 8B_GateUp"
)
FACTORS=(2 4 8)

for u in "${FACTORS[@]}"; do
  for line in "${SHAPES[@]}"; do
    read M N K LABEL <<< "$line"
    echo ">>> u=$u shape=${M}x${N}x${K} ($LABEL)"
    rm -f tk_mxfp8_layouts*.so
    # Capture VGPRs & spill.
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
      CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DMXFP8_RRR_MAIN_UNROLL=$u" \
      2>&1 | grep -A 6 "_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EE" \
      | grep -E "VGPRs|Spill|ScratchSize" | head -5 \
      | tee r48e_unroll${u}_${LABEL}_resource.log
    for i in 1 2 3; do
      HIP_VISIBLE_DEVICES=3 MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=50 MXFP8_ITERS=100 MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=rrr MXFP8_PRESHUFFLE_QUANT=1 \
        python3 test_mxfp8_python.py $M $N $K 2>&1 | grep TFLOPS \
        | tee -a r48e_unroll${u}_${LABEL}_run${i}.log
      sleep 20
    done
  done
done

echo "SWEEP DONE"
