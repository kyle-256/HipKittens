#!/bin/bash
# R50 Dev D Phase 2: Full RRR sweep (7 cells) for winning mask vs baseline
# Strict SCLK protocol: 5 runs/cell, 30s cooldown, 60s rebuild cooldown.
# Usage: WIN_MASK=<n> bash r50d_phase2_bench.sh
set -e
cd /shared_nfs/kyle/test/Hipkittens2/.claude/worktrees/agent-ac10186c/analysis/fp8_gemm/mi350x
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2
export ROCM_PATH=/opt/rocm
export PYTHONPATH=$THUNDERKITTENS_ROOT/src/common/pyutils:$PYTHONPATH

WIN_MASK=${WIN_MASK:?must set WIN_MASK env var}

mkdir -p r50d_results

SHAPES=(
  "8192 8192 8192 8192cube"
  "4096 4096 4096 8B_QO"
  "4096 14336 4096 8B_GateUp"
  "4096 4096 14336 8B_Down"
  "4096 8192 8192 70B_QO"
  "4096 28672 8192 70B_GateUp"
  "4096 8192 28672 70B_Down"
)

# Tags: baseline (mask=0) and winner (mask=$WIN_MASK)
TAGS=("baseline:0" "winner:${WIN_MASK}")

for entry in "${TAGS[@]}"; do
  tag="${entry%%:*}"
  mask="${entry##*:}"
  for line in "${SHAPES[@]}"; do
    read M N K LABEL <<< "$line"
    # Skip 8B_GateUp for winner since Phase 1 already covered it
    if [ "$tag" = "winner" ] && [ "$LABEL" = "8B_GateUp" ]; then
      continue
    fi
    echo "=========="
    echo ">>> tag=$tag mask=$mask shape=${M}x${N}x${K} ($LABEL) $(date)"
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
      CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DMXFP8_RRR_SCHED_BARRIER=$mask" \
      > r50d_results/build_phase2_${tag}_${LABEL}.log 2>&1
    echo ">>> Build done, sleeping 60s rebuild cooldown..."
    sleep 60
    for i in 1 2 3 4 5; do
      echo ">>> tag=$tag run=$i $(date)"
      HIP_VISIBLE_DEVICES=4 MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=100 MXFP8_ITERS=200 MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=rrr MXFP8_PRESHUFFLE_QUANT=1 \
        python3 test_mxfp8_python.py $M $N $K 2>&1 | grep TFLOPS \
        | tee -a r50d_results/r50d_phase2_${tag}_${LABEL}.log
      sleep 30
    done
  done
done

echo "ALL DONE $(date)"
