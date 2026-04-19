#!/bin/bash
# R45 Dev A — within-GPU same-.so N=3 reps for R44/R45 NEW rule 1 (within-GPU
# variance baseline). Runs on GPU2 only, 4 INCLUDED cells × RCR layout × 3 reps.
# Goal: report cross-rep stdev to compare against cross-GPU spread.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p r45a_runs

GPU=${GPU:-2}
SHAPES=${SHAPES:-"4x4kx4k 4x8kx8k 8x8kx8k 16x8kx8k"}
LAYOUT=${LAYOUT:-rcr}
N_REPS=${N_REPS:-3}

for shape in $SHAPES; do
  for rep in $(seq 1 $N_REPS); do
    label="rep${rep}_${shape}_${LAYOUT}"
    PHYS_GPU=$GPU LABEL=$label SHAPE=$shape LAYOUT=$LAYOUT \
      N_PAIRS=5 PREHEAT=30 \
      ./r45a_orchestrate.sh > /dev/null 2>&1
    rc=$?
    echo "[within-gpu rep=$rep $shape $LAYOUT rc=$rc]"
  done
done
echo "WITHIN-GPU REPS DONE"
