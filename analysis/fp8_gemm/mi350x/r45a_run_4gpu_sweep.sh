#!/bin/bash
# R45 Dev A — fan paired BABA bench across GPU2/3/6/7 for INCLUDED R44A cells
# × 3 layouts (RCR / RRR / CRR).
#
# Cells: M ∈ {4 × {4kx4k, 8kx8k}, 8 × 8kx8k, 16 × 8kx8k} (4 INCLUDED cells)
# Layouts: rcr (primary), rrr, crr (geometry-symmetric, R44A coded but unbenched)
# Total runs: 4 cells × 3 layouts × 4 GPUs = 48
#
# Each GPU runs its 12 cell+layout combos serially; GPUs run in parallel.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p r45a_runs

GPUS=${GPUS:-"2 3 6 7"}
SHAPES=${SHAPES:-"4x4kx4k 4x8kx8k 8x8kx8k 16x8kx8k"}
LAYOUTS=${LAYOUTS:-"rcr rrr crr"}
N_PAIRS=${N_PAIRS:-5}
PREHEAT=${PREHEAT:-30}

run_gpu_serial() {
  local gpu=$1
  for shape in $SHAPES; do
    for layout in $LAYOUTS; do
      label="${shape}_${layout}"
      echo "[gpu$gpu] === $label ==="
      PHYS_GPU=$gpu LABEL=$label SHAPE=$shape LAYOUT=$layout \
        N_PAIRS=$N_PAIRS PREHEAT=$PREHEAT \
        ./r45a_orchestrate.sh > /dev/null 2>&1
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "[gpu$gpu] FAIL $label rc=$rc"
      else
        echo "[gpu$gpu] DONE $label"
      fi
    done
  done
}

PIDS=()
for g in $GPUS; do
  echo "spawn GPU$g sweep"
  run_gpu_serial $g &
  PIDS+=($!)
done

echo "waiting for ${#PIDS[@]} GPU sweeps..."
for pid in "${PIDS[@]}"; do
  wait $pid
done
echo "ALL DONE"
