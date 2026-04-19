#!/bin/bash
# R47 Dev C: Scale-load cachepolicy sweep for MXFP8 RCR/RRR/CRR fastpath kernels.
#
# Sweep MXFP8_{RCR,RRR,CRR}_V2_SCALE_CACHEPOLICY ∈ {0,1,2,3} for the 7
# compute-bound shapes. AMD CDNA cachepolicy bits:
#   0 = default
#   1 = GLC (globally coherent)
#   2 = SLC (streaming, skip L1)
#   3 = GLC + SLC
#
# Build pattern: a single binary bakes ALL THREE layout cachepolicy macros to
# the same value $P. We then run all 3 layouts in one Python invocation
# (MXFP8_LAYOUTS=rcr,rrr,crr). Result: 4 policies × 7 shapes = 28 builds and
# 28 invocations = 84 layout×policy×shape data points.
#
# v2 hygiene (after first sweep produced 2 transient outliers):
#   - 200 warmup + 400 iters (was 50/100)
#   - thermal preheat: 1 spurious warmup-only run before each measurement
#   - 3 repeats per cell, take median
#   - rm -f tk_mxfp8_layouts*.so before each build
#   - log per-build md5
#
# Outputs:
#   r47c_reports/build_${shape}_p${P}.log
#   r47c_reports/run_${shape}_p${P}_r${rep}.log
#   r47c_reports/summary.tsv  (median across reps)
#   r47c_reports/raw.tsv      (every rep)

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"
PYEXT="$(python3-config --extension-suffix)"
PHYS_GPU=2
OUT="$HERE/r47c_reports"
mkdir -p "$OUT"

export THUNDERKITTENS_ROOT="$WT"

SHAPES=(
  "8192 8192 8192"           # 8192cube
  "4096 4096 4096"           # 4096cube
  "4096 14336 4096"          # 8B Gate/Up-style (large N small K)
  "4096 4096 14336"          # 8B Down (small N large K)
  "4096 8192 8192"           # 70B Q/O
  "4096 28672 8192"          # 70B Gate/Up
  "4096 8192 28672"          # 70B Down
)
SHAPE_TAGS=(
  "8192cube"
  "4096cube"
  "8B_GateUp_4kx14kx4k"
  "8B_Down_4kx4kx14k"
  "70B_QO_4kx8kx8k"
  "70B_GateUp_4kx28kx8k"
  "70B_Down_4kx8kx28k"
)

POLICIES=(0 1 2 3)
N_REPS=3
WARMUP=${WARMUP:-200}
ITERS=${ITERS:-400}

SUMMARY="$OUT/summary.tsv"
RAW="$OUT/raw.tsv"
echo -e "shape\tM\tN\tK\tpolicy\tlayout\ttflops_median\ttflops_min\ttflops_max\tbuild_md5" > "$SUMMARY"
echo -e "shape\tM\tN\tK\tpolicy\trep\tlayout\ttflops\tavg_ms\tbuild_md5" > "$RAW"

build_one() {
  local tag=$1 m=$2 n=$3 k=$4 p=$5
  local log="$OUT/build_${tag}_p${p}.log"
  echo "===== BUILD ${tag} (${m}x${n}x${k}) policy=${p} ====="
  rm -f tk_mxfp8_layouts*.so
  make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=$m -DN_DIM=$n -DK_DIM=$k \
              -DMXFP8_RCR_V2_SCALE_CACHEPOLICY=$p \
              -DMXFP8_RRR_V2_SCALE_CACHEPOLICY=$p \
              -DMXFP8_CRR_V2_SCALE_CACHEPOLICY=$p" \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "BUILD FAIL rc=$rc tag=$tag p=$p"
    return 1
  fi
  local so_file=$(ls tk_mxfp8_layouts*.so 2>/dev/null | head -1)
  [ -z "$so_file" ] && { echo "BUILD MISSING .so"; return 1; }
  local md5=$(md5sum "$so_file" | awk '{print $1}')
  echo "BUILD OK $so_file md5=$md5"
  echo "$md5" > "$OUT/build_${tag}_p${p}.md5"
  return 0
}

bench_one() {
  # Args: shape_tag M N K P rep
  local tag=$1 m=$2 n=$3 k=$4 p=$5 rep=$6
  local log="$OUT/run_${tag}_p${p}_r${rep}.log"
  HIP_VISIBLE_DEVICES=$PHYS_GPU \
    MXFP8_BUILD_M=$m MXFP8_BUILD_N=$n MXFP8_BUILD_K=$k \
    MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS \
    MXFP8_LAYOUTS=rcr,rrr,crr \
    MXFP8_PRESHUFFLE_QUANT=1 \
    MXFP8_CHECK=0 \
    python3 test_mxfp8_python.py $m $n $k > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "BENCH FAIL rc=$rc tag=$tag p=$p rep=$rep"
    return 1
  fi
  return 0
}

# For each (shape, policy):
#   build, then preheat (1 throwaway bench), then 3 measured reps.
for i in "${!SHAPES[@]}"; do
  read M N K <<< "${SHAPES[$i]}"
  TAG="${SHAPE_TAGS[$i]}"
  for P in "${POLICIES[@]}"; do
    if ! build_one "$TAG" $M $N $K $P; then continue; fi
    md5=$(cat "$OUT/build_${TAG}_p${P}.md5")
    # Preheat throwaway
    HIP_VISIBLE_DEVICES=$PHYS_GPU \
      MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
      MXFP8_WARMUP=20 MXFP8_ITERS=20 \
      MXFP8_LAYOUTS=rcr,rrr,crr MXFP8_PRESHUFFLE_QUANT=1 MXFP8_CHECK=0 \
      python3 test_mxfp8_python.py $M $N $K > "$OUT/preheat_${TAG}_p${P}.log" 2>&1
    declare -A LAYOUT_VALS
    LAYOUT_VALS=([RCR]="" [RRR]="" [CRR]="")
    for r in $(seq 1 $N_REPS); do
      echo "===== BENCH ${TAG} (${M}x${N}x${K}) p=${P} rep=${r}/${N_REPS} ====="
      bench_one "$TAG" $M $N $K $P $r
      python3 - <<PYEOF >> "$RAW"
import re
text = open("$OUT/run_${TAG}_p${P}_r${r}.log").read()
sections = re.split(r"--- (RCR|RRR|CRR) Layout", text)
for i in range(1, len(sections), 2):
    label = sections[i]
    body = sections[i+1] if i+1 < len(sections) else ""
    m = re.search(r"Avg time:\s*([\d.]+)\s*ms,\s*TFLOPS:\s*([\d.]+)", body)
    if m:
        avg_ms, tflops = m.group(1), m.group(2)
        print("\t".join(["$TAG", "$M", "$N", "$K", "$P", "$r", label, tflops, avg_ms, "$md5"]))
PYEOF
    done
    # Compute median for this (shape, policy) per layout, append to summary.
    python3 - <<PYEOF >> "$SUMMARY"
import re, statistics
results = {"RCR": [], "RRR": [], "CRR": []}
for r in [1,2,3]:
    text = open("$OUT/run_${TAG}_p${P}_r%d.log" % r).read()
    sections = re.split(r"--- (RCR|RRR|CRR) Layout", text)
    for i in range(1, len(sections), 2):
        label = sections[i]
        body = sections[i+1] if i+1 < len(sections) else ""
        m = re.search(r"Avg time:\s*([\d.]+)\s*ms,\s*TFLOPS:\s*([\d.]+)", body)
        if m:
            results[label].append(float(m.group(2)))
for layout in ["RCR", "RRR", "CRR"]:
    vals = results[layout]
    if vals:
        med = statistics.median(vals)
        mn = min(vals); mx = max(vals)
        print("\t".join(["$TAG", "$M", "$N", "$K", "$P", layout,
                         f"{med:.2f}", f"{mn:.2f}", f"{mx:.2f}", "$md5"]))
PYEOF
  done
done

echo ""
echo "===== SWEEP DONE ====="
wc -l "$SUMMARY" "$RAW"
