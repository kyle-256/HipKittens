#!/bin/bash
# R43 Reviewer Phase 2/3: paired bench dispatcher (one_so_layout or two_so).
# Mirrors r40_reviewer_phase2_pair.sh interface.
#
# R44 Dev D PATCH (R43 NEW methodology rule 1, MANDATORY):
#   Wrapped bench in R36 NEW 3-gate retry loop:
#     G1  sclk-post-preheat >= SCLK_GATE_MHZ (2200 default)
#     G2a sclk-post-bench   >= SCLK_POSTBENCH_GATE_MHZ (2200 default)
#     G2b per-run stdev/mean <= STDEV_MEAN_GATE (0.01 default)
#   Reference impl: r38c_8bdown_orchestrate.sh:79-140
#   Lack of G1 caused R42 Reviewer P3.3 false-positive (~1 day escalation).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

OUTDIR_P2="$HERE/r43_reviewer_phase2"
OUTDIR_P3="$HERE/r43_reviewer_phase3"
mkdir -p "$OUTDIR_P2" "$OUTDIR_P3"

PHYS_GPU=${PHYS_GPU:-2}
LABEL=${LABEL:-cell}
N_PAIRS=${N_PAIRS:-10}
WARMUP=${WARMUP:-30}
ITERS=${ITERS:-50}
PREHEAT=${PREHEAT:-60}
WARMUP_PAIRS=${WARMUP_PAIRS:-2}
BENCH_KIND=${BENCH_KIND:-one_so_layout}
PHASE=${PHASE:-2}

# R44 Dev D: 3-gate retry parameters (R36 NEW, R43 NEW MANDATORY)
SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
SCLK_POSTBENCH_GATE_MHZ=${SCLK_POSTBENCH_GATE_MHZ:-2200}
STDEV_MEAN_GATE=${STDEV_MEAN_GATE:-0.01}
MAX_RETRIES=${MAX_RETRIES:-3}

if [ "$PHASE" = "2" ]; then OUTDIR="$OUTDIR_P2"; else OUTDIR="$OUTDIR_P3"; fi

OUT="$OUTDIR/${LABEL}_gpu${PHYS_GPU}.log"
echo "===== R43 Phase=$PHASE label=$LABEL gpu=$PHYS_GPU N_PAIRS=$N_PAIRS BENCH_KIND=$BENCH_KIND PREHEAT=$PREHEAT =====" | tee "$OUT"
echo "[orchestrate] R36/R43 3-gate: G1>=${SCLK_GATE_MHZ}MHz G2a>=${SCLK_POSTBENCH_GATE_MHZ}MHz G2b<=${STDEV_MEAN_GATE} MAX_RETRIES=${MAX_RETRIES}" | tee -a "$OUT"
date | tee -a "$OUT"

# R44 Dev D: helper to run one bench attempt + parse 3 gates.
# Inputs: $BENCH_KIND $LABEL $PHYS_GPU $OUTDIR $attempt + env vars exported by caller
# Sets: PASS (0/1), POST_MHZ, POSTB_MHZ, RATIO_A, RATIO_B
run_bench_attempt() {
  local attempt=$1
  BENCH_OUT="$OUTDIR/${LABEL}_gpu${PHYS_GPU}_attempt${attempt}.txt"
  BENCH_ERR="$OUTDIR/${LABEL}_gpu${PHYS_GPU}_attempt${attempt}.err"
  if [ "$BENCH_KIND" = "two_so" ]; then
    MXFP8_DISPATCH_TRACE=1 \
    ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
      M=$M N=$N K=$K \
      SO_A="$SO_A" MOD_A="$MOD_A" \
      SO_B="$SO_B" MOD_B="$MOD_B" \
      N_PAIRS=$N_PAIRS MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS \
      PREHEAT_S=$PREHEAT WARMUP_PAIRS=$WARMUP_PAIRS \
      python3 r37_paired_bench_2so.py > "$BENCH_OUT" 2> "$BENCH_ERR"
  else
    MXFP8_DISPATCH_TRACE=1 \
    ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
      M=$M N=$N K=$K SO="$SO" MOD="$MOD" \
      LAYOUT_A=$LAYOUT_A LAYOUT_B=$LAYOUT_B \
      N_PAIRS=$N_PAIRS MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS \
      PREHEAT_S=$PREHEAT WARMUP_PAIRS=$WARMUP_PAIRS \
      python3 r33c_paired_bench.py > "$BENCH_OUT" 2> "$BENCH_ERR"
  fi
  brc=$?
  echo "[bench attempt=$attempt rc=$brc]" | tee -a "$OUT"
  cat "$BENCH_OUT" | tee -a "$OUT"
  echo "--- stderr (head) ---" | tee -a "$OUT"
  head -20 "$BENCH_ERR" | tee -a "$OUT"

  # G1: sclk-post-preheat
  POST_LINE=$(grep "sclk-post-preheat" "$BENCH_OUT" | head -1)
  POST_MHZ=$(echo "$POST_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')
  # G2a: sclk-post-bench
  POSTB_LINE=$(grep "sclk-post-bench" "$BENCH_OUT" | head -1)
  POSTB_MHZ=$(echo "$POSTB_LINE" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p')

  # G2b: per-run stdev/mean for both medians.  r33c_paired_bench prints
  # "<label_a>  median=... mean=... stdev=...".  r37_paired_bench_2so prints
  # "CRR_DEFAULT" / "CRR_HBSHRINK".  Parse both via permissive regex.
  RATIO_A=$(grep -E "^[A-Za-z_]+ +median=" "$BENCH_OUT" | sed -n '1p' | sed -nE 's/.*median=([0-9.]+).*stdev=([0-9.]+).*/\1 \2/p' | python3 -c "import sys
try:
  m,s=map(float,sys.stdin.read().split())
  print(f'{(s/m):.6f}' if m>0 else 'NaN')
except: print('NaN')" 2>/dev/null)
  RATIO_B=$(grep -E "^[A-Za-z_]+ +median=" "$BENCH_OUT" | sed -n '2p' | sed -nE 's/.*median=([0-9.]+).*stdev=([0-9.]+).*/\1 \2/p' | python3 -c "import sys
try:
  m,s=map(float,sys.stdin.read().split())
  print(f'{(s/m):.6f}' if m>0 else 'NaN')
except: print('NaN')" 2>/dev/null)
  if [ -z "$RATIO_A" ]; then RATIO_A="NaN"; fi
  if [ -z "$RATIO_B" ]; then RATIO_B="NaN"; fi

  RATIO_PASS=0
  if [ "$RATIO_A" != "NaN" ] && [ "$RATIO_B" != "NaN" ]; then
    RATIO_PASS=$(python3 -c "print(1 if max(${RATIO_A}, ${RATIO_B}) <= ${STDEV_MEAN_GATE} else 0)" 2>/dev/null || echo 0)
  fi

  echo "[orchestrate] attempt=$attempt rc=$brc G1 sclk-post-preheat=${POST_MHZ:-unknown}MHz (gate=${SCLK_GATE_MHZ}) G2a sclk-post-bench=${POSTB_MHZ:-unknown}MHz (gate=${SCLK_POSTBENCH_GATE_MHZ}) G2b stdev/mean A=${RATIO_A} B=${RATIO_B} (gate=${STDEV_MEAN_GATE} pass=${RATIO_PASS})" | tee -a "$OUT"

  PASS=1
  if [ $brc -ne 0 ]; then PASS=0; fi
  if [ -z "$POST_MHZ" ] || [ "$POST_MHZ" -lt "$SCLK_GATE_MHZ" ]; then PASS=0; fi
  GATE2_PASS=0
  if [ -n "$POSTB_MHZ" ] && [ "$POSTB_MHZ" -ge "$SCLK_POSTBENCH_GATE_MHZ" ]; then GATE2_PASS=1; fi
  if [ "$RATIO_PASS" = "1" ]; then GATE2_PASS=1; fi
  if [ $GATE2_PASS -eq 0 ]; then PASS=0; fi
}

if [ "$BENCH_KIND" = "two_so" ]; then
  : ${SO_A:?missing SO_A}; : ${MOD_A:?missing MOD_A}; : ${SO_B:?missing SO_B}; : ${MOD_B:?missing MOD_B}
  : ${M:?missing M}; : ${N:?missing N}; : ${K:?missing K}
  echo "[two_so] SO_A=$SO_A MOD_A=$MOD_A SO_B=$SO_B MOD_B=$MOD_B M=$M N=$N K=$K" | tee -a "$OUT"
elif [ "$BENCH_KIND" = "one_so_layout" ]; then
  : ${SO:?missing SO}; : ${MOD:?missing MOD}
  : ${M:?missing M}; : ${N:?missing N}; : ${K:?missing K}
  : ${LAYOUT_A:=crr}; : ${LAYOUT_B:=rrr}
  echo "[one_so_layout] SO=$SO MOD=$MOD LAYOUT_A=$LAYOUT_A LAYOUT_B=$LAYOUT_B M=$M N=$N K=$K" | tee -a "$OUT"
fi

attempt=1
PASS=0
while [ $attempt -le $MAX_RETRIES ]; do
  echo "[bench attempt $attempt/$MAX_RETRIES]" | tee -a "$OUT"
  run_bench_attempt $attempt
  if [ $PASS -eq 1 ]; then
    echo "[orchestrate] PASS R36 3-gate logic at attempt=$attempt; using as final." | tee -a "$OUT"
    BENCH_OUT="$OUTDIR/${LABEL}_gpu${PHYS_GPU}_attempt${attempt}.txt"
    BENCH_ERR="$OUTDIR/${LABEL}_gpu${PHYS_GPU}_attempt${attempt}.err"
    cp "$BENCH_OUT" "$OUTDIR/${LABEL}_gpu${PHYS_GPU}_clean.txt"
    cp "$BENCH_ERR" "$OUTDIR/${LABEL}_gpu${PHYS_GPU}_clean.err"
    # Also retain the legacy non-attempt-suffixed path for downstream parsers.
    cp "$BENCH_OUT" "$OUTDIR/${LABEL}_gpu${PHYS_GPU}_bench.txt"
    cp "$BENCH_ERR" "$OUTDIR/${LABEL}_gpu${PHYS_GPU}_bench.err"
    break
  fi
  echo "[orchestrate] attempt=$attempt FAILED gates; retry after 5s..." | tee -a "$OUT"
  sleep 5
  attempt=$((attempt+1))
done

if [ $PASS -ne 1 ]; then
  echo "[orchestrate] FAIL: R36 3-gate logic did not PASS within MAX_RETRIES=$MAX_RETRIES attempts (likely thermal/power throttle; investigate)" | tee -a "$OUT"
  exit 2
fi

date | tee -a "$OUT"
echo "DONE label=$LABEL gpu=$PHYS_GPU"
