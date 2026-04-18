#!/bin/bash
# R38 Dev D — Methodology hardening: G1' fallback gate (drop-in replacement
# for r36_reviewer_orchestrate.sh / r37a_orchestrate.sh).
#
# Successor to r36_reviewer_orchestrate.sh and r37a_orchestrate.sh. Adds the
# R37 NEW G1' fallback gate to recover samples lost by R36 G1's
# over-aggressive post-preheat MHz check on contended hosts.
#
# Gate logic per attempt:
#   G1  (R34 carry-forward): sclk-post-preheat >= SCLK_GATE_MHZ (default 2200)
#   G2a (R36 NEW):           sclk-post-bench   >= SCLK_GATE_MHZ (default 2200)
#   G2b (R36 NEW):           per-run stdev/mean (CV) <= STDEV_RATIO_MAX (0.01)
#   G1' (R37 NEW fallback):  sclk-post-bench >= SCLK_GATE_MHZ AND
#                            median_tflops > G1P_MIN_MEDIAN_TF (default 500)
#
# Acceptance ladder (R38 NEW):
#   1. Run bench. Compute G1, G2a, G2b.
#   2. If (G1 AND G2a AND G2b) — accept as STRICT (gate_path="G1+G2a+G2b").
#   3. Else if G2a AND G2b AND G1' — RETRY up to MAX_G1_RETRIES=3 with
#      back-off, hoping for a clean G1 pass. After exhausting retries,
#      accept the *best* G1' run (one with G2a+G2b passing) as fallback.
#      Mark sample with gate_path="G1'_fallback".
#   4. Else if G2a AND G2b — retry up to MAX_RETRIES=3. If still failing,
#      mark EXHAUSTED.
#
# All accepted samples emit a JSON record with gate_path so reviewers can
# audit which gate path each sample took. Aggregate JSON written to
# $OUTDIR/${CELL}_gpu${PHYS_GPU}_samples.json
#
# Methodology rules (R29-R37 closures, MUST follow):
#   1. rm -f tk_mxfp8_layouts*.so + log per-build md5 (R29 Dev C rule)
#      [R37 Dev C noted md5 unreliable; we still log it for traceability,
#       but the load-bearing build-hygiene check is now r38_nm_gate.sh.]
#   2. rocm-smi -d $PHYS_GPU not -d 0 (R31 Reviewer rule)
#   3. R32: cross-GPU triangulation on >=2 GPUs
#   4. R33: any GPU >+1.5% above others ⇒ use min-of-GPUs
#   5. R34: auto-retry on G1 failure (now also relaxes to G1')
#   6. R36 NEW: G2a + G2b mandatory (load-bearing)
#   7. R37 NEW: G1' fallback when G1 over-rejects under contention
#
# Usage:
#   PHYS_GPU=4 ./r38_orchestrate.sh
#   PHYS_GPU=0 BASE_SO=/tmp/r38_base.so PROD_SO=/tmp/r38_prod.so \
#     CELL=70b_kv_crr ./r38_orchestrate.sh
#
# Optional env:
#   N_RUNS              (default 5)
#   WARMUP              (default 50)
#   ITERS               (default 100)
#   SCLK_GATE_MHZ       (default 2200)  - both G1 and G2a use this
#   STDEV_RATIO_MAX     (default 0.01)  - G2b CV ceiling
#   G1P_MIN_MEDIAN_TF   (default 500)   - G1' minimum sane median TF
#   MAX_G1_RETRIES      (default 3)     - tries to recover full G1 pass
#   MAX_RETRIES         (default 3)     - hard ceiling per (tag, rep)
#   M / N / K           (default 4096 / 1024 / 8192 — 70B-KV V2-CRR fastpath)
#   LAYOUT              (default crr)
#   CELL                (default 70b_kv_crr)
#   N_PAIRS             (default 3 — BABA paired prod/base reps)
#   BASE_SO             (default /tmp/r37a_baseline_70bkv.so)
#   PROD_SO             (default /tmp/r37a_prod_70bkv.so)
#   OUTSUBDIR           (default r38_runs)
#   BENCH_SCRIPT        (default r35_reviewer_bench5x.py)
#
# Acceptance contract: this script is a drop-in replacement for
# r36_reviewer_orchestrate.sh — it accepts every clean run that R36/R37
# would accept, and additionally accepts G1' fallbacks that R36 would
# have lost to over-rejection.

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"

PHYS_GPU=${PHYS_GPU:-0}
N_RUNS=${N_RUNS:-5}
WARMUP=${WARMUP:-50}
ITERS=${ITERS:-100}
SCLK_GATE_MHZ=${SCLK_GATE_MHZ:-2200}
STDEV_RATIO_MAX=${STDEV_RATIO_MAX:-0.01}
G1P_MIN_MEDIAN_TF=${G1P_MIN_MEDIAN_TF:-500}
MAX_G1_RETRIES=${MAX_G1_RETRIES:-3}
MAX_RETRIES=${MAX_RETRIES:-3}
M=${M:-4096}; N=${N:-1024}; K=${K:-8192}
LAYOUT=${LAYOUT:-crr}
CELL=${CELL:-70b_kv_crr}
N_PAIRS=${N_PAIRS:-3}
BASE_SO=${BASE_SO:-/tmp/r37a_baseline_70bkv.so}
PROD_SO=${PROD_SO:-/tmp/r37a_prod_70bkv.so}
OUTSUBDIR=${OUTSUBDIR:-r38_runs}
BENCH_SCRIPT=${BENCH_SCRIPT:-r35_reviewer_bench5x.py}

OUTDIR="$HERE/$OUTSUBDIR"
mkdir -p "$OUTDIR"

OUT="$OUTDIR/${CELL}_gpu${PHYS_GPU}_orchestrate.log"
SAMPLES_JSON="$OUTDIR/${CELL}_gpu${PHYS_GPU}_samples.json"

echo "===== R38 Dev D orchestrate cell=$CELL gpu=$PHYS_GPU layout=$LAYOUT M=$M N=$N K=$K =====" | tee "$OUT"
echo "[r38_orchestrate] G1+G2a+G2b primary; G1' fallback (bench_mhz>=${SCLK_GATE_MHZ} AND median>${G1P_MIN_MEDIAN_TF})" | tee -a "$OUT"
echo "[r38_orchestrate] N_PAIRS=$N_PAIRS BASE_SO=$BASE_SO PROD_SO=$PROD_SO" | tee -a "$OUT"
date | tee -a "$OUT"

echo "[pre-orchestrate sclk check]" | tee -a "$OUT"
rocm-smi --showclocks -d $PHYS_GPU 2>&1 | grep -i sclk | head -3 | tee -a "$OUT"

extract_mhz() { echo "$1" | sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p'; }

# Initialize JSON output
echo '{"cell":"'$CELL'","gpu":'$PHYS_GPU',"M":'$M',"N":'$N',"K":'$K',"samples":[' > "$SAMPLES_JSON"
first_sample=1

emit_sample() {
  # args: tag rep gate_path attempt md5 median mean stdev cv preMhz benchMhz
  local tag=$1 rep=$2 gate_path=$3 attempt=$4 md5=$5 med=$6 mean=$7 stdev=$8 cv=$9 preMhz=${10} benchMhz=${11}
  if [ $first_sample -eq 0 ]; then echo "," >> "$SAMPLES_JSON"; fi
  first_sample=0
  printf '{"tag":"%s","rep":%d,"gate_path":"%s","attempt":%d,"md5":"%s","median":%s,"mean":%s,"stdev":%s,"cv":%s,"preMhz":%s,"benchMhz":%s}' \
    "$tag" "$rep" "$gate_path" "$attempt" "$md5" "${med:-null}" "${mean:-null}" "${stdev:-null}" "${cv:-null}" "${preMhz:-null}" "${benchMhz:-null}" >> "$SAMPLES_JSON"
}

# Run a single bench attempt and populate ATT_* shell vars.
do_one_attempt() {
  local so_path=$1 tag=$2 rep=$3 attempt=$4
  cp -f "$so_path" "$HERE/tk_mxfp8_layouts.cpython-310-x86_64-linux-gnu.so"
  local md5=$(md5sum "$HERE/tk_mxfp8_layouts.cpython-310-x86_64-linux-gnu.so" | awk '{print $1}')
  local BENCH_OUT="$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_attempt${attempt}.txt"
  local BENCH_ERR="$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_attempt${attempt}.err"
  ROCR_VISIBLE_DEVICES=$PHYS_GPU HIP_VISIBLE_DEVICES=0 PHYS_GPU=$PHYS_GPU \
    N_RUNS=$N_RUNS WARMUP=$WARMUP ITERS=$ITERS \
    python3 "$BENCH_SCRIPT" mxfp8 $LAYOUT $M $N $K \
    > "$BENCH_OUT" 2> "$BENCH_ERR"
  local brc=$?
  ATT_BENCH_OUT="$BENCH_OUT"
  ATT_BENCH_ERR="$BENCH_ERR"
  ATT_RC=$brc
  ATT_MD5=$md5
  ATT_PREMHZ=$(extract_mhz "$(grep 'sclk-post-preheat' "$BENCH_ERR" | head -1)")
  ATT_BENCHMHZ=$(extract_mhz "$(grep 'sclk-post-bench' "$BENCH_ERR" | head -1)")
  local SUMMARY_LINE=$(grep "^SUMMARY" "$BENCH_OUT" | head -1)
  ATT_MEAN=$(echo "$SUMMARY_LINE" | sed -nE 's/.*mean=([0-9.]+).*/\1/p')
  ATT_STDEV=$(echo "$SUMMARY_LINE" | sed -nE 's/.*stdev=([0-9.]+).*/\1/p')
  ATT_MED=$(echo "$SUMMARY_LINE" | sed -nE 's/.*median=([0-9.]+).*/\1/p')
  ATT_CV=""
  if [ -n "$ATT_MEAN" ] && [ -n "$ATT_STDEV" ] && [ "$ATT_MEAN" != "0" ]; then
    ATT_CV=$(python3 -c "print(f'{$ATT_STDEV/$ATT_MEAN:.6f}')")
  fi
  ATT_G1=0; ATT_G2A=0; ATT_G2B=0; ATT_G1P=0
  [ -n "$ATT_PREMHZ" ] && [ "$ATT_PREMHZ" -ge "$SCLK_GATE_MHZ" ] && ATT_G1=1
  [ -n "$ATT_BENCHMHZ" ] && [ "$ATT_BENCHMHZ" -ge "$SCLK_GATE_MHZ" ] && ATT_G2A=1
  [ -n "$ATT_CV" ] && ATT_G2B=$(python3 -c "print(1 if $ATT_CV <= $STDEV_RATIO_MAX else 0)")
  if [ -n "$ATT_BENCHMHZ" ] && [ "$ATT_BENCHMHZ" -ge "$SCLK_GATE_MHZ" ] && [ -n "$ATT_MED" ]; then
    ATT_G1P=$(python3 -c "print(1 if $ATT_MED > $G1P_MIN_MEDIAN_TF else 0)")
  fi
}

# Run a (tag, rep) and attempt the gate ladder.
run_with_gate_ladder() {
  local so_path=$1 tag=$2 rep=$3
  local best_g1p_attempt=0
  local best_g1p_med="" best_g1p_mean="" best_g1p_stdev="" best_g1p_cv="" best_g1p_pre="" best_g1p_bench="" best_g1p_md5=""
  local attempt=1
  while [ $attempt -le $MAX_RETRIES ]; do
    do_one_attempt "$so_path" "$tag" "$rep" "$attempt"
    echo "[$tag rep=$rep att=$attempt md5=${ATT_MD5:0:8} rc=$ATT_RC med=${ATT_MED:-NA} cv=${ATT_CV:-NA} G1=$ATT_G1 G2A=$ATT_G2A G2B=$ATT_G2B G1P=$ATT_G1P preMhz=${ATT_PREMHZ:-NA} benchMhz=${ATT_BENCHMHZ:-NA}]" | tee -a "$OUT"

    if [ $ATT_RC -eq 0 ] && [ $ATT_G1 -eq 1 ] && [ $ATT_G2A -eq 1 ] && [ $ATT_G2B -eq 1 ]; then
      echo "  -> ACCEPT (gate_path=G1+G2a+G2b, attempt=$attempt)" | tee -a "$OUT"
      cp "$ATT_BENCH_OUT" "$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_clean.txt"
      cp "$ATT_BENCH_ERR" "$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_clean.err"
      echo "$ATT_MED" > "$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_clean.med"
      emit_sample "$tag" "$rep" "G1+G2a+G2b" "$attempt" "$ATT_MD5" "$ATT_MED" "$ATT_MEAN" "$ATT_STDEV" "$ATT_CV" "$ATT_PREMHZ" "$ATT_BENCHMHZ"
      return 0
    fi

    # Track best G1' candidate (G2a+G2b pass, median sane) as we go
    if [ $ATT_RC -eq 0 ] && [ $ATT_G2A -eq 1 ] && [ $ATT_G2B -eq 1 ] && [ $ATT_G1P -eq 1 ]; then
      best_g1p_attempt=$attempt
      best_g1p_med="$ATT_MED"; best_g1p_mean="$ATT_MEAN"; best_g1p_stdev="$ATT_STDEV"
      best_g1p_cv="$ATT_CV"; best_g1p_pre="$ATT_PREMHZ"; best_g1p_bench="$ATT_BENCHMHZ"
      best_g1p_md5="$ATT_MD5"
      cp "$ATT_BENCH_OUT" "$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_g1pcandidate.txt"
      cp "$ATT_BENCH_ERR" "$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_g1pcandidate.err"
    fi

    sleep 5
    attempt=$((attempt+1))
  done

  # Exhausted MAX_RETRIES without full G1 pass. If we have a G1' candidate, accept it.
  if [ $best_g1p_attempt -gt 0 ]; then
    echo "  -> ACCEPT G1' FALLBACK (gate_path=G1'_fallback, attempt=$best_g1p_attempt) — G1 over-rejected, G2a+G2b+median-sane all pass" | tee -a "$OUT"
    cp "$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_g1pcandidate.txt" "$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_clean.txt"
    cp "$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_g1pcandidate.err" "$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_clean.err"
    echo "$best_g1p_med" > "$OUTDIR/${tag}_rep${rep}_gpu${PHYS_GPU}_clean.med"
    emit_sample "$tag" "$rep" "G1'_fallback" "$best_g1p_attempt" "$best_g1p_md5" "$best_g1p_med" "$best_g1p_mean" "$best_g1p_stdev" "$best_g1p_cv" "$best_g1p_pre" "$best_g1p_bench"
    return 0
  fi

  echo "  -> EXHAUSTED (gate_path=EXHAUSTED) — no acceptable run after $MAX_RETRIES attempts" | tee -a "$OUT"
  emit_sample "$tag" "$rep" "EXHAUSTED" "$MAX_RETRIES" "$ATT_MD5" "$ATT_MED" "$ATT_MEAN" "$ATT_STDEV" "$ATT_CV" "$ATT_PREMHZ" "$ATT_BENCHMHZ"
  return 1
}

# BABA pattern: prod, base, prod, base, ... (N_PAIRS prod and N_PAIRS base)
SEQ=()
for ((i=0; i<N_PAIRS; i++)); do SEQ+=("prod" "base"); done
for i in "${!SEQ[@]}"; do
  case "${SEQ[$i]}" in
    prod) run_with_gate_ladder "$PROD_SO" prod $i ;;
    base) run_with_gate_ladder "$BASE_SO" base $i ;;
  esac
done

# Close JSON
echo '' >> "$SAMPLES_JSON"
echo ']}' >> "$SAMPLES_JSON"

# Quick gate-path summary
G1_COUNT=$(grep "gate_path...G1+G2a+G2b" "$SAMPLES_JSON" 2>/dev/null | wc -l)
G1P_COUNT=$(grep "_fallback" "$SAMPLES_JSON" 2>/dev/null | wc -l)
EX_COUNT=$(grep "EXHAUSTED" "$SAMPLES_JSON" 2>/dev/null | wc -l)
echo "[done gpu=$PHYS_GPU] G1+G2a+G2b=$G1_COUNT  G1'_fallback=$G1P_COUNT  EXHAUSTED=$EX_COUNT" | tee -a "$OUT"
date | tee -a "$OUT"
