#!/bin/bash
# R52 Dev Q: V2 RRR data-load cachepolicy sweep — strict-SCLK harness.
# Variants:
#   CP0: A=0 (cache_all)     B=0 (cache_all)     [baseline]
#   CP1: A=2 (cache_stream)  B=0 (cache_all)     [R52P primary]
#   CP2: A=1 (cache_global)  B=1 (cache_global)  [GLC both]
#   CP3: A=3 (non_temporal)  B=0 (cache_all)     [A NT]
#   CP4: A=2 (cache_stream)  B=3 (non_temporal)  [control — B NT — should hurt]
# Protocol: 5 runs/cell, 30s cooldown, 60s rebuild cooldown.
# WARMUP=100, ITERS=200 per spec.
# GPU: HIP_VISIBLE_DEVICES=1 (R52Q exclusive).
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"
export ROCM_PATH=/opt/rocm
export PYTHONPATH="$THUNDERKITTENS_ROOT/src/common/pyutils"

GPU=${HIP_VISIBLE_DEVICES:-1}
WARMUP=${MXFP8_WARMUP:-100}
ITERS=${MXFP8_ITERS:-200}
RUNS=${RUNS:-5}
COOL=${COOL:-30}
REBUILD_COOL=${REBUILD_COOL:-60}
OUTDIR="r52q_results"
mkdir -p "$OUTDIR"

# Args: $1=label  $2=M  $3=N  $4=K  $5=tag(cp0..cp4)  $6=A_COH  $7=B_COH
run_cell() {
    local LABEL=$1 M=$2 N=$3 K=$4 TAG=$5 A=$6 B=$7
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K -DMXFP8_RRR_A_COHERENCY=$A -DMXFP8_RRR_B_COHERENCY=$B" \
        > "$OUTDIR/${LABEL}_${TAG}_build.log" 2>&1
    for r in $(seq 1 $RUNS); do
        HIP_VISIBLE_DEVICES=$GPU \
        MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=rrr MXFP8_PRESHUFFLE_QUANT=1 \
        python3 test_mxfp8_python.py $M $N $K \
            > "$OUTDIR/${LABEL}_${TAG}_run${r}.log" 2>&1
        if [ $r -lt $RUNS ]; then sleep $COOL; fi
    done
}

# Args: $1=label  $2=M  $3=N  $4=K  $5=tag  $6=A  $7=B
run_check() {
    local LABEL=$1 M=$2 N=$3 K=$4 TAG=$5 A=$6 B=$7
    HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=20 MXFP8_ITERS=20 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 \
    MXFP8_SNR_THRESHOLD_DB=45 MXFP8_LAYOUTS=rrr MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py $M $N $K \
        > "$OUTDIR/${LABEL}_${TAG}_check.log" 2>&1 || true
}

run_variant_at() {
    local LABEL=$1 M=$2 N=$3 K=$4 TAG=$5 A=$6 B=$7
    echo "  --> $LABEL ($M x $N x $K) RRR  [$TAG: A=$A B=$B]"
    run_cell "$LABEL" "$M" "$N" "$K" "$TAG" "$A" "$B"
    sleep $COOL
    run_check "$LABEL" "$M" "$N" "$K" "$TAG" "$A" "$B"
    echo "      sleep ${REBUILD_COOL}s"
    sleep $REBUILD_COOL
}

run_full_sweep() {
    local LABEL=$1 M=$2 N=$3 K=$4
    run_variant_at "$LABEL" "$M" "$N" "$K" "cp0" 0 0
    run_variant_at "$LABEL" "$M" "$N" "$K" "cp1" 2 0
    run_variant_at "$LABEL" "$M" "$N" "$K" "cp2" 1 1
    run_variant_at "$LABEL" "$M" "$N" "$K" "cp3" 3 0
    run_variant_at "$LABEL" "$M" "$N" "$K" "cp4" 2 3
}

PHASE=${R52Q_PHASE:-primary}

if [ "$PHASE" = "primary" ]; then
    echo "=== R52 Dev Q: V2 RRR cachepolicy sweep — PRIMARY (8B Gate/Up) ==="
    run_full_sweep "8B_GateUp" 4096 14336 4096
elif [ "$PHASE" = "cross" ]; then
    echo "=== R52 Dev Q: V2 RRR cachepolicy cross-cell — K=8192 ==="
    # Cross-cell — only baseline + best variant (TBD post-primary). Default
    # bench cp0 + cp1 here; orchestrator will rerun w/ best variant.
    BEST=${R52Q_BEST:-cp1}
    BEST_A=${R52Q_BEST_A:-2}
    BEST_B=${R52Q_BEST_B:-0}
    run_variant_at "70B_GateUp" 4096 28672 8192 "cp0" 0 0
    run_variant_at "70B_GateUp" 4096 28672 8192 "$BEST" "$BEST_A" "$BEST_B"
    run_variant_at "70B_QO"     4096 8192 8192 "cp0" 0 0
    run_variant_at "70B_QO"     4096 8192 8192 "$BEST" "$BEST_A" "$BEST_B"
fi

echo ""
echo "========== R52 Dev Q — strict-SCLK summary =========="
python3 - "$OUTDIR" <<'PYEOF'
import sys, re, statistics, glob
D = sys.argv[1]
def get_tf(p):
    try:
        t = open(p).read()
        m = re.search(r'TFLOPS:\s+([0-9.]+)', t)
        return float(m.group(1)) if m else None
    except Exception:
        return None
def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return (None, None, None)
    return (statistics.median(xs), min(xs), max(xs))
labels = sorted({f.split("/")[-1].split("_")[0]+"_"+f.split("/")[-1].split("_")[1] for f in glob.glob(f"{D}/*_run*.log")})
tags_all = sorted({f.split("/")[-1].split("_")[2] for f in glob.glob(f"{D}/*_run*.log")})
for lbl in labels:
    print()
    print(f"=== Label: {lbl} ===")
    print(f"{'tag':<6} | {'med':>7} {'min':>7} {'max':>7} {'spread%':>7}  | vs cp0")
    base = None
    for tag in tags_all:
        runs = [get_tf(f"{D}/{lbl}_{tag}_run{r}.log") for r in (1,2,3,4,5)]
        med, lo, hi = stats(runs)
        if med is None:
            continue
        sp = (hi - lo) / med * 100
        if tag == "cp0":
            base = med
            dlt = ""
        else:
            dlt = f"{100*(med-base)/base:+.2f}%" if base else "n/a"
        flag = " *FLAG*" if sp > 5 else ""
        print(f"{tag:<6} | {med:7.1f} {lo:7.1f} {hi:7.1f} {sp:6.2f}%{flag}  | {dlt}")
PYEOF
