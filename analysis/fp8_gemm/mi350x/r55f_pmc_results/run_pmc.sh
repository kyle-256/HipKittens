#!/bin/bash
# R55F PMC harness — adapted from R54B/run_pmc_v2.sh
# Targets RRR layout (V2 scaled kernel for MXFP8) on 8B Gate/Up + cross-shape cells.
# GPU 5, MXFP8_WARMUP=100 ITERS=200 per task spec, 30s cooldown between rocprof invocations.
set -uo pipefail

WORKSPACE="/shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x/r55f_workspace"
OUTROOT="/shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x/r55f_pmc_results"
export THUNDERKITTENS_ROOT="/shared_nfs/kyle/test/Hipkittens2"
export ROCM_PATH=/opt/rocm

GPU=${HIP_VISIBLE_DEVICES:-5}
WARMUP=${WARMUP:-100}
ITERS=${ITERS:-200}
COOLDOWN=${COOLDOWN:-30}

PMC1="$OUTROOT/pmc_set1.txt"
PMC3="$OUTROOT/pmc_set3.txt"

# Cells: LABEL|M|N|K
# Primary cell first, then cross-shape triangulation
CELLS=(
    "8B_GateUp|4096|14336|4096"
    "8B_QO|4096|4096|4096"
    "70B_QO|4096|8192|8192"
    "70B_Down|4096|8192|28672"
)

build_for_shape() {
    local M=$1 N=$2 K=$3 label=$4 kind=$5
    local shape_dir="$WORKSPACE/builds/${label}"
    mkdir -p "$shape_dir"
    # Symlink source/include files (these are stable)
    for f in "$WORKSPACE"/*.cpp "$WORKSPACE"/*.inc "$WORKSPACE"/*.h \
             "$WORKSPACE/Makefile" "$WORKSPACE/include_link"; do
        [ -e "$f" ] && ln -sfn "$f" "$shape_dir/" 2>/dev/null
    done
    # CRITICAL: COPY test scripts (do not symlink) — Python resolves symlinks
    # for sys.path[0], which would pull a stale tk_*_layouts.so from the
    # parent mi350x dir owned by another agent. See R55F findings.
    for f in test_python.py test_mxfp8_python.py; do
        if [ ! -f "$shape_dir/$f" ] || [ -L "$shape_dir/$f" ]; then
            rm -f "$shape_dir/$f"
            cp "$WORKSPACE/$f" "$shape_dir/$f" 2>/dev/null || \
              cp "$THUNDERKITTENS_ROOT/analysis/fp8_gemm/mi350x/r54b_workspace/$f" "$shape_dir/$f"
        fi
    done

    cd "$shape_dir"
    if [ "$kind" = "fp8" ]; then
        local SO="tk_fp8_layouts.cpython-310-x86_64-linux-gnu.so"
        local LOG="$OUTROOT/${label}_${kind}_build.log"
        if [ -f "$SO" ] && [ -f "${SO}.shape" ] && [ "$(cat ${SO}.shape)" = "${M}x${N}x${K}" ]; then
            echo "  cached fp8 build for $label"
            return 0
        fi
        rm -f "$SO" "${SO}.shape"
        make TARGET=tk_fp8_layouts SRC=kernel_fp8_layouts.cpp \
            CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" > "$LOG" 2>&1
        if [ -f "$SO" ]; then echo "${M}x${N}x${K}" > "${SO}.shape"; fi
    else
        local SO="tk_mxfp8_layouts.cpython-310-x86_64-linux-gnu.so"
        local LOG="$OUTROOT/${label}_${kind}_build.log"
        if [ -f "$SO" ] && [ -f "${SO}.shape" ] && [ "$(cat ${SO}.shape)" = "${M}x${N}x${K}" ]; then
            echo "  cached mxfp8 build for $label"
            return 0
        fi
        rm -f "$SO" "${SO}.shape"
        make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
            CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" > "$LOG" 2>&1
        if [ -f "$SO" ]; then echo "${M}x${N}x${K}" > "${SO}.shape"; fi
    fi
}

run_pmc() {
    local kind=$1 label=$2 M=$3 N=$4 K=$5 pmcset=$6 pmcfile=$7
    local outdir="$OUTROOT/${label}_${kind}_rrr_${pmcset}"
    mkdir -p "$outdir"
    if [ -f "$outdir/pmc_5/pmc_counter_collection.csv" ]; then
        echo "  skip $kind $label $pmcset (already have pmc_5 csv)"
        return
    fi
    local shape_dir="$WORKSPACE/builds/${label}"
    cd "$shape_dir"
    echo "  running $kind $label $pmcset"
    if [ "$kind" = "fp8" ]; then
        PYTHONPATH=. HIP_VISIBLE_DEVICES=$GPU \
        FP8_BUILD_M=$M FP8_BUILD_N=$N FP8_BUILD_K=$K \
        FP8_WARMUP=$WARMUP FP8_ITERS=$ITERS FP8_CHECK=0 FP8_LAYOUTS=rrr \
        rocprofv3 -i "$pmcfile" --output-format csv -d "$outdir" \
            -- python3 test_python.py $M $N $K \
            > "$outdir/run.log" 2>&1
    else
        PYTHONPATH=. HIP_VISIBLE_DEVICES=$GPU \
        MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
        MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
        MXFP8_LAYOUTS=rrr MXFP8_PRESHUFFLE_QUANT=1 \
        MXFP8_RRR_PRESHUFFLE_V2_RUNTIME=1 \
        rocprofv3 -i "$pmcfile" --output-format csv -d "$outdir" \
            -- python3 test_mxfp8_python.py $M $N $K \
            > "$outdir/run.log" 2>&1
    fi
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "    ABORT $kind $label $pmcset rc=$rc"
        tail -15 "$outdir/run.log"
    fi
}

for CELL in "${CELLS[@]}"; do
    IFS='|' read -r LABEL M N K <<< "$CELL"
    echo "=== $LABEL ${M}x${N}x${K} ==="
    build_for_shape $M $N $K $LABEL fp8
    build_for_shape $M $N $K $LABEL mxfp8
    run_pmc fp8 $LABEL $M $N $K set1 "$PMC1"
    sleep $COOLDOWN
    run_pmc fp8 $LABEL $M $N $K set3 "$PMC3"
    sleep $COOLDOWN
    run_pmc mxfp8 $LABEL $M $N $K set1 "$PMC1"
    sleep $COOLDOWN
    run_pmc mxfp8 $LABEL $M $N $K set3 "$PMC3"
    sleep $COOLDOWN
done

echo "DONE"
