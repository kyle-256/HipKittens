#!/bin/bash
# R45 — All compute-bound LLaMA shapes: MXFP8 vs FP8 per-tensor comparison
# Rebuilds .so for each shape (compile-time M_DIM/N_DIM/K_DIM gates)
set -euo pipefail

cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2

GPU=${HIP_VISIBLE_DEVICES:-2}
WARMUP=100
ITERS=200
RESULTS_DIR="r45_shape_comparison"
mkdir -p "$RESULTS_DIR"

# All compute-bound shapes
declare -a SHAPES=(
    "8192 8192 8192"
    "4096 4096 4096"
    "4096 14336 4096"
    "4096 4096 14336"
    "4096 8192 8192"
    "4096 28672 8192"
    "4096 8192 28672"
)

declare -a LABELS=(
    "8192cube"
    "8B_QO"
    "8B_GateUp"
    "8B_Down"
    "70B_QO"
    "70B_GateUp"
    "70B_Down"
)

echo "============================================================"
echo "R45 All-Shape MXFP8 vs FP8 Per-Tensor Comparison"
echo "GPU=$GPU  WARMUP=$WARMUP  ITERS=$ITERS"
echo "============================================================"

for i in "${!SHAPES[@]}"; do
    read -r M N K <<< "${SHAPES[$i]}"
    LABEL="${LABELS[$i]}"
    TAG="${M}x${N}x${K}"

    echo ""
    echo ">>> [$((i+1))/${#SHAPES[@]}] $LABEL ($TAG)"

    # Build FP8
    echo "    [FP8] building ..."
    rm -f tk_fp8_layouts*.so
    make TARGET=tk_fp8_layouts SRC=kernel_fp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" \
        > "$RESULTS_DIR/${LABEL}_fp8_build.log" 2>&1

    echo "    [FP8] benching ..."
    HIP_VISIBLE_DEVICES=$GPU \
    FP8_BUILD_M=$M FP8_BUILD_N=$N FP8_BUILD_K=$K \
    FP8_WARMUP=$WARMUP FP8_ITERS=$ITERS FP8_CHECK=0 FP8_LAYOUTS=rcr,rrr,crr \
    python3 test_python.py $M $N $K \
        > "$RESULTS_DIR/${LABEL}_fp8.log" 2>&1

    # Build MXFP8
    echo "    [MXFP8] building ..."
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" \
        > "$RESULTS_DIR/${LABEL}_mxfp8_build.log" 2>&1

    echo "    [MXFP8] benching ..."
    HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=$WARMUP MXFP8_ITERS=$ITERS MXFP8_CHECK=0 \
    MXFP8_LAYOUTS=rcr,rrr,crr MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py $M $N $K \
        > "$RESULTS_DIR/${LABEL}_mxfp8.log" 2>&1

    echo "    done."
done

echo ""
echo "============================================================"
echo "Parsing results..."
echo "============================================================"

python3 - "$RESULTS_DIR" <<'PYEOF'
import sys, re, os
results_dir = sys.argv[1]

labels_full = {
    "8192cube": "8192³ (default)",
    "8B_QO": "8B Q/O (4096³)",
    "8B_GateUp": "8B Gate/Up (4096×14336×4096)",
    "8B_Down": "8B Down (4096×4096×14336)",
    "70B_QO": "70B Q/O (4096×8192×8192)",
    "70B_GateUp": "70B Gate/Up (4096×28672×8192)",
    "70B_Down": "70B Down (4096×8192×28672)",
}

order = ["8192cube", "8B_QO", "8B_GateUp", "8B_Down",
         "70B_QO", "70B_GateUp", "70B_Down"]

def extract_tflops(path, layout):
    try:
        with open(path) as f:
            text = f.read()
        pattern = rf'---\s+{layout.upper()} Layout.*?TFLOPS:\s+([0-9.]+)'
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return float(m.group(1))
    except:
        pass
    return None

print()
hdr = f"{'Shape':<35} | {'FP8 RCR':>8} {'FP8 RRR':>8} {'FP8 CRR':>8} | {'MX RCR':>8} {'MX RRR':>8} {'MX CRR':>8} | {'RCR%':>6} {'RRR%':>6} {'CRR%':>6}"
sep = "-"*130
print(hdr)
print(sep)

summary_lines = [hdr, sep]

for label in order:
    fp8_path = os.path.join(results_dir, f"{label}_fp8.log")
    mx_path = os.path.join(results_dir, f"{label}_mxfp8.log")

    fp8_rcr = extract_tflops(fp8_path, "rcr")
    fp8_rrr = extract_tflops(fp8_path, "rrr")
    fp8_crr = extract_tflops(fp8_path, "crr")
    mx_rcr = extract_tflops(mx_path, "rcr")
    mx_rrr = extract_tflops(mx_path, "rrr")
    mx_crr = extract_tflops(mx_path, "crr")

    def pct(mx, fp):
        if mx and fp and fp > 0:
            v = 100*mx/fp
            flag = " OK" if v >= 95.0 else " XX"
            return f"{v:.1f}%{flag}"
        return "N/A"

    def tf(v):
        return f"{v:.1f}" if v else "FAIL"

    line = f"{labels_full[label]:<35} | {tf(fp8_rcr):>8} {tf(fp8_rrr):>8} {tf(fp8_crr):>8} | {tf(mx_rcr):>8} {tf(mx_rrr):>8} {tf(mx_crr):>8} | {pct(mx_rcr,fp8_rcr):>9} {pct(mx_rrr,fp8_rrr):>9} {pct(mx_crr,fp8_crr):>9}"
    print(line)
    summary_lines.append(line)

print()

# Write summary
with open(os.path.join(results_dir, "SUMMARY.txt"), "w") as f:
    f.write("\n".join(summary_lines) + "\n")
print(f"Written to {results_dir}/SUMMARY.txt")
PYEOF
