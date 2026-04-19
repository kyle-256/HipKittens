#!/bin/bash
# R45 Full bench: perf + correctness + determinism for all compute-bound shapes
set -euo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/Hipkittens2

GPU=${HIP_VISIBLE_DEVICES:-2}

declare -a SHAPES=("8192 8192 8192" "4096 4096 4096" "4096 14336 4096" "4096 4096 14336" "4096 8192 8192" "4096 28672 8192" "4096 8192 28672")
declare -a LABELS=("8192cube" "8B_QO" "8B_GateUp" "8B_Down" "70B_QO" "70B_GateUp" "70B_Down")

OUTDIR="r45_full_results"
mkdir -p "$OUTDIR"

for i in "${!SHAPES[@]}"; do
    read -r M N K <<< "${SHAPES[$i]}"
    L="${LABELS[$i]}"
    echo ""
    echo "=== [$((i+1))/${#SHAPES[@]}] $L (${M}x${N}x${K}) ==="

    # Build FP8
    rm -f tk_fp8_layouts*.so
    make TARGET=tk_fp8_layouts SRC=kernel_fp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" > /dev/null 2>&1
    echo "  FP8 built"

    HIP_VISIBLE_DEVICES=$GPU \
    FP8_BUILD_M=$M FP8_BUILD_N=$N FP8_BUILD_K=$K \
    FP8_WARMUP=100 FP8_ITERS=200 FP8_CHECK=0 FP8_LAYOUTS=rcr,rrr,crr \
    python3 test_python.py $M $N $K > "$OUTDIR/${L}_fp8.log" 2>&1
    echo "  FP8 benched"

    # Build MXFP8
    rm -f tk_mxfp8_layouts*.so
    make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
        CXXFLAGS="-w -DM_DIM=$M -DN_DIM=$N -DK_DIM=$K" > /dev/null 2>&1
    echo "  MXFP8 built"

    HIP_VISIBLE_DEVICES=$GPU \
    MXFP8_BUILD_M=$M MXFP8_BUILD_N=$N MXFP8_BUILD_K=$K \
    MXFP8_WARMUP=100 MXFP8_ITERS=200 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 \
    MXFP8_SNR_THRESHOLD_DB=45 MXFP8_LAYOUTS=rcr,rrr,crr MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py $M $N $K > "$OUTDIR/${L}_mxfp8.log" 2>&1
    echo "  MXFP8 benched"
done

echo ""
echo "========== RESULTS =========="

python3 - "$OUTDIR" <<'PYEOF'
import sys, re, os
D = sys.argv[1]

names = {
    "8192cube": "8192³",
    "8B_QO": "8B Q/O (4096³)",
    "8B_GateUp": "8B Gate/Up",
    "8B_Down": "8B Down",
    "70B_QO": "70B Q/O",
    "70B_GateUp": "70B Gate/Up",
    "70B_Down": "70B Down",
}
order = list(names.keys())

def get_tf(path, lay):
    try:
        t = open(path).read()
        m = re.search(rf'---\s+{lay.upper()} Layout.*?TFLOPS:\s+([0-9.]+)', t, re.DOTALL)
        return float(m.group(1)) if m else None
    except: return None

def get_snr(path, lay):
    try:
        t = open(path).read()
        # find the section for this layout, then SNR
        sec = re.search(rf'---\s+{lay.upper()} Layout.*?(?=---\s+\w+ Layout|Results saved|$)', t, re.DOTALL)
        if sec:
            m = re.search(r'SNR:\s+([0-9.]+)\s+dB', sec.group())
            return float(m.group(1)) if m else None
    except: return None

def get_det(path, lay):
    try:
        t = open(path).read()
        sec = re.search(rf'---\s+{lay.upper()} Layout.*?(?=---\s+\w+ Layout|Results saved|$)', t, re.DOTALL)
        if sec:
            return "PASS" in re.search(r'Determinism.*?:\s+(PASS|FAIL)', sec.group()).group(1)
    except: return None

print()
print(f"{'Shape':<20} | {'Layout':>6} | {'FP8 TF':>8} {'MX TF':>8} {'MX/FP8':>7} | {'SNR dB':>7} {'Det':>4} | {'Status':>6}")
print("-"*90)

lines = []
for label in order:
    fp = os.path.join(D, f"{label}_fp8.log")
    mx = os.path.join(D, f"{label}_mxfp8.log")
    for lay in ["rcr", "rrr", "crr"]:
        fp_tf = get_tf(fp, lay)
        mx_tf = get_tf(mx, lay)
        snr = get_snr(mx, lay)
        det = get_det(mx, lay)

        ratio = f"{100*mx_tf/fp_tf:.1f}%" if (mx_tf and fp_tf) else "N/A"
        snr_s = f"{snr:.1f}" if snr else "N/A"
        det_s = "OK" if det else ("FAIL" if det is not None else "N/A")

        perf_ok = mx_tf and fp_tf and (mx_tf/fp_tf >= 0.95)
        snr_ok = snr and snr >= 45.0
        det_ok = det is True
        all_ok = perf_ok and snr_ok and det_ok
        status = "PASS" if all_ok else "FAIL"

        line = f"{names[label]:<20} | {lay.upper():>6} | {fp_tf:8.1f} {mx_tf:8.1f} {ratio:>7} | {snr_s:>7} {det_s:>4} | {status:>6}"
        print(line)
        lines.append(line)
    print()

with open(os.path.join(D, "SUMMARY.txt"), "w") as f:
    f.write(f"{'Shape':<20} | {'Layout':>6} | {'FP8 TF':>8} {'MX TF':>8} {'MX/FP8':>7} | {'SNR dB':>7} {'Det':>4} | {'Status':>6}\n")
    f.write("-"*90 + "\n")
    for l in lines:
        f.write(l + "\n")
print(f"\nWritten to {D}/SUMMARY.txt")
PYEOF
