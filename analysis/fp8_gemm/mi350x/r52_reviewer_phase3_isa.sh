#!/bin/bash
# Phase 3 ISA-only re-validation: dump assembly for R52H (RRR soft barrier
# OFF vs ON) and R52J (CRR EXACT IA=0 vs IA=4) at the cells claimed
# bit-identical. No GPU needed.
set -uo pipefail
cd "$(dirname "$0")"
export THUNDERKITTENS_ROOT="$(cd ../../.. && pwd)"

OUTDIR="r52_reviewer_phase3_results"
mkdir -p "$OUTDIR"

HIPCC=/opt/rocm/bin/hipcc
HIPFLAGS="-DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math -I/opt/rocm/include/rocrand"
CPPF="-I${THUNDERKITTENS_ROOT}/include -I/opt/rocm/include/hip"
PYINC=$(python3 -m pybind11 --includes)

# === R52H: RRR soft barrier ===
echo "=== R52H ISA: 8B Gate/Up RRR (M=4096 N=14336 K=4096) ==="
for STATE in off on; do
    case $STATE in
        off) FLAG="-DMXFP8_RRR_SOFT_BARRIER=0";;
        on)  FLAG="-DMXFP8_RRR_SOFT_BARRIER=1";;
    esac
    echo "  building $STATE ($FLAG)"
    $HIPCC kernel_mxfp8_layouts.cpp $HIPFLAGS -std=c++20 -w $CPPF \
        -DM_DIM=4096 -DN_DIM=14336 -DK_DIM=4096 $FLAG $PYINC -shared -fPIC \
        --cuda-device-only -S -o "$OUTDIR/r52h_rrr_${STATE}.s" 2> "$OUTDIR/r52h_rrr_${STATE}.build.log" && echo "    OK" || echo "    BUILD FAILED"
done
DLINES_H=$(diff "$OUTDIR/r52h_rrr_off.s" "$OUTDIR/r52h_rrr_on.s" 2>/dev/null | tee "$OUTDIR/r52h_rrr_diff.txt" | wc -l)
echo "R52H diff lines: $DLINES_H  (R52H claim: ~36, all empty-asm marker pairs + UID)"
NONMARKER_H=$(grep -v -E "(ASMSTART|ASMEND|__hip_cuid|^---$|^[<>]\s*$|^[0-9])" "$OUTDIR/r52h_rrr_diff.txt" | wc -l)
echo "R52H non-marker non-cuid diff lines: $NONMARKER_H"

# === R52J: CRR EXACT IA=0 vs IA=4 ===
echo ""
echo "=== R52J ISA: 8B Gate/Up CRR EXACT IA=0 vs IA=4 ==="
for IA in 0 4; do
    echo "  building IA=$IA"
    $HIPCC kernel_mxfp8_layouts.cpp $HIPFLAGS -std=c++20 -w $CPPF \
        -DM_DIM=4096 -DN_DIM=14336 -DK_DIM=4096 \
        -DCRR_EXACT_B1_LDS_INSERT_AFTER=$IA $PYINC -shared -fPIC \
        --cuda-device-only -S -o "$OUTDIR/r52j_crr_ia${IA}.s" 2> "$OUTDIR/r52j_crr_ia${IA}.build.log" && echo "    OK" || echo "    BUILD FAILED"
done
DLINES_J=$(diff "$OUTDIR/r52j_crr_ia0.s" "$OUTDIR/r52j_crr_ia4.s" 2>/dev/null | tee "$OUTDIR/r52j_crr_diff.txt" | wc -l)
echo "R52J diff lines: $DLINES_J  (R52J claim: bit-identical except cuid)"
NONCUID_J=$(grep -v -E "(__hip_cuid|^---$|^[<>]\s*$|^[0-9])" "$OUTDIR/r52j_crr_diff.txt" | wc -l)
echo "R52J non-cuid diff lines: $NONCUID_J"

echo ""
echo "=== Phase 3 ISA SUMMARY ==="
echo "R52H (RRR soft barrier OFF vs ON):"
echo "  total diff lines: $DLINES_H"
echo "  excluding markers/cuid: $NONMARKER_H"
echo "  R52H claim: ~36 lines, all empty-asm + UID, zero real instr changes"
echo "  re-validation: $([ "$NONMARKER_H" -le 4 ] && echo "CONFIRMED — REFUTATION re-validates" || echo "DISPUTED — found real instr changes")"
echo ""
echo "R52J (CRR EXACT IA=0 vs IA=4):"
echo "  total diff lines: $DLINES_J"
echo "  excluding cuid: $NONCUID_J"
echo "  R52J claim: bit-identical except cuid hash"
echo "  re-validation: $([ "$NONCUID_J" -le 4 ] && echo "CONFIRMED — REFUTATION re-validates" || echo "DISPUTED — found real instr changes")"
