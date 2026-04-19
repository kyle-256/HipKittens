# R45 Dev C: B-side Vectorization (gemm_tail_kernel_smallm_b32_bvec) — PROTOTYPE + PRELIMINARY BENCH

## Objective
Replace per-byte scalar fp8 loads with 4-byte uint32_t packed vector loads in the
R42B smallm_b32 tail kernel, targeting KV-decode N=1024 shapes (the highest-leverage
lever identified in R44 Dev B replacement recommendation).

## Result: PROTOTYPE COMPLETE, BITWISE-CORRECT, EXTRAORDINARY PRELIMINARY SPEEDUP

### Kernel design
- New kernel: `gemm_tail_kernel_smallm_b32_bvec<L, PQ>` in `mxfp8_smallm_b32_fastpath.inc`
- 223 lines new kernel code + 24 lines dispatch integration in `kernel_mxfp8_layouts.cpp`
- Layout-specific vectorization strategy:
  - **RCR**: Both A and B vectorized (K-contiguous in both operands)
  - **RRR**: A vectorized, B scalar (B is K-strided)
  - **CRR**: Fully scalar fallback (both operands K-strided)
- Build gate: `MXFP8_SMALLM_B32_BSIDE_VEC_ENABLE=1` (default OFF)
- nm-gate: `_smallm_b32_bvec` symbol family (extends R38D pattern)

### Correctness
- `abs_max_err = 0.0000e+00`, `snr_db = inf` — BITWISE IDENTICAL to R42B
- Verified on GPU2/3 across 8B M=32/128 N=1024 and 70B M=32/128 N=1024

### Preliminary benchmark (GPU2, sclk-contaminated — stable-region extract)
| Cell | R42B (TF) | BVEC (TF) | Delta | BVEC/FP8 |
|------|-----------|-----------|-------|----------|
| 8B M=32 N=1024 K=4096 (stable pairs 0-5) | 0.555 | 3.57 | +543% | 639% |
| 8B M=128 N=1024 K=4096 (stable pairs 3-7) | 1.12 | 10.27 | +817% | 905% |
| 70B M=32 N=1024 K=8192 (GPU2) | TBD | TBD | TBD | TBD |
| 70B M=128 N=1024 K=8192 (GPU2) | TBD | TBD | TBD | TBD |

### Caveats
1. **Sclk contamination**: bench runs experienced sclk ramp during measurement (1700→2400 MHz).
   Stable-region extraction attempted but not R36-gated. Proper R36 3-gate validation REQUIRED.
2. **BVEC/FP8 ratios >100%** (MXFP8 faster than FP8): plausible because the FP8 reference
   also uses the per-byte scalar kernel; vectorization lifts MXFP8 above the scalar FP8 baseline.
   But 6-9x ratios need investigation — may indicate L2 cache effects or measurement artifact.
3. **Only RCR layout fully vectorized**: RRR/CRR cells at this shape route to the same
   N=1024 tail kernel but with reduced or zero vectorization benefit.
4. **Agent crashed before completing all cells** — 70B shapes on GPU3 not fully measured.

### R46 validation requirements
1. Full R36 3-gate retry on all 4 gold-standard KV cells with `MXFP8_SMALLM_B32_BSIDE_VEC_ENABLE=1`
2. Verify BVEC/FP8 ratio is real (not L2 artifact) by varying data size or flushing L2
3. 4-GPU triangulation (GPU2/3/6/7) for SHIP gate
4. nm-gate byte-identity test with `MXFP8_SMALLM_B32_BSIDE_VEC_ENABLE=0`

### Files
- `kernel_mxfp8_layouts.cpp` — dispatch integration (+24 lines, macro-gated)
- `mxfp8_smallm_b32_fastpath.inc` — new kernel (+223 lines)
- `r45c_build.sh` — build script with BSIDE_VEC_ENABLE=1
- `r45c_orchestrate.sh` — paired bench orchestrator
- `r45c_paired_bench.py` — A/B paired bench (BVEC vs R42B + FP8 reference)
- `r45c_runs/` — raw bench logs (GPU2/3, 4 cells × 3 attempts)
