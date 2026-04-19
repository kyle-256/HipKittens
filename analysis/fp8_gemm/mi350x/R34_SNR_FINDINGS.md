# R34 SNR Validation — Findings (2026-04-19)

## Summary
Attempted full 42-shape SNR validation vs torch float32 reference. Discovered the
kernel has **structural correctness issues** AND the SNR-via-deterministic-cells
methodology is **not robust** beyond a narrow setup.

## What works
At a single specific configuration (M=4096, N=4096, K=2048, n_runs=5, ext_br variant)
the consistent-cells SNR is **~47 dB** — the only reproducible PASS we have.
- ref_finite=100%, kernel_finite=88%, det_frac=60-65%
- SNR_det=47.06-47.82 dB across {zero, const_-4, rand_-1_2, rand_-2_3} scale modes
- Confirms FP4 E2M1 dequant table + E8M0 scale interpretation are CORRECT

## What breaks
1. **All M > 4096** — same kernel, same N, same K, same scales, but SNR_det collapses to -600 to -700 dB (max|diff|=3.39e+38). Even cells consistent across 5 runs disagree from torch reference by bf16-overflow magnitudes.
2. **All n_runs < 5** — 3-run consistent cells include race-condition borderline cells with 4e+36 errors.
3. **All R25_FINAL_v2 BEST_VARIANTS** at their bench shapes — the ts_v12_gm7_*, ts_lgk2_gm6_*, etc. variants all fail SNR even when called at their compiled (M, N, K).

## Diagnosis
The kernel produces **two classes of "deterministic" cells**:
- ~70% truly stable cells → match torch reference within 47 dB
- ~17% "deterministically wrong" cells → kernel writes huge garbage values consistently across runs (bf16-max territory, ±3.39e38)
- ~13% non-deterministic cells (race conditions)

The 17% "deterministic wrong" tier is the killer:
- Cannot be filtered by run-consistency
- Magnitude swamps the signal in any aggregate metric (SNR collapses)
- Has been present across all rounds R25-R34 (visible in finite_frac < 90%)

## Implication for benchmarking
- Performance numbers from `bench_all_42.py` measure **wall-clock time of whatever the kernel computes**, including the ~17% wrong cells
- Without a true correctness reference, TFLOPS comparisons vs aiter assume aiter's output is the contract
- The 41/42 win record is **conditional on tolerating ~17% incorrect output**

## What's actually needed
A proper correctness gate would require:
1. Build with explicit M_DIM macro (does the kernel use it?)
2. Cell-by-cell accumulator-precision study to identify which path produces the wrong values
3. A sane correctness reference (likely aiter binary output, not torch)

## Files
- `snr_diag_random.py` — diagnostic that reproducibly hits 47 dB at M=N=4096
- `snr_all_42_shapes.py` — full 42-shape script (broken methodology, outputs -700 dB)
- `R34_SNR_ALL_42_SHAPES.log` — full output showing methodology break-down
- `R34_DECIDER_VERDICT.md` — original R34 hypothesis (vmcnt(15) via VGPR-PF)
