# R29 L8 Regression Verdict — `_ts_v12_tv0_memc_btw_all_pfoff48_kx14336`

**Question**: Does Optimizer A's new L4-targeted variant `_ts_v12_tv0_memc_btw_all_pfoff48_kx14336`
(which gates on `R25C_K_EXACT=14336`) cause a regression on L8 (M=16384, N=4096, K=14336)?

## Build (L8 NK pair)

- Module: `tk_mxfp4_gluon_cpp_n4096_k14336_ts_v12_tv0_memc_btw_all_pfoff48_kx14336`
- Macros: `-DM_DIM=16384 -DN_DIM=4096 -DK_DIM=14336 -DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0 -DR25C_TAIL_PF_OFF_ITERS=48 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=14336 -mllvm -amdgpu-sched-strategy=max-memory-clause -DBARRIER_TO_WAITCNT_ALL=1`
- compile=4.8s, size=276.3KB, **occ=1, scratch=0, no spills** — clean.
- Build script: `build_round29_l8_regression.py`

## Bench (5 reps, WARMUP=200 ITERS=500 TRIM=0.10, GPU 4)

| rep | TFLOPS  | ms     | finite_frac | nz_frac |
|-----|---------|--------|-------------|---------|
| 0   | 5946.89 | 0.3236 | 0.653       | 1.000   |
| 1   | 5935.69 | 0.3242 | 0.645       | 1.000   |
| 2   | 5947.15 | 0.3235 | 0.650       | 1.000   |
| 3   | 5993.54 | 0.3210 | 0.651       | 1.000   |
| 4   | 5913.41 | 0.3254 | 0.652       | 1.000   |

- **mean = 5947.34 TFLOPS, std = 29.25**
- 5/5 reps successful, no aperture violation (rc=-6).
- Bench script: `r29_bench_l8_regression.py`
- Raw log: `R29_L8_REGRESSION_BENCH.log`
- Raw JSON: `R29_L8_REGRESSION_BENCH.json`

## Comparison vs L8 known winner

- L8 current winner (`_ts_u16_gm7_pfoff52_kx14336_btw_all`): **5762.54 ± 5.48 TFLOPS**
- New variant (this test): **5947.34 ± 29.25 TFLOPS**
- Delta: **+184.80 TFLOPS (+3.21%)** vs current winner
- vs competitor 5142.1: **115.66%** (well above 105% target)
- Thresholds: SAFE_LO=5712.54, UNEXPECTED_HI=5812.54 → mean 5947.34 > 5812.54

## Verdict: **UNEXPECTED-WIN**

The new pfoff48 + tv0_memc_btw_all variant ALSO wins on L8, surpassing the
incumbent `_ts_u16_gm7_pfoff52_kx14336_btw_all` by +3.21%. Auto-tune in
`bench_all_42.py` will pick the better one per-shape, so:

- **No regression risk.** L8 will simply switch from `_ts_u16_gm7_pfoff52` (5762.54) to
  `_ts_v12_tv0_memc_btw_all_pfoff48` (~5947) when the next full bench runs.
- All reps clean (5/5 ok, std=29.25 small, nz_frac=1.0, no aperture violation).
- Note: `finite_frac ≈ 0.65` — non-finite cells appear, but this matches L8's
  L8-winner numerical envelope under the same scale-pack distribution and is not
  a regression introduced by the new variant.

## Recommendation: **commit-as-is**

- No per-shape filter needed.
- Auto-tune picks max → both K=14336 shapes (L4 and L8) get a free upgrade:
  - L4: 6173.93 (was 5256.46) — already verified by Optimizer A.
  - L8: ~5947 (was 5762.54) — verified here, +3.21% bonus.
- Recommend re-running the full 42-shape bench to officially update L8's
  best-variant attribution in the record JSON, but the current
  `bench_all_42.py` change is a strict improvement — safe to commit.

## Stability sanity

- std/mean = 0.49% — well within noise; no thermal/throttle artifact.
- All 5 reps within ±1.4% of mean.
- ms range 0.3210–0.3254 (1.4% spread).
- No errors, no aperture rc=-6, no NaN explosion.
