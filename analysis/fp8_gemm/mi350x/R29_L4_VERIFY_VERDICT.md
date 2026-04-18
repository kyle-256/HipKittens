# R29 L4 Verification Verdict

**Shape**: L4 = 4096 x 32768 x 14336
**Variant**: `ts_v12_tv0_memc_btw_all`
**SO**: `tk_mxfp4_gluon_cpp_n32768_k14336_ts_v12_tv0_memc_btw_all.cpython-310-x86_64-linux-gnu.so`
**Setup**: GPU 0 (idle), WARMUP=200, ITERS=500, TRIM=0.10, REPS=5

## Reps (TFLOPS)
| rep | TFLOPS  | ms     |
|-----|---------|--------|
| 0   | 5248.74 | 0.7332 |
| 1   | 5269.55 | 0.7303 |
| 2   | 5263.67 | 0.7311 |
| 3   | 5255.91 | 0.7322 |
| 4   | 5244.41 | 0.7338 |

## Statistics
- **mean**: 5256.46 TFLOPS
- **std**:  10.35 TFLOPS
- **min**:  5244.41
- **max**:  5269.55
- **stability (std/mean)**: 0.197% (target <1%, well within)

## Comparison vs competitor (5296.1)
- delta: -39.64 TFLOPS
- ratio: **99.25%** (was reported as 98.5% in v2 single-rep)
- mean (5256.46) < comp - 1*std (5296.1 - 10.35 = 5285.75)

## VERDICT: LOSE (stable)

The result is stable LOSE — the deficit (-39.64 TFLOPS) is ~3.8x the std (10.35),
far outside the noise band. Stability is excellent (0.197% std/mean), so this is
NOT a noise artifact. The v2 single-rep at 5217 (98.5%) was on the low end of the
distribution; the true mean is ~5256 (99.25%), still a clear LOSE.

Note: also checked `min_finite_frac=0.319, min_nz_frac=1.000` — the low finite_frac
is consistent across reps (random fp4 inputs producing many inf/nan results); this
is the same numeric regime as other shapes and not a kernel bug.

## Recommendation
- Do **NOT** flag for re-bench — the v2 reading was directionally correct.
- L4 remains a true LOSE shape (~0.75% gap to comp). It is small enough that L4
  belongs with the residual "gap-close" candidates rather than the noise-band wins.
- Suggest investigating L4-specific tile/pfoff variants (similar in spirit to R28-C
  for L8) before declaring saturated.
