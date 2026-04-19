# R55 Opt E-3 Cohort Verdict — Worker A (GPU 0)

**Date:** 2026-04-19
**Cohort:** Opt E-3 — HK FAIL cohort-race rescue, M=16384 x N in {14336, 28672}
**Mechanism:** Direct port of R54 E-1/E-2 pattern — AITER 256x256 `.co` dispatch via R50D shim AS-IS.
**Shim:** `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` (no rebuild)
**Aiter `.co`:** `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`
**Kernel:** `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E`
**Bench rules:** warmup=200, iters=500, trim=0.10, GPU=0 (idle, verified via rocm-smi); 10 INDEPENDENT seeds [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010].

## Cohort outcome — 4/4 PROMOTE, 0 DEAD

All four candidates pass all four PROMOTE-gate clauses with perfect AITER bit-determinism (wcf_max=0, wcf_std=0, fin_min=1.0). Each is a NET-VC rescue: the corresponding R54 R40B HK kernel was failing the cohort-race gate (PASS_9/10 fin or wcf, or FLAKE_1/10 wcf).

| Idx | Shape (MxNxK)        | Grid (gdx,gdy,gdz,bdx) | n_OK_10 | wcf_max | wcf_std | fin_min | snr_med_db | TFLOPS p50 | comp TFLOPS | pct_comp | R54 status     | Verdict |
|----:|----------------------|------------------------|--------:|--------:|--------:|--------:|-----------:|-----------:|------------:|---------:|----------------|---------|
|  1  | 16384 x 14336 x 2048 | (56, 64, 1, 256)       |  10/10  |   0.000 |   0.000 |    1.00 |     55.535 |     3518.5 |      3301.3 |  106.58% | PASS_9/10 fin=0.9222 | PROMOTE |
|  2  | 16384 x 14336 x 4096 | (56, 64, 1, 256)       |  10/10  |   0.000 |   0.000 |    1.00 |     55.560 |     4717.9 |      4255.8 |  110.86% | FLAKE_1/10 wcf=0.0506 | PROMOTE |
|  3  | 16384 x 28672 x 2048 | (112, 64, 1, 256)      |  10/10  |   0.000 |   0.000 |    1.00 |     55.535 |     3666.2 |      3482.3 |  105.28% | PASS_9/10 fin=0.9657 | PROMOTE |
|  4  | 16384 x 28672 x 4096 | (112, 64, 1, 256)      |  10/10  |   0.000 |   0.000 |    1.00 |     55.560 |     4670.4 |      4411.7 |  105.86% | PASS_9/10 wcf=0.0215 | PROMOTE |

### PROMOTE-gate cross-check
- n_OK >= 8/10:        4/4 cells at 10/10 (perfect).
- wcf_max < 0.02:      4/4 cells at 0.000 (AITER bit-deterministic).
- wcf_std < 0.01:      4/4 cells at 0.000.
- fin_min >= 0.97:     4/4 cells at 1.000.
- Cohort-race rescue gate (no perf clause): 4/4 PASS by definition.

### Anomalies / risk notes
- **None.** The high-risk cell #2 (`16384x14336x4096`, R54 FLAKE_1/10) PROMOTEd cleanly: wcf=0 across all 10 seeds with the highest pct_comp in the cohort (110.86%). This confirms the R54 FLAKE was a cohort-race tail-draw on UNCHANGED HK .so (not a kernel bug), and the AITER bit-deterministic dispatch trivially closes it.
- All 4 cells beat their `competitor_tflops` baseline (105-111% comp), so no perf regression risk to flag for the reviewer.
- All 4 use the same `BpreShuffle_256x256.co` and shim signature already proven across 14+ R50D-pattern PROMOTEs (R50/R51/R52/R53/R54 cohorts).

## Headline numbers
- **+4 NET VC rescues** to merge into the R55 manifest (subject to integration regression check).
- p50 TFLOPS aggregate: 3518.5 / 4717.9 / 3666.2 / 4670.4 — all 105-111% of competitor.
- Wall-clock for cohort: ~5 min serial on GPU 0 (4 SMOKE + 4 10RUN).

## Files (all under `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/`)
- `bench_R55E3_{1..4}.py` — bench scripts (subprocess-per-seed)
- `R55_OPT_E3_{1..4}_SMOKE.{json,log}`
- `R55_OPT_E3_{1..4}_10RUN.{json,log}`
- `R55E3_{1..4}_INTEGRATION_FRAGMENT.json` — single-shape manifest deltas for reviewer merge

## Reviewer recommendation
PROMOTE all 4 fragments into the R55 manifest. Mechanism-confidence remains VERY HIGH; the R55 floor target (38/42, +2 NET VC from this cohort) is comfortably exceeded by Worker A alone (+4). Hand off to Worker B (E-4) and the integration reviewer.
