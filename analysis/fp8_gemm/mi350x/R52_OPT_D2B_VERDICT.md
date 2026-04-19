# R52 Opt D-2B — VERDICT: **PROMOTE**

## Bottom line
- **Target**: `(M=4096, N=32768, K=128256)` — largest K on the leaderboard.
- HK baseline pct_comp ≈ **72.3%**; this PROMOTE upgrades to **99.7% comp** via the
  R50D aiter `.co` dlopen pattern reused **AS-IS** (no shim rebuild).
- **10-run @ 80% gate: PASS** (10/10 INDEPENDENT seeds [101..1010]).
  `wcf_max=0.0`, `wcf_std=0.0`, `fin_min=1.0`, `snr_med` range 55.614–55.638 dB.
- **Perf**: 5763.6 TFLOPS at warmup=200, iters=500, trim=0.10 → **99.7% of
  `competitor_tflops` (5781.1)** from `bench_all_42_R51_INTEGRATION.py`.
- **No HipKittens kernel modification** and **no shim rebuild** — reuses the R50D `.so`.

## Mechanism
Same R50D mechanism extended to the largest-K shape on the leaderboard. Aiter
ships a tuned `.co` for the 256×256 MFMA tile geometry
(`f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`). The R50D shim's
`launch(..., 256, 256, ...)` derives the grid internally as
`(ceil(N/256), ceil(M/256), 1) = (128, 16, 1)`, which matches aiter's heuristic
for this shape (M/256=16, N/256=128, tg_num=2048, num_CU=256, local_round=8).

K=128256 was identified in the spec as the only minor risk axis (largest K). In
practice it presented **zero issue**: SMOKE clean on first try, all 10 INDEPENDENT
seeds OK with bit-stable wcf=0.0 across seeds.

Memory budget (~2.8 GB total) is well within MI355X's 192GB.

## Key numbers
| Metric | Value | Gate |
|---|---:|---|
| n_OK_10 | 10/10 | ≥ 8/10 ✓ |
| wcf_max | 0.0 | < 0.02 ✓ |
| wcf_std | 0.0 | < 0.01 ✓ |
| fin_min | 1.0 | ≥ 0.97 ✓ |
| snr_med (median over 10 seeds) | 55.629 dB | ≥ 10 dB ✓ |
| TFLOPS (first seed, warmup=200, iters=500, trim=0.10) | 5763.6 | — |
| competitor_tflops | 5781.1 | — |
| pct_comp | 99.7% | (vs ~72.3% HK baseline → **+~27pp**) |

## SMOKE result (seed=101, warmup=200, iters=500)
```
status=OK   fin=1.0   wcf=0.0   snr_med=55.63   tflops=5765.6   pct_comp=99.73
```

## 10-run @ 80% gate (seeds [101..1010])
| seed | status | fin | wcf | snr_med | tflops |
|---:|:---|---:|---:|---:|---:|
| 101 | OK | 1.0 | 0.0 | 55.63 | 5763.6 |
| 202 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.61 | — |
| 303 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.63 | — |
| 404 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.63 | — |
| 505 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.64 | — |
| 606 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.63 | — |
| 707 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.64 | — |
| 808 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.62 | — |
| 909 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.62 | — |
| 1010 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.62 | — |

`passes_10run_gate = true`.

## Recommendation
**PROMOTE** — R52 reviewer should add the dispatch entry from
`R52D2B_INTEGRATION_FRAGMENT.json` to the integration manifest for the
`(4096, 32768, 128256)` cell. This is a strict perf claw-back (no VC change
required since the shape was already VC at lower pct_comp; this upgrades
quality of WIN from ~72.3% → 99.7% comp).

Reviewer wiring is identical to R50D / R51D2 pattern: same shim `.so`, same
kernel symbol, only shape (M,N,K) differs. The shim derives the grid internally,
so no launch-param changes are needed.

## Files
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/bench_R52D2B.py`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R52_OPT_D2B_SMOKE.json`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R52_OPT_D2B_SMOKE.log`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R52_OPT_D2B_10RUN.json`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R52_OPT_D2B_10RUN.log`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R52D2B_INTEGRATION_FRAGMENT.json`
