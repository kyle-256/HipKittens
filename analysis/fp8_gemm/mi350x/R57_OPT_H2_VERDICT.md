# R57 Cohort H-2 (Opt L) — Verdict

**Worker:** Cohort H-2 (Opt L) — kept-HK-cell hardening via 192×256 alt-tile AITER swap
**GPUs used:** 1, 2, 3 (idle, verified via `rocm-smi`)
**Mechanism:** R55 D-5B/1 found 256×256 AITER underperforms HK on N=14336 (-1.33pp). R57 hypothesis: next-highest-eff non-256×256 tile = 192×256 (eff=109.7) might clear HK + 1.0pp.
**Result:** **0/3 PROMOTE — 3/3 ACCEPT_FALLBACK (DEAD on perf, all clean on correctness).**

## Summary table

| Cand | Shape (M, N, K) | HK pct_comp | R57 (192×256) pct_comp | Δ (pp) | Smoke SNR | wcf | fin | D-3A-1 outcome |
|------|-----------------|------------:|-----------------------:|-------:|----------:|----:|----:|----------------|
| **L-1** | (16384, 4096, 2048) | 105.36% | 92.77% | **-12.59** | 55.52 dB | 0.0 | 1.0 | **ACCEPT_FALLBACK** (>2pp shortfall — 10-run SKIPPED per STOP rule) |
| **L-2** | (16384, 4096, 3072) | 103.06% | 94.63% | **-8.43**  | 55.57 dB | 0.0 | 1.0 | **ACCEPT_FALLBACK** (>2pp shortfall — 10-run SKIPPED per STOP rule) |
| **L-3** | (32768, 14336, 2048) | 100.39% | 86.01% | **-14.38** | 55.52 dB | 0.0 | 1.0 | **ACCEPT_FALLBACK** (>2pp shortfall — 10-run SKIPPED per STOP rule) |

## Mechanism falsified

The R57 plan hypothesis (192×256 grid-saturation balance beats 256×256 on N=14336 cell L-3) is **falsified**: 192×256 is **even worse** than 256×256 vs HK on L-3 (256×256 was -1.33pp per R55; 192×256 is -14.38pp here). Same falsification on L-1/L-2 (smaller shapes, where HK is already 103-105% comp). The eff=109.7 ranking does not predict end-to-end perf on these grid-bound cells.

## Closures

- 192×256 alt-tile axis CLOSED on cells L-1, L-2, L-3 (HK kept-cell pool).
- All 3 surviving HK cells in R56 manifest **REMAIN HK** in R57.
- Combined HK→AITER swaps from this cohort: **0**. AITER bit-deterministic share unchanged from R56 (39/42). HK survivor count unchanged from R56 (3/42).

## Correctness

All 3 candidates passed bit-determinism / SNR / wcf / finite-frac gates. The shim worked correctly with the 192×256 `.co` for the 4th time (after R56 G-4 C3 first usage). Shim continues to be tile-generic (256×256, 128×512, 192×256, 96×640, 64×1024 all proven AS-IS — 9th consecutive R50D shim AS-IS reuse round).

## Recommendation to decider

- KEEP all 3 HK cells (R40B_HK kernel) as baseline for R57 integration.
- Mark 192×256 axis CLOSED for HK kept-cell pool. Do not retry next round.
- Possible R58 follow-up: per the plan's "fallback exploration if 192×256 DEAD", a 5-seed SMOKE on **128×256** (eff=85.3) is permitted on these cells; however, eff=85.3 < eff=109.7 so prior is even more strongly DEAD. Recommend SKIP unless decider has spare GPU budget.

## Outputs

- `bench_R57H2_L1.py`, `bench_R57H2_L2.py`, `bench_R57H2_L3.py`
- `R57_OPT_L1_SMOKE.{json,log}`, `R57_OPT_L2_SMOKE.{json,log}`, `R57_OPT_L3_SMOKE.{json,log}`
- `R57H2_L1_INTEGRATION_FRAGMENT.json`, `R57H2_L2_INTEGRATION_FRAGMENT.json`, `R57H2_L3_INTEGRATION_FRAGMENT.json`
- `R57_OPT_H2_VERDICT.md` (this file)
