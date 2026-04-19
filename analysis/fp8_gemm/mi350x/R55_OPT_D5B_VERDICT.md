# R55 Opt D-5B Verdict — Marginal HK-VC perf claw-back, M=32768

Worker D | GPU 3 | warmup=200, iters=500, trim=0.10 | seeds=[101..1010 step 101]

All 3 candidates use AITER 256x256 via R50D shim AS-IS (no rebuild).
HK-VC baseline = R40B PASS_10/10 (R54).

## Per-candidate results

| Idx | Shape | HK pct | AITER SMOKE pct | AITER 10-run pct | Delta pp | n_OK | wcf_max | fin_min | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 32768x14336x2048 | 103.03% | 101.70% | (skipped) | -1.33 | -- | -- | -- | NO_PROMOTE (ACCEPT_FALLBACK) |
| 2 | 32768x28672x2048 | 99.68%  | 101.65% | 101.94% | +2.26 | 10/10 | 0.0 | 1.0 | **PROMOTE** |
| 3 | 32768x6144x2048  | 103.27% | 107.18% | 107.23% | +3.96 | 10/10 | 0.0 | 1.0 | **PROMOTE** |

## Cohort summary

- **2 PROMOTE / 1 NO_PROMOTE** (D-3A-1 ACCEPT_FALLBACK on D-5B/1)
- Net VC delta: 0 (all 3 already HK-VC; no shape changes VC status)
- Perf claw-back: +2.26pp on `32768x28672x2048` and +3.96pp on `32768x6144x2048` (sum +6.22pp)
- D-5B/1 (32768x14336x2048) AITER -1.33pp under HK; correctly rejected at SMOKE per ACCEPT_FALLBACK rule
- Both PROMOTEs show bit-deterministic correctness (wcf_max=0, fin_min=1.0, snr_med ~55.5dB) across 10 INDEPENDENT seeds

## Mechanism notes

- D-5B/1 (M=32768, N=14336): largest aspect-ratio of the 3, AITER 256x256 underperforms HK by 1.33pp. Consistent with aiter heuristic-vs-actual mismatch on N/256 = 56 (non-power-of-two grid-x).
- D-5B/2 (M=32768, N=28672): N/256=112 (power-of-two adjacent), AITER wins +2.26pp.
- D-5B/3 (M=32768, N=6144): N/256=24, AITER wins +3.96pp (largest D-5B gain). Smallest N suggests HK 256x256 path was leaving headroom on small-N shapes at large M.

## Files

- `bench_R55D5B_{1,2,3}.py` — bench harnesses (cloned from R54 D-4A template)
- `R55_OPT_D5B_{1,2,3}_SMOKE.{json,log}` — single-seed smoke output
- `R55_OPT_D5B_{2,3}_10RUN.{json,log}` — 10-seed independent verification
- `R55D5B_{1,2,3}_INTEGRATION_FRAGMENT.json` — reviewer merge manifest deltas

## Compliance

- GPU 3 only (verified idle via rocm-smi pre-run; %util=0)
- warmup=200, iters=500, trim_frac=0.10 throughout (NO exceptions)
- 10 INDEPENDENT seeds via separate Python subprocesses (no in-process re-seed)
- R50D shim untouched; aiter .co untouched
- D-5B/1 D-3A-1 ACCEPT_FALLBACK applied as specified (HK >= 100% AND AITER < HK + 0.5pp)
