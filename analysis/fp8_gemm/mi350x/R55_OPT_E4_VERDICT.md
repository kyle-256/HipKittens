# R55 Opt E-4 — Cohort-race rescue (smaller-N + 6144x32768) verdict

**Worker B, GPU 1.** 2 candidates, both AITER 256x256 via R50D shim AS-IS.

## Per-shape results (10-run @ 80% gate; warmup=200, iters=500, trim=0.10)

| Idx | Shape (MxNxK)      | R54 status         | R55 verdict | n_OK_10 | wcf_max | wcf_std | fin_min | snr_med_min | TFLOPS | pct_comp |
|-----|--------------------|--------------------|--------|---------|---------|---------|---------|-------------|--------|----------|
| 1   | 16384x6144x4096    | FLAKE_4/10 wcf=0.0631 | PROMOTE | 10/10   | 0.0     | 0.0     | 1.0     | 55.55       | 4595.2 | 113.67%  |
| 2   | 6144x32768x4096    | PASS_9/10 wcf=0.0209  | PROMOTE | 10/10   | 0.0     | 0.0     | 1.0     | 55.55       | 4503.8 | 104.96%  |

Competitor TFLOPS sources (R54 INTEGRATION 10RUN consensus):
- 16384x6144x4096: comp=4042.5
- 6144x32768x4096: comp=4291.0

## Headline

**2/2 PROMOTE / 0 DEAD.** Both candidates passed all four gates with perfect bit-determinism
(wcf_max=0, wcf_std=0, fin_min=1.0) across 10 INDEPENDENT seeds, plus perf claw-backs:

- E4_1 (16384x6144x4096): HK R40B 100.71% (FLAKE_4/10) -> AITER 113.67% PASS_10/10. **+12.96pp pct_comp.**
- E4_2 (6144x32768x4096): HK R40B 94.66% (PASS_9/10) -> AITER 104.96% PASS_10/10. **+10.30pp pct_comp.**

Mechanism: identical to R54 E-1/E-2; R50D shim reused AS-IS (7th consecutive AS-IS reuse round
across the cohort), no kernel rebuild. Aiter heuristic uniformly picks 256x256 tile for both
shapes (eff=128 tiebreak), and `BpreShuffle_256x256.co` clears the cohort-race that the HK
R40B kernel cannot escape on these (M=16384, N=6144) and (M=6144, N=32768) cells.

## NET VC delta contribution to R55

**+2 NET VC** (both shapes were R54 strict 10-run FAIL). Fully consistent with E-3/E-4 floor
projection of "+2 to +6 NET VC" in R55_DECIDER_PLAN.md.

## Files

- bench: `bench_R55E4_1.py`, `bench_R55E4_2.py`
- SMOKE results: `R55_OPT_E4_{1,2}_SMOKE.{json,log}`
- 10run results: `R55_OPT_E4_{1,2}_10RUN.{json,log}`
- Manifest deltas: `R55E4_{1,2}_INTEGRATION_FRAGMENT.json`
