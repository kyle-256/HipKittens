# R55 Opt D-5A — Marginal HK-VC perf claw-back probe (M=16384/4096)

**Worker C, GPU 2.** All 3 candidates: R50D shim AS-IS + aiter `.co` dlopen, tile 256x256.
Bench params: warmup=200, iters=500, trim=0.10. 10-run seeds [101,202,...,1010].

## Per-shape result table

| # | Shape (MxNxK)        | HK R54 pct | AITER SMOKE pct | AITER 10-run pct | n_OK_10 | wcf_max | fin_min | delta_pp | Verdict   |
|---|----------------------|------------|-----------------|------------------|---------|---------|---------|----------|-----------|
| 1 | 16384x4096x4096      | 101.16%    | 108.90%         | 108.06%          | 10/10   | 0.0     | 1.0     | +6.90pp  | PROMOTE   |
| 2 | 16384x6144x2048      | 106.19%    | 107.25%         | 108.00%          | 10/10   | 0.0     | 1.0     | +1.81pp  | PROMOTE   |
| 3 | 4096x4096x8192       | 100.16%    | 117.47%         | 117.15%          | 10/10   | 0.0     | 1.0     | +16.99pp | PROMOTE   |

(`pct` here = `tflops / HK_R54_consensus_tflops_p50` so AITER pct - HK pct == raw perf delta in pp.)

## Cohort summary

- **3/3 PROMOTE / 0 DEAD / 0 REJECT**.
- All 3 pass D-5A marginal gate (`AITER pct_comp > HK pct_comp + 0.5pp`, `n_OK_10 >= 8`, bit-determinism gates).
- Aggregate perf delta over HK p50: **+25.70pp** across the 3 shapes (mean +8.57pp).
- All 3 strict 10-run PASS (wcf=0, fin=1.0 every seed; SNR median 55.5-55.6 dB).
- Net VC delta = 0 (all 3 were already VC HK PASS_10/10); D-3A-1 risk averted on every cell because AITER perf strictly exceeds HK by >=1.81pp.
- **R50D shim reused AS-IS for the 7th consecutive round**; no rebuild, no kernel modification.

## Mechanism notes

- D-5A confirms aiter `.co` 256x256 tile is a stronger choice than HK on these 3 marginal-HK-VC shapes despite HK already running >=100%; biggest win on the wide-K small-MN shape (4096x4096x8192, +17pp).
- All 3 shapes use the same .co + same shim binary — no per-shape rebuild required.
- AITER bit-determinism (wcf=0 across 10 INDEPENDENT seeds) is a side-benefit, but the PROMOTE decision is driven by the perf delta alone.

## Files

- `bench_R55D5A_{1,2,3}.py`
- `R55_OPT_D5A_{1,2,3}_SMOKE.{json,log}`
- `R55_OPT_D5A_{1,2,3}_10RUN.{json,log}`
- `R55D5A_{1,2,3}_INTEGRATION_FRAGMENT.json`
