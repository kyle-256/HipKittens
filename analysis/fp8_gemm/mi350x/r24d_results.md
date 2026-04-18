# R24D — A-only Non-Temporal Cache Hint on DLA shapes

**Date:** 2026-04-18  **Machine:** MI355X (gfx950)  **GPUs used:** 0–3
**Bench params:** warmup=200, iters=500, trim=10% (per .claude/rules/benchmark-rules.md)
**Gate:** Δ ≥ +1.5pp vs `_r24d_baseline`

## Hypothesis
R22B (commit `b657e5e8`) showed B-NT and B+A NT regress on all 3 DLA shapes
because B-tile is shared across K-iters within a CTA — NT bypass kills L2 reuse.
A-tile is M-streamed (no K-reuse within a CTA), so A-only NT might help by not
polluting L2 with A data we will never re-fetch.

## Variants
- `_r24d_baseline`  A=0, B=0  (fresh rebuild for fair compare)
- `_r24d_ant1`      A=1 (`non_temporal`, slc+glc bypass), B=0
- `_r24d_ant2`      A=2 (`cache_stream`, glc only),       B=0

## Results (TFLOPS)

| Shape | baseline | ant1 (Δ %)             | ant2 (Δ %)             | best     | verdict   |
|-------|----------|------------------------|------------------------|----------|-----------|
| DLA1  | 5222.87  | 4984.29 (−4.57%) LOSE  | 4957.36 (−5.08%) LOSE  | baseline | DEAD-END  |
| DLA2  | 4221.82  | 3939.21 (−6.69%) LOSE  | 3873.47 (−8.25%) LOSE  | baseline | DEAD-END  |
| DLA7  | 4122.33  | 3929.40 (−4.68%) LOSE  | 3956.28 (−4.03%) LOSE  | baseline | DEAD-END  |

## Verdict: DEAD END (all 3 DLA shapes)

Every A-NT variant regresses by 4–8% on every DLA shape; no variant clears the
+1.5pp gate. Combined with R22B (B-NT and B+A NT also regress), this fully
falsifies the "selective NT hint frees L2 for the other operand" hypothesis on
these shapes.

### Mechanistic refutation
The premise was that A-tile has no K-reuse within a CTA, so bypassing L2 for A
would be free. The data shows the bypass is *not* free:
- A is M-streamed *across CTAs* but the same A-line is consumed by multiple
  warps within a CTA (M-tile rows × K-iters with rotating buffers); evicting
  early forces re-fetch.
- Even `cache_stream` (GLC-only, ant2) loses 4–8% — it's not just an
  invalidation issue; the L2 hit rate on A is materially positive.
- The TCP_DATA_STALL identified in R21-recon is not load-bypass-fixable; it is
  shared-memory / vmcnt structural.

## Files
- Build script: `build_round24_optD.py`
- Bench script: `bench_round24_optD_smoke.py`
- Bench JSON:   `bench_round24_optD_smoke.json`
- Bench log:    `bench_round24_optD_smoke.log`
- Built .so:    `build_all42/tk_mxfp4_gluon_cpp_n*_k*_ts_*_r24d_{baseline,ant1,ant2}.cpython-310-*.so`

## Recommendation
Stop exploring `*_LOAD_NONTEMPORAL` axis on DLA shapes. R22B + R24D together
exhaust the {A,B,both} × {non_temporal, cache_stream} cache-hint matrix; all
six combinations LOSE. Future DLA work should pivot to either (a) reducing
LDS pressure / vmcnt waits, or (b) restructuring the K-loop to lengthen
in-CTA B-tile reuse so a wider tile fits in HBM-bandwidth budget.
