# R49 Dev E — Dormant CRR variants resweep (hbnshrink, rect) — REFUTED

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ c5140f11
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=6
**Hypothesis:** R47-era dormant CRR variants `crr_hbnshrink_fastpath.inc`
(BLK_N=128) and `rect` (`MXFP8_RECT_BLK_N=64`) deserve re-bench under
strict SCLK protocol now that the R48 baseline shows CRR is structurally
~92% ceiling-bound. Plausible: at a different BLK_N geometry, the
A-side LDS reuse pattern flips and unlocks +2-4%.

## TL;DR — VERDICT: REFUTED (both variants, all cells)

| Cell        | Variant   | base med | treat med | Δ% | spread |
|-------------|-----------|---------:|----------:|---:|-------:|
| 8B Gate/Up  | hbnshrink | 2349.0  | 1338.0 | -43.04% | 61.73% (treat unstable) |
| 8B Gate/Up  | rect      | 2349.0  | 1522.9 | -35.17% | 0.30%  |
| 70B Gate/Up | hbnshrink | 2527.8  | 1270.7 | -49.73% | 7.84% |
| 70B Gate/Up | rect      | 2527.8  | 1394.0 | -44.85% | 0.47% |
| 70B Down    | hbnshrink | 2728.6  | 1403.9 | -48.55% | 44.71% (treat unstable) |
| 70B Down    | rect      | 2728.6  | 1625.4 | -40.43% | 0.43% |

All 6 cells regress 35–50%. SHIP gate fails on every axis.

## Diagnosis

Both dormant variants change BLK_N (rect → 64, hbnshrink → 128) which
shifts the wave-tile geometry at the same BLK_M=256:
- `rect` (BLK_N=64): doubles tile count → 4× wavefront occupancy at
  fixed grid, but each tile does 1/4 the MFMA work — net result wave-tail
  becomes the dominant cost at our 4096-M family.
- `hbnshrink` (BLK_N=128): halves tile count → 1/2 wave count but
  per-tile B-side LDS doubles. The instability (61% / 44% spread) on two
  cells indicates SCLK-throttle from sustained LDS pressure.

Neither geometry beats the production BLK_N=256 layout for any of the
target cells. R47-era's "dormant" status is consistent with this measurement.

## Outcome

No source landed. Variant files retained in tree for archival
(`crr_mxfp8_exact_8wave_hbnshrink_fastpath.inc`, the `MXFP8_RECT_BLK_N`
macro arm in `crr_mxfp8_exact_8wave_fastpath.inc`). Both closed as
production levers.

## Files

- `r49e_bench.sh` — bench driver (3 cells × 3 variants × 5 runs/cell)
- `r49e_bench.run.log` — orchestrator stdout (full summary in tail)
- `r49e_results/` — per-cell per-variant run logs
- `crr_mxfp8_exact_8wave_hbnshrink_fastpath.inc` — variant header
  retained, default OFF via `MXFP8_USE_HBNSHRINK`
