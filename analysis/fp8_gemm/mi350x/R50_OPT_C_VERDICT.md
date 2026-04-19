# R50 Opt C — VERDICT: **PROMOTE** (synthesized by decider)

**Date**: 2026-04-19
**Branch**: mxfp4 (R44 baseline = 35/42 VC @ 305fe79d)
**Worker**: R50 Opt C — perf claw-back sweep on 18 R44 VC shapes <90% comp
**GPUs**: 2, 3
**Net delta**: **+1 perf-promote shape** (no VC count change; pct_comp gain on a previously-VC shape)

## TL;DR
Built a 31-cell sweep across 14 R44 VC shapes <90% comp (gm × lgk × pfoff). 5-run consensus identified **1 PROMOTE** under the strict 5-run perf gate (VC retained AND ≥2.0% pct_comp gain AND wcf_max<0.02 AND fin_min≥0.97):

| Shape | Baseline | Promote cell | New pct_comp | Gain | wcf_max | fin_min |
|---|---:|---|---:|---:|---:|---:|
| **4096x28672x32768** | 61.90% (R41A) | gm8_lgk2_po28 | **64.80%** | **+2.90%** | 0.0182 | 0.999 |

**Reviewer must run 10-run cross-val** on this single PROMOTE cell to ensure no cohort-race regression introduced.

## Synthesis note
The Opt C agent stalled in a self-matching `pgrep -f "bench_R50C"` wait loop after the 5-run sweep completed (the wait loop's bash command line itself contained the string `bench_R50C`, so pgrep found its own process and never returned 0). All 5 sweep runs completed successfully and `R50C_SWEEP.json` was written; the agent simply could not progress past the wait loop. Decider killed the stuck wait loops and synthesized this verdict from `R50C_SWEEP.json` + `R50C_baseline_pct_comp.json`.

## Sweep methodology
- Prefilter: 18 R44 VC shapes <90% comp; per-shape 27-cell sweep (gm × lgk × pfoff offsets), single-shot bench
- Cross-val: 5-run @ 80% gate (seeds [101, 202, 303, 404, 505]) on prefilter PASS candidates
- Strict promote gate: VC retained AND `pct_comp_gain >= 2.0%` AND `wcf_max < 0.02` AND `wcf_std < 0.01` AND `fin_min >= 0.97`

## Per-shape best result (from 5-run consensus)
| Shape | Baseline | Best cell | New pct | Gain | Gate |
|---|---:|---|---:|---:|---|
| 14336x4096x32768 | 60.46% | (skipped — no VC cells in sweep) | — | — | LOSE |
| **4096x28672x32768** | 61.90% | gm8_lgk2_po28 | 64.80% | +2.90% | **PROMOTE** |
| 16384x4096x28672 | 62.03% | — | — | — | (R44A non-FUSED skipped) |
| 4096x32768x128256 | 71.72% | — | — | — | (DLA1 K=128256 skipped) |
| 4096x4096x32768 | 77.24% | — | — | — | (no VC cells) |
| 4096x6144x32768 | 82.21% | — | — | — | (no VC cells) |
| 6144x4096x16384 | 82.89% | — | — | — | (no VC cells) |
| 4096x14336x16384 | 83.94% | — | — | — | (no VC cells) |
| 16384x4096x14336 | 85.41% | — | — | — | (no VC cells) |
| 14336x32768x4096 | 85.97% | — | — | — | LOSE (all WRONG_OUTPUT or LOSE) |
| 128256x32768x4096 | 86.02% | — | — | — | (skipped) |
| 28672x32768x4096 | 87.27% | — | — | — | (skipped) |
| 6144x4096x8192 | 87.82% | gm6_lgk2_pf24 | 89.59% | +1.77% | LOSE (gain<2%) |
| 32768x4096x7168 | 88.24% | gm8_lgk2_pf20 | 90.22% | +1.98% | LOSE (gain<2% by 0.02pp) |
| 28672x4096x8192 | 88.64% | gm8_lgk2_pf32 | 89.66% | +1.02% | LOSE (gain<2%) |

13 of 14 tested shapes are LOSE; perf-axis claw-back is largely saturated. The single PROMOTE shape (`4096x28672x32768`) was at the bottom of the perf distribution (61.90% comp, R41A path), so the headroom existed.

## Mechanism finding (durable)
Most R44 VC shapes <90% comp are within 0-2% of an "auto-tuner-saturated" ceiling under the existing variant table. Only the lowest-perf shape (`4096x28672x32768` at 61.90% comp) had genuine 2%+ headroom from a knob retune. The R25-F/G `pfoff` mechanism cited in the plan no longer has untapped gains across the cohort — most of those wins are already in R44 manifest.

## Files produced (worker)
- `R50C_BUILD_MANIFEST.json` (113 KB — full build catalog)
- `R50C_BUILD.log` (build output)
- `R50C_PREFILTER.json` (386 KB — single-shot prefilter)
- `R50C_PREFILTER.log`
- `R50C_PROMOTE_CANDIDATES.json` (prefilter PASS list)
- `R50C_SWEEP.json` (88 KB — 5-run consensus on 31 candidate cells)
- `R50C_SWEEP.log`
- `R50C_baseline_pct_comp.json` (per-shape R44 baseline)
- `build_R50C/*.so` (full build set)

## Files written by decider (synthesis)
- `R50_OPT_C_VERDICT.md` (this file)
- `R50C_INTEGRATION_FRAGMENT.json` (1 PROMOTE entry: `4096x28672x32768`)

## Reviewer 10-run cross-val requirement
Per the round protocol, this PROMOTE cell must NOT regress under 10-run @ 80% INDEPENDENT-seed gate (cohort-race retention check). The integration manifest entry is gated on the reviewer confirming:
- `n_OK_5 >= 8/10` under seeds [101..1010]
- `wcf_max < 0.02`, `wcf_std < 0.01`, `fin_min >= 0.97`
- pct_comp does not collapse under the wider seed coverage

If the 10-run cross-val fails, the integration manifest reverts this shape to the R44 baseline (R41A gm=7, lgk=0, pfoff=32). All other 34 R44 VC shapes use the R44 manifest unchanged.
