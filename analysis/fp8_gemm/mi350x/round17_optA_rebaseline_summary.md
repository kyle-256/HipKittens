# Round 17 Optimizer A — 42-shape Re-baseline Summary

**Date**: 2026-04-17
**Author**: Kyle.Zhao@amd.com / kyle-256
**Tool**: bench_all42_parallel.py with `0,1,2,3,4,5,6` (7 GPUs in parallel; GPU 7 reserved for profile)
**Settings**: warmup=200, iters=500, trim=10%, 115 variants
**Status**: **PARTIAL — 26 / 42 shapes completed at time-out**
**Output**: `bench_all42_results_r17_rebaseline.json` (parsed from log)
**Comparison file**: `round17_optA_rebaseline_compare.txt`

---

## 1. Status

- 26/42 shapes finished in ~56 minutes wall (started 21:04, last shape printed 22:00)
- 16 shapes still in-flight (mostly the K=128256 deep-LOSE quartet + larger M/N variants)
- Time-out caused by user 50-min wall budget

## 2. Drift Findings (26-shape sample)

### Shape ratio shifts ≥ 2pp vs Round 2

| Shape | R2 ratio | R17 ratio | Δpp | R2 best | R17 best | Reading |
|---|---|---|---|---|---|---|
| 16384×6144×2048   | 114.6 % | 112.6 % | **−2.0** | ts_v4_tv0           | ts_v12_tv0_memc       | regress (lost +2pp) |
| 4096×14336×16384  |  98.6 % |  96.6 % | **−2.0** | ts_lgk2             | ts_memc               | regress |
| 4096×4096×8192    | 110.9 % | 108.0 % | **−2.9** | default             | lgk2_dc               | regress |
| 4096×4096×16384   | 104.4 % | 106.9 % | **+2.5** | ts_v24              | ts_pf6_6_lgk2_v12     | improve |
| 6144×4096×16384   |  98.6 % | 100.8 % | **+2.2** | ts_lgk2             | ts_v12_tv0_memc       | improve (LOSE→WIN) |
| 4096×4096×32768   | 100.9 % | 103.6 % | **+2.7** | ts                  | ts_v12_tv0            | improve |
| 6144×4096×8192    | 102.7 % | 104.9 % | **+2.2** | v32                 | v20_memc              | improve |
| 6144×32768×4096   | 101.1 % | 103.6 % | **+2.5** | ts_v4_memc          | ts_lgk2_memc          | improve |

**8 shapes shifted ≥ 2pp**: 5 improved (+), 3 regressed (−). Net: **+0.4 pp average** on shifted shapes.

### Shapes that did NOT shift (within ±1.9pp): 18 / 26 confirmed

The other 18 shapes show **<2pp drift** vs Round 2 — baseline is largely stable.

### Variant churn (best-tag changed even when ratio steady)

8 of 26 shapes pick a DIFFERENT best variant in R17 even where ratio is stable
(e.g. 16384×14336×2048: ts_lgk2_v20 → ts_memc; 32768×4096×2048: ts_v12_tv0_memc → ts_lgk2_memc).
This is consistent with cross-GPU bias noted in R12B — the optimal tag is sensitive to physical
GPU differences (same chip, different thermal/clock state across the 7 GPUs).

## 3. Missing shapes (still running)

Did NOT complete in time-out window (largest K, largest M/N):
- 4096×32768×128256 (DLA1) — being profiled separately on GPU 7
- 4096×128256×32768 (DLA2)
- 28672×32768×4096 (DLA7)
- 28672×4096×16384 (P1)
- 16384×4096×{4096,6144,7168,14336,28672}, 16384×{6144,14336,28672}×4096
- 28672×4096×8192, 32768×4096×{7168,14336}
- 128256×32768×4096

These can be re-tried in a follow-up R17 sub-task.

## 4. Verdict for re-baseline

- **Baseline is broadly STABLE** — most shapes within ±2pp of Round 2.
- **Net 5 improvements / 3 regressions in shifted shapes** — slight positive drift, plausibly from
  variant-set additions in Round 2 finally reaching equilibrium for some shapes.
- **Variant-tag instability remains** (cross-GPU bias) — for any future critical bench, pin a single
  GPU and re-run target shapes 2-3 times.
- **No saturation breakthrough** — none of the formerly LOSE shapes flipped to WIN at the deep-LOSE
  level (4096×14336×16384 actually regressed by 2pp, now joins LOSE pool).

**Recommendation**: do NOT update `bench_all42_results_round2.json` from this partial run. Either
(a) finish the re-baseline overnight on dedicated 8 GPUs, or (b) accept Round 2 as canonical until a
true 8-GPU full run completes.
