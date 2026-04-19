# R54 INTEGRATION VERDICT (reviewer 10-RUN @ 80%)

**Date:** 2026-04-19
**Reviewer:** Claude Opus 4.7
**Bench:** `bench_all_42_R54_INTEGRATION.py --runs 10 --gpus 0,1,2,3` warmup=200 iters=500 trim=0.10
**Seeds:** [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010] (INDEPENDENT, not same-seed×10)
**Gate:** n_OK >= 8/10 AND wcf_max < 0.02 AND wcf_std < 0.01 AND fin_min >= 0.97
**Manifest:** R54_INTEGRATION_MANIFEST.json (27 aiter overrides + 15 HK shapes)
**Wall:** SMOKE 1.4 min/pass; 10-RUN ~14 min total

## Overall verdict: COMMIT

- **12/12 R54 PROMOTE shapes re-confirmed PASS_10/10** (perfect determinism: wcf_max=0.0, wcf_std=0.0, fin_min=1.0 across all 12)
- **27/27 AITER cells PASS** (100% bit-deterministic across all 10 INDEPENDENT seeds)
- **36/42 strict 10-run VC**, vs R53's 33/42 = **+3 net VC**; vs R52's 36/42 = **0 net but kernel base now bit-deterministic for 27 of 42 shapes**
- **Mean perf delta on 30 shared-VC shapes: +4.42 pp** vs R53 baseline
- **Zero D-4 reverts needed**: all 6 D-4A/D-4B perf claw-backs hold +17.66pp to +28.28pp over R53 HK baselines
- **6 NET VC rescues** (E-1 ×3 + E-2 ×3): R53 cohort-LOSS shapes are now strict 10-run VC via bit-deterministic aiter .co dispatch

## R54 PROMOTE re-confirmation (12/12 PASS)

| Shape | Source | R54 verdict | n_OK | pct_comp | wcf_max | wcf_std | fin_min |
|---|---|---|---|---|---|---|---|
| 32768x4096x2048   | R54E1_1 (256x256) | PASS_10/10 | 10/10 | **105.16%** | 0.0 | 0.0 | 1.0000 |
| 32768x4096x3072   | R54E1_2 (256x256) | PASS_10/10 | 10/10 | **118.58%** | 0.0 | 0.0 | 1.0000 |
| 28672x4096x8192   | R54E1_3 (256x256) | PASS_10/10 | 10/10 | **108.72%** | 0.0 | 0.0 | 1.0000 |
| 4096x32768x4096   | R54E2_1 (256x256) | PASS_10/10 | 10/10 | **105.66%** | 0.0 | 0.0 | 1.0000 |
| 4096x32768x6144   | R54E2_2 (256x256) | PASS_10/10 | 10/10 | **106.96%** | 0.0 | 0.0 | 1.0000 |
| 16384x4096x6144   | R54E2_3 (256x256) | PASS_10/10 | 10/10 | **115.20%** | 0.0 | 0.0 | 1.0000 |
| 4096x14336x16384  | R54D4A_1 (256x256) | PASS_10/10 | 10/10 | **105.87%** | 0.0 | 0.0 | 1.0000 |
| 32768x4096x7168   | R54D4A_2 (256x256) | PASS_10/10 | 10/10 | **106.63%** | 0.0 | 0.0 | 1.0000 |
| 6144x4096x8192    | R54D4A_3 (256x256) | PASS_10/10 | 10/10 | **117.82%** | 0.0 | 0.0 | 1.0000 |
| 4096x4096x16384   | R54D4B_1 (256x256) | PASS_10/10 | 10/10 | **110.96%** | 0.0 | 0.0 | 1.0000 |
| 4096x14336x8192   | R54D4B_2 (256x256) | PASS_10/10 | 10/10 | **114.17%** | 0.0 | 0.0 | 1.0000 |
| 16384x4096x7168   | R54D4B_3 (256x256) | PASS_10/10 | 10/10 | **111.89%** | 0.0 | 0.0 | 1.0000 |

## D-3C/D-4-style marginal-promotion check (all PASS, no reverts)

| Shape | R54 pct_comp | HK baseline | Δ vs HK | VC | Decision |
|---|---|---|---|---|---|
| 4096x14336x16384  | 105.87% | 84.41% (R40B) | **+21.46pp** | PASS | KEEP |
| 32768x4096x7168   | 106.63% | 88.36% (R40B) | **+18.27pp** | PASS | KEEP |
| 6144x4096x8192    | 117.82% | 89.53% (R40B) | **+28.29pp** | PASS | KEEP |
| 4096x4096x16384   | 110.96% | 93.30% (R40B) | **+17.66pp** | PASS | KEEP |
| 4096x14336x8192   | 114.17% | 91.92% (R40B) | **+22.25pp** | PASS | KEEP |
| 16384x4096x7168   | 111.89% | 93.76% (R40B) | **+18.13pp** | PASS | KEEP |

All 6 D-4A/D-4B perf claw-back promotions are well past D-3C 0.5pp gate (smallest +17.66pp). Zero reverts.

## Per-shape PASS/FAIL table (42 rows, sorted by source)

| Shape | Source | Verdict | VC | pct_comp | wcf_max | fin_min |
|---|---|---|---|---|---|---|
| 16384x14336x2048   | R40B            | PASS_9/10  | FAIL | 100.3% | 0.0013 | 0.9222 |
| 16384x14336x4096   | R40B            | FLAKE_1/10 | FAIL |  94.5% | 0.0506 | 0.9904 |
| 16384x28672x2048   | R40B            | PASS_9/10  | FAIL |  97.1% | 0.0133 | 0.9657 |
| 16384x28672x4096   | R40B            | PASS_9/10  | FAIL |  91.8% | 0.0215 | 0.9853 |
| 16384x4096x2048    | R40B            | PASS_10/10 | PASS | 105.2% | 0.0006 | 0.9985 |
| 16384x4096x3072    | R40B            | PASS_10/10 | PASS | 101.6% | 0.0150 | 0.9937 |
| 16384x4096x4096    | R40B            | PASS_10/10 | PASS | 101.2% | 0.0114 | 0.9953 |
| 16384x6144x2048    | R40B            | PASS_10/10 | PASS | 106.2% | 0.0006 | 0.9791 |
| 16384x6144x4096    | R40B            | FLAKE_4/10 | FAIL | 100.7% | 0.0631 | 0.9912 |
| 32768x14336x2048   | R40B            | PASS_10/10 | PASS | 103.0% | 0.0074 | 0.9864 |
| 32768x28672x2048   | R40B            | PASS_10/10 | PASS |  99.7% | 0.0105 | 0.9832 |
| 32768x6144x2048    | R40B            | PASS_10/10 | PASS | 103.3% | 0.0113 | 0.9764 |
| 4096x4096x8192     | R40B            | PASS_10/10 | PASS | 100.2% | 0.0151 | 1.0000 |
| 6144x32768x4096    | R40B            | PASS_9/10  | FAIL |  94.7% | 0.0209 | 0.9791 |
| 4096x128256x32768  | R41A            | PASS_10/10 | PASS |  97.9% | 0.0002 | 0.9997 |
| 4096x32768x28672   | R50D_AITER      | PASS_10/10 | PASS | 100.8% | 0.0000 | 1.0000 |
| 14336x4096x32768   | R51D1_AITER     | PASS_10/10 | PASS | 103.1% | 0.0000 | 1.0000 |
| 16384x4096x28672   | R51D2_AITER     | PASS_10/10 | PASS | 105.0% | 0.0000 | 1.0000 |
| 28672x4096x16384   | R51D3_AITER     | PASS_10/10 | PASS | 102.5% | 0.0000 | 1.0000 |
| 4096x28672x32768   | R52D2A_AITER    | PASS_10/10 | PASS | 101.9% | 0.0000 | 1.0000 |
| 4096x32768x128256  | R52D2B_AITER    | PASS_10/10 | PASS |  99.7% | 0.0000 | 1.0000 |
| 4096x4096x32768    | R52D2C_AITER    | PASS_10/10 | PASS | 105.8% | 0.0000 | 1.0000 |
| 6144x4096x16384    | R53D3A_2_AITER  | PASS_10/10 | PASS | 107.7% | 0.0000 | 1.0000 |
| 4096x6144x32768    | R53D3A_3_AITER  | PASS_10/10 | PASS | 131.5% | 0.0000 | 1.0000 |
| 4096x32768x14336   | R53D3B_1_AITER  | PASS_10/10 | PASS |  66.9% | 0.0000 | 1.0000 |
| 32768x4096x14336   | R53D3B_2_AITER  | PASS_10/10 | PASS |  85.9% | 0.0000 | 1.0000 |
| 128256x32768x4096  | R53D3B_3_AITER  | PASS_10/10 | PASS |  86.2% | 0.0000 | 1.0000 |
| 14336x32768x4096   | R53D3C_1_AITER  | PASS_10/10 | PASS |  88.0% | 0.0000 | 1.0000 |
| 28672x32768x4096   | R53D3C_2_AITER  | PASS_10/10 | PASS |  87.7% | 0.0000 | 1.0000 |
| 16384x4096x14336   | R53D3C_3_AITER  | PASS_10/10 | PASS |  91.2% | 0.0000 | 1.0000 |
| 4096x14336x16384   | R54D4A_1_AITER  | PASS_10/10 | PASS | 105.9% | 0.0000 | 1.0000 |
| 32768x4096x7168    | R54D4A_2_AITER  | PASS_10/10 | PASS | 106.6% | 0.0000 | 1.0000 |
| 6144x4096x8192     | R54D4A_3_AITER  | PASS_10/10 | PASS | 117.8% | 0.0000 | 1.0000 |
| 4096x4096x16384    | R54D4B_1_AITER  | PASS_10/10 | PASS | 111.0% | 0.0000 | 1.0000 |
| 4096x14336x8192    | R54D4B_2_AITER  | PASS_10/10 | PASS | 114.2% | 0.0000 | 1.0000 |
| 16384x4096x7168    | R54D4B_3_AITER  | PASS_10/10 | PASS | 111.9% | 0.0000 | 1.0000 |
| 32768x4096x2048    | R54E1_1_AITER   | PASS_10/10 | PASS | 105.2% | 0.0000 | 1.0000 |
| 32768x4096x3072    | R54E1_2_AITER   | PASS_10/10 | PASS | 118.6% | 0.0000 | 1.0000 |
| 28672x4096x8192    | R54E1_3_AITER   | PASS_10/10 | PASS | 108.7% | 0.0000 | 1.0000 |
| 4096x32768x4096    | R54E2_1_AITER   | PASS_10/10 | PASS | 105.7% | 0.0000 | 1.0000 |
| 4096x32768x6144    | R54E2_2_AITER   | PASS_10/10 | PASS | 107.0% | 0.0000 | 1.0000 |
| 16384x4096x6144    | R54E2_3_AITER   | PASS_10/10 | PASS | 115.2% | 0.0000 | 1.0000 |

**Cohort breakdown:**
- AITER cells: **27/27 PASS** (100% bit-deterministic — wcf_max=0, wcf_std=0, fin_min=1.0 across all 27)
- HK cells: **9/15 PASS** (6 FAIL, all R40B with cohort-race wcf or fin gate failures)

## Cohort-race churn report (UNCHANGED .so files)

R45 documented phenomenon: cohort-race tail draws on UNCHANGED .so files between rounds. R54 churn vs R53 (UNCHANGED HK shapes only):

| Shape | Source | R53 status | R54 verdict | Mechanism |
|---|---|---|---|---|
| 16384x14336x2048 | R40B | PASS (VC) | PASS_9/10 (NV, fin_min=0.9222) | cohort-race LOSS |
| 16384x28672x2048 | R40B | PASS (VC) | PASS_9/10 (NV, fin_min=0.9657) | cohort-race LOSS |
| 6144x32768x4096  | R40B | PASS (VC) | PASS_9/10 (NV, wcf_max=0.0209) | cohort-race LOSS |

Net cohort-race churn on UNCHANGED HK .so cells: **0 / -3 = -3**. These are byte-identical R40B kernels — the loss is tail-draw distribution, NOT a regression caused by R54. See `project_mxfp4_R45_cohort_tail_draw.md`.

## Net VC delta calculation

```
R53 baseline strict 10-run VC: 33/42
R54 NEW VC rescues (E-1 ×3 + E-2 ×3, all aiter bit-deterministic):  +6
R54 cohort-race losses on UNCHANGED HK R40B .so files:               -3
                                                                     -----
R54 strict 10-run VC: 36/42                                          net +3
```

vs R52 baseline (36/42): **net 0** in count, but R54's 36 are 27 bit-deterministic AITER + 9 HK; R52's 36 were 7 AITER + 29 HK. R54's distribution is dramatically more stable round-over-round (no cohort-race tail attrition risk on the 27 AITER cells).

The R54 contribution is +6 NET VC (E-1, E-2) plus 6 perf claw-backs (D-4A 3 sub-90 +18-28pp; D-4B 3 marginal +17-22pp).

## Mean perf delta vs R53 baseline

- 30 shared-VC shapes (in both R53 and R54 strict 10-run set): **+4.42 pp mean pct_comp**
- All 12 R54 PROMOTE shapes were already factored in
- Top gains driven by D-4 perf claw-backs: +28.28pp (6144x4096x8192), +22.25pp (4096x14336x8192), +21.47pp (4096x14336x16384)
- AITER 27-cell aggregate: 100% strict-deterministic (no perf jitter at all across 10 seeds; max delta ≤ 0.15pp on R53→R54 carry-over cells)

## Recommendation: **COMMIT**

Justification:
1. All 12 R54 PROMOTE candidates re-confirmed at 10/10 with perfect bit-determinism (wcf_max=0.0, wcf_std=0.0, fin_min=1.0).
2. 6 NET VC rescues (E-1 + E-2) deliver bit-deterministic verified-correct status for shapes that R53 lost to cohort-race.
3. 6 perf claw-backs (D-4A/D-4B) deliver +17-28pp over HK baselines on shapes that were sub-95% HK-VC; all well past D-3C 0.5pp gate.
4. Mean **+4.42 pp** perf gain on 30 shared-VC shapes vs R53.
5. Apparent net 0 vs R52 hides a major structural improvement: 27/42 cells are now bit-deterministic AITER (vs R52's 7/42), eliminating cohort-race tail-draw risk on that majority of shapes.
6. Apparent -3 cohort losses on UNCHANGED HK R40B .so are R45+ documented tail-draw, NOT R54 regression.
7. 6th consecutive R50D-AS-IS reuse round; zero kernel rebuild required.
8. The trend trajectory is unmistakable: R44 (35) → R52 (36) → R53 (33 attrition) → R54 (36 with 27 AITER bit-deterministic baseline) — the AITER .co dlopen pattern is now load-bearing for stable round-over-round VC retention.

## Artifacts

- `R54_INTEGRATION_MANIFEST.json` — 27 aiter overrides + 15 HK baseline (full per-shape dispatch with tile_M/tile_N/co_path/kernel_name)
- `bench_all_42_R54_INTEGRATION.py` — generic per-shape aiter dispatch (256x256, 96x640, 64x1024)
- `R54_INTEGRATION_SMOKE1.json` / `.log` / `.console` — single-run all-42 SMOKE (41/42 VC; only 16384x14336x4096 wrong, single-run cohort-race)
- `R54_INTEGRATION_10RUN.json` / `.log` / `.console` — full 10-run @ 80% reviewer bench (36/42 strict VC)
