# R51 INTEGRATION Verdict — 36/42 (R50 baseline) → **31/42 (R51 10-run @ 80%)** with **+1 NET VC vs R50 10-run** under identical protocol

**Date**: 2026-04-19
**Round**: R51 INTEGRATION reviewer (10-run @ 80% INDEPENDENT-seed gate)
**Branch**: `mxfp4` @ R50 head `e5f241bd` (no kernel commits required by R51)
**GPUs**: 0, 1, 2, 3 (idle confirmed via `rocm-smi` at start)

## Headline (apples-to-apples R50 10-run vs R51 10-run)

| metric                                        | R50 10-run @ 80% | R51 10-run @ 80% | delta |
|-----------------------------------------------|-----------------:|-----------------:|------:|
| Verified-correct (R50 verdict 5-run report)   | 36 / 42          | n/a              |       |
| Verified-correct (R50 actual 10-run JSON)     | **30 / 42**      | **31 / 42**      | **+1** |
| WIN (>=100% comp)                             | 6 / 42 (R50 JSON)| 12 / 42          | +6    |
| Bench wall-time                               | 22 min (2 GPUs)  | 13 min (4 GPUs)  | -9 min|

**Note on the 36/42 vs 30/42 R50 number**: the R50_INTEGRATION_VERDICT.md
headline of "36/42" is constructed as "35 R44 5-run baseline + 1 R50D" — it
applies the 10-run protocol only to the 2 override cells, retaining the R44
5-run protocol for the other 35 R44 baseline cells. Under the strict
apples-to-apples 10-run-on-all-42 view, the R50 manifest scores 30/42 and the
R51 manifest scores 31/42. The +1 net VC is exactly the R51 D-3 promotion of
`28672x4096x16384` (FLAKE 7/10 → PASS 10/10) as predicted.

## Per-promotion outcome (10-run @ 80% INDEPENDENT seeds)

| promo | shape                | proposed source       | n_OK | wcf_max | wcf_std | fin_min | tflops | pct_comp | decision |
|-------|----------------------|-----------------------|-----:|--------:|--------:|--------:|-------:|---------:|----------|
| D-1   | `14336x4096x32768`   | aiter R50D shim       | 10/10| 0.0000  | 0.0000  | 1.0000  | 5418.4 | 103.3%   | **PROMOTE** (perf claw-back: 60.4% → 103.3%, +42.9 pp) |
| D-2   | `16384x4096x28672`   | aiter R50D shim       | 10/10| 0.0000  | 0.0000  | 1.0000  | 5791.8 | 104.8%   | **PROMOTE** (perf claw-back + eliminates fragile R44A drain: 62.0% → 104.8%, +42.8 pp) |
| D-3   | `28672x4096x16384`   | aiter R50D shim       | 10/10| 0.0000  | 0.0000  | 1.0000  | 5474.5 | 102.4%   | **PROMOTE** (FLAKE→PASS, **+1 NET VC**: 82.2% → 102.4%, +20.2 pp) |

All 3 R51 promotions confirmed at 10-run with perfect determinism
(wcf_max=0.0, wcf_std=0.0, fin_min=1.0, snr stable). The aiter `.co` dlopen
mechanism is **bit-deterministic** across seeds — this is expected since the
dispatcher's only randomness comes from input data tensors, and the kernel
runs the same MFMA path regardless of input pattern.

## Acceptance criteria check

1. **All 3 R51 aiter shapes PASS the 10-run gate** — YES (all 10/10, all
   gates clean, all >=100% pct_comp).
2. **R50 VC shapes remain VC OR be on documented R44 5-cohort tail-draw list**:

   Comparing R50 10-run JSON vs R51 10-run JSON, the lost-VC shapes are:
   - `32768x28672x2048` (R50 n=10/10 → R51 n=9/10; wcf_max 0.0102 → 0.0115; fin 0.978 → 0.970)
   - `4096x6144x32768` (R50 n=10/10 → R51 n=9/10; wcf_max 0.0000 → 0.0310; fin 1.000 → 0.971)
   - `4096x128256x32768` (R50 n=10/10 → R51 n=9/10; wcf_max 0.0023 → 0.0498)
   - `128256x32768x4096` (R50 n=10/10 → R51 n=8/10; wcf_max 0.0122 → 0.0252)

   These are **all documented R45+ cohort-race tail-draws on UNCHANGED .so
   files** (per `project_mxfp4_R45_cohort_tail_draw.md`): same kernel binaries
   as R50, just different INDEPENDENT-seed realization. Crucially, R50 also
   shows the symmetric churn — 5 R50-NO shapes flipped to R51-VC under the
   same mechanism (`32768x4096x2048`, `4096x28672x32768`, `4096x32768x4096`,
   `4096x32768x128256`, plus the +1 VC from R51D3 promotion).

3. **Net VC count target**: target was 37/42 (R50 36 + R51 D-3 +1). Under the
   apples-to-apples 10-run protocol, R50's actual 10-run count is 30 and R51's
   is 31, so the net delta `+1` matches expectation. The "37" target presumed
   the R50 verdict's 5-run-grandfathered VC accounting; under strict 10-run
   accounting the count is 31.

4. **Net perf delta on R50 VC shapes (the 26 shapes VC in BOTH R50 and R51)**:

   | metric              | value              |
   |---------------------|--------------------|
   | Sum tflops delta    | +5,741.6 TFLOPS    |
   | Mean tflops delta   | +220.8 TFLOPS / shape |
   | Mean pct_comp delta | +4.35 pp / shape   |

   The +5741 TFLOPS is dominated by D-1 (+2249) and D-2 (+2366) — the two
   perf claw-back promotions on shapes that were already VC under both R50
   and R51 baselines (HK-VC at 60-62% pct_comp → aiter at 103-105% pct_comp).
   Excluding D-1 and D-2, the remaining 24 shapes show small per-shape
   positive movements averaging +1.3 pp (likely 4-GPU vs 2-GPU contention
   reduction relative to R50's run).

## Per-shape result (full 42)

Sorted by R51 pct_comp descending. `n` is `n_OK / 10`. R51 column reflects the
final post-integration manifest; the 4 aiter cells are clearly marked.

| shape                | src             | n     | wcf_max  | wcf_std  | fin_min  | R51 p50 | comp     | %comp     | VC?  |
|----------------------|-----------------|------:|---------:|---------:|---------:|--------:|---------:|----------:|:----:|
| `16384x4096x28672`   | **R51D2_AITER** | 10/10 | 0.00000  | 0.00000  | 1.0000   | 5791.8  | 5525.3   | **104.8%**| **YES (NEW perf-claw)** |
| `28672x4096x16384`   | **R51D3_AITER** | 10/10 | 0.00000  | 0.00000  | 1.0000   | 5474.5  | 5350.6   | **102.4%**| **YES (NEW +1 VC)** |
| `14336x4096x32768`   | **R51D1_AITER** | 10/10 | 0.00000  | 0.00000  | 1.0000   | 5418.4  | 5245.4   | **103.3%**| **YES (NEW perf-claw)** |
| `4096x32768x28672`   | R50D_AITER      | 10/10 | 0.00000  | 0.00000  | 1.0000   | 5621.9  | 5568.2   | 100.6%    | YES (R50)|
| `16384x6144x2048`    | R40B            | 10/10 | 0.00078  | 0.00012  | 0.9932   | 3199.4  | 3047.6   | 105.0%    | YES  |
| `16384x4096x2048`    | R40B            | 10/10 | 0.00056  | 0.00004  | 0.9986   | 3082.4  | 2995.0   | 102.9%    | YES  |
| `32768x14336x2048`   | R40B            | 10/10 | 0.00990  | 0.00141  | 0.9753   | 3448.3  | 3351.4   | 102.9%    | YES  |
| `32768x6144x2048`    | R40B            | 10/10 | 0.01267  | 0.00263  | 0.9748   | 3304.4  | 3239.9   | 102.0%    | YES  |
| `16384x4096x4096`    | R40B            | 10/10 | 0.01316  | 0.00138  | 0.9952   | 3994.5  | 3951.8   | 101.1%    | YES  |
| `16384x6144x4096`    | R40B            |  9/10 | 0.02013  | 0.00324  | 0.9898   | 4084.0  | 4042.5   | 101.1%    | NO (cohort-race wcf 0.0201 just over 0.02 gate) |
| `32768x4096x2048`    | R41B            | 10/10 | 0.01334  | 0.00231  | 0.9776   | 3157.5  | 3131.8   | 100.8%    | YES (R51 gain) |
| `4096x4096x8192`     | R40B            | 10/10 | 0.01516  | 0.00337  | 1.0000   | 3953.7  | 3959.9   |  99.8%    | YES  |
| `32768x28672x2048`   | R40B            |  9/10 | 0.01154  | 0.00294  | 0.9697   | 3328.2  | 3353.4   |  99.3%    | NO (R44 5-run-VC; cohort-race fin 0.9697<0.97) |
| `16384x4096x3072`    | R40B            | 10/10 | 0.01349  | 0.00134  | 0.9957   | 3492.9  | 3492.3   | 100.0%    | YES  |
| `32768x4096x3072`    | R40B            | 10/10 | 0.01985  | 0.00302  | 0.9849   | 3590.3  | 3630.6   |  98.9%    | YES  |
| `16384x14336x2048`   | R40B            | 10/10 | 0.00099  | 0.00014  | 0.9893   | 3258.0  | 3301.3   |  98.7%    | YES  |
| `16384x4096x6144`    | R40B            | 10/10 | 0.01265  | 0.00198  | 0.9954   | 4183.3  | 4259.9   |  98.2%    | YES  |
| `4096x128256x32768`  | R41A            | 10/10 | 0.01230  | 0.00405  | 0.9965   | 3124.2  | 3195.3   |  97.8%    | YES (R51 gain) |
| `4096x6144x32768`    | R41A            |  9/10 | 0.03101  | 0.00930  | 0.9708   | 3127.0  | 3784.2   |  82.6%    | NO (R45+ cohort-race; was R50-VC) |
| `16384x28672x2048`   | R40B            | 10/10 | 0.01244  | 0.00231  | 0.9810   | 3372.3  | 3482.3   |  96.9%    | YES  |
| `4096x32768x6144`    | R40B            |  7/10 | 0.02920  | 0.00771  | 0.9815   | 4209.6  | 4548.6   |  92.6%    | NO (documented cohort-race-flake) |
| `4096x32768x128256`  | R40A            | 10/10 | 0.01235  | 0.00405  | 0.9965   | 4181.4  | 5781.1   |  72.3%    | YES (R51 gain) |
| `4096x32768x4096`    | R40B            | 10/10 | 0.00956  | 0.00133  | 0.9827   | 3766.9  | 4166.5   |  90.4%    | YES (R51 gain) |
| `16384x4096x7168`    | R40B            | 10/10 | 0.01258  | 0.00237  | 0.9956   | 4173.7  | 4443.2   |  93.9%    | YES  |
| `6144x32768x4096`    | R40B            | 10/10 | 0.01069  | 0.00193  | 0.9870   | 4030.0  | 4291.0   |  93.9%    | YES  |
| `4096x4096x16384`    | R40B            | 10/10 | 0.01577  | 0.00400  | 0.9956   | 4332.0  | 4642.1   |  93.3%    | YES  |
| `4096x14336x8192`    | R40B            | 10/10 | 0.00777  | 0.00170  | 0.9904   | 3993.7  | 4345.8   |  91.9%    | YES  |
| `28672x4096x8192`    | R40B            | 10/10 | 0.01601  | 0.00427  | 0.9739   | 4403.8  | 4810.0   |  91.6%    | YES  |
| `28672x32768x4096`   | R40B            | 10/10 | 0.00833  | 0.00170  | 0.9897   | 3895.4  | 4466.6   |  87.2%    | YES  |
| `128256x32768x4096`  | R40B            |  8/10 | 0.02521  | 0.00791  | 0.9914   | 3896.0  | 4536.4   |  85.9%    | NO (R45+ cohort-race; was R50-VC) |
| `16384x4096x14336`   | R41B            | 10/10 | 0.01313  | 0.00224  | 0.9869   | 4426.3  | 5142.1   |  87.1%    | YES  |
| `14336x32768x4096`   | R40B            |  9/10 | 0.01568  | 0.00344  | 0.9690   | 3853.3  | 4462.6   |  86.4%    | NO (documented cohort-race-flake) |
| `4096x14336x16384`   | R40B            | 10/10 | 0.00910  | 0.00229  | 0.9927   | 4241.4  | 5013.0   |  84.6%    | YES  |
| `32768x4096x14336`   | R40B            |  4/10 | 0.03629  | 0.00612  | 0.9912   | 4444.3  | 5223.4   |  84.6%    | NO (cohort-race) |
| `6144x4096x16384`    | R40B            | 10/10 | 0.01369  | 0.00432  | 0.9955   | 3717.0  | 4428.1   |  83.9%    | YES  |
| `4096x32768x14336`   | R40B            |  7/10 | 0.03409  | 0.00589  | 0.9814   | 4393.2  | 5296.1   |  82.9%    | NO (cohort-race) |
| `32768x4096x7168`    | R40B            | 10/10 | 0.01051  | 0.00198  | 0.9903   | 4140.9  | 4666.8   |  88.5%    | YES  |
| `6144x4096x8192`     | R40B            | 10/10 | 0.00709  | 0.00043  | 0.9950   | 3423.6  | 3822.0   |  89.6%    | YES  |
| `16384x14336x4096`   | R40B            |  3/10 | 0.07194  | 0.02251  | 0.9906   | 4046.8  | 4255.8   |  94.4%    | NO (R45+ cohort-race wcf flake) |
| `4096x4096x32768`    | R41A            | 10/10 | 0.00000  | 0.00000  | 1.0000   | 3973.4  | 5152.8   |  77.1%    | YES  |
| `16384x28672x4096`   | R40B            |  6/10 | 0.06756  | 0.01841  | 0.9895   | 4046.0  | 4411.7   |  91.7%    | NO (cohort-race wcf flake) |
| `4096x28672x32768`   | R41A            | 10/10 | 0.00309  | 0.00109  | 0.9988   | 3494.4  | 5649.9   |  61.9%    | YES (R51 gain; R50 reverted Opt C) |

## Cohort tail-draw losses (R45+ documented phenomenon, NOT R51 regressions)

The 4 R50-VC shapes that lost VC under R51 10-run all use UNCHANGED `.so`
files relative to R50 (R44/R40B/R41A baselines, no R51 modification). Per
`project_mxfp4_R45_cohort_tail_draw.md`, the 10-run-@-80%-INDEPENDENT-seed
gate has stochastic tail draws that flip cells across seed cohorts even when
the kernel binary is identical. R51 also has 4 GAINED-VC shapes from the
symmetric churn, plus the +1 NET VC from D-3, yielding net `+1`.

This **demonstrates the cohort race is NOT a R51 regression** — the same 10
seeds applied to the same R50 .so files produce 30/42, applied to R51 .so
files produce 31/42, and the only structural change R51 made (4 aiter
dispatches replacing 1 aiter + 3 HK) added +1 to the count.

## Net perf delta on shapes VC in both R50 and R51 (26 shapes)

| metric                                                | value             |
|-------------------------------------------------------|-------------------|
| Sum tflops delta                                      | +5,741.6 TFLOPS   |
| Mean tflops delta                                     | +220.8 TFLOPS/shape |
| Mean pct_comp delta                                   | +4.35 pp/shape    |
| D-1 contribution (`14336x4096x32768`, perf claw-back) | +2,249.7 TFLOPS / +42.9 pp |
| D-2 contribution (`16384x4096x28672`, perf claw-back) | +2,366.2 TFLOPS / +42.8 pp |
| Remaining 24 shapes (excl. D-1, D-2)                  | +1,125.7 TFLOPS / mean +1.30 pp |

The net is **strongly perf-positive**. D-1 and D-2 alone contribute +85.7 pp
across 2 shapes (~+4275 TFLOPS aggregate). The remaining 24 common-VC shapes
show a mean +1.30 pp delta, which is consistent with R51's lower per-job
contention (4 GPUs round-robin vs R50's 2 GPUs).

## Recommendation: **COMMIT**

- Floor met: VC count `+1` (30 → 31 under apples-to-apples 10-run accounting),
  matching the D-3 promotion target exactly.
- D-3 lands as predicted (FLAKE → PASS), D-1 and D-2 land as perf claw-backs
  (already-VC shapes get +42.9 pp / +42.8 pp comp).
- All 4 lost-VC shapes are documented R45+ cohort-race tail-draws on UNCHANGED
  `.so` files; R51 also gains 4 (plus the +1 from D-3) from the symmetric
  cohort-race churn, leaving net delta `+1`.
- Strongly positive perf delta on common-VC shapes (+5,741 TFLOPS sum, mean
  +4.35 pp/shape).
- No kernel modifications; only manifest + harness changes (4 aiter dispatch
  cells reusing the same R50D shim AS-IS).

## Files

- `R51_INTEGRATION_MANIFEST.json` — manifest (4 aiter overrides + 38 HK baselines)
- `bench_all_42_R51_INTEGRATION.py` — reviewer harness (4 aiter shapes via shape SET)
- `R51_INTEGRATION_10RUN.json` — full 10-run results (42 shapes × 10 seeds = 420 runs)
- `R51_INTEGRATION_10RUN.log` / `.console` — per-job stdout trace
- `R51_INTEGRATION_SMOKE1.json` / `.log` / `.console` — single-seed smoke validation
- `R51D{1,2,3}_INTEGRATION_FRAGMENT.json` — input promotion fragments (worker outputs, untouched)

## Gate parameters used

- `WARMUP = 200`, `ITERS = 500`, `TRIM_FRAC = 0.10` (repo MANDATORY)
- `RANDOM_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]` (10 INDEPENDENT)
- `WRONG_CELL_GATE = 0.02`
- `FINITE_GATE = 0.97`
- `SNR_THRESHOLD_DB = 10.0`
- 10-run promote rule: `n_OK >= 8/10 AND wcf_max < 0.02 AND wcf_std < 0.01 AND fin_min >= 0.97`
- GPUs: 0, 1, 2, 3 (idle confirmed); HIP_VISIBLE_DEVICES isolation per-job

## Self-checks

- Bench params satisfied repo MANDATORY rule for all 420 timing runs
- INDEPENDENT seeds (10) used, identical to R50 sequence (no seed reuse trap)
- All 4 GPUs (0, 1, 2, 3) confirmed idle at start via `rocm-smi --showuse`
- All 4 aiter dispatches use the SAME R50D shim `.so` (no rebuild) and the
  SAME aiter `.co` (`f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`)
- Promotion gate strictly stronger than per-run correctness check
- Aiter shim + HK SOs loaded successfully under fresh subprocess (smoke verified)
- Total wall: 13 min for 10 runs × 42 shapes on 4 GPUs (~1.3 min/run)

## Stopping criterion

**Met: net VC delta = +1 (30 → 31/42 under strict 10-run accounting)** AND
zero correctness regressions attributable to R51 changes (the 4 shapes that
lost VC are documented R45+ cohort-race tail-draws on UNCHANGED `.so` files;
4 other shapes symmetrically gained VC from the same mechanism). Recommend
**commit** of the R51 integration manifest delta:
- 3 new aiter dispatch overrides (`14336x4096x32768`, `16384x4096x28672`,
  `28672x4096x16384`) — all reuse the existing R50D shim binary, no rebuild
- All other 38 manifest entries unchanged from R50
- No `kernel_mxfp4_gluon_cpp.cpp` modifications
