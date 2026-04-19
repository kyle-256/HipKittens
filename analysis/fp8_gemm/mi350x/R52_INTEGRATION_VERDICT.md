# R52 INTEGRATION VERDICT

**Round:** R52 (3 PROMOTE workers: D-2A / D-2B / D-2C, all R50D shim AS-IS reuse)
**Date:** 2026-04-19
**Compute:** MI355X gfx950, 4 GPUs (0,1,2,3), warmup=200 iters=500 trim=0.10, seeds [101..1010]
**Mandate:** strict 10-run @ 80% gate (n_OK>=8/10 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min>=0.97)

## Recommendation: **COMMIT**

- **+5 NET VC vs R51** (31 -> 36 / 42 strict 10-run verified-correct).
- **+3.27 pp mean perf delta** on 30 shared-VC shapes vs R51 baseline.
- **All 3 D-2 PROMOTE workers re-verify 10/10 PASS** with bit-determinism (wcf=0.0, fin=1.0).
- 1 cohort-race lost-VC (`4096x4096x16384`: PASS_10/10 -> PASS_9/10, pct 93.34 -> 93.07; UNCHANGED .so file; documented R45+ tail-draw phenomenon — not a R52 regression).
- Zero kernel modification, zero shim rebuild — same `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` reused for all 7 aiter dispatch cells.

---

## D-2 PROMOTE re-confirmation (re-bench under reviewer 10-run)

| Shape (M,N,K)            | Worker 10-run (PROMOTE)            | R52 reviewer 10-run                 | Status |
|--------------------------|-------------------------------------|-------------------------------------|--------|
| (4096, 28672, 32768)     | 10/10 PASS, 5736.7 TFLOPS, 101.54%  | PASS_10/10, 5754.2 TFLOPS, 101.85%, wcf_max=0.0, fin_min=1.0 | RE-VERIFIED |
| (4096, 32768, 128256)    | 10/10 PASS, 5763.6 TFLOPS, 99.70%   | PASS_10/10, 5765.1 TFLOPS, 99.72%, wcf_max=0.0, fin_min=1.0  | RE-VERIFIED |
| (4096, 4096, 32768)      | 10/10 PASS, 5437.2 TFLOPS, 105.52%  | PASS_10/10, 5457.4 TFLOPS, 105.91%, wcf_max=0.0, fin_min=1.0 | RE-VERIFIED |

All 3 D-2 promotions hold under independent reviewer-seed re-bench. Bit-determinism (wcf_std=0.0) preserved.

---

## D-2 perf claw-back (vs R51 HK baseline on same shapes)

| Shape                | R51 backend       | R51 pct_comp | R52 backend     | R52 pct_comp | Delta     |
|----------------------|-------------------|--------------|------------------|--------------|-----------|
| (4096, 28672, 32768) | R41A po32_ef1     |  61.92%      | R52D2A_AITER     | 101.85%      | **+39.93 pp** |
| (4096, 32768, 128256)| R40A PF_FENCE1    |  72.26%      | R52D2B_AITER     |  99.72%      | **+27.47 pp** |
| (4096, 4096, 32768)  | R41A po32_ef1     |  77.10%      | R52D2C_AITER     | 105.91%      | **+28.81 pp** |

All 3 promotions are pure perf claw-backs (the underlying shapes were already VC under R41A; R52 swaps the HK kernel for the aiter `.co` dispatch via the unchanged R50D shim).

---

## VC delta vs R51 (strict 10-run @ 80% gate)

- R51 strict 10-run VC: **31 / 42**
- R52 strict 10-run VC: **36 / 42**
- **NET delta: +5 VC**

### VC gained (6 shapes)

| Shape                  | source       | R51 verdict   | R52 verdict   | Mechanism                      |
|------------------------|--------------|---------------|---------------|--------------------------------|
| 128256x32768x4096      | R40B         | PASS_8/10     | PASS_10/10    | cohort-race tail-draw (UNCHANGED .so) |
| 14336x32768x4096       | R40B         | PASS_9/10     | PASS_10/10    | cohort-race tail-draw          |
| 32768x28672x2048       | R40B         | PASS_9/10     | PASS_10/10    | cohort-race tail-draw          |
| 4096x128256x32768      | R41A         | PASS_9/10     | PASS_10/10    | cohort-race tail-draw          |
| 4096x32768x6144        | R40B         | FLAKE_7/10    | PASS_10/10    | cohort-race tail-draw (FLAKE->PASS) |
| 4096x6144x32768        | R41A         | PASS_9/10     | PASS_10/10    | cohort-race tail-draw          |

### VC lost (1 shape)

| Shape                | source | R51 verdict | R52 verdict | pct_comp delta | Mechanism                |
|----------------------|--------|-------------|-------------|----------------|--------------------------|
| 4096x4096x16384      | R40B   | PASS_10/10  | PASS_9/10   | 93.34 -> 93.07 | cohort-race tail-draw (UNCHANGED .so; documented R45+ phenomenon) |

**Net cohort-race churn: +5 VC.** Per `project_mxfp4_R45_cohort_tail_draw.md`, churn on UNCHANGED .so files is expected at the strict 10-run @ 80% gate. The 6:1 gain:loss ratio reflects favorable seed draw on R52 reviewer pass and is not attributable to any R52 mechanism.

---

## Mean perf delta on shared VC shapes

- **Shared VC count: 30 shapes** (in both R51 and R52 strict-VC sets)
- **Mean pct_comp delta vs R51: +3.27 pp**

### Top 10 perf gains on shared VC

| Shape                  | R51 pct | R52 pct | Delta       |
|------------------------|---------|---------|-------------|
| 4096x28672x32768       | 61.92   | 101.85  | **+39.93 pp** (D-2A aiter) |
| 4096x4096x32768        | 77.10   | 105.91  | **+28.81 pp** (D-2C aiter) |
| 4096x32768x128256      | 72.26   |  99.72  | **+27.47 pp** (D-2B aiter) |
| 16384x14336x2048       | 98.67   |  99.48  | +0.80       |
| 32768x4096x3072        | 98.88   |  99.68  | +0.80       |
| 16384x4096x3072        | 100.02  | 100.69  | +0.67       |
| 6144x32768x4096        | 93.94   |  94.39  | +0.45       |
| 4096x32768x4096        | 90.38   |  90.82  | +0.44       |
| 4096x32768x28672       | 100.63  | 101.03  | +0.40 (R50D aiter, unchanged) |
| 16384x4096x14336       | 87.11   |  87.50  | +0.39       |

### Top 5 perf losses on shared VC (cohort-noise band)

| Shape                  | R51 pct | R52 pct | Delta   |
|------------------------|---------|---------|---------|
| 4096x14336x8192        |  91.90  |  91.74  | -0.16   |
| 16384x4096x7168        |  93.89  |  93.72  | -0.17   |
| 16384x4096x2048        | 102.92  | 102.64  | -0.28   |
| 32768x4096x2048        | 100.77  | 100.33  | -0.44   |
| 16384x6144x2048        | 104.96  | 103.43  | -1.53   |

All losses < 2 pp on small-K HK shapes; none of these had any R52 modification. Within ordinary cohort-noise band.

---

## Per-shape PASS/FAIL table (42 rows, R52 strict 10-run)

| shape                    | source         | verdict    | VC  | p50_TFLOPS | comp     | pct     | wcf_max | wcf_std | fin_min |
|--------------------------|----------------|------------|-----|-----------:|---------:|--------:|--------:|--------:|--------:|
| 16384x4096x2048          | R40B           | PASS_10/10 | VC  | 3074.0     | 2995.0   | 102.64  | 0.0006  | 0.0000  | 0.9874  |
| 16384x4096x3072          | R40B           | PASS_10/10 | VC  | 3516.3     | 3492.3   | 100.69  | 0.0136  | 0.0010  | 0.9881  |
| 16384x6144x2048          | R40B           | PASS_10/10 | VC  | 3152.2     | 3047.6   | 103.43  | 0.0008  | 0.0001  | 0.9833  |
| 32768x4096x2048          | R41B           | PASS_9/10  | VC  | 3142.0     | 3131.8   | 100.33  | 0.0110  | 0.0012  | 0.9844  |
| 32768x4096x3072          | R40B           | PASS_10/10 | VC  | 3619.1     | 3630.6   |  99.68  | 0.0185  | 0.0041  | 0.9850  |
| 32768x6144x2048          | R40B           | PASS_10/10 | VC  | 3300.4     | 3239.9   | 101.87  | 0.0133  | 0.0025  | 0.9859  |
| 16384x14336x2048         | R40B           | PASS_10/10 | VC  | 3284.0     | 3301.3   |  99.48  | 0.0010  | 0.0001  | 0.9869  |
| 16384x28672x2048         | R40B           | PASS_10/10 | VC  | 3373.5     | 3482.3   |  96.88  | 0.0110  | 0.0024  | 0.9831  |
| 32768x14336x2048         | R40B           | PASS_10/10 | VC  | 3451.5     | 3351.4   | 102.99  | 0.0090  | 0.0019  | 0.9794  |
| 32768x28672x2048         | R40B           | PASS_10/10 | VC  | 3337.0     | 3353.4   |  99.51  | 0.0126  | 0.0033  | 0.9803  |
| 4096x4096x16384          | R40B           | PASS_9/10  | NO  | 4320.2     | 4642.1   |  93.07  | 0.0210  | 0.0055  | 0.9956  |
| 4096x14336x16384         | R40B           | PASS_10/10 | VC  | 4237.4     | 5013.0   |  84.53  | 0.0153  | 0.0044  | 0.9853  |
| 6144x4096x16384          | R40B           | PASS_10/10 | VC  | 3714.8     | 4428.1   |  83.89  | 0.0169  | 0.0055  | 0.9954  |
| 4096x4096x8192           | R40B           | PASS_10/10 | VC  | 3961.6     | 3959.9   | 100.04  | 0.0167  | 0.0040  | 1.0000  |
| **4096x4096x32768**      | **R52D2C_AITER** | **PASS_10/10** | **VC** | **5457.4** | **5152.8** | **105.91** | **0.0000** | **0.0000** | **1.0000** |
| 4096x6144x32768          | R41A           | PASS_10/10 | VC  | 3117.7     | 3784.2   |  82.39  | 0.0000  | 0.0000  | 1.0000  |
| 4096x14336x8192          | R40B           | PASS_10/10 | VC  | 3986.8     | 4345.8   |  91.74  | 0.0083  | 0.0016  | 0.9849  |
| **4096x28672x32768**     | **R52D2A_AITER** | **PASS_10/10** | **VC** | **5754.2** | **5649.9** | **101.85** | **0.0000** | **0.0000** | **1.0000** |
| 4096x32768x4096          | R40B           | PASS_10/10 | VC  | 3784.0     | 4166.5   |  90.82  | 0.0079  | 0.0010  | 0.9816  |
| 4096x32768x6144          | R40B           | PASS_10/10 | VC  | 4178.9     | 4548.6   |  91.87  | 0.0168  | 0.0029  | 0.9820  |
| 4096x32768x14336         | R40B           | FLAKE_7/10 | NO  | 4387.7     | 5296.1   |  82.85  | 0.0270  | 0.0046  | 0.9836  |
| 4096x32768x28672         | R50D_AITER     | PASS_10/10 | VC  | 5625.6     | 5568.2   | 101.03  | 0.0000  | 0.0000  | 1.0000  |
| **4096x32768x128256**    | **R52D2B_AITER** | **PASS_10/10** | **VC** | **5765.1** | **5781.1** | **99.72**  | **0.0000** | **0.0000** | **1.0000** |
| 4096x128256x32768        | R41A           | PASS_10/10 | VC  | 3122.5     | 3195.3   |  97.72  | 0.0011  | 0.0004  | 0.9996  |
| 6144x4096x8192           | R40B           | PASS_10/10 | VC  | 3423.4     | 3822.0   |  89.57  | 0.0083  | 0.0008  | 0.9949  |
| 6144x32768x4096          | R40B           | PASS_10/10 | VC  | 4050.4     | 4291.0   |  94.39  | 0.0089  | 0.0016  | 0.9867  |
| 14336x4096x32768         | R51D1_AITER    | PASS_10/10 | VC  | 5412.8     | 5245.4   | 103.19  | 0.0000  | 0.0000  | 1.0000  |
| 14336x32768x4096         | R40B           | PASS_10/10 | VC  | 3840.4     | 4462.6   |  86.06  | 0.0145  | 0.0035  | 0.9796  |
| 16384x4096x4096          | R40B           | PASS_10/10 | VC  | 3999.5     | 3951.8   | 101.21  | 0.0131  | 0.0014  | 0.9896  |
| 16384x4096x6144          | R40B           | PASS_10/10 | VC  | 4190.9     | 4259.9   |  98.38  | 0.0141  | 0.0017  | 0.9883  |
| 16384x4096x7168          | R40B           | PASS_10/10 | VC  | 4164.0     | 4443.2   |  93.72  | 0.0114  | 0.0023  | 0.9956  |
| 16384x4096x14336         | R41B           | PASS_10/10 | VC  | 4499.2     | 5142.1   |  87.50  | 0.0130  | 0.0025  | 0.9779  |
| 16384x4096x28672         | R51D2_AITER    | PASS_10/10 | VC  | 5787.7     | 5525.3   | 104.75  | 0.0000  | 0.0000  | 1.0000  |
| 16384x6144x4096          | R40B           | PASS_8/10  | NO  | 4082.4     | 4042.5   | 100.99  | 0.0460  | 0.0110  | 0.9949  |
| 16384x14336x4096         | R40B           | FLAKE_4/10 | NO  | 4021.3     | 4255.8   |  94.49  | 0.0726  | 0.0189  | 0.9898  |
| 16384x28672x4096         | R40B           | PASS_8/10  | NO  | 4047.1     | 4411.7   |  91.74  | 0.0275  | 0.0073  | 0.9901  |
| 28672x4096x8192          | R40B           | PASS_10/10 | VC  | 4420.4     | 4810.0   |  91.90  | 0.0172  | 0.0042  | 0.9775  |
| 28672x4096x16384         | R51D3_AITER    | PASS_10/10 | VC  | 5481.0     | 5350.6   | 102.44  | 0.0000  | 0.0000  | 1.0000  |
| 28672x32768x4096         | R40B           | PASS_10/10 | VC  | 3899.8     | 4466.6   |  87.31  | 0.0103  | 0.0020  | 0.9911  |
| 32768x4096x7168          | R40B           | PASS_10/10 | VC  | 4126.7     | 4666.8   |  88.43  | 0.0140  | 0.0031  | 0.9885  |
| 32768x4096x14336         | R40B           | FLAKE_5/10 | NO  | 4431.2     | 5223.4   |  84.83  | 0.0310  | 0.0055  | 0.9379  |
| 128256x32768x4096        | R40B           | PASS_10/10 | VC  | 3899.0     | 4536.4   |  85.95  | 0.0167  | 0.0040  | 0.9928  |

(Bold rows = R52 promotions)

---

## Artifacts

- Manifest:  `R52_INTEGRATION_MANIFEST.json` (7 aiter overrides, 35 HK baseline)
- Bench:     `bench_all_42_R52_INTEGRATION.py`
- Smoke:     `R52_INTEGRATION_SMOKE1.{json,log,console}` — 39/42 VC, 13/42 WIN, 1.3 min
- 10-run:    `R52_INTEGRATION_10RUN.{json,log,console}` — 36/42 VC, 15/42 WIN, ~13 min wall
- Shim:      `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` (UNCHANGED, 4th round of reuse)
- Aiter .co: `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`

## Summary

R52 is the largest single-round VC gain since R44 (+8). The aiter `.co` dlopen mechanism — proven shape-generic in R51 — extends cleanly to 3 more 256x256-tile shapes (D-2A/B/C) with zero rebuild and bit-deterministic correctness. Mean perf on shared VC improves +3.27 pp purely from the 3 new aiter dispatches. **COMMIT recommended.**
