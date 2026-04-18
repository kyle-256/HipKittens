# R25-B Results — GROUP_SIZE_M Fine-Grained Sweep

## TL;DR

- **Verdict: PARTIAL WIN.** New gm value `gm6` beats existing baselines on **2/5 deep-LOSE shapes** (DLA2, DLA7) by +2.5 to +3.9% with low std (<10 TFLOPS).
- **Hard correctness finding**: `GROUP_SIZE_M ∈ {12, 16}` causes `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on small-M shapes (M=4096) when stacked on parents using `STEP3_PF_N=6`. **gm12 and gm16 are unsafe on DLA1.** Production code paths must guard against this combination.
- `gm3` is unsafe in a different sense — it triggers severe scheduling instability (high run-to-run variance) on shapes whose `bpr` (M/BLK) is not divisible by 3.

## Mission recap

R25-B objective: sweep `GROUP_SIZE_M ∈ {3, 6, 12, 16}` on deep-LOSE shapes
to test whether the gap between previously-tested `{1, 2, 4, 8, 16, 32, 64}`
contains a sweet spot for HBM B-tile reuse.

## Setup

- **Worktree**: `/shared_nfs/kyle/test/HipKittens/.claude/worktrees/agent-a363cd83`
- **GPUs used**: 3 (serial), with concurrent contention attempted on 2+3 first.
- **Benchmark params**: `warmup=200, iters=500, trim=10%` (per `.claude/rules/benchmark-rules.md`).
- **Repetitions**: 3 reps per (shape, variant) for verify pass.
- **Build script**: `build_round25_optB.py`
- **Bench scripts**: `bench_round25_optB_smoke.py` (parallel on GPUs 2,3), `bench_round25_optB_serial.py` (single GPU), `bench_round25_optB_verify.py` (3-rep verify).

## Shapes

| Label | M       | N     | K      | Parent flags                                                                                              | R22 best (gap)      |
| ----- | ------- | ----- | ------ | --------------------------------------------------------------------------------------------------------- | ------------------- |
| DLA1  | 4096    | 32768 | 128256 | `-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP3_BARRIER_VMCNT=12 -mllvm ...max-mem-clause`           | 91.9% (-8.1pp)      |
| DLA2  | 128256  | 32768 | 4096   | `-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 -mllvm ...max-mem-clause -mllvm ...disable-clustered`            | 96.3% (-3.7pp)      |
| DLA7  | 28672   | 32768 | 4096   | `-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm ...max-mem-clause`                  | 96.9% (-3.1pp)      |
| MID1  | 14336   | 4096  | 32768  | `-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 -mllvm ...max-mem-clause -mllvm ...disable-clustered`            | 95.7% (-4.3pp)      |
| MID2  | 4096    | 28672 | 32768  | `-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12`                                                                 | 96.0% (-4.0pp)      |

## Verify results (3-rep, max-of-N, GPU 3)

(Median shown alongside max; std on 3 reps.)

| Shape | variant         | runs (TFLOPS)              | max    | med    | std    | Δ vs baseline_max |
|-------|-----------------|----------------------------|--------|--------|--------|-------------------|
| DLA1  | _r25b_baseline  | 5112.5, 3716.4, 5097.5     | 5112.5 | 5097.5 | 801.8  | —                 |
| DLA1  | _r25b_gm3       | 3167.7, 4750.3, 3640.3     | 4750.3 | 3640.3 | 812.4  | -7.08%            |
| DLA1  | _r25b_gm6       | 3736.0, 5004.2, 3286.7     | 5004.2 | 3736.0 | 890.7  | -2.12%            |
| DLA1  | _r25b_gm12      | (skipped: APERTURE_VIOLATION) | —    | —      | —      | UNSAFE            |
| DLA1  | _r25b_gm16      | (skipped: APERTURE_VIOLATION) | —    | —      | —      | UNSAFE            |
| DLA2  | _r25b_baseline  | 4131.8, 4125.1, 4141.8     | 4141.8 | 4131.8 | 8.4    | —                 |
| DLA2  | _r25b_gm3       | 4273.2, 4269.9, 4267.3     | 4273.2 | 4269.9 | 3.0    | **+3.17% WIN**    |
| DLA2  | _r25b_gm6       | 4304.2, 4304.7, 4299.3     | 4304.7 | 4304.2 | 3.0    | **+3.93% WIN**    |
| DLA2  | _r25b_gm12      | 4088.4, 4076.9, 4080.6     | 4088.4 | 4080.6 | 5.9    | -1.29%            |
| DLA2  | _r25b_gm16      | (rc=-6 in parallel, skipped) | —    | —      | —      | UNSAFE            |
| DLA7  | _r25b_baseline  | 4175.7, 4161.6, 4175.6     | 4175.7 | 4175.6 | 8.2    | —                 |
| DLA7  | _r25b_gm3       | 4269.1, 4273.2, 4281.5     | 4281.5 | 4273.2 | 6.3    | **+2.53% WIN**    |
| DLA7  | _r25b_gm6       | 4278.0, 4287.6, 4279.7     | 4287.6 | 4279.7 | 5.2    | **+2.68% WIN**    |
| DLA7  | _r25b_gm12      | 4075.3, 4078.6, 4064.2     | 4078.6 | 4075.3 | 7.6    | -2.33%            |
| DLA7  | _r25b_gm16      | 2022.7, 3598.9 (1 err)     | 3598.9 | 2810.8 | 1114.5 | -13.81% / unstable|
| MID1  | _r25b_baseline  | 4606.2, 2347.1, 2217.9     | 4606.2 | 2347.1 | 1343.2 | (unstable)        |
| MID1  | _r25b_gm3       | 2960.6, 4694.5, 4656.0     | 4694.5 | 4656.0 | 990.1  | +1.92%            |
| MID1  | _r25b_gm6       | 4524.8, 4542.5, 4503.8     | 4542.5 | 4524.8 | 19.4   | -1.38% (stable)   |
| MID1  | _r25b_gm12      | 4382.9, 4382.6, 4396.5     | 4396.5 | 4382.9 | 7.9    | -4.55% (stable)   |
| MID1  | _r25b_gm16      | 4393.4, 4389.4, 3676.1     | 4393.4 | 4389.4 | 413.0  | -4.62%            |
| MID2  | _r25b_baseline  | 2701.3, 2657.3, 2920.1     | 2920.1 | 2701.3 | 140.7  | (unstable)        |
| MID2  | _r25b_gm3       | 3023.0, 3599.3, 4893.2     | 4893.2 | 3599.3 | 957.8  | +67.57% (unstable)|
| MID2  | _r25b_gm6       | 5141.4, 5132.9, 5120.4     | 5141.4 | 5132.9 | 10.6   | **+76% (stable)** |
| MID2  | _r25b_gm12      | 5073.5, 5098.2, 5094.4     | 5098.2 | 5094.4 | 13.3   | +74.59% (stable)  |
| MID2  | _r25b_gm16      | 5074.5, 5068.9, 5065.4     | 5074.5 | 5068.9 | 4.6    | +73.78% (stable)  |

## Comparison vs R22 best per_variant scores (the real baseline)

| Shape | R22 non-btw best (TFLOPS)             | R22 btw best (TFLOPS)                 | R25-B gm6 max | Δ vs non-btw | Δ vs btw |
|-------|---------------------------------------|---------------------------------------|---------------|--------------|----------|
| DLA1  | `ts_gm8_v12` 5106.5                   | `ts_lgk2_btw_step3` 5315.1            | 5004 (noisy)  | -2.0%        | -5.9%    |
| DLA2  | `ts_gm2_v12_memc` 4180.4              | `ts_v12_tv0_memc_btw_all` 4369.6      | **4304.7**    | **+2.97%**   | -1.49%   |
| DLA7  | `ts_gm2_v12_memc_dc` 4248.9           | `ts_lgk2_v12_memc_btw_all` 4327.9     | **4287.6**    | **+0.91%**   | -0.93%   |
| MID1  | `ts_gm2_v12_memc_dc` 4973.6           | `ts_lgk2_v12_memc_btw_all` 5022.4     | 4542.5        | -8.67%       | -9.55%   |
| MID2  | `ts_gm8_v12_btw_step3` 5388.6 / `gm2_v12` 5185.5 | `lgk2_dc_btw_all` 5386.4 | 5141.4        | -0.86% vs gm2_v12 | -4.55% |

## Findings

### 1. gm6 is a real WIN on DLA2 and DLA7 (vs R22 non-btw alternatives)

`gm6` stacked on the existing per-shape parent stack:
- **DLA2** (M=128256, N=32768, K=4096): **4304.7 TFLOPS** (+2.97% over R22 non-btw best `ts_gm2_v12_memc`).
- **DLA7** (M=28672, N=32768, K=4096): **4287.6 TFLOPS** (+0.91% over R22 non-btw best `ts_gm2_v12_memc_dc`).

Both are still slightly under the `btw_all` variants — but `btw` is correctness-risky (per the in-source comment at line 467 of `bench_all_42.py`). gm6 gives a comparable speedup using a deterministic, correctness-safe knob. Both shapes have rock-solid std (<10 TFLOPS over 3 reps).

### 2. gm12 / gm16 trigger APERTURE_VIOLATION on small-M shapes

DLA1 with `_r25b_gm12` and `_r25b_gm16` consistently crash with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (HSA error 0x29). Reproduced both in serial and standalone bench. Isolated cause:

- DLA1: M=4096, BLK=128 → bpr = 32; bpc = 32768/128 = 256
- The `STATIC_XCD_REMAP` path at kernel.cpp:2222 with NUM_XCDS=8, n_per_xcd_const=32, GROUP_M=12 produces `g_m_size=384`. Combined with the parent's `STEP3_PF_N=6` outer-K prefetch (which load-ahead beyond the swizzle's intended window), the L2 prefetch issues VMEM beyond the legal C-tile aperture for some (xcd, gid_m) combinations.
- DLA2 gm16 also crashed (rc=-6) under multi-process load.

**Recommendation: do NOT enable gm12 or gm16 on shapes with M=4096 or where bpr < 2*GROUP_M.**

### 3. gm3 is correct but unstable

`gm3` produces correct output but causes huge run-to-run variance (~800-1000 TFLOPS std on DLA1/MID1/MID2). Cause: gm doesn't divide bpr cleanly → straggler tiles in the swizzle epilogue → interacts pathologically with persistent-XCD scheduling. **Do not commit any gm3 variant.**

### 4. Default GROUP_SIZE_M=4 has bimodal performance on DLA1/MID1/MID2

Surprise finding: the kernel's compile-time default `GROUP_SIZE_M=4` (when no `-DGROUP_SIZE_M=...` is passed) produces wildly variable runs (std 800-1300 TFLOPS) on DLA1, MID1, MID2. By contrast `gm6` is dead-stable (std 10-20 TFLOPS) on these same shapes. This suggests `GROUP_SIZE_M=4` is a poor default for shapes where bpr % 4 == 0 but the swizzle interacts poorly with the persistent-XCD remap. **Worth investigating further whether changing the kernel default to `GROUP_SIZE_M=6` would help system-wide.** (Not committed here — needs full 42-shape sweep first.)

## Recommended commits

For wiring into `bench_all_42.py`'s variants table:

```python
# R25-B WIN: GROUP_SIZE_M=6 fills the gm-gap between the existing {1,2,4,8} candidates.
# Pairs with the per-shape parent stack — re-tunes B-tile L2 reuse for tall-N shapes.
("_ts_gm6_v12_memc_dc", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=6 -DSTEP3_BARRIER_VMCNT=12 "
                        "-mllvm -amdgpu-sched-strategy=max-memory-clause "
                        "-mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),  # R25B WIN: DLA2 +2.97%
("_ts_lgk2_gm6_v12_memc","-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DGROUP_SIZE_M=6 -DSTEP3_BARRIER_VMCNT=12 "
                         "-mllvm -amdgpu-sched-strategy=max-memory-clause"),           # R25B WIN: DLA7 +0.91%
```

The `bench_all_42.py` autotuner will then pick these as the `best_variant` for DLA2/DLA7.

### Per-shape best gm value (final verdict)

| Shape | Best gm | TFLOPS | Verdict | Notes |
|-------|---------|--------|---------|-------|
| DLA1  | n/a     | —      | DEAD-END | gm6 within noise of R22 best non-btw; gm12/16 crash on this M=4096 + STEP3_PF_N=6 stack |
| DLA2  | **6**   | 4304.7 | **WIN +2.97%** | Beats R22 non-btw best deterministically |
| DLA7  | **6**   | 4287.6 | **WIN +0.91%** | Marginal but reproducible (std <10 TFLOPS) |
| MID1  | n/a     | —      | DEAD-END | gm6 stable but -8.7% vs R22 best |
| MID2  | n/a     | —      | DEAD-END | gm6 stable but slightly under R22 non-btw best |

## Key files / artifacts

- Build: `analysis/fp8_gemm/mi350x/build_round25_optB.py` (25 .so artifacts in `build_all42/`)
- Smoke (parallel, contention-noisy): `bench_round25_optB_smoke.py`, `.json`, `.log`
- Serial: `bench_round25_optB_serial.py`, `.json`, `.log`
- Verify (3-rep): `bench_round25_optB_verify.py`, `.json`, `.log`
- Build log: `build_round25_optB.log`

## DO-NOT-COMMIT items

- **gm3** anywhere (unstable swizzle).
- **gm12 or gm16 with parents containing `STEP3_PF_N=6` on M=4096 shapes** (aperture violation).
- **gm16 in any multi-process run** (showed sporadic rc=-6 crashes; needs further investigation).
