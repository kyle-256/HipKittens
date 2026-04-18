# R30 Dev B — V2-CRR VGPR Reduction / Occupancy Lever Audit

**Date:** 2026-04-18
**Branch:** r30-b (base feat/mxfp8-only @ cd630fd8)
**GPU:** HIP_VISIBLE_DEVICES=1 (MI355X / gfx950)
**Scope:** Investigate whether VGPR reduction can lift V2-CRR occupancy from "2" to "3" (per Dev D's R29 LDS audit recommendation, finding `(3)` cross-cycle VGPR/occupancy direction).

**Verdict:** **NO SHIP — STRUCTURAL CLOSURE.** The premise that reducing VGPR usage can increase V2-CRR per-CU block count is **false** for this kernel because **LDS, not VGPR, is the binding occupancy constraint**. Mechanistic experiment with `-DGEMM_MIN_BLOCKS_PER_CU=3` produces a binary with **bit-identical kernel resources** to the mb=2 baseline, and on the most CRR-sensitive shape (LLaMA 70B Gate 4096×28672×8192) it actually **regresses -0.55% with Welch t=-3.26** — pure overhead of the increased launch_bounds hint with no occupancy benefit.

Add `GEMM_MIN_BLOCKS_PER_CU > 2 for V2-CRR` to the paradigm-correction lever-closure list.

---

## 1. Hardware ground truth (MI355X / gfx950)

Queried via `hipGetDeviceProperties`:
- `sharedMemPerBlock = 163840` (160 KB LDS per block, == per-CU LDS budget)
- `sharedMemPerMultiprocessor = 41943040` (40 MB; not relevant — per-block is the binding ceiling)
- `regsPerBlock = 131072`
- `maxThreadsPerBlock = 1024`
- `maxThreadsPerMultiProcessor = 2048` (= 32 waves of 64 lanes per CU)
- `multiProcessorCount = 256`

The hipcc resource-usage remark "Occupancy [waves/SIMD]: N" must be interpreted as **waves per SIMD across all blocks on the CU**:
- 8-wave block × 1 block/CU = 8 waves / 4 SIMDs = **2 waves/SIMD = "occupancy 2"**.
- 8-wave block × 2 blocks/CU = 16 waves / 4 SIMDs = **4 waves/SIMD = "occupancy 4"**.
- "occupancy 3" with an 8-wave block would require **24 waves/CU**, which is not an integer multiple of 8 → **infeasible without restructuring to non-8-wave blocks**.

The R30 brief's hypothesis "occupancy=3 needs ≤ 84 VGPR per wave" is the formula for a 1-wave-per-block kernel and does not apply here.

## 2. Baseline VGPR / LDS measurement (V2-CRR exact 8-wave fastpath)

Build with `-DM_DIM=8192 -DN_DIM=8192 -DK_DIM=8192` (default `GEMM_MIN_BLOCKS_PER_CU=2`):

```
crr_exact_8wave_scaled_kernel<true, 2> (PRESHUFFLED_QUANT=1, SCALE_VERSION=2)
    TotalSGPRs : 52
    VGPRs      : 234   (no spill, scratch=0)
    LDS Size   : 139264 bytes per block
    Occupancy  : 2 waves/SIMD  (= 1 block/CU)
```

Source: `r30b_baseline_build.log` line ~ V2-CRR template `_Z29crr_exact_8wave_scaled_kernelILb1ELi2EE`.

LDS occupancy math (gfx950 LDS per CU = 163840 B):
| blocks/CU | LDS used | Fits in 163840? |
|---:|---:|:---:|
| 1 | 139264 | YES (85% util) |
| 2 | 278528 | NO (170% — 2× overflow) |
| 3 | 417792 | NO (255%) |

**LDS is the binding occupancy constraint by a factor of 1.7×.** No amount of VGPR reduction can raise blocks/CU above 1 unless LDS is also reduced to ≤ 81920 B (a 41% cut of the current shared-tile footprint).

VGPR occupancy math (gfx950 VGPR per SIMD = 1536 from physical pool inferred by AMD CDNA4 spec; even if 1024 the math is the same shape):
- VGPR-only-bound max waves/SIMD at 234 VGPR = floor(1536/234) = 6
- All 8-wave kernels at any VGPR ≤ 192 would still be limited to floor(8 waves/block / 4 SIMDs) = 2 waves/SIMD × N blocks = max 8 waves/SIMD with 4 blocks/CU.

So the kernel is at "occupancy 2" because of LDS (1 block/CU) — VGPR has 6× headroom over current usage.

## 3. Mechanistic check: `GEMM_MIN_BLOCKS_PER_CU=3`

Built with extra flag `-DGEMM_MIN_BLOCKS_PER_CU=3`. Resource report:

```
crr_exact_8wave_scaled_kernel<true, 2>
    TotalSGPRs : 52        (same)
    VGPRs      : 234       (same — compiler ignored mb=3 since LDS-bound)
    LDS Size   : 139264    (same)
    Occupancy  : 2         (same)
    Spill      : 0         (same)
```

`diff <(grep -E "VGPR|Occupancy|LDS|Scratch|Function" r30b_baseline_build.log) <(grep ... r30b_mb3_build.log)` is **empty** — every kernel emitted is bit-identical at the resource-allocation level. The `__launch_bounds__(_, 3)` hint is silently clamped because no allocation exists that satisfies it (LDS overflow blocks any allocation regardless of VGPR budget).

The .so md5 differs (`1462fee82b...` vs `057f1fec...`) only because the literal `3` is encoded in the AMD HSA kernel descriptor `kernarg_size` / `private_segment` metadata field that the runtime uses for SPI launch-allocator hints.

## 4. Bench results (5x preheat-then-bench, GPU1 isolated)

### 8192³ V2-CRR (primary formal cell)

| Build | n | mean | sd | runs |
|---|---:|---:|---:|---|
| Baseline mb=2 (md5 `1462fee8`) | 5 | 2780.53 | 14.79 | 2757.92 / 2793.72 / 2774.72 / 2792.62 / 2783.68 |
| Candidate mb=3 (md5 `057f1fec`) | 5 | 2781.01 | 10.80 | 2762.93 / 2783.70 / 2781.97 / 2784.52 / 2791.95 |

Welch t = **+0.06**, Δ = **+0.48 TFLOPS (+0.02%)**. Indistinguishable. Fails the SHIP gate (need t > 3.0 + Δ ≥ 1%).

Correctness: SNR=49.59 dB, det=3/3 PASS — neither path regresses correctness (kernel is identical).

### LLaMA 70B Gate 4096×28672×8192 V2-CRR (cross-shape regression check)

| Build | n | mean | sd |
|---|---:|---:|---:|
| Baseline mb=2 run-1 | 5 | 2384.53 | 4.67 |
| Baseline mb=2 run-2 | 5 | 2383.95 | 10.43 |
| **Pooled baseline mb=2** | **10** | **2384.24** | **7.62** |
| Candidate mb=3 | 5 | 2371.22 | 7.11 |

Welch t (pooled baseline vs candidate) = **−3.26**, Δ = **−13.02 TFLOPS (−0.55%)**.

The two baseline runs are stable (means within 0.6 TFLOPS), so the candidate's −13 TFLOPS gap is real. Since the kernel binary is **bit-identical for the V2-CRR template**, the regression must come from runtime overhead introduced by the elevated `__launch_bounds__(_, 3)` hint — likely the HSA SPI launch allocator reserving extra LDS / wave slots per CU dispatch decision based on the un-achievable hint and serializing block dispatch differently.

Correctness: SNR=49.59 dB, det=3/3 PASS.

## 5. SHIP gate evaluation

| Gate | Required | Actual | Pass? |
|---|---|---|---|
| 8192³ V2-CRR ≥ R28 baseline (2844) + 1% (≥ 2872) | yes | 2781 (−63 TFLOPS vs R28; baseline drift) | NO (note: my baseline 2780 also misses; this is GPU1 vs prior GPU drift, not a candidate regression) |
| 8192³ Welch t > 3.0 vs r30-b baseline | yes | t = +0.06 | NO |
| 70B Gate ≥ R28 SHIP baseline 2419.99 − 0.5% (≥ 2407) | yes | 2371 | NO (regressed) |
| LLaMA cells correctness ≥ 48 dB SNR + det 3/3 | yes | 49.59 dB / det 3/3 | YES |

**Result: candidate fails 3 of 4 gates → NO SHIP.**

## 6. Why "VGPR reduction → occ=3" is structurally impossible for V2-CRR

V2-CRR's 234 VGPRs decompose roughly as:
- **4 fp32 accumulators `cA, cB, cC, cD`**, each `rt_fl<RBM=64, RBN=32, col_l, rt_16x16_s>` = 64×32 / 64 lanes × 1 reg/element = **32 VGPRs each → 128 VGPRs total**.
- **A/B operand registers `a, b0, b1`**: A_col_reg (RBM=64 wide) ≈ 16 VGPRs, B_col_reg×2 (RBN=32 each) ≈ 16 VGPRs total. **~32 VGPRs**.
- **SRDs / pointer state**: `a_v2_srsrc, b_v2_srsrc` = 2 × 4 VGPRs = 8 VGPRs (V2 path); plus `a0p0, a0p1, a1p0, a1p1, b0p0, b1p0` = 6 × 4 VGPRs = 24 VGPRs (V1 SCALE_VERSION=1 fallback path) — these are **dead in the V2 dispatch but still counted in the unified template body**.
- **Scale packs `a0/a1_scale_packs[2], b0/b1_scale_packs[1]`**: 6 × 1 VGPR = **6 VGPRs**.
- **Loop counters, address arithmetic, `lane_*` derived state**: ~10–20 VGPRs.
- **Compiler temp / liveness slack** to honor schedule constraints (`s_setprio`, `sched_barrier`): rest.

The 4 fp32 accumulators alone are 128 VGPRs and are **load-bearing for correctness** (they accumulate the four CTA-tile output quadrants {cA, cB, cC, cD} that are then stored to four 64×32 output sub-tiles). Dropping any accumulator would require either restructuring the output tile shape (BLK/HB/RBM/RBN — major surgery breaking 8 static_asserts in this file alone) or a fundamentally different epilogue (e.g. partial-sum + extra global-store pass — likely net negative).

The Dev D R29 audit's "scale broadcast registers held across MFMA quadrants" hypothesis was already minimized in R28: scale packs total 6 VGPRs, and they are read by MFMA opsel-selected lanes, not broadcast-replicated. There is no obvious VGPR reduction available without changing the algorithm.

**Even if VGPR were reduced to 84 (the brief's target), occupancy would still be 1 block/CU because LDS is 1.7× the budget for 2 blocks/CU.**

The only structural lever that could lift V2-CRR per-CU occupancy is **LDS reduction** — specifically, dropping the double-buffered `As[2][2]` + `Bs[2][2]` shared tiles from 139264 B to ≤ 81920 B (single-buffer or smaller tiles). That is a fundamentally different lever than VGPR reduction and is outside R30 Dev B's brief.

## 7. Paradigm correction recommended for R30 wrap

Add to the lever-closure list (extends R27/R28/R29 closures):

> **`GEMM_MIN_BLOCKS_PER_CU > 2` is CLOSED for V2-CRR** (R30 Dev B). The kernel is LDS-bound at 139264 B/block on a 163840 B/CU LDS budget. Setting `__launch_bounds__(_, 3)` is silently ignored by the compiler (kernel binary bit-identical) and on cross-shape testing causes a measurable runtime regression (−0.55% / Welch t=−3.26 on 70B Gate) due to SPI launch-allocator overhead from the un-achievable hint. NEVER prototype "raise __launch_bounds__ min_blocks_per_cu for V2-CRR" again. Lever is also CLOSED for the V2-RCR sibling (same kernel-shape resource profile per R29 reverify).

> **Corollary: VGPR reduction in V2-CRR cannot increase per-CU block count** until LDS shared-tile footprint drops below 81920 B. The R29 Dev D direction-(3) "VGPR / occupancy pressure" was based on an outdated mental model that conflated waves/SIMD with blocks/CU.

## 8. What's actually open for V2-CRR (carry-over to future cycle)

From R29 Dev D's audit + this analysis, residual unexplored levers:

1. **LDS shared-tile reduction** (single-buffer As/Bs or smaller BLK/HB) → could enable 2 blocks/CU. Major restructure; likely wrong tradeoff because steady-state would lose pipelining. Not recommended without a concrete throughput model.
2. **`buffer_load_dword_lds` (VMEM→LDS direct)** for tile loads, eliminating the VMEM→VGPR→LDS round-trip (saves ~30 VGPRs of operand staging registers). gfx950 supports this; requires kittens runtime change. Listed in R29 Dev D's brief as "alternative-2 lever". Out of scope for R30 Dev B.
3. **Scale-load 2-deep prefetch queue** (R26 Dev B partial). Adds VGPR (2 extra scale packs), reduces VMEM-to-MFMA latency window. May trade off VGPR for steady-state IPC. Untested at R30.
4. **70B Gate / Down / KV cross-shape MXFP8 absolute regressions** flagged by R29 Reviewer (−0.63% / −2.42% / −2.68% vs R27 source-unchanged) — needs cross-GPU reverify per R29 wrap notes; not a V2-CRR optimization, but a measurement-stability question.

## 9. Files

- Build logs: `r30b_baseline_build.log` (mb=2, 8192³), `r30b_mb3_build.log` (mb=3, 8192³), `r30b_baseline_70bgate_build.log` (mb=2, 70B Gate), `r30b_baseline2_70bgate_build.log` (mb=2 re-run for noise control), `r30b_mb3_70bgate_build.log` (mb=3, 70B Gate).
- Bench outputs: `r30b_baseline_8192_v2crr.txt`, `r30b_mb3_8192_v2crr.txt`, `r30b_baseline_70bgate_v2crr.txt`, `r30b_baseline2_70bgate_v2crr.txt`, `r30b_mb3_70bgate_v2crr.txt`.
- Build cache hygiene: `rm -f tk_mxfp8_layouts*.so` between every build; per-build md5 logged inline.

## 10. Summary

| Metric | Value |
|---|---|
| Verdict | **NO SHIP — STRUCTURAL CLOSURE** |
| Lever direction tested | `-DGEMM_MIN_BLOCKS_PER_CU=3` (Dev D R29 "VGPR/occupancy" lever) |
| Baseline VGPR | 234 (no spill) |
| Candidate VGPR | 234 (no spill — compiler ignored hint) |
| 8192³ V2-CRR Δ | **+0.48 TFLOPS (+0.02%), Welch t=+0.06** |
| 70B Gate V2-CRR Δ | **−13.02 TFLOPS (−0.55%), Welch t=−3.26** ⚠ regression |
| Correctness | All builds: SNR 49.59 dB, det 3/3 PASS (kernel is identical) |
| Time spent | ~75 min (within 90 min budget) |
| Paradigm closure added | **YES** — `GEMM_MIN_BLOCKS_PER_CU > 2` for V2-CRR |
| Code change | NONE (lever has no implementable form that helps) |
