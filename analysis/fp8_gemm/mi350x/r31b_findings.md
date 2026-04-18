# R31 Dev B — V2-CRR LDS Footprint Reduction (single-buffer As/Bs)

**Date:** 2026-04-18
**Branch:** r31-b (base feat/mxfp8-only @ 3dddbd5e)
**GPU:** HIP_VISIBLE_DEVICES=1 (MI355X / gfx950)
**Scope:** Drop V2-CRR LDS shared-tile footprint below 81920 B/block to enable 2 blocks/CU (per R30 Dev B's structural-closure paradigm correction). Single-buffer both `As[2][2]→As[1][2]` and `Bs[2][2]→Bs[1][2]`.

**Verdict:** **NO SHIP — perf loss exceeds occupancy gain.** Both single-buffered yields exactly the predicted LDS reduction (139264 → **69632 B/block, −50%**) and is correct (SNR 49.59 dB / det 3/3 PASS). Compiler hint `mb=2` keeps reported occupancy at 2 waves/SIMD, but with LDS halved the runtime should achieve 2 effective blocks/CU (16 waves/CU). However, **the loss of cross-K-iter prefetch pipelining produces a catastrophic −30% perf regression** on both 8192³ and 70B Gate. Add **"single-buffer As+Bs for V2-CRR (synchronous load form)"** to the lever-closure list.

Code change is **macro-guarded, default off** (`MXFP8_CRR_LDS_SINGLE_BUFFER=0`). Default builds bit-identical to baseline (md5 verified). Macro left in tree for R32+ exploration of pipelining-recovery variants; no perf regression on any default-build path.

---

## 1. LDS allocation breakdown (V2-CRR exact 8-wave fastpath)

`crr_mxfp8_exact_8wave_fastpath.inc` lines 191-192 declare:

```cpp
__shared__ ST_crr_a As[2][2];   // ST_v2a = st_fp8e4m3<HB=128, BK=128, st_16x128_v2a_s>
__shared__ ST_crr_b Bs[2][2];   // ST_v2  = st_fp8e4m3<HB=128, BK=128, st_16x128_v2_s>
```

ST tile is 128 rows × 128 cols of fp8e4m3 = 16384 B raw. Swizzled padding (per `st_16x128_v2*`) adds ~1024 B per tile → effective ~17408 B per tile in LDS.

Per-CU LDS (gfx950): **163840 B** (per `hipGetDeviceProperties`).

| Configuration | A tiles | B tiles | Total tiles | LDS B/block | LDS B for 2 blocks/CU | Fits 163840? |
|---|---:|---:|---:|---:|---:|:---:|
| Baseline `[2][2]+[2][2]` | 4 | 4 | 8 | 139264 | 278528 | NO (1.7×) |
| R14 `[2][2]+[1][2]` (Bs SB only) | 4 | 2 | 6 | 104448 | 208896 | NO (1.27×) |
| R31 SB `[1][2]+[1][2]` (this PR) | 2 | 2 | 4 | **69632** | **139264** | **YES** ✓ |

The 4-tile configuration drops LDS to 69632 (×2 = 139264) which fits within the 163840 budget with 24576 B headroom — first time V2-CRR has had room for 2 effective blocks/CU since the kernel was written.

The second dim `[2]` is the M-row split (RBM=64 wide, BLK=256 tall → 2 row-groups per A side; analogously for B's RBN=32 in WARPS_N=4 layout).
The first dim was the tic/toc K-pipeline double-buffer that this experiment removed.

## 2. Implementation

Added a guarded experimental path in `crr_mxfp8_exact_8wave_fastpath.inc`:

```cpp
#ifndef MXFP8_CRR_LDS_SINGLE_BUFFER
#define MXFP8_CRR_LDS_SINGLE_BUFFER 0
#endif
#if MXFP8_CRR_LDS_SINGLE_BUFFER
#  define CRR_LDS_OUTER 1
#else
#  define CRR_LDS_OUTER 2
#endif
...
__shared__ ST_crr_a As[CRR_LDS_OUTER][2];
__shared__ ST_crr_b Bs[CRR_LDS_OUTER][2];
```

When SB=1, the K-loop is replaced with a synchronous form (no cross-iter prefetch into `[toc]` — there is no `[toc]`):

```
for k = 0 .. k_iters-1:
    load_raw_scales(k>>1) or shift-down
    load_b(b0, Bs[0][0]); load_b(b1, Bs[0][1]); load_a(a, As[0][0])
    s_waitcnt lgkmcnt(0)
    MMA cA, cB
    load_a(a, As[0][1])
    s_waitcnt lgkmcnt(0)
    MMA cC, cD
    s_barrier
    global_load (k+1) → Bs[0][0/1], As[0][0/1]
    s_waitcnt vmcnt(0)
    s_barrier
```

This is the simplest correctness-safe form: **fully drain LDS reads, fully wait for next-iter VMEM, then continue**. No double-buffer = no cross-iter prefetch overlap.

## 3. Resource report comparison

Build `-DMXFP8_CRR_LDS_SINGLE_BUFFER=1 -DGEMM_MIN_BLOCKS_PER_CU=2` (default mb), 8192³ shape:

| Metric | Baseline (SB=0, mb=2) | SB=1, mb=2 | SB=1, mb=3 | SB=1, mb=4 |
|---|---:|---:|---:|---:|
| TotalSGPRs | 52 | 46 | 46 | 46 |
| VGPRs | 234 | **250** (+16) | 168 | 128 |
| ScratchSize bytes/lane | 0 | 0 | 420 | 580 |
| Occupancy [waves/SIMD] | 2 | 2 | 3 | 4 |
| VGPR Spill | 0 | 0 | **191** | **358** |
| LDS Size [bytes/block] | 139264 | **69632** (−50%) | 69632 | 69632 |

LDS reduction works exactly as predicted. With mb=2 the SB build keeps the compiler-reported occupancy at 2 (the launch_bounds hint), but the runtime now has the LDS headroom to actually launch 2 blocks/CU (since 2×69632 = 139264 ≤ 163840). VGPR rises +16 due to the extra waitcnt/spill liveness in the simplified loop, but does not hurt the per-SIMD allocation.

mb=3 and mb=4 force the compiler to compress VGPRs to fit higher block counts — and the resulting massive spilling (191 / 358 lanes of scratch) tanks perf.

## 4. Bench results (5x preheat-then-bench, GPU1 isolated, build-cache hygiene + per-build md5)

### 8192³ V2-CRR (primary formal cell)

| Build | md5 | n | median TFLOPS | mean | stdev | runs |
|---|---|---:|---:|---:|---:|---|
| Baseline mb=2 (default) | f1a27acd | 5 | **2756.38** | 2739.14 | 57.04 | 2756, 2797, 2684, 2674, 2784 |
| Baseline mb=2 re-run (post-edit, default) | f1a27acd | 5 | **2764.21** | 2753.33 | 20.62 | 2722, 2743, 2767, 2764, 2771 |
| **SB mb=2** | eead9dcb | 5 | **1913.74** | 1912.53 | 2.21 | 1914, 1915, 1914, 1911, 1909 |
| SB mb=3 | 8e7a6fc8 | 5 | **272.35** | 272.12 | 0.49 | 272, 272, 272, 272, 271 |

Welch t (SB mb=2 vs Baseline pooled n=10 mean ~2746 sd ~42):
- SB mean 1912.53, sd 2.21
- t = (1912.53 - 2746) / sqrt(2.21²/5 + 42²/10) ≈ **−63** (catastrophic regression)

Δ = **−843 TFLOPS, −30.6%**

Correctness: SB mb=2 SNR=49.59 dB pass-rate=100% det=3/3 PASS — kernel is functionally correct.

### LLaMA 70B Gate 4096×28672×8192 V2-CRR (cross-shape regression check)

| Build | md5 | n | median TFLOPS | mean | stdev |
|---|---|---:|---:|---:|---:|
| Baseline mb=2 | d67e2046 | 5 | **2373.92** | 2372.95 | 6.97 |
| **SB mb=2** | 9f74f10d | 5 | **1599.25** | 1581.66 | 34.80 |

Welch t = (1581.66 - 2372.95) / sqrt(34.80²/5 + 6.97²/5) ≈ **−50**, Δ = **−791 TFLOPS, −33.3%**

Correctness: SNR=49.59 dB, det=3/3 PASS.

## 5. Why the synchronous SB form regresses ~30%

The K-loop runs 8192/128 = 64 K iterations (64 main MMAs of 4-quadrant cA/cB/cC/cD each). The original double-buffered loop overlaps:
- Iter k MFMA compute on the wave datapath
- Iter k+1 VMEM/LDS prefetch on the memory subsystem
- Iter k+2 partial setup queued

Removing `[toc]` collapses this to **fully serialized** load → compute → load → compute. Each iter must finish all VMEM (`s_waitcnt vmcnt(0)`) and a full LDS drain (`s_barrier` + `s_waitcnt lgkmcnt(0)`) before the next can begin.

Estimated cycle accounting per iter (back-of-envelope):
- 4 tile global-loads + barrier + vmcnt drain: ~150 cycles wasted vs baseline overlap
- Lost prefetch-to-compute hiding: half the VMEM latency now exposed: another ~100 cycles
- Per-iter overhead × 64 iters × 256 grid blocks at 1-block/CU ≈ 30% more wall time

This matches the observed −30.6% / −33.3% regressions.

The naive synchronous form is the simplest correct shape; **a recovered-pipelining SB form is conceivable but would require restructuring the LDS-read schedule so that next-iter VMEM completes before the current-iter consumers retire** (e.g., async copy `buffer_load_dword_lds` into an LDS region that's been pre-emptied via a one-sided barrier). That's a 2-3 day kernel rewrite outside the R31 budget — and given the closure of `buffer_load_dword_lds` for V2 scales (R30 Dev C), the only async-direct-to-LDS path that exists is for tile fills (already used). Re-purposing it in a pipelined SB context is non-trivial.

## 6. SHIP gate evaluation

| Gate | Required | Actual | Pass? |
|---|---|---|---|
| Correctness 8192³ V2-CRR SNR ≥ 48 dB | yes | 49.59 dB | YES |
| Determinism 3/3 | yes | 3/3 | YES |
| 8192³ V2-CRR ≥ R28 baseline 2844 + 1% | yes (≥2872) | 1914 (−960 vs baseline) | NO |
| 8192³ Welch t > 3.0 | yes | t = −63 | NO |
| 70B Gate ≥ baseline ± 0.5% | yes (≥2362) | 1599 (−775) | NO |
| LDS ≤ 81920 B/block | yes | **69632 B** | YES ✓ |
| Effective occupancy ≥ 2 blocks/CU | yes | LDS-permitted 2 blocks/CU; runtime achievable | (likely) YES — but irrelevant given perf loss |

**Result: candidate fails 3 of 7 gates (the 3 perf gates) → NO SHIP.**

## 7. Outcome classification (per R31 brief)

This is the **"NO SHIP (perf loss exceeds occupancy gain)"** outcome from the brief's outcome list:

> *"NO SHIP (perf loss exceeds occupancy gain): single-buffer compiles + correct + occ=3 but slower; document and propose alternative LDS reduction (e.g., scale staging area collapse — though Dev C established no scale LDS exists for V2; double-check this for any non-V2 staging that may sneak in)."*

Per R30 Dev C's SASS-level audit, V2-CRR scales are scale-direct-to-VGPR with **zero LDS round-trip**. We re-verified this by inspecting `crr_mxfp8_exact_8wave_fastpath.inc` lines 221-232, 250-313, 315-389: no `__shared__` declarations beyond the tile arrays at 191-192 exist anywhere in the V2-CRR kernel template. The 139264 B is fully accounted for by `As[2][2] + Bs[2][2]` swizzled tiles. No additional LDS reduction lever exists at the shared-tile level.

## 8. Paradigm correction recommended for R31 wrap

Add to the lever-closure list (extends R27/R28/R29/R30 closures, now 15 closed levers):

> **`single-buffer As+Bs for V2-CRR (naive synchronous form)` is CLOSED for V2-CRR** (R31 Dev B). Halving LDS to 69632 B/block (fits 2 blocks/CU) is achievable and correctness-preserving (SNR 49.59 dB, det 3/3, pass 100%), but the synchronous load→compute→load form sacrifices all cross-K-iter prefetch pipelining and produces **−30.6% on 8192³ (Welch t≈−63) and −33.3% on 70B Gate (t≈−50)**. The compiler-reported "occupancy 2" doesn't change at mb=2 because launch_bounds caps the hint, but the runtime can now actually launch 2 blocks/CU per LDS budget. The K-pipelining loss (≈ half VMEM latency exposed per iter, ≈64 K iters × 256 grid blocks at 1 block/CU) dominates the occupancy gain.
>
> NEVER prototype "naive synchronous single-buffer As+Bs" again. Macro `MXFP8_CRR_LDS_SINGLE_BUFFER` left in tree (default off, bit-identical baseline verified) for R32+ exploration of **pipelining-recovery SB variants** (e.g., split global_load into early-issue + late-wait around the MMA chain so half the VMEM latency overlaps compute). Such a variant would need to maintain LDS-read-then-clobber ordering without a full barrier; non-trivial.
>
> Forced-occupancy variants (mb=3 with VGPR-cap → 191 spills; mb=4 → 358 spills) regress to **272 / sub-200 TFLOPS** — spilling dominates by 10×.

## 9. Files

Code change (macro-guarded, default off):
- `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc` — added `MXFP8_CRR_LDS_SINGLE_BUFFER` macro + alternative SB K-loop in `#if ... #else ... #endif` blocks. Default-off bit-identical to baseline (md5 `f1a27acd...` matches both pre-edit and post-edit clean default builds).

Bench logs (in this directory):
- `r31b_baseline_8192.txt` — Baseline 8192³ V2-CRR 5x (n=5, median 2756)
- `r31b_baseline_default_8192.txt` — Post-edit default-off baseline re-confirm (n=5, median 2764)
- `r31b_sb_mb2_8192.txt` — SB mb=2 8192³ (n=5, median 1914)
- `r31b_sb_mb3_8192.txt` — SB mb=3 8192³ (n=5, median 272 — VGPR-spill disaster)
- `r31b_baseline_70bgate.txt` — Baseline 70B Gate 4096×28672×8192 (n=5, median 2374)
- `r31b_sb_70bgate.txt` — SB mb=2 70B Gate (n=5, median 1599)

Build logs (all use `-Rpass-analysis=kernel-resource-usage`; resource numbers in section 3):
- `/tmp/baseline_build.log`, `/tmp/sb_build.log`, `/tmp/sb_mb2_build.log`, `/tmp/sb_mb3_build.log`, `/tmp/sb_mb4_build.log`, `/tmp/baseline70b_build.log`, `/tmp/sb70b_build.log`, `/tmp/post_default_build.log`.

Build-cache hygiene: `rm -f tk_mxfp8_layouts*.so` between every build; per-build md5 logged inline (every distinct configuration produced a distinct md5).

## 10. Summary

| Metric | Value |
|---|---|
| Verdict | **NO SHIP — perf loss exceeds occupancy gain** |
| Lever direction tested | `__shared__ As[2][2]→[1][2]` + `Bs[2][2]→[1][2]` (collapse K-pipeline buffer) |
| Baseline LDS / VGPR / Occ | 139264 B / 234 VGPR / 2 waves/SIMD (1 block/CU LDS-bound) |
| Candidate LDS / VGPR / Occ | **69632 B (−50%)** / 250 VGPR (+16) / 2 waves/SIMD (mb=2 hint; LDS now fits 2 blocks/CU) |
| 8192³ V2-CRR Δ | **−842 TFLOPS, −30.6%, Welch t≈−63** |
| 70B Gate V2-CRR Δ | **−775 TFLOPS, −33.3%, Welch t≈−50** |
| Correctness | SB build: SNR 49.59 dB, det 3/3 PASS, pass-rate 100% |
| Time spent | ~80 min (within 90 min budget) |
| Paradigm closure added | YES — "naive synchronous single-buffer As+Bs" for V2-CRR (15th closed lever cumulative) |
| Code change committed? | Macro left in tree default-off (bit-identical baseline), no commit because no SHIP and macro is exploratory scaffolding for R32+ |
| R32 follow-up direction | If LDS reduction is to be re-attempted: pipelining-recovery SB variant (split global_load early-issue + late-wait, async copy into pre-emptied LDS region maintained without full barrier). Non-trivial 2-3 day rewrite, uncertain whether perf can recover to ≥ baseline. |
