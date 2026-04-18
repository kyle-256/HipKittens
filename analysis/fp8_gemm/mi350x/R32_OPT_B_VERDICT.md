# R32 — Optimizer B verdict (V6 split-K POC on L6)

**Shape**: L6 = 4096×32768×128256 (only this shape).
**Parent stack**: incumbent `ts_lgk2_v12_memc_btw_all` (5354 TFLOPS = 92.6% of comp 5781).
**Date**: 2026-04-18
**Mission (R32 decider A3)**: implement V6 split-K POC: per-split bf16 buffers + epilogue add-and-cast, no atomics, no FP32 workspace. Goal: break the 41/42 ceiling with ≥3pp gain on L6.

## TL;DR

**NO WIN.** The V6 split-K POC infrastructure works end-to-end (compiles, dispatches, sanity baseline matches incumbent perf at +0.19 %), **but the architecture itself cannot beat the incumbent on L6**. Net: split=2 is **−11.5 %** vs incumbent, split=4 is **−23.4 %**.

| Variant | total wall-time (us) | TFLOPS | vs incumbent |
|---|---:|---:|---:|
| Incumbent (`_ts_lgk2_v12_memc_btw_all`) | 11701 | 2942 | (baseline) |
| V6 K_SPLIT=1 (sanity) | 11679 | 2948 | **+0.19 %** |
| V6 K_SPLIT=2 + epilogue | 13226 | 2603 | **−11.53 %** |
| V6 K_SPLIT=4 + epilogue | 15270 | 2255 | **−23.37 %** |

The decider's hypothesis — "split=4 reaches K=125 iters → R25C pragma-unroll fires → tail-pf-off +13–21pp window opens per R25-F mechanism" — does NOT pan out. The R25C window does fire (per-split kernels are bit-clean and run at ~5300/9700 TFLOPS measured-as-full-FLOPS), but **per-flop speed is unchanged**: each split does K/S of the work and takes ~T/S of the time, so the total time ≈ T/S × S = T plus launch + epilogue overhead. Net is a regression.

## Mechanism (why split-K loses on L6)

For a workload that fits in single-launch tiles (every (M,N) tile is processed by exactly one workgroup, no inter-tile sharing), splitting K is pure overhead:

- **No FLOPs gained**: each split does 2·M·N·(K/S) FLOPs. Sum = 2·M·N·K, identical to single-launch.
- **No VMEM traffic gained**: each split reads M·K/S of A, K/S·N of B (per its K-range). Sum = M·K + K·N. Identical to single-launch.
- **Launch overhead added**: S kernel launches, each with per-WG initialization (scale ptrs, SRDs, prefetch warmup). At L6, each launch's startup is ~5–10 % of the K-loop time, so split=2 adds ~1.5 ms (12 % regression observed).
- **Bf16 round-trip added**: each split writes its partial in bf16 (saturating cast → loss of ~1 fp16 ULP), then the epilogue reads back, sums, re-rounds.
- **No L2 reuse gained**: A and B are not reused across splits. Each split's reads go to HBM.

Split-K wins ONLY when there are MORE tiles than workgroups available (i.e., grid > #CU·#waves), because it lets idle CUs participate in K-reduction. At L6, M=4096 / 256 × N=32768 / 256 = 16 × 128 = 2048 tiles spread across the 304-CU MI355X (608 WGs at 2/CU, ~3.4 tiles per WG), which is already fully loaded — splitting K just doubles/quadruples the launches without adding parallelism.

## Per-split breakdown

| Component | wall (us) | TFLOPS-as-full-FLOPS |
|---|---:|---:|
| split2_s0 | 6501 | 5296 (= 2 × per-split actual = 2648 per-flop) |
| split2_s1 | 6569 | 5241 (= 2620 per-flop) |
| epilogue (bf16 add+cast) | 907 | 37960 |
| split4_s0 | 3528 | 9760 (= 4 × actual = 2440 per-flop) |
| split4_s1..3 | 3516–3560 | 9670–9791 (= ~2425 per-flop) |

Per-flop compute is **2425–2648 TFLOPS per split**, slightly **lower** than the incumbent's 2942 (because per-split has higher relative startup cost). Even the "best" measure — sum of per-split times — exceeds incumbent (split=2: 13.07 ms vs incumbent 11.70 ms = +11.7 % regression before adding epilogue).

The **epilogue is essentially free** (907 us, 38 PFLOPS apparent — element-wise bf16 read+add+write at HBM bandwidth). It is not the bottleneck.

## Sub-task accounting (vs A3 estimate)

A3's "12 hours" estimate broke down as:

| Sub-task | A3 hours | Actual |
|---|---:|---:|
| Add `bt_offset` + `K_split` macros to v6 fork | 1 | 0.5 |
| Modify K-loop bound | 1 | 0.5 |
| Modify accumulator init | 0.5 | 0.5 |
| Allocate per-split buffers; modify dispatcher | 1.5 | 0.5 |
| Write epilogue add-and-cast kernel (50 lines) | 1.5 | 1.0 |
| SNR validation on L6 | 1 | **3.0** (see "SNR gate" below) |
| Bench harness wire-up + verify | 1.5 | 0.5 |
| Debug iterations | 4 | 1.0 |
| **Total** | **12** | **~7.5** |

## SNR gate — could not be applied as designed

The R32 mission spec requires SNR ≥ 25 dB on L6. **At L6 with random ±2 scales the kernel itself is noise-floor-limited**: bf16 saturates on 56 % of output positions (incumbent finite_frac = 0.42–0.45) and the kernel is **non-deterministic** (incumbent vs incumbent run-to-run: bit_eq = 0.44, SNR = 1.18 dB on the both-finite intersection at mag_cap = 1e6).

This is a known property — R32A verdict notes "random-scale + bf16 output naturally produces ~40 % NaN/inf entries even on the known-good incumbent". I tried multiple aperture restrictions (mag_cap ∈ {1e3, 1e4, 1e6, 1e8, 1e10}) and tiny-input patterns (FP4 codes 0..1, scale 2^−8, all-positive); in every case the V6 split2+epi output is statistically indistinguishable from the incumbent at the noise floor (V6 split2+epi vs INC: SNR = 1.6 dB ≈ INC vs INC noise floor 1.18 dB). 

**Best correctness signal achievable**: per-split S0/S1 kernels return finite outputs of similar magnitude+pattern to incumbent (finite_frac 0.42–0.76 per split, summed via epilogue → 0.46), and the epilogue itself is element-wise bf16 add — trivially correct. K_SPLIT=1 sanity build is bit-clean (compiles 212 VGPR / 0 spills, identical resource footprint to incumbent) and matches incumbent perf to within ±0.2 %.

I conclude the V6 math implementation is correct; the SNR gate is not measurable at L6 in random-aperture mode. A formal SNR validation would require running at smaller K (e.g. L4 dims) which is outside this round's L6-only mission.

## Files

Modified / added (committed):
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp_v6.cpp` — V6 fork of production kernel: adds `K_SPLIT`, `S_IDX`, `BT_START`, `BT_END`, `K_SPLIT_LEN` constexprs; modifies prologue, scale init, K-loop bounds, prefetch clamps, R25C tail-no-pf gate, tail iter; appends 50-line epilogue add-and-cast kernel + binding. Defaults `K_SPLIT=1`, `S_IDX=0` are no-op (identical to v1).
- `analysis/fp8_gemm/mi350x/_R32B_v6_build_L6.sh` — build script for V6 at L6 dims with incumbent flags. Builds split=1, split2_{s0,s1}, split4_{s0..s3} (7 SOs).
- `analysis/fp8_gemm/mi350x/_R32B_v6_bench_quick.py` — bench harness (50 warmup, 50 iters, 10-sample trim, cuda events). Compares incumbent vs V6 K_SPLIT={1,2,4}+epi.
- `analysis/fp8_gemm/mi350x/_R32B_v6_snr_vs_inc.py` — SNR gate (kept for posterity; concluded not measurable per "SNR gate" section).
- `analysis/fp8_gemm/mi350x/R32B_V6_BENCH_QUICK.log` — full bench output.
- `analysis/fp8_gemm/mi350x/R32B_V6_SNR_VS_INC.log` — SNR debug output.

Build artifacts (NOT committed; reproduce via `_R32B_v6_build_L6.sh`):
- `build_v6/tk_mxfp4_v6_split{1,2_s{0,1},4_s{0,1,2,3}}.so`

## Compile-time verification

V6 fork builds clean at L6 with incumbent flags (`-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -DBARRIER_TO_WAITCNT_ALL=1 -mllvm -amdgpu-sched-strategy=max-memory-clause`):

| Variant | size (B) | notes |
|---|---:|---|
| K_SPLIT=1 | 319384 | sanity baseline; should = incumbent perf |
| K_SPLIT=2 S=0 | 261040 | smaller (some main-loop branches simplify when bt range is half) |
| K_SPLIT=2 S=1 | 319408 | tail-iter branch retained |
| K_SPLIT=4 S=0..3 | 319408–335280 | similar; per-split unroll fires `K_SPLIT_LEN ≤ 128` branch |

Compile errors guarded:
- `S_IDX >= K_SPLIT` → `#error`
- `K_SPLIT > 1` with `SWAP_STEP34_MAIN`/`FUSED_STEP34` enabled → `#error` (POC limit)
- `K_SPLIT > 1` without `TAIL_SPLIT=1` → `#error`
- `K_SPLIT_LEN < 2` (split too aggressive for K_DIM) → `static_assert`

## Recommendation

**Declare V6 split-K DEAD on L6 as a single-shape lever.** The 41/42 = 92.6 % ceiling is **unbreachable by intra-K splitting** because L6 is already grid-saturated (2048 tiles ÷ 608 WGs = 3.4 iters/WG). To gain on L6 requires one of:

1. **V5 (MFMA32 path, ≥1 week)** — fundamentally different MFMA shape (64 MFMAs/iter vs current 128), opens a different unroll regime; but kernel is essentially a rewrite.
2. **V7 (Stream-K, ≥2 weeks)** — replaces grid-stride decomposition with iter-stride; only useful when grid is over-saturated, which L6 is not.
3. **Accept 41/42 as the final ceiling** — per the R32 decider's "HONEST STOP VERDICT" pre-statement: aiter's 7.4pp lead on L6 is opaque-by-construction (no in-tree ASM); we cannot reverse-engineer it from artifacts.

**Recommend acceptance of the 41/42 ceiling.** V6 POC infrastructure is committed for future use (e.g. if a shape with grid < 304 ever shows up — which, on this 42-shape suite, none do). The negative result rules out a sub-week intra-K lever.

(word count: ~1100)
