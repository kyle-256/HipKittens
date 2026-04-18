# R28 Plan — Residual LOSE shapes after v2 bench

Date: 2026-04-18  Branch: `mxfp4`  Author: R28-Plan (Opus 4.7)
Status: pure analysis / planning, no kernel changes

---

## 0. v2 progress snapshot (as of 30/42 logged)

| v1 LOSE | M×N×K | v1 ratio | v2 ratio | v2 best tag | Verdict |
|---|---|---:|---:|---|---|
| L1 | 4096×14336×16384 | 97.2% | **115.3%** | `_pfoff56_kx16384_btw_all` | FLIPPED |
| L2 | 4096×28672×32768 | 96.5% | **115.6%** | `_pfoff120_kx32768_btw_all` (tv0) | FLIPPED |
| L3 | 4096×32768×6144 | 99.7% | **116.2%** | `_pfoff19_kx6144_btw_all` | FLIPPED |
| L4 | 4096×32768×14336 | 98.9% | **98.5%** | `_ts_v12_tv0_memc_btw_all` (no K_EXACT) | NOISE-BAND LOSE |
| L5 | 4096×32768×28672 | 94.7% | **116.6%** | `_pfoff104_kx28672_btw_all` (lgk2) | FLIPPED |
| L7 | 14336×4096×32768 | 94.7% | **115.9%** | `_pfoff120_kx32768_btw_all` (tv0) | FLIPPED |
| L6 | 4096×32768×128256 | 92.8% | pending | — | residual (structural) |
| L8 | 16384×4096×14336 | 97.9% | pending | — | residual (likely noise) |
| L9 | 16384×4096×28672 | 95.0% | pending | — | residual (lgk2 .so launch in v1 was broken) |

**Projection**: 5 firm flips, 1 unchanged (L4 noise), 3 unknown (L6/L8/L9 still in flight). Best case 38/42 WIN; worst case 36/42.

### L4 root-cause analysis (shape 21 regression)

L4's v2 winner `ts_v12_tv0_memc_btw_all` ran 5217 TFLOPS vs v1's `ts_lgk2_memc_btw_all` 5240 TFLOPS — Δ = 23 TFLOPS (0.44%).
- The `_kx14336` K_EXACT .so for n=32768 IS built and on disk.
- v1 picked `lgk2`, v2 picked `tv0` — both BTW-all variants of equivalent class.
- Both are within parent-noise (~0.5% trim-mean variability at warmup=200/iters=500).
- **Not a real regression**, not a kernel issue — pure auto-tune `max()` instability when the top 2 variants are within stdev.
- Mechanism: in `bench_all_42.py:598-605`, ties are broken by iteration order in `variants[]`. Run-to-run TFLOPS for two near-tied variants will alternate which wins.

**Implication**: L4 does not need an R28 mechanism. Either (a) re-run with 5-rep median selection, or (b) accept ~98.5-99% as the achievable plateau for that shape.

---

## 1. Residual LOSE shapes after v2 (projected)

| Rank | Shape (M×N×K) | Proj ratio | v1 best variant | Class | Why still LOSE |
|---|---|---:|---|---|---|
| 1 | **4096×32768×128256** (L6, DLA1) | 92.8% | `ts_gm8_v12_btw_step3` | mega-K + 大N | R25C cannot be used (kernel:85-91); R27-C bypass DEAD; only structural axis is V5 32×32 |
| 2 | **16384×4096×28672** (L9) | 95.0% (or +20pp if launch fixes) | `v20_memc_btw_step3` | 大K + 大M | v1 had broken `_kx28672` lgk2 launch (263 TFLOPS) — v2 will retest cleanly; if launch OK, could hit ~115% |
| 3 | **16384×4096×14336** (L8) | 97.9% | `ts_u16` | 大K + 大M, near-noise | u16 family has no K_EXACT entry; gap is small |
| 4 | **4096×32768×14336** (L4) | 98.5-99% | tv0 / lgk2 (tied within noise) | mega-N + mid-K | noise plateau; no structural axis |

L3 will WIN in v2 (already 116.2% logged), so dropped from residual.

---

## 2. R28 candidate mechanisms

### R28-A — L9 launch-clean re-bench (shape 16384×4096×28672)
**Mechanism**: rebuild `tk_mxfp4_gluon_cpp_n4096_k28672_ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all` (already on disk per `R26-D`) and 5-rep verify on a clean GPU. v1 reported 263 TFLOPS — driver-state poisoning, not a real perf result.
- **Compile flags**: already built. No new compile.
- **Verify**: `bench_variant(GPU=N, m=16384, n=4096, k=28672, suffix='_ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all')` × 5 reps with warmup=200 iters=500 trim=10%, on idle GPU.
- **Success criterion**: median ≥ 5800 TFLOPS (vs comp 5525, would be ~105% WIN).
- **Cost**: 0.1 hr. **Risk**: zero. **EV**: P=0.7 × +20pp.

### R28-B — L9 tv0-parent K_EXACT@K=28672 (insurance for R28-A)
**Mechanism**: clone existing `_ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all` recipe but on tv0 parent stack (since L9's non-K_EXACT v1 winner was `v20_memc_btw_step3`, suggesting parent affinity may differ).
- **Compile flags**:
  ```
  -DTAIL_SPLIT=1 -DGROUP_SIZE_M=7 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0
  -DR25C_TAIL_PF_OFF_ITERS=104 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=28672
  -mllvm -amdgpu-sched-strategy=max-memory-clause
  -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule
  -DBARRIER_TO_WAITCNT_ALL=1
  ```
  Suffix: `_ts_v12_tv0_memc_dc_gm7_pfoff104_kx28672_btw_all`. Build for `n=4096, k=28672`.
- **Success criterion**: ≥ 5800 TFLOPS at M=16384.
- **Cost**: 0.5 hr (1 build + 5-rep). **Risk**: zero (K_EXACT-gated). **EV**: P=0.4 × +5pp.

### R28-C — L8 u16-family K_EXACT@K=14336 (closes the wrong-parent gap)
**Mechanism**: u16 (UNROLL_K=16) parent dominates 16384×4096×14336 (97.9%). No `_kx14336` exists with u16 parent. Build the smallest possible new K_EXACT entry on top of u16.
- **Compile flags**:
  ```
  -DTAIL_SPLIT=1 -DUNROLL_K=16 -DGROUP_SIZE_M=7 -DSTEP3_BARRIER_VMCNT=12
  -DR25C_TAIL_PF_OFF_ITERS=52 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=14336
  -mllvm -amdgpu-sched-strategy=max-memory-clause
  -DBARRIER_TO_WAITCNT_ALL=1
  ```
  Suffix: `_ts_u16_gm7_pfoff52_kx14336_btw_all`. K_iters=14336/256=56 → pfoff=52 (4 prefetches fire, matching R25-G recipe). Build for `n=4096, k=14336`.
- **Success criterion**: ≥ 5300 TFLOPS at M=16384 (vs comp 5142 → 103% WIN).
- **Cost**: 0.5 hr. **Risk**: zero. **EV**: P=0.4 × +3pp.

### R28-D — L6 DLA1 V5 MFMA_32X32X64 scout build (BACKBURNER fund decision)
**Mechanism**: only structural axis remaining. R25C is disqualified (K=128256 unroll-8 runtime branch causes 13× slowdown + memory faults, see `R27C_FIX_VERDICT.md`). Per `R27_V5_MFMA32_SCOUT.md` §6.3: the gating experiment is a single-shape DLA1 microbench of `MXFP4_USE_32X32` macro-gated `kpair_64mfma_step12`.
- **Phase 1 (≤2 days)**: build `mfma323264_scaled<opsel_a, opsel_b>` wrapper at `include/ops/warp/register/tile/mma.cuh` (mirror L128-148, ~25 LOC), add 32x32 dispatch arm at L218-225 (~20 LOC). Compile-only, no kernel change.
- **Phase 2 (≤3 days)**: build `MXFP4_USE_32X32` macro path of `kpair_64mfma_step12` at `kernel_mxfp4_gluon_cpp.cpp:1255-1391`. Use `acc_A0Bl[4]` (`fp4_floatx16_t`) instead of `acc_A0Bl[16]` (`fp4_floatx4_t`). 8 MFMAs/half instead of 32.
- **Compile flags**: `-DMXFP4_USE_32X32=1` plus existing R25-G stack for K=128256.
- **Success criterion (DECISION GATE)**: 5-rep median ≥ +1pp on DLA1 (≥5416 TFLOPS) with no SNR drop. If <+1pp or any SNR regression → **DEAD, abandon V5**.
- **Cost**: 5 days for gate. Full rewrite if gated through: ~1.5 weeks.
- **Risk**: HIGH (regression risk on 33+ already-WIN shapes during full re-tune).
- **EV**: P=0.3 × +6pp on DLA1 alone; gated by Phase 1+2 outcome.

---

## 3. Priority ordering

1. **R28-A** (L9 re-bench, 0.1 hr, zero risk, P=0.7 × +20pp) — DO FIRST.
2. **R28-B** (L9 tv0-parent K_EXACT, 0.5 hr, zero risk, insurance) — DO IF R28-A < +5pp.
3. **R28-C** (L8 u16-K_EXACT, 0.5 hr, zero risk, +3pp) — DO IN PARALLEL with R28-A.
4. **R28-D** (V5 32×32 scout) — FUND ONLY IF user explicitly authorizes 1-week sprint AND R28-A/B/C all complete without flipping ≥7 of 9 LOSEs. Currently we project 5 firm + 3 likely flips = 8/9, so V5 EV is marginal.

---

## 4. DO NOT TRY (explicit dead list)

1. **Any R25C variant with K≥65536 or K=128256** — kernel:85-91 documents pragma-unroll-8 runtime branch causes code-size doubling and catastrophic regression. R27-C bypass attempts confirmed DEAD with memory-aperture violations.
2. **R26-A pf495 wire on DLA1** — `verify_round26_pf495_clean.log` shows reps [3586, 1672, 1680, 5546, 1667] TFLOPS, stdev=1729 (32-39% catastrophic regressions). The single 6173-6503 stable run on GPU 0 was coincidental scheduling.
3. **R25C_TAIL_PF_OFF_ITERS = 497** for K=128256 — HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION on 5/5 reps.
4. **STEP3_PF_N / STEP4_PF_N tuning** on R25-G stacks — `R26B_PF_N_dead.md` (DLA2 monotone worse).
5. **TAIL_BARRIER_VMCNT** on R25-best stacks — `R26C_tail_vmcnt_dead.md` (flat across {0,4,8,12,16}).
6. **STEP12_BR_LGKMCNT 2nd round** — commit `5d0b5fd2`.
7. **gm5/gm9 sweep** — `R26E_gm_axis_dead.md` (gm7 already at local optimum).
8. **B-tile `__builtin_prefetch`** — `emit_one_pf` already emits `buffer_load_lds`.
9. **WAVES_PER_EU=3** — `__launch_bounds__(_NUM_THREADS, 1)` clamps to 1 wave/SIMD on K-bound shapes.
10. **Cache hints / NT stores / persistent-XCD / EARLY_SCALE_PF / EARLY_BL_PF / DIRECT_BL** — all DEAD pre-R26.
11. **outer-K pull-forward / extra L2 pf** — R24B/C: VMEM-issue-bound, NOT VMEM-latency-bound.
12. **iterative-ilp LLVM sched-strategy** — compiler bug (HSA aperture violation on ≥10 shape categories).
13. **per-shape K-loop unrolling UNROLL=8/16** — Round 6 B refuted ±0.16pp.
14. **static XCD-aware block_id remap** — Round 7 B: -0.37pp.
15. **ASM rewriter (s_nop removal)** — breaks correctness.

---

## 5. GPU dispatch plan

**Pre-condition**: wait for v2 to finish (currently 30/42 done, projected ~60-90 min more wall-clock based on 8-GPU parallel pace). v2 occupies GPUs 0,1,5,6,7. GPUs 2,3,4 are idle now and could be used for R28 prep work, but build steps don't need a GPU.

| Step | Agent | GPU | Action | Est duration |
|---|---|---|---|---|
| S0 | wait | — | v2 finishes; parse final JSON; confirm L6/L8/L9 statuses | 60-90 min |
| S1 | R28-Build | none (CPU) | If L9 / L8 still LOSE: build R28-B (`_ts_v12_tv0_memc_dc_gm7_pfoff104_kx28672_btw_all` for n=4096,k=28672) AND R28-C (`_ts_u16_gm7_pfoff52_kx14336_btw_all` for n=4096,k=14336) IN PARALLEL | 5 min (2 .so builds) |
| S2a | R28-Bench-A | GPU 0 | R28-A: re-verify `_ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all` at M=16384,N=4096,K=28672, 5 reps | 5 min |
| S2b | R28-Bench-B | GPU 1 | R28-B: bench new `_ts_v12_tv0...kx28672_btw_all` at M=16384,N=4096,K=28672, 5 reps | 5 min |
| S2c | R28-Bench-C | GPU 5 | R28-C: bench new `_ts_u16...kx14336_btw_all` at M=16384,N=4096,K=14336, 5 reps | 5 min |
| S2d | R28-Bench-L4 | GPU 6 | Optional: 5-rep median verify of L4's existing `_ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all` at M=4096,N=32768,K=14336 to confirm L4 is noise-band, not real regression | 5 min |
| S3 | R28-Wire | none | If any R28-B/C beats existing best by ≥1pp at warmup=200 iters=500 trim=10%, ADD entry to `bench_all_42.py:519-576` variants list | 10 min |
| S4 | R28-Verify | GPUs 0,1,5,6,7 | Re-run `bench_all42_parallel_R25_FINAL.py 0,1,5,6,7` with new entries (v3 bench) | 90-180 min |
| S5 (gated) | R28-V5-Scout | none + GPU 7 | ONLY if user authorizes. Per R28-D Phase 1+2 plan above. | 5 days |

**Total turnaround for non-V5 work**: ~3-4 hours after v2 finishes. Net WIN gain projection: 36/42 → 37-38/42.

---

## 6. References

- `bench_all42_results_R25_FINAL_v2.log` (in flight, line 30 at planning time)
- `R27C_FIX_VERDICT.md` — DLA1 R25C bypass DEAD
- `R27_V5_MFMA32_SCOUT.md` — V5 BACKBURNER reasoning, Phase 1+2 plan
- `R27_DIAGNOSE.md` — original 9-LOSE diagnosis matrix
- `verify_round26_pf495_clean.log` — pf495 instability proof (stdev=1729 TFLOPS)
- `kernel_mxfp4_gluon_cpp.cpp:80-110` — R25C K-gate documentation
- `bench_all_42.py:519-576` — current K_EXACT variant list
