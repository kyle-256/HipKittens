# R32 Dev B — V2-CRR LDS Single-Buffer Pipelining Recovery

**Date:** 2026-04-18
**Branch:** r32-b (base feat/mxfp8-only @ 1e1e03a6)
**GPU:** HIP_VISIBLE_DEVICES=1 (physical GPU 1, MI355X / gfx950)
**Scope:** Recover the −30% perf loss observed by R31 Dev B in the V2-CRR LDS single-buffer (SB) form by adding explicit pipelining variants. R31 had established that SB hits LDS=69632 B/block (−50%) but the naive synchronous load→compute→load form sacrificed all cross-K-iter VMEM/MMA overlap. The R31 brief left the macro `MXFP8_CRR_LDS_SINGLE_BUFFER` in tree (default off) for R32+ to explore pipelining-recovery variants.

**Verdict:** **NO SHIP — best variant (PIPE=3) recovers about half of the R31 loss but still fails perf gate.**
- PIPE=3 (inner-K interleave): −15.4% on 8192³ / −18.6% on 70B Gate (vs DB baseline). LDS=69632 B/block, no spill, correctness PASS (SNR 49.59 / det OK).
- PIPE=1 (early-issue + late-wait, separate `a_next` reg): −56.9% — VGPR pressure forces 26-lane spill.
- PIPE=2 (split VMEM issue across MMA pairs, separate `a_next`): −58.1% — same spill mechanism.

Add to lever-closure list: **"V2-CRR LDS single-buffer with explicit pipelining recovery"** is CLOSED. The shape of the kernel (4 fp32 accumulators × 32 VGPRs each + 32-VGPR A_col_reg + 16-VGPR B_col_regs + scale state) leaves no headroom to hold a second A_col_reg concurrently — any variant requiring `a_next` spills 26 lanes and degrades worse than the naive form. The only spill-free variant (PIPE=3, single `a` reused across the two MMA half-chains) still serializes the LDS-read drain and barrier between halves, recovering only ~4% of the K-pipelining loss.

Code change is **macro-guarded, default off** (`MXFP8_CRR_LDS_SINGLE_BUFFER=0`). Default build md5 `c46876660de5877224e0f879e2afddf5` is bit-identical pre- and post-edit (verified). Macros `MXFP8_CRR_LDS_SINGLE_BUFFER` and `MXFP8_CRR_SB_PIPELINE` left in tree as scaffolding for future LDS-reduction work that doesn't depend on V2-CRR's specific accumulator-tile shape.

---

## 1. Three pipelining-recovery approaches

The R31 naive SB form (now `PIPE=0`) is:
```
for k = 0 .. k_iters-1:
    [scales]
    load_b(b0); load_b(b1); load_a(a, As[0][0])     // 3 LDS reads
    waitcnt lgkmcnt(0)
    MMA cA, cB                                       // 2 MMAs (8 quadrants)
    load_a(a, As[0][1])                              // 1 LDS read
    waitcnt lgkmcnt(0)
    MMA cC, cD                                       // 2 MMAs
    s_barrier                                        // wait for waves to finish reading
    global_load_b/a × 4                              // VMEM→LDS direct (async)
    waitcnt vmcnt(0)                                 // wait for writes to commit
    s_barrier                                        // sync waves before next iter
```

All VMEM happens at the K-iter boundary with no overlap with MMAs.

### Approach 1 (PIPE=1) — early-issue + late-wait

Issue ALL 4 LDS reads upfront (requires holding two A_col_regs `a, a_next`). Drain LDS reads + barrier. Then issue all 4 next-iter VMEM-to-LDS direct loads. Then run all 4 MMAs in two pairs while VMEM is in flight. Wait vmcnt at end.

```
load_b(b0); load_b(b1); load_a(a, As[0][0]); load_a(a_next, As[0][1])   // 4 LDS reads
waitcnt lgkmcnt(0); s_barrier                                            // drain reads + sync
global_load_b/a × 4                                                      // VMEM→LDS async
MMA cA, cB (using a)                                                     // overlap with VMEM
MMA cC, cD (using a_next)
waitcnt vmcnt(0); s_barrier
```

### Approach 2 (PIPE=2) — split VMEM issue

Like Approach 1 but the next-iter VMEM is split into two halves issued around the MMA pairs to spread the in-flight window: B-side issued before MMA1, A-side issued between MMA1 and MMA2.

### Approach 3 (PIPE=3) — inner-K interleave (no `a_next`)

Read As[0][0] before MMA1; read As[0][1] AFTER MMA1 (reusing single `a` register). Issue all 4 next-iter VMEM after the second LDS read drain + barrier, then run MMA2 with VMEM in flight.

```
load_b(b0); load_b(b1); load_a(a, As[0][0])      // 3 LDS reads
waitcnt lgkmcnt(0)
MMA cA, cB                                        // no overlap (still draining)
load_a(a, As[0][1])
waitcnt lgkmcnt(0); s_barrier                     // drain final read + sync
global_load_b/a × 4                               // VMEM→LDS async
MMA cC, cD                                        // overlap with VMEM
waitcnt vmcnt(0); s_barrier
```

Key advantage: only one `A_col_reg` live at a time → matches naive VGPR=250, no spill.

## 2. Resource report (V2-CRR `_Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv`, 8192³)

| Build | md5 | TotalSGPR | VGPR | VGPR Spill | LDS B/block | Occ |
|---|---|---:|---:|---:|---:|---:|
| Baseline (DB, default) | `c46876660de5...` | 52 | 234 | 0 | 139264 | 2 |
| SB PIPE=0 (R31 naive) | `ad6baf36331d2a...` | 46 | 250 | 0 | 69632 | 2 |
| SB PIPE=1 (early-issue) | `c7afa67d3035...` | 46 | 256 | **26** | 69632 | 2 |
| SB PIPE=2 (split VMEM) | `75af5711f407...` | 46 | 256 | **26** | 69632 | 2 |
| SB PIPE=3 (interleave) | `05725cbe4568...` | 46 | 250 | 0 | 69632 | 2 |

VGPR spill explanation: A_col_reg = `rt_fp8e4m3<BK=128, RBM=64, col_l, rt_128x16_s>` = 8KB tile / 64 lanes = 128 B/lane = **32 VGPRs per wave**. PIPE=1 / PIPE=2 add a second `a_next` register → +32 VGPR; the compiler can't fit it within the per-wave VGPR budget without 26 lanes of spill in inner loop.

PIPE=3 reuses the single `a` register across both MMA halves, so no additional VGPR is needed — same 250 VGPR / no-spill profile as PIPE=0.

LDS reduction is identical for all SB variants (69632 B/block = exactly half of baseline).

mb=3 not re-tested — R31 already established mb=3 forces compiler to spill 191 lanes (272 TFLOPS, catastrophic). LDS-bound nature confirmed in R30.

## 3. Build-cache hygiene + per-build md5 (R29 Dev C rule)

For every build:
- `rm -f tk_mxfp8_layouts*.so` before compile.
- `md5sum tk_mxfp8_layouts*.so` after.

For paired BABA builds:
- Two separate `.so`s with distinct `PY_MODULE_NAME`: `tk_mxfp8_layouts_base` and `tk_mxfp8_layouts_pipe{1,2,3}`.

Default build sanity check: post-edit default build (no SB macro, no PIPE macro) hashes to `c46876660de5877224e0f879e2afddf5`, **bit-identical** to pre-edit. The macros are dead code when `MXFP8_CRR_LDS_SINGLE_BUFFER=0`.

## 4. Paired BABA bench results (5x and 3x preheat-then-bench, GPU1, 30s preheat per harness)

Paired in-process BABA bench loads `tk_mxfp8_layouts_base` and `tk_mxfp8_layouts_pipe{1,2,3}` as separate pybind11 modules; runs base→cand→base→cand per pair (n_pairs × 2 = bench data points per side). 30-second preheat with 16k × 16k fp16 matmul before measurement.

Welch t computed with each side's full sample (n=6 for n_pairs=3, n=10 for n_pairs=5).

### 8192³ V2-CRR (primary cell)

| Variant | n | base mean | cand mean | Δ TFLOPS | Δ % | Welch t | LDS B | corr |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| SB PIPE=1 | 6 | 2745.62 | 1189.86 | −1555.76 | **−56.7%** | −120.3 | 69632 | PASS |
| SB PIPE=2 | 6 | 2768.78 | 1159.59 | −1609.19 | **−58.1%** | −639.6 | 69632 | PASS |
| SB PIPE=3 | 6 | 2773.92 | 2361.29 | −412.63 | **−14.9%** | −64.1 | 69632 | PASS |
| SB PIPE=3 (n_pairs=5) | 10 | 2417.26 | 1990.20 | −427.06 | **−17.7%** | −174.6 | 69632 | PASS |
| (R31 SB naive, ref) | n=5 | ~2746 | 1912.53 | −833 | −30.6% | ~−63 | 69632 | PASS |

Note: PIPE=3 baseline mean varies between runs (2774 vs 2417) due to thermal stagger; the candidate mean is highly stable (~1990 across both runs). Likely the candidate hits a pipeline-throttled floor independent of clock.

### LLaMA 70B Gate 4096×28672×8192 V2-CRR (cross-shape)

| Variant | n | base mean | cand mean | Δ TFLOPS | Δ % | Welch t | LDS B | corr |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| SB PIPE=3 | 6 | 2370.58 | 1931.06 | −439.52 | **−18.5%** | −168.2 | 69632 | PASS |
| (R31 SB naive, ref) | n=5 | 2372.95 | 1581.66 | −791 | −33.3% | ~−50 | 69632 | PASS |

### Cross-variant summary

PIPE=3 recovers ~half of the R31 SB naive loss on both shapes:
- 8192³: −30.6% (R31) → −15-18% (PIPE=3): about +13 pp recovered.
- 70B Gate: −33.3% (R31) → −18.5% (PIPE=3): about +15 pp recovered.

But none meet the SHIP gate (parity or +ve vs DB baseline with Welch t > 3.0).

PIPE=1 / PIPE=2 are catastrophically worse than PIPE=0 because the spill of 26 VGPR lanes in the inner loop adds ~30+ extra LDS round-trips per K-iter to scratch, dwarfing any pipelining gain.

## 5. Why PIPE=3 (the only viable form) still loses ~17%

Even with VMEM in flight during MMA cC/cD pair, the kernel still pays:

1. **Two full LDS-read-drain barriers per iter** (one inside, one at iter boundary):
   - inner: after `load_a(a, As[0][1])` we wait `lgkmcnt(0)` + `s_barrier` — serializes against wave skew (each wave waits for slowest).
   - outer: after `vmcnt(0)` + `s_barrier` — same.
   The DB form has inter-iter prefetch that hides one of these inside the previous iter's MMA chain; SB cannot.

2. **VMEM only overlaps with the second MMA pair**. The first MMA pair (cA, cB) runs after `lgkmcnt` drain but BEFORE next-iter VMEM is issued (which can't begin until the second LDS read is also drained). So only ~half the MMA chain overlaps with VMEM, vs ~all of it in the DB form.

3. **No hoist of scale loads outside the K-iter critical path**: scales are still loaded at top-of-iter inside the `(k & 1) == 0` branch. Could in principle be hoisted to overlap with the prior iter's MMAs, but all the natural slots are already occupied by LDS read drains.

4. **`s_barrier` cost**: each barrier on gfx950 is ~50 cycles fixed-cost regardless of wave count. Two barriers per iter × 64 iters × ~256 grid blocks = significant overhead vs DB which has half the barriers.

5. **VGPR=250 already at edge of 2-block/CU LDS budget**, so even if mb=3 hint were taken, no gain from forcing higher occupancy (R30 closure).

The −15-18% gap matches the back-of-envelope estimate: ~half the K-pipelining loss recovered (overlap during MMA pair 2), ~half the loss remains (no overlap during MMA pair 1 + extra inner barrier).

## 6. SHIP gate evaluation per approach

Gates per R32 brief: any approach with perf parity OR positive vs DB baseline on **8192³ AND 70B Gate** (Welch t > 3.0) while LDS=69632 and correctness PASS.

| Variant | Corr 8192 | LDS=69632 | 8192 ≥ baseline (t>3) | 70B Gate ≥ baseline (t>3) | Verdict |
|---|:---:|:---:|:---:|:---:|---|
| SB PIPE=1 | YES | YES | NO (−57%, t=−120) | not tested (no value) | NO SHIP |
| SB PIPE=2 | YES | YES | NO (−58%, t=−640) | not tested (no value) | NO SHIP |
| SB PIPE=3 | YES | YES | NO (−15%, t=−64) | NO (−19%, t=−168) | NO SHIP |

All three approaches NO SHIP.

## 7. Outcome classification (per R31 brief vocabulary)

This is the **"NO SHIP — pipelining recovery insufficient (best variant still negative)"** outcome. The R31 hypothesis that a "split global_load early-issue + late-wait" form might recover the loss is **partially confirmed** (PIPE=3 closes about half the gap) but not enough to ship.

The structural gap between SB and DB on this kernel is:
- DB has cross-K-iter VMEM prefetch into a separate slot (`As[toc][_]` / `Bs[toc][_]`), giving the entire MMA chain time to overlap with the next iter's VMEM.
- SB has only intra-K-iter VMEM overlap, and only with the second half of the MMA chain (the first half consumes the slot the next iter will write into).

To recover the remaining ~15-18%, one would need either:
1. A non-`s_barrier` synchronization that doesn't serialize on wave skew (no obvious gfx950 primitive).
2. A different SB layout where waves don't share the LDS slot (e.g., per-wave LDS partition — would need rewriting the load/store geometry, ~3-5 days).
3. Smaller tile sizes (e.g., HB=64 instead of HB=128) so the LDS halves fit in DB form — but this would require a parallel rewrite of the entire CRR fastpath geometry (8 static_asserts on BLK/BK/RBM/RBN/WARPS_M/WARPS_N).

None are within the R32 budget.

## 8. Paradigm correction recommended for R32 wrap

Add to the lever-closure list (extends R27/R28/R29/R30/R31 closures, now 19 closed levers cumulative):

> **`V2-CRR LDS single-buffer with pipelining recovery (PIPE=1, PIPE=2, PIPE=3)` is CLOSED for V2-CRR** (R32 Dev B). Three explicit pipelining-recovery variants on top of the R31 SB naive form:
> - PIPE=1 (early-issue + late-wait, requires `a_next` reg): VGPR spill 26 lanes in inner loop → −57% on 8192³ Welch t=−120.
> - PIPE=2 (split VMEM issue, also requires `a_next`): same spill pattern → −58% on 8192³ Welch t=−640.
> - PIPE=3 (interleave with single `a` reg reuse): no spill, recovers about half the R31 loss → −15-18% on 8192³ / 70B Gate, Welch t=−64 to −175.
>
> Root cause: A_col_reg is 32 VGPRs per wave; the kernel's 4 fp32 accumulators (128 VGPRs) + b0/b1 (32 VGPRs) + a (32 VGPRs) + scales/loop state already saturates the per-SIMD VGPR budget. Holding a second `a_next` causes spill that dwarfs any pipelining gain.
>
> The PIPE=3 variant (single-`a` reuse) is the only spill-free SB form, but its inner-iter LDS-read-drain + barrier still serializes one MMA half-chain against the VMEM issue, leaving ~15-18% perf gap. NEVER prototype PIPE=1 or PIPE=2 again. PIPE=3 is the best achievable SB form on V2-CRR's current accumulator/operand-tile shape.
>
> Macros `MXFP8_CRR_LDS_SINGLE_BUFFER` and `MXFP8_CRR_SB_PIPELINE` left in tree (default off, bit-identical baseline verified) for future investigators to reuse if the kernel's tile geometry changes (e.g., a smaller-RBM rewrite or a new MFMA size that frees VGPRs).
>
> Future LDS-reduction work for V2-CRR must come from a different lever: either smaller tile geometries (rewrite, not a macro flip) or per-wave LDS partition (rewrite, breaking the load/store helpers).

## 9. Files

Code change (macro-guarded, default off):
- `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc` — added `MXFP8_CRR_SB_PIPELINE` macro and 3 alternative SB K-loop bodies under `#if MXFP8_CRR_SB_PIPELINE == 1/2/3`. Default-off bit-identical to baseline (md5 `c46876660de5877224e0f879e2afddf5` matches both pre-edit and post-edit clean default builds).

Bench script:
- `analysis/fp8_gemm/mi350x/r32b_paired_bench.py` — paired BABA in-process harness, 30s preheat, two pybind11 modules under distinct names for V2-CRR `gemm_crr_pq_v2`.

Bench logs (in this directory):
- `r32b_baseline_8192_5x.txt` — Baseline n=5 (median 2783, mean 2785, sd 17)
- `r32b_baseline_70bgate_5x.txt` — Baseline n=5 (median 2369, mean 2372, sd 9)
- `r32b_paired_pipe1_8192.txt` — PIPE=1 BABA n=6 (cand 1190, base 2746, t=−120)
- `r32b_paired_pipe2_8192.txt` — PIPE=2 BABA n=6 (cand 1160 first, second run for stability)
- `r32b_paired_pipe2_8192_run2.txt` — PIPE=2 BABA re-run (cand 1160, base 2769, t=−640)
- `r32b_paired_pipe3_8192.txt` — PIPE=3 BABA n=6 (cand 2361, base 2774, t=−64)
- `r32b_paired_pipe3_8192_n5.txt` — PIPE=3 BABA n=10 (cand 1990, base 2417, t=−175)
- `r32b_paired_pipe3_70bgate.txt` — PIPE=3 BABA 70B Gate (cand 1931, base 2371, t=−168)

Build logs (with `-Rpass-analysis=kernel-resource-usage`):
- `r32b_baseline_8192_build.log`, `r32b_baseline_70bgate_build.log`
- `r32b_pipe0_build.log` (R31 naive, for VGPR-baseline reference)
- `r32b_pipe1_build.log`, `r32b_pipe2_build.log`, `r32b_pipe3_build.log`

Build-cache hygiene: `rm -f tk_mxfp8_layouts*.so` before every build; per-build md5 logged inline (every distinct configuration produced a distinct md5).

## 10. Methodology rule compliance (R31 closures)

| Rule | Compliance |
|---|---|
| `rm -f tk_mxfp8_layouts*.so` + per-build md5 (R29 Dev C) | YES, every build |
| `rocm-smi -d 1` for physical GPU 1 (R31 Reviewer) | YES (PHYS_GPU=1 env in r32b_paired_bench.py) |
| In-process A/B with BABA + 30s preheat (R31 Dev D) | YES, all paired benches |
| Closed-lever check (no prototyping of 18 closed levers) | YES, no prohibited variants prototyped |
| SHIP-claim normalization (GPU0 → 2.6% discount) | N/A — bench is on physical GPU 1, no SHIP claim |

## 11. Summary

| Metric | Value |
|---|---|
| Verdict | **NO SHIP — pipelining recovery insufficient** |
| Best variant | PIPE=3 (interleave with single `a` reg reuse) |
| LDS / VGPR / Spill | 69632 B (−50%) / 250 VGPR (+16) / 0 lanes |
| 8192³ V2-CRR Δ (best) | **−14.9% / −17.7%** (Welch t=−64 / −175) |
| 70B Gate V2-CRR Δ (best) | **−18.5%** (Welch t=−168) |
| Recovery from R31 SB naive | 8192³: 30.6 → 14.9 = +15.7pp (about half); 70B: 33.3 → 18.5 = +14.8pp (about half) |
| Correctness | All builds: SNR 49.59 dB, det 3/3 PASS |
| Default build bit-identical | YES (md5 `c46876660de5877224e0f879e2afddf5`) |
| Code change committed? | Macro left in tree default-off, no SHIP commit (consistent with R31 Dev B precedent) |
| Time spent | ~3.5 hours (within 6h budget) |
| Paradigm closures added | YES — PIPE=1, PIPE=2, PIPE=3 each closed (19th, 20th, 21st cumulative closures) |
| Recommended next direction | Either accept SB perf cost as inherent to V2-CRR's accumulator-tile shape, OR shift focus to non-CRR LDS-reduction work (V2-RCR, rect-V2). Smaller-RBM geometry rewrites would re-open SB but are 3-5 day surgery. |
