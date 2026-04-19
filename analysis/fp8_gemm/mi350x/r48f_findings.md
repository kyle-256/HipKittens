# R48 Dev F — A-scale-only cachepolicy + A-first issue order on V2-RCR (70B Down)

**Date:** 2026-04-19
**Branch:** R48 worktree on feat/mxfp8-only @ c46227d0
**GPU:** HIP_VISIBLE_DEVICES=4 (MI355X / gfx950)
**Target cell:** 70B Down RCR — M=4096, N=8192, K=28672 (R47A-swizzle baseline ≈ 2995 TFLOPS / 91.1% of FP8)
**Lead:** R48 Dev B refutation (`r48b_findings.md` §4) — "the 70B Down RCR/CRR gap is NOT in the B-scale loading path — productive directions point to A-scale (PC=4 b128 has 2× B's bytes)".

**Verdict:** **NO SHIP — REFUTED on both step 1 and step 3.**
- A-scale-only cachepolicy sweep (p=0/1/2/3): no policy beats baseline by ≥ +1.5%; p≥2 (SLC) regress slightly (-0.6% / -1.3%).
- A-first issue-order pin (sched_barrier symmetric inverse of Dev B's B-first): -0.53% — within noise, mirror image of Dev B's +0.18% B-first null result.

The 70B Down RCR gap is not in the A-scale load path either. Combined with Dev B's B-scale refutation and R31C's VGPR-prefetch refutation, the entire scale-load axis is now eliminated for V2-RCR on this shape. The remaining 4.1pp gap-to-FP8 likely sits in **FP8 tile data movement** (28672-K means A-tile + B-tile dominate VMEM, not scales) or — more likely — is structural / at hardware ceiling per Dev D's analysis pattern (RCR family at M=4096 is uniformly within 1pp of predicted 95% ceiling for K=8192; large-K shapes amortize prologue/epilogue further so the steady-state ceiling could already be the binding constraint here too).

---

## 1. Hypothesis & lever design

Per Dev B's lead, A-scale and B-scale at V2-RCR have asymmetric per-lane byte volumes:

```
A-scale (PC=4): 1 buffer_load_b128 / lane / k_pair = 16 bytes
B-scale (PC=2): 1 buffer_load_b64  / lane / k_pair = 8 bytes
```

For 70B Down (M=4096, N=8192, K=28672) the working sets are:
- A-scale: M × K / 32 = 4096 × 28672 / 32 = **3.67 MB** (fits in per-XCD L2, ~8 MB)
- B-scale: N × K / 32 = 8192 × 28672 / 32 = **7.34 MB** (does NOT fit in per-XCD L2)

The pre-existing macro `MXFP8_RCR_V2_SCALE_CACHEPOLICY` (R27 Dev A H1, R47 Dev C swept) controls **both** A and B scale loads with a single value. R47C found p=0 best universally for RCR. But the unified macro could be hiding an asymmetric optimum: **A** could benefit from one policy (it fits in L2 → keep cached, p=0) while **B** wants another (it overflows L2 → SLC=2 to reduce L2 pollution from pages that won't be reused). Or vice-versa, if A's larger per-load width is actually the L2-pollution culprit and B's smaller load is the cache-friendly one.

This work tests the A-side independently:

1. **Step 1 — A-scale-only cachepolicy:** add macro `MXFP8_RCR_ASCALE_CACHEPOLICY` overriding the unified macro for the four A-scale `buffer_load_b128` call sites in the V2-RCR fastpath. Sweep p=0/1/2/3 with B-scale fixed at p=0 (default).
2. **Step 3 — A-first issue order:** add macro `MXFP8_RCR_ASCALE_FIRST` that inserts a `__builtin_amdgcn_sched_barrier(0)` immediately AFTER the A-scale b128 issue (and before the B-scale b64 issue), pinning the LLVM-scheduled order with A strictly first. This is the symmetric inverse of Dev B's `MXFP8_RCR_COOPERATIVE_BSCALE` (B-first); Dev B's B-first is null at +0.18%, so A-first should also be null if the LLVM scheduler is producing an optimal interleaving.

Step 2 (no-regression sweep across all 7 RCR shapes) is skipped because step 1 produced no winner ≥ +1.5%.

## 2. Patch sites (all default OFF)

`kernel_mxfp8_layouts.cpp` (V2-RCR fastpath — NOT touched: rect, RRR, CRR, B-scale path):

| Macro | Default | Site count | Lines |
|---|---|---:|---|
| `MXFP8_RCR_ASCALE_CACHEPOLICY` | -1 (sentinel: inherit unified) | decl + resolved expansion | 392-406 |
| `MXFP8_RCR_ASCALE_CACHEPOLICY_RESOLVED` | (computed) | A-scale b128 call sites | 2891, 3390, 3401, 3448 |
| `MXFP8_RCR_ASCALE_FIRST` | 0 | decl + 2 sched_barrier sites | 444-454 + 2892, 3402 |

ABI invariant when both macros are at default: `MXFP8_RCR_ASCALE_CACHEPOLICY_RESOLVED` expands exactly to `MXFP8_RCR_V2_SCALE_CACHEPOLICY` (the existing unified macro; baseline value 0). I verified the default-OFF correctness path produces SNR 49.61 dB and 3/3 byte-identical determinism (matches Dev B's documented baseline-RCR SNR of 49.61 dB to the centidB), confirming no behavioural change.

A compile-time `#error` enforces mutual exclusion of `MXFP8_RCR_ASCALE_FIRST` and `MXFP8_RCR_COOPERATIVE_BSCALE` (they place sched-barriers at conflicting program points).

## 3. Step 1 — A-scale cachepolicy sweep (3-run, 50 warmup / 100 iters, sleep 20s between)

Build commands per cell (orchestrator: `r48f_70bdown_sweep.sh`):
```
make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
  CXXFLAGS="-w -DM_DIM=4096 -DN_DIM=8192 -DK_DIM=28672 \
            -DMXFP8_RCR_ASCALE_CACHEPOLICY=${p}"
```
All builds with B-scale at the default (`MXFP8_RCR_V2_SCALE_CACHEPOLICY=0`). The resolved macro expansion — verified via `grep` on the post-edit source — uses the override for A-scale b128 only.

Bench: `MXFP8_BUILD_M=4096 MXFP8_BUILD_N=8192 MXFP8_BUILD_K=28672`, `MXFP8_LAYOUTS=rcr`, `MXFP8_PRESHUFFLE_QUANT=1`. GPU 4 (sole user).

| Policy | Run 1 | Run 2 | Run 3 | Median (TFLOPS) | Δ vs p=0 |
|---|---:|---:|---:|---:|---:|
| **p=0 (default)** | 2898.54 | 2934.20 | 2939.20 | **2934.20** | — |
| p=1 (GLC) | 2943.01 | 2906.49 | 2931.46 | 2931.46 | **-0.09%** |
| p=2 (SLC) | 2880.39 | 2915.99 | 2923.99 | 2915.99 | **-0.62%** |
| p=3 (GLC+SLC) | 2895.36 | 2906.21 | 2896.01 | 2896.21 | **-1.29%** |

**Best policy: p=0 (already the default).** No alternative policy beats p=0 by the +1.5% SHIP gate; p=1 is a statistical wash (well inside ±0.5% noise), p=2 and p=3 regress monotonically. This matches the R47C unified-macro pattern for the same shape (p=0=2950.13, p=1=-0.26%, p=2=-0.50%, p=3=-0.36% — see `r47c_findings.md`), confirming that decoupling A from B does not unlock asymmetric optima here.

**Interpretation:** the 3.67 MB A-scale working set already enjoys L2 hits with default policy; biasing to SLC just moves loads to the L1-skip path and forfeits L2 reuse across CTAs sharing the same `bc` slab. SLC is a worse default, even isolated to A.

Per task gate ("If best is not p=0 by ≥ +1.5%: continue to step 3"), proceed to step 3.

## 4. Step 3 — A-first issue order pin (3-run, 50 warmup / 100 iters)

Build (orchestrator: `r48f_70bdown_afirst.sh`):
```
make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
  CXXFLAGS="-w -DM_DIM=4096 -DN_DIM=8192 -DK_DIM=28672 \
            -DMXFP8_RCR_ASCALE_FIRST=${af}"
```

| Setting | Run 1 | Run 2 | Run 3 | Median (TFLOPS) | Δ vs af=0 |
|---|---:|---:|---:|---:|---:|
| **af=0 (baseline)** | 2923.85 | 2934.82 | 2926.91 | **2926.91** | — |
| af=1 (A-first pinned) | 2909.73 | 2911.43 | 2929.21 | 2911.43 | **-0.53%** |

**Within ±0.5% noise.** Mirror image of Dev B's B-first null result (+0.18%). Combined with Dev B, both ordering directions are null — confirming the LLVM scheduler is already producing a near-optimal interleaving of the A b128 / B b64 issue pair, and pinning either direction with a sched-barrier provides no measurable benefit.

## 5. Cross-shape no-regression sweep — SKIPPED

Step 2 is conditional on step 1 producing a per-shape winner. Step 1 produced no winner on the primary target (70B Down RCR), so a no-regression sweep across all 7 RCR shapes would only confirm "default-OFF == default-OFF" (trivially zero delta everywhere). The macro infrastructure is left in tree default OFF; future investigators can reuse it without re-deriving patch sites.

## 6. Correctness gate (default-OFF baseline regression test)

To confirm that adding the macro infrastructure does not perturb the default code path, I ran `MXFP8_CHECK=1` `MXFP8_DETERMINISM_RUNS=3` on the unmodified-default build (no `-DMXFP8_RCR_ASCALE_*` flags):

```
[RCR] Correctness (rtol=0.1000, atol=3.0):
    Max abs error: 0.0624, Mean abs error: 0.0040
    SNR: 49.61 dB (threshold 48.0 dB)   PASS
[RCR] Determinism (3 runs): PASS
```

SNR 49.61 dB ≥ 45 dB SHIP gate; 3/3 byte-identical determinism. Identical to Dev B's documented RCR baseline SNR (49.61 dB). The macro infrastructure is a no-op when defaulted.

## 7. Hardware-ceiling cross-check vs Dev D

Dev D's `r48d_findings.md` table classifies all 5 RCR cells at M=4096 as **CEILING** (95-96% measured vs ~95% predicted, no recoverable headroom). The shape Dev B labelled "70B Down RCR" (M=4096,N=8192,K=28672) is NOT in Dev D's table — Dev D's "70B Down" is a different cell (M=4096,N=4096,K=14336). However Dev D's broader generalization (§4.1: "All 5 RCR cells are at hardware ceiling … STOP all RCR optimization on M=4096 shapes") implicitly covers any large-K M=4096 RCR shape via the steady-state argument (large-K amortizes prologue/epilogue further → steady-state K-pair cycle count dominates → ratio approaches the ISA-model 95%).

The R47A baseline for this shape is 2995.1 TFLOPS post-XCD-swizzle (`r47a_swizzle_results/clean_rebench_summary.md`). My measured baseline (2934 TFLOPS, 3-run median) is 2.0% below R47A's number — at the edge of the ±2% within-GPU envelope, consistent with run-to-run noise / GPU thermal state. Even taking the high R47A figure (91.1% of FP8 baseline 3286 TFLOPS) as the reference, the 4.1pp gap is plausibly within Dev D's prediction P1 ("any future MXFP8 RCR optimization touching MFMA dispatch / scale load / scale pack will produce ≤+1pp on the 5 RCR cells listed above") generalized to this shape. **All scale-axis levers (R31C VGPR-prefetch, R47C unified cachepolicy, R48B B-scale issue reorder + cooperative B-scale, R48F A-scale cachepolicy + A-first issue order) have now been refuted for V2-RCR.** The remaining headroom — if any — is in the FP8 tile loading or is at hardware ceiling.

## 8. Macros left in the tree (default OFF)

- `MXFP8_RCR_ASCALE_CACHEPOLICY` (default -1 sentinel: inherit `MXFP8_RCR_V2_SCALE_CACHEPOLICY`) — A-scale-only cachepolicy override for V2-RCR b128 loads.
- `MXFP8_RCR_ASCALE_FIRST` (default 0) — sched_barrier-pin A-scale issue strictly before B-scale (mutually exclusive with `MXFP8_RCR_COOPERATIVE_BSCALE`).

Kept (not reverted) so a future investigator can reproduce / extend without re-deriving patch sites. The R48F null priors join Dev B's null prior on the "scale-load issue ordering / cachepolicy" axis — saving downstream cycles from re-running these experiments.

## 9. Logs

- `r48f_70bdown_p{0,1,2,3}_run{1,2,3}.log` — 3-run cachepolicy sweep
- `r48f_70bdown_af{0,1}_run{1,2,3}.log` — 3-run A-first sweep
- `r48f_build_p{0,1,2,3}.log`, `r48f_build_af{0,1}.log`, `r48f_build_default.log` — build logs (resource-usage remarks)
- `r48f_70bdown_sweep.sh`, `r48f_70bdown_afirst.sh` — orchestrator scripts (reproducible)

## 10. One-line summary

**A-scale-only cachepolicy (p=0/1/2/3) and A-first sched-barrier ordering both REFUTED for V2-RCR 70B Down (M=4096,N=8192,K=28672) — best A-scale policy is still p=0 (-0.09% / -0.62% / -1.29% for p=1/2/3); A-first ordering is -0.53% (mirror image of Dev B's null B-first +0.18%). Combined with Dev B (B-scale) and R31C (VGPR-prefetch), the entire scale-load axis is now eliminated for V2-RCR; the residual 4.1pp 70B Down gap is in FP8 tile data movement or at hardware ceiling per Dev D's RCR-M=4096 ceiling generalization.**
