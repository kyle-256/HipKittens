# R29 L6 Decider Verdict — 4096×32768×128256 (DLA1)

**Date**: 2026-04-18  Decider: claude-opus-4-7 (R29-Scout, no-build)
**Status**: L6 stuck at 92.6% of competitor (5353.9 / 5781.1 TFLOPS), gap = 7.4pp.
**Top v2 variants** all saturate within 0.5% (92.1-92.6%) across BARRIER_TO_WAITCNT
permutations — confirms BTW axis is fully saturated.

---

## 1. Three candidate mechanisms NOT yet tried for L6

### V6 — **Split-K with global atomic FP32 reduce** (single round, K-only split)
- **What**: Launch K=128256 as a **split-K** grid: spawn `S` extra grid blocks
  along the K axis (e.g. S=2 → each WG processes K/2=64128, two WGs per (m,n)
  tile reducing into a single FP32 output via `atomicAdd`). Optionally a
  `__threadfence`+second-pass bf16 cast at the end. Atomics live on a workspace
  FP32 `[M,N]` buffer (~1 GiB at 4096×32768) allocated host-side.
- **Why for K=128256**: shape is 4× longer than next deepest in-tree shape
  (32768). At K_iters=501, even a perfect K-loop spends ≥3 ms of MFMA-bound
  compute per WG; splitting halves wall-time *if* HBM/L2 bandwidth isn't the
  binder. It is **not** binder-bound — R24B/C found VMEM-issue saturation, not
  latency. Splitting reduces per-WG K-iters to a regime where R25C
  K_EXACT *can* fire (K/2 = 64128 is still > 32768 gate, but K/4 = 32064 fits;
  and a per-split `K_EXACT` recompile is straightforward).
- **Risk**: HIGH. Atomic FP32 contention across (M,N) overlap could nullify
  gains; output buffer doubles memory pressure; correctness bar is a full SNR
  regression on DLA1.
- **Expected gain**: 3-8pp on L6 (best case wins; worst case zero / DEAD).
- **Effort**: **multi-day** (≥3 days: workspace alloc, atomic store path,
  per-shape K_EXACT plumbing, SNR validate). NOT < 2 hr.

### V7 — **Stream-K dynamic K partitioning** (Davis et al. 2023)
- **What**: Replace fixed (m,n,k)→WG mapping with a Stream-K scheduler: each
  CU claims a contiguous range of K-iters across the global `M*N*K_iters`
  output, accumulating a per-CU partial that gets reduced via fixup kernel.
  Uses persistent-XCD machinery (already in tree at L2199) plus a fixup pass.
- **Why for K=128256**: Stream-K's main benefit is exactly L6's pathology —
  long-K + low-occupancy "tail effect" where the last few WGs run alone
  while CUs idle. With M=4096,N=32768 → 32×128 = 4096 tiles on 304 CUs of
  MI355X (8 XCDs × 38 CU each), the steady-state tail is significant.
- **Risk**: HIGH. Requires fixup kernel + atomic semaphore protocol; touches
  30%+ of the kernel. Very high regression risk on the 40 already-WIN shapes.
- **Expected gain**: 2-6pp on L6, 0-3pp on other deep-K LOSEs (none
  currently exist post-v2).
- **Effort**: **weeks** (≥2 weeks). NOT < 2 hr.

### V8 — **K_EXACT=128256 K-LIMIT bypass via static-loop-split** (R25E peel
  rebirth, **not** R26-A pf495 wire)
- **What**: Distinct from R26-A (which set R25C_TAIL_PF_OFF_ITERS=497 inside
  the runtime-branched loop and faulted). Instead **statically split** the
  K-loop into TWO `#pragma unroll 8` loops: `[0, K_iters - PEEL)` with full
  prefetch, and `[K_iters - PEEL, K_iters)` with `PF_N=0`. No runtime branch
  inside the hot body → no code-size doubling → no aperture violation. This
  was R25-E's `R25E_K_LOOP_PEEL` worktree (`r25e-kpeel`); it was **never run
  on DLA1** because R26-A pf495 (the runtime-branch variant) was tried first
  and false-alarmed. The peel macro itself was not bench-tested at K=128256.
- **Why for K=128256**: same R25-F insight — B is L2-resident after iter 2,
  so the trailing ~6 iters of `buffer_load_lds` are dead weight × 501 / 16 = 31×
  more dead loads than at K=4096. If the peel emits clean two-loop ASM
  (verified by R25-E worktree compile artifacts), we get the R25-G mechanism
  on a shape that is structurally locked out of the runtime gate.
- **Risk**: MED (compiler may still merge the two loops; need ISA
  inspection). Failure mode is regression to current 92.6%.
- **Expected gain**: 1-4pp on L6 (matches R25-G tier on the proportional
  K-iter dead-pf fraction).
- **Effort**: ~1.5 hr to rebuild + smoke-bench, IF the `r25e-kpeel`
  worktree macro still exists and compiles on the post-R28-C kernel.

---

## 2. Top recommendation

**Defer all three.** None fits the <2 hour bar with quantifiable outcome:

- **V6 (split-K)** and **V7 (Stream-K)** are multi-day kernel rewrites with
  structural regression risk on the 40 WIN shapes; they need their own sprint.
- **V8 (R25E static peel rebirth)** is the only sub-day option. However, it
  has two pre-conditions that block a clean <2 hr scout:
  1. The `r25e-kpeel` worktree macro must still be live and apply to the
     post-R28 kernel (kernel:80-110 has been rewritten three times since R25-E
     was authored on commit `7ada8c70`-era code).
  2. ISA inspection is required after build to confirm the compiler emits
     two distinct loops vs merging back into one runtime-branched body
     (the precise failure mode that killed R26-A). That is a 30-min ISA
     dump + `awk` audit, not a yes/no.

  Even with both green, the smoke is 1 build (~3 min) + 5-rep bench (~5 min)
  + ISA audit (~30 min) + 5-rep verify on a different GPU for noise control
  (~5 min) ≈ 50 min IF everything works first try. With one build retry
  (highly likely on a never-tested macro) it spills past 2 hr.

## 3. Verdict

**L6 saturated at the < 2 hr work bar.** The remaining structural levers
(V5 MFMA32 ≥ 1 week per `R27_V5_MFMA32_SCOUT.md`; V6 split-K ≥ 3 days;
V7 Stream-K ≥ 2 weeks) all require dedicated multi-day sprints. **V8
(R25E static-peel revival)** is the cheapest unexplored axis but still
needs ≥ 2 hr including the ISA correctness audit that R26-A skipped — and
which is *the* reason R26-A produced a false-alarm wire.

**Defer L6 to next sprint.** Recommend funding V8 (1-day scout: pull
worktree, port macro to current kernel, ISA-audit, 5-rep × 3-pfoff
sweep) before V5 (which is 5-10× more expensive for similar predicted gain).

If user insists on a sub-2-hr action this session: re-bench L6's current
top-7 BTW variants 5-rep median on a guaranteed-idle GPU to **shrink the
93%-of-comp confidence interval**. Will not move L6, but eliminates noise
hypothesis as a free byproduct.

---

## References
- `R27_V5_MFMA32_SCOUT.md` — V5 BACKBURNER, 1.5-2 weeks, 0-5pp p50
- `R27C_VERIFY_VERDICT.md` — R27-C K_EXACT bypass DEAD (HSA aperture)
- `kernel_mxfp4_gluon_cpp.cpp:80-110` — R25C K_EXACT hard-gate K≤32768
- `R26_PLAN.md:32-41` — R25E_K_LOOP_PEEL macro (V8 source)
- `R28_PLAN.md:107-124` — explicit DEAD list (no overlap with V6/V7/V8)
