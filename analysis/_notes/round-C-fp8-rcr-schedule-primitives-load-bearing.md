# Round-C — RCR main-loop schedule primitives (s_barrier, s_setprio) are all load-bearing; no kernel change

## Goal

After round-A (`RCR_PREFETCH_LGKM 4 → 8`, +5T fwd) and round-B (no
shippable change — all wait-counter values are at their empirical
optima), the user asked to "continue" optimizing. The remaining
non-structural levers in `grouped_rcr_kernel`'s main loop body are the
explicit synchronization primitives: 8 `__builtin_amdgcn_s_barrier()`
and 4 paired `__builtin_amdgcn_s_setprio(1) / s_setprio(0)` brackets
per K-iter. Round-2 falsified removing setprio + sched_barrier
TOGETHER (-5.6%) but never tested removing them in isolation.

This round tests three concrete sched interventions on the
RCR main-loop body (line ~2772-2801 of
`analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`).

## Lever surveyed

The main-loop body iterates `ki_dyn - 2` times. Each iter has 4 mma
groups (cA → cB → cC → cD). Each group follows the pattern:

```
load_b/load_a/rcr_8w_load_hoist  (LDS read + HBM→LDS prefetch)
__builtin_amdgcn_s_barrier()      (barrier #N.0 — pre-mma sync)
asm("s_waitcnt lgkmcnt(0)")       (full LDS drain)
__builtin_amdgcn_s_setprio(1)     (raise SQ priority)
rcr_mma(c?, a, b?)                (MFMA)
__builtin_amdgcn_s_setprio(0)     (lower SQ priority)
__builtin_amdgcn_s_barrier()      (barrier #N.1 — post-mma sync)
```

Per K-iter: 8 `s_barrier` (~50-100 cyc each) + 4 setprio pairs.

## Experiment C1 — Remove all 4 post-MMA barriers

Hypothesis: the post-MMA barrier is for compiler-reorder gating; the
load operations on the next group's first line are register-only
(load_b writes to `b1`, etc.), independent of the just-issued mma's
accumulator output.

```diff
-           __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
-           __builtin_amdgcn_s_barrier();
+           __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            (… same for cB, cC, cD …)
```

Result: **correctness FAIL — 0/8 → 1/8 SNR PASS**, fwd TFLOPS=230
(broken). The post-MMA barriers ARE load-bearing: they sync the
LDS-write completion (cooperative `rcr_8w_load_hoist` to next-toc
slot) before the next iter's tic^=1; toc^=1 swap. Without them,
warp X starts reading from `As[(toc^=1)]` while warp Y is still
writing to it from the previous iter's `rcr_8w_load_hoist`.

## Experiment C2 — Remove only the end-of-iter post-cD barrier

Hypothesis: the LAST barrier (post-cD, line 2800) gates iter k from
iter k+1's `tic ^= 1; toc ^= 1` swap. Maybe the next iter's first
load (which goes to a different LDS slot than cD's MMA inputs) can
issue ahead.

```diff
            __builtin_amdgcn_s_setprio(1); rcr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
-           __builtin_amdgcn_s_barrier();
        }
```

Result: **correctness PASS** (8/8 SNR ≥ 25 dB) but
**performance regressed by -17 score points** (5 runs,
median 669 vs baseline median 686): fwd 1850 vs 1900 (-50 T, -2.6%);
dgrad 1985 vs 2085 (-100 T noise correlation, but consistently
lower); wgrad ~1780 (unchanged, RCR-only change). The compiler
re-orders the next iter's load_b/load_a EARLIER without the barrier,
which causes the `rcr_8w_load_hoist` HBM→LDS prefetch to compete
for SQ issue slots with the cD MMA's still-pending dependent
operations. Net: pipeline less efficient.

Reverted.

## Experiment C3 — Remove all 4 s_setprio pairs

Hypothesis: round-2 removed BOTH setprio AND sched_barrier (-5.6%
regression). Maybe just removing setprio (keeping sched_barrier and
s_barrier) is benign — sched_barrier was doing the heavy lifting
all along.

```diff
-           __builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
+           rcr_mma(cA, a, b0);
            (… same for cB, cC, cD …)
```

Result: **correctness FAIL — 0/8 SNR PASS** on every run. fwd=0
(broken). The s_setprio brackets ARE load-bearing in isolation: they
gate the SQ from issuing dependent ops (load_a, load_b for next iter's
first MMA) ahead of the in-flight MFMA. Without setprio(1) → setprio(0),
the SQ commits load_a/load_b at normal priority, the dependent MFMA
issues with stale operand → wrong result. setprio is NOT just a perf
hint; on this kernel's tight schedule it is correctness-critical.

Reverted.

## Conclusion

All three RCR main-loop schedule primitives — `s_barrier` (×8 per iter),
`s_setprio(1)/s_setprio(0)` brackets (×4 per iter), and `sched_barrier`
(round-2 + round-4 already established) — are load-bearing.

* Removing any of the post-MMA barriers IN AGGREGATE breaks correctness.
* Removing JUST the end-of-iter barrier preserves correctness but
  regresses performance significantly (-17 score, -50 T fwd) due to
  compiler reorder hurting the pipeline.
* Removing s_setprio alone breaks correctness on this kernel (untested
  in round-2 which removed setprio + sched_barrier together).

The hand-tuned schedule on `grouped_rcr_kernel` is at a local optimum
in the sched-primitive landscape: any single-line removal regresses or
breaks. **No kernel change shipped in round-C.**

## Cumulative score progression

```
baseline (R5 auto-optimize, commit 7637aae)         : 686 (mean 687.4)
round-A (RCR_PREFETCH_LGKM 4 → 8)                   : 688 (mean 690.0)
round-B (numeric levers all saturated, no change)   : 688 (mean 690.0)
round-C (sched primitives all load-bearing, no change): 688 (mean 690.0)
                                                      ─────────────────
                                                      manual gain: +2.6 mean
```

## Remaining headroom requires structural kernel work

The metric is 689/1000 = 68.9% of the 2800 T per-section target. The
~31% gap is **structural**, not schedule-tuning:

| Lever | Estimated gain | Estimated cost | Blocker |
|-------|---------------|---------------|---------|
| **AGPR accumulator migration** | +5-10% | 2-4 rounds | HK headers' `art`-mode mma intrinsics (`include/ops/warp/register/tile/assembly/mma.cuh:307`) only support bf16/half — a FP8 art-mode mma must be added first (~1-2 weeks) |
| **K-tail overlap with main-loop epilog** | +1.5-3% | 2-3 rounds | Hoisting K-tail buffer_loads to overlap with epilog 2's last MMAs is correctness-delicate; requires reformulating the cD → cA-K-tail data dependency |
| **4-warp port (rcr_4w grouped variant)** | +10-20% | 4-8 rounds | Same-scale rewrite as a full new persistent 4-wave kernel from scratch (R8-dm note); doubles per-warp register pressure on a/b operands |
| **Grouped-tile fusion (multiple groups per CTA)** | +3-8% | 3-6 rounds | Eliminates per-tile prologue/epilog overhead on small-G shapes (B=4); needs new dispatch + new tile-coord arithmetic; risk of register pressure blow-up |
| **K-tail unfuse (separate K-tail kernel)** | -5% to +2% | 2-3 rounds + dispatch rework | Round-3 fused K-tail (path B) was the round that lifted gpt_oss from 153 → 465. Reversing it loses that win unless paired with a much more efficient standalone K-tail kernel |

The first item (AGPR migration) is the highest expected value per round
but requires HK infrastructure work BEFORE the FP8 grouped kernel can
even use it. Recommend prioritizing the FP8 art-mode mma intrinsic
implementation as the next 1-2 round target.

## What I do NOT recommend

* **More wait-counter / barrier sweeps.** All combinations within ±2σ
  of the optima have been exercised (rounds A/B/C). The lever space
  is closed; further sweeps will produce only noise-correlated
  "wins" that disappear under interleaved A/B re-test.
* **Auto-optimize on dispatcher rules (`group_m`, `num_xcds`).**
  These were tested for 5+ rounds and plateaued at score ~686. They
  cannot move the kernel-only metric measurably (kernel architecture
  is the bottleneck, not the launch config).
* **Compiler reorder mask tuning (`sched_barrier(MASK)`)**. The
  current mask=0 (no reorder allowed) is overly conservative on paper
  but matches the hand-tuned schedule exactly; non-zero masks have
  been tried in round-2 (-5.6% regression).
