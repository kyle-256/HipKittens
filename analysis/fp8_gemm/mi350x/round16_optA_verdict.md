# Round 16 Optimizer A — Verdict

**Date:** 2026-04-17
**GPUs used:** 0, 1 (idle MI355X / gfx950)
**Methodology:** warmup=200, iters=500, 10% trim mean
**Hypothesis:** Adding `-mllvm -greedy-regclass-priority-trumps-globalness=true`
(regalloc family) on top of an existing `-mllvm -amdgpu-sched-strategy=iterative-ilp`
WIN should NOT collide (different LLVM option families) and may compound.

## Verdict: **DEAD END / NEGATIVE FINDING — 0/5 wins**

The compound iterilp + regclassglob did NOT amplify the R10/R11 iterilp
wins on any of the 5 winning shapes. Every candidate produced a
NEGATIVE single-shot delta vs the existing iterilp-only winner,
ranging from sub-noise (-0.32pp) to catastrophic (-14.82pp).

## Per-shape table

| Shape | Variant suffix | asm vs parent | asm vs iterilp | Smoke parent | Smoke iterilp | Smoke compound | Δ vs iterilp (pp) | Smoke gate (≥+0.5pp) |
|---|---|---|---|---|---|---|---|---|
| S1 14336×4096×32768   | `_lgk2_dc_r16a_iterilp_regclassglob`     | DIFF | DIFF | 4799.5 | 4897.5 | 4119.8 | **-14.82** | fail (catastrophic) |
| S2 16384×4096×28672   | `_u32_r16a_iterilp_regclassglob`         | DIFF | DIFF | 4987.5 | 5046.2 | 4972.6 |  -1.33     | fail |
| S3 4096×32768×28672   | `_v20_memc_r16a_iterilp_regclassglob`    | DIFF | DIFF | 5247.8 | 5346.3 | 5322.4 |  -0.43     | fail (within noise) |
| S4 4096×28672×32768   | `_u16_r16a_iterilp_regclassglob`         | DIFF | DIFF | 5292.8 | 5407.6 | 5389.7 |  -0.32     | fail (within noise) |
| S5 4096×32768×14336   | `_ts_lgk2_memc_r16a_iterilp_regclassglob`| DIFF | DIFF | 5086.6 | 5146.7 | 5083.2 |  -1.20     | fail |

All 5 produced DIFF .text bytes vs both the bare parent AND the
iterilp-only winner — confirming regclassglob really did affect register
allocation (not silently no-op'd by LLVM).

## Verify (5-run): SKIPPED

Per R16A methodology, only smoke-gate-PASS candidates qualify for 5-run
replication. 0/5 passed. No commits.

## Interpretation

1. **regclassglob is NOT a free axis to stack on iterilp.** Despite being
   in a different LLVM option family (regalloc vs scheduler), the resulting
   register allocation interferes with the iterilp scheduler's preferred
   register layout, producing inferior code on ALL 5 shapes.
2. **R14A's +0.236pp signal on P1 was iterilp-INDEPENDENT.** The P1 parent
   `_ts_gm8` does NOT use iterilp; on iterilp WINs the flag has the
   opposite sign.
3. **S1's -14.82pp** is the most extreme — using parent `_lgk2_dc` (which
   inserts `-mllvm -amdgpu-disable-clustered-low-occupancy-reschedule`)
   together with iterilp + regclassglob may have triggered a cascading
   misallocation. Note that the existing S1 iterilp-only winner used a
   DIFFERENT parent macro set (`_v16_wpe2`), so the prompt's "stack on
   `_lgk2_dc` parent" effectively tested a brand-new iterilp variant for
   S1, not a stack on the shipped winner.
4. **regalloc × scheduler interaction is not orthogonal in this kernel.**
   The compiled register pressure profile for iterilp is sensitive to
   the priority queue used by the greedy regalloc.

## Compound axis status (post-R16A)

| Pair | Result |
|---|---|
| iterilp × regclassglob | **DEAD END** on all 5 R10/R11 winners |
| memc × iterilp         | structurally impossible (single LLVM option, last-wins) |
| iterilp × padding-ratio | byte-identical no-op (R12) |
| iterilp × STEP12_BR_LGKMCNT | catastrophic (R12) |
| iterilp × STEP3_BARRIER_VMCNT=24 | sub-threshold (R12) |

The iterilp WIN ridge remains locally optimal; surrounding flag
combinations either no-op or regress. **R16A confirms that the regalloc
axis (regclassglob) is also OFF the table for stacking on iterilp.**

## Files

- `build_round16_optA_iterilp_regclassglob.{py,log}` — 5/5 OK builds (~5s)
- `asm_diff_probe_r16a.{py,log,json}` — 5/5 DIFF vs both parent and iterilp
- `bench_round16_optA_smoke.{py,log,json}` — 0/5 gate-pass (all NEG deltas)
- `bench_round16_optA_verify.{py,log,json}` — SKIPPED (no candidates qualified)
- `round16_optA_verdict.md` — this file

## Recommendation for R17

Stop searching the `-mllvm` flag space for stacks on iterilp. The
compound-flag search is exhausted. The next round should pivot to:

1. **Kernel-source frontier already explored in R14 — confirmed dead end.**
2. **TODO**: try stacking regclassglob on the 4 NON-iterilp shapes (DLA1,
   DLA2, DLA7, P1). R15A only confirmed the +0.24pp on P1 was sub-threshold;
   R15B/R15C may have other data here. (Not in R16A scope.)
3. **TODO**: investigate whether different mfma scheduling priorities
   (e.g., `-mllvm -amdgpu-mfma-scheduling=...`) might be a fresh axis.
