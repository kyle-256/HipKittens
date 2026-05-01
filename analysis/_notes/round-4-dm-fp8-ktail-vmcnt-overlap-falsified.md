# Round 4-dm — FP8 K-tail vmcnt(8)-overlap + load reorder FALSIFIED (-20 score)

**Date**: 2026-05-01 (Round 4 of 100 in `dm` run)
**HK HEAD on entry**: `6b47f420` (round-3-dm unroll saturated)
**HK HEAD after this round**: commit + kernel revert (net no kernel
change; docs only)
**Primus-Turbo HEAD**: `9195a5f` (round-3-dm docs; unchanged)

## TL;DR

Two independent K-tail schedule probes this round:

1. **Add `s_setprio(1/0)` wrap to 4 K-tail mfmas** (the only mfma
   site without setprio — main loop + Epilog 1 + Epilog 2 all wrap).
   5-run result: mean 815.2 vs baseline 816.2 ⇒ **-1 score (noise)**.
   Neutral, not commit-worthy.

2. **Reorder K-tail loads so `a_kt1` issues LAST + split vmcnt(0) to
   vmcnt(8) + vmcnt(0)** so cA+cB mfmas overlap with a_kt1's HBM
   latency. Build OK with spill +6 on `<FUSED_KTAIL=true, N_MASKED=false>`
   and +2 on `<true, true>`. Single-run result: **score 797, -20 vs
   baseline 817**. Uniform regression across all 8 gpt_oss FP8
   shapes (3-4 pp per shape). Falsified.

Both probes reverted in-place; kernel restored to round-3-dm baseline.
Final metric verify: 823 (top of noise band).

## Probe 2 full per-shape delta (baseline → reorder+split)

| shape | baseline | round-4 probe | Δ pp |
|---|---|---|---|
| GateUP-B4-M2048  | 0.933 | 0.899 | -3.4 |
| Down-B4-M2048    | 0.974 | 0.932 | -4.2 |
| GateUP-B4-M4096  | 0.936 | 0.902 | -3.4 |
| Down-B4-M4096    | 0.928 | 0.889 | -3.9 |
| GateUP-B32-M2048 | 0.981 | 0.946 | -3.5 |
| Down-B32-M2048   | 0.983 | 0.952 | -3.1 |
| GateUP-B32-M4096 | 0.971 | 0.927 | -4.4 |
| Down-B32-M4096   | 0.974 | 0.950 | -2.4 |

All 8 gpt_oss FP8 cases regressed by 2-4 pp. DSV3 ratios unchanged
(`FUSED_KTAIL=false` variant untouched). Clear falsification.

## Mechanism hypothesis for why (2) regressed

Three factors that compound:

1. **VGPR live-range extension of `b0, b1, a`** across the extra 2
   mfmas between vmcnt(8) and vmcnt(0): the register allocator gives
   b0/b1/a longer lifetime, squeezing cD-accumulator / loop-temp
   slots and raising spill from 45 → 51 on `<0,0,1>` and 54 → 56 on
   `<0,1,1>`. Each additional spill slot adds ~20-50 cyc per K-tail
   block. This is the **same spill-cascade mechanism** that killed
   round-27 (`a_kt1` hoist), applied here to (b0, b1, a) instead.

2. **HBM-ordering assumption broken**: `vmcnt(8)` means "at most 8
   vmem ops outstanding", but the HBM controller does NOT guarantee
   strict FIFO completion order (especially across L2 cache levels).
   Assuming that 16 drained = "a + b0 + b1" and 8 remaining =
   "a_kt1 exactly" is too optimistic. Actual completion could be
   "some a_kt1 done, some a/b0/b1 still pending" → register wait
   stall inside cA/cB mfma execution path.

3. **setprio transitions fragment the mfma issue queue**: splitting
   4 sequential mfmas into 2 groups of 2, each with `setprio(1) ...
   setprio(0)` brackets, creates 4 priority transitions instead of 2.
   On gfx950, each transition has a 1-2 cycle bubble on the SALU
   issue port that shifts mfma start. The per-group-of-2-mfma mfma
   throughput drops by a few %.

The K-tail ~2-4% wall-time budget doesn't have headroom for these
three compounding tax items. Net cost > overlap benefit.

## What this means for the round-8 §1 "K-tail amortize" plan

Round-27 and round-4-dm both show the same failure pattern: ANY
schedule change that extends a register's live range (`a_kt1` in
round-27; `b0/b1/a` here) past the register allocator's current
bin-packing envelope triggers a spill cascade that erases the
latency-hiding benefit and often makes things worse.

The round-8 §1 "multi-tile-M amortize" requires:
- sharing b0/b1 K-tail registers across MULTIPLE output tiles
  (effectively a large live-range extension of both)
- plus doubling the accumulator lifetime (cA/cB/cC/cD across two
  output tiles instead of one)

VGPR budget in the `<FUSED_KTAIL=true, N_MASKED=false>` specialization
is already at 256 VGPR ceiling with 45 B/lane spill. Doubling
accumulators + extending b0/b1 lifetime will push well past the
ceiling. **The round-8 §1 design is almost certainly unfeasible in
the current register-tile layout.**

Hence: K-tail structural optimization requires **register-tile
re-derivation first**, not schedule tweaks. That's a 3-5 round
project: rework RBM/RBN or the A_row_reg/B_row_reg packing to free
VGPR headroom, THEN attempt the amortize.

Alternative path: **task-body lever E (direct HBM→register main
loop, skip A-tile LDS staging)** has a completely different failure
mode (potentially multiplying HBM traffic by 4× instead of VGPR
pressure). May be a cleaner attack vector.

## Cumulative saturated/falsified knobs after round-4-dm

Added two more falsification entries:

| knob | tested round | result |
|---|---|---|
| K-tail mfma setprio wrap | 4-dm | neutral (noise) |
| K-tail reorder + vmcnt(8) overlap | 4-dm | **-20 score** (VGPR cascade) |

The K-tail epilog schedule is now **very thoroughly falsified** —
round-3 shipped the current single-drain design, round-27 falsified
the a_kt1 hoist, round-4-dm falsified the reorder+split-drain variant
AND the setprio-wrap-only variant.

K-tail is essentially at a local minimum within the current register
tile layout. Can't improve without deeper structural change.

## Round-5 suggestion

Pivot away from K-tail.

**Option A — DSV3-Down scheduling probe (4 shapes, ratio 0.95-0.97,
NO K-tail / NO N-tail)**: DSV3-Down hits main-loop only. All round
4-27 knobs were saturated on the DSV3-Down geomean (0.95-0.97). The
task-body explicitly diagnoses this as "main loop throughput"
bottleneck. A 1-round probe worth exploring: check whether Triton's
per-K-iter `s_waitcnt`/`s_barrier` placement for K=2048 (ki=16) is
structurally different from HK's K-iter (which is calibrated for
ki=22 gpt_oss).

**Option B — Start multi-round project: FP8 register-tile
re-derivation** (pre-requisite for round-8 §1 amortize). This is
3-5 rounds of work with no guaranteed metric win until the end.
Pick only if willing to burn 5 rounds on structural prep.

Recommended: **Option A (1-round DSV3-Down probe)**. Quick data
gather; result informs whether any fast single-knob lever is still
live on the DSV3 side (where K-tail failure modes don't apply).

## Files touched

- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:2414-2427` —
  K-tail block edit + revert (net no change).
- `analysis/_notes/round-4-dm-fp8-ktail-vmcnt-overlap-falsified.md`
  (this file).

## Commit

```
docs(round-4-dm): FP8 K-tail vmcnt(8)-overlap + load-reorder falsified (-20 score, VGPR live-range cascade mirrors round-27)
```
