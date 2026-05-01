# Round 3-dm — FP8 grouped RCR main-loop `#pragma unroll` saturated

**Date**: 2026-05-01 (Round 3 of 100 in `dm` run)
**HK HEAD on entry**: `f9f2b545` (round-2-dm ST_v3 falsified)
**HK HEAD after commit**: this round (comment-only code delta; pragma
stays at shared `RCR_MAIN_UNROLL` macro = 2)
**Primus-Turbo HEAD**: `b732ade` (round-2-dm ST_v3 docs; unchanged
this round)

## TL;DR

Probed the unswept `TK_PRAGMA_UNROLL(N)` override on
`grouped_rcr_kernel`'s main-loop `for (k = 0; k < ki_dyn - 2; …)`.
Current code uses the shared `RCR_MAIN_UNROLL = 2` macro (same hint
as the dense kernel); grouped-specific sweep was NOT in the round-25
saturated-knobs inventory. Round-26 docs claimed "repeatedly tested"
but produced no numerical evidence.

**5-run sweep on the round-2-dm baseline**:

| unroll hint | n | mean score | median | range  |
|---|---|---|---|---|
| 1 (no unroll) | 5 | **816.2** | 818 | [810, 822] |
| 2 (baseline)  | 5 | **816.2** | 816 | [812, 819] |
| 4 (deeper)    | 5 | **816.4** | 818 | [812, 820] |

All three cells within 0.2 score of each other — perfectly flat.
**Saturated in both directions**: `#pragma unroll N` is an ignorable
hint at this body size; LLVM's own unroll heuristic picks the
effective unroll factor from body size + trip count (ki_dyn = 16 for
DSV3 K=2048, 22 for gpt_oss K=2880) and the resulting machine code
is indistinguishable across hint values.

Build-resource confirmation: VGPR spill identical at hints 1 / 2 / 4
(67 / 76 / 45 / 54 for the four `<FUSED_KTAIL, N_MASKED>` specializations
— matching round-2-dm baseline). No code-size or spill pressure
difference either. LLVM has already stabilised on its own unroll choice.

Code change: kept `TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)` at the grouped
main-loop site (unchanged from baseline); added comment block
documenting this round's sweep so future rounds don't re-probe. No
kernel semantic change. Metric restored to 813 post-revert, within
±3 noise of the 816 mean.

## Why probe this now

Round-1-dm / round-2-dm exhausted all exposed single-knob levers by
data — LDS layout swap (round 2 falsified correctness), config rules
(round 1 within noise), VMCNT / LGKM / sched_barrier / chunk_size /
(group_m, num_xcds) (rounds 4–27). Round-25 saturation inventory
notably omitted `RCR_MAIN_UNROLL`. Round-6 touched UNROLL=2 but only
inside the failed 2-tile body port, not on the current 1-tile body.

Hypothesis: at `ki_dyn ≈ 16–22`, deeper unroll may expose more
scheduling freedom across adjacent K-iters' load ↔ MMA dependencies,
or shallower unroll may reduce I$ pressure on the persistent grouped
kernel's larger binary. Both hypotheses falsified at 5-run sample
depth.

## What this leaves on the table

The cumulative "saturated single-knob" inventory after round-3-dm:

| knob | tested rounds | result |
|---|---|---|
| main-loop `sched_barrier` | 4, 8, 25 | removed (+0.45 pp) |
| epilog `sched_barrier` | 8, 25 | **load-bearing**, keep |
| `RCR_PREFETCH_LGKM` ∈ {2,4,8} | 5 | saturated |
| `RCR_STEADY_VMCNT` ∈ {4,8,12} | 8 | saturated |
| `RCR_INIT0/INIT1_VMCNT` | 24 | saturated; raise races |
| `RCR_EPILOGUE_VMCNT` ∈ {1..4} | 25 | saturated |
| `RCR_TWO_TILE_MIN_KI` 28→20 | 15 | no-op for ki=22 |
| `RCR_TWO_TILE_MID_VMCNT` | 6, 15 | grouped 2-tile catastrophic |
| `chunk_size` 64 → 32 | 22 | B=4 regresses |
| BN=128 dispatch | 24 | Triton uses 256 too |
| `(group_m, num_xcds)` | 21, 23, 1-dm | per-shape saturated |
| `BLOCK_SWIZZLE_NUM_XCDS` = 8 | 12-14 | hard MI355X HW |
| K-tail `load_a_kt` hoist | 27 | VGPR live-range cliff |
| `ST_rcr` LDS layout v2 → v3 | 2-dm | correctness FAIL |
| **`RCR_MAIN_UNROLL` {1,2,4}** | **3-dm** | **flat, compiler-driven** |

Every exposed hint / macro / pragma is now saturated or formally
falsified. Remaining path forward is multi-round structural only.

## Remaining multi-round structural options (unchanged from round-2-dm)

1. **FP8 K-tail amortize across M-slab** — 2-3 rounds. Round-8 §1
   design sketch. Low risk, graceful failure mode. Affects 8 gpt_oss
   shapes only (K=2880, K_REM=64). Estimated +1-3 pp per shape.
2. **FP8 direct HBM → register main loop** (task-body lever E,
   round-4 §9.1.c) — 4-8 rounds. Highest documented yield (+5-7 pp).
   Risk: naïve approach multiplies HBM traffic by 4× (WARPS_N) since
   A-tile is currently cross-warp shared via LDS; real design needs
   to preserve sharing differently.
3. **FP8 MFMA cell-shape rework** (16x16x128 → 32x32x64 scaled) —
   2-3 rounds. Round-12 rocprof partially falsified at MFMA-count
   level; but register-tile & LDS-layout knock-ons may still unlock.
4. **BF16 BK=32 + ns=3 port** — out of scope (BF16 `[watch]`, score
   doesn't move per task-body RED line).

## Round-4 suggestion

Commit to structural project (1) K-tail amortize. Round-4 step 1:
read-only audit of `grouped_rcr_kernel`'s outer persistent loop +
FUSED_KTAIL epilog (lines 2045-2331), identify the M-slab boundary,
check VGPR-pressure headroom, write round-4-dm design doc. First
kernel edit in round-5.

Rationale for picking (1) over (2) or (3):

- Graceful failure: if VGPR spills regress, revert is trivial;
  DSV3 (`FUSED_KTAIL=false` path) is untouched.
- Already has a mechanism hypothesis (round-8 §1).
- Direct counter-evidence to round-8's "K-tail relative cost 2-4%":
  will we hit the 0.5-1% target? Even partial win (2% → 1.5% = 0.5 pp
  per shape × 8 shapes = +0.25 pp overall = +2-3 score) is commit-
  worthy at this noise floor.

## Files touched this round

- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:2155-2171` — comment
  block explaining the UNROLL sweep (sandwiched between round-2
  baseline setprio/sched_barrier comments and the main `for` loop).
  No functional code change.
- `analysis/_notes/round-3-dm-fp8-grouped-main-loop-unroll-saturated.md`
  (this file).

## Commit

```
docs(round-3-dm): FP8 grouped RCR main-loop #pragma unroll sweep saturated ({1,2,4} means all 816.2-816.4 over 5 runs)
```
