# Round 8 — gpt_oss focus: 3 micro-knob experiments, ALL falsified

Branch: `dev/kyle_hipkitten_bf16` (Primus-Turbo) /
        `save/fp8-progress-20260319-native-layouts` (HipKittens)
HipKittens HEAD on entry: round-7 notes (`07248152`)
Primus-Turbo HEAD on entry: `fd9580fd` (round-13 docs)

## TL;DR

Three independent micro-knob experiments this round, all neutral-to-
slightly-negative at the metric level after 8-10 sample comparisons:

1. **FAIL — BF16 grouped `main_loop_iter` `sched_barrier(0)` removal**
   (gated `!FUSED_KTAIL`). 8-run mean 792.6 vs baseline 793.6, **min
   790 vs baseline 793 (−3 pts worst case)**.

2. **FAIL — FP8 grouped Epilog 1 & Epilog 2 `RCR_SCHED_BARRIER()` removal**
   (3 hints/output-tile, untouched by round-4). 8-run mean 793.5 vs
   baseline 793.6.

3. **FAIL — `RCR_STEADY_VMCNT` knob sweep** (default 8 → tested 4 and 12).
   Initial 5-run RCR_STEADY_VMCNT=4 batch looked promising (mean 794.6,
   median 795, max 796) but wider 10-run comparison showed it was sample
   variance: 10-run RCR_STEADY_VMCNT=4 mean 793.9 / median 794 / max 796,
   versus 10-run baseline RCR_STEADY_VMCNT=8 mean **794.2** / median 794 /
   max **797**. Baseline is **slightly better** at this sample size.
   RCR_STEADY_VMCNT=12 was clearly neutral (5-run mean 793.4).

Net round-8 metric: 794 → **794** (no improvement; commit is notes-only).
**Important calibration:** baseline metric has true mean ~794 and ceiling
~797 over 10 runs, even though `auto_optimize.py` reports `best=794`
based on single-run per round. The metric noise band is ±3 pts.

## Why round-4's FP8 main-loop sched_barrier win does NOT generalize

Round-4 (commit `333074d6`, see `round-4-fp8-grouped-rcr-sched-barrier-
removal.md`) gained +0.45 pp on grp_FP8 by removing 2 instances of
`RCR_SCHED_BARRIER()` in the FP8 grouped RCR main loop. The mechanism
documented there: `sched_barrier(0)` is a pure compile-time hint that
prevents LLVM machine-scheduler reordering across the barrier; removing
it lets the back-end scheduler co-issue VMEM loads with MFMA across what
were previously separated regions. Per-K-iter × ~22 K-iters per tile
× ~768 tiles = ~16,896 sites where the back-end gains scheduling freedom.

Two follow-up extrapolations both failed to produce measurable gains:

### (a) BF16 main-loop sched_barrier removal (this round, attempt 1)

The BF16 grouped path in `kernel_bf16_dynamic.cpp` shares
`device_gemm_tile_body` with dense GEMM (round-4's FP8 has separate
files for dense vs grouped). The 4 `sched_barrier(0)` calls in
`main_loop_iter` are inside the shared template, so any unconditional
removal would touch dense (forbidden by task scope). Gating on
`!FUSED_KTAIL` confines the change to gpt_oss BF16 grouped (the only
BF16 K-misaligned variant — DSV3 is K-aligned, dense is K-aligned) but
the metric showed no median gain and a −3 pt worst-case dip.

**Hypothesis on why it didn't help:** the BF16 main_loop_iter processes
2 K-iters per call with 4 barriers total = 2 barriers per K-iter,
matching FP8's density. But the BF16 lambda already has tighter
operand dependencies (4 `s_waitcnt lgkmcnt(0)` + 1 `TK_WAIT_LGKM` per
2-K-iter call) than FP8, leaving the back-end scheduler less freedom
to reorder even with the hint removed. Mechanism that worked for FP8
relied on FP8's looser `s_waitcnt`-around-MMA pattern.

### (b) FP8 epilog sched_barrier removal (this round, attempt 2)

The 3 epilog hints are per-output-tile, not per-K-iter. With ~768
output tiles per metric run and ~22 K-iters/tile, the main-loop hints
were ~16,896 sites; the epilog hints are only ~2,304 sites — 7× fewer.
Even if mechanism applied, expected gain is ≤ +0.06 pp (7× smaller),
inside metric noise band (±3 pts). The metric was indeed indistinguishable
from baseline. Without ≥ +0.05 pp signal at this site, can't justify
shipping a kernel change here.

### (c) RCR_STEADY_VMCNT knob sweep (this round, attempt 3)

Default RCR_STEADY_VMCNT=8 controls the per-K-iter `s_waitcnt vmcnt(N)`
threshold inside the FP8 RCR main loop. With UNROLL=2 and 4 HBM loads
per K-iter, 8 outstanding allows ~2 K-iters in flight without VMEM stall.

Round-5 sweep tested RCR_PREFETCH_LGKM ∈ {2,4,8} and found ±2 TF
insensitive but did **not** sweep RCR_STEADY_VMCNT. This round filled
that gap:

| RCR_STEADY_VMCNT | n  | mean  | median | min | max | comment       |
| ---------------- | -- | ----- | ------ | --- | --- | ------------- |
| 8 (baseline)     | 10 | 794.2 | 794    | 792 | 797 | best mean     |
| 4 (tighter)      | 10 | 793.9 | 794    | 792 | 796 | initial 5-run |
|                  |    |       |        |     |     | mean was 794.6|
|                  |    |       |        |     |     | (sample var)  |
| 12 (looser)      | 5  | 793.4 | 793    | 792 | 795 | clearly worse |

**Hypothesis on why it didn't help:** the back-end OoO scheduler already
deals with VMEM latency hiding well at the HW level; the explicit
`s_waitcnt vmcnt(8)` hint at the *end* of each K-iter is mostly serving
as a fence to prevent next-iter LDS reads from starting before the
fetched HBM data has been observed by the L2. Tightening to 4 forces
earlier drains but adds back-pressure on subsequent loads (smaller
in-flight queue depth → more L2 misses uncoalesced → marginally slower).
Loosening to 12 is essentially never-wait, which removes the fence and
lets unrelated work scramble — slight regression too. Default 8 is
already the sweet spot.

The first 5-run RCR_STEADY_VMCNT=4 batch happened to sample the upper
tail (5/5 above 791), making the ratio look like a +1 pp improvement.
Wider sampling regressed back to baseline ±0.3 pp. **This is a useful
calibration data point: 5-run signals at this micro-tune scale should be
dismissed; require 10+ runs for any genuine ±0.5 pp shift.**

## Round-7's BF16 K-tail single-wait port stays falsified

This round's BF16 sched_barrier experiment is a SECOND independent
falsification of "FP8 round-X micro-optimization → BF16 will benefit
similarly". Combined with round-7's BF16 K-tail single-wait failure
(−11 score from +1 VGPR / +5 SGPR spill cascade), the pattern is clear:
**BF16 grouped is operating at a different point on the
register-pressure × scheduler-freedom surface than FP8 grouped**, so
mechanical port of FP8 wins is no longer a viable strategy. Future
BF16-side optimizations need to be designed *for BF16*, not ported.

## Calibration: auto_optimize "best" tracker under-counts true ceiling

`auto_optimize.py` records `best=794` based on one metric run per round.
The 10-run baseline distribution this round shows true mean = 794.2
and max = **797** (a +3 pt ceiling above the recorded best). This means:

- Single-round +1 score gains on this metric are inside the noise band.
- Real wins need either (a) a clear shift in median (+1.5 pp or more)
  *or* (b) a structural change with mechanism evidence (rocprof / VGPR
  delta / per-shape ratio shift).
- The plateau at "score ~794" does NOT mean we're stuck — it means we're
  measurement-limited at the current sampling level. Chasing +1 pp via
  micro-knob tuning is below the metric's discriminating power.

## What's NOT yet falsified (next-round candidates)

Strict ranking by hypothesis weight:

1. **K-tail amortize (M-dim multi-tile per persistent wg)** — round-30
   start guidance ④. A persistent wg processes multiple BM-rows of one
   K-tile, sharing one K-tail epilog across them. Reduces K-tail relative
   cost from 2-4% → 0.5-1%. Requires kernel structural change (shared
   accumulator per-row, single end-of-tile K-tail call). Risk: VGPR
   pressure (multiple BM-rows × MMA = +N VGPRs/wg).

2. **N=2880 / N=5760 BN=128 sweep** — round-30 start guidance ②. Default
   BN=256 leaves 64-col-wide N-tail at 25% utilization. BN=128 doubles
   that to 50% but doubles grid (768 → 1536 tiles ≈ 2 waves). Worth
   measuring per-shape; round-30 start said Triton uses BN=256, but that
   data point is from before round 11/12/13 host-overhead trim that
   bumped overall ratio +1-2pp. Re-measure.

3. **Per-shape XCD-pinning sweep on B=32 unbound shapes** — small,
   typically +0.1-0.3 pp. Only worth doing in batch (sweep once, write
   rules into `select_default_config` once).

## Files / commits

- HipKittens: `analysis/_notes/round-8-grouped-sched-barrier-removal-falsified.md`
  (this file). All three kernel changes reverted; HEAD-on-disk = baseline.
- Primus-Turbo: no change.
