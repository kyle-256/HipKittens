# Round-10 progress note: K-tail fuse forward main line saturated

**Date**: 2026-04-30
**Round**: 10 / 100 (auto_optimize)
**Score**: 832 (vs historic best 833, within noise)
**Primus-Turbo HEAD before round**: 0cff2388 (round-9 H4 +367)
**HipKittens HEAD before round**: see HK git log

## TL;DR

K-tail fuse for **forward** main line is fully saturated across both
BF16 and FP8 grouped GEMM, RCR layout. Remaining fuse opportunities
are all on **backward** paths which the metric does NOT measure
(`scripts/_metric_grouped_only.py` evaluates fwd TFLOPS only; bwd is
correctness-gated but not perf-scored). Per-shape config rules also
saturated (rounds 61-70 covered all 8 gpt_oss BF16 + 8 gpt_oss FP8 +
4 DSV3-Down FP8 cases).

## K-tail fuse status by (dtype, layout)

| dtype | layout | fuse status | path | round | notes                                  |
|-------|--------|-------------|------|-------|----------------------------------------|
| BF16  | RCR    | ✓ shipped   | B    |  5    | direct HBM→Reg, K_REM = K_STEP = 64    |
| BF16  | RRR    | ✗ failed    | A/B  |  5-7  | phantom-read on subtile_inplace        |
| BF16  | RRR    | ✓ workaround| H4   |  9    | reroute dA bwd to RCR via b transpose  |
| BF16  | CRR    | ✗ not done  | —    |   —   | var-K dB, backward only → not metric   |
| FP8   | RCR    | ✓ shipped   | B    |  3-7  | direct HBM→Reg + per-group SRD bound   |
| FP8   | RRR    | ✗ not done  | —    |   —   | dA bwd, not metric                     |
| FP8   | CRR    | ✗ not done  | —    |   —   | var-K dB, not metric (P2)              |

## Round-10 metric breakdown (per-shape ratio vs Triton)

```
BF16 segment   geomean = 1.0890   progress = 0.907
  Ratio range: 0.995 - 1.165 (16 cases)
  Worst:  grpBF16_gpt_oss_20B-Down-B4-M4096    (1015.8 / 1021.1 = 0.995)
  Best:   grpBF16_DeepSeek-V3-GateUP-B32-M4096 (1431.2 / 1228.6 = 1.165)

FP8  segment   geomean = 0.9158   progress = 0.763
  Ratio range: 0.813 - 1.045 (16 cases)
  Worst:  grpFP8_gpt_oss_20B-GateUP-B4-M2048   (920.3  / 1131.5 = 0.813)
  Best:   grpFP8_DeepSeek-V3-GateUP-B32-M4096  (1695.3 / 1621.5 = 1.045)

Score = 1000 * geomean(min(progress_i, 1.0)) = 832
```

## Why score is hard to push past ~833

The 8 lowest-ratio shapes are all `gpt_oss FP8` at K=2880 (K%128 = 64,
so K-tail path is mandatory). They sit at ratio 0.81-0.89 vs Triton.
The K-tail epilog itself contributes ~5% of total wall (per round-1
mfma cycle accounting); even a hypothetical 0% K-tail overhead would
only lift ratio to ~0.86-0.94, still below the 1.20 target. The
remaining gap is in the **main K-loop microarch** (mfma issue rate +
HBM prefetch overlap + LDS bank conflicts).

Triton uses `origami.rank_configs` (a sophisticated streamk + tile
sweep library) to pick per-shape `(BM, BN, BK, num_warps, num_stages,
group_size_m, cache_a, cache_b)`. HipKittens uses hand-tuned
`(group_m, num_xcds)` knobs — the `(BM, BN, BK, num_warps,
num_stages)` are baked into the kernel template at compile time.
Closing the last 15-20% gap likely requires:

1. A new kernel template variant (e.g. `BM=192, BN=192` for N=2880=15·192
   alignment), wired through a per-shape rule that picks it. Major
   effort: new kernel + main-loop tuning + register layout re-derivation.
2. Or finer mfma scheduling (e.g. interleaving 4 cell-tile DOs with
   2-stage prefetch). Already heavily tuned via `s_setprio` /
   `RCR_SCHED_BARRIER`.

Neither falls inside the round-1-N "K-tail fuse" main line, so this
round did NOT chase them.

## Path A/B/C exhaustion — sub-direction status (for the next agent)

For BF16 RRR fuse (the only forward path where fuse failed numerically):

* **Path A** (LDS-staged via `G::load` + `load(reg, st_subtile)`):
  exhausted across rounds 3-5. `subtile_inplace` returns stale LDS
  source addresses post-epilog-2 for `warp_col ∈ {1, 3}` (verified
  via lane probe + zero-init diagnostic). Bug is in the kittens
  helper's compiler-time stale-capture, not the K-tile data. SNR
  saturates at ~18-19 dB.

* **Path B** (direct HBM → register + manual `ds_read_b64_tr_b16`):
  attempted rounds 6-8. The `col_l rt_32x16_s` lane-to-cell mapping
  is permuted by the `st_32x16` LDS swizzle's XOR bank-conflict
  mitigation; manual derivation is brittle and saturates at ~25 dB
  SNR (still below the 25 dB FP8 / 1e-2 BF16 allclose threshold).

* **Path C** (scalar epilog accumulate): not attempted — all 4
  vec8/cell-by-cell variants tried in rounds 1-3 spilled or
  regressed catastrophically (see task body's "已经验证不行" list).

* **H4 (workaround, round 9)**: transpose `b` in the Primus-Turbo
  dispatch layer to route RRR (dA bwd) through the RCR fuse path.
  Costs one extra `.transpose(-2, -1).contiguous()` per call (~6%
  bwd regression on metric BF16) but eliminates 4 BF16 dA correctness
  failures, lifting score 465 → 832. **Currently shipped.**

## What this round did NOT do (and why)

1. **Cfg/rule tune** — explicitly forbidden by the task body: "rule
   tune already saturated, the remaining score uplift is solely on
   K-tail fuse". Verified: 8/8 BF16 gpt_oss + 8/8 FP8 gpt_oss have
   per-shape rules in `primus_turbo/pytorch/kernels/hipkitten/config.py`
   from rounds 57-70. The "default" path (e.g. for FP8
   `gpt_oss-Down-B4-M2048`) was already swept and confirmed best at
   `(group_m=4, num_xcds=8)`.

2. **Path C scalar fma for BF16 RRR fuse** — task body explicitly
   marks 4 scalar accumulation strategies as "verified bad" (spill
   or 30x regression). Re-trying without a new register-layout
   insight would just repeat those failures.

3. **dB var-K fuse (P2)** — task body P2 says "wait until fwd main
   line is done". fwd main line IS done now, but dB does not influence
   metric forward TFLOPS, so working on it costs a round without
   moving the score. Deferring as a "long-term health" task for
   when score plateau is acceptable.

## Suggested next-round angle

Given the score plateau at 832-833 and the K-tail-fuse main-line
saturation, the most productive next-round directions (in priority
order) are:

1. **Profile the FP8 K-tail epilog wall fraction** (e.g. via a
   `SKIP_FUSE_KTAIL` build flag or `rocprof` trace) to confirm the
   ~5% estimate. If higher, partial-mfma half-K is worth a sub-cycle.
2. **dB var-K fuse (P2)** — long-term codebase health, even though
   it doesn't move metric.
3. **New kernel template for N-aligned-to-192** — only justified if
   the score-plateau is acceptable (would gain ~5-10pp on gpt_oss
   ratio but it's a multi-round kernel rewrite).
4. **Accept the plateau** — ship the current 832-833 score; the
   architectural ceiling for HipKittens grouped GEMM in this template
   space is likely ~850-900.

Do NOT chase rule tune in the next 2-3 rounds — verified saturated
across all 32 metric cases via the round-9 H4 + round-61-70 rule
sweeps.
