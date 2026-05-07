# Round F — FP8 grouped BLOCK_SIZE=128 tile-port: correctness-fix landed, perf falsified

**Status (2026-05-07 EOD)**: structural port complete, correctness fix landed,
perf hypothesis falsified at end-to-end metric. Kernel kept as env-gated
scaffolding (`TURBO_FP8_B128=1` only; off in production); production score
unchanged at 691.

---

## TL;DR

* Round F hypothesised that a `BLOCK_SIZE=128` variant of `grouped_rcr_kernel`
  would lift the gpt_oss-20B B=4 dgrad shapes (currently tile-starved at
  ~1.26 tiles/CU under the outer `BLOCK_SIZE=256` kernel) by getting closer to
  the saturating ~5 tiles/CU the larger B=32 shapes already enjoy.
* M1+M2a built the namespace-shadowed type system + a byte-identical kernel
  body in `namespace kernel_b128`. Resources land cleanly: VGPR 104, AGPR 0,
  spill 0 (vs outer 256 / 0 / 30-54), occupancy 4 waves/SIMD vs outer 1.
* M2-debug-3 found and fixed a real correctness bug introduced by the port:
  the file-scope wait-counter macros (`RCR_INIT0_VMCNT=4`, `RCR_INIT1_VMCNT=6`,
  `RCR_PREFETCH_LGKM=8`, `RCR_STEADY_VMCNT=8`, `RCR_EPILOGUE_VMCNT=4`) are
  loose enough to be no-ops in the b128 prologue (which has half the LDS data
  per tile → half the vmcnt/lgkm increments per `rcr_8w_load_hoist` call), so
  the first MFMA fires before its operand registers retire from `ds_read`.
  Halving the 5 thresholds (replaced with hardcoded `vmcnt(2)/(3)/(4)/(2)` and
  `lgkmcnt(4)` inside the b128 body) restores 8/8 correctness PASS on the
  gpt_oss kernel-only metric (SNR 297 dB vs outer 297 dB on every shape).
* M5 perf falsification: with correctness restored, b128 is **slower than
  outer on every gpt_oss shape that enters the b128 path**, including the
  tile-starvation B=4 dgrad cases the port targeted (1487-1664 T vs outer
  1934-2564 T = -23 to -38 %). Round G picks a different lever.

---

## Per-shape numbers (gpt_oss kernel-only metric, MI355X, env=1 force-on)

| shape              | sec    | OUTER (env=0) | B128  (env=1) | Δ       |
|--------------------|--------|---------------|---------------|---------|
| GateUP_B4_M2048    | fwd    | 1878          | 1875          |  -3     |
| GateUP_B4_M2048    | dgrad  | 1934          | 1487          | **-447 (-23%)** |
| GateUP_B4_M4096    | fwd    | 2051          | 2077          |  +26    |
| GateUP_B4_M4096    | dgrad  | 2529          | 1606          | **-923 (-37%)** |
| GateUP_B32_M2048   | fwd    | 2022          | 2021          |  -1     |
| GateUP_B32_M2048   | dgrad  | 2564          | 1664          | **-900 (-35%)** |
| GateUP_B32_M4096   | fwd    | 2108          | 2120          |  +12    |
| GateUP_B32_M4096   | dgrad  | 2602          | 1608          | **-994 (-38%)** |
| Down_*             | all    | (unchanged)   | (unchanged)   | 0 (b128 dispatcher rejects K%128=64) |

All Down shapes go through the outer kernel because their `K_kern=2880`
is K%128=64 not aligned (M2a doesn't yet handle K-tail). So we only see the
b128 path on the GateUP H4-rerouted dgrad shapes (where `K_kern=N_orig=5760`
is 128-aligned).

Section averages with b128 forced on:

| section | OUTER avg | B128 avg | Δ     |
|---------|-----------|----------|-------|
| fwd     | 1913      | 1907     | -6    |
| dgrad   | 2105      | 1693     | -412  |
| wgrad   | 1790      | 1776     | -14   |

Score: production (env=0) = 691. With b128 forced (env=1) = 640. **-51 points
if we shipped b128 with auto-trigger** — clear regression.

---

## Why the correctness bug

`buffer_load_dwordx4 ... offen lds` (the workhorse of `rcr_8w_load_hoist`)
increments BOTH vmcnt and lgkm and decrements both when the LDS write
completes. Per `rcr_8w_load_hoist` call:

* OUTER ST_v2 (HB=128 × BK=128 = 16 KB): `memcpy_per_tile = 16384 / (16 * 512) = 2`
  → 2 vmcnt ops per thread per call.
* b128 ST_v2 (HB=64 × BK=128 = 8 KB):     `memcpy_per_tile = 8192  / (16 * 512) = 1`
  → 1 vmcnt op per thread per call.

The kernel prologue makes 4 `rcr_8w_load_hoist` calls before
`TK_WAIT_VMCNT(RCR_INIT0_VMCNT=4)`. In OUTER, that's 4 × 2 = 8 vmcnt
outstanding, so vmcnt(4) drains 4 of 8 → first 2 ST tiles guaranteed to
land. In b128, that's 4 × 1 = 4 vmcnt outstanding, so vmcnt(4) drains 0 of 4
→ no guarantee any prologue tile has landed → first `load_b(b0, b_tile(tic, 0))`
ds_read at line 3612 races with the buffer_load_lds writes for `Bs[0][0]`. The
explicit `s_waitcnt lgkmcnt(0)` at line 3617 will eventually drain those, but
between line 3612 (issue) and line 3617 (drain) the ds_read result may already
be captured into `b0`'s VGPR slot from stale LDS bytes. MFMA at line 3618
multiplies that stale operand. Errors propagate via `cA`'s accumulator into
every subsequent K-iter and every subsequent stored cell.

The fix halves the 5 wait thresholds proportionally to the HB ratio (b128 = 1/2
the LDS volume → 1/2 the wait counter). The lgkm steady-state hint at line 3616
(`TK_WAIT_LGKM`) is also halved; the explicit `lgkmcnt(0)` after it provides
the hard correctness guarantee, the hint just shapes when the compiler/hardware
starts the drain (no perf cost shown for either choice; default to 4 since 8
allowed correctness drift on B=4 K=5760 dgrad without the hardcoded vmcnt fix).

This bug is specific to the port — the outer kernel is correct because its
LDS volume matches the macro values that were tuned for it (Round-A and
documentation in the macro definition block).

---

## Why the perf hypothesis falsified

The original Round-F EV math (`round-F-fp8-tile-size-128-port-plan-and-EV.md`)
projected:

> tile-starvation cases (B=4 dgrad: 1.26 tiles/CU under BLK=256) should lift
> by ~1.5-2x as b128 brings them to ~5 tiles/CU, matching the saturated B=32
> regime.

Reality: GateUP B=4 dgrad does land at 4.84 tiles/CU under b128 (4x more
tiles than outer's 1.26), but performance drops 23-38 % instead of rising.

Per-tile overhead doesn't scale with tile size. A b128 tile and an outer tile
both pay:
* binary group search (~70 cyc),
* 8 `rcr_8w_load_hoist` issue cycles (4 prologue + steady-state ramp),
* 4-mul + 4-store epilog,
* prefetch issue + wait counter drain.

But the MFMA throughput per tile in b128 is 1/4 of outer's (each warp
processes 2 cells × 1 MFMA per K-step in b128 vs 8 cells × 1 MFMA in outer).
With 4x more tiles, total MFMA work is identical — but per-tile overhead
multiplies 4x. The CU-utilization gain from going 1.26 → 5 tiles/CU is
~1.5x in latency-hiding terms (B=4 outer was 51 % MFMA-busy, B=32 outer is
~80 % MFMA-busy per Round-D PMC analysis), but the 4x overhead inflation
swamps it.

The B=32 shapes (already saturated under outer at ~5 tiles/CU) lose more
in b128 (B32_M4096 dgrad -38 %) than B=4 shapes (B4_M2048 dgrad -23 %),
which is consistent with this story: saturated shapes have nothing to gain
from b128's "bring tiles_per_CU up" but pay the 4x overhead penalty in full.

Possible follow-ups (NOT this round):
* 4-wave (vs 8-wave) b128 layout. R47/R48 saw rcr_4w outer (BLK=256, 4 waves)
  beat 8w outer in some isoshape configs by reducing wave-contention. With
  b128, a 4-wave layout would have 2 CTAs/CU and possibly different
  per-tile-overhead profile.
* Hybrid dispatch: b128 only on the *most* tile-starved shapes (e.g.,
  tiles_per_CU < 0.5) where MFMA-busy% is lowest. The current metric has
  no such shape.
* Sub-tile-fusion: process 2 b128 tiles per CTA-iter, sharing prologue/epilog
  costs across them. Heavy refactor, distant payoff.

---

## What we shipped (commit dc4be6a2 → today)

1. `kernel_b128` namespace with shadowed BLOCK_SIZE / HB / RBM / RBN / ST /
   register-tile types (M1, commit `a844df35`).
2. Mechanical port of `grouped_rcr_kernel` body into `kernel_b128` (M2a,
   commit `dc4be6a2`).
3. M2-debug-3 wait-counter halving (this commit) — hardcoded inline-asm
   replaces the 5 file-scope `TK_WAIT_*` macro calls inside the b128 body.
4. M3 dispatcher gate quarantined (env=1 only; production traffic
   unaffected).
5. M5 falsification: this note + commit message.

What did NOT ship: M2b K-tail port, M5 auto-trigger, M5 metric ship.

Production score = 691 (unchanged from before Round F).
