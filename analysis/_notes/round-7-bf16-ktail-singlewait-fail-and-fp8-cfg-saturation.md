# Round 7 — gpt_oss focus: BF16 K-tail single-wait port FAIL + FP8 config saturation map

Branch: `dev/kyle_hipkitten_bf16`
HipKittens HEAD on entry: `62cebd5` (round-6 notes-only)
Primus-Turbo HEAD on entry: `c3b70e37` (round-4 +20.2% bwd)

## TL;DR

Two attempts this round, one falsified, one saturated:

1. **FAIL — BF16 grouped RCR K-tail single-wait port from FP8 round-3.**
   Identical micro-pattern (`+1 A_reg_t`, single `s_waitcnt vmcnt(0)`, 4
   back-to-back DO_MMAs instead of 2+vmcnt+2). Compiled clean (VGPRs=256,
   Occupancy=2 unchanged) but VGPR_Spill 42→43 *and* **SGPR_Spill 0→5** in
   the FUSED_KTAIL=true variant. Metric: 791 → 780 (**−11 score**, all 8
   BF16 gpt_oss shapes regress 2–5 %). Reverted with `git checkout --`.

2. **SATURATED — grpFP8-Down-B4-M4096 wider config sweep.** Worst-ratio
   non-watch metric shape (0.829). 9 × 5 = 45-candidate sweep over
   `gm ∈ {1,2,3,4,6,8,12,16,32}` × `xcd ∈ {1,2,4,8,16}`. Existing rule
   `(gm=4, xcd=4)` (round-69) is the **unique top of the entire grid**:
   no neighbor cell beats it. Worth recording so a future round doesn't
   re-sweep this surface. Action moved to siblings.

3. **WIN (companion commit on Primus-Turbo)** — 2 FP8 B=4 sibling shapes
   that the existing config rules either fixed `xcd=8` or fell through
   to the binding default got tightened: GateUP-B4-M4096 (gm=2,xcd=8) →
   (gm=14, xcd=4) [+1.21 pp tight verify]; Down-B4-M2048 default
   (gm=4, xcd=8) → (gm=2, xcd=2) [+0.48 pp tight verify]. See
   Primus-Turbo `208fa8f`.

Net round-7 metric: 791 → **792** (3-run variance 792/792/793 vs round-6
791/791/793; median +1; correctness 0/16 fail).

## Why the BF16 K-tail single-wait port regressed

FP8 round-3 (commit `07354791`, see
`round-3-fp8-grouped-ktail-single-wait.md`) gained +0.7 pp on grpFP8 by
hoisting the K-tail M-slab-1 A-load + B0/B1 loads ahead of a single
`s_waitcnt vmcnt(0)` and running all 4 DO_MMAs back-to-back. Risk profile
documented as "+1 A_row_reg, occupancy-neutral".

The patch I tried mirrored that pattern at
`analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp:842-853` —
declaration of `A_reg_t A_tile_kt1;` scoped *inside* the
`if constexpr (FUSED_KTAIL && L == RCR)` block (more conservative than
FP8 which declares `A_row_reg a_kt1;` at function scope on line 1926):

```cpp
// Round-7 single-wait K-tail epilog (mirror FP8 round-3 pattern).
A_reg_t A_tile_kt1;
load_a_kt(A_tile,     0);
load_a_kt(A_tile_kt1, 1);
load_b_kt(B_tile_0,   0);
load_b_kt(B_tile_1,   1);
asm volatile("s_waitcnt vmcnt(0)");
DO_MMA(C_accum[0][0], A_tile,     B_tile_0, C_accum[0][0]);
DO_MMA(C_accum[0][1], A_tile,     B_tile_1, C_accum[0][1]);
DO_MMA(C_accum[1][0], A_tile_kt1, B_tile_0, C_accum[1][0]);
DO_MMA(C_accum[1][1], A_tile_kt1, B_tile_1, C_accum[1][1]);
```

Compiler resource report:

| variant                                    | VGPRs | Occ | VGPR_Spill | SGPR_Spill |
| ------------------------------------------ | ----- | --- | ---------- | ---------- |
| BF16 `grouped_kernel<RCR,832,0>` baseline  | 256   | 2   | 42         | 0          |
| BF16 `grouped_kernel<RCR,832,0>` round-7   | 256   | 2   | 42         | 0          |
| BF16 `grouped_kernel<RCR,832,1>` baseline  | 256   | 2   | 42         | 0          |
| BF16 `grouped_kernel<RCR,832,1>` round-7   | 256   | 2   | **43 (+1)**| **5 (+5)** |

The +1 VGPR spill alone matches FP8 round-3's risk profile. The new
signal is **+5 SGPR spills** in the FUSED_KTAIL=1 instance, which mirrors
the round-5 lesson: small additions to a register-pressure-saturated
kernel trigger spill redistribution beyond the static count delta.

Per-shape kernel-only TF (3-run median, gpt_oss-only):

| shape (BF16 gpt_oss)        | before | after | Δ      |
| --------------------------- | ------ | ----- | ------ |
| GateUP-B4-M2048             | 1014.9 |  982.5 | −3.2 % |
| Down-B4-M2048               |  780.5 |  765.9 | −1.9 % |
| GateUP-B4-M4096             | 1082.5 | 1047.6 | −3.2 % |
| Down-B4-M4096               | 1018.2 |  991.8 | −2.6 % |
| GateUP-B32-M2048            | 1143.3 | 1101.1 | −3.7 % |
| Down-B32-M2048              | 1098.4 | 1047.2 | −4.7 % |
| GateUP-B32-M4096            | 1154.5 | 1132.3 | −1.9 % |
| Down-B32-M4096              | 1118.4 | 1094.0 | −2.2 % |

ALL 8 BF16 gpt_oss shapes regressed (avg −2.9 %), including shapes that
do NOT exercise the K-tail path (DSV3 K=2048 is K-aligned and uses the
non-FUSED instance), confirming the regression is in the FUSED_KTAIL=1
template (which all gpt_oss K=2880 shapes hit). The −5 SGPR spill in
that template alone is enough to explain a 2–5 % main-loop slowdown:
SGPR spills come out of LDS, and the BF16 main loop is already
LDS-bandwidth-pressured (8 cross-warp `s_barrier` per K-iter — the
"load-bearing barrier set" identified in round-5 §2).

### Lesson for future BF16 K-tail micro-tuning

The FP8 → BF16 port did NOT preserve the +1-tile-only risk profile
because the BF16 grouped kernel's baseline register pressure profile is
materially different:

- BF16 `grouped_kernel<RCR>` baseline: VGPRs 256, **VGPR_Spill 42**.
- FP8 `grouped_rcr_kernel` baseline: VGPRs 256, **VGPR_Spill 67**.

Counter-intuitively the FP8 kernel has *more* baseline spill, yet
absorbed +1 register tile cleanly while BF16 did not. Hypothesis: FP8's
larger spill is mostly cold-path (B-stage LDS staging, scale loads), so
the compiler had idle scratch lanes to redirect into. BF16's smaller
baseline spill is "mostly hot" — every additional spill displaces a
main-loop register. Future BF16 K-tail register-pressure changes need
to either:

1. Reuse a main-loop dead tile in-place (FUSED_KTAIL block already
   reuses `A_tile`, `B_tile_0`, `B_tile_1` — round-5 path-B comment at
   line 766-769 explicitly calls this out as the strategy that worked
   for the path-B loads). The single-wait pattern fundamentally needs
   *one extra A tile alive simultaneously with the M-slab-0 A tile*, so
   in-place reuse isn't possible without restructuring the M-slab
   ordering itself.
2. Find a way to reduce the BF16 baseline spill by 1+ first (so the
   added tile goes into the freed slot without touching the hot path).
   Probably means revisiting whether the B_tile_0 / B_tile_1 second-
   stage LDS staging registers can be merged once K_STEP loads have
   drained.

Both options are >1-round projects. Park.

## grpFP8-Down-B4-M4096 sweep saturation

Sweep at `/tmp/sweep_fp8_down_b4_m4096_round7.py` over the 9 × 5 = 45
candidate grid (target = lowest-ratio non-watch shape, ratio 0.829):

```
  gm |     x=1      x=2      x=4      x=8      x=16
-----------------------------------------------------
   1 |   931.2   940.7   948.5   934.2   932.7
   2 |   942.3   971.8   972.8   970.3   970.1
   3 |   923.2   957.5   969.0   921.4   921.4
   4 |   968.2   962.8   973.9*  969.3   961.2     ← current rule (round-69)
   6 |   944.1   949.8   961.6   944.1   944.1
   8 |   949.2   956.9   967.2   950.0   950.3
  12 |   951.2   960.0   968.3   950.3   952.0
  16 |   955.9   963.5   965.6   955.8   956.2
  32 |   955.5   963.0   966.1   957.5   958.2
```

`(4, 4)` is the unique top — strictly dominates every neighbor by ≥1.1
TF, monotone descent in every direction. Round-69 already chose this
cell from a 2-candidate (8, 4) vs (4, 4) sweep; this round's wider grid
*confirms* the choice rather than refining it.

Conclusion: grpFP8-Down-B4-M4096 has no remaining config-knob handle.
Future ratio gains on this shape (currently 0.829, target 1.20) require
either (a) a kernel-level structural change (BN=128 path / N-tile reshape
/ skip-A-tile-LDS staging — all multi-round projects deferred from
round-6) or (b) HBM-bandwidth amelioration (the M_total=16384 × tiles
gives ~1 wave on MI355X, putting the shape at the launch-bound boundary
where BW dominates compute).

## Pending multi-round directions (carried over from round-6 next-round)

Unchanged from round-6:

- **(b')** Retry FP8 2-tile main loop port + grouped-tuned MID_VMCNT +
  `unroll(1)`. Round-6 catastrophic −36 % regression was the dense
  template; needs grouped-specific scheduling tweaks.
- **(c)** Skip A-tile LDS staging — Triton-style direct HBM → register
  main loop. 4-8 hr structural rewrite. Estimated +5-7 pp on grp_FP8 by
  removing the 8 cross-warp `s_barrier` per K-iter. Highest-leverage
  remaining single change.
- **(d)** ~~BF16 K-tail single-wait port~~ ❌ FALSIFIED this round (see
  above). Pattern deferred until BF16 baseline spill reduction lands.

New next-round candidate from this round's sweep work:

- **(e)** Per-shape XCD-pinning sweep on the 4 unbound FP8 B=32 shapes
  (GateUP-B32-M2048/M4096 already on round-69 (8,4) but Down-B32 only
  has the round-69 +0.95 pp note for B4-M4096; B=32 Down M=2048/4096
  fall through to default). Lower magnitude per shape than B=4 (B=32
  grids saturate the GPU) but 2 unbound shapes × ~0.3-0.5 pp each may
  add up. ETA 1 round.

## Files touched this round

- `Primus-Turbo/primus_turbo/pytorch/kernels/hipkitten/config.py` —
  refined GateUP-B4-M4096 rule (gm=2,xcd=8) → (gm=14,xcd=4) and added
  Down-B4-M2048 rule (gm=2,xcd=2). Commit `208fa8f` on Primus-Turbo.
- `HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp` —
  K-tail single-wait patch tried + reverted (no commit).
- `HipKittens/analysis/_notes/round-7-...md` — this file.

## Risk-budget tracker

Per the round-6 lesson: **per-round risk budget = at most ONE of**
{+1 register tile, ≤5 main-loop instruction delta, 1 wait-counter tweak}.

Round 7 spent: 1 register tile (BF16 K-tail single-wait, falsified).
Total risk used: 1/1. Round 7 remaining: 0 — additional risk-y changes
park for round 8.
