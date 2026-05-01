# Round 8 — rcr_4w port plan invalidated by occupancy data

## Baseline (round-8 entry)

`_metric_grouped_only.py`: `score=822, geomean=0.9858` (new high-side
of noise band; prior best 820). Per-round mean still ~816.

Lowest 5:

| shape                         | ratio |
|-------------------------------|------:|
| gpt_oss-Down-B4-M4096         | 0.934 |
| gpt_oss-GateUP-B4-M2048       | 0.940 |
| gpt_oss-GateUP-B4-M4096       | 0.946 |
| DSV3-Down-B16-M4096           | 0.951 |
| DSV3-Down-B16-M2048           | 0.955 |

## Finding: `rcr_4w::kernel` does NOT have higher occupancy

Round 7 recommended porting `rcr_4w::kernel` (line 805, 4-wave, WM=2 WN=2,
256 threads) to grouped on the claim that it achieves **4 blocks/CU = 4
waves/SIMD** (README line 34 and task body hint).

Actual `-Rpass-analysis=kernel-resource-usage` remarks show:

```
_ZN6rcr_4w6kernelE14layout_globals:
  TotalSGPRs: 46
  VGPRs: 198
  AGPRs: 256
  Occupancy [waves/SIMD]: 1      ← NOT 4
  VGPRs Spill: 0
  LDS Size [bytes/block]: 131072
```

Compare:

```
_Z11gemm_kernelIL6Layout0ELi0EE  (dense 8-wave RCR):
  VGPRs: 256, AGPRs: 0, Occupancy: 2 waves/SIMD, Spill: 12

_Z18grouped_rcr_kernelILi0ELb0ELb0EE  (grouped 8-wave RCR):
  VGPRs: 256, AGPRs: 0, Occupancy: 2 waves/SIMD, Spill: 67
```

`rcr_4w` has **LOWER** per-SIMD occupancy than both 8-wave variants
(1 vs 2 waves/SIMD). The `__launch_bounds__(NT=256, 2)` hint targets 2
blocks/CU but the compiler falls back to 1 because VGPR(198) + AGPR(256)
= 454 register footprint exceeds the 512-register budget for 2 blocks at
this register mode on gfx950.

## Why dense nevertheless dispatches to rcr_4w for large grids

README line 34 claim ("4 blocks/CU") misleads. The actual reason
rcr_4w wins on dense large grids (`aligned_grid >= 3200`) is **empirical**
— measured before this chat. Likely hypotheses:

1. **AGPR=256**: accumulators in AGPR free VGPR for memory/ALU ops,
   enabling deeper ILP inside the warp even at lower block-parallelism.
2. **256-thread block** (vs 512): finer-grained work distribution across
   CUs; better load-balancing on thousand-tile grids.
3. **Non-persistent (`grid=aligned_grid`)**: one tile per block
   eliminates the persistent stride stall/RAW inside the outer loop.

None of these is a "free occupancy" win. A grouped port would have to:

- **Invert (3)**: persistent single-launch is a task-body RED LINE.
- **Preserve (1)**: adapt AGPR accumulator pattern, which is NOT a
  constexpr change in the main loop.
- **Preserve (2)**: requires 256-thread-block setup, which invalidates
  the current cooperative-load code (same class of failure as round-6's
  WARPS flip).

Blast radius is higher than round-7 estimated: the "port rcr_4w" plan
is not a 2-3 round job — it's a **same-scale rewrite as a full new
persistent 4-wave kernel from scratch**, comparable to the WARPS flip
but with additional AGPR migration. Not feasible as the next step.

## Residual structural option: AGPR accumulator migration (grouped)

The only identified lever that MIGHT give a measurable gain without
repeating round-6's load-layout failure:

**Switch `cA/cB/cC/cD` from VGPR-backed to AGPR-backed rt_fl.**
Currently `rt_fl<RBM, RBN, col_l, rt_16x16_s>` resolves to VGPR-backed
register tiles; `mma_ABt` wraps the AGPR-output MFMA with a
v_accvgpr_read/write that moves MFMA output into VGPR. Migrating to
`mma_ABt_base` (direct AGPR) would save ~128 VGPR (the accumulator
footprint) and free those for spill reduction / memory ILP.

Scope: modify the 4-accumulator declarations + 4 `rcr_mma` call sites
in the main loop + 4 epilog call sites. AGPR can't be read by regular
ALU ops; the scale (`mul(cA, cA, scale)`) and store (`store(g.c, cA, ...)`)
would need v_accvgpr_read to migrate to VGPR first. Net VGPR saving
depends on whether the accumulator-to-VGPR moves are still at scale/store
time (should be OK — accumulator is live only across main loop iters,
scale is at end).

Risk: compiler may not track AGPR live ranges tightly, could still
allocate shadow VGPR. Same saturation/regression risk as other
refactors.

Scope estimate: 1-2 rounds. Much narrower than 4-wave port. Round 9
candidate.

## Round-8 action

Plan correction only. **No kernel change this round.** Metric unchanged
(post-compile re-run 822 → baseline band, same as entry).

## Commits

- HipKittens: this note (+ correction to round-7-dm note's plan).
- Primus-Turbo: mirror note.
