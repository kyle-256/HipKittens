# Round 6 — FP8 grouped WARPS_M/WARPS_N flip catastrophically falsified

## Baseline (round-6 entry)

`score=820, geomean=0.9837, n=16`. Worst shapes:

- gpt_oss-GateUP-B4-M2048 @ 0.933
- gpt_oss-Down-B4-M4096 @ 0.940
- gpt_oss-GateUP-B4-M4096 @ 0.945
- DSV3-Down-B16-M2048 @ 0.961

Round-5 plan: try Option A (register-tile orientation via WARPS_M/WARPS_N
flip — `2/4 → 4/2`, auto-deriving `RBM=64,RBN=32 → RBM=32,RBN=64`).
Rationale: swap "tall" accumulators (64M×32N per warp) for "wide"
(32M×64N), flipping LDS read pattern on A vs B.

## Probe: `WARPS_M=4, WARPS_N=2`

Single-line change to `kernel_fp8_layouts.cpp:13-14`. All downstream
types auto-derive:

- `RBM = BLK/WARPS_M/2 = 256/4/2 = 32`
- `RBN = BLK/WARPS_N/2 = 256/2/2 = 64`
- `_NUM_WARPS = WARPS_M*WARPS_N = 8` (unchanged)
- `_NUM_THREADS = 512` (unchanged)
- `A_row_reg = rt_fp8e4m3<RBM=32, BK, ...>`
- `B_row_reg = rt_fp8e4m3<RBN=64, BK, ...>`

Build: clean. Resource remarks (grouped_rcr_kernel):

|                               | baseline (2/4) | flipped (4/2) | Δ      |
|-------------------------------|:--------------:|:-------------:|:------:|
| VGPRs                         | 256            | 256           | 0      |
| Occupancy [waves/SIMD]        | 2              | 2             | 0      |
| Spill (`<0,false,false>`)     | 67             | —             | —      |
| Spill (`<0,true,false>`)      | 76             | 90            | **+14**|
| Spill (`<0,false,true>`)      | 45             | 68            | **+23**|
| Spill (`<0,true,true>`)       | 54             | 80            | **+26**|
| ScratchSize (false,false)     | 272            | 344           | +72    |
| ScratchSize (true,false)      | 308            | 364           | +56    |
| ScratchSize (false,true)      | 184            | 228           | +44    |
| ScratchSize (true,true)       | 212            | 260           | +48    |

Spill + scratch increased across the board. Still occ=2 so no free
lunch.

## Metric (post-flip)

**Score: 8 / geomean: 0.0100 — ALL 16 shapes FAIL SNR** (fwd-snr < 25 dB):

| shape                          | fwd-snr (dB) |
|--------------------------------|-------------:|
| DSV3-GateUP-B16-M2048          |          7.4 |
| DSV3-Down-B16-M2048            |          3.1 |
| DSV3-GateUP-B16-M4096          |          2.6 |
| DSV3-Down-B16-M4096            |          0.8 |
| DSV3-GateUP-B32-M2048          |          2.7 |
| DSV3-Down-B32-M2048            |          0.9 |
| DSV3-GateUP-B32-M4096          |          0.9 |
| DSV3-Down-B32-M4096           |         −0.1 |
| gpt_oss-GateUP-B4-M2048        |         20.3 |
| gpt_oss-Down-B4-M2048          |  14.3 (dB)   |
| gpt_oss-GateUP-B4-M4096        |          9.0 |
| gpt_oss-Down-B4-M4096          |         20.3 |
| gpt_oss-GateUP-B32-M2048       |          1.6 |
| gpt_oss-Down-B32-M2048         |          4.2 |
| gpt_oss-GateUP-B32-M4096       |          0.4 |
| gpt_oss-Down-B32-M4096         |          1.4 |

Numerical garbage. **Falsified.** Reverted in the same round.

## Root cause (why it can't be a 1-line flip)

`rcr_8w_load_hoist<_NUM_THREADS>` (lines 435+) is hand-derived for the
2×4 warp layout: thread-to-swizzled-offset mapping assumes 8 warps
cooperate on a 128×128 LDS tile with a specific wave-stride pattern
that the 4×2 flip disrupts. Similarly, the `prefill_swizzled_offsets`
in `kittens::group<_NUM_WARPS>::prefill_swizzled_offsets` emits
warp-id-dependent offsets that are correct for 8 warps in a 2×4
spatial arrangement but scramble the cooperative load under 4×2.

What auto-derived (correctly): `rt_fl<RBM,RBN,...>` accumulator size,
`A_row_reg` / `B_row_reg` shapes, `subtile_inplace<RBM,BK>` / `<RBN,BK>`
row/col slicing, `rcr_mma` MFMA call shape.

What did NOT auto-derive: LDS-side cooperative-load thread-to-element
mapping in `rcr_8w_load_hoist`, `G::prefill_swizzled_offsets`, and
any `wm==1` / `wn*RBN` style inline indexing in prologue / main loop /
store.

## Round-6 outcome

- WARPS flip: **falsified catastrophically** (all 16 SNR < 10 dB).
  Kernel reverted + rebuilt + re-verified: 5-run metric post-revert
  lands 815 (within σ≈2 of baseline 816). Clean state restored.
- Blocks round-5's "Option A" path: register-tile orientation swap
  requires a multi-round LDS/cooperative-load rewrite, not a
  constexpr flip.

## Remaining structural options (ordered by blast radius)

1. **Port 4-wave RCR kernel (rcr_4w, namespace at line 805) to
   grouped**. Dense RCR uses a 4-wave variant (WM=2, WN=2, 4 blocks/CU)
   when `aligned_grid >= RCR_4WAVE_MIN_GRID=3200` AND `k <=
   RCR_4WAVE_MAX_K=8192`. All DSV3 shapes qualify (grid ≥ 4096,
   K ≤ 7168). Grouped has NO 4-wave variant — only the 8-wave
   `grouped_rcr_kernel`. Porting would lift occupancy from 2→4
   waves/SIMD for DSV3 (high-grid) shapes and open a ~15–25% TFLOPS
   headroom. **Cost: 2–4 rounds (rcr_4w has its own g2s/s2r
   helpers and kernel body; need to add group-offs binary search,
   per-group M, and at minimum the `N_MASKED_STORE=false,
   FUSED_KTAIL=false` variant — DSV3 hits this happy path).**

2. **MFMA cell shape 32x32x64 → 16x16x128**. Task-body Option C.
   Higher issue parallelism (2x MFMA/cycle) but RBM/RBN shrinks,
   more accumulator tiles. **Cost: 3–5 rounds (rewrite rt_fl,
   load helpers, rt_16x16_s → rt_16x128_s; falsified in round-12
   historical on gpt_oss but DSV3 has looser VGPR budget).**

3. **Reduce accumulator count 4 → 2 (drop cC/cD)**. Would free ~64
   VGPR at cost of processing half the tile per iter (doubled
   iter count). May lift occ 2 → 4 via VGPR pressure drop.
   **Cost: 3+ rounds (redesign cA/cB store ↔ double-tile schedule;
   high risk of regressing gpt_oss B=4 launch-bound shapes).**

## Recommendation for round-7

Start Option 1 (4-wave grouped port). Narrowest initial scope:

- Clone `rcr_4w::kernel` → `grouped_rcr_4w_kernel` with the same
  `grouped_layout_globals` signature.
- Reuse the existing `s_offs` / `s_cum_tiles` binary-search machinery
  from `grouped_rcr_kernel` (lines 2017–2072).
- Gate in `dispatch_grouped_rcr` with the same `aligned_grid ≥ 3200
  && k ≤ 8192` condition (+ add `K_rem == 0 && N aligned` to skip
  the K-tail / N-tail code path entirely; DSV3 qualifies).
- Target: DSV3-Down-B16-M4096 (ratio 0.949) first, measure lift.

## Commits

- HipKittens: this note (kernel reverted).
- Primus-Turbo: mirror note.
