# Round-F — FP8 grouped-RCR `BLOCK_SIZE=128` tile-size port: plan, EV, milestones

## User direction

> "应该换一个tiling会好，我们的经验了"
> ("Change tiling and it'll be better, that's our experience")

User flagged tile-size dispatch as the next high-EV lever. Round-D PMC
established that the 51 % FP8-peak ceiling on best shapes is structural
for the 8-warp / 16x16-base / 4-acc design; Round-E falsified the
32x32-base port at EV gate. This round shifts attack from per-prim /
per-iter optimization to **per-tile load-balance** by adding a
`BLOCK_SIZE=128` (128x128 per-CTA tile) variant of `grouped_rcr_kernel`
and dispatching to it on shapes with low `total_tiles / NUM_CUS`.

---

## Per-shape evidence (current FP8 baseline, score=685, mean ~1917 T)

| Shape | Section | TFLOPS | tiles/CU @ BLK=256 | tiles/CU @ BLK=128 |
|-------|---------|-------:|--------------------:|--------------------:|
| Down  B4 M2048   | fwd  | **1465** | **1.5**  | 6.0  |
| Down  B4 M2048   | dgrad | 1408   | 1.5      | 6.0  |
| GateUP B4 M2048  | fwd  |  1846   | 2.9      | 11.6 |
| GateUP B4 M2048  | dgrad| 1898   | 2.9      | 11.6 |
| Down  B4 M4096   | fwd  |  1866   | 3.0      | 12.0 |
| GateUP B4 M4096  | fwd  |  2055   | 5.75     | 23.0 |
| Down  B32 M2048  | fwd  |  1872   | 12       | 48   |
| Down  B32 M4096  | dgrad |  1946  | 24       | 96   |
| GateUP B32 M2048 | dgrad |  2546  | 23       | 92   |
| GateUP B32 M4096 | dgrad | **2572** | **46** | 184  |

Strong correlation between `tiles/CU @256` and TFLOPS up to ~6 tiles/CU,
then plateaus. `B=4 M=2048` shapes hit 28-29 % FP8 peak (vs 51 % on best
shape) — well below the structural ceiling. Pure load-balance issue.

If `BLOCK_SIZE=128` lifts B=4 shapes from 28-29 % to 40-45 % peak (=
2014-2266 TFLOPS), expected gains:

| Shape (worst 4) | Current avg (3 sec) | Projected | Δ per shape |
|-----------------|--------------------:|----------:|------------:|
| Down  B4 M2048  | 1382 T | ~1900 T | +518 T |
| Down  B4 M4096  | 1788 T | ~2000 T | +212 T |
| GateUP B4 M2048 | 1782 T | ~2050 T | +268 T |
| GateUP B4 M4096 | 2182 T | ~2250 T |  +68 T |

Total gain across 12 B=4 timings ≈ +3200 TFLOPS = +267 mean across 24 →
section avgs fwd 1899 → ~2010, dgrad 2077 → ~2160, wgrad 1774 → ~1850.
**Score 685 → ~720..735 (+35..+50 score points).**

This is a **conditional dispatch**: B=32 shapes (12-46 tiles/CU) already
saturate amortization with BLK=256; routing them to BLK=128 would
give 48-184 tiles/CU = per-tile setup overhead dominates → expected
regression. So `kernel_b128` only fires when
`total_tiles / NUM_CUS < THRESHOLD` (initial T = 8).

---

## Why this is the right next round (vs prior options)

| Option | EV (score) | Cost | Status |
|--------|-----------:|-----:|--------|
| A: AGPR migration | +5..+10 % full = +35..+70 | 1-2 wk infra + 1 round | Blocked on missing FP8 art-mode mma intrinsics in HK headers |
| B: 32x32x64 main loop | +1..+2 % = +7..+15 | 1-2 wk port | EV-falsified at microbench (Round E) |
| C: 4-warp port (256x256/4w) | +5..+10 % | Tried R54-R61 | **BLOCKED**: deterministic LLVM AGPR allocation bug (cAB[0][0].tiles[{0,1}][1] wrong on 256-AGPR/64x64-cell shape; 4 mitigation attempts failed; pivoted to Lever A) |
| D: K-tail/main-loop overlap | +1..+2 % | 3-6 days | Low EV |
| **F: BLOCK_SIZE=128 (this round)** | **+5..+8 % full = +35..+50** | **3-7 days** | **Highest unblocked EV** |

Note that C tried to attack the same load-balance issue (4 warps → 2 CTAs/CU → effective 2× tile parallelism) but was blocked by a CODEGEN bug, NOT a fundamental EV problem. Option F attacks the same root cause via a different mechanism (smaller CTA tile → more total tiles → more parallelism per CU) and avoids the 4w/256-AGPR codegen path entirely (stays in 8-warp / VGPR-only / 4-cells per warp).

---

## Concrete file deltas required

### Type changes (compile-only, M1 — this commit, ~80 LoC)

* New `namespace kernel_b128 { ... }` after
  `lever_c2_round_54_step1_scaffold` (line 312)
* Constants: `BLOCK_SIZE_b128 = 128`, `HB_b128 = 64`, `RBM_b128 = 32`,
  `RBN_b128 = 16`, `_NUM_THREADS_b128 = 512` (8 warps unchanged)
* Type aliases:
    * `ST_v2_b128 = st_fp8e4m3<HB_b128=64, BK=128, st_16x128_v2_s>`
        — 4 sub-tiles of 16x128 (was 8 in HB=128)
    * `ST_v2a_b128 = st_fp8e4m3<HB_b128=64, BK=128, st_16x128_v2a_s>`
    * `A_row_reg_b128 = rt_fp8e4m3<RBM_b128=32, BK=128, row_l, rt_16x128_s>`
    * `B_row_reg_b128 = rt_fp8e4m3<RBN_b128=16, BK=128, row_l, rt_16x128_s>`
* `static_assert` cross-checks:
    * `sizeof(ST_v2_b128) == sizeof(ST_v2) / 2`
    * `sizeof(A_row_reg_b128) == sizeof(A_row_reg) / 2`
    * `sizeof(B_row_reg_b128) == sizeof(B_row_reg) / 2`
    * Per-warp acc footprint: 4 acc * RBM_b128 * RBN_b128 * sizeof(fp32)
      = 4 * 32 * 16 * 4 = 8192 B / warp / tile (vs 32768 B for BLK=256)
      = 16 fp32/lane/acc * 4 acc = 64 fp32/lane (vs 128 fp32/lane)

### Kernel function port (M2 — separate commit, ~700 LoC)

Direct copy of `grouped_rcr_kernel<KI_HINT, N_MASKED_STORE, FUSED_KTAIL,
FUSE_ACT>` (line 2542-3231) into namespace, replacing:
* All `BLOCK_SIZE` → `BLOCK_SIZE_b128` (literal 128)
* All outer-scope type names (`ST_v2`, `A_row_reg`, `B_row_reg`) →
  `_b128` suffixed namespace-local aliases
* All outer-scope constant names (`RBM`, `RBN`, `HB`, `_NUM_THREADS`)
  → `_b128` aliases
* `for (br ...)` and `for (bc ...)` loop bounds use `bpr_g = M_g /
  128` and `bpc = ceil_div(g.n, 128)` (4× more tiles per group)
* Cooperative load helpers (`rcr_8w_load_hoist<_NUM_THREADS, ST, GL,
  COORD>`) are already templated on ST and N_THREADS — no helper
  changes needed (verified at templates)

### Dispatcher gate (M3 — separate commit, ~30 LoC)

In `dispatch_grouped_rcr` (line 6350), before the existing kernel
launches:

```cpp
const int total_tiles_estimate = g.G * (g.M_total / g.G / 128) * kittens::ceil_div(g.n, 128);
// ^^ approximate: real total_tiles is computed inside the kernel via
//    cumulative scan; we use M_total/G as a proxy for the per-group M.
const int tiles_per_cu = total_tiles_estimate / NUM_CUS;
constexpr int B128_THRESHOLD = 8;  // tunable
const bool use_b128 =
    (tiles_per_cu < B128_THRESHOLD) &&
    (g.k % K_BLOCK == 0 || /* or accept K-tail variant */) &&
    (g.n >= 128);  // safety: at least 1 col-tile

if (use_b128) {
    kernel_b128::grouped_rcr_kernel<0, n_masked, fuse_ktail, fuse_act>
        <<<dim3(NUM_CUS), dim3(_NUM_THREADS_b128), 0, g.stream>>>(g);
} else {
    grouped_rcr_kernel<0, n_masked, fuse_ktail, fuse_act>
        <<<dim3(NUM_CUS), g.block(), 0, g.stream>>>(g);
}
```

### Correctness validation (M4)

Single-shape probe: Down B=4 M=2048 (worst shape, K=N=2880 — both
K-tail AND N-mask paths exercised). Gate: SNR > 25 dB on output.

If correctness passes, run full 8-shape FP8 metric (`scripts/_metric_gpt_oss_fp8_kernel.py`)
to measure score delta.

### Per-shape benchmark + threshold tune (M5)

Sweep `B128_THRESHOLD ∈ {4, 6, 8, 10, 12}` to find the crossover
point. Expected: B=4 shapes prefer b128, B=32 shapes prefer 256.

---

## Risk register (from rcr_4w precedent — R54-R61, BLOCKED)

Unlike rcr_4w which changed warp partitioning (4 warps × 64x64 cells),
the b128 variant changes ONLY tile size (still 8 warps × 64x32 cells →
shrunk to 8 warps × 32x16 cells). Per-warp accumulator footprint drops
to 64 fp32/lane (vs 128 in BLK=256). LLVM should pick standard VGPR
allocation (no AGPR), avoiding the AGPR-allocator codegen bug that
blocked 4w.

Specific risks:

1. **Per-tile setup overhead becomes dominant.** With 4× smaller per-tile
   work, the binary search + LDS prologue + epilog overhead is
   relatively 4× larger. Mitigation: only fire b128 when tiles/CU is
   low (the conditional dispatch). Risk: discovers no shape benefits
   at all → falsified.

2. **K-tail / N-mask path bugs.** The K-tail block in
   `grouped_rcr_kernel` (lines 2858-3107) hardcodes K_BLOCK and uses
   buffer_load offsets indexed by RBM=64 / RBN=32. Re-deriving for
   RBM=32 / RBN=16 is mechanical but error-prone. Mitigation: M1
   (compile-only) defers this; M2 ports without K-tail first
   (FUSED_KTAIL=false), M3 only dispatches b128 to K-aligned shapes
   (no K-tail needed) initially, then M4 ports K-tail.

3. **LDS layout swizzle re-validation.** ST_v2/v2a swizzle is per-
   sub-tile (16x128); HB=64 just means 4 sub-tiles instead of 8. No
   intrinsic incompatibility expected. PMC bank-conflict count should
   stay 0 (Round-D verified for HB=128 case).

4. **Metric regression on B=32 shapes if threshold mistuned.**
   Mitigation: pre-commit per-shape sweep; refuse to ship if any
   B=32 shape regresses > 2 %.

---

## Milestones

| M | Deliverable | Cost | Stop-here-if |
|---|-------------|------|--------------|
| **M1** | Namespace skeleton + type aliases + static_asserts compile | 1-2 hr | Type system rejects (e.g. ST_v2 doesn't accept HB=64 — unlikely) |
| M2 | `kernel_b128::grouped_rcr_kernel` body ported (no K-tail), compiles & links | 4-8 hr | Helper template instantiation fails on smaller types |
| M3 | Dispatcher gate added; b128 launches for low-tiles/CU shapes | 1-2 hr | Dispatcher API surface complications |
| M4 | Correctness PASS on Down B=4 M=2048 (or single-K-aligned proxy if K-tail not yet ported) | 4-12 hr | LLVM codegen bug (rcr_4w precedent) — escape hatch via VGPR-only allocation since accumulator footprint is now 64 fp32/lane = below AGPR threshold |
| M5 | Full 8-shape metric improvement, threshold tuned | 4-8 hr | Score regresses or no gain → falsified |

This file: **M1 deliverable.** M2-M5 are subsequent commits.

---

## EV summary (committed pre-M1)

* Best case: +35..+50 score (= +5..+8 % full kernel via B=4 shape gains)
* Worst case: 0 (falsified at M4 / M5 → port reverted)
* Cost: 3-7 days end-to-end
* Risk: medium-high (rcr_4w 7-round precedent shows codegen surprises;
  but b128 stays in standard 8-warp/VGPR allocation regime, avoiding
  the AGPR-specific bug that blocked C)

EV per day: 5-15 score / day in the best case; 0 if falsified.
Comparable to Round-A (5 score/hour but tiny scope) on per-day basis,
but Round-F is the only structural lever left after Rounds B-E
exhausted the small ones.
