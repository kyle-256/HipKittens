# R52 Dev R — Per-shape N-tile shrink for V2 RRR at K=4096 — REFUTED-PLUMBING (pre-bench)

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 72a3c217 (R52N landed)
**GPU:** MI355X (gfx950) — pre-bench static / structural forensics; no GPU
benches run because the lever is not authorable inside the agent budget.
**Mandate:** R52P §5.3 — reduce per-CTA `N_tile` for K=4096 RRR shapes to
shrink the B working-set per CTA, with the goal of letting more B fit in
L2 simultaneously across dispatched CTAs and reducing TC return contention.
R52Q REFUTED R52P's primary cachepolicy lever (§5.1); §5.3 was the only
remaining V2 RRR data-feed lever R52P recommended.

## TL;DR — VERDICT: REFUTED-PLUMBING (gate triggered: "deep template surgery beyond a parameter change")

| Variant            | per-CTA N_tile | Authorable in budget? | SNR/det | Bench | vs baseline |
|--------------------|---------------:|-----------------------|---------|-------|------------:|
| Baseline (V2 RRR)  | 256            | YES (already shipped) | 49.61dB / 3-of-3 | 2549 TFLOPS (R52Q ref) | — |
| Half-N (BLK_N=128) | 128            | **NO — multi-day surgery** | n/a | n/a | n/a |
| Quarter-N (BLK_N=64) | 64           | **NO — same blocker**       | n/a | n/a | n/a |
| 1.5×N (BLK_N=384)  | 384            | **NO — same blocker (and BLK%128≠0 violates LDS swizzle)** | n/a | n/a | n/a |

**Why REFUTED-PLUMBING:** The MXFP8 RRR exact 8-wave fastpath
(`rrr_mxfp8_exact_8wave_fastpath.inc`) hard-codes the per-CTA tile via two
`static_assert`s — `BLK == 256` and `WARPS_N == 4` — and the V2 scale-pack
slab math is hardcoded for `pack_count=2` on B (`slab_bytes_b = 64 * padded_k_blocks`
with byte stride literals `lane_kblk*128 + lane_nonk*8` and `k_pair * 512`).
Authoring an N-tile-shrunk RRR variant requires coordinated changes across
(a) the kernel template, (b) the host-side preshuffle in
`preshuffle_scale_matrix_mfma16_v2_rcr_b` (Python), (c) the LDS allocation,
(d) the dispatch wrapper, and (e) the output-store coordinate math.
Numeric correctness is binary: any mistake in the scale-pack stride math
yields silent SNR collapse rather than a compile error. Total scope ≥ 1
day of careful work plus several rebuild/verify cycles.

Per the agent constraints — *"If the N-tile shrink requires deep template
surgery beyond just changing a parameter (e.g., recomputing accumulator
layouts, scale-pack indexing, etc.), document the scope and decide
pragmatically — don't go down a rabbit hole."* — the gate triggers.

**The only existing N-tile shrink scaffold (`MXFP8_RECT_BLK_N=64`, R28D)
is CRR-only and explicitly force-disables the V2 fastpath. There is no
RRR analogue.**

## 1. Where the per-CTA N-tile is configured

The RRR V2 fastpath uses `BLK = GEMM_BLOCK_SIZE = 256` for both M-tile and
N-tile per CTA, partitioned across `WARPS_M=2 × WARPS_N=4 = 8` warps.
Per-CTA tiling math (`kernel_mxfp8_layouts.cpp:335–352`):

```cpp
constexpr int BLK = GEMM_BLOCK_SIZE, BK = GEMM_K_BLOCK;  // 256, 128
constexpr int HB  = BLK / 2;                              // 128
constexpr int WARPS_M = 2, WARPS_N = 4;
constexpr int RBM = BLK / WARPS_M / 2;                    // 64
constexpr int RBN = BLK / WARPS_N / 2;                    // 32
```

Per-CTA M-tile = `WARPS_M * 2 * RBM = 256`; per-CTA N-tile = `WARPS_N * 2 * RBN = 256`.
At 8B Gate/Up RRR (M=4096, N=14336, K=4096):

| Quantity                               | Value           |
|----------------------------------------|-----------------|
| Grid                                   | 16 × 56 = 896 CTAs |
| Per-CTA B working-set (N_tile × K)     | 256 × 4096 × 1B = 1 MiB (data) + scales |
| Per-CTA LDS (Bs[2][2] @ 128×128 each) | 64 KiB             |
| K-iters per CTA                        | 4096 / 128 = 32   |

Half-N would yield a per-CTA B working-set of 512 KiB and a 32 KiB LDS Bs
allocation, doubling the grid to 1792 CTAs; quarter-N halves both again
(256 KiB B working-set, 16 KiB Bs LDS, 3584 CTAs).

The `static_assert`s at the top of the RRR fastpath
(`rrr_mxfp8_exact_8wave_fastpath.inc:216–219`) fix the legal values:

```cpp
static_assert(BLK == 256, "MXFP8 exact 8-wave fast path requires BLK=256");
static_assert(WARPS_M == 2, "MXFP8 exact 8-wave fast path requires WARPS_M=2");
static_assert(WARPS_N == 4, "MXFP8 exact 8-wave fast path requires WARPS_N=4");
```

The 4-wave variant (`rrr_mxfp8_4wave_fastpath.inc:85`) shares the
`BLK==256` constraint. There is no RRR fastpath that compiles at any
other BLK.

## 2. What an N-tile shrink would need to touch

A `BLK_N=128` RRR fastpath would require coordinated edits across at
minimum:

### 2.1 Kernel template (`rrr_mxfp8_exact_8wave_fastpath.inc`)

* Replace `BLK` in N-coordinate paths (`bc * BLK + half * HB + wn * RBN`,
  `blocks_per_col = N_DIM / BLK`, the dispatch grid `(g.n / BLK)`) with a
  new `BLK_N` constant.
* Reshape `RBN`. With WARPS_N=4 and BLK_N=128, RBN = 16. This changes the
  accumulator template `rt_fl<RBM, RBN, …>` and propagates into every
  `rcr_exact_floatx4_t regs[(RBM/16)*(RBN/16)]` (regs from 4 → 2), every
  scale-mul and store loop.
* Recompute `rrr_b_pack_count = (RBN+31)/32`. With RBN=16 this becomes 1
  (was 2). Every `b0_scale_packs[]`/`b1_scale_packs[]` length and every
  `if constexpr (rrr_b_pack_count > 1)` branch needs to be re-validated.
* Resize `Bs[2][2]` LDS allocations. Type `ST_rrr_b = st_fp8e4m3<HB, BK, …>`
  becomes `st_fp8e4m3<HB_N, BK, …>` with `HB_N = BLK_N/2 = 64`.
* Recompute the V2 scale slab math. Currently:
  `slab_bytes_b = 64 * padded_k_blocks` (= `pack_count_b=2 * 32 * padded`).
  At RBN=16 / pack_count=1 it would be `32 * padded_k_blocks`. The
  `slab_idx_b = bc * WARPS_N + wn` formula still works but bc now ranges
  over `N/BLK_N` (= 14336/128 = 112) instead of `N/BLK` (= 56), doubling
  num_slabs.
* Recompute the V2 scale per-(k_pair, lane) load offsets. Currently:
  - A side: `voff = lane_kblk*256 + lane_nonk*16`, `soff = k_pair*1024`
    (b128 fetches all 4 packs).
  - B side: `voff = lane_kblk*128 + lane_nonk*8`, `soff = k_pair*512`
    (b64 fetches both packs).
  At pack_count_b=1 the B-side becomes a b32 load with new strides:
  `voff = lane_kblk*64 + lane_nonk*4`, `soff = k_pair*256`. Any off-by-
  one or wrong shift here yields silent SNR collapse.
* Recompute the output-store coordinates
  (`rrr_mxfp8_exact_8wave_fastpath.inc:887–890`) — `bc * WARPS_N * 2 + …`
  works structurally but the wave-mapping math needs re-verification under
  the new accumulator shape.

### 2.2 Host-side preshuffle (`test_mxfp8_python.py`)

`preshuffle_scale_matrix_mfma16_v2_rcr_b` currently has parameters
`(scale_exp, blk=256, hb=128, rbn=32, warps_n=4)` and computes a layout
exactly matching the kernel's slab math. To support BLK_N=128:

* Add a parallel callable (or parameterize) for `(blk_n=128, hb_n=64,
  rbn=16, warps_n=4)`.
* Recompute `pack_count = 1` (was 2), `rbn_rg = rbn // 32 = 0` (was 1) —
  the `rg_base = wn * rbn_rg` branch breaks at rbn_rg=0; the layout
  algorithm itself needs a rewrite for sub-32 RBN.

This Python-side change must be co-developed with the kernel scale-load
math; bench harnesses (`r52q_bench.sh`, `test_mxfp8_python.py`) must be
extended to call the right preshuffle for each variant.

### 2.3 Dispatch wiring

A new `dispatch_rrr_exact_8wave_scaled_v2_blkn128<PRESHUFFLED_QUANT>`
wrapper plus a runtime gate in the `gemm_rrr_pq_v2` selection logic to
route K=4096 shapes to the new path while leaving K=8192 on the existing
BLK=256 fastpath. The agent prompt explicitly anticipates this
("recommend a per-shape gate (K≤4096 only)") — but the gate cannot be
benched without the kernel itself.

### 2.4 Build and resource validation

Each new BLK_N variant needs `-Rpass-analysis=kernel-resource-usage`
inspection to confirm:
* VGPR usage stays under 256 to keep occupancy=2 (or ideally rises to 4).
* LDS [bytes/block] decreases as expected (Bs halved at BLK_N=128).
* No new scratch traffic introduced in the K-loop body.
* SNR ≥ 45 dB and 3/3 deterministic at the target shape.

## 3. Why the existing rect scaffold doesn't help

The R28D rectangular-tile scaffold (`MXFP8_RECT_BLK_N=64`, comments at
`kernel_mxfp8_layouts.cpp:317–352`) defines `HB_N`, `BLK_N`, `RBN_RECT`
constants — but explicitly notes:

> SCAFFOLDING ONLY: when MXFP8_RECT_BLK_N=64 the V2-CRR exact-8wave
> fastpath is force-disabled (mirrors R27 Dev D MXFP8_BLK128 prototype)
> because its scale slab math + load_col_from_v2_st helpers hardcode
> HB_N=128. Real perf path requires a dedicated rect-V2 fastpath in a
> future cycle; this gate ensures clean compile + V1 fallback PASS.

The scaffold is consumed only by a CRR-side helper variant
(`load_col_from_v2_st_half_rect`). Nothing in the RRR fastpath uses
`HB_N`/`BLK_N`/`RBN_RECT`. Activating `MXFP8_RECT_BLK_N=64` does not
shrink any RRR-side N-tile; it would, at best, force RRR to fall through
to a non-fastpath kernel (which would be far slower than the V2 RRR
baseline regardless of any N-tile working-set effect).

A symmetric `MXFP8_RRR_BLK_N` scaffold + a dedicated RRR-rect fastpath
file (analogous to `crr_mxfp8_exact_8wave_rect_fastpath.inc` — which is
itself 1k+ lines of bespoke kernel code per the file list) would be the
correct precedent. That is multi-day work; out of scope for an agent
mission.

## 4. Sketch of expected outcome (sanity check on the hypothesis)

Even with the surgery completed, the hypothesis itself is shaky in light
of R52Q's refutation:

* R52Q established that A and B are *both* heavily reused inside the V2
  RRR K-loop (each K-pair loads both A0/A1 and B0/B1, and each is
  consumed 2× via paired MFMA quartets). Both sides want maximum L2
  retention; CP4 (B-side `nt`) regressed 16%.
* Halving N_tile per CTA halves the per-CTA B working-set, but doubles
  the grid. Each CTA still reads K * N_tile bytes of B from L2 across
  all K-iters; the *aggregate* B traffic across all CTAs is unchanged.
  The only L2-resident-fraction win comes if the smaller per-CTA B
  panel stays warm in L2 across the K-loop where the larger one was
  evicting itself mid-loop. R52Q's measurement that L2 hit rate is
  79.4% at baseline (essentially the same as the fast 70B Q/O cell at
  80.6%) is evidence that B is *already* mostly L2-resident; the gap
  is in TC→TA *return rate*, not in cache occupancy.
* MFMA work per CTA halves with N_tile (from `2 * RBN * RBM * BK = 2 *
  32 * 64 * 128 = 524288` MAC-ops per K-iter × 32 K-iters × 2 wave-pairs
  = …). That means CTA dispatch overhead and barrier overhead become a
  larger fraction of total runtime. R52P explicitly cautioned this
  trade-off ("Trade-off: fewer ops per CTA, more CTA dispatch
  overhead").
* Cross-cell concern: the per-shape gate (K≤4096 only) means K=8192
  cells (70B Q/O, 8B Down) keep BLK_N=256, requiring TWO RRR templates
  to be compiled and dispatched. This is a meaningful test-matrix and
  binary-size cost.

A *measured* refutation would be more authoritative than this
structural argument, but the cost-to-build is too high relative to the
expected leverage given (a) R52Q's data on actual A/B reuse and (b) the
fact that doubling the CTA count fights against the existing
`MXFP8_RRR_BLOCK_SWIZZLE` XCD-aware swizzle which is already tuned for
the 256-tile grid geometry.

## 5. What this leaves open for R53+

R52Q closed cachepolicy biasing. R52R closes per-shape N-tile shrink at
the agent budget level (REFUTED-PLUMBING — would need a multi-day
human-developer cycle). The R52P §5.3 lever is therefore *deferred,
not refuted*; if a future cycle is allocated to authoring an RRR-rect
fastpath analogous to `crr_mxfp8_exact_8wave_rect_fastpath.inc`, the
hypothesis remains testable.

The remaining R52P R53 directions:

* **§5.2 K-superblock CTA-level reuse / persistent CTA** — also
  multi-day kernel surgery, but conceptually orthogonal to N-tile.
  Reuses the *existing* per-CTA tile size, just extends a CTA's
  responsibility to multiple M-stripes against a single B-column-block.
  Could amortize each L2-fetch of B over more MFMA work without
  changing the N-tile.
* **A-side LDS double-buffer at K=4096 occ=1** — flagged in R52Q §4 as
  "may merit a measured reattempt focused on A-prefetch overlap" given
  the new PMC understanding. Still kernel-internal but not as deep as
  full N-tile re-templating.

These remain the highest-priority candidates for the next cycle.

## 6. Process notes

Per orchestrator gate in the prompt — *"If a smaller N-tile breaks
numerics … report REFUTED-PLUMBING and document what would need to
change"* — and *"If the N-tile shrink requires deep template surgery
beyond just changing a parameter (e.g., recomputing accumulator
layouts, scale-pack indexing, etc.), document the scope and decide
pragmatically — don't go down a rabbit hole."* — both gates short-
circuit the bench step. No GPU work was performed; no source patch was
authored; `HIP_VISIBLE_DEVICES=3` was not touched.

| Step | Action                                          | Outcome |
|------|--------------------------------------------------|---------|
| 1    | `pwd` worktree verification                      | OK (`.claude/worktrees/agent-ae5320f8`) |
| 2    | Read R52P / R52Q / R52N findings and RRR inc    | DONE    |
| 3    | Locate N-tile parameter (BLK, WARPS_N, RBN)      | DONE — `BLK==256` and `WARPS_N==4` `static_assert`'d |
| 4    | Plan N-tile sweep                                | DONE — half-N (BLK_N=128) is the cleanest target; all variants share the same blocker |
| 5    | Pre-bench audit per variant                      | SKIPPED — no variant compileable without kernel surgery |
| 6    | Numeric validation FIRST                         | SKIPPED — see (5) |
| 7    | Strict-SCLK A/B per-variant bench                | SKIPPED — see (5) |
| 8    | Cross-cell validation                            | SKIPPED — see (5) |
| 9    | Write findings                                   | DONE — this file |

## 7. One-line summary

**The R52P §5.3 per-shape N-tile shrink at K=4096 RRR is REFUTED-PLUMBING
at the agent budget: the V2 RRR exact 8-wave fastpath hard-codes BLK=256
and WARPS_N=4 via `static_assert`, the V2 scale slab math hardcodes
B-side pack_count=2 with byte-stride literals, and authoring an N-tile-
shrunk RRR fastpath requires coordinated changes across the kernel
template, host-side preshuffle (Python), LDS allocation, dispatch
wrapper, and output-store coordinates — total scope multi-day. The
existing R28D `MXFP8_RECT_BLK_N` scaffold is CRR-only and explicitly
force-disables the V2 fastpath, so cannot be repurposed. R53 should
pursue R52P §5.2 (K-superblock / persistent CTA) or the R52Q-flagged
A-side LDS double-buffer reattempt instead.**

## 8. Deliverables

* `r52r_findings.md` — this file.
* No source patch (none authorable in budget).
* No bench script (no variant to bench).
* No GPU activity on `HIP_VISIBLE_DEVICES=3`.
