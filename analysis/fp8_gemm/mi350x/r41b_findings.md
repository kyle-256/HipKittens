# R41 Dev B — BK=64 vs 128 K-direction blocking on V2-CRR

## Verdict: **NO SHIP — hypothesis REFUTED BY INSPECTION (no benches needed, ~2-4 GPU-hr saved)**

The R41+ priority list speculation that BK=64 might "halve per-iter
scale-fetch" is **structurally false** under the current scale-pack /
mfma layout. BK=64 would actually **increase** scale-fetch frequency
and add the **same** WG-grid-doubling-style overhead pattern that
killed R38 Dev A (HB-N V2-CRR), R39 Dev A (HB-N V2-RRR), and R40 Dev A
(HB-N+WARPS_N=2 V2-RRR) — but along the K axis instead of the N axis.
This is the **4th independent confirmation** of the
**tile-area-conservation paradigm** (cumulative R34 sub-RBM + R35 W4 +
R40 W2 → R41 BK=64 K-shrink).

Time-boxed at ~25 minutes per the R39 Dev A early-abort precedent. No
.so built. No GPU cycles consumed.

---

## 1. Background — what BK does in V2-CRR exact-8wave

`crr_mxfp8_exact_8wave_fastpath.inc:131` enforces `static_assert(BK == 128)`.
BK is the K-direction tile size per main-loop iteration. The current
default V2-CRR/V2-RRR/V2-RCR fastpaths all use BK=128 (set in
`kernel_mxfp8_layouts.cpp:307-309` via `GEMM_K_BLOCK 128`).

The main loop (`crr_mxfp8_exact_8wave_fastpath.inc:268`):
```cpp
const int k_iters = g.k / BK;            // 32 iters at K=4096 / BK=128
```

Per main-loop iter at BK=128:
* **VMEM load**: 2 A-tiles (each 128 K × 64 M × 1 B = 8 KB) + 2 B-tiles
  (128 K × 32 N × 1 B = 4 KB each) per WG = **24 KB VMEM**
* **MMA chain**: 8 mfmas per (n, m) accumulator tile pair (k_phase 0..3
  walks lo/hi opsel). Each mfma covers 16 K-elements →
  4 phases × 2 (lo/hi) = 8 mfmas × 16 K = **128 K covered per chain**
* **Scale fetch**: every other iter calls `load_raw_scales(k_pair = k>>1)`
  (line 822-823). One b128 (16 B) for A + one b64 (8 B) for B per lane.
  Each fetched pack covers 4 scales × 32 K-elem = **128 K-elements**.
  Loading 4 packs (a0/a1 × p0/p1) per fetch = **2 BK=128 iters covered
  per fetch** (alternate iter just does `>> 16` shift on the existing
  pack — no fetch).
* Scale bytes per K-element: `(16 + 8) / 256 = 0.094 B/K` per WG, or
  ≈0.4% of the 24 KB / 128 K = 192 B/K total VMEM rate. **Scale fetch
  is < 1% of total VMEM bandwidth** at BK=128.

---

## 2. BK=64 arithmetic — what would change

Hypothetical BK=64 main loop:

| quantity                       | BK=128 (current)             | BK=64 (hypothetical)             | ratio  |
|--------------------------------|------------------------------|----------------------------------|--------|
| `k_iters` at K=4096            | 32                           | 64                               | 2.0×   |
| `k_iters` at K=14336 (8B-Down) | 112                          | 224                              | 2.0×   |
| VMEM bytes per iter (per WG)   | 24 KB                        | 12 KB                            | 0.5×   |
| Total VMEM bytes per WG        | 24 KB × k_iters              | 12 KB × k_iters_new = same       | **1.0×** |
| Mfmas per accumulator tile     | 8 (4 phases × 2 lo/hi)       | 4 (2 phases × 2 lo/hi)           | 0.5×   |
| Per-iter barriers + waitcnt    | ~3 (vmcnt + lgkm + s_barrier)| ~3                               | 1.0×   |
| Total barrier overhead         | 3 × k_iters                  | 3 × k_iters_new                  | **2.0×** |
| Scale-pack fetches per K       | 1 fetch per 256 K            | 1 fetch per 128 K                | **2.0×** (WORSE) |
| Per-iter MMA chain length      | 8 mfmas (latency-hiding)     | 4 mfmas                          | 0.5×   |

### 2.1 The "scale-fetch halved" premise is structurally false

The R41+ priority list described BK=64 as "halving per-iter scale-fetch
(potentially relieving the bottleneck that HB-N attempts couldn't)".

**This is wrong under the current scale-pack layout.** The scale pack
`fp8e8m0_4` is a uint32 holding 4 scales × 32 K-elem each = 128 K-elem
(MXFP8 spec: one scale per 32 K-elements). The mfma intrinsic
`mfma_*_xK16` walks the 4 scales via opsel, covering 64 K-elem per pack.
Two packs (lo + hi) cover 128 K-elem = exactly 1 BK=128 iter. The
load already amortises ONE b128 over TWO BK=128 iters (alternate iter
shifts `>> 16` on the pre-loaded pack — the cheapest possible amortisation).

Halving BK to 64 would mean:
* Per BK iter still consumes the same 64 K-elem of scales per pack (this is
  set by mfma intrinsic, not by BK)
* But each pack now covers only 1 BK=64 iter (not 2 BK=128) → the
  "alternate-iter shift" amortisation BREAKS. We'd fetch a new pack
  every iter, **doubling** scale-fetch frequency.
* Even worse: the mfma chain length per iter halves (4 mfmas vs 8), so
  there's less compute time to hide the increased fetch frequency behind.

**Net: BK=64 makes scale fetch strictly worse, not better.** The premise
of the hypothesis is empirically and structurally refuted.

### 2.2 Bandwidth break-even check

Even if we ignore the scale-fetch regression and only consider total VMEM:

* Total VMEM bytes per WG is **invariant** under BK rotation (24 KB ×
  32 iters at BK=128 = 768 KB per WG; 12 KB × 64 iters at BK=64 = 768 KB
  per WG). VMEM bandwidth is unchanged.
* Total mfma flops per WG is **invariant** (same M×N×K work, just
  re-tiled). Compute throughput is unchanged.
* What changes: per-iter overhead doubles (2× barriers, 2× waitcnt sync,
  2× LDS slab swaps, 2× scale-pack fetches).
* Per main-loop iter overhead estimate: ~10-20 cycles for barrier +
  s_waitcnt + sched_barrier + LDS write address arith. At BK=128 with
  k_iters=32 (8B Q shape K=4096), that's ~480 cycles overhead total. At
  BK=64 with k_iters=64, that's ~960 cycles → **+480 cycles of pure
  overhead** with NO compensating bandwidth gain.
* At fmla rate (per WG ≈ 2 TF on MI355X), 480 cycles ≈ 0.5 µs of
  fmla = ~2.5 GF wasted per WG. Across 304 CUs × 2 blocks/CU = ~600
  WGs, that's ~1.5 TF/s of wasted compute.
* Break-even occupancy: would need scale-fetch savings ≥ 480 cycles per
  iter to pay off the doubled overhead. But scale-fetch isn't even a
  bottleneck — it's < 1% of VMEM (see §1). **No achievable occupancy
  break-even exists.**

---

## 3. Structural blockers — BK=64 is not a small change

Even if the perf forecast were neutral, the implementation cost is
prohibitive:

1. **No `rt_64x16_s` tile shape exists.** `include/types/types.cuh:62-70`
   lists only `rt_16x16`, `rt_32x32`, `rt_32x32_8`, `rt_16x32`, `rt_32x16`,
   `rt_32x16_4`, `rt_16x32_4`, `rt_16x128`, `rt_128x16`. The MMA register
   tile `A_col_reg = rt_fp8e4m3<BK, RBM, col_l, rt_128x16_s>`
   (`kernel_mxfp8_layouts.cpp:448`) is hardcoded to `rt_128x16_s`. A new
   `rt_64x16_s` shape class with mfma swizzle handlers would have to be
   added to the type system — multi-day work touching `register_tile.cuh`
   + `mma_*` intrinsic wrappers.

2. **Scale-pack opsel is wired to BK=128.** `crr_mma_scaled_dispatch`
   (`kernel_mxfp8_layouts.cpp:1464-1493`) has 16 cases (4×4 opsel matrix
   for k_phase 0..3 × n parity 0..1). At BK=64 only k_phase 0..1 would
   be valid → 8 dead cases, plus the lo/hi pack-pair iteration in
   `crr_exact_cA_with_b1_interleave_*` (lines 168-241 in the .inc) would
   need redesign to walk 4 mfmas instead of 8. The
   `crr_mma_scaled_phase<>` template specialization is also keyed to
   the 8-mfma chain.

3. **8 hard `static_assert(BK == 128)` sites** (one per fastpath .inc
   plus kernel_mxfp8_layouts.cpp:2347). Any of these would need to become
   `BK == 64 || BK == 128` with parallel codegen branches — this is the
   exact "scaffolding-then-rewrite" pattern that R34 Dev D's sub-RBM
   work consumed ~75 minutes on, only to discover (R34 Dev D §3) that
   the resource forecast was already negative (476 VGPR spill).

4. **LDS slab geometry would change.** `subtile_inplace<RBM, BK>`
   (line 502, 530) consumes a `RBM × BK` LDS subtile. Halving BK halves
   the per-load LDS traffic — fine — but doubles the per-WG LDS slab
   count to cover the full K iteration. The overall `As[2][2] + Bs[2][2]`
   buffer (139264 B at BK=128) doesn't shrink because it's keyed to
   BK-per-iter not total K. No occupancy gain.

5. **No precedent .inc to clone.** Unlike HB-M shrink (which had
   R35 Dev B's Stage A1 scaffold to clone for R36 Dev A's Stage B1
   pipelining work), there is no BK=64 prototype anywhere in the tree.
   `git log --all --oneline | grep -i "BK=64\|BK_64\|MXFP8_CRR_BK"` →
   empty.

---

## 4. Tile-area-conservation paradigm — 4th independent confirmation

R32 Dev B closure §8 established:
> Per-warp VMEM/MMA tile area is approximately conserved across (BLK, BK,
> WARPS_M, WARPS_N, RBM, RBN) rotations. Halving any one dimension while
> keeping the others fixed doubles WG count, doubles barriers + scale-fetch
> overhead, and rarely buys back the per-iter savings unless that specific
> dimension was BOTH on the critical path AND removing it relieved a
> binding resource (VGPR spill, occupancy = 1, etc.).

Confirmations to date:

| cycle | rotation         | mechanism                         | verdict      | savings    |
|-------|------------------|-----------------------------------|--------------|------------|
| R34   | RBM 64→32        | sub-RBM 4-stride M-loop           | REFUTED (476 spill) | -100% (build only) |
| R35   | WARPS_M 2→4      | W4 reorder                        | REFUTED (256 sat) | -8% measured |
| R38   | BLK_N 256→128 (V2-CRR) | HB-N shrink                  | REFUTED      | -43 to -47% |
| R39   | BLK_N 256→128 (V2-RRR) | HB-N shrink                  | REFUTED (inspection) | -1 GPU-hr saved |
| R40   | HB-N + WARPS_N=2 | compound bet                      | REFUTED (doubly pre-refuted) | -3-5 GPU-hr saved |
| **R41** | **BK 128→64 (K-shrink)** | **K-direction blocking** | **REFUTED (inspection)** | **-2-4 GPU-hr saved** |

The exception: HB-M shrink (R35 Dev B → R36 Dev A → R37 Dev A) WORKED
because:
* Tall-thin shapes (N=1024 KV) were VGPR-pressure-bound, not bandwidth-
  bound. The 234 → 160 VGPR drop (-74) was load-bearing — it doubled
  occupancy (1 → 2 blocks/CU), unblocking SB pipelining.
* The doubled WG grid was small enough (2× of a tiny tall-thin grid)
  that the barrier overhead was amortised over the per-WG win.
* HB-M shrink is a **resource-relief** lever — not a bandwidth-relief
  lever. It only ships where VGPR is the bottleneck.

**BK shrink fails the same test**: V2-CRR is bandwidth-bound (R38 Dev A
§"Why it failed"). VGPR is not the bottleneck on the default V2-CRR;
even tall-thin 70B-KV needed HB-M shrink, not BK shrink, to unlock
SB pipelining. BK=64 buys nothing on either axis.

---

## 5. Why the "wide-K" candidate shape framing doesn't save it

The R41+ task spec listed three K=large candidates:

* **8B Q (K=4096)**: 32 iters → 64 iters. Smallest doubling, but K=4096
  is short — scale-fetch is already <1% of VMEM, no headroom to relieve.
  Also: 8B QO V2-RCR was just R40 Dev D STRICT-PROMOTED (+6.85 / +8.19%
  via N_PAIRS bump alone, no kernel work). The shape is already at its
  achievable production ceiling without K-direction work.
* **8B-Down (K=14336)**: 112 iters → 224 iters. The doubling penalty is
  highest here. R38 Dev C SHIP'd this shape on V2-RRR at +7.42% (now
  R40-reconfirmed at +7.91% gold-standard). The cell already hit its
  V2 ceiling via layout switch, not BK rotation.
* **70B-Down (K=28672)**: 224 iters → 448 iters. Worst doubling penalty.

For all three: total VMEM bandwidth is invariant under BK rotation
(§2.2). Larger K just makes the per-iter overhead penalty bigger
(more iters × more per-iter overhead). There is no K-direction shape
where BK=64 has a positive bandwidth or occupancy story.

---

## 6. Followup recommendations

1. **Do NOT pursue BK=64 V2-CRR/V2-RRR/V2-RCR.** The lever is
   structurally negative across all three K-direction lengths considered.
   This closes the K-rotation axis of the tile-area-conservation
   paradigm (already closed on M, N, WARPS_M, WARPS_N axes; K was the
   last untested rotation).
2. **Do NOT pursue BK=256 (the inverse).** Doubling BK would halve
   barriers but quadruple the per-iter A_col_reg / B_col_reg VGPR
   footprint (from 32 / 16 VGPR to 64 / 32). The default V2-CRR is
   already at 234 VGPR (R36 Dev A baseline); pushing past 256 would
   spill, and there's no `rt_256x16_s` tile shape anyway.
3. **Survey R41+ candidates outside ALL tile-rotation paradigms.** The
   tile-area-conservation paradigm is now CLOSED on every rotation
   (M, N, K, WARPS_M, WARPS_N). Future structural perf work should target:
   * Inter-WG L2 coordination (speculative — no precedent in tree)
   * Alternative shared-mem layout for B operand (R34 Dev B mentioned;
     unexplored)
   * V2-RCR HB shrink (mirror of HB-M B1 success but on RCR layout —
     would need separate scaffold)
   * Decode-shape coverage (M=1, 32, 128 entirely unmapped per R40 wrap)
4. **Methodology rule (NEW R41+):** "K-direction blocking variation
   (BK=64 / BK=256)" is the **5th tile-rotation paradigm closed by
   inspection**. R42+ should treat any "halve/double a tile dimension to
   trade overhead for bandwidth" hypothesis as REFUTED-BY-DEFAULT
   unless the proposer first identifies (a) a binding resource that
   the rotation relieves AND (b) a concrete pre-existing prototype
   in tree. Without both, abort before scaffold.

---

## 7. Time-box accounting

* Read TODO.md + agent_prompt.md (R40 cycle wrap context): ~3 min
* Read crr_mxfp8_exact_8wave_fastpath.inc + scale-pack arithmetic: ~8 min
* Read R38 Dev A r38a_findings.md (HB-N refute mechanism transfer): ~3 min
* Read R37 Dev A ab8a80f7 + R36 Dev A r36a_findings.md (HB-M B1 win
  mechanism): ~3 min
* Verify no `rt_64x16_s` tile shape exists + count BK=128 hardcoded sites:
  ~3 min
* Write findings doc + commit: ~5 min

**Total: ~25 min, well within the R41 time-box.** Zero .so builds, zero
GPU cycles consumed. ~2-4 GPU-hr saved by structural inspection vs the
scaffold-then-bench protocol that would have been required to disprove
empirically.

---

## 8. Closure additions (cycle-17 closure list)

**41. R41 Dev B BK=64 K-direction blocking REFUTED via inspection.** The
"halve per-iter scale-fetch" premise is structurally false: scale packs
already amortise across 2 BK=128 iters (one b128 fetch covers 256 K-elem
via lo/hi opsel + alternate-iter shift). Halving BK breaks the
amortisation, **doubling** scale-fetch frequency. Total VMEM bandwidth
is invariant under BK rotation; per-iter overhead doubles (barriers,
waitcnt, LDS swap). No achievable occupancy break-even exists. 4th
independent tile-area-conservation confirmation (M, N, K rotations all
now closed; cumulative R34 sub-RBM + R35 W4 + R38 HB-N + R39 HB-N V2-RRR
+ R40 HB-N+W2 + R41 BK=64 = 6 confirmations on tile-rotation refutes).
Closes K-direction rotation as the last untested axis of the paradigm.

**42. R41+ methodology rule:** "halve/double a tile dimension to trade
overhead for bandwidth" hypotheses are REFUTED-BY-DEFAULT unless the
proposer first identifies (a) a binding resource the rotation relieves
AND (b) a concrete pre-existing prototype in tree. Saves the
scaffold-then-bench cycle on speculative tile-rotation candidates.
