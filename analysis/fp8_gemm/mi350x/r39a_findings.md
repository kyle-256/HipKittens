# R39 Dev A — HB-N shrink prototype on V2-RRR wide-N shapes

## Verdict: **NO SHIP — early abort per R38 time-box rule (V2-RRR tile config matches V2-CRR)**

Per the R39+ task spec ("if V2-RRR's tile config matches V2-CRR — already
bandwidth-saturated — abort early; don't waste cycles on a guaranteed
REFUTED variant"), the V2-RRR wide-N HB-N shrink was **not built or
benched**. R38 Dev A's V2-CRR refute mechanism transfers directly.

## Tile-config inspection (load-bearing finding)

The default V2-RRR exact-8wave fastpath asserts **identical** tile geometry
to V2-CRR (`analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc:18-19`):

```cpp
static_assert(BLK == 256, "MXFP8 exact 8-wave fast path requires BLK=256");
static_assert(BK == 128,  "MXFP8 exact 8-wave fast path requires BK=128");
static_assert(WARPS_M == 2, "...WARPS_M=2");
static_assert(WARPS_N == 4, "...WARPS_N=4");
```

`kernel_mxfp8_layouts.cpp:333-339` defines, for ALL Layout::* fastpaths:
* `BLK = GEMM_BLOCK_SIZE = 256`
* `HB  = BLK / 2 = 128`
* `WARPS_M = 2`, `WARPS_N = 4`
* `RBM = BLK / WARPS_M / 2 = 64`
* `RBN = BLK / WARPS_N / 2 = 32`

Per-warp accumulator footprint (`rrr_mxfp8_exact_8wave_fastpath.inc:43`):
```cpp
rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;  // 4 tiles × 64×32 = same as V2-CRR
```

**Conclusion**: the only difference between V2-RRR and V2-CRR fastpaths is the
A-fetch layout (row-shared LDS in RRR vs col-shared LDS in CRR, with
`load_transpose` for B in `RRR_ROW_SHARED_TRANSPOSE`). The N-direction
partition (4 warps × RBN=32 = 128 per HB-half) is **byte-for-byte identical**.

## Why the V2-CRR refute mechanism transfers

R38 Dev A established that HB-N shrink on wide-N V2-CRR fails because:
1. Default V2-CRR is bandwidth-bound on wide-N (B-operand throughput),
   not VGPR-bound — `WARPS_N=4 + RBN=32` already partitions N tightly.
2. Halving N (`BLK_N=128, HB_N=64`) doubles the WG grid in N → 2× barriers,
   2× scale-fetch overhead, 2× CR reduction across smaller per-WG tile.
3. The accumulator drop (cB, cD → -64 VGPR/warp) does not relieve the
   binding constraint.
4. Even with PIPE=1 cross-buffer DB recovery (1594 TF on contended GPU0
   vs 1260 PIPE=0), still well below 2400 TF baseline.
5. Net: -43% to -45% across both wide-N shapes (8B + 70B Gate/Up).

V2-RRR's `WARPS_N=4 + RBN=32` is identical → the N-partition is equally
tight; the same WG-grid-doubling overhead applies; the same accumulator
"saving" is equally non-load-bearing. The B-operand bandwidth ceiling
that bounds V2-CRR also bounds V2-RRR (B is fetched the same way modulo
the row-shared-transpose layout, which doesn't change bytes/SM/cycle).

## What V2-RRR's win over V2-CRR actually buys (R34 Dev B context)

R34 Dev B 4-GPU triangulated V2-RRR > V2-CRR by +5–6% on 8B Gate/Up
4096×14336×4096 (STRICT SHIP for c5, SHIP-LITE for c6). The win comes
from the **A-fetch layout** (row-shared B with transpose vs col-shared B),
not from a different tile geometry. So the R34 Dev B finding does NOT
imply V2-RRR has bandwidth headroom that HB-N shrink could exploit; it
implies V2-RRR is even **closer** to its B-bandwidth ceiling at the same
tile geometry.

If anything, V2-RRR's higher absolute TFLOPS (~2538 TF on c5 vs CRR's
~2415) means V2-RRR is *more* bandwidth-bound on wide-N than V2-CRR, so
HB-N shrink — which trades VGPR for WG-grid overhead — would produce an
even worse delta.

## Time-box rationale (R38 explicit rule applied)

The R39 task spec said:
> "Time-box: if V2-RRR's tile config matches V2-CRR (already bandwidth-saturated),
> abort early — don't waste cycles on a guaranteed REFUTED variant."

V2-RRR's tile config matches V2-CRR exactly. Building the scaffold and
benching would consume ~1 GPU-hour to reproduce a guaranteed -40%-class
regression. Aborting honors the time-box and frees R39 cycles for
higher-EV candidates (priority items 1, 2, 4, ...).

## What would be different (and might warrant a future cycle)

The HB-N shrink hypothesis would only have a chance to win on a kernel
that has WARPS_N=2 + RBN=64 (large N-direction accumulator under-partitioned
across few warps), mirroring HB-M's win on `WARPS_M=2 + RBM=64` tall-thin.
Neither V2-CRR nor V2-RRR (nor V2-RCR) has that geometry today; all use
WARPS_N=4 + RBN=32.

A potential R40+ candidate: design a NEW V2-* variant with WARPS_N=2 + RBN=64
(symmetric to R35 Dev C's WARPS_M=4 V2-CRR variant), then test HB-N shrink
ON THAT VARIANT. This is a 2-cycle compound bet — first prove the
asymmetric tiling is competitive at a wide-N shape, then layer HB-N shrink.
Both legs are speculative; not recommended for R39.

## Files

* No new kernel `.inc` written.
* No changes to `kernel_mxfp8_layouts.cpp`.
* This findings doc (`r39a_findings.md`) records the inspection + abort
  rationale.

## Followups / R40+ candidates

1. **Do NOT build V2-RRR HB-N shrink** under the current tile geometry.
   The R38 Dev A V2-CRR refute mechanism transfers directly.
2. **WARPS_N=2 V2-* variant** (compound bet). Speculative — would need
   to first prove a `WARPS_N=2 + RBN=64` V2-CRR/RRR is competitive at
   wide-N (R35 Dev C's WARPS_M=4 prototype was negative on its target
   shape, suggesting under-partitioned tiles are not generically winning).
3. **B-bandwidth-side optimization** for wide-N rect MXFP8 — e.g.,
   B-side LDS prefetch, B-operand cache-policy bits, B re-laid-out
   format (RCR-style per-tile B). This targets the binding constraint
   directly. Would need a separate scaffold; not a tile-geometry knob.
4. **Generalize the time-box rule**: when a paradigm (HB-* shrink) is
   refuted on one variant, before mirroring to a sibling variant,
   inspect the load-bearing tile-config invariants. If they match,
   abort early. R38 Dev A's findings doc → R38 cycle wrap → R39 task
   spec → this abort: the chain worked. Lock this into the methodology.

## Conclusion

V2-RRR's exact-8wave fastpath shares all load-bearing tile-geometry
invariants with V2-CRR (BLK=256, HB=128, WARPS_M=2, WARPS_N=4, RBM=64,
RBN=32, accumulator footprint cA/cB/cC/cD=64×32×4). The HB-N shrink
mechanism that failed on V2-CRR (R38 Dev A: -43% to -45%) will fail
identically on V2-RRR. Per the R38 time-box rule, R39 Dev A aborts
early without building the V2-RRR HB-N scaffold or burning bench
cycles.

**SHIP/NO-SHIP verdict: NO SHIP (early abort, hypothesis pre-refuted by
tile-config equivalence).**
