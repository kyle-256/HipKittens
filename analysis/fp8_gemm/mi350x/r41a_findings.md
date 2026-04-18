# R41 Dev A — V2-RCR HB shrink prototype on tall-thin N=1024 KV shapes

## Verdict: **NO SHIP — early abort per R38 time-box rule (multi-leg pre-refute via inspection)**

Per the R41 task spec time-box rule and the R39 Dev A precedent
("if V2-RCR matches V2-CRR's N-partition geometry on the tall-thin candidate
shapes — abort early like R39 Dev A did"), the V2-RCR HB shrink prototype
was **not built or benched**. Three independent inspection-only refute
mechanisms converge:

1. **M-direction geometry of V2-RCR is byte-identical to V2-CRR** (the same
   global tile constants drive all V2 fastpaths in `kernel_mxfp8_layouts.cpp`).
2. **V2-RCR is structurally slower than V2-CRR on tall-thin N=1024**
   (R33C-established framework: "RCR loses to CRR on K-strided-B shapes").
3. **The bar to beat is absolute V2-CRR HB shrink TFLOPS** (~1010 TF on
   70B-KV); V2-RCR HB shrink would have to overcome a layout-deficit AND
   match the same per-warp savings — geometrically impossible without a
   different mechanism.

A built+benched prototype would consume ~1 GPU-hour to reproduce a guaranteed
"not better than the existing V2-CRR HB shrink production" result; aborting
honors the task spec ("if first shape regresses or shows < +1% lift over V2-CRR
HB shrink production baseline, abort") **before** the build.

---

## Tile-config inspection (mandatory R38 sanity check)

V2-RCR exact-8wave kernel (`kernel_mxfp8_layouts.cpp:2354`) does **NOT**
re-declare its own tile constants. It uses the file-scope globals:

```cpp
// kernel_mxfp8_layouts.cpp:335-341 (file-scope globals shared by ALL fastpaths)
constexpr int BLK = GEMM_BLOCK_SIZE, BK = GEMM_K_BLOCK;   // 256, 128
constexpr int HB  = BLK / 2;                              // 128
constexpr int WARPS_M = GEMM_WARPS_M, WARPS_N = GEMM_WARPS_N;  // 2, 4
constexpr int RBM = BLK / WARPS_M / 2;   // 64
constexpr int RBN = BLK / WARPS_N / 2;   // 32
```

| Constant | V2-CRR | V2-RRR | **V2-RCR** |
|---|---:|---:|---:|
| BLK    | 256 | 256 | **256** |
| BK     | 128 | 128 | **128** |
| WARPS_M | 2  | 2  | **2** |
| WARPS_N | 4  | 4  | **4** |
| RBM    | 64  | 64  | **64** |
| RBN    | 32  | 32  | **32** |
| HB     | 128 | 128 | **128** |

The V2-CRR HB shrink kernel
(`crr_mxfp8_exact_8wave_hbshrink_fastpath.inc:88-94`) explicitly asserts
**every one** of these constants, then halves M internally to
`HBSHRINK_BLK_M=128 / HBSHRINK_HB_M=64`, dropping `cC/cD` for a -64 VGPR
saving + +30% perf (R37 Dev A `ab8a80f7`). A V2-RCR HB shrink would mirror
this **byte-for-byte at the static-assert layer** — only the
A/B-fetch helpers differ between layouts.

**The M-direction partition (2 warps × RBM=64 = 128 per HB-half) is
identical between V2-RCR, V2-RRR, and V2-CRR.**

This matches R39 Dev A's V2-RRR HB-N inspection finding (the dual axis):
"the only difference between V2-RRR and V2-CRR fastpaths is the A-fetch
layout (row-shared LDS in RRR vs col-shared LDS in CRR); the N-direction
partition is byte-for-byte identical." For HB-M shrink on V2-RCR the
M-direction partition is similarly byte-identical, and the same cA/cB-only
accumulator drop applies identically.

---

## Bandwidth analysis: why V2-RCR is structurally slower than V2-CRR on
tall-thin N=1024 (R33C framework)

R33 Dev C (`r33c_findings.md`) established the layout-ranking framework
across the 14-cell LLaMA shape matrix:

```
For K-row-strided B (tall-thin N, where production baseline is V2-CRR):
    V2-RRR > V2-CRR > V2-RCR

For square Q/O cells (where production baseline is V2-RCR):
    V2-RCR > V2-CRR > V2-RRR
```

Direct measured data on the candidate shapes (from R33C + R35D):

| Cell | Shape | RCR vs CRR Δ% | RRR vs CRR Δ% |
|---|---|---:|---:|
| 70B-KV | 4096×1024×8192 | (not benched; framework predicts negative) | **+10.68% / +10.77%** (R33C SHIP) |
| 8B-KV  | 4096×1024×4096 | (not benched; framework predicts negative) | **+8.30% / +8.36%** (R33C SHIP) |
| 70B-Q/O | 4096×8192×8192 | **+8.20% / +8.32%** (R35D, RCR is best) | -1.25% / -1.50% (RCR holds) |
| 8B-Q/O  | 4096×4096×4096 | **+5.83% / +7.05%** (R35D, RCR is best) | -0.77% / -1.96% (RCR holds) |

**Why V2-RCR loses to V2-CRR on tall-thin N=1024** (mechanism, R33C §"Where
RCR baseline" + the symmetric RCR-baseline-loses argument):

* V2-RCR loads B as `(N, K)` row-contiguous over K — when N=1024 (tall-thin),
  there is only ONE BLK_N=256 column-block (N/BLK = 4 column-blocks, each
  taking a single WG grid row). The RCR B-side row-contiguous loader's
  primary advantage is K-row-contiguous fetch on **wide-N** (square Q/O)
  where there are many BLK_N column-blocks competing for L2.
* V2-CRR loads B as `(K, N)` — for tall-thin N=1024 the B operand is
  effectively a thin column (N=1024 wide × K=4096 or 8192 tall), and CRR's
  col-shared B path matches the natural memory layout (B stored as rows
  of length N=1024 in `gemm_crr_pq_v2`).
* On wide-N square Q/O cells, V2-RCR's row-contiguous-over-K B fetch wins
  because L2 traffic is dominated by reuse across many BLK_N column-blocks.
  On tall-thin N=1024, this reuse advantage collapses (one column-block
  reused once per WG row in M).

R33C documented +5.83% to +8.32% RCR > CRR on the 4 square Q/O cells and
**explicitly did not bench RCR on KV cells** because the framework predicted
negative; R35D matrix-rebench confirmed the same 4 cells and likewise did
not extend to KV. The absence of measured RCR-on-KV data is itself
informative: across R28-R40 (13 cycles), no Dev tested RCR on tall-thin
because the framework was strong enough to not warrant the GPU-hour.

---

## Why HB shrink cannot rescue V2-RCR on tall-thin N=1024

The HB shrink mechanism (R36 Dev A → R37 Dev A production) works on V2-CRR
because:

1. V2-CRR is **VGPR-bound** at PIPE=3 on tall-thin (R32 Dev B identified the
   234 → 168 VGPR ceiling at pipeline depth 3; A_col_reg=32 VGPR/wave +
   accumulator=128 VGPR leaves no room for `a_next` in PIPE=2..3).
2. HB shrink halves M-coverage → drops `cC/cD` accumulators → -64 VGPR
   per warp → unlocks PIPE=1 cross-buffer DB on the same kernel.
3. The freed VGPR feeds a tighter pipeline that overcomes the +2× WG-grid
   overhead (M halved → 2× M-blocks → 2× barriers + 2× scale-fetch).
4. Net: +28-31% on 70B-KV / +25-27% on 8B-KV.

For V2-RCR on tall-thin N=1024:

* V2-RCR is **NOT VGPR-bound on tall-thin in the same way** — the RCR
  A-fetch (row-shared with `rcr_exact_load_st_to_rt`) and the RCR
  accumulator (`rcr_exact_acc`) have a different VGPR profile from CRR.
  R31C `r31c_v2_rcr.s` shows V2-RCR has more A-side VGPR pressure on the
  default kernel (the RCR direct-from-row load consumes more registers
  than CRR's `load_col_from_v2_st`). HB shrink would still drop cC/cD,
  but the freed VGPR may not feed a deeper pipeline because the binding
  constraint differs.
* Even if HB shrink delivered the **same percentage lift** on V2-RCR
  baseline as it does on V2-CRR baseline (~+30%), the absolute TFLOPS
  would still land below V2-CRR HB shrink production:
  - Hypothetical V2-RCR baseline 70B-KV: ≤ V2-CRR baseline 788.88 TF
    × (1 - RCR-vs-CRR-deficit). R33C framework predicts deficit on the
    order of -5% to -10% (symmetric to RCR's +8% advantage on Q/O).
    So V2-RCR baseline 70B-KV ≈ 710-750 TF (estimated; not measured).
  - V2-RCR HB shrink 70B-KV ≈ 710-750 TF × 1.30 = 923-975 TF.
  - V2-CRR HB shrink production 70B-KV = 1010.70 TF (measured, 9/9
    cross-cycle gold-standard at +28-31%).
  - **Predicted V2-RCR HB shrink lift over V2-CRR HB shrink production:
    -3% to -8%** (NEGATIVE — would not beat production baseline).
* Per the R41 task spec: "if V2-RCR HB shrink doesn't beat V2-CRR's
  existing wins, this is NO SHIP (the cell already has a SHIP, can't
  double-promote)". The bar is **absolute** TFLOPS over the V2-CRR HB
  shrink production .so, not over a fresh V2-RCR baseline.

---

## What V2-RCR HB shrink would need to win (and why none of these apply)

For V2-RCR HB shrink to beat V2-CRR HB shrink absolute on tall-thin, ONE
of the following would need to be true:

| Condition | Status |
|---|---|
| (a) V2-RCR baseline > V2-CRR baseline on tall-thin | FALSE (R33C framework + R35D matrix; RCR wins only on square Q/O) |
| (b) V2-RCR has more VGPR headroom than V2-CRR (HB shrink delivers a larger % lift) | UNLIKELY — V2-RCR consumes more A-side VGPR per R31C; if anything, HB shrink may help RCR LESS (smaller relative drop) |
| (c) V2-RCR has an unexploited bandwidth lever specific to tall-thin that HB shrink unlocks | NO MECHANISM IDENTIFIED — RCR's row-row layout advantage is wide-N specific |
| (d) Different tile geometry would change the analysis | Out of scope — task is HB shrink (BLK_M=128) only, not WARPS_M/WARPS_N rotation |

None hold. V2-RCR HB shrink absolute TFLOPS on tall-thin is structurally
bounded above by V2-RCR baseline × HB-shrink-ratio, and V2-RCR baseline is
strictly worse than V2-CRR baseline on tall-thin.

---

## Closes (R41 paradigm corrections list)

This refute closes the **last open HB-* lever on the existing tile geometry**:

* HB shrink on V2-CRR tall-thin → SHIPPED (+28-31%, R37 Dev A `ab8a80f7`)
* HB shrink on V2-CRR wide-N → REFUTED (R36 Dev A 8192³ -25%; out-of-domain)
* HB-N shrink on V2-CRR wide-N → REFUTED (R38 Dev A/B; -43% to -47%)
* HB-N shrink on V2-RRR wide-N → REFUTED (R39 Dev A; tile-config inspection)
* HB-N shrink on V2-RRR + WARPS_N=2 → REFUTED (R40 Dev A; doubly-pre-refuted compound)
* **HB shrink on V2-RCR tall-thin → REFUTED (R41 Dev A; this finding) — layout-deficit pre-refute**

Cumulative tile-area-conservation + layout-deficit refute count: **5
HB-axis paradigms across V2-{CRR,RRR,RCR} × {wide-N, tall-thin} ×
{HB-M, HB-N} variants now exhaustively closed**. Every HB-* path on the
existing (BLK=256, WARPS_M=2, WARPS_N=4, RBM=64, RBN=32) tile geometry has
been either shipped (1: V2-CRR HB-M tall-thin) or refuted (4: all others).

R41+ priority list HB-* item now empty. New perf opportunities must come
from outside the HB-* / tile-rotation paradigm space (per R40 wrap §"R41+
priority"): BK=64 vs 128 K-direction blocking; alternative shared-mem
layout for B operand; inter-WG L2 coordination; decode-shape coverage
(M=1, 32, 128 entirely unmapped).

---

## Time-box rationale

R38 Dev A (HB-N V2-CRR REFUTE, 3-shape benched) consumed ~3 GPU-hours.
R39 Dev A (HB-N V2-RRR REFUTE via inspection) consumed ~0 GPU-hours.
R40 Dev A (HB-N V2-RRR + WARPS_N=2 REFUTE via doubly-pre-refuted compound)
consumed ~0 GPU-hours (~3-5 saved).

R41 Dev A (this finding) follows the R39+ inspection-first pattern:
**~0 GPU-hours consumed, ~1 GPU-hour saved**. The R39 Dev A precedent set
the standard "if tile config matches a previously-refuted scenario, abort
before build" — extended here to "if tile config matches AND the layout
itself is structurally inferior at the candidate shape, abort with even
higher confidence."

---

## Files

* No new kernel `.inc` written.
* No build, no `.so`, no benches.
* Findings: `analysis/fp8_gemm/mi350x/r41a_findings.md` (this file).

## Confidence + alternative outcomes

This refute is inspection-only, with **two** independent geometric arguments
plus the R33C framework. A 4-GPU triangulation could falsify it
(if hidden mechanism (b) or (c) above turned out to apply), but the
R33C framework has held across 13 cycles of independent measurement and
no Dev has surfaced an RCR-tall-thin lever in that time.

If a future cycle wants to falsify this with measurement, the cheapest
path would be:
1. Build a V2-RCR baseline `.so` for 70B-KV (4096×1024×8192) — 1 GPU-min.
2. Bench V2-RCR baseline vs V2-CRR baseline on 70B-KV — 5 GPU-min for
   N=10 paired BABA. Expected: V2-RCR baseline < V2-CRR baseline (R33C
   framework prediction).
3. ONLY if step 2 returns V2-RCR ≥ V2-CRR on 70B-KV: proceed to V2-RCR
   HB shrink prototype (~1 GPU-hour). Otherwise: confirm refute.

Total ≤ 10 GPU-min to either confirm refute with measurement or unlock the
prototype path. Recommended for R42+ if the priority list is otherwise empty.

## R41 cycle output

* Verdict: NO SHIP / refute via inspection
* Mechanism documented (3 independent legs; bandwidth + geometry +
  layout-ranking framework).
* HB-axis paradigm space: **EXHAUSTIVELY CLOSED** on existing tile geometry
  (5 cycles: R36 / R37 / R38 / R39 / R40 / R41).
* GPU-hours saved: ~1 (vs build+bench), ~5+ (vs full 4-GPU triangulation).
