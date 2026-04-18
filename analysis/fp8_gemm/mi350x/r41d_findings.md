# R41 Dev D — Alternative shared-memory layout for B operand: feasibility study

**Date:** 2026-04-18
**Branch:** r41-dev-d (base feat/mxfp8-only @ bc89af67, R40 cycle wrap)
**Scope:** Pure feasibility / closure check / R42+ scoping for "broadcast-B" or "swizzled-B" alternatives to the V2-CRR / V2-RRR / V2-RCR B-operand fastpaths. **No kernel implementation.**

## TL;DR — VERDICT: **NO-GO** (paradigm-closed under R19 Dev B + R29 Dev D + tile-area-conservation triple)

| Alternative | Feasibility | Lift Ceiling | VGPR Δ | LDS Δ | Closure rule |
|---|---|---:|---:|---:|---|
| **Broadcast-B via cross-lane** (intra-wave permlane/DPP, eliminate B's LDS round-trip) | **Architecturally impossible** for the inter-warp share that motivates the broadcast | n/a (0) | n/a | n/a | R19 Dev B Angle 2 (CDNA3/4 has no inter-wave register-to-register primitive) |
| **Broadcast-B via VGPR cache** (load B once per warp into VGPR, eliminate per-MFMA `ds_read_b64_tr_b8`) | Architecturally possible but VGPR-budget-exclusive | **0–negative** (8B Gate would lose RRR's lead) | **+32–64 VGPR** (V2-CRR 234→266–298; V2-RRR 256→saturate hard, +10–40 lane spill) | −16 KB B-LDS save | tile-area-conservation (3-confirm), VGPR ceiling (R34 Dev B build log) |
| **Swizzled-B / re-permute B in LDS** (different bank pattern, e.g. row-pad or different `sw_k` mask) | Architecturally trivial but mathematically void | **0%** (audited: bank pattern is already optimal — 0 conflicts, all 32 banks per cycle) | 0 | 0 to +B-LDS pad | R29 Dev D LDS-bank-conflict closure (already-conflict-free swizzle) |

This entire class of "alternate B-operand shared-memory layout" levers falls under **three independently confirmed closures**: (1) the R19 Dev B inter-wave-broadcast architectural impossibility, (2) the R29 Dev D LDS-bank-conflict-already-zero audit, and (3) the tile-area-conservation paradigm (3-confirm: R34 Dev D + R35 Dev C + R40 Dev A) which forbids "free" reductions in per-warp B residency.

The R41+ priority list candidate #3 ("alternate B-operand shared-memory layout") should be **closed** with a cumulative tally entry. Recommended new closed-paradigm tag: `B-operand-layout` (4th class lever family, joins HB-N / HB-M / tile-area-conservation).

---

## 1. Survey of prior B-operand work (R19–R40)

### 1.1 R19 Dev B — Cross-wave broadcast scout (`TODO.md:1801–1805`)

> **Angle 2 cross-wave broadcast via permlane/DPP**：NO. AMD CDNA3/4 **没有 inter-wave register-to-register primitive**（`ds_bpermute` / `permlane16` 都是 intra-wave，32-lane / 64-lane only）. Inter-wave broadcast 必须走 LDS = SCALE_LDS

**Implication for broadcast-B:** The 8-wave V2-CRR / V2-RRR fastpath has WARPS_N=4. B-tile data covering distinct N-cols is consumed by distinct warps. To "broadcast B to multiple warps without LDS" is structurally impossible on gfx950 — there is no inter-wave register primitive. Any broadcast-B scheme that eliminates LDS for B must either (a) reduce WARPS_N to 1 (collapsing to a 2-warp kernel — already CLOSED via tile-area-conservation, R40 Dev A WARPS_N=2 refuted) or (b) replicate B redundantly per warp via per-warp VMEM (doubles B-VMEM, strictly worse than the current LDS round-trip).

### 1.2 R19 Dev B — Reuse hunt (3 angles all NO; `TODO.md:1801–1805`)

- Angle 1 (cross-iter A reuse): NO — k_pair stride 256B = 64 dwords, fully disjoint.
- Angle 2 (cross-wave broadcast): NO — no HW primitive (above).
- Angle 3 (B scale share between A halves): NO — half=0/1 by HB=128 = distinct row_bases = distinct VMEM transactions.

**3 angles all NO** → only the preshuffle-V2 layout (Dev A R19) survived as a structural lever. That eventually shipped as the V2 layout family (R20–R28) which is the current baseline.

### 1.3 R29 Dev D — V2-CRR LDS bank-conflict audit (`r29d_lds_bank_audit.md`)

Static analysis of `load_col_from_v2_st_half` + `load_col_from_v2a_st_half` + ST_v2/v2a swizzle:

- For every 16-lane dispatch cycle, the existing swizzle `(nc ^ sw_k)` places lanes on all 32 banks exactly once.
- Both `ds_read_b64_tr_b8` instructions (offset:0, offset:1024) inherit the conflict-free pattern.
- Cycle-0/1/2/3 bank patterns identical (translation by 0 mod 32 banks).
- K_HALF=0 and K_HALF=1 produce identical bank patterns.
- Empirically corroborated: SQ_LDS_BANK_CONFLICT/SQ_INSTS_LDS < 1% (R22-B / R26-A profiling).

**Verdict:** "LDS bank conflict reduction" is a DEAD-END for V2-CRR. Sub-experiments rejected: ST_v2 row-pad (waste LDS without benefit), re-mask `sw_k` to `(k_row & 15) << 4` (per-cycle row-set is already exhaustively covered).

### 1.4 R34 Dev B — V2-RRR vs V2-CRR triangulation (`r34b_findings.md`)

V2-RRR (row-shared B + transpose in registers) > V2-CRR (column-shared B) by **+5–6%** on 8B Gate / 8B Up (4096×14336×4096). Mechanism: A-fetch coalesce wins (A is row-major = same direction as the dot product, so A loads are cheaper); the cost is +16 VGPR `B_row_reg tmp` for the transpose bridge.

Build log resource:
- V2-CRR fastpath @ `crr_mxfp8_exact_8wave_fastpath.inc:246` → **234 VGPR / 0 spill**
- V2-RRR fastpath @ `rrr_mxfp8_exact_8wave_fastpath.inc:23` → **256 VGPR / 1 spill** (already at the ceiling)

**Implication:** V2-RRR has zero VGPR headroom. Any extra VGPR cost (broadcast-B, B-cache) on V2-RRR triggers immediate lane-spill. V2-CRR has 22 VGPR headroom (234 → 256 ceiling) — a broadcast-B that costs ≤22 VGPR might fit, but see §2 arithmetic.

### 1.5 R34/R35/R40 — tile-area-conservation closure (3-confirm)

Three independent confirmations (`TODO.md:112`):
- R34 Dev D: sub-RBM (smaller per-warp tile, more warps) — **REFUTED**. Shrinking RBM does not reduce per-warp VGPR because total tile area = WARPS_M·WARPS_N·RBM·RBN is invariant under the rotation that conserves NUM_WARPS=8.
- R35 Dev C: WARPS_M=4 on V2-CRR (W4 reorder; equivalent to byte-shuffle of WARPS_M and WARPS_N) — **REFUTED**. 234→256 VGPR + 7-lane spill.
- R40 Dev A: WARPS_N=2 on V2-RRR (the only remaining structural rotation) — **REFUTED** (doubly-pre-refuted compound).

> Per-warp tile area (RBM·RBN·4) is structurally invariant under (WARPS_M, WARPS_N) rotation when product is fixed.

**Implication for broadcast-B:** Any scheme that puts B in VGPR (instead of LDS) does not reduce the per-warp working-set; it shifts the storage class. The tile-area-conservation closure says you cannot route around a fixed per-warp footprint by moving warps around — the same applies to moving B from LDS to VGPR.

### 1.6 R29 Reviewer / R30 — VGPR reduction for occupancy=2→3

R30 Dev B/C dug for VGPR reductions to allow occupancy=3 on V2-CRR. Conclusion (`r30b_findings.md:128`):

> The Dev D R29 audit's "scale broadcast registers held across MFMA quadrants" hypothesis was already minimized in R28: scale packs total 6 VGPRs, and they are read by MFMA opsel-selected lanes, not broadcast-replicated. There is no obvious VGPR reduction available without changing the algorithm.

The 4 fp32 accumulators (cA/cB/cC/cD) alone are 128 VGPR and load-bearing for correctness. **Even if VGPR were reduced to 84, occupancy would still be 1 block/CU because LDS is 1.7× the budget for 2 blocks/CU.**

**Implication:** The current V2-CRR / V2-RRR resource bottleneck is LDS, not VGPR per-block-occupancy. So a "trade LDS for VGPR" lever (broadcast-B) does not improve occupancy and only buys ds_read elimination — which was just shown (R29 Dev D) to be already conflict-free / not the critical path.

### 1.7 R36 Dev A / R37 Dev B / R38 — HB shrink B1 (the only B-operand SHIP)

The ONE B-operand-touching SHIP across R28–R40 was R36 Dev A's HB-M shrink + B1 pipeline (8B-Down V2-CRR rect, +28%). This shipped as `crr_mxfp8_exact_8wave_hbshrink_fastpath.inc`. Mechanism: shrink BLK_M=128 (HB_M=64) frees -66 VGPR; the freed budget feeds a B1 LDS-interleave + cross-buffer DB pipeline. **Note:** this is *not* a different B-operand layout — it shrinks the M-tile so that fewer A-rows are LDS-resident, which secondarily reduces B-pipeline pressure. Does not change B's LDS swizzle, fetch granularity, or per-warp residency.

### 1.8 HB-N shrink (R38 / R39 / R40) — REFUTED via bandwidth saturation

HB-N shrink (BLK_N=128, HB_N=64 — the symmetric mirror of HB-M shrink, on the B-side) was prototyped by R38 Dev A on V2-CRR (-43% to -47% across 3 wide-N shapes), R39 Dev A on V2-RRR (early-abort — refuted via the same mechanism), and R40 Dev A on V2-RRR + WARPS_N=2 (refuted via tile-area-conservation compound).

The R38 closure (`TODO.md:281`):
> V2-CRR is bandwidth-saturated on wide-N (WARPS_N=4); freed VGPR structurally has no pipeline to feed.

**Implication:** B-side bandwidth on the V2 family is already at the line. A broadcast-B or swizzled-B alternative that does NOT relieve real bandwidth (only trades LDS bytes for VGPR bytes, or swaps one swizzle for another at the same bank-conflict count of zero) is in the same closed-paradigm class.

---

## 2. Architectural arithmetic — broadcast-B alternative

### 2.1 Current B-tile geometry (V2-CRR default)

- BLK = 256, HB = 128, BK = 128, WARPS_M = 2, WARPS_N = 4, RBM = 64, RBN = 32
- B-LDS: `__shared__ ST_B Bs[2][2]` where each tile is HB × BK = 128 × 128 bytes = **16 KB/tile** (FP8 = 1 byte)
  - Total B-LDS = 2 (DB) × 2 (k_half) × 16 KB = **64 KB** (mainline V2-CRR/RRR; per `kernel_mxfp8_layouts.cpp:2369, 4443`)
- B-LDS per CTA-iteration consumed: 2 tiles × 16 KB = **32 KB/iter** (matches task brief)
- Per-warp B-residency in registers (during dot product): each warp consumes RBN=32 N-cols × BK=128 K-rows = **4 KB/warp from LDS** (loaded into `B_col_reg`/`B_row_reg`, ~16 VGPR `b0` + 16 VGPR `b1` = 32 VGPR after the transpose-bridge tax on RRR)
- Per-warp dot-product uses `b0` and `b1` (2 K_HALF instances, 4 KB total at 4-byte/lane after the b64_tr_b8 unpack)

(Brief's "8 KB/warp from LDS each iter" tracks total B-LDS bytes touched per warp across both K_HALFs; the residency in VGPR at any one time is ~32 VGPR for both `b0` and `b1`.)

### 2.2 Broadcast-B option (load B once per warp into VGPR, no per-MFMA `ds_read_b64_tr_b8`)

**Architectural sketch:** Pre-load the entire per-warp B-tile (RBN=32 cols × BK=128 K-rows = 4 KB) into VGPR once per CTA-iteration; consume from VGPR in the inner MFMA loop instead of issuing `ds_read_b64_tr_b8` 4 times per j-loop iteration.

**VGPR cost:**
- B-tile per warp = 32 × 128 = 4096 bytes = 1024 dwords / 64 lanes = **64 dword per lane = 64 VGPR per warp** (if held all at once)
- Even in the most aggressive "split into b0/b1 K-halves" scheme: each half = 32 VGPR per warp; you must double-buffer one or both = 32–64 VGPR
- This is *additive* to the existing 32 VGPR for `b0` + `b1` register destinations (which would be aliased into the new B-cache; net add = +32 VGPR best case, +64 VGPR worst case)

**VGPR budget check:**
| Layout | Baseline VGPR | + Broadcast-B (best/worst) | Verdict |
|---|---:|---:|---|
| V2-CRR | 234 | 266 / 298 | **Spill** at best, hard-saturate at worst |
| V2-RRR | 256 (1-lane spill) | 288 / 320 | **Hard-saturate** (already at ceiling) |
| V2-RCR | ~242 (per `r34a_build_c5_8b_gate.log` median over RCR builds) | 274 / 306 | **Spill** at best |

Even the "best case" exceeds the 256 VGPR ceiling by 10 VGPR on V2-CRR and by 32 VGPR on V2-RRR. **Forecast: 5–30 lane spill across the layout family**. Lane spill on AMD CDNA3/4 costs 4–8 cycles per spill/reload, in the inner loop = catastrophic regression (precedent: R35 Dev C W4 reorder, 7-lane spill = -12.16% standalone before re-pipelining).

**LDS savings:** -16 KB B-LDS per buffer = -32 KB total (DB). LDS budget per V2-CRR CTA is currently 139,264 B (TODO.md:152 reference). Saving 32 KB → 107 KB. This is *below* the 2-blocks/CU threshold and could in principle enable occupancy=2 *if* VGPR is also <128. But VGPR is going *up* not down, so occupancy stays at 1.

**Bandwidth-relief:** Does not relieve real VMEM bandwidth (VMEM→LDS load count is unchanged; the broadcast just changes how many ds_reads are issued downstream). Does not relieve VGPR pressure (it adds VGPR). The only thing it relieves is `ds_read_b64_tr_b8` issue count — which R29 Dev D audit established is already conflict-free at zero per-cycle stalls.

**Result:** broadcast-B trades LDS bytes for VGPR bytes with no occupancy win (occupancy already capped by VGPR not LDS in the best case, and would worsen here), no bandwidth win, no bank-conflict win, and a strict-positive issue cost from spill. **NO-GO.**

### 2.3 Swizzled-B option (different LDS bank pattern)

**Architectural sketch:** Replace the existing `(nc ^ sw_k)` swizzle in `load_col_from_v2_st_half` with a different bit-pattern, OR pad the ST_v2 row by 4 B (cols=132 instead of 128), aiming to reduce LDS bank conflicts on the dot product fetch path.

**Bank-conflict baseline:** Per R29 Dev D static audit, the existing swizzle achieves **all 32 banks used exactly once per cycle** for both `ds_read_b64_tr_b8` instructions in `load_col_from_v2_st_half` and `load_col_from_v2a_st_half`, on both K_HALF=0 and K_HALF=1, on all `j ∈ [0, RT::width)` iterations.

**Lift ceiling:** 0%. There is no bank-conflict cycle to reduce. The R29 Dev D audit explicitly rejects:
- Row-padding (cols=132): "wastes LDS without benefit and would increase per-CTA LDS footprint above the current 139264 B"
- Re-mask `sw_k` to `(k_row & 15) << 4`: "per-cycle row-set is `{8 contiguous rows}` so all row patterns ∈ `{0..7}` are already exercised — no improvement possible"

**Result:** swizzled-B is mathematically void — already optimal. **NO-GO.**

### 2.4 Hybrid: VMEM→LDS direct (`buffer_load_dword_lds`) on B-side

The R29 Dev D recommendation list (`r29d_lds_bank_audit.md:144`) lists this as the alternative to its closed audit:
> **Explicit `buffer_load_dword_lds` (VMEM→LDS direct)** — gfx950 supports this, V2-CRR currently goes VMEM→VGPR→LDS. Big restructure (the original alternative-2 lever in R29 Dev D's brief).

This is *not* a change to B's *layout* (the topic of this task); it is a change to the *path* by which B reaches LDS (eliminating the VGPR intermediate). It's a separate lever, partially explored under R30+ scale-LDS work. Out of scope for this feasibility study but flagged as a potentially-open lever (the only B-side lever NOT closed).

---

## 3. Comparison to closed paradigms

| Closed paradigm | Closure cycle/dev | Same-class with this task? | Reasoning |
|---|---|:---:|---|
| Cross-wave register broadcast (no HW primitive) | R19 Dev B | **YES** | Broadcast-B reduces to "share B between warps without LDS" = inter-wave register primitive needed = HW impossible |
| LDS bank conflict reduction | R29 Dev D | **YES** | Swizzled-B is direct sub-class; already 0 conflicts, no lift possible |
| Tile-area-conservation under (WARPS_M, WARPS_N) rotation | R34 Dev D + R35 Dev C + R40 Dev A | **YES (extends)** | Broadcast-B = "trade LDS for VGPR while keeping per-warp tile area fixed" = same conservation law |
| HB-N shrink (V2-CRR R38, V2-RRR R39, V2-RRR+W2 R40) | R38/R39/R40 Dev A | **PARTIAL** | HB-N shrink relieved B-LDS but not bandwidth; broadcast-B has identical bandwidth-relief failure mode |
| HB-M shrink (R37 Dev A wired, R36 Dev A SHIP) | R36/R37 | NO (ship-class) | HB-M+B1 shipped because shrinking M frees A-LDS, which is bandwidth-limited orthogonal to B |
| Rect-V2 paradigm (CRR Path 1 -8.5%, RCR -33-36%) | R33 Dev A + Dev B | NO (different mechanism) | Rect changes BLK_N geometry, not B-tile residency or fetch path |

The B-operand-layout class (broadcast-B, swizzled-B) is **fully covered** by the union of (R19 Dev B + R29 Dev D + tile-area-conservation 3-confirm). It is the **direct intersection** of "no inter-wave register primitive" and "swizzle is already optimal" and "VGPR ceiling is binding".

### Cumulative tally addition

Per `TODO.md:136`: 39 closed levers as of R40. This study adds **the 40th**: `B-operand-alt-layout` (broadcast-B + swizzled-B subclasses).

---

## 4. Recommendation

**NO-GO.** Do not allocate R42+ effort to a broadcast-B or swizzled-B prototype.

**Closure scaffold for R41 cycle wrap:**

> **B-operand alt-layout REFUTED on feasibility (R41 Dev D)**: Two sub-classes scoped — (a) broadcast-B via VGPR cache costs +32–64 VGPR, breaks 256 ceiling on all 3 layouts (V2-CRR 234→266/298; V2-RRR 256→288/320; V2-RCR ~242→274/306), with no bandwidth or occupancy win; (b) swizzled-B / LDS re-permute is mathematically void per R29 Dev D audit (existing `(nc ^ sw_k)` already achieves all-32-banks-per-cycle, 0 conflicts). Cross-wave broadcast variant ruled out architecturally (R19 Dev B: gfx950 has no inter-wave register primitive). **Closed: B-operand alternate shared-memory layout class** (joins HB-N / HB-M shrink + tile-area-conservation in the closed-lever tally — 40th cumulative lever). The only B-side lever NOT closed by this study is `buffer_load_dword_lds` (VMEM→LDS direct), which is a *path* change rather than a *layout* change — flagged separately for R42+ as the sole remaining structural B-side direction.

**No source modifications. No build artifacts. No runtime data. Pure analysis / closure.**

---

## 5. Audit checklist (per R29 Dev D template)

- [x] Read `kernel_mxfp8_layouts.cpp:471–508` (V2 col-load helper).
- [x] Read `kernel_mxfp8_layouts.cpp:637–673` (V2a col-load helper).
- [x] Read `kernel_mxfp8_layouts.cpp:2367–2427` (V2-RCR `Bs[2][2]` allocation + dispatch).
- [x] Read `kernel_mxfp8_layouts.cpp:4443–4519` (V2-RRR `Bs[2][2]` allocation + load_b lambda + transpose bridge).
- [x] Read `kernel_mxfp8_layouts.cpp:4782–4830` (V2-CRR `Bs[2][2]` block).
- [x] Read `analysis/fp8_gemm/mi350x/r29d_lds_bank_audit.md` (LDS bank-conflict closure).
- [x] Read `analysis/fp8_gemm/mi350x/r34b_findings.md` (V2-RRR > V2-CRR mechanism, VGPR figures).
- [x] Read `analysis/fp8_gemm/mi350x/r30b_findings.md` (VGPR reduction closure / occupancy=2 LDS-binding finding).
- [x] Searched `analysis/fp8_gemm/mi350x` for `permlane|ds_bpermute|DPP|dpp_|cross-lane|broadcast` — only R19 Dev B (TODO.md:1801) and the unrelated scale-LDS R30B/C references.
- [x] Cross-checked TODO.md R28–R40 cycle wraps for prior B-operand experiments (R36 Dev A HB-M B1 SHIP is the only B-touching SHIP, and it does not change B layout).
- [x] Verified the closed-paradigm intersection (R19 Dev B + R29 Dev D + tile-area-conservation 3-confirm) covers both broadcast-B and swizzled-B sub-classes.
- [x] Verdict and 40th-cumulative-lever closure scaffold drafted for R41 cycle wrap.

---

## 6. References (all paths absolute under `/tmp/wt-r41-d/`)

- Source: `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp`
- B-load helpers: `kernel_mxfp8_layouts.cpp:471–508` (`load_col_from_v2_st_half` + `load_col_from_v2_st`); `:637–673` (`load_col_from_v2a_st_half` + `load_col_from_v2a_st`)
- V2-RCR B alloc: `kernel_mxfp8_layouts.cpp:2367–2427`
- V2-RRR B alloc + transpose-bridge: `kernel_mxfp8_layouts.cpp:4443–4519`
- V2-CRR B alloc: `kernel_mxfp8_layouts.cpp:4782–4830`
- Prior closures cited:
  - `analysis/fp8_gemm/mi350x/r29d_lds_bank_audit.md` (R29 Dev D LDS-bank-conflict closure)
  - `analysis/fp8_gemm/mi350x/r34b_findings.md` (V2-RRR > V2-CRR triangulation, VGPR figures)
  - `analysis/fp8_gemm/mi350x/r30b_findings.md` (VGPR reduction closure)
  - `TODO.md:1801–1805` (R19 Dev B inter-wave broadcast architectural impossibility)
  - `TODO.md:108–112` (tile-area-conservation 3-confirm: R34 Dev D + R35 Dev C + R40 Dev A)
  - `TODO.md:281` (HB-N shrink V2-CRR closure — bandwidth-saturation argument)
  - `TODO.md:136` (39 closed levers as of R40)
- Build log VGPR figures: `analysis/fp8_gemm/mi350x/r34a_build_c5_8b_gate.log` (V2-CRR 234, V2-RRR 256+1-spill, V2-RCR ~242)
