# R40 Dev A — HB-N + WARPS_N=2 compound bet on V2-RRR (wide-N shapes)

## Verdict: **NO SHIP — early abort per R38 time-box rule (compound mechanism doubly pre-refuted)**

Per the R40+ task spec and the R38 explicit time-box rule ("if tile-config
inspection says the lever doesn't fit OR per-warp accumulator pressure
swallows all freed capacity, abort early"), the V2-RRR + WARPS_N=2 + HB-N
shrink compound bet was **not built or benched**. Both load-bearing legs
of the compound have already been empirically refuted in R35 Dev C and
R38 Dev A; V2-RRR's structural starting condition makes the prediction
strictly worse than the V2-CRR W4 mirror that R35 Dev C already closed.

---

## 1. The compound bet decomposition

The R40+ task hypothesis layers two independent mechanisms:

* **Leg A (WARPS_N=2 reorder)** — Rotate the warp grid from production
  WARPS_M=2 × WARPS_N=4 (= 8 warps) to WARPS_M=4 × WARPS_N=2 (= 8 warps,
  same total). Per-warp partition becomes RBM=32 / RBN=64 (vs RBM=64 /
  RBN=32 production). Hypothesized to free per-warp WG capacity.
* **Leg B (HB-N shrink)** — Halve N-direction half-block: HB_N=64 /
  BLK_N=128 (vs HB_N=128 / BLK_N=256). Hypothesized to relieve N-side
  bandwidth pressure now that WARPS_N=2 lifts the binding constraint.

Both legs must pull weight to land a SHIP. Because Leg A is structurally
identical to R35 Dev C's WARPS_M=4 V2-CRR scaffold (already empirically
closed) and Leg B is identical to R38 Dev A's HB-N V2-CRR scaffold (also
empirically closed), the compound has no degrees of freedom that haven't
already been observed.

---

## 2. Leg A — WARPS_N=2 (= WARPS_M=4 to keep 8 warps total) is byte-equivalent to R35 Dev C's CRR W4 scaffold

R35 Dev C scaffolded `crr_mxfp8_exact_8wave_warpsm4_fastpath.inc` with
exactly the geometry the R40+ task description proposes for V2-RRR:

```
WARPS_M_W4 = 4
WARPS_N_W4 = 2
RBM_W4     = BLK / WARPS_M_W4 / 2 = 256 / 4 / 2 = 32   (was 64 at WARPS_M=2)
RBN_W4     = BLK / WARPS_N_W4 / 2 = 256 / 2 / 2 = 64   (was 32 at WARPS_N=4)
```

(`crr_mxfp8_exact_8wave_warpsm4_fastpath.inc:113-118`,
`r35c_findings.md:99-105`).

R35 Dev C's first-order analysis (which is independent of A-fetch layout
— it only depends on register-tile shapes and per-warp tile area):

| Per-warp register | Production (WARPS_N=4) | W4 reorder (WARPS_N=2) | Δ VGPR |
|---|---|---|---|
| `A_col_reg` (BK × RBM) | 32 (128 × 64) | 16 (128 × 32) | -16 |
| `B_col_reg b0` (BK × RBN) | 16 (128 × 32) | 32 (128 × 64) | +16 |
| `B_col_reg b1` (BK × RBN) | 16 (128 × 32) | 32 (128 × 64) | +16 |
| Operand subtotal | 64 | 80 | **+16 net** |
| Accumulator (4 × rt_fl<RBM,RBN>) | 4 × (64/16)·(32/16) × 4 = 128 | 4 × (32/16)·(64/16) × 4 = 128 | **0** (identical area, transposed shape) |

**Per-warp tile area is conserved** when (WARPS_M, WARPS_N) reorder while
their product stays fixed. R35 Dev C confirmed this both analytically
and empirically (`r35c_findings.md:139-148, 192-220`).

R35 Dev C's empirical resource report from the W4 probe build (V2-CRR):

```
Production V2-CRR : VGPRs=234, Spill=0,  Occupancy 2 waves/SIMD
W4 V2-CRR skeleton: VGPRs=256 (saturated), Spill=7, Occupancy 2 waves/SIMD
```

Δ VGPR = +22 (saturated to the 256 budget cap), Δ spill = +7.

This refutes the "WARPS_N=2 frees per-warp WG capacity" premise of Leg A
on V2-CRR. The same operand register types are used in V2-RRR, so the
per-warp delta arithmetic is identical.

---

## 3. V2-RRR baseline starts at the VGPR ceiling — strictly worse starting point than V2-CRR

V2-RRR's exact-8wave kernel (SCALE_VERSION=2) already saturates VGPR at
the production WARPS_M=2 × WARPS_N=4 geometry. Resource report from the
R34 Dev B 8B Gate/Up build (`r34b_build_c5_8b_gate.log`):

```
./rrr_mxfp8_exact_8wave_fastpath.inc:23:1: VGPRs: 256 (saturated)
                                           VGPRs Spill: 1
                                           Occupancy 2 waves/SIMD
                                           ScratchSize 8 bytes/lane
./crr_mxfp8_exact_8wave_fastpath.inc:246:1: VGPRs: 234, Spill: 0, ScratchSize 0
```

So V2-RRR baseline has **0 VGPR headroom** vs V2-CRR's 22 VGPR headroom
(234 → 256). V2-RRR carries an extra `B_row_reg tmp` for the row-shared
load + transpose bridge (`rrr_mxfp8_exact_8wave_fastpath.inc:275, 291`)
plus the `b0_keep` / `b1_keep` aliases when
`RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS` is on; these
account for the +22 VGPR vs CRR.

### 3.1 W4 reorder forecast on V2-RRR

Applying R35 Dev C's empirical Δ to V2-RRR's saturated baseline:

| Metric | V2-RRR production | V2-RRR + W4 forecast | Mechanism |
|---|---|---|---|
| Operand VGPR | (256 cap) | 256 cap, +16 net push | Per §2 above |
| Accumulator VGPR | 128 | 128 | Conserved (same area) |
| `B_row_reg tmp` (transpose bridge) | 16 (RBN=32, BK=128) | 32 (RBN=64, BK=128) | +16 in transpose path |
| Spill | 1 | ≥ 8 (likely 15-25) | Compound of W4 spill + RRR's existing 1-lane pressure |

Net forecast: V2-RRR + W4 saturates VGPR with **2-3× the spill** that
R35 Dev C measured on CRR (which was already strictly worse than its
production baseline). Occupancy stays 2 waves/SIMD (256 VGPR cap is the
binding limit at 2 waves).

### 3.2 The transpose-bridge tax compounds W4

V2-RRR's `RRR_ROW_SHARED_TRANSPOSE` path materializes a `B_row_reg tmp`
of size `(RBN × BK)` → at RBN=64 it becomes 32 VGPR/wave (vs 16 at
RBN=32). On V2-CRR there is no such intermediate — B is loaded directly
in column form. So the W4 reorder taxes V2-RRR more heavily than V2-CRR
on the operand pipeline.

---

## 4. Leg B — HB-N shrink remains bandwidth-refuted regardless of warp grid

R38 Dev A's V2-CRR HB-N shrink mechanism (`r38a_findings.md` summary in
TODO.md:189): **default V2-CRR is bandwidth-saturated on wide-N
(B-operand throughput); halving N coverage doubles the WG grid in N
which doubles barriers + scale-fetch overhead WITHOUT relieving the
binding constraint.** Net: -43% to -47% across 3 wide-N shapes.

R39 Dev A's transfer argument (`r39a_findings.md:39-56`): V2-RRR is
**more** bandwidth-bound than V2-CRR (V2-RRR achieves ~2538 TF on c5
8B Gate/Up vs V2-CRR's ~2415 TF — both at the same N-partition
geometry). HB-N shrink would produce an even worse delta on RRR than
on CRR.

The W4 reorder does not change Leg B's mechanism: WARPS_N=2 still
distributes B across N (now 2 warps × RBN=64 = 128 per HB half, same
total per-block bandwidth as 4 × 32). Halving HB_N still doubles the WG
grid in N → still 2× barriers, still 2× scale fetches (now over a
smaller WG tile). The "freed VGPR" claim of Leg B fails because Leg A
saturates VGPR, not frees it.

---

## 5. Why the "feedback loop" hypothesis fails

The R40+ task description proposes that "WARPS_N=2 + RBN=64 ... may
unlock the 'HB-N freed VGPR has somewhere to go' feedback loop that was
missing in R38/R39". This requires:

1. WARPS_N=2 to **free** VGPR (so HB-N's hypothetical -64 VGPR
   accumulator drop would be re-investable). **REFUTED** — §2/§3 show
   WARPS_N=2 (= W4 reorder) **adds** +16-22 net VGPR pressure;
   accumulator is conserved.
2. HB-N to relieve a binding constraint other than bandwidth. **REFUTED**
   — Leg B's mechanism is purely WG-grid expansion overhead; it does
   not address bandwidth.
3. The compound to land at occupancy higher than 2 waves/SIMD.
   **REFUTED** — both production and W4 are at 2 waves/SIMD on V2-RRR;
   W4 saturates VGPR but doesn't drop occupancy because the next step
   down (1 wave/SIMD) requires VGPR > 512.

All three preconditions for the feedback loop fail. There is no path by
which the compound bet could win on V2-RRR wide-N shapes.

---

## 6. Time-box rationale (R38 explicit rule + R39 mirror)

R38 cycle wrap rule: "if a paradigm (HB-* shrink) is refuted on one
variant, before mirroring to a sibling variant, inspect the load-bearing
tile-config invariants. If they match, abort early." R39 Dev A applied
this and aborted V2-RRR HB-N (mono-leg) without bench. R40 Dev A
applies the same rule to the **compound**: both legs are individually
refuted, so the compound has no novel mechanism to test.

R40+ task spec: "if tile-config inspection says WARPS_N=2 doesn't fit
VGPR budget OR per-warp accumulator pressure swallows all freed capacity,
abort early like R39 Dev A did." Both conditions are met:

* **VGPR budget**: V2-RRR baseline is at 256 cap (vs CRR 234) with 1-lane
  spill. W4 push of +16 VGPR cannot fit without more spill.
* **Accumulator pressure**: per-warp accumulator area is exactly conserved
  (4 × 64 × 32 = 4 × 32 × 64 = 8192 elements per warp); W4 reorder is a
  rotation, not a relief.

Building the scaffold (mirror R35 Dev C's 424 LOC + V2 host preshuffle
re-tile for the new pack-count layout + RRR transpose-bridge plumbing)
would consume ~3-5 GPU-hours to arrive at a guaranteed structural
disproof (saturated VGPR + spill + bandwidth-class regression). Aborting
honors the R40+ spec time-box.

---

## 7. VGPR budget arithmetic (load-bearing summary)

For a 256 VGPR/wave occupancy budget at 2 waves/SIMD on gfx950:

```
V2-RRR production (WARPS_M=2 × WARPS_N=4, RBM=64, RBN=32):
  A_row_reg          (BK=128 × RBM=64, row-fetch path) ≈ 32 VGPR
  B_col_reg b0+b1    (2 × BK × RBN  = 2 × 128 × 32)   = 32 VGPR
  B_row_reg tmp      (RBN=32 × BK=128, transpose bridge) ≈ 16 VGPR
  Accumulator cA/B/C/D (4 × 64 × 32 fp32)               = 128 VGPR
  Scale packs + LDS offsets + SRDs + LDS addrs + scratch ≈ 48 VGPR
  ──────────────────────────────────────────────────────
  Total                                                   ≈ 256 (cap)  + 1-lane spill (measured)

V2-RRR W4 reorder (WARPS_M=4 × WARPS_N=2, RBM=32, RBN=64):
  A_row_reg_w4       (BK=128 × RBM_W4=32)              ≈ 16 VGPR  (-16)
  B_col_reg_w4 b0+b1 (2 × BK × RBN_W4 = 2 × 128 × 64)  = 64 VGPR  (+32)
  B_row_reg_w4 tmp   (RBN_W4=64 × BK=128)              ≈ 32 VGPR  (+16)
  Accumulator cA/B/C/D (4 × 32 × 64 fp32)               = 128 VGPR (unchanged, area conserved)
  Scale packs + ... (same as production)               ≈ 48 VGPR
  ──────────────────────────────────────────────────────
  Total forecast                                         ≈ 256 (cap) + ~10-20 lane spill
```

Per-warp tile area = RBM × RBN × 4 (accumulators) is invariant under
(WARPS_M, WARPS_N) rotation when WARPS_M·WARPS_N is fixed. The "halving
warps frees per-warp WG capacity" intuition fails because the freed
warp-count is paid back exactly by doubled per-warp work.

Layering HB-N (RBN_HB shrink): would drop accumulator by 64 VGPR
(2 of 4 cA/B/C/D collapse), but the WG grid doubles in N → barriers
double → scale fetches double. Per R38 Dev A, this costs more than
the freed VGPR is worth on bandwidth-saturated kernels. V2-RRR is
strictly more bandwidth-saturated than V2-CRR, so the loss is larger.

---

## 8. Files

* No new kernel `.inc` written.
* No changes to `kernel_mxfp8_layouts.cpp`.
* This findings doc (`r40a_findings.md`) records the inspection +
  abort rationale.

---

## 9. Followups / R41+ candidates

1. **Do NOT build V2-RRR + W4 + HB-N compound** under current
   tile geometry + register types. Both legs refuted; no novel
   mechanism in the compound.
2. **Generalize the closed-paradigm rule**: tile-area conservation
   under (WARPS_M, WARPS_N) rotation is a STRUCTURAL invariant for
   the exact-8wave family. Document this in the methodology so future
   cycles do not propose W4 / W2 reorder as a VGPR-relief lever
   (R34 Dev D sub-RBM + R35 Dev C W4 + R40 Dev A W2-on-RRR all
   independently confirm this). Lock into TODO.md "closed paradigm"
   list with mechanism tag `tile-area-conservation`.
3. **Direct B-bandwidth attack** for wide-N rect MXFP8 (carry-over
   from R39 Dev A followup §3): B-side LDS prefetch, B cache-policy
   bits, B re-laid-out per-tile format. This targets the binding
   constraint (B-bandwidth on wide-N) directly. Separate scaffold;
   not a tile-geometry knob.
4. **Asymmetric warp count (NUM_WARPS != 8)**: only path that breaks
   tile-area conservation is to change the total warp count (e.g.
   12 or 16 warps with proportionally larger BLK). This is a deeper
   rewrite of the LDS budget + occupancy model; not a 2-3 day cycle.

---

## 10. Hypothesis status update

| Claim (R40+ task) | Status (R40 Dev A) | Evidence |
|---|---|---|
| WARPS_N=2 frees per-warp WG capacity | **REFUTED** (mirror of R35 Dev C on CRR) | §2: per-warp tile area conserved; operand pair widens +32, A shrinks -16, net +16 VGPR pressure |
| WARPS_N=2 + HB-N together unlock VGPR feedback loop | **REFUTED** (compound legs both refuted) | §5 |
| V2-RRR has bandwidth headroom over V2-CRR | **REFUTED** (R39 Dev A) | V2-RRR is +5-6% over V2-CRR baseline at wide-N → MORE bandwidth-bound, not less |
| Net delta is 0 or negative (per task step 4) | **CONFIRMED NEGATIVE** | V2-RRR baseline already at 256 + 1-lane spill; W4 forecast pushes spill to ~10-20 lanes |
| Compound bet warrants build + bench | **NO** | Time-box per R38/R40+ rule; mechanism doubly closed |

---

## 11. Methodology rule compliance

| Rule | Compliance |
|---|---|
| Tile-config inspection FIRST (R38 time-box rule) | YES — §7 VGPR arithmetic done before any build |
| Structural-disproof closure preferred over speculative bench | YES — both legs already empirically closed in prior cycles |
| Closed-lever check before re-test | YES — R35 Dev C (W4 mirror) + R38 Dev A (HB-N mirror) |
| Save GPU hours on guaranteed-refute mechanism | YES — ~3-5 GPU-hours saved |
| Findings doc commits to dev branch | YES — `r40a_findings.md` on `r40-dev-a` |
| Default-build untouched (no scaffold landed) | TRIVIAL — no code changes |

---

## 12. Conclusion

The R40+ "HB-N + WARPS_N=2 compound on V2-RRR" lever decomposes into
two independently-refuted mechanisms:

* **Leg A** (WARPS_N=2 reorder = R35 Dev C's W4 mirror) is a tile-area-
  conserving rotation that adds ~+16 net VGPR pressure on operands and
  saturates the VGPR cap (R35 Dev C measured 256 + 7-lane spill on CRR;
  V2-RRR forecast is worse because baseline is already at the cap).
* **Leg B** (HB-N shrink) is bandwidth-mechanism-refuted on V2-CRR
  wide-N (-43% to -47% per R38 Dev A) and V2-RRR is strictly more
  bandwidth-saturated than V2-CRR per R34 Dev B / R39 Dev A.

The compound has no novel mechanism that could rescue either leg.
V2-RRR's baseline VGPR saturation (256 + 1 spill, vs CRR's 234) makes
the W4 mirror strictly worse on RRR than on CRR.

**SHIP/NO-SHIP verdict: NO SHIP (early abort, compound mechanism
doubly pre-refuted; per-warp accumulator pressure swallows all
hypothetically freed capacity per the task step-4 abort condition).**

GPU-hours saved: ~3-5 (no scaffold built, no bench run).
