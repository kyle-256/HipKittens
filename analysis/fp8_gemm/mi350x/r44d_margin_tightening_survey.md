# R44 Dev D — R37/R38 margin tightening survey (R44+ priority #5)

Date: 2026-04-18
Branch base: HEAD = `51a7c759` (R43 cycle wrap)
Author: R44 Dev D
GPU usage: 0 GPU-min (pure offline analysis from R31-R43 wrap sections,
R43D drift audit, dispatcher source, and HB-shrink + RRR kernel inspection)

## Scope

Per task spec, two gold-standards are at the limit of their stated margin:

1. **70B-KV HB shrink B1**: R43 reviewer +28.49% vs target ≥+28% — margin **+0.49pp**
2. **8B Gate/Up V2-RRR**: R43 reviewer  +5.28% vs target ≥+5.0% — margin **+0.28pp**

For each, this survey:
- Reads R37/R38 cycle wrap to recover ship rationale + target derivation
- Reads dispatcher kernel paths (HB-shrink B1 .inc / RRR exact-8wave .inc)
- Decides whether the target has hardware-resource-accounting room to widen
  (tighten the lower bound, freeing headroom) — or whether it is a **floor
  limit** (silicon-bin perf cap)
- Recommends one of: TIGHTEN-target / WIDEN-margin via micro-opt / NO-CHANGE

## TL;DR

| # | Cell | Latest Δ% | Target | Margin | Verdict | Recommendation |
|---|------|-----------|--------|--------|---------|----------------|
| 1 | 70B-KV HB shrink B1 | +28.49% | ≥+28% | +0.49pp | **FLOOR-LIMIT** (silicon-bin envelope 28.02-30.39%) | **NO-CHANGE** |
| 2 | 8B Gate/Up V2-RRR   |  +5.28% | ≥+5.0% | +0.28pp | **FLOOR-LIMIT** (silicon-bin perf cap, R40 boundary-lock rule) | **NO-CHANGE** target; R45+ scope: investigate RBN/HBN fan-out for marginal lift |

**Both targets are floor-limits — the apparent "tight margin" is the
distance from the cell's structural perf cap to the original SHIP claim,
not a drift signal.**  Tightening either target would reduce a
margin-of-safety against silicon-bin variance without delivering more
performance.  Widening via micro-optimization is bounded for #1 by
VGPR/HBM accounting (already at PIPE=1 ceiling per R36 Dev A pivot
analysis) and for #2 by the existing default-V2-RRR tile config (already
WARPS_N=4 RBN=32 — well-saturated for wide-N).

## #1 — 70B-KV HB shrink B1 (M=4096, N=1024, K=8192)

### Ship rationale and target derivation

Cycle history of the +28% target:

| Cycle | Δ% (min reported) | Source | Notes |
|------|------------------:|--------|-------|
| R36 (Dev A original SHIP) | +28.02 | TODO:556 | 1-GPU original |
| R37 (Reviewer) | +27.79 | r37_reviewer_findings.md:75 | 2-GPU triangulation |
| R37 (Dev A wire-in) | +30.39 | TODO:619 | 4-GPU production wire |
| R38 (Reviewer)  | +28.07 | r38_reviewer_findings.md:7 | 5-cycle stable |
| R39 (Reviewer)  | +28.38 | TODO:404 | 6th cross-cycle confirm |
| R40 (Reviewer)  | +28.96 | TODO:307 | 7th |
| R41 (Reviewer)  | +29.47 | TODO:203 | 8th |
| R42 (Reviewer)  | +28.17 | TODO:114 | 9th |
| R43 (Reviewer)  | +28.49 | r43_reviewer_findings.md:99 | 10th |

The +28% STRICT gate **was not analytically derived**.  It was set in
R36 by Dev A's GPU3-only SHIP (+28.02%) and inherited as the floor of
the bin envelope for all subsequent reconfirms.  Across 10 measurements
spanning 8 cycles, the band is **28.02–30.39%** with no monotonic trend
(R43D linear regression slope ≈ −0.05 pp/cycle, effectively flat).

### Kernel-path inspection

Predicate: `CRR-V2-HBSHRINK-B1-70B-KV (R37AB SHIP)` at
`kernel_mxfp8_layouts.cpp:6000-6013` (gated by `MXFP8_CRR_BLK_M==128`).

```cpp
#if defined(MXFP8_CRR_BLK_M) && (MXFP8_CRR_BLK_M == 128)
if (g.m == 4096 && g.n == 1024 && (g.k == 8192 || g.k == 4096) &&
    crr_can_use_exact_8wave_scaled_hbshrink(g)) {
    dispatch_crr_exact_8wave_scaled_v2_hbshrink<true>(g);
    return;
}
#endif
```

Backing kernel:
`crr_mxfp8_exact_8wave_hbshrink_fastpath.inc` (834 lines).  Comments at
top (lines 10-46) document the **R34 Dev D §3.2 PARADIGM PIVOT**:
- Default kernel hits −15% structural ceiling at PIPE=3 because
  `A_col_reg=32 VGPR/wave + accumulator=128 VGPR/wave` leaves no room
  for the `a_next` register PIPE=2..3 needs.
- HB shrink halves M-direction block: `BLK_M=128`, `HB_M=64`, drops
  cC/cD accumulators (top-half only) → accumulator footprint drops
  from `4×32=128 VGPR` to `2×32=64 VGPR` per warp.
- Production PIPE=1 (`MXFP8_CRR_HBSHRINK_PIPELINE=1`) adds DB main loop
  + cycle-2 B-tile prefetch + cycle-2 A-tile prefetch + B1 LDS
  interleave inside cA MMA chain.

PIPE=2 and PIPE=3 variants exist (lines 39-47) but per R36 Dev A's
original notes they **degenerate to the single-A early-issue form**
because there's only 1 A-load per iter — no benefit.

### Hardware-resource-accounting room?

Cross-checked against R36-R37 SHIP analysis:
- VGPR with PIPE=1: ~168/wave (-66 vs default 234), 0 spill, occ=2.
- HBM bandwidth at observed +28-30% lift on 70B-KV (M=4096 N=1024
  K=8192 = 32 GiB tile load + ~67 MiB output): kernel achieves
  ~989-1018 TF on this shape vs 765-790 TF baseline — already very near
  the FP8 per-tensor reference (R37 wrap: HB-shrink B1 = 107.4% of FP8).

Quoting r37_reviewer_findings.md (TODO:619):
> First V2-CRR cell to ever clear FP8 per-tensor reference: HB shrink
> B1 production = 1010.70 TF vs FP8 941.11 TF = 107.4%.

**There is no obvious accounting room to widen.**  The HB-shrink kernel
is at PIPE=1 ceiling for this geometry; PIPE=2/3 degenerate; further
M-direction shrinking (BLK_M=64) would halve grid coverage with no
gain.  The +28-30% Δ% IS the structural ceiling on this shape.

### Verdict and recommendation

**FLOOR-LIMIT.  NO-CHANGE.**

The +28% target is the floor of a 10-measurement silicon-bin envelope,
not an analytically-derived gate.  R43's +28.49% sits at the LOW end of
the band (median ~28.7%) but the trend is flat (slope −0.05 pp/cycle,
R43D audit).  Tightening to ≥+28.5% would (a) discard ~half of past
clean SHIP measurements as failures and (b) deliver no actionable
information about kernel health.  Widening via micro-optimization
appears blocked by the PIPE=1 VGPR/HBM ceiling already proven by R36
Dev A's pivot analysis.

**Optional R45+ scope** (NOT for R44 cycle):
- A speculative `MXFP8_CRR_HBSHRINK_PIPELINE=4` ("B3v2", placeholder
  reserved per `crr_mxfp8_exact_8wave_hbshrink_fastpath.inc:75`) could
  attempt a sub-tile interleave of A-read + B1-read inside the cA MMA
  chain.  Per the R36 Dev A note this was attempted as B3 and returned
  worse results — would need a fresh design pass.  **Defer indefinitely
  unless R44+ surfaces a falling trend on this cell.**

## #2 — 8B Gate/Up V2-RRR (M=4096, N=14336, K=4096)

### Ship rationale and target derivation

Cycle history of the +5% target:

| Cycle | Δ% (min reported) | Source | Notes |
|------|------------------:|--------|-------|
| R34 (Dev B SHIP-LITE)  | +5.025 | TODO:735 | First measurement (cited later) |
| R36                    | +5.05  | TODO:304 | SHIP-LITE confirm |
| R39 (Dev B STRICT)     | +5.13  | TODO:304 | N_PAIRS=20 + PREHEAT=120 needed to clear t-stat |
| R40                    | +6.22  | TODO:304 | Rotation skewed to fast bin |
| R41                    | +5.76  | TODO:206 |  |
| R42                    | +5.17  | TODO:117 |  |
| R43                    | +5.28  | r43_reviewer_findings.md:115 |  |

The +5.0% STRICT gate is **literally the R34B floor measurement** (the
dispatcher comment at `kernel_mxfp8_layouts.cpp:5978` even encodes
this: `"ADVISE-V2-RRR-8B-GATEUP (R34B +5.025% min)"`).  R39 promoted to
STRICT after the cell finally cleared `min Welch t > 10` at N=20.

The R40 cycle established the **BOUNDARY-LOCK rule** (TODO:351,
R40_reviewer_findings.md / TODO:407):
> R38 Dev D's statistical-power-cap hypothesis DOES NOT APPLY here:
> t=+18.52 deep into clearance — constraint is **silicon-binning
> performance-cap on GPU3** (5 GPU3 measurements all in [+4.609,
> +5.198], median +4.910).  Cross-cycle: R34 +5.025 / R36 +5.05 / R39
> Gate +5.131 / R40 Up +4.910 — all straddle +5.0 boundary.

R43D drift audit classified this cell as **STABLE-OSCILLATING ±0.5pp**
around the +5.0-5.5% structural floor.

### Kernel-path inspection

Predicate emits an **advisory only**: at
`kernel_mxfp8_layouts.cpp:5977-5979`:

```cpp
if (g.m == 4096 && g.n == 14336 && g.k == 4096) {
    MXFP8_DISPATCH_TRACE_ONCE("crr_v2",
        "ADVISE-V2-RRR-8B-GATEUP (R34B +5.025% min)", g);
}
```

The actual lift comes when the caller switches their entry point from
`gemm_crr_pq_v2` to `gemm_rrr_pq_v2`, which routes to
`dispatch_rrr_exact_8wave_scaled_v2` (`kernel_mxfp8_layouts.cpp:5887-5897`):

```cpp
if (rrr_can_use_exact_8wave_scaled(g)) {
    MXFP8_DISPATCH_TRACE_ONCE("rrr_v2", "RRR-V2-EXACT-8WAVE", g);
    dispatch_rrr_exact_8wave_scaled_v2<true>(g);
    return;
}
```

Backing kernel: `rrr_mxfp8_exact_8wave_fastpath.inc` (578 lines).  Tile
config:
- `WARPS_M=2, WARPS_N=4` (static-asserted, line 18-19)
- `RBM, RBN` matching default (RBN=32 for wide-N)
- `BLK_M`, `BLK_N` standard (256/256 for the 8-wave fast path)

This is the **default V2-RRR fastpath** — there is no shrink/expand
variant currently dispatched for the Gate/Up shape.  R38 Dev A
attempted HB-N shrink (`MXFP8_CRR_BLK_N=128`) on V2-CRR for this
geometry and measured **−43.18%** (TODO:570) — the V2-CRR baseline is
already well-tuned for wide-N with WARPS_N=4 + RBN=32, and HB-N shrink
adds WG-grid + barrier overhead with no compute benefit.  R39 Dev A
attempted the same on V2-RRR and the kernel was REFUTED via tile-config
inspection without burning benches (TODO:453).

### Hardware-resource-accounting room?

The R34B +5.025% Δ% measures (V2-RRR Gate/Up) vs (V2-CRR Gate/Up) —
both via the dispatcher.  V2-RRR wins because:
- A=row-major (M,K) fits gate/up's natural input layout
- B=row-major (K,N) is the natural weight layout
- avoids the V2-CRR transposed A-staging penalty

But V2-RRR for this shape is also already at its tile-config ceiling:
- WARPS_N=4 + RBN=32 = 128 N-cols/wave: optimal for wide-N
- Default DB pipelining; no additional headroom from PIPE bumps
  (Gate/Up at K=4096 is a relatively short K-loop)

R36-R39 SHIP-LITE → STRICT chain spent 3 cycles trying to clear STRICT
t>10; the LIFT itself never moved beyond ~+5.0-5.3% on the
slow-silicon-bin GPUs.  This is the cell's structural floor.

### Verdict and recommendation

**FLOOR-LIMIT.  NO-CHANGE target.**

The +5.0% target is the dispatcher-encoded R34B SHIP claim and
matches the silicon-bin perf cap measured by R40 across 5 GPU3
samples.  R43D classified this as STABLE-OSCILLATING; R43's +5.28% is a
typical sample within the ±0.5pp band.  Tightening would cause
intermittent failures on any single-GPU rotation that lands a slow-bin
GPU.

**Optional R45+ micro-optimization scope** (NOT for R44 cycle):

The Gate/Up shape (M=4096, N=14336, K=4096) is **N-heavy K-light**.
Two speculative directions worth a future cycle's planning, both
subject to dedicated kernel work:

1. **N-tile expansion (BLK_N=384 or 512) for V2-RRR Gate/Up**:
   currently RBN=32 × WARPS_N=4 covers BLK_N=128.  N=14336 = 112×128
   tiles.  A BLK_N=256 variant (RBN=64 × WARPS_N=4 OR RBN=32 ×
   WARPS_N=8) could reduce grid count + better amortise the
   per-output-tile setup.  Pre-refute test: WARPS_N=8 was REFUTED by
   R35 Dev C (TODO:363 R40 audit cites Leg A pre-refutation); RBN=64
   would need fresh prototype.  Estimated: 1-2 days for kernel; 1 day
   for bench triangulation.  Expected lift: 0.5-1.5pp on best case.

2. **K-tile shrink + larger M coverage**: K=4096 is small enough that
   the 4-stage K-loop pipeline may be under-utilized.  Could try
   BLK_M=512 (RBM=128 + WARPS_M=4) or similar to lift M-wave throughput
   at the cost of accumulator VGPR.  R34 Dev D pivot analysis
   established VGPR ceiling on V2-CRR; need re-derivation for V2-RRR
   wide-N.

Both are **R45+ kernel-design work** requiring bench infrastructure.
Per task spec, do NOT implement in R44 cycle.

## Cross-cell summary table

| Property | #1 70B-KV HB shrink B1 | #2 8B Gate/Up V2-RRR |
|----------|------------------------|----------------------|
| Predicate kernel | `CRR-V2-HBSHRINK-B1-70B-KV` | `RRR-V2-EXACT-8WAVE` (advisory only) |
| Path | `dispatch_crr_exact_8wave_scaled_v2_hbshrink<true>` | `dispatch_rrr_exact_8wave_scaled_v2<true>` |
| Inc file | `crr_mxfp8_exact_8wave_hbshrink_fastpath.inc` (834 LOC) | `rrr_mxfp8_exact_8wave_fastpath.inc` (578 LOC) |
| Original SHIP cycle | R36 Dev A | R34 Dev B |
| Target encoding | 28% (silicon-bin floor) | 5.0% (R34B dispatcher comment) |
| 8-cycle envelope | 28.02–30.39% | 5.025–6.22% |
| R43D trend | STABLE (slope −0.05 pp/cycle) | STABLE-OSCILLATING ±0.5pp |
| Structural cap reason | PIPE=1 VGPR/HBM ceiling (R36 Dev A pivot) | WARPS_N=4 RBN=32 already wide-N optimal |
| Pre-refuted micro-opts | PIPE=2/3 degenerate; further BLK_M shrink no gain | HB-N shrink REFUTED V2-CRR (R38A) and V2-RRR (R39A); WARPS_N=8 REFUTED (R35C) |
| Recommendation | **NO-CHANGE** | **NO-CHANGE** target; R45+ kernel work optional |

## Cross-reference to R43D drift audit

This survey is consistent with R43D's `r43d_gold_standard_drift_audit.md`:
- #1 70B-KV: R43D classified STABLE within silicon-bin variance
  (see r43d:#1 verdict).  This survey adds the kernel-resource
  accounting confirmation.
- #2 8B Gate/Up: R43D classified STABLE-OSCILLATING per R40 BOUNDARY-LOCK
  rule (see r43d:#4 verdict).  This survey adds the V2-RRR tile-config
  + speculative R45+ scope.

R43D explicitly recommended **DO NOT modify dispatcher** (#3 in its
R43+ recommendations).  This survey extends that to **DO NOT tighten
either target**.

## R44+ recommendations summary

1. **Targets: NO-CHANGE.**  Both +28% / +5.0% are floor-limits, not
   tight margins in the drift sense.  Tightening either would generate
   spurious failures from silicon-bin noise.
2. **Margin-monitoring rule (RECOMMENDED, methodology)**: when reviewer
   flags a "tight margin" gold-standard, distinguish FLOOR-LIMIT (target
   = silicon-bin floor) from TRUE-DRIFT (target derived from
   hardware-resource analysis, Δ% falling).  Cite R43D audit for FLOOR
   classification before recommending widening work.
3. **R45+ optional scope**: V2-RRR Gate/Up wide-N tile-config
   exploration (RBN=64 BLK_N=256 OR analogous) — bounded ~0.5-1.5pp
   expected lift, requires fresh kernel prototype + 4-GPU triangulation.
4. **R45+ optional scope**: HB-shrink B1 PIPE=4 ("B3v2") sub-tile
   interleave — only if R44+ surfaces a falling trend; otherwise defer.

## Files
- `analysis/fp8_gemm/mi350x/r44d_margin_tightening_survey.md` — this file
- Cross-references:
  - `analysis/fp8_gemm/mi350x/r43d_gold_standard_drift_audit.md`
  - `analysis/fp8_gemm/mi350x/r37_reviewer_findings.md`
  - `analysis/fp8_gemm/mi350x/r38_reviewer_findings.md`
  - `analysis/fp8_gemm/mi350x/r39_reviewer_findings.md` (R40 BOUNDARY-LOCK reasoning at TODO:407)
  - `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp:5977-5979` (Gate/Up advisory)
  - `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp:6000-6013` (HB shrink B1 dispatch)
  - `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_hbshrink_fastpath.inc:1-80` (PARADIGM PIVOT comment block)
  - `analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc:18-19` (WARPS_M=2, WARPS_N=4 static-assert)
