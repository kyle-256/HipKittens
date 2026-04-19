# R54 Dev F findings — 70B Gate/Up CRR XCD swizzle granularity tuning

**Cycle:** R54
**Cell:** 70B Gate/Up CRR (M=4096, N=28672, K=8192)
**Baseline (shipped):** default = `MXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS=8` + `GROUP_M=4`
**FP8 reference (R53 Reviewer):** 2807.1 TFLOPS
**MX/FP8 ratio (default, this run):** 88.45% (HEADROOM ≈ -6.6pp vs 95% gate; cell-card recorded 89.3%)
**Lever:** XCD swizzle granularity sweep across `NUM_XCDS ∈ {4, 8(default), 16, 32}`
(GROUP_M=4 fixed) and `GROUP_M ∈ {1, 2, 4(default), 8, 16}` (NUM_XCDS=8 fixed).
**Origin:** R47B/R52G already SHIPPED XCD swizzle on/off for CRR but never tuned
the (NUM_XCDS, GROUP_M) tuple — Dev F closes that residual lever.

## Verdict — REFUTED-EMPIRICAL (no granularity slack)

The default (NUM_XCDS=8, GROUP_M=4) is at-or-above the entire 7-config swept
neighbourhood within 1× run-to-run noise. Best alternative (xcds04_gm04) is
+0.35% over default — strictly inside the ±0.5% noise band on this cell — and
no other config improves materially. All 8 configs build with byte-identical
resources (CRR kernel: 254 VGPR / 0 spill / 131072 LDS / occ 2). The granularity
axis carries zero exploitable slack at this shape.

## Phase 0: scope and config matrix

8 configs × 5 runs each = 40 strict-SCLK runs (MXFP8_WARMUP=100,
MXFP8_ITERS=200, 30 s inter-run cool, 60 s inter-build cool, GPU 1 isolated).
Per `r54f_bench.sh`:

| Tag           | NUM_XCDS | GROUP_M | Notes |
|---|---|---|---|
| default       | 8 (impl) | 4 (impl) | Currently shipped (R47B/R52G) |
| xcds04_gm04   | 4        | 4        | NUM_XCDS sweep down |
| xcds16_gm04   | 16       | 4        | NUM_XCDS sweep up |
| xcds32_gm04   | 32       | 4        | NUM_XCDS sweep up (limit) |
| xcds08_gm01   | 8        | 1        | GROUP_M sweep down (no row grouping) |
| xcds08_gm02   | 8        | 2        | GROUP_M sweep down |
| xcds08_gm08   | 8        | 8        | GROUP_M sweep up |
| xcds08_gm16   | 8        | 16       | GROUP_M sweep up |

`xcds08_gm04` was skipped — identical to `default`.

## Phase 1: ISA / resource verification

All 8 configs built successfully. The `MXFP8_CRR_BLOCK_SWIZZLE_*` macros are
host-side block-index remapping only — they do not touch the inner-loop kernel
body, so resource use is structurally invariant. Confirmed by inspection of the
CRR kernel resource block (`kernel_mxfp8_layouts.cpp:2415:1`) across all 8
build logs:

| Tag           | VGPRs | VGPR Spill | TotalSGPRs | LDS (B) | Scratch | Occ |
|---|---|---|---|---|---|---|
| default       | 254   | 0          | 52         | 131072  | 0       | 2   |
| xcds04_gm04   | 254   | 0          | 52         | 131072  | 0       | 2   |
| xcds08_gm01   | 254   | 0          | 52         | 131072  | 0       | 2   |
| xcds08_gm02   | 254   | 0          | 52         | 131072  | 0       | 2   |
| xcds08_gm08   | 254   | 0          | 52         | 131072  | 0       | 2   |
| xcds08_gm16   | 254   | 0          | 52         | 131072  | 0       | 2   |
| xcds16_gm04   | 254   | 0          | 52         | 131072  | 0       | 2   |
| xcds32_gm04   | 254   | 0          | 52         | 131072  | 0       | 2   |

**All 8 configs are byte-identical at the CRR kernel resource level.** Any
performance delta is attributable purely to the XCD scheduling reorder, not
register/LDS pressure.

(`r54f_results/isa/` directory exists but is empty — `r54f_isa_verify.sh` was
authored to dump per-config ISA but the run log shows resource verification
was sourced from the per-tag `*_build.log` files in `r54f_results/` instead.
Resource invariance is fully established without per-config ISA disassembly.)

## Phase 2: per-config performance (median of 5)

Strict-SCLK 5-run TFLOPS extracted from `*_run{1..5}.log`
(`Avg time: X ms, TFLOPS: Y` line, CRR layout, preshuffle-quant on):

| TAG          | med TFLOPS | min    | max    | spread | MX/FP8 | vs default |
|---|---|---|---|---|---|---|
| default      | 2482.9     | 2477.8 | 2490.7 | 0.52%  | 88.45% | +0.00%     |
| xcds04_gm04  | 2491.5     | 2479.7 | 2495.9 | 0.65%  | 88.76% | **+0.35%** |
| xcds16_gm04  | 2485.2     | 2483.8 | 2488.7 | 0.20%  | 88.53% | +0.09%     |
| xcds32_gm04  | 2480.1     | 2452.4 | 2492.8 | 1.63%  | 88.35% | -0.11%     |
| xcds08_gm01  | 2483.2     | 2471.2 | 2488.0 | 0.68%  | 88.46% | +0.01%     |
| xcds08_gm02  | 2481.4     | 2418.1 | 2485.6 | 2.72%  | 88.40% | -0.06%     |
| xcds08_gm08  | 2481.9     | 2441.0 | 2485.3 | 1.78%  | 88.41% | -0.04%     |
| xcds08_gm16  | 2483.1     | 2480.9 | 2492.0 | 0.45%  | 88.46% | +0.01%     |

(Reproduced from the inline summary block in `r54f_bench.run.log`; medians
re-verified by hand against the per-run logs.)

### Best-config analysis: xcds04_gm04 (+0.35% vs default)

- Median delta = +8.6 TFLOPS = +0.35% — strictly inside the ±0.5%
  cell-noise band documented for 70B Gate/Up CRR.
- Run spread for xcds04_gm04 (0.65%) overlaps default's spread (0.52%); no
  shift-of-distribution evidence.
- All other 6 configs cluster in [-0.11%, +0.09%] vs default — flat plateau.
- Conclusion: NO config beats default by ≥1% (SHIP threshold) and best
  config matches default within ±0.5% (REFUTED-EMPIRICAL threshold) — the
  default (NUM_XCDS=8, GROUP_M=4) sits on a flat granularity plateau.

### Worst-config analysis: xcds32_gm04 (-0.11% vs default)

- NUM_XCDS=32 over-shards block placement on a 304-CU MI355X (one swizzle
  domain per ~9.5 CU), introducing scheduling noise (run spread 1.63%) but
  no systematic regression.
- xcds08_gm08 / xcds08_gm02 / xcds08_gm16 also exhibit per-run spread
  >1.7% — non-default GROUP_M values increase variance without improving
  the median.

## Resource summary (CRR kernel, identical across all 8 configs)

| Metric            | Value     |
|---|---|
| VGPRs             | 254       |
| VGPR Spill        | 0         |
| TotalSGPRs        | 52        |
| Scratch (B/lane)  | 0         |
| LDS (B/block)     | 131072    |
| Occupancy (w/SIMD)| 2         |

V2 RRR fastpath (line 263) and MX decode kernel (line 3932) are also bit-
identical across configs (254 VGPR / 254 VGPR / 212 VGPR respectively), as
expected — the swizzle macros are CRR-host-only.

## Reasoning

The XCD block swizzle is a host-side block-index remap that controls which
CTA goes to which XCD/CU; on MI355X (8 XCDs × 38 CUs = 304 CUs) for this
shape (M=4096, N=28672, K=8192 → 64×112 = 7168 CTAs at 128-tile, ≈23.6
waves/CU), the swizzle's role is to balance XCD-local L2 reuse against
inter-XCD HBM bank conflicts.

The fact that the **entire 8-config sweep clusters within ±0.4%** of default
indicates that at this shape:

1. The CTA count (7168) is large enough that any reasonable swizzle achieves
   near-perfect XCD load balance — the granularity tuple is below the
   noise floor.
2. The L2 reuse pattern is dominated by intra-CTA K-loop streaming, not
   inter-CTA neighbour reuse — so GROUP_M variation does not move the
   needle.
3. The CRR ~88-89% MX/FP8 ratio at this cell is bottlenecked elsewhere
   (scale-shift critical path per R49A/R50A/R53A/R54A/R54E closure family),
   not in the XCD scheduler.

This is the **6th REFUTED axis on the 70B Gate/Up CRR HEADROOM-recovery
search**, complementing:
1. R49A — host-side scale repack (REFUTED-EMPIRICAL, v_perm penalty)
2. R50A — lead-distance / inter-iter live state (REFUTED, VGPR spill)
3. R53A — opsel-keyed K-phase MMA dispatch (REFUTED, VGPR spill + body dup)
4. R54A — LDS-resident pre-shifted scale layout (REFUTED-EMPIRICAL, LDS overflow)
5. R54E — `v_pk_lshrrev_b32` packed shift (REFUTED-EMPIRICAL-ISA, opcode absent)
6. **R54F — XCD swizzle granularity tuning (REFUTED-EMPIRICAL, no slack)** ← this work

The CRR scale-shift critical-path family is now closed across 5 axes (R49A
through R54E) and the orthogonal swizzle/scheduling axis is also closed
(R54F). Subsequent CRR HEADROOM work should pivot to:
- VMEM/LDS pipelining (R32-class) — orthogonal to scale shift and swizzle
- Wave-tile/block recomposition at a deeper level than (NUM_XCDS, GROUP_M)
- Or accept the structural ceiling and re-prioritize toward higher-headroom cells.

## Artifacts

- `r54f_bench.sh` — Phase 2 strict-SCLK 5-run sweep driver
- `r54f_isa_verify.sh` — per-config CRR-kernel resource extractor (Phase 1)
- `r54f_bench.run.log` — full Phase 2 run transcript + summary table
- `r54f_results/{tag}_build.log` × 8 — build resource remarks per config
- `r54f_results/{tag}_run{1..5}.log` × 40 — per-run TFLOPS measurements
- `r54f_results/isa/` — empty (resource verification sourced from build logs)
- `r54f_workspace/` — build symlinks + .so + JSON results

No source modifications. No `MXFP8_CRR_BLOCK_SWIZZLE_*` default change
recommended — current shipped (NUM_XCDS=8, GROUP_M=4) is empirically optimal
within the swept neighbourhood at this cell.

## Verdict line for cycle wrap

`R54 Dev F: 70B Gate/Up CRR XCD swizzle granularity tuning — REFUTED-EMPIRICAL — best alt (xcds04_gm04) +0.35% over default (inside ±0.5% noise band); 8/8 configs cluster within ±0.4%; all builds byte-identical at 254 VGPR / 0 spill / 131072 LDS; default (NUM_XCDS=8, GROUP_M=4) sits on flat granularity plateau; 6th-axis closure of 70B Gate/Up CRR HEADROOM-recovery search.`
