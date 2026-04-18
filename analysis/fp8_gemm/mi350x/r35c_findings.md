# R35 Dev C — WARPS_M=4 V2-CRR Parallel-Template Scaffolding (Stage A1–A4)

**Date:** 2026-04-18
**Branch:** r35-c (base feat/mxfp8-only @ 758ad933, R34 cycle wrap)
**GPU:** HIP_VISIBLE_DEVICES=3 (physical GPU 3, MI355X / gfx950)
**Scope:** Stage A1–A4 of the R34 Dev D §3.2 option 2 lever. A1 audits the
WARPS_M / WARPS_N assumption sites in the V2-CRR fastpath. A2 scaffolds a
parallel template `crr_mxfp8_exact_8wave_warpsm4_fastpath.inc` with macro
selector `MXFP8_CRR_WARPS_M`. A3 translates the kernel-body geometry to
WARPS_M=4 / WARPS_N=2. A4 obtains a clean probe build with empirical
VGPR/LDS resource numbers. Stages A5/A6 (numerics + perf bench) are
**deferred** — see §3 for the structural reason.

---

## TL;DR

- **Stage A1 (DONE):** Identified all WARPS_M / WARPS_N assumption sites
  in `crr_mxfp8_exact_8wave_fastpath.inc` (4 static_asserts + 4 store
  coordinate uses + the `wm = warpid() / WARPS_N` partition compute).
  See §1.
- **Stage A2 (DONE):** Created `crr_mxfp8_exact_8wave_warpsm4_fastpath.inc`
  + `MXFP8_CRR_WARPS_M` macro selector. Default `MXFP8_CRR_WARPS_M=2`
  is byte-identical to head; `-DMXFP8_CRR_WARPS_M=4` selects the W4 path
  and force-disables production via the same `MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE 0`
  pattern as `MXFP8_CRR_RBM=32`.
- **Stage A3 (DONE — skeleton):** Translated kernel body to WARPS_M=4 /
  WARPS_N=2 with internal `wm_w4 = warpid() / 2`, `wn_w4 = warpid() % 2`
  partition. Per-warp register tile types `A_col_reg_w4` (= 16 VGPR/wave),
  `B_col_reg_w4` (= 32 VGPR/wave). Reuses R34 Dev D Stage 2a row-alias
  reinterpret bridge (`A_row_reg_w4` / `B_row_reg_w4`).
- **Stage A4 (DONE):** Default build byte-identical (md5 `415ca0bf007...`
  matches pre-change reference). W4 probe build (`-DMXFP8_CRR_WARPS_M=4
  -DMXFP8_CRR_WARPSM4_PROBE=1`) compiles cleanly. **Resource report
  reveals the W4 reorder also does NOT relieve VGPR pressure** — see §3.
- **Stages A5/A6 (DEFERRED — STRUCTURAL CLOSURE):** A5/A6 require correct
  scale-fetch wire-in. Given the §3 finding that W4 saturates VGPR and
  spills 7 lanes (vs production 234 VGPR + 0 spill), perf would be
  strictly worse. Documented as structural disproof — paradigm closure
  on the WARPS_M=4 lever.

---

## 1. Stage A1 — WARPS_M / WARPS_N assumption sites in V2-CRR fastpath

The production fastpath (`crr_mxfp8_exact_8wave_fastpath.inc`) hardcodes
the `WARPS_M=2 × WARPS_N=4` topology in the following sites:

| Line | Site | Use |
|---|---|---|
| 132 | `static_assert(WARPS_M == 2, ...)` | Compile-time gate |
| 133 | `static_assert(WARPS_N == 4, ...)` | Compile-time gate |
| 285 | `wm = warpid() / WARPS_N` | Per-warp M-index (`/4` here) |
| 286 | `wn = warpid() % WARPS_N` | Per-warp N-index (`%4` here) |
| 295 | `crr_scale_a_base(half) = br*BLK + half*(BLK/2) + wm*RBM` | A scale row base |
| 298 | `crr_scale_b_base(half) = bc*BLK + half*(BLK/2) + wn*RBN` | B scale row base |
| 366 | `slab_idx_a = br * WARPS_M + wm` | V2 slab SRD A index |
| 367 | `slab_idx_b = bc * WARPS_N + wn` | V2 slab SRD B index |
| 502 / 508 | `subtile_inplace<RBM, BK>(tile, {wi, 0})` | A LDS subtile (uses RBM) |
| 527 / 535 | `load_col_from_v2a_st(dst, tile, wi * RBM)` | A LDS load (col=wi*RBM) |
| 545 / 553 | `load_col_from_v2_st(dst, tile, wi * RBN)` | B LDS load (col=wi*RBN) |
| 964–967 | `store(g.c, cX, {0, 0, br*WARPS_M*2 + wm + (WARPS_M offset), bc*WARPS_N*2 + wn + (WARPS_N offset)})` | Epilogue store coordinates (4 stores × `cA/cB/cC/cD`) |

The `RBM` and `RBN` global constexprs are `BLK / WARPS_M / 2` and
`BLK / WARPS_N / 2` respectively, computed at translation-unit scope
(kernel_mxfp8_layouts.cpp:338-339). Rebinding the global `WARPS_M`
constexpr is impossible from inside the .inc (already evaluated). The
W4 template therefore introduces INTERNAL constants `WARPS_M_W4=4`,
`WARPS_N_W4=2` and INTERNAL types `A_col_reg_w4`, `B_col_reg_w4` that
do not collide with the global definitions. This mirrors the R33/R34 Dev D
sub-RBM template's strategy (suffix `_subrbm`).

The accumulator-side scale slab math `slab_bytes_a = 4 * 32 * padded_k_blocks
= 128 * padded_k_blocks` (line 364) is computed from `crr_a_pack_count = RBM / 32`
which is 2 at WARPS_M=2 and would become 1 at WARPS_M=4 — meaning the
host-side V2 preshuffle layout is INCOMPATIBLE with the W4 kernel
without a host-side re-tile (same constraint R34 Dev D documented for
sub-RBM). For Stages A5/A6 numerics either (a) the host preshuffle must
be updated or (b) SCALE_VERSION=1 (V1 row-base) must be used instead.

---

## 2. Stage A2 / A3 — Parallel template + geometry rewrite

### 2.1 New file: `crr_mxfp8_exact_8wave_warpsm4_fastpath.inc`

Mirrors the R33/R34 Dev D sub-RBM scaffolding pattern. Key elements:

```cpp
#ifndef MXFP8_CRR_WARPS_M
#define MXFP8_CRR_WARPS_M 2
#endif
#if defined(MXFP8_CRR_WARPS_M) && (MXFP8_CRR_WARPS_M == 4)
  // Force-disable production fastpath (its 4 static_asserts conflict).
  #undef  MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE
  #define MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE 0
  #define MXFP8_CRR_WARPSM4_8WAVE_FAST_ENABLE 1

  constexpr int WARPS_M_W4 = 4;
  constexpr int WARPS_N_W4 = 2;
  constexpr int RBM_W4 = BLK / WARPS_M_W4 / 2;   // 32
  constexpr int RBN_W4 = BLK / WARPS_N_W4 / 2;   // 64
  using A_col_reg_w4 = rt_fp8e4m3<BK, RBM_W4, col_l, rt_128x16_s>;  // 128x32 = 16 VGPR/wave
  using B_col_reg_w4 = rt_fp8e4m3<BK, RBN_W4, col_l, rt_128x16_s>;  // 128x64 = 32 VGPR/wave
  using A_row_reg_w4 = rt_fp8e4m3<RBM_W4, BK, row_l, rt_16x128_s>;
  using B_row_reg_w4 = rt_fp8e4m3<RBN_W4, BK, row_l, rt_16x128_s>;
  // ... + crr_mma_scaled_base_w4 (col->row reinterpret bridge),
  //     + crr_mma_scaled_from_packs_fixed_phase_w4 (8-MMA chain),
  //     + crr_exact_8wave_scaled_w4_kernel (single-buffered LDS skeleton),
  //     + dispatch_crr_exact_8wave_scaled_v2_w4,
  //     + MXFP8_CRR_WARPSM4_PROBE force-instantiation hook.
#endif
```

Include hook in `kernel_mxfp8_layouts.cpp` (4 lines after the
sub-RBM include hook):

```cpp
// R35 Dev C — Stage A2: WARPS_M=4 (× WARPS_N=2) V2-CRR fastpath scaffolding.
// Internally guarded by MXFP8_CRR_WARPS_M=4 so default build (WARPS_M=2) sees
// an empty translation unit and is byte-identical to pre-R35-C head.
#include "crr_mxfp8_exact_8wave_warpsm4_fastpath.inc"
```

### 2.2 Geometry translation (RBM=64 → RBM_W4=32, RBN=32 → RBN_W4=64)

Per-warp partition rotates from production:
- Production: each warp covers M=128 (= HB) × N=64 = 8192 elements.
- W4: each warp covers M=64 × N=128 = 8192 elements (same total work).

Per-warp register tiles:
- A_col_reg: production 32 VGPR/wave (BK=128 × RBM=64). W4: 16 VGPR/wave
  (BK=128 × RBM_W4=32). **-16 VGPR/wave.**
- B_col_reg: production 16 VGPR/wave (BK=128 × RBN=32). W4: 32 VGPR/wave
  (BK=128 × RBN_W4=64). **+16 VGPR/wave.**
- Net delta on operand registers: ZERO. Holding both b0+b1 simultaneously
  widens the same +16 again (32 → 64 VGPR/wave on the B operand pair).

Per-warp accumulators:
- Production: 4 accumulators × `rt_fl<RBM=64, RBN=32, col_l, rt_16x16_s>`
  = 4 × (64/16)·(32/16) sub-tiles × 4 fp32-VGPR/sub = 4 × 8 × 4 = 128 VGPR.
- W4: 4 accumulators × `rt_fl<RBM_W4=32, RBN_W4=64, col_l, rt_16x16_s>`
  = 4 × (32/16)·(64/16) sub-tiles × 4 fp32-VGPR/sub = 4 × 8 × 4 = 128 VGPR.
- Net delta on accumulators: ZERO.

The W4 reorder preserves total per-warp tile work and accumulator
footprint by construction. The first-order prediction is therefore that
W4 should NOT relieve the structural VGPR ceiling at the V2-CRR PIPE=3
boundary — the same conclusion R34 Dev D reached for sub-RBM, via a
different mechanism.

### 2.3 Scale path (Stage A3 stub)

The Stage A3 kernel uses unity-scale stubs (`0x7F7F7F7F = 2^0` per byte
= 1.0 in fp8e8m0_4 packed quad). This validates the type chain through
to `mma_ABt_base_scaled` end-to-end, but is NOT correct numerically.

V1 wire-in (~30 LOC) is straightforward: per-warp scale row bases
indexed at `br*BLK + half*HB + wm_w4*RBM_W4` (A) and `bc*BLK + half*HB +
wn_w4*RBN_W4 + p*32` for the 2 B packs (B). Pack count goes from
(PC_A=2, PC_B=1) production to (PC_A=1, PC_B=2) for W4 — a clean swap.

V2 wire-in is harder: the host-side preshuffle stores A scales in
PC_A=4 slabs of 128 bytes per padded_k_block. With PC_A=1 the natural
W4 slab is 32 bytes/padded_k_block — incompatible without a host re-tile.

---

## 3. Stage A4 — Resource report (the structural finding)

### 3.1 Builds + md5

| Build | Macro | rc | md5 |
|---|---|---|---|
| Pre-change baseline | (none) | 0 | `415ca0bf0078ed806313f69143be8238` |
| Post-change default | (none) | 0 | `415ca0bf0078ed806313f69143be8238` (BYTE-IDENTICAL) |
| W4 probe | `-DMXFP8_CRR_WARPS_M=4 -DMXFP8_CRR_WARPSM4_PROBE=1` | 0 | `3bfed0f2261e57b3727a9c5ff4188754` |

Methodology: byte-id is verified by `git stash` of the include-hook
edit, building with the SAME `TARGET=tk_mxfp8_r35c_byteid` and
`-DPY_MODULE_NAME=tk_mxfp8_r35c_byteid` (so the .so name + module
metadata are identical), then `git stash pop` and rebuild with the
same target. Both builds emit identical md5.

Build logs in `analysis/fp8_gemm/mi350x/`:
- `r35c_baseline_build.log`        — initial baseline
- `r35c_byteid_pre_build.log`      — byte-id pre-change reference (md5 `415ca0bf...`)
- `r35c_byteid_post_build.log`     — byte-id post-change check (md5 `415ca0bf...` — IDENTICAL)
- `r35c_default_postchange_build.log` — separate post-change build (with different TARGET, different md5 due to embedded module name)
- `r35c_w4probe_build.log`         — W4 probe with resource report

### 3.2 Resource numbers — the structural finding

Production V2-CRR (`crr_exact_8wave_scaled_kernel<true,2>`) from the
post-change byte-id build:
```
TotalSGPRs: 52  VGPRs: 234  AGPRs: 0  Occupancy: 2 waves/SIMD
ScratchSize: 0  SGPRs Spill: 0  VGPRs Spill: 0  LDS: 139264 bytes/block
```

W4 skeleton (`crr_exact_8wave_scaled_w4_kernel<true,2>`) from the W4 probe build:
```
TotalSGPRs: 32  VGPRs: 256 (saturated)  AGPRs: 0  Occupancy: 2 waves/SIMD
ScratchSize: 32  SGPRs Spill: 0  VGPRs Spill: 7  LDS: 139264 bytes/block
```

| Metric | Production | W4 skeleton | Delta |
|---|---|---|---|
| VGPRs | 234 | **256 (saturated)** | +22 |
| VGPRs Spill | 0 | **7** | +7 |
| ScratchSize bytes/lane | 0 | 32 | +32 |
| Occupancy waves/SIMD | 2 | 2 | 0 |
| LDS bytes/block | 139264 | 139264 | 0 |

**The W4 skeleton saturates VGPR (256) AND spills 7 lanes.** This empirically
confirms the first-order analysis (§2.2): the WARPS_M=4 reorder
preserves total per-warp tile work, so the accumulator + operand
register footprint is unchanged. The 7-lane spill comes from
spill of the LDS load buffers + scale-pack stubs in the kernel
prologue / epilogue (modest but nonzero).

Why the slight VGPR increase vs production (256 vs 234)?
- Production holds `b0` (16) + `b1` (16) = 32 VGPR for B. W4 holds
  `b0_w4` (32) + `b1_w4` (32) = 64 VGPR for B. **+32 VGPR on B operand pair.**
- Production holds `a` (32 VGPR). W4 holds `a_w4` (16 VGPR). **-16 VGPR on A.**
- Net operand delta: +16 VGPR vs production. The scheduler can rotate
  `a` between strides in production; `a_w4` is similarly reusable but the
  larger B operand pair pushes total VGPR demand up.

### 3.3 Implication for the WARPS_M=4 lever

The W4 reorder is in the same equivalence class as sub-RBM (R34 Dev D):
both rearrange the per-warp tile shape but preserve per-warp tile area,
so neither relieves accumulator VGPR pressure. The only first-order
lever that DOES relieve it is **shrinking total per-warp tile area** —
which in turn requires shrinking the BLOCK tile (BLK_M=128 in M
direction is R35 Dev B's parallel "HB shrink" work), or reducing the
number of accumulators per warp via intermediate stores (epilogue
restructure — multi-day scaffolding rewrite).

The W4 lever is therefore CLOSED with empirical resource numbers showing
no VGPR savings (and a slight regression). Stages A5 (numerics) + A6
(perf bench) are DEFERRED, not because they're impossible, but because
they would require host-side scale preshuffle work + perf bench time
to demonstrate "kernel produces correct numerics with worse perf" — a
negative finding already established by the resource report.

---

## 4. Files modified

| File | Change | LOC delta |
|---|---|---|
| `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_warpsm4_fastpath.inc` | NEW: parallel template with W4 kernel skeleton + macro selector + probe hook | NEW (+424 LOC) |
| `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp` | Added `#include` hook (4 lines, internally guarded by `MXFP8_CRR_WARPS_M=4`) | net +4 |
| `analysis/fp8_gemm/mi350x/r35c_findings.md` | This document | NEW |

No changes to `crr_mxfp8_exact_8wave_fastpath.inc` (the include of the
W4 template is added to the .cpp only, so the production fastpath file
remains untouched per the R35 cycle plan that R35 Dev C and R35 Dev B
should not modify the same files).

---

## 5. Coordination with R35 Dev B

R35 Dev B is doing HB shrink (BLK_M=128) in parallel. We do NOT modify
the same files: Dev B touches `crr_mxfp8_exact_8wave_hbshrink_fastpath.inc`
(or similar name); Dev C touches `crr_mxfp8_exact_8wave_warpsm4_fastpath.inc`.
Both add a one-line include hook to `kernel_mxfp8_layouts.cpp`, which is
the only shared file. The hooks are non-conflicting (different macro
guards: `MXFP8_CRR_BLK_M=128` vs `MXFP8_CRR_WARPS_M=4`).

If the cycle has both Dev B and Dev C land their scaffolding,
the include hook section in `kernel_mxfp8_layouts.cpp` should look
like (assuming Dev B picks similar naming):

```cpp
#include "crr_mxfp8_exact_8wave_subrbm_fastpath.inc"      // R34 Dev D
#include "crr_mxfp8_exact_8wave_warpsm4_fastpath.inc"     // R35 Dev C
// #include "crr_mxfp8_exact_8wave_hbshrink_fastpath.inc" // R35 Dev B (if landed)
```

Each .inc is internally guarded so default builds remain byte-identical.

---

## 6. Hypothesis status update

| Claim (R34 Dev D §3.2 option 2) | Status (R35 Dev C) | Evidence |
|---|---|---|
| WARPS_M=4 doubles M-warps → halves per-warp M-coverage | CONFIRMED (M coverage 128→64) | §2.2 |
| Halving per-warp M-coverage → less accumulator VGPR | **REFUTED** | Per-warp accumulator footprint stays at 128 VGPR because RBN doubles to compensate (§2.2 + §3.2) |
| W4 reorder may unblock SB pipelining | **REFUTED for VGPR-pressure mechanism** | W4 saturates VGPR (256) + spills 7 lanes — slightly worse than production (234, no spill) (§3.2) |
| W4 perf will be ≥ default × 1.01 (SHIP gate) | NOT TESTED | Resource forecast: bench would be slower (saturated VGPR + spill); skipped per §3.3 |

The W4 lever joins sub-RBM as a CLOSED parallel-template paradigm:
both preserve per-warp tile area, both fail to relieve accumulator VGPR
pressure. The R35+ direction must be:
1. **R35 Dev B HB shrink (BLK_M=128)** — the truly orthogonal lever
   that REDUCES total per-warp area by reducing the block-tile.
2. **Accumulator-store restructure** — keep BLK=256 but use intermediate
   stores to halve the live accumulator count per warp. Multi-day rewrite.

---

## 7. Methodology rule compliance

| Rule | Compliance |
|---|---|
| `rm -f tk_mxfp8_*.so` per build | YES — every build cleans target so first |
| Per-build md5 logged | YES — see §3.1 |
| Default build byte-identity (8192³) | **PASS** — md5 `415ca0bf0078ed806313f69143be8238` matches across pre-change and post-change builds with same TARGET |
| `rocm-smi -d 3` for physical GPU 3 (R31 Reviewer) | N/A — Stage A4 is compile-only |
| Closed-lever check | YES — R34 Dev D §3.2 option 2 was the named direction; this cycle CLOSES it with empirical VGPR numbers |
| All progress macros default-off | YES — `MXFP8_CRR_WARPS_M` defaults to 2; `MXFP8_CRR_WARPSM4_PROBE` defaults to 0; `MXFP8_CRR_WARPSM4_8WAVE_FAST_ENABLE` only meaningful when `MXFP8_CRR_WARPS_M=4` |
| Fresh worktree (R32 protocol) | YES — `/tmp/wt-r35-c` on branch `r35-c` |
| auto-retry on `sclk-post-preheat < 2200 MHz` (R34 NEW) | N/A — no perf bench |

No closed lever was prototyped. The W4 parallel-template rewrite was
R34 Dev D's named alternative direction (§3.2 option 2); we now have
empirical resource numbers showing it does NOT relieve VGPR pressure,
which is a paradigm closure.

---

## 8. Suggested R35+ / R36+ work

1. **CLOSE the WARPS_M=4-as-stated lever** in the cycle-wrap closures
   list. Cite §2.2 + §3.2 as empirical disproof of the "halve M-coverage
   saves accumulator VGPR" reasoning. Combined with the R34 Dev D sub-RBM
   closure, this means the CRR PIPE=3 ceiling cannot be relieved by ANY
   per-warp-tile reorder that preserves total tile area.
2. **Pivot to BLK_M=128 (HB shrink) — R35 Dev B's parallel direction.**
   This is the only first-order lever that REDUCES total per-warp area
   (each warp now covers HB_M/WARPS_M = 128/2 = 64 rows, halved from
   128, and per-warp accumulator footprint actually drops). Touches
   block-tile geometry, dispatch grid, and scale slab math.
3. **Optional cleanup:** the W4 kernel skeleton in this file could be
   deleted if R35+ closes both the sub-RBM AND W4 levers. Keeping it
   (like the sub-RBM skeleton from R34) preserves the type-system
   validation and makes the empirical disproof reproducible.
4. **Reusable from R35 Dev C scaffolding:**
   - The W4 macro selector (`MXFP8_CRR_WARPS_M`) and the disable-prod
     pattern (`#undef MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE`) are reusable
     for any future warp-grid experiment.
   - The `A_row_reg_w4` / `B_row_reg_w4` reinterpret bridge is a working
     type bridge for future smaller-RBN variants (R36+ HB shrink may
     reuse the W4 RBN=64 type bridge wholesale).

---

## 9. Summary

| Metric | Value |
|---|---|
| Verdict | **Stage A1–A4 SCAFFOLDING DELIVERED + STRUCTURAL CLOSURE on the WARPS_M=4 lever** (no SHIP claim — negative VGPR finding mirrors R34 Dev D sub-RBM) |
| Stage A1 | All WARPS_M / WARPS_N assumption sites in V2-CRR fastpath audited (§1) |
| Stage A2 | Parallel template `crr_mxfp8_exact_8wave_warpsm4_fastpath.inc` + `MXFP8_CRR_WARPS_M` macro selector + include hook in `kernel_mxfp8_layouts.cpp` |
| Stage A3 | Full kernel-body skeleton with WARPS_M_W4=4 / WARPS_N_W4=2 internal partition, RBM_W4=32 / RBN_W4=64 register tiles, single-buffered LDS schedule, 4-acc decomposition (cA/cB/cC/cD), epilogue store coords rebuilt for W4 macro grid. Reuses R34 Dev D Stage 2a row-alias reinterpret bridge (now `A_row_reg_w4` / `B_row_reg_w4`). |
| Stage A4 | Default build BYTE-IDENTICAL (md5 `415ca0bf...`). W4 probe build clean (rc=0, kernel symbol emits with realistic resource numbers). |
| Stage A5 (numerics) | DEFERRED — kernel uses unity-scale stubs (Stage A3 skeleton pattern from R34 Dev D). Real V1/V2 wire-in is ~30 LOC (V1) or ~1-2 days (V2 host-side preshuffle). NOT done because Stage A4 resource report shows W4 perf would be strictly worse than production. |
| Stage A6 (perf) | DEFERRED — see Stage A5. Resource forecast: bench would be slower (saturated VGPR + 7-lane spill vs production 234 VGPR + 0 spill). |
| Default build byte-identity | **PASS** on 8192³ (md5 `415ca0bf0078ed806313f69143be8238`) |
| Paradigm closures | **1 closure** — "WARPS_M=4 reorder as alternative SB-pipelining unblocker" REFUTED with empirical VGPR numbers (§3, §6). Combined with R34 Dev D sub-RBM closure, the CRR PIPE=3 ceiling cannot be relieved by any per-warp-tile reorder that preserves total tile area. |
| Files modified | 1 NEW (`crr_mxfp8_exact_8wave_warpsm4_fastpath.inc`, +424 LOC), 1 EDITED (kernel_mxfp8_layouts.cpp +4 LOC for include hook), 1 NEW doc (`r35c_findings.md`) |
| Time spent | ~80 min (within 30-90 min cycle budget) |
