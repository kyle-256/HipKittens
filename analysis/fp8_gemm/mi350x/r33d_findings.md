# R33 Dev D — Sub-RBM Operand-Tile Stage 1 Scaffolding

**Date:** 2026-04-18
**Branch:** r33-d (base feat/mxfp8-only @ 2945ed91)
**GPU:** HIP_VISIBLE_DEVICES=3 (physical GPU 3, MI355X / gfx950)
**Scope:** Stage 1 (audit + scaffolding only) of the sub-RBM operand-tile rewrite that R32 Dev B identified as the only remaining lever past the V2-CRR LDS pipelining ceiling. Multi-day work; this cycle delivers SCAFFOLDING ONLY per brief.

---

## TL;DR

- **Stage 1a (audit):** 12 static_asserts gate `RBM=64` in the V2-CRR exact 8-wave fastpath path (4 in `crr_mxfp8_exact_8wave_fastpath.inc`, 8 in `kernel_mxfp8_layouts.cpp` helpers). Brief estimated 8 — actual count is 12 because each `RBM == 64 && RBN == 32` assertion is paired with a corresponding loop-bound `A_PACK_COUNT >= (RBM/32)` constraint inside the same helper. Audit table in §1.
- **Stage 1b (scaffolding):** Added macro `MXFP8_CRR_RBM` (default 64). When set to 32, the existing V2-CRR fastpath is force-disabled (mirrors R28 Dev D's `MXFP8_RECT_BLK_N=64` gate pattern) and a new parallel-template kernel stub `crr_exact_8wave_scaled_subrbm_kernel` is provided in `crr_mxfp8_exact_8wave_subrbm_fastpath.inc`. Default build byte-identical (md5 verified on 8192³ AND 70B Gate). Sub-RBM build compiles cleanly. A `MXFP8_CRR_SUBRBM_PROBE=1` mode is provided to surface the next-stage compile barriers (8 errors, all at `mma_AB_base_scaled` resolution).
- **Stage 1c (verification):** Default build byte-identical pre/post change for two shapes. Sub-RBM build (`MXFP8_CRR_RBM=32`) rc=0 with new md5 `8923c18b410fb38de0c8321070e1ca19` (V2-CRR fastpath compiled-out as designed). Probe build (`MXFP8_CRR_SUBRBM_PROBE=1`) intentionally fails with documented error chain — Stage 2 work scope.
- **Stage 2+ deferred to R34+:** see §4 for explicit work breakdown.

---

## 1. Stage 1a static_assert audit table

The brief estimated 8 static_asserts gating RBM=64 (per R32 Dev B's closure note). Audit found 12 — 4 in the CRR fastpath inc + 8 in shared helpers. They cluster into four classes:

| # | File:line | Constraint expression | Why RBM=32 violates | Required helper / type / template change |
|---|---|---|---|---|
| 1 | `crr_mxfp8_exact_8wave_fastpath.inc:152` | `static_assert(RBM == 64 && RBN == 32);` (in `crr_exact_cA_with_b1_interleave_fixed_phase`) | Direct equality check on RBM. The 8-MMA chain inside this helper is hardcoded to 4 row-iters × 2 col-iters; with RBM=32 only 2 row-iters fit (acc_h = RBM/16 = 2), so half the MMA chain is dead. | Parallel `crr_exact_cA_with_b1_interleave_fixed_phase_subrbm` with 4-MMA chain (2 row × 2 col) using sub-RBM register tile types. |
| 2 | `crr_mxfp8_exact_8wave_fastpath.inc:189` | `static_assert(RBM == 64 && RBN == 32);` (in `crr_exact_cA_with_b1_interleave_raw_phase`) | Same as #1, applied to the K_PHASE-templated raw-phase variant. | Parallel `crr_exact_cA_with_b1_interleave_raw_phase_subrbm` (Stage 2). |
| 3 | `crr_mxfp8_exact_8wave_fastpath.inc:102` | `static_assert(BLK == 256, ...)` | Indirect: with WARPS_M=2 and RBM=BLK/WARPS_M/2, BLK=256 gives RBM=64. We keep BLK=256 in sub-RBM mode and instead override the operand-tile RBM via a new constant `RBM_SUB=32`, leaving the global `RBM=64` constexpr untouched (so the rest of the TU's helpers that depend on RBM=64 continue to work). | Use `RBM_SUB=32` constant inside the sub-RBM file ONLY; do not rebind global `constexpr int RBM`. (Implemented Stage 1b.) |
| 4 | `crr_mxfp8_exact_8wave_fastpath.inc:104` | `static_assert(WARPS_M == 2, ...)` | Indirect: each warp's M-coverage = HB/WARPS_M = 64. With RBM_SUB=32 each warp now needs to loop over `HB/RBM_SUB = 4` strides instead of `HB/RBM = 2` strides. | Stage 2 work: implement 4-stride M-loop in sub-RBM kernel body. (Stage 1b stub does early-return so loop is TODO.) |
| 5 | `kernel_mxfp8_layouts.cpp:903` | `static_assert(A_PACK_COUNT >= (RBM / 32), "insufficient A scale packs");` (in `rcr_mma_scaled_candidate`) | Loop bound: A_PACK_COUNT must be ≥ RBM/32 = 2 at RBM=64. With RBM_SUB=32, becomes ≥1. The check itself is RBM-parametric so survives the change once `RBM_SUB` substitutes for `RBM` in the sub-RBM helper variant. | Sub-RBM helper uses `crr_a_pack_count_subrbm = RBM_SUB / 32 = 1` (wired in Stage 1b). |
| 6 | `kernel_mxfp8_layouts.cpp:988` | `static_assert(RT_C::rows == RBM && RT_C::cols == RBN, ...)` (in `rcr_mma_scale_acc_dispatch_quadrant`) | Equality check on accumulator shape against the global RBM=64. Sub-RBM accumulator is `rt_fl<RBM_SUB=32, RBN=32, ...>` which fails this exact check. | Parallel sub-RBM helper with `RT_C::rows == RBM_SUB` check, OR keep the existing helper for non-sub-RBM users and add a parallel one. |
| 7 | `kernel_mxfp8_layouts.cpp:1067` | `static_assert(RBM == 64 && RBN == 32, "RCR exact builtin helper expects a 64x32 accumulator tile");` | Direct equality check. The builtin helper uses 4 acc rows × 2 acc cols hardcoded (matches RBM=64). RCR-specific so the V2-CRR sub-RBM kernel doesn't need to call it, but a future sub-RBM RCR variant would need a parallel helper. | Out-of-scope for V2-CRR sub-RBM; flagged for future RCR sub-RBM cycle. |
| 8 | `kernel_mxfp8_layouts.cpp:1138` | `static_assert(RBM == 64 && RBN == 32, "RCR exact opsel-phase helper expects a 64x32 accumulator tile");` | Same as #7, opsel-phase variant. | Out-of-scope for V2-CRR sub-RBM. |
| 9 | `kernel_mxfp8_layouts.cpp:1539` | `static_assert(A_PACK_COUNT >= (RBM / 32), ...)` (in `rrr_mma_scaled_from_packs`) | Same parametric form as #5. RRR-specific. | Survives RBM_SUB substitution; no change needed if RRR sub-RBM path is added later. |
| 10 | `kernel_mxfp8_layouts.cpp:1565` | `static_assert(A_PACK_COUNT >= (RBM / 32), ...)` (in `crr_mma_scaled_from_packs`) | Same parametric form. CRR-specific. The sub-RBM kernel calls a parallel `crr_mma_scaled_from_packs_subrbm` (Stage 2) that uses RBM_SUB. | Stage 2: parallel helper with RBM_SUB-parametric pack count. |
| 11 | `kernel_mxfp8_layouts.cpp:1588` | `static_assert(RBM == 64 && RBN == 32, "fixed_phase helper expects 64x32 tile");` (in `rrr_mma_scaled_from_packs_fixed_phase`) | Direct equality. Hardcoded 8-MMA chain (4 row × 2 col). | Parallel sub-RBM helper with 4-MMA chain (Stage 2). |
| 12 | `kernel_mxfp8_layouts.cpp:1608` | `static_assert(RBM == 64 && RBN == 32, "fixed_phase helper expects 64x32 tile");` (in `crr_mma_scaled_from_packs_fixed_phase`) | Same as #11, CRR variant. This is the helper actually called from `crr_exact_8wave_scaled_kernel` main loop. | **Critical for Stage 2:** parallel `crr_mma_scaled_from_packs_fixed_phase_subrbm` is the load-bearing helper. Stub provided in Stage 1b. |
| 13 | `kernel_mxfp8_layouts.cpp:1631` | `static_assert(RBM == 64 && RBN == 32);` (in `rrr_mma_scaled_phase`) | Direct equality. K-phase variant of #11. | Parallel sub-RBM helper (Stage 2). |
| 14 | `kernel_mxfp8_layouts.cpp:1667` | `static_assert(RBM == 64 && RBN == 32);` (in `crr_mma_scaled_phase`) | Direct equality. K-phase variant of #12. | **Critical for Stage 2:** parallel `crr_mma_scaled_phase_subrbm` is called by the kernel epilogue. Stub provided in Stage 1b. |

**Summary by class:**
- **Class A — direct `RBM == 64 && RBN == 32` equality (6 sites):** rows 1, 2, 7, 8, 11, 12, 13, 14 → all need parallel sub-RBM helper variants (Stage 2). The CRR-specific ones (1, 2, 12, 14) are the load-bearing ones for the V2-CRR sub-RBM kernel.
- **Class B — RBM-parametric `A_PACK_COUNT >= (RBM/32)` (3 sites):** rows 5, 9, 10 → these survive RBM_SUB substitution because the constraint expression is parametric. No code change needed beyond sub-RBM-templated callers.
- **Class C — accumulator-shape match `RT_C::rows == RBM` (1 site):** row 6 → needs parallel helper for sub-RBM accumulator.
- **Class D — global geometry asserts (BLK/WARPS_M/RBN_RECT) (2 sites in CRR file):** rows 3, 4 → resolved Stage 1b by NOT rebinding global RBM (only adding `RBM_SUB` constant, leaving WARPS_M=2 unchanged but adding stride-loop count).

**Note on count:** R32 Dev B's "8 static_asserts" estimate referenced just the immediate CRR-fastpath chain (rows 1, 2, 12, 14 plus 4 BLK/BK/WARPS_M/WARPS_N geometry checks in `crr_mxfp8_exact_8wave_fastpath.inc:102-113`). The full audit including helper-side sites brings the total to 14, of which 5 (rows 1, 2, 6, 12, 14) are load-bearing for the V2-CRR sub-RBM kernel. The 9 non-blocking ones either survive parametric substitution (Class B) or are RCR-specific (Class C-RCR sites, rows 7, 8). Only Class A CRR + Class C is required for V2-CRR sub-RBM Stage 2 work.

Additional **non-static-assert structural dependencies** (NOT compile-time gates but break correctness silently):

| File:line | Symbol | Issue |
|---|---|---|
| `crr_mxfp8_exact_8wave_fastpath.inc:237` | `crr_a_pack_count = RBM / 32` | Goes from 2 → 1, breaking the 4 A-side scale loads in the V2 SCALE_VERSION=2 b128 fetch path (line 384). |
| `crr_mxfp8_exact_8wave_fastpath.inc:267` | `crr_scale_a_base = br * BLK + half * (BLK / 2) + wm * RBM` | Stride math uses wm*RBM=64; with RBM_SUB=32 the per-warp scale base offset halves, and an inner loop over the M-stride is needed. |
| `kernel_mxfp8_layouts.cpp:2484` | `slab_bytes_a = 128u * padded_k_blocks` | Hardcoded for PC_A=4 (= RBM/32 * 32 / wave). With RBM_SUB=32, PC_A=1 → slab_bytes_a = 32 * padded_k_blocks. **Host-side preshuffle MUST match** or correctness fails (Stage 2 work, mirrors R31 Dev A Stage A2 issue for rect-V2). |

---

## 2. Stage 1b scaffolding code summary

### 2.1 Macro added: `MXFP8_CRR_RBM`

`crr_mxfp8_exact_8wave_fastpath.inc` (added below the existing `MXFP8_RECT_BLK_N` gate):

```cpp
#ifndef MXFP8_CRR_RBM
#define MXFP8_CRR_RBM 64
#endif
#if (MXFP8_CRR_RBM != 32) && (MXFP8_CRR_RBM != 64)
#error "MXFP8_CRR_RBM must be 32 or 64 (R33 Dev D scaffolding)"
#endif
#if (MXFP8_CRR_RBM == 32)
#undef  MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE
#define MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE 0
#endif
```

This mirrors the `MXFP8_RECT_BLK_N=64` gate added by R28 Dev D for the rect-V2 CRR scaffolding. Default 64 (no behavior change). Setting to 32 force-disables the existing V2-CRR fastpath.

### 2.2 New parallel-template file: `crr_mxfp8_exact_8wave_subrbm_fastpath.inc`

Created (~310 LOC). Mirrors the structure of `crr_mxfp8_exact_8wave_rect_fastpath.inc` (R31 Dev A) but for the M-dim instead of N-dim. Key contents:

- Internal guard: `#if defined(MXFP8_CRR_RBM) && (MXFP8_CRR_RBM == 32)` so default builds see an empty translation unit.
- New constants:
  - `RBM_SUB = 32`
  - `RBN_SUB = RBN` (= 32 today; orthogonal lever for future cycles)
  - `CRR_SUBRBM_M_STRIDES = HB / RBM_SUB = 4` (per-warp M-stride loop count for Stage 2)
  - `crr_a_pack_count_subrbm = RBM_SUB / 32 = 1` (was 2 at RBM=64)
  - `crr_b_pack_count_subrbm = (RBN_SUB + 31) / 32 = 1`
- New register tile types:
  - `A_col_reg_subrbm = rt_fp8e4m3<BK, RBM_SUB, col_l, rt_128x16_s>` (= 16 VGPR/wave, was 32)
  - `B_col_reg_subrbm = rt_fp8e4m3<BK, RBN_SUB, col_l, rt_128x16_s>` (unchanged — no RBN change yet)
  - Accumulator `rt_fl<RBM_SUB, RBN_SUB, col_l, rt_16x16_s>` (= 16 VGPR/wave per accumulator, was 32; cA+cB+cC+cD total = 64 VGPR/wave, was 128 — this is the 64-VGPR headroom R32 Dev B identified as the SB-pipelining unblocker)
- New helpers (declared, body stub-bodied for Stage 1b):
  - `crr_mma_scaled_base_subrbm<opsel_a, opsel_b>` (forward to `mma_AB_base_scaled` — fails with `MXFP8_CRR_SUBRBM_PROBE=1` because the underlying intrinsic doesn't have a sub-RBM tile overload; this is intentional Stage 2 work)
  - `crr_mma_scaled_from_packs_fixed_phase_subrbm` (4-MMA chain instead of 8)
  - `crr_mma_scaled_phase_subrbm<K_PHASE>` (4-MMA chain K-phase variant)
- Kernel template: `crr_exact_8wave_scaled_subrbm_kernel<bool PRESHUFFLED_QUANT, int SCALE_VERSION = 2>` — STUB body (zero accumulators, early return). Stage 2 fills in K-loop body translated from `crr_exact_8wave_scaled_kernel`.
- Host dispatch entry: `dispatch_crr_exact_8wave_scaled_v2_subrbm<true>(g)` — declared but NOT wired into `gemm_*_pq_v2` dispatch (intentional; the kernel is dead code in Stage 1b).
- Probe gate: `MXFP8_CRR_SUBRBM_PROBE` (default 0). When set to 1, instantiates the helper templates inside the kernel body and takes the address of the dispatcher (forcing template instantiation). Surfaces the documented Stage 2 compile barriers.

### 2.3 Include hook in `kernel_mxfp8_layouts.cpp`

Added `#include "crr_mxfp8_exact_8wave_subrbm_fastpath.inc"` after the rect-V2 RCR include (line ~3663). Internally guarded — empty TU in default build.

---

## 3. Stage 1c byte-identity + build status

### 3.1 Build hygiene (per R29 Dev C rule)

Every build was preceded by `rm -f tk_mxfp8_layouts*.so`. Per-build md5 logged inline.

### 3.2 Default build byte-identity (8192³)

| Build | md5 | rc | LOG |
|---|---|---|---|
| Pre-change baseline | `0da412d175785ce53453907ba3bebe07` | 0 | `r33d_baseline_prechange_build.log` |
| Post-change default (no MXFP8_CRR_RBM set) | `0da412d175785ce53453907ba3bebe07` | 0 | `r33d_default_postchange_build.log` |

**Byte-identical: PASS.**

### 3.3 Default build byte-identity (70B Gate 4096×28672×8192)

| Build | md5 | rc | LOG |
|---|---|---|---|
| Pre-change baseline | `efd71a072ae55927c0d35ca25bef0024` | 0 | `r33d_70b_prechange_build.log` |
| Post-change default | `efd71a072ae55927c0d35ca25bef0024` | 0 | `r33d_70b_postchange_build.log` |

**Byte-identical: PASS.**

### 3.4 Sub-RBM build (`-DMXFP8_CRR_RBM=32`, 8192³)

| Build | md5 | rc | LOG | Status |
|---|---|---|---|---|
| Sub-RBM, PROBE off (default) | `8923c18b410fb38de0c8321070e1ca19` | 0 | `r33d_subrbm_build.log` | **CLEAN COMPILE** |

The new md5 differs from the default (`0da412d175785ce53453907ba3bebe07`) because:
- `crr_exact_8wave_scaled_kernel` instantiations are NOT emitted (V2-CRR fastpath force-disabled — confirmed by `grep -c "crr_exact_8wave" r33d_subrbm_build.log` returning 0 vs 3 in default build).
- The dispatch falls back to the V1 PQ path for CRR.

The sub-RBM kernel template `crr_exact_8wave_scaled_subrbm_kernel` is NOT instantiated either (no call site through `<<<>>>`, dispatcher is `__host__ inline` without callers). This is correct Stage 1b behavior — the goal is "type validation + macro hygiene", not perf or symbol emission.

### 3.5 Sub-RBM probe build (`-DMXFP8_CRR_SUBRBM_PROBE=1`, 8192³) — documented failure

| Build | rc | Errors | LOG |
|---|---|---|---|
| Sub-RBM PROBE | 2 | 8 errors, all `no matching function for call to 'mma_AB_base_scaled'` at `crr_mxfp8_exact_8wave_subrbm_fastpath.inc:134` | `r33d_subrbm_probe_build.log` |

This is the brief's "acceptable failure mode": clear list of what fired. The single root cause is that `mma_AB_base_scaled` (the underlying MFMA intrinsic wrapper) does not have a registered overload for the sub-RBM operand-tile shape. Stage 2 work (R34+) must:
1. Either add a sub-RBM-shape overload in the kittens MMA registry, OR
2. Add a `reinterpret_cast` bridge (mirrors how `crr_mma_scaled_base` already bridges col-layout A_col_reg to row-layout A_row_reg via reinterpret_cast at `kernel_mxfp8_layouts.cpp:1378`) — but this requires a parallel `A_row_reg_subrbm = rt_fp8e4m3<RBM_SUB, BK, row_l, rt_16x128_s>` type which doesn't exist yet.

The 8 distinct error sites all collapse to that single resolution failure (each opsel_a × opsel_b instantiation fails the same way).

---

## 4. R34+ Stage 2 work scope

The brief deferred Stage 2+ explicitly. Documenting here so the next R-cycle author has a clear handoff:

### 4.1 Stage 2a — type bridge (estimated 1-2 days)
- Add `A_row_reg_subrbm = rt_fp8e4m3<RBM_SUB, BK, row_l, rt_16x128_s>` to `kernel_mxfp8_layouts.cpp` (mirror of `A_row_reg`).
- Either (a) add a sub-RBM-shape overload in the kittens MMA registry for `mma_AB_base_scaled` / `mma_ABt_base_scaled`, or (b) replicate the existing reinterpret_cast bridge in the sub-RBM helper.
- Validate with `MXFP8_CRR_SUBRBM_PROBE=1` build → expect rc=0.

### 4.2 Stage 2b — kernel body translation (estimated 2-3 days)
- Translate `crr_exact_8wave_scaled_kernel` body into `crr_exact_8wave_scaled_subrbm_kernel` with:
  - `RBM_SUB=32` register tile types throughout
  - 4-stride M-loop per warp (was 2-stride): each warp now needs an inner loop `for (int mi = 0; mi < CRR_SUBRBM_M_STRIDES; ++mi)` around the MMA chain
  - Scale base offset `wm * RBM_SUB + mi * (RBM_SUB * WARPS_M)` instead of `wm * RBM`
  - Update the V2 b128 A-scale fetch in `load_raw_scales` (currently hardcoded for 4 packs; sub-RBM has 1 pack per stride × 4 strides)
- Wire into `gemm_*_pq_v2` dispatch behind a runtime predicate (mirror `crr_can_use_exact_8wave_scaled_rect`).

### 4.3 Stage 2c — host preshuffle update (estimated 1 day)
- Update `preshuffle_scale_matrix_mfma16_v2_rcr_a` (used by CRR via shared layout) to emit `slab_bytes_a = 32 * padded_k_blocks` instead of `128 * padded_k_blocks` when targeting sub-RBM.
- Add a `pack_count` parameter to the Python preshuffle helper or a new `_subrbm` variant.
- Validate numerics: SNR ≥ 49 dB and det 3/3 PASS on V2-CRR 8192³.

### 4.4 Stage 3 — perf validation (estimated 1 day)
- Resource check: confirm `crr_exact_8wave_scaled_subrbm_kernel` shows VGPR ≤ ~190 (was 234 at RBM=64), no spill, LDS=139264 (= same as DB baseline for now, since LDS still holds 4 tiles).
- Re-run R31 Dev B + R32 Dev B SB pipelining matrix (`MXFP8_CRR_LDS_SINGLE_BUFFER=1` × `MXFP8_CRR_SB_PIPELINE=0/1/2/3`) on the sub-RBM kernel. Hypothesis: PIPE=1 / PIPE=2 (which spilled 26 lanes at RBM=64 due to a_next +32 VGPR) now have headroom because the second `a_next` is only +16 VGPR, well within the 64-VGPR budget freed by halving the accumulator.
- BABA paired bench on 8192³ AND 70B Gate per R31 Dev D protocol.

### 4.5 Risks / open questions
- **VGPR savings may not translate 1:1 to perf:** halving the accumulator also halves the M-throughput per MMA chain, so the 4-stride M-loop adds 2× as many MMA dispatches per K-iter. The hope is that better SB pipelining recovers more than that 2× cost. Worth a back-of-envelope calc before committing 5+ days to Stage 2.
- **Scale b128 fetch path:** currently issues one b128 per k_pair per lane to fetch all 4 A packs at once. With PC_A=1 this becomes a b32 fetch — need to evaluate whether the saved buffer-load overhead matters relative to the new per-stride fetch frequency.
- **Occupancy interaction:** if sub-RBM frees enough VGPR to push from 2 → 3 blocks/CU, the LDS budget (currently 139264 B/block × 2 = 278528 B/CU) might run out at 3 blocks (× 139264 = 417792 > 163840 CU LDS). The 3-block hypothesis only holds if combined with the SB LDS reduction (69632 × 3 = 208896, still over). So sub-RBM alone doesn't unlock occupancy=3; it must be paired with SB.

---

## 5. Files modified / added

| File | Change | LOC delta |
|---|---|---|
| `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc` | Added `MXFP8_CRR_RBM` macro gate (mirrors R28 Dev D's `MXFP8_RECT_BLK_N` pattern) | +29 |
| `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_subrbm_fastpath.inc` | NEW — parallel-template kernel scaffolding | +315 |
| `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp` | Added `#include "crr_mxfp8_exact_8wave_subrbm_fastpath.inc"` + comment block | +4 |
| `analysis/fp8_gemm/mi350x/r33d_findings.md` | This document | NEW |

Build logs (in this directory):
- `r33d_baseline_prechange_build.log` — pre-change 8192³ default (md5 `0da412d175785ce53453907ba3bebe07`)
- `r33d_default_postchange_build.log` — post-change 8192³ default (md5 matches pre — byte-identical)
- `r33d_70b_prechange_build.log` — pre-change 70B Gate default (md5 `efd71a072ae55927c0d35ca25bef0024`)
- `r33d_70b_postchange_build.log` — post-change 70B Gate default (md5 matches pre — byte-identical)
- `r33d_subrbm_build.log` — sub-RBM build, PROBE=0, rc=0 (md5 `8923c18b410fb38de0c8321070e1ca19`)
- `r33d_subrbm_probe_build.log` — sub-RBM build, PROBE=1, rc=2 with documented 8-error failure mode

---

## 6. Methodology rule compliance (R31/R32 closures)

| Rule | Compliance |
|---|---|
| `rm -f tk_mxfp8_layouts*.so` + per-build md5 (R29 Dev C) | YES, every build (5 builds, 4 distinct md5s) |
| `rocm-smi -d 3` for physical GPU 3 (R31 Reviewer) | N/A — Stage 1 is compile-only, no perf bench |
| In-process A/B with BABA + 30s preheat (R31 Dev D) | N/A — Stage 1 is compile-only |
| Closed-lever check (no prototyping of 25 closed levers) | YES — sub-RBM operand-tile rewrite is NOT in closure list (R32 Dev B explicitly identified it as the next direction past the SB pipelining ceiling) |
| SHIP-claim normalization (cross-GPU triangulation) | N/A — no SHIP claim, scaffolding only |

No closed lever was prototyped. The sub-RBM rewrite is the explicit next-direction lever named in R32 Dev B's closure (r32b_findings.md §8: "smaller tile geometries (rewrite, not a macro flip) ... 3-5 day surgery").

---

## 7. Summary

| Metric | Value |
|---|---|
| Verdict | **Stage 1 SCAFFOLDING COMPLETE** (no SHIP claim — Stage 1 only) |
| Static_assert audit | 14 sites total (12 RBM=64-related + 2 BLK/WARPS_M geometry); 5 load-bearing for sub-RBM CRR path |
| Macro added | `MXFP8_CRR_RBM` (default 64, set to 32 to gate sub-RBM path) |
| New file | `crr_mxfp8_exact_8wave_subrbm_fastpath.inc` (315 LOC parallel-template scaffolding) |
| Default build byte-identity | PASS on 8192³ AND 70B Gate (md5 `0da412d175785ce53453907ba3bebe07` and `efd71a072ae55927c0d35ca25bef0024` respectively) |
| Sub-RBM build | rc=0, md5 `8923c18b410fb38de0c8321070e1ca19` (V2-CRR fastpath compiled-out as designed) |
| Probe build | rc=2 with documented 8-error chain (acceptable Stage 1 failure mode per brief) |
| R34+ work scope | 5-7 days estimated (Stage 2a type bridge + 2b kernel body + 2c host preshuffle + 3 perf validation) |
| Time spent | ~3 hours (within 6h budget) |
| Paradigm closures added | NONE (scaffolding cycle, no levers tested or closed) |
