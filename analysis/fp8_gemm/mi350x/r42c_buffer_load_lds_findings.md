# R42 Dev C — `buffer_load_dword_lds` VMEM→LDS direct path scoping

**Date:** 2026-04-18
**Branch:** r42-dev-c (base feat/mxfp8-only @ R41 cycle wrap, 211b0125)
**Scope:** Scoping/feasibility for the R29 Dev D recommendation #4 / R41 Dev D §2.4 flagged lever — replace the assumed `buffer_load_b128 → ds_write_b128` chain with a single `buffer_load_dword_lds` (VMEM→LDS direct) instruction in the V2-CRR/RRR/RCR B-load path.

## TL;DR — VERDICT: **REFUTED-IN-TREE / ALREADY-IMPLEMENTED** (the substitution is the production state)

The B-load (and A-load) path in V2-CRR / V2-RRR / V2-RCR **already uses `llvm.amdgcn.raw.buffer.load.lds`** through every overload of `kittens::load` invoked via `G::load`. There is **no `buffer_load_b128 → ds_write_b128` chain anywhere in the V2 dispatch path** to substitute. The R29 Dev D recommendation #4 (`r29d_lds_bank_audit.md:144`) was already implemented prior to R30 and confirmed by the R30C and R31C SASS inventories. The R41 Dev D feasibility note (`r41d_findings.md:144-149`) carried this stale recommendation forward as "the only B-side lever not yet closed" without a fresh SASS audit; the audit already exists in tree and shows zero `ds_write*` of any kind in the V2 fastpath.

**Cumulative tally:** add 1 closed lever (`buffer-load-LDS-direct already-shipped`). The R41 cumulative tally goes 42 → 43.

**No prototype built. No GPU time burned. ~3-5 GPU-hr saved.**

## 1. Toolchain availability check (gfx950)

- ROCm 7.1.0 / HIP 7.1.25424 / clang 20.0.0git (`hipcc --version`).
- Intrinsic `llvm.amdgcn.raw.buffer.load.lds` is declared in `/opt/rocm-7.1.0/include/ck_tile/core/arch/amd_buffer_addressing.hpp:1273-1281` and used pervasively in CK-tile's gfx950 path.
- gfx950 supports `bytes_per_thread ∈ {4, 12, 16}` (1, 3, 4 dwords per thread), per `amd_buffer_addressing.hpp:1793-1796` and `:2786-2792`. gfx942 only supports 1 dword. **Our path uses 16 bytes/thread** (matches `buffer_load_dwordx4 ... lds`).
- The intrinsic is also declared locally in `kernel_mxfp8_layouts.cpp` (used at `:2183` and `:2214` from `load_transpose`) and in `include/ops/warp/memory/tile/global_to_shared.cuh:63, 101, 215, 239, 308`.

**Conclusion:** intrinsic is available, supported on gfx950 with the 16B/lane granularity our tile fills require, and is not gated by any compiler/driver flag we'd need to flip.

## 2. Mechanism inspection — what does `G::load` actually emit?

`G::load` is `kittens::group<_NUM_WARPS>::load` (`kernel_mxfp8_layouts.cpp:432`) which forwards to `kittens::load<...>` (`include/ops/group/memory/tile/global_to_shared.cuh:6-43`). The four `kittens::load` overloads in `include/ops/warp/memory/tile/global_to_shared.cuh` are:

| Lines | Variant | What it emits |
|---|---|---|
| `:17-111` | `(dst, src, idx)` — base | `llvm_amdgcn_raw_buffer_load_lds` × `memcpy_per_tile` (+leftover) |
| `:187-249` | `(dst, src, idx, swizzled_offsets)` — pre-baked offsets | `llvm_amdgcn_raw_buffer_load_lds` × `memcpy_per_tile` (+leftover) |
| `:264-318` | `(dst, src, idx, swizzled_offsets, SRD, base_ptr, lds_base)` — pre-baked SRD + LDS base | `llvm_amdgcn_raw_buffer_load_lds` (16B/lane) |
| `:332-417` | `store(...)` | scalar element-wise convert + global store (no LDS path involved) |

**All four `load` overloads emit `llvm_amdgcn_raw_buffer_load_lds` directly. None of them goes VMEM→VGPR→LDS via a `buffer_load_b128 → ds_write_b128` pair.**

`grep "ds_write\|raw_buffer_store\|ds_store"` over both `global_to_shared.cuh` files returns **zero matches**.

The V2-CRR fastpath (`kernel_mxfp8_layouts.cpp:4782-4830`), V2-RRR (`:4443-4519`, with `RRR_ROW_SHARED_TRANSPOSE` further routing to the 4-dword-per-lane `load_transpose<>` form when the row-shared transpose mode is enabled), and V2-RCR (`:2367-2427`) all reach LDS through one of these four overloads.

## 3. Empirical confirmation — pre-existing SASS inventories

- `analysis/fp8_gemm/mi350x/r30c_sass_inventory.log`:
  ```
  ds_write*                       : 0
  buffer_load_dwordx4 ... lds     : 24    # tile fills (VMEM->LDS DIRECT via TK G::load)
  buffer_load_dwordx4 (no lds)    : 2     # A scale b128 -> VGPR (warmup + steady)
  buffer_load_dwordx2 (no lds)    : 2     # B scale b64  -> VGPR (warmup + steady)
  # Conclusion: 0 ds_write of any kind. Scales go VMEM->VGPR->MMA directly.
  # Tile fills already use buffer_load_lds (the lever R29 Dev D recommended).
  # No remaining buffer_load + ds_write pair to convert.
  ```
- `analysis/fp8_gemm/mi350x/r31c_sass_audit_summary.log`:
  ```
  ds_write / ds_store        : 0
  buffer_load* (total)       : 112
  buffer_load_*x4 ... lds    : 96    (TK G::load tile fills; VMEM->LDS direct)
  buffer_load_*x4|2 ... offen (no lds)   <- scale b128/b64 to VGPR
  ```

**Two independent SASS audits (R30C and R31C) confirm: zero `ds_write` instructions in the production V2 fastpath. All 96 tile-fill VMEM ops in R31C's count are already `buffer_load_*x4 ... lds`.**

The only `buffer_load_*` instructions that do NOT have the `lds` modifier are scale-pack loads (`b128`/`b64` to VGPR for the MMA opsel-selected scale lanes), which are out of scope (scales are not a tile fill).

## 4. Genealogy of the stale recommendation

1. **R29 Dev D** (`r29d_lds_bank_audit.md:144`, dated 2026-04-18): "Explicit `buffer_load_dword_lds` (VMEM→LDS direct) — gfx950 supports this, V2-CRR currently goes VMEM→VGPR→LDS. Big restructure."
2. **R30 (Dev unknown)**: implemented the substitution (likely as part of `kittens::load` rework — the four overloads in `include/ops/warp/memory/tile/global_to_shared.cuh` all use the intrinsic, and the most-likely commit predates the R30C audit).
3. **R30C SASS audit** confirmed `ds_write* = 0`, `buffer_load_dwordx4 ... lds = 24`. The recommendation was effectively closed at R30 but never struck from the R29D recommendation list.
4. **R31C SASS audit** re-confirmed at 96 LDS-direct loads.
5. **R41 Dev D** (`r41d_findings.md:144-149`): scoped the B-operand alt-LAYOUT class and explicitly excluded path changes; the §2.4 hybrid note flagged `buffer_load_dword_lds` as "out of scope … flagged as a potentially-open lever (the only B-side lever NOT closed)" — based on the R29D recommendation list, **without re-checking R30C / R31C SASS audits**.
6. **R42 Dev C (this study)**: re-checked the SASS inventories and the source. The lever is **closed in tree**.

## 5. Why no prototype is needed

The "prototype phase" of the task asks to "build a candidate .so with `buffer_load_dword_lds` substituted". The substitution is **the production state**. There is nothing to substitute *in* (the alternative path doesn't exist) and nothing to substitute *out* (no `buffer_load_b128 → ds_write_b128` chain exists in V2-CRR/RRR/RCR steady state). Building a candidate .so against `feat/mxfp8-only` HEAD would emit byte-identical assembly to the production .so at the B-load level — violating the protocol's "default 8192³ MXFP8 byte-identical unless your change is the new default" rule with no Δ to measure.

The only actionable variation in this category would be:
- **Toggle the `..., SRD, base_ptr, lds_base` overload (`:264-318`) on or off**: this is a different *invocation form* of the same intrinsic (16B/lane fixed instead of `bytes_per_thread`-templated, `to_sgpr_u32` SRD/lds-base hoisting, manual SOFF SGPR keep-in-class). This has been explored in R20-R28 and is the production form for `c_v2`-class loads when `swizzled_offsets` are pre-baked (the V2 family); whether a per-fastpath toggle to the simpler base form changes ISA scheduling is a different, much smaller-EV question — and not what either R29 Dev D or R41 Dev D were proposing.
- **Async / multi-buffer prefetch chain depth**: orthogonal to the VMEM→LDS instruction choice; partially explored under R26 Dev B, and would not be a "buffer_load_dword_lds substitution".

## 6. Performance implications

The R41 Dev D estimate ("5–15% lift if VMEM dispatch is binding") is moot: the optimization is the production state. The 11-cycle 70B-KV V2-CRR median 766–790 TF / 3.04% envelope (R41 Reviewer Phase 1) reflects this configuration. Any 5–15% lift R41 Dev D forecast is already baked into the baseline.

## 7. Recommendation

- **Add to closed-lever tally:** `buffer-load-LDS-direct (R29 Dev D rec #4) already-shipped at R30+, confirmed by R30C/R31C SASS audits, re-confirmed by R42C source audit`. **43rd cumulative closed lever** (R41: 42 → R42: 43).
- **Strike** the R41 Dev D §2.4 / `TODO.md:142-145` priority-list entry "`buffer_load_dword_lds` (VMEM→LDS direct path)" with a back-reference to this finding.
- **Methodology gap (NEW R42+ rule, recommended):** any "lever from a prior recommendation list" should be re-audited against the most-recent SASS inventory before being scoped/prototyped. R41 Dev D should have triggered R30C/R31C re-read before forwarding the R29D recommendation. (This costs minutes, saves hours.)

## 8. References

- `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp:432` (`using G = kittens::group<_NUM_WARPS>;`)
- `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp:2141-2225` (`load_transpose<>` — the 4-dword-per-lane explicit-intrinsic form, used by `RRR_ROW_SHARED_TRANSPOSE`)
- `include/ops/group/memory/tile/global_to_shared.cuh:6-43` (`G::load` forwarders)
- `include/ops/warp/memory/tile/global_to_shared.cuh:17-318` (four `kittens::load` overloads, all emit `llvm_amdgcn_raw_buffer_load_lds`)
- `analysis/fp8_gemm/mi350x/r29d_lds_bank_audit.md:144` (original R29 Dev D recommendation #4)
- `analysis/fp8_gemm/mi350x/r30c_sass_inventory.log` (post-R30 SASS confirmation, ds_write=0)
- `analysis/fp8_gemm/mi350x/r31c_sass_audit_summary.log` (post-R31 SASS re-confirmation, ds_write=0, 96 LDS-direct tile fills)
- `analysis/fp8_gemm/mi350x/r41d_findings.md:144-149` (R41 Dev D § 2.4 — stale forward of R29D recommendation #4)
- `/opt/rocm-7.1.0/include/ck_tile/core/arch/amd_buffer_addressing.hpp:1273-1281, 1793-1796, 2786-2792` (intrinsic declaration + gfx950 size support)
- `TODO.md:113-118, 142-145` (R41 Dev D's flag + R42+ priority list entry)

## 9. Audit checklist

- [x] Toolchain check: ROCm 7.1, gfx950, intrinsic declared, 4-dword granularity supported.
- [x] Read all four `kittens::load` overloads in `include/ops/warp/memory/tile/global_to_shared.cuh`.
- [x] Read `G::load` forwarders in `include/ops/group/memory/tile/global_to_shared.cuh`.
- [x] Read V2-CRR/RRR/RCR B-allocation sites in `kernel_mxfp8_layouts.cpp`.
- [x] Read `load_transpose<>` (RRR row-shared explicit form).
- [x] Re-read R29 Dev D recommendation list.
- [x] Re-read R30C SASS inventory + R31C SASS audit summary.
- [x] Re-read R41 Dev D §2.4 hybrid note.
- [x] Confirmed `grep "ds_write|raw_buffer_store|ds_store"` returns 0 hits in both `global_to_shared.cuh` files.
- [x] No prototype built. No GPU time burned. Pure scoping/closure.
