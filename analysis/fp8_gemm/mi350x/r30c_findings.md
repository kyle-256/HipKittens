# R30 Dev C — `buffer_load_dword_lds` Feasibility Audit for V2-CRR Scale Loads

**Date:** 2026-04-18
**Branch:** r30-c (base feat/mxfp8-only @ cd630fd8)
**GPU:** GPU2 (HIP_VISIBLE_DEVICES=2)
**Brief:** R29 Dev D recommended exploring `buffer_load_dword_lds` (VMEM→LDS direct) for V2-CRR scale loads to eliminate VGPR staging and free occupancy headroom. (`r29d_lds_bank_audit.md` §"What this means", item 4.)

**Verdict: AUDIT-ONLY (lever N/A for V2-CRR scales).**

V2-CRR scales already follow a VMEM→VGPR→MMA path with **zero LDS round-trip**. There is no `ds_write` of scale data anywhere in the V2 fastpath — neither in source nor in compiler-emitted SASS. `buffer_load_*_lds` is only useful when the destination is LDS, so the builtin has no eligible call site for scales. (Bonus discovery below: the only LDS-bound VMEM traffic in V2-CRR — A/B tile fills — **already uses `llvm_amdgcn_raw_buffer_load_lds`** via TK's `G::load`, so the lever is also maxed out for the residual LDS-bound traffic.)

---

## 1. Data-flow trace (the audit's primary value)

Source files inspected (all paths under `/tmp/wt-r30-c/`):
- `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc`
- `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp`
- `include/ops/warp/register/tile/mma.cuh`
- `include/ops/warp/memory/tile/global_to_shared.cuh`

### 1.1 Scale storage declaration (registers, not LDS)

`crr_mxfp8_exact_8wave_fastpath.inc:221-224`:

```cpp
fp8e8m0_4 a0_scale_packs[crr_a_pack_count];
fp8e8m0_4 a1_scale_packs[crr_a_pack_count];
fp8e8m0_4 b0_scale_packs[crr_b_pack_count];
fp8e8m0_4 b1_scale_packs[crr_b_pack_count];
```

These are **per-thread automatic arrays** of `fp8e8m0_4` (32-bit packed E8M0 quartet). They live in **VGPRs** (or scratch on spill, but no spill here — see VGPR/spill table below). There is no `__shared__` decoration anywhere in the V2-CRR scale path. (Compare: the `MXFP8_RCR_EXACT_PQ_SCALE_LDS_ENABLE` block at `kernel_mxfp8_layouts.cpp:2192-2502` declares `__shared__ volatile uint32_t scale_stage_dwords[...]` for an experimental RCR scale-LDS staging — but it is **disabled by default** (=0) and the README at line 141 records: "Shared/LDS scale cache experiments: produced correctness problems and did not survive validation.")

### 1.2 Population (VMEM→VGPR direct, no LDS)

`crr_mxfp8_exact_8wave_fastpath.inc:316-358`, `SCALE_VERSION == 2` branch in `load_raw_scales`:

```cpp
// A side: 1× b128 fetches all 4 A packs ({a0p0, a1p0, a0p1, a1p1}).
const __uint128_t a_raw =
    llvm_amdgcn_raw_buffer_load_b128(a_v2_srsrc, a_voff, a_soff,
                                     MXFP8_CRR_V2_SCALE_CACHEPOLICY);
a0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw      ));
a1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 32));
if constexpr (crr_a_pack_count > 1) {
    a0_scale_packs[1] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 64));
    a1_scale_packs[1] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 96));
}
// B side: 1× b64 fetches both B packs ({b0p0, b1p0}).
const uint64_t b_raw =
    llvm_amdgcn_raw_buffer_load_b64(b_v2_srsrc, b_voff, b_soff,
                                    MXFP8_CRR_V2_SCALE_CACHEPOLICY);
b0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(b_raw      ));
b1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(b_raw >> 32));
```

Two intrinsics (`__builtin_amdgcn_raw_buffer_load_b128` and `_b64`), both producing scalar/vector returns straight into VGPRs. No LDS pointer is involved.

### 1.3 Consumption (VGPR→MMA scale operand)

The packs are forwarded by-value into `crr_mma_scaled_base<opsel_a, opsel_b>(...)` (`kernel_mxfp8_layouts.cpp:1243-1284`), which calls `mma_ABt_base_scaled` (`include/ops/warp/register/tile/mma.cuh:311-338`), which calls `mfma1616128_scaled` (`mma.cuh:128-150`):

```cpp
*(floatx4_t*)D = {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
    *(intx8_t*)A, *(intx8_t*)B, *(floatx4_t*)C,
    0, 0, opsel_a, opsel_b,
    /* scale_a, scale_b passed as VGPR operands */ ...)};
```

The MFMA scale operands are **VGPR inputs** to `v_mfma_scale_f32_16x16x128_f8f6f4`. There is no requirement, opportunity, or mechanism to route scales through LDS in this path.

### 1.4 End-to-end summary

```
DRAM (g.a_scale, g.b_scale)
   │
   │   buffer_load_b128 / buffer_load_b64   [VMEM, cachepolicy=auto-selected by R28 gate]
   ▼
VGPR  a0/a1/b0/b1_scale_packs[]             [no LDS round-trip]
   │
   │   passed by-value to MMA dispatch
   ▼
v_mfma_scale_f32_16x16x128_f8f6f4 scale operands  [VGPR-resident at issue]
```

**Therefore: there is no LDS load and no LDS store of V2-CRR scale data.** The path matches the R27 paradigm correction #1 ("V2 paradigm: scale-direct-to-VGPR (no LDS)") byte-for-byte.

---

## 2. SASS confirmation

Built `kernel_mxfp8_layouts.cpp` for gfx950 with `-save-temps`; isolated the V2-CRR kernel `_Z29crr_exact_8wave_scaled_kernelILb1ELi2EE...` (`PRESHUFFLED_QUANT=true, SCALE_VERSION=2`). Body: 1756 lines of SASS. Resource usage: **VGPRs=232, SGPRs=66, AGPRs=0, scratch=0, occupancy=2 waves/SIMD, LDS=139264 B/block, no register spill**.

Instruction inventory in V2-CRR kernel body:

| Instruction class | Count | Purpose |
|---|---:|---|
| `ds_write*` (any width) | **0** | — (no LDS writes from inside this kernel) |
| `ds_read_b64_tr_b8` | 144 | A/B tile column reads (V2 / V2a `load_col_from_v2_st`) |
| `ds_read*` (other widths) | 0 | — |
| `buffer_load_dwordx4 ... lds` | **24** | A/B tile fills going **VMEM→LDS direct** (already using the lever via TK `G::load`) |
| `buffer_load_dwordx4` (no lds) | 2 | Scale b128 loads → VGPR (1× warmup k_pair, 1× steady) |
| `buffer_load_dwordx2` (no lds) | 2 | Scale b64 loads → VGPR (1× warmup, 1× steady) |
| `global_load_*` / `flat_load_*` / `scratch_*` | 0 | — |

The 4 non-`lds` buffer loads' scale-offset arithmetic matches the V2 spec exactly (e.g. `s_lshl_b32 s1, s0, 10` for A and `s0, s0, 9` for B where `s0 = k_pair >> 1` — i.e. `a_soff = k_pair << 10`, `b_soff = k_pair << 9` in source). They are unambiguously the scale loads, and their destination is VGPR (`v[18:21]` and `v[172:173]`).

**Confirms the source trace: zero `ds_write` of any kind, zero LDS round-trip for scales.**

---

## 3. Why `buffer_load_dword_lds` does not apply to V2-CRR scales

`__builtin_amdgcn_raw_ptr_buffer_load_lds` (LLVM `int_amdgcn_raw_buffer_load_lds`) signature (per `/opt/rocm-7.1.0/llvm/include/llvm/IR/IntrinsicsAMDGPU.td`):

```
[],  // void return
[ rsrc(SGPR),                  // v4i32 buffer descriptor
  LDS-base-pointer(addrspace 3),  // destination (mandatory LDS)
  i32 byte_size,               // 1/2/4 (or 12/16 for gfx950)
  i32 voffset(VGPR),
  i32 soffset(SGPR/imm),
  i32 imm_offset,
  i32 cachepolicy ]
```

Two structural blockers for scale use:

1. **Destination must be addrspace(3) LDS.** V2-CRR scale registers are private (addrspace(5) → VGPR) by construction (`fp8e8m0_4 a0_scale_packs[...]`). Routing scales through LDS would require:
   - Allocating LDS for scale staging (≥ `WAVES_PER_CTA × packs/wave × 4 B = 8 × 4 × 4 B = 128 B`/k_pair minimum; scales up if double-buffered).
   - Adding a `ds_read_b32` after `s_waitcnt vmcnt(0)` to pull the scale back into a VGPR for MMA.
   - This adds **one LDS round-trip per scale** that does not exist today, and the resulting code would be strictly slower (extra `s_waitcnt lgkmcnt`, extra `ds_read` issue slot, extra LDS bank pressure) on the very same bytes the current path already gets directly.
   - This is the experimental RCR `MXFP8_RCR_EXACT_PQ_SCALE_LDS_ENABLE` path and the README explicitly states it "produced correctness problems and did not survive validation".

2. **`buffer_load_lds` byte-size is 1/2/4 universally and 12/16 only on gfx950 — `b64` (8 B) is not supported.** The B-side scale `buffer_load_b64` cannot be replaced by `buffer_load_lds` even hypothetically; it would have to split into 2× `buffer_load_dword` (4 B) which doubles VMEM issue slots. The A-side `b128` (16 B) is supported, but per (1) routing it through LDS only adds latency.

The lever is **N/A for scale loads** in the V2 paradigm.

---

## 4. Bonus discovery: the lever is already applied where it matters (tile fills)

The 24 `buffer_load_dwordx4 ... lds` instructions in V2-CRR's body are the A/B tile fills issued by ThunderKittens' `G::load(tile, g.a, ...)` and `G::load(tile, g.b, ...)`. Source: `include/ops/warp/memory/tile/global_to_shared.cuh:63, 101, 215, 239, 308`:

```cpp
llvm_amdgcn_raw_buffer_load_lds(
    srsrc,         // buffer resource
    lds_ptr,       // destination LDS pointer
    bytes_per_thread,
    swizzled_global_byte_offset,
    0, 0,
    static_cast<int>(coherency::cache_all));
```

**TK already uses VMEM→LDS direct for tile fills.** This is the only LDS-bound traffic in V2-CRR (scales bypass LDS by design). So:

- The VMEM→LDS direct lever is **already maxed out** for the only data path that benefits from it.
- There is no remaining `buffer_load + ds_write` pair anywhere in V2-CRR to convert.
- VGPR pressure attributable to "tile staging in VGPRs before LDS" is **already zero**.

This independently corroborates R29 Dev D's static LDS bank-conflict analysis (which assumed `ds_write_b128` for tile fill but found 0 conflicts); the actual fill goes VMEM→LDS without intermediate VGPR.

---

## 5. Implications for occupancy / VGPR pressure (Dev D's hoped-for win)

R29 Dev D hoped that converting scale loads to VMEM→LDS would free VGPRs and allow occupancy=3 (for R30 Dev B's parallel push). Re-examining this with the actual data flow:

| Per-thread VGPR cost of scales today | Bytes |
|---|---:|
| `a0_scale_packs[0..1]` (2× fp8e8m0_4 = 2× u32) | 8 |
| `a1_scale_packs[0..1]` (2× fp8e8m0_4) | 8 |
| `b0_scale_packs[0]` (1× fp8e8m0_4) | 4 |
| `b1_scale_packs[0]` (1× fp8e8m0_4) | 4 |
| **Total scale-pack VGPR footprint** | **24 B = 6 VGPRs** |

Total kernel VGPRs = **232**. Removing all 6 scale-pack VGPRs (impossible without breaking MMA) would yield 226 VGPRs. The occupancy bin for 8-wave CTAs on gfx950 is **256 VGPRs → 1 wave/SIMD**, **224 VGPRs → 2 waves/SIMD**. We are already at occupancy=**2** (per `-Rpass-analysis=kernel-resource-usage`). Hitting occupancy=3 needs ≤170 VGPRs (`512/3` rounded down to 4-VGPR alloc granularity → ≈168), a 64-VGPR (-27.6%) reduction. Scale-pack VGPRs are 2.6% of the budget; removing them is **not the bottleneck for occupancy=3**.

The dominant VGPR consumers in V2-CRR are: (i) the A/B `col_reg` operand registers (tile-shape × 4-byte fragments), (ii) the float accumulator `cA`/`cB`/`cC`/`cD` tiles (`RBM × RBN × float = 64 × 32 × 4 = 8192 B = 2048 VGPRs aggregate, distributed across 64 lanes = 32 VGPRs/lane per accumulator × 4 accumulators = 128 VGPRs/lane`). Even completely eliminating the scale-pack VGPRs would not unlock occupancy=3.

**Conclusion for Dev B:** the scale-load lever cannot help R30 Dev B's occupancy=3 push. The path to occupancy=3 must come from tile-shape reduction (RBM/RBN reshape, halving accumulator live-range), accumulator splitting, or 4-wave layout (smaller per-wave tile).

---

## 6. Closure recommendation (paradigm correction candidate for R30)

Add to the lever-closure list:

> **`buffer_load_dword_lds` is N/A for V2-CRR scale loads.** The V2 paradigm is scale-direct-to-VGPR by construction (R27 paradigm correction #1) — there is no LDS round-trip in source nor in SASS for any scale operand, so there is no `buffer_load + ds_write` pair to convert. Furthermore, the only LDS-bound VMEM traffic in V2-CRR (A/B tile fills via `kittens::G::load`) **already uses** `llvm_amdgcn_raw_buffer_load_lds`. Lever is **maxed out** in V2-CRR; further work on this lever requires source modification that introduces (not removes) LDS scale staging, which has been historically tried and abandoned. NEVER prototype "VMEM→LDS direct for scales" again unless the V2 paradigm itself is replaced.

This is a **dual-purpose closure**: (a) closes the R29 Dev D follow-up lever, (b) re-confirms R27 paradigm correction #1 with empirical SASS evidence (0 `ds_write*`, 0 non-tile `ds_read*`).

---

## 7. Audit methodology checklist

- [x] Identify `SCALE_VERSION==2` branch in `load_raw_scales` lambda (`crr_mxfp8_exact_8wave_fastpath.inc:316-358`).
- [x] Trace storage class of `a0/a1/b0/b1_scale_packs` (private/VGPR via `fp8e8m0_4 [...]` declaration at line 221-224).
- [x] Trace consumption path via `crr_mma_scaled_dispatch` → `crr_mma_scaled_base` → `mma_ABt_base_scaled` → `mfma1616128_scaled` → `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` (VGPR scale operand).
- [x] Search source for `__shared__` scale declarations: only `MXFP8_RCR_EXACT_PQ_SCALE_LDS_ENABLE` (RCR-only, default off, README-documented as failed experiment).
- [x] Look up `__builtin_amdgcn_raw_ptr_buffer_load_lds` signature in `IntrinsicsAMDGPU.td` (gfx950 byte-sizes 1/2/4/12/16; b64 not supported).
- [x] Build for gfx950, dump SASS for `_Z29crr_exact_8wave_scaled_kernelILb1ELi2EE...`.
- [x] Confirm: 0 `ds_write*`, 144 `ds_read_b64_tr_b8` (all tile-load), 24 `buffer_load_dwordx4 ... lds` (tile fills already VMEM→LDS direct), 4 non-`lds` `buffer_load` (the 4 scale fetches per k_pair, A b128 + B b64, both warmup and steady).
- [x] Verified: scale-load offset arithmetic in SASS (`s_lshl_b32 s1, s0, 10` for A `<<` 10 = ×1024-byte stride per k_pair; `<< 9` for B = ×512-byte stride).
- [x] Computed VGPR reduction headroom: 6 VGPRs from scale packs vs the 64-VGPR reduction needed for occupancy=3 → not the bottleneck.

---

## 8. Outcome

**AUDIT-ONLY closure.** The lever Dev D recommended (replace `raw_buffer_load_b128 + ds_write_b128` with `buffer_load_lds` for V2-CRR scales) does not exist as a code site to convert: V2 has neither `ds_write_b128` nor any other LDS write of scale data. Dev D's brief assumed an LDS round-trip that V2 does not have. The recommendation was based on the canonical "VMEM→VGPR→LDS" pattern from other GEMMs but does not apply to TK's V2 paradigm.

Bonus side-effect: confirmed that **TK `G::load` already uses `buffer_load_lds`** for tile fills, so the lever is maxed out for the only LDS-bound VMEM traffic in V2-CRR.

No bench performed (no code change to bench). No commit beyond this findings file.
