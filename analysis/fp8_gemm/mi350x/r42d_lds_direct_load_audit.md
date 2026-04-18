# R42 Dev D — Phase 3: `buffer_load_dword_lds` independent in-tree audit

**Branch**: r42-dev-d
**Date**: 2026-04-18
**Scope**: Independent verification (R42 Dev C is the lead) — search for prior in-tree usage of `buffer_load_dword_lds` / `__builtin_amdgcn_buffer_load_lds` / similar VMEM→LDS direct intrinsics. Don't duplicate Dev C's work; just establish ground truth on existing infrastructure.

## TL;DR

The intrinsic is **broadly used in tree** for A/B tile fills via the `kittens::load(ST&, GL&, COORD)` family. The current MXFP8 V2 fastpaths already invoke this path via `G::load(...)` — confirmed by call sites in `kernel_mxfp8_layouts.cpp`. **The R42 Dev C scope is narrower: applying VMEM→LDS direct to operands that currently route VMEM→VGPR→LDS** (specifically B-side scale prefetch and any non-`G::load` data movement).

## Symbol the LLVM compiler emits

The HIP intrinsic of interest is `llvm.amdgcn.raw.buffer.load.lds` (LLVM IR name) → emits `buffer_load_dword_lds` (or 96-bit / 128-bit variants depending on `bytes_per_thread`) on gfx9 ISA. The C-side declaration in tree (single source of truth):

```
include/ops/warp/memory/util/util.cuh:117-124
  extern "C" __device__ void
  llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,
                                  as3_uint32_ptr lds_ptr,
                                  int size,
                                  int voffset,
                                  int soffset,
                                  int offset,
                                  int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");
```

Argument `size` is bytes-per-lane (4 for `dword`, 16 for the 128-bit variant if available; current call sites pass 4 or 16).

## Call-site inventory (in-tree)

7 distinct call sites across 5 files:

| file | lines | context |
|--|--|--|
| `include/ops/warp/memory/util/util.cuh`           | 117-124   | declaration only |
| `include/ops/warp/memory/vec/global_to_shared.cuh` | 52, 76    | shared **vector** load (`kittens::load(SV&, GL&, ...)`) |
| `include/ops/warp/memory/tile/global_to_shared.cuh`| 63, 101, 215, 239, 308 | shared **tile** load — primary A/B fill path for all V2 fastpaths via `G::load` |
| `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`  | 707, 738  | FP8 reference kernel — directly invokes the intrinsic, mirror of the tile helper |
| `kernels/gemm/fp8fp32/FP8_4wave/utils.cpp`         | 88        | FP8 4-wave kernel — directly invokes the intrinsic via a `prefill_*` helper |
| `kernels/gemm/bf16fp32/archive/utils.cpp`          | 263, 342  | archived bf16 kernel — historical reference |

## How MXFP8 V2 currently uses the intrinsic

V2-CRR / V2-RCR / V2-RRR fastpaths fill the A and B operand shared tiles via
`G::load(Xs[...], g.x, coord, swizzled_offsets_x)` calls (templated `G::load`
resolves to the `tile/global_to_shared.cuh:215` overload, which directly issues
`llvm_amdgcn_raw_buffer_load_lds` with `bytes_per_thread = ST::underlying_subtile_bytes_per_thread`).

Sample call sites in `kernel_mxfp8_layouts.cpp` (V2-CRR and V2-RCR):
- 2956-2970 (V2-RCR steady-state K-loop A/B fills, double-buffered)
- 3785-3818 (V2-CRR pre-load chain via `b_co(...)` / `a_co(...)` coord helpers)
- 3891-3900 (V2-CRR sub-RBM scaffold, same lift via `b_tile(...)` helper)

So **A and B operand tile fills are already on the VMEM→LDS direct path.** The R30 Dev C audit (TODO.md:1002) identified this and documented:

> `buffer_load_dword_lds` is N/A for V2-CRR scales (Dev C): V2 paradigm is
> scale-direct-to-VGPR with zero LDS round-trip; no `buffer_load + ds_write`
> pair exists to convert. Tile fills already use the lever via TK `G::load`.

(R30 Dev C closure entry, item 12 in the cumulative paradigm-corrections list.)

## What's left for R42 Dev C to scope

**The R42+ open lever is NOT for tile fills (already done) and NOT for V2 scales (closed in R30 Dev C).** Per R41 Dev D `r41d_findings.md` and R30 Dev C closure rationale, the remaining lever is **B-side scale prefetch**:

- Currently V2-CRR / V2-RRR scale loads go through inline ld+st pairs (e.g. `llvm_amdgcn_raw_buffer_load_b128(...)` at lines 2781, 3260, 3306) that feed VGPR registers directly. Some scale paths go VMEM→VGPR (no LDS).
- The R29 Dev D recommendation flagged for R42+ (TODO.md:116) is a "path change" — **convert any remaining VMEM→VGPR→LDS chains** (if any exist for B-operand tile data outside the `G::load` helpers) **to a single `buffer_load_dword_lds`**.
- After R30 Dev C's audit, the only candidates left are **non-`G::load` B-side movements** (e.g., manual `prefill_transpose_swizzled_offsets` paths in V2-CRR transpose bridge, or any scale-fetch path NOT currently going through the LDS-direct intrinsic).

**Verification recommendation for Dev C**: grep for `llvm_amdgcn_raw_buffer_load_b128` and `llvm_amdgcn_raw_buffer_load_b64` invocations in `kernel_mxfp8_layouts.cpp` that pair with a subsequent `ds_write` or `move<...>::sts` — those are the conversion candidates.

## Cross-arch note (gfx950 / MI355X confirmation)

The intrinsic is supported on gfx950. Existing `make` builds compile and link
this code path on the project's MI355X CI (R28-R41 production .so all use
`G::load` → `llvm_amdgcn_raw_buffer_load_lds`). No new arch-enable flag
required for R42 Dev C exploration.

## "c2k" / external ROCm ASM check

No `c2k` directory or symbol present in tree. Searched:
- `kernels/`     — no `c2k` matches
- `include/`     — no `c2k` matches
- `analysis/`    — no `c2k` matches

(The user's mention of "c2k" likely referred to a ROCm/AMD external sample
not vendored into this repo. The in-tree ground truth is the
`include/ops/warp/memory/{tile,vec}/global_to_shared.cuh` helpers + the
`util.cuh` intrinsic declaration.)

## Recommendations (independent of Dev C, for R42+)

1. **Dev C's prototype scope is correctly narrow**: focus on B-side scale paths or any non-`G::load` VMEM-staged data, NOT on retrofitting tile fills (already on LDS-direct).
2. **R30 Dev C's "N/A for V2 scales" closure stands**: scale loads route VMEM→VGPR by design (no LDS round-trip exists to bypass). Don't re-prototype this branch.
3. **Profiling first**: per R30 highest-EV recommendation, verify VMEM dispatch is the binding resource for 8B Up V2-RRR boundary-lock cell (4096×14336×4096) BEFORE committing to a prototype. Use `rocprof --hsa-trace` or VMEM-stall counters.
4. **Compatibility note**: if Dev C extends the `tile/global_to_shared.cuh` template family to a new size (e.g. `bytes_per_thread = 8` for FP8 byte-level loads), verify the LLVM intrinsic accepts `size = 8` on gfx950 — the existing call sites use 4 or 16.

## Time-box

This audit: ~25 min wall (grep + cross-reference + writeup). Zero GPU usage.
Independent of Dev C — does not duplicate any prototype work.
