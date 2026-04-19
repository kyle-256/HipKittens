# R38 Opt A — Inline-asm `buffer_load_dwordx4 ... lds` (2026-04-19)

## Hypothesis

The 19 R37 WRONG_OUTPUT shapes failed because their BEST_VARIANTS flag stack
contains scheduler-aggressive LLVM flags (or implicit clang scheduling) that
reorder `__builtin_amdgcn_raw_buffer_load_lds` calls across K-iteration
boundaries, despite R37 stripping `-mllvm -amdgpu-sched-strategy=
max-memory-clause`. Replace the intrinsic call inside `emit_one_pf` with an
`asm volatile("buffer_load_dwordx4 ... offen lds")` block carrying a
`"memory"` clobber so the compiler cannot reorder it regardless of any
`-mllvm` flag.

## Implementation

### Source change
`kernel_mxfp4_gluon_cpp.cpp` `emit_one_pf` (line ~903):
```cpp
__device__ __forceinline__ void emit_one_pf(const tile_pf_params& p, int idx) {
#if R38A_INLINE_BUFLOAD_LDS
    uint32_t lds_addr = p.lds_addrs[idx];
    uint32_t voff = p.voffs[idx];
    asm volatile(
        "s_mov_b32 m0, %0\n"
        "buffer_load_dwordx4 %1, %2, %3 offen lds\n"
        :
        : "s"(lds_addr), "v"(voff), "s"(p.srd), "s"(p.soff)
        : "memory"
    );
#else
    llvm_amdgcn_raw_buffer_load_lds(...);  // original intrinsic
#endif
}
```

Macro `R38A_INLINE_BUFLOAD_LDS` defaults to OFF (does not break R37 WIN
modules). Builder `build_R38A.py` adds `-DR38A_INLINE_BUFLOAD_LDS=1` and
strips `-mllvm -amdgpu-sched-strategy=max-memory-clause` (R37 baseline).

### Verification of inline-asm emission

Disassembled `tk_..._kx32768_btw_all_R38A.so` via `roc-obj-extract` +
`llvm-objdump --mcpu=gfx950`:
```
s_mov_b32 m0, s38
buffer_load_dwordx4 v68, s[20:23], s27 offen lds   // E05D1000 1B050044
s_mov_b32 m0, s39
buffer_load_dwordx4 v80, s[20:23], s27 offen lds
...
```
Inline asm IS emitted with the expected `... offen lds` form. Encoding
`E05D1000` matches the gfx950 `buffer_load_dwordx4` LDS form.

## Smoke test (`m4096_n4096_k32768`,
variant `ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all_R38A`)

`R38A_smoke_one.py`, constant scale=-4, seed=42:

| Run | kernel_finite | nan%   | inf%   |
|-----|---------------|--------|--------|
| 1   | 0.6573        | 7.15%  | 27.11% |
| 2   | 0.6478        | 7.55%  | 27.67% |
| 3   | 0.6530        | 7.30%  | 27.40% |

R37 baseline (same shape, same module name minus the inline-asm change):
- kernel_finite = 0.7178 (1 run, same seed)
- R37 published bench number: 0.7445

R38A produces **slightly worse** (or comparable, well within run-to-run
non-determinism) finite-fraction than R37. **Both fail the 0.995 correctness
gate by a wide margin.**

## Verdict on hypothesis

**REFUTED**: replacing the `raw_buffer_load_lds` intrinsic with an
`asm volatile` carrying `"memory"` clobber does NOT fix the WRONG_OUTPUT
class. The bug is not LLVM reordering of the prefetch issue across
K-iteration boundaries. Possible alternative root causes (for downstream
investigation, not this round):

1. **Within-iteration reordering of MFMA ↔ ds_read ↔ scale-load**: the bug
   is between step3/step4 fragments and the *consumption* of LDS data, not
   between the LDS-write prefetches.
2. **LDS bank conflicts on m0-managed direct LDS writes** under the
   tail-pf-off path interacting with `BARRIER_TO_WAITCNT_ALL=1`.
3. **Wrong cache-hint dropped**: the inline-asm path drops the `cache_hint`
   parameter; if any variant relies on `cache_streaming` or similar (R22B
   hint), behavior could diverge. The `_dc` variants (e.g.
   `ts_v12_tv0_memc_dc_gm7_pfoff120_...`) hint at cache-hint sensitivity,
   though none of the R37 WIN modules regress in this test (see full sweep).

## Build manifest

`build_R38A.py 16 all` — 30 unique modules built clean in 10.1s
(parallel); 0 spills; VGPRs=244, AGPRs=256, occupancy=1 wave/SIMD on
the canonical large-K module.

## Files produced

- `kernel_mxfp4_gluon_cpp.cpp` — added `R38A_INLINE_BUFLOAD_LDS` macro
  (default OFF) and the inline-asm path in `emit_one_pf`. R37 WIN modules
  unchanged.
- `build_R38A.py` — clones build_R37 with `-DR38A_INLINE_BUFLOAD_LDS=1`.
- `bench_all_42_R38A.py` — clones bench_all_42_R37 against `build_R38A/`.
- `R38A_smoke_one.py` — single-shape correctness probe.
- `R38A_BUILD_MANIFEST.json` — built-modules manifest.
- `R38_OPT_A_BENCH.log` + `bench_all42_results_R38_optA.json` — full sweep
  results (in flight).
- `R38_OPT_A_VERDICT.md` — final verdict (TBD).
