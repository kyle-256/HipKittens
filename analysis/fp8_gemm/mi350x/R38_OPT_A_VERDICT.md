# R38 Opt A — Verdict: **FAIL** (catastrophic regression) (2026-04-19)

## Summary

| Round | WIN  | LOSE | WRONG_OUTPUT | CRASH |
|-------|------|------|--------------|-------|
| R37   | 14/42 | 0/42 | 19/42        | 9/42  |
| R38A  | **0/42** | 0/42 | **36/42**   | **6/42** |

The R38A inline-asm replacement of `__builtin_amdgcn_raw_buffer_load_lds` with
`asm volatile("buffer_load_dwordx4 ... offen lds")` **broke ALL 14 R37 WIN
shapes** and **did not fix any of the 19 R37 WRONG_OUTPUT shapes**. Net delta:
−14 WINs, +17 WRONG_OUTPUT, −3 CRASH.

The hypothesis ("LLVM scheduler reorders prefetches across iteration
boundaries via implicit clang scheduling") is **REFUTED** with high
confidence.

## Per-shape comparison (correctness only — no TFLOPS recorded for any R38A shape)

| Shape (M×N×K) | R37 status | R37 finite | R38A status | R38A finite | Delta |
|---------------|------------|-----------:|-------------|------------:|-------|
| 16384×4096×2048   | CRASH        | N/A    | WRONG_OUTPUT | 0.9865 | up (CRASH→WRONG) |
| 16384×4096×3072   | CRASH        | N/A    | WRONG_OUTPUT | 0.9861 | up (CRASH→WRONG) |
| 16384×6144×2048   | OK (WIN)     | 0.9996 | WRONG_OUTPUT | 0.9782 | **DOWN** |
| 32768×4096×2048   | CRASH        | N/A    | WRONG_OUTPUT | 0.9622 | up (CRASH→WRONG) |
| 32768×4096×3072   | OK (WIN-like)| 0.9932 | WRONG_OUTPUT | 0.8789 | **DOWN** |
| 32768×6144×2048   | CRASH        | N/A    | CRASH        | N/A    | same |
| 16384×14336×2048  | OK (WIN-like)| 0.9944 | WRONG_OUTPUT | 0.9699 | **DOWN** |
| 16384×28672×2048  | CRASH        | N/A    | CRASH        | N/A    | same |
| 32768×14336×2048  | WRONG_OUTPUT | 0.9912 | WRONG_OUTPUT | 0.9654 | DOWN  |
| 32768×28672×2048  | CRASH        | N/A    | CRASH        | N/A    | same |
| 4096×4096×16384   | OK (WIN)     | 0.9991 | WRONG_OUTPUT | 0.6418 | **DOWN** |
| 4096×14336×16384  | OK (WIN)     | 0.9964 | WRONG_OUTPUT | 0.6066 | **DOWN** |
| 6144×4096×16384   | OK (WIN)     | 0.9995 | WRONG_OUTPUT | 0.6452 | **DOWN** |
| 4096×4096×8192    | OK (WIN)     | 1.0000 | WRONG_OUTPUT | 0.9864 | **DOWN** |
| 4096×4096×32768   | WRONG_OUTPUT | 0.7445 | WRONG_OUTPUT | 0.6429 | DOWN  |
| 4096×6144×32768   | WRONG_OUTPUT | 0.6623 | WRONG_OUTPUT | 0.6146 | DOWN  |
| 4096×14336×8192   | OK (WIN)     | 0.9987 | WRONG_OUTPUT | 0.9643 | **DOWN** |
| 4096×28672×32768  | WRONG_OUTPUT | 0.6216 | WRONG_OUTPUT | 0.5301 | DOWN  |
| 4096×32768×4096   | WRONG_OUTPUT | 0.9913 | WRONG_OUTPUT | 0.9104 | DOWN  |
| 4096×32768×6144   | CRASH        | N/A    | WRONG_OUTPUT | 0.9271 | up    |
| 4096×32768×14336  | WRONG_OUTPUT | 0.9883 | WRONG_OUTPUT | 0.7629 | DOWN  |
| 4096×32768×28672  | WRONG_OUTPUT | 0.6145 | WRONG_OUTPUT | 0.5268 | DOWN  |
| 4096×32768×128256 | WRONG_OUTPUT | 0.7613 | WRONG_OUTPUT | 0.2871 | DOWN  |
| 4096×128256×32768 | WRONG_OUTPUT | 0.6357 | WRONG_OUTPUT | 0.5766 | DOWN  |
| 6144×4096×8192    | OK (WIN)     | 0.9993 | WRONG_OUTPUT | 0.9851 | **DOWN** |
| 6144×32768×4096   | WRONG_OUTPUT | 0.9903 | WRONG_OUTPUT | 0.9381 | DOWN  |
| 14336×4096×32768  | WRONG_OUTPUT | 0.6603 | WRONG_OUTPUT | 0.6141 | DOWN  |
| 14336×32768×4096  | CRASH        | N/A    | CRASH        | N/A    | same  |
| 16384×4096×4096   | OK (WIN)     | 0.9951 | WRONG_OUTPUT | 0.9609 | **DOWN** |
| 16384×4096×6144   | OK (WIN)     | 0.9995 | WRONG_OUTPUT | 0.9864 | **DOWN** |
| 16384×4096×7168   | OK (WIN)     | 0.9995 | WRONG_OUTPUT | 0.9873 | **DOWN** |
| 16384×4096×14336  | WRONG_OUTPUT | 0.9789 | WRONG_OUTPUT | 0.9563 | DOWN  |
| 16384×4096×28672  | WRONG_OUTPUT | 0.6709 | WRONG_OUTPUT | 0.6185 | DOWN  |
| 16384×6144×4096   | OK (WIN)     | 0.9996 | WRONG_OUTPUT | 0.9437 | **DOWN** |
| 16384×14336×4096  | WRONG_OUTPUT | 0.9929 | WRONG_OUTPUT | 0.9159 | DOWN  |
| 16384×28672×4096  | WRONG_OUTPUT | 0.9753 | WRONG_OUTPUT | 0.9219 | DOWN  |
| 28672×4096×8192   | OK (WIN)     | 0.9955 | WRONG_OUTPUT | 0.9489 | **DOWN** |
| 28672×4096×16384  | OK (WIN)     | 0.9983 | WRONG_OUTPUT | 0.6208 | **DOWN** (collapse) |
| 28672×32768×4096  | CRASH        | N/A    | CRASH        | N/A    | same  |
| 32768×4096×7168   | OK (WIN)     | 0.9964 | WRONG_OUTPUT | 0.9731 | **DOWN** |
| 32768×4096×14336  | WRONG_OUTPUT | 0.9896 | WRONG_OUTPUT | 0.9788 | DOWN  |
| 128256×32768×4096 | CRASH        | N/A    | CRASH        | N/A    | same  |

**14 WIN→WRONG_OUTPUT regressions** (large bold-marked rows above), every
single R37 WIN now fails the 0.995 gate.

## Key observations

1. **Inline-asm replacement consistently degrades correctness by 0.5–4 pts**
   on shapes that previously hit ≥0.995. Several K=16384 shapes drop from
   0.999 to ~0.62 — a *collapse* of the same magnitude as the original
   WRONG_OUTPUT class.
2. **The original WRONG_OUTPUT shapes do NOT recover** — they get marginally
   worse (0.7445 → 0.6429 on 4096×4096×32768).
3. The 9 CRASH shapes (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION) split:
   3 stay CRASH, 6 turn into WRONG_OUTPUT. The inline-asm path apparently
   does not reproduce the OOB-load behavior of the
   `_ts_lgk2_gm6_v12_memc_pfoff4` variant — it produces garbage instead of
   crashing.

## Why R38A regressed instead of fixed

Several plausible mechanisms (not investigated further in this round):

1. **Lost cache-hint argument**: the inline asm hardcodes `aux=0` (default
   cache mode), dropping the `cache_all` (=0) hint that the intrinsic
   passes. For all R37 BEST_VARIANTS this should be a no-op since
   `cache_all == 0` *encodes the same bits*, but the gfx950 LDS-form
   `buffer_load_dwordx4` may interpret aux differently than the non-LDS
   form. (Audit needed.)
2. **m0 not preserved across non-LDS uses**: the inline asm clobbers m0,
   but the kernel uses ds_read instructions that depend on the previous
   m0 value being constant. The intrinsic-emission path coalesces multiple
   m0 sets across an entire block of LDS loads; the per-call inline-asm
   form forces re-set per load and may interleave m0 changes with ds_read
   in ways that violate ds_read addressing.
3. **`"memory"` clobber over-restricts the scheduler** to the point that
   register pressure spikes (we saw VGPRs go from 244 baseline to 244 — same
   — so this is NOT the issue; no spills).
4. **GFX950 buffer_load_dwordx4 LDS-form encoding differences**: the
   intrinsic may emit `buffer_load_dwordx4 vN, s[A:A+3], s[M] offen lds:1
   sc0:0 sc1:0 nt:0` (full bit pattern); our string form
   `buffer_load_dwordx4 %1, %2, %3 offen lds` may be treated by the
   assembler as a default that differs in some bit (e.g. nt or scope).
   The encoded bytes for our inline-asm form are `E05D1000`; the R37
   intrinsic-emitted form should be checked against this.

## Recommended next directions

- **Compare encoded bytes** of intrinsic-emitted vs inline-asm-emitted
  `buffer_load_dwordx4 ... lds` instructions side-by-side. If they differ,
  R38A's mnemonic string is wrong.
- **Try inline asm WITHOUT `"memory"` clobber** but with explicit
  `"~{m0}"` clobber and SSA-style operand ties; see if that
  preserves R37 WINs.
- **Pursue R38 Opt B/C alternative hypotheses** that don't depend on the
  scheduler theory: e.g. fork the BEST_VARIANTS for the 19 WRONG_OUTPUT
  shapes to use less aggressive flag stacks (the original Fix A from
  R35_WRONG_CELLS_DIAGNOSIS.md), or audit the `_ts_lgk2_gm6_v12_memc_pfoff4`
  CRASH variant for the OOB-load bug.

## Methodology

- Bench: `bench_all_42_R38A.py`, warmup=200, iters=500, trim=0.10,
  correctness gate kernel_finite ≥ 0.995, 8 GPUs (GPU0..7), elapsed 3.9 min.
- Build: `build_R38A.py` with `-DR38A_INLINE_BUFLOAD_LDS=1`,
  `-mllvm -amdgpu-sched-strategy=max-memory-clause` stripped.
- Smoke test: `R38A_smoke_one.py` (3-run reproducibility on
  4096×4096×32768).
- All raw output: `R38_OPT_A_BENCH.log`,
  `bench_all42_results_R38_optA.json`.

## Files

- `kernel_mxfp4_gluon_cpp.cpp` — `R38A_INLINE_BUFLOAD_LDS` macro added
  (default OFF; **R37 WIN modules are unaffected** because the macro is
  not set when building build_R37 modules).
- `build_R38A.py`, `bench_all_42_R38A.py`, `R38A_smoke_one.py`
- `bench_all42_results_R38_optA.json` — full sweep
- `R38_OPT_A_PROGRESS.md` — work log
- `R38_OPT_A_VERDICT.md` — this file
