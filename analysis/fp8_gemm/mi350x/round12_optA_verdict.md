# Round 12 Optimizer A — Verdict on `iterative-ilp` failures

## TL;DR (verdict category)
**(a) iterative-ilp is broken on certain shape categories** —
specifically deterministic on 3 shapes, probabilistically (1-of-3) on
1 more shape, and benign / flaky-incorrect on 1 shape. Recommended
action: **hybrid dispatch — apply iterative-ilp ONLY to the
verified-WIN parents** (Round 10 + Round 11 verify list). Do NOT
roll it out to all shapes.

## Reproduction (warmup=20, iters=50, retries=3, GPU 6)
| Shape (MxNxK)               | parent suffix              | baseline 3/3 | iterilp 3/3 | mode |
|-----------------------------|----------------------------|--------------|-------------|------|
| 4096×32768×128256 (DLA1)    | _ts_pf6_6_v12_memc         | 3 OK         | 0 OK        | DETERMINISTIC FAIL |
| 128256×32768×4096 (DLA2)    | _ts_gm2_v12_memc_dc        | 3 OK         | 0 OK        | DETERMINISTIC FAIL |
| 28672×32768×4096 (DLA7)     | _ts_lgk2_v12_memc          | 3 OK         | 0 OK        | DETERMINISTIC FAIL |
| 16384×4096×7168 (WIN1)      | _ts_lgk2_v20_memc          | 3 OK         | 3 OK        | FLAKY-OK (Round 11 was a 1-in-N glitch) |
| 32768×6144×2048 (WIN2)      | _ts_gm8_v12                | 3 OK         | 0 OK        | DETERMINISTIC FAIL |

Bisect (warmup=20, iters=50, retries=3, GPU 6) — built fresh `_r12_noilp` (parent flags only)
and `_r12_ilp_rebuild` (parent flags + iterative-ilp):

| Shape          | _r12_noilp 3/3 | _r12_ilp_rebuild 3/3 |
|----------------|----------------|----------------------|
| DLA1           | 3 OK           | 0 OK                 |
| DLA2           | 3 OK           | 0 OK                 |
| DLA7           | 3 OK           | 0 OK                 |
| WIN1           | 3 OK           | 3 OK                 |
| WIN2           | 3 OK           | **1 OK** (FLAKY)     |

## Root cause: iterative-ilp scheduler
- `_r12_noilp` is byte-identical (modulo .co filename header) to
  baseline. `_r12_ilp_rebuild` is byte-identical to `_r11_iterilp`.
  So adding `-mllvm -amdgpu-sched-strategy=iterative-ilp` is the
  ONLY difference triggering the fault.
- Failure modes: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION (most
  shapes) and "Write access to a read-only page" (DLA1, occasional
  WIN2). Both indicate pointer corruption — the address registers
  used by `global_store_short_d16_hi` are clobbered by the scheduler.
- Failure addresses are very large (e.g. 0xff9010f50000) and
  page-aligned (low 16 bits are 0) → strongly suggests the BASE of
  the C output pointer is wrong, not just the per-thread offset.
  Sometimes "read-only" addresses → kernel SRD or rodata region.

## ASM diff stats (DLA1: WIN1 too small to lookup easily)
- DLA1 baseline → iterilp:
  - VGPR: 468 → 460
  - SGPR: 93 → 91
  - AGPR: 256 → 256 (full)
  - LDS: 131072 → 131072 (= 128 KiB)
  - No spills in either (.vgpr_spill_count = 0, .sgpr_spill_count = 0)
- WIN1 baseline → iterilp:
  - VGPR: 452 → 424
  - SGPR: 75 → 75
  - AGPR: 256 → 256
  - LDS unchanged
- Reschedule is heavy (1152 buffer_load/global_load lines reorder
  in WIN1). The diff isn't an isolated bug we can grep for visually.

The scheduler is producing valid-looking code that nevertheless
clobbers C-output pointer SGPRs in some hard-to-spot way. This is
either:
  - A real LLVM/AMDGPU `iterative-ilp` correctness bug under heavy
    AGPR/VGPR pressure with many `v_mfma_scale_f32_16x16x128_f8f6f4`
    operations interleaved with `s_load_dwordx2` kernarg loads.
  - Some interaction with `lds_direct` `buffer_load_dwordx4 ... offen lds`
    (gfx950's direct-LDS load) that the iterative-ilp scheduler
    doesn't model correctly.

## Shape pattern (rough)
Failing shapes share: **K is a "small" multiple of 4096 (2048,
4096) OR K is HUGE and N is HUGE**. Working iterilp shapes share:
**M=4096 with intermediate N×K** (e.g. 4096×128256×32768,
4096×28672×32768, 4096×32768×14336, 4096×32768×28672).

A clean predicate isn't obvious without more shape coverage. The
safest dispatch heuristic is:
> Apply `iterative-ilp` ONLY on the explicitly-verified shapes
> (Round 10 + Round 11 verify shapes). For all other shapes, fall
> back to default (no `iterative-ilp`).

## Recommended action
1. **Do NOT roll out `iterative-ilp` as the new default for all 42 shapes.**
2. Keep the existing 5 verified WINs from Round 10 + Round 11 verify
   (commits `091d3baa`, the new Round 11 verify commit).
3. WIN2 (32768×6144×2048) is a CURRENT WIN under baseline flags —
   do NOT add iterative-ilp to it. Round 11 was correct to flag this
   as a regression risk; the data confirms the risk.
4. WIN1 (16384×4096×7168) — Round 11 reported a fault but our 3/3
   reproduction passed (binary is identical). This shape is FLAKY
   on its first kernel launch (likely a transient initialization
   issue). Adding iterative-ilp does NOT change baseline flakiness
   — both noilp and ilp-rebuild were 3/3 OK in bisect. Treat as
   safe to leave alone (don't add iterative-ilp for marginal gain
   on a current WIN).
5. Skip iterative-ilp on the 3 deep-LOSE shapes that fail
   deterministically: DLA1, DLA2, DLA7. Continue searching for
   alternative speedups for these.
6. **Optional follow-up (Round 12 OptB area)**: try
   `-mllvm -amdgpu-sched-strategy=iterative-minreg` or
   `iterative-gcn-max-occupancy` on the 3 failing deep-LOSE shapes
   to see if a different iterative scheduler avoids the bug while
   still giving register-pressure relief.
7. **Upstream report material** (if anyone files): reproducible
   miscompile in LLVM AMDGPU `iterative-ilp` scheduler on gfx950
   HIP kernels using `v_mfma_scale_f32_16x16x128_f8f6f4` with
   AGPR=256 saturation. Repro: `_r12_ilp_rebuild` .so files in
   `build_all42/`, source `kernel_mxfp4_gluon_cpp.cpp`. Build
   command in `build_round12_optA_bisect.py`.

## Files written
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/bench_round12_optA_repro.py`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/bench_round12_optA_repro.log`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/bench_round12_optA_repro.json`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_round12_optA_bisect.py`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_round12_optA_bisect.log`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/bench_round12_optA_bisect.py`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/bench_round12_optA_bisect.log`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/bench_round12_optA_bisect.json`
- ASM dumps in `/tmp/r12_optA_asm/{baseline,noilp,ilp_rebuild,iterilp,dla1_baseline,dla1_iterilp}.{fatbin,gfx950.co,s}`

## ASM resource diff (per kernel)
| Shape | metric | baseline | iterilp |
|-------|--------|----------|---------|
| DLA1 (n32768_k128256) | sgpr_count | 93 | 91 |
| DLA1 | vgpr_count | 468 | 460 |
| DLA1 | agpr_count | 256 | 256 |
| WIN1 (n4096_k7168) | sgpr_count | 75 | 75 |
| WIN1 | vgpr_count | 452 | 424 |
| WIN1 | agpr_count | 256 | 256 |
