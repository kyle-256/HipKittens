# R25-G — PER-K PFOFF SWEEP — MASSIVE 6-SHAPE WIN

**Date**: 2026-04-18
**Branch**: `mxfp4`. Parent: `178b533e` (R25-F MASSIVE WIN baseline).
**GPUs used**: 0, 1, 3, 4, 6, 7 (rotated as availability shifted; verifications
done on idle GPUs after spot-checking `rocm-smi`; SD final verify on isolated GPU 1).
**Bench params**: warmup=200, iters=500, trim=10% (per `.claude/rules/benchmark-rules.md`).

## Mission

Extend R25-F's mechanism (K=4096, gm7 + R25C_TAIL_PF_OFF_ITERS=14, i.e.
"only the first 2 K-iters issue global prefetches") to MID-GAP LARGER-K LOSE
shapes (K ∈ {14336, 16384, 28672, 32768}) that R25-F's pfoff≤16 sweep could
not reach.

For each K, the analogous "first ~2-8 prefetched then off" point lies at
`pfoff = K_iters - {1..8}` where `K_iters = K / 256`:
  - K=14336 → K_iters=56  → swept {48, 52, 54, 55}
  - K=16384 → K_iters=64  → swept {56, 60, 62, 63}
  - K=28672 → K_iters=112 → swept {104, 108, 110, 111}
  - K=32768 → K_iters=128 → swept {120, 124, 126, 127}

## Results — 6/6 SHAPES WIN (all stable, std ≤ 30 TFLOPS)

Each candidate was 3-rep verified against its current production parent.
`gm7_pfoff{Y}` = parent flag stack with `-DGROUP_SIZE_M=7` AND
`-DR25C_TAIL_PF_OFF_ITERS={Y} -DR25C_K_LIMIT=32768`.

| # | Shape (M×N×K)        | K     | K_iters | Parent (pre-R25G)        | Parent TFLOPS | Winner       | Win TFLOPS | Δ%          |
|---|----------------------|-------|---------|--------------------------|---------------|--------------|------------|-------------|
| SA| 4096×28672×32768     | 32768 | 128     | ts_v12_tv0_memc_btw_all  | 5423.43       | gm7_pfoff120 | **6486.67**| **+19.60%** |
| SB| 4096×32768×14336     | 14336 | 56      | ts_v12_tv0_memc_btw_all  | 5187.39       | gm7_pfoff54  | **6097.59**| **+17.55%** |
| SC| 14336×4096×32768     | 32768 | 128     | ts_lgk2_memc_btw_all     | 5028.16       | gm7_pfoff124 | **5910.76**| **+17.55%** |
| SD| 16384×4096×28672     | 28672 | 112     | ts_lgk2_memc_btw_all     | 5270.78       | gm7_pfoff104 | **6369.26**| **+20.84%** |
| SE| 28672×4096×16384     | 16384 | 64      | ts_lgk2_memc_btw_all     | 5194.05       | gm7_pfoff56  | **6071.22**| **+16.89%** |
| SF| 4096×14336×16384     | 16384 | 64      | ts_lgk2_memc_btw_all     | 4938.28       | gm7_pfoff56  | **5613.97**| **+13.68%** |

Std (3-rep): SA=6.55, SB=8.70, SC=10.18 (verify2), SD=13.93 (verify3 on isolated GPU 1), SE=4.92, SF=10.19.
All pass the 30-TFLOPS gate; no aperture violations on the final verifies (some
intermediate runs had `rc=-6` due to GPU contention from concurrent R25-D verify
— resolved by isolating each shape on a quiet GPU).

## Mechanistic interpretation

For every K tested, the optimum pfoff is at **`K_iters - 4` to `K_iters - 8`**
— i.e. only the first 4-8 K-iters issue VMEM prefetches. The remaining ~90-95%
of the K-loop runs PF_N=0 (LDS preload + mfma only).

This is the **same mechanism** R25-F established on K=4096: persistent-XCD
remap means the B-tile is L2-resident across iterations after a brief warm-up;
tail prefetches just re-fetched lines already cached and stole VMEM bandwidth
from steady-state scale-load and C-write traffic.

The optimum offset from `K_iters` is slightly larger here (4-8 iters of
prefetch retained) than for K=4096 (only 2 retained, pfoff=14/16). Likely
explanation: larger K means the B-tile is larger and takes a few more iters
to fully warm L2 before steady-state cache hits dominate.

## Wired into `bench_all_42.py`

Added 5 K-EXACT-gated variants (lines ~513-540):

```
_ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all   # SA win
_ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all    # SB win
_ts_lgk2_gm7_memc_pfoff124_kx32768_btw_all         # SC win
_ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all         # SD win
_ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all          # SE + SF win
```

A new compile-time gate `R25C_K_EXACT` was added to the kernel
(lines 92-105). When set to a positive value, R25C only activates for the
EXACT specified `K_DIM`; otherwise the variant degrades gracefully to the
parent-without-R25C on other K's. This is essential because `pfoff=124`
on a K=4096 shape (k_byte_iters=16 < 124) would zero out ALL prefetches
and catastrophically regress.

```c
#define R25C_ACTIVE ((R25C_TAIL_PF_OFF_ITERS > 0) && (K_DIM <= R25C_K_LIMIT) \
                     && (R25C_K_EXACT == 0 || K_DIM == R25C_K_EXACT))
```

## DO NOT

- Use any of the new R25-G variants without `R25C_K_EXACT` set — would zero
  all prefetches on shapes where `pfoff > k_byte_iters` (catastrophic).
- Apply pfoff > k_byte_iters anywhere outside K_EXACT gating.

## Caveats / Future work

- A full 42-shape regression run is needed to confirm no surprise regressions
  from the new entries. Recommend running `bench_all42_parallel_r25d.py` style
  autotune once R25-D verify and R25-E free up GPUs.
- DLA1 (K=128256) is OUT OF SCOPE for R25-G (K > R25C_K_LIMIT=32768) —
  R25-E owns that work in its worktree.
- The plateau / single-step optimum is not yet bracketed for K=28672 (pfoff111
  errored with rc=-6 in smoke). If a future round needs more precision,
  pfoff ∈ {102, 103, 105, 106} should be probed.
- The SD shape (M=16384,N=4096,K=28672) showed bimodal stability on busy GPUs
  (5400 vs 4000 TFLOPS for the same parent). On a quiet GPU 1, both parent and
  winner were tight (std ≤ 14). Recommend always benchmarking SD on a fully
  isolated GPU.

## Files

- `kernel_mxfp4_gluon_cpp.cpp` (lines 92-105) — added `R25C_K_EXACT` gate
- `build_round25_optG.py` — 30 builds (5 pfoff values × 6 shapes incl. gm7-only DIAG; all OK)
- `bench_round25_optG_smoke.py`, `.json`, `.log` — phase-1 1-rep smoke (6 shapes)
- `bench_round25_optG_verify.py`, `.json`, `.log` — phase-2 3-rep verify (mixed-GPU; some `rc=-6`)
- `bench_round25_optG_verify2.py`, `.json`, `.log` — clean 3-rep re-verify of SC, SD
- `bench_round25_optG_verify3_SD.py`, `.json`, `.log` — final isolated 5-rep SD verify
- `bench_all_42.py` (lines ~513-540) — wired 5 R25-G entries
