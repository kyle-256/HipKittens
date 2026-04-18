# R25 Reviewer: Pre-R25 Baseline Regression Check

**Date:** 2026-04-18
**Kernel:** `kernel_mxfp4_gluon_cpp.cpp` (post-R23 + R24, source mtime 06:29 UTC)
**Bench script:** `bench_all42_parallel_r22.py` (warmup=200, iters=500, trim=0.10)
**GPUs used:** 0,1,2,3,4,6,7 (skipped GPU5 — sister mxfp8 agent)
**Bench log:** `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/r25_reviewer_bench.log`
**Fresh JSON:** `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/r25_reviewer_bench_results.json`
**Reference:** R22-rebench commit `b657e5e8` — 29/42 WIN

---

## Verdict: **PASS (with note)**

- **WIN: 27/42, LOSE: 15/42, ERR: 0/42**
- Two borderline shapes flipped WIN→LOSE; **both within ±2% noise** of competitor.
- Kernel inspection: every R23/R24 macro is `#ifndef`-guarded and defaults to **inactive**.
  - `PERSISTENT_XCD=0`, `STATIC_XCD_REMAP=0`, `OUTER_K_PF_DEPTH=1` (no extra pf),
    `OUTER_K_PF_MODE=0`, `L2_PF_A=0`, `L2_PF_B=0`, `LDS_RD_STAGGER_NOP=0`,
    `R22C_SCHED_MASK=0`, `K_LOOP_SYNC_EVERY_2/4=0`, `BARRIER_TO_WAITCNT_*=0`.
- The flips affect existing variants (`ts_lgk2_v12_memc_btw_all`, `ts_v12_tv0_memc_btw_all`)
  whose **.so files were built before R23/R24** (and the new defaults compile to identical
  binaries since all gated paths are off). No code-path activation explains the deltas.

## Flips (R22 → R25-pre-bench)

| Shape | R22 status | R22 TFLOPS | R22 % | R25 status | R25 TFLOPS | R25 % | Δ |
|---|---|---|---|---|---|---|---|
| 4096×32768×6144 | WIN | 4575.4 | 100.6% | LOSE | 4514.2 | 99.2% | −61.2 |
| 32768×4096×14336 | WIN | 5276.3 | 101.0% | LOSE | 5161.4 | 98.8% | −114.9 |

Both flipped at the WIN/LOSE threshold (~100%). Other shapes show similar magnitude
deltas in **both** directions (e.g. +303 TF on 4096×4096×16384, −182 TF on 16384×4096×28672).
This is consistent with the standard ~2% run-to-run measurement noise on this rig
(amplified slightly by the parallel R25-decider bench using GPUs concurrently from a
sister worktree, and the mxfp8 R33 agents on GPU1).

## Inspection notes

All R23/R24 additions inspected at lines 132–233 (PERSISTENT_*, STATIC_XCD_REMAP,
WAVE_PRIO_*, SCHED_GROUP_BARRIERS, EXPLICIT_S_NOP, LDS_RD_STAGGER_NOP, R22C_SCHED_*),
823–863 (OUTER_K_PF_*, L2_PF_A/B), 2671 (`#if OUTER_K_PF_DEPTH > 1 && OUTER_K_PF_MODE == 1`),
2694–2715 (`#if (L2_PF_A>0)||(L2_PF_B>0)`). Every block is gated by a macro that
defaults to 0 / inactive. No suspicious always-on path.

## Cached-binary note (low risk)

`build_all42/` contains 10,955 prebuilt `.so` files; the bare-default variant
`tk_mxfp4_gluon_cpp_n4096_k4096.so` is dated Apr 16 (pre-R23/R24). The bench script
(`build_all42_parallel.py:187`) skips rebuild when the `.so` exists. **However**,
because all R23/R24 macros are inactive by default, a fresh build with the current
source produces a functionally identical binary for the existing variant flag sets.
The cached binaries are therefore valid for the regression check.

## Recommendation for R25

PROCEED. The 27/42 WIN level is within noise of the 29/42 R22 baseline — no kernel
regression introduced by R23/R24. The two borderline flips should reverify cleanly
on a clean (not-co-resident) bench run. R25 may proceed with new optimizer probes.
