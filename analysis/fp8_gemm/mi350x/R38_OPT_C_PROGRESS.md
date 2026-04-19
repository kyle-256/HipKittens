# R38 Opt C — Tail-iter L2-only prefetch (progress log)

Date: 2026-04-19
Branch: `mxfp4`
Goal: Fix B3 — when R25-C tail-pf-off would skip a tail prefetch, route it
through `emit_full_pf_l2only<>` so that:
  1. The buffer is fetched into L2 cache (warming next-iter access),
  2. NO LDS write happens (no double-buffer slot collision),
  3. The compiler-tracked `vmcnt` stays consistent (no skipped load → no
     barrier-vs-load race),
hopefully eliminating R37 CRASH while preserving R37 WINs that R38B regressed.

## Macro added

`R38C_TAIL_L2ONLY` (default OFF). When ON, gates a new `if (_r25c_tail_no_pf)`
block placed AFTER the existing `if (!_r25c_tail_no_pf) emit_pf_tail<0>(...)`
block in the `R37_FIX_B && !FUSED_STEP34` branch (kernel ~line 3019). The
block calls `emit_full_pf_l2only<PF_MPT>` on each of `pf_a0_p`, `pf_a1_p`,
`pf_bl_p` (skipped when `DIRECT_BL`), `pf_br_p`. All four `pf_*_p` structs
were already constructed at the loop top (line 2621), so no extra struct
construction cost. R38C and R38B are mutually exclusive (compile-time
`#error` if both ON).

The L2-only emit (`emit_one_pf_l2only`) was already present in the kernel
since R24B/R24C (line 999): an `asm volatile` `buffer_load_dwordx4` into
a `float4 sink` VGPR with a `"memory"` clobber. The compiler cannot DCE
inline asm volatile — vmcnt metadata is preserved.

## Smoke test on canonical CRASH shape

`m32768_n4096_k2048` variant `ts_lgk2_gm6_v12_memc_pfoff4` (R37: CRASH on rep 1
after producing finite=0.984 on rep 0; R38B: WRONG_OUTPUT, no fault).

Built clean (7s). 50 reps via `R38C_smoke_one.py` on GPU 0:
- **No GPU fault** across all 50 reps (deliverable goal a).
- finite range 0.984–0.989 — same as R37's pre-crash rep, same as R38B's
  finite range. **Below the 0.995 gate** (deliverable goal b NOT met).

This was the same outcome R38B saw on this exact shape. The hypothesis that
the L2-only path would recover finite was WRONG — the wrong-output condition
was never about LDS slot collisions; it's about MFMAs reading stale LDS while
R37_FIX_B is active without the always-emit prefetch state machine.

## Full 42-shape bench (R38C build)

Methodology: warmup=200, iters=500, trim=10%, finite≥0.995 gate, 8 GPUs.
Build: 30 unique .so files in `build_R38C/`, 0 failures, 21.7s wall.
Bench: 0.8 min wall.

| Bucket | R37 | R38B | R38C |
|---|---|---|---|
| WIN  | 14 | 5  | 5  |
| LOSE | 0  | 14 | 13 |
| WRONG_OUTPUT | 19 | 23 | 24 |
| CRASH | 9  | **0** | **0** |

### CRASH transitions (9 R37 CRASH shapes under R38C)

| Shape (MxNxK) | R38C status | finite | tflops | %comp |
|---|---|---|---|---|
| 16384x4096x3072  | **WIN**          | 0.9997 | 3505.2 | 100.4% |
| 28672x32768x4096 | LOSE             | 0.9962 | 3885.7 | 87.0%  |
| 128256x32768x4096| LOSE             | 0.9974 | 3882.6 | 85.6%  |
| 32768x4096x2048  | WRONG_OUTPUT     | 0.9821 | —      | —      |
| 32768x6144x2048  | WRONG_OUTPUT     | 0.9947 | —      | —      |
| 16384x28672x2048 | WRONG_OUTPUT     | 0.9856 | —      | —      |
| 32768x28672x2048 | WRONG_OUTPUT     | 0.9911 | —      | —      |
| 4096x32768x6144  | WRONG_OUTPUT     | 0.9886 | —      | —      |
| 14336x32768x4096 | WRONG_OUTPUT     | 0.9808 | —      | —      |

**0/9 still crash. 3/9 correct + competitive (1 WIN, 2 LOSE).**
**6/9 WRONG_OUTPUT, finite 0.981–0.995 (just below gate).**

### WIN preservation (14 R37 WINs under R38C)

| Shape | R37 tflops | R38C status | finite | R38C tflops | class |
|---|---|---|---|---|---|
| 16384x4096x3072  | (CRASH→WIN)| OK           | 0.9997 | 3505.2 | WIN  100.4% |
| 16384x4096x4096  | 4561.2     | OK           | 0.9974 | 3970.1 | WIN  100.5% |
| 16384x6144x4096  | 4768.1     | OK           | 0.9974 | 4067.2 | WIN  100.6% |
| 16384x6144x2048  | 3436.2     | OK           | 0.9954 | 2701.0 | LOSE 88.6%  |
| 4096x4096x8192   | 4371.6     | OK           | 1.0000 | 3777.9 | LOSE 95.4%  |
| 4096x14336x8192  | 4785.6     | OK           | 0.9970 | 3927.4 | LOSE 90.4%  |
| 6144x4096x8192   | 4287.6     | OK           | 0.9965 | 3384.7 | LOSE 88.6%  |
| 16384x4096x6144  | 4926.9     | OK           | 0.9977 | 4201.6 | LOSE 98.6%  |
| 16384x4096x7168  | 5063.3     | OK           | 0.9976 | 4215.8 | LOSE 94.9%  |
| 4096x4096x16384  | 4983.0     | WRONG_OUTPUT | 0.9899 | —      | WRONG       |
| 4096x14336x16384 | 5249.4     | WRONG_OUTPUT | 0.9418 | —      | WRONG       |
| 6144x4096x16384  | 4625.5     | WRONG_OUTPUT | 0.9855 | —      | WRONG       |
| 28672x4096x8192  | 5243.3     | WRONG_OUTPUT | 0.9920 | —      | WRONG       |
| 28672x4096x16384 | 5770.0     | WRONG_OUTPUT | 0.9483 | —      | WRONG       |
| 32768x4096x7168  | 5159.7     | WRONG_OUTPUT | 0.9909 | —      | WRONG       |

R37 WIN preservation: 3 WIN + 6 LOSE + 6 WRONG_OUTPUT (-11 net).

### Why didn't R38C beat R38B?

R38C and R38B both completely close the wait-count race that caused R37 CRASH —
0/9 CRASH under either macro. The L2-only emission has the same compiler-visible
buffer_load semantics as the LDS-write emission (asm-volatile, "memory"
clobber, vmcnt-tracked). The CRASH was about scheduler metadata consistency,
not about cache state.

But R38C did NOT recover the perf+correctness that R38B sacrificed:
- The 6 R37→WRONG_OUTPUT shapes that R38B converted from CRASH and the 6 R37
  WINs that R38B demoted to WRONG_OUTPUT all sit in the 0.94–0.99 finite range
  on R38C too.
- The 9 R37 WINs that R38B demoted to LOSE remained LOSE on R38C, with similar
  perf gaps (88–99%).

The reason: the LDS slot collision hypothesis was WRONG. The actual perf cost
in R38B is from the EXTRA buffer_load instructions issued on tail iters (4*PF_MPT
loads per tail iter). R38C issues the SAME number of loads (just discarded
instead of LDS-written), so the bandwidth cost is identical.

The reason R37 was faster on those 9 LOSE shapes: it skipped the loads ENTIRELY
(R25-C tail-pf-off). The wait-count metadata happened to land in a state where
the kernel didn't crash IF the variant didn't trigger the race (33/42), and
crashed IF it did (9/42). R38C makes the kernel safe by re-emitting the loads
in any form, but every "safe" form has the same bandwidth cost.

## Conclusion

R38C structurally eliminates CRASH (deliverable goal achieved on the
"no fault" axis), exactly matching R38B. Net leaderboard impact is essentially
identical to R38B: WIN goes 14→5 (-9), CRASH goes 9→0 (+0 saved usable
shapes vs R38B: 3 OK out of 9 CRASH shapes, identical to R38B).

The Fix B3 hypothesis as stated ("L2-only keeps the per-iter LDS-write cost
out") was incorrect — there is no per-iter LDS-write cost difference. The
real cost is the buffer_load bandwidth itself, which both fixes pay equally.
The right next steps are listed in the verdict.

## Files produced

- `kernel_mxfp4_gluon_cpp.cpp` — `R38C_TAIL_L2ONLY` macro (default OFF) +
  L2-only branch wired in the R37_FIX_B && !FUSED_STEP34 path. R38B-mutex `#error`.
- `build_R38C.py` — builder; sets `-DR38C_TAIL_L2ONLY=1`, strips the bad
  `-mllvm -amdgpu-sched-strategy=max-memory-clause`. `--all` builds all 42.
- `bench_all_42_R38C.py` — bench harness pointing at `build_R38C/`, gate
  finite≥0.995, output to `bench_all42_results_R38_optC.json`.
- `bench_all42_results_R38_optC.json` — full 42-shape results.
- `R38C_BUILD_RUN.log` — build run log.
- `R38C_BENCH_RUN.log` — bench run log.
- `R38C_BUILD_MANIFEST.json` — build manifest (30 unique builds, 0 failures).
- `R38C_smoke_one.py` — single-shape repro harness.
- `build_R38C/` — 30 .so files (one per (N,K,variant)).
