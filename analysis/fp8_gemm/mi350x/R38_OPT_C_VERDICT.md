# R38 Opt C — VERDICT

**Status: STRUCTURAL PARITY WITH R38B — CRASH eliminated, but the Fix B3
mechanism failed to recover the perf+correctness R38B sacrificed. Do NOT
default `R38C_TAIL_L2ONLY=1` for the whole leaderboard.**

## Headline

| Metric | R37 | R38B | **R38C** |
|---|---|---|---|
| CRASH (HSA fault) | 9/42 | 0/42 | **0/42** |
| WIN  | 14/42 | 5/42 | **5/42** |
| LOSE | 0/42  | 14/42 | **13/42** |
| WRONG_OUTPUT | 19/42 | 23/42 | **24/42** |

## What Fix B3 actually does

When R25-C signals `_r25c_tail_no_pf` (the last `R25C_TAIL_PF_OFF_ITERS`
iters), instead of skipping the four `emit_pf_tail<0>(pf_*_p)` calls (R37
default → CRASH on 9 variants), R38C calls `emit_full_pf_l2only<PF_MPT>`
on each of the four pf_*_p structs. This issues the SAME four
`buffer_load_dwordx4` GMEM fetches as the LDS-bound prefetch but discards
the result into a scratch VGPR (no `lds:1` modifier → no LDS write).

The `emit_full_pf_l2only<>` helper already existed in the kernel for the
R24C outer-K pull-forward L2-prefetch path (line 999, `asm volatile` with
"memory" clobber).

Macro `R38C_TAIL_L2ONLY` is gated default OFF. The macro is in the same
`R37_FIX_B && !FUSED_STEP34` branch R38B targets. R38B and R38C are
mutually exclusive (compile-time `#error`).

## Smoke test

`m32768_n4096_k2048` variant `ts_lgk2_gm6_v12_memc_pfoff4` (canonical R37
CRASH case, R38B WRONG_OUTPUT case): 50 reps clean (no fault), finite
0.984–0.989 (below 0.995 gate but stable). Same finite range as R37
produced on its single rep before crashing, and same range R38B produced.
**Goal (a) met (no fault). Goal (b) NOT met (finite below gate).**

## CRASH transition (9 R37 CRASH shapes)

3 OK, 6 WRONG_OUTPUT — IDENTICAL outcome to R38B.

| Shape | R38C status | finite | tflops | %comp |
|---|---|---|---|---|
| 16384x4096x3072  | **WIN**      | 0.9997 | 3505 | 100.4% |
| 28672x32768x4096 | LOSE         | 0.9962 | 3886 | 87.0%  |
| 128256x32768x4096| LOSE         | 0.9974 | 3883 | 85.6%  |
| 32768x4096x2048  | WRONG_OUTPUT | 0.9821 | —    | —      |
| 32768x6144x2048  | WRONG_OUTPUT | 0.9947 | —    | —      |
| 16384x28672x2048 | WRONG_OUTPUT | 0.9856 | —    | —      |
| 32768x28672x2048 | WRONG_OUTPUT | 0.9911 | —    | —      |
| 4096x32768x6144  | WRONG_OUTPUT | 0.9886 | —    | —      |
| 14336x32768x4096 | WRONG_OUTPUT | 0.9808 | —    | —      |

## R37 WIN preservation (14 shapes)

3 WIN + 6 LOSE + 6 WRONG_OUTPUT — same regression pattern as R38B
(R38B was 2 WIN + 9 LOSE + 3 WRONG_OUTPUT, slightly worse on WIN
preservation, slightly better on WRONG count). Net leaderboard impact
is identical (WIN 14→5).

## Why didn't R38C beat R38B?

The premise of the Fix B3 mission was that R38B's perf cost came from
"per-iter LDS-write" overhead and that L2-only loads would dodge it.
**That premise was incorrect**:

1. The L2-only and LDS-bound `buffer_load_dwordx4` instructions issue the
   SAME GMEM bandwidth. The only difference is whether the destination is
   LDS or a discarded VGPR. There is no measurable LDS-write cost on
   gfx950 for the small bursts in question — the DRAM/L2 bandwidth is the
   binding constraint, not LDS write port pressure.

2. The CRASH was about **scheduler metadata consistency**, not about LDS
   slot collisions. R37's R25-C "skip" left the compiler emitting an
   `s_waitcnt(vmcnt(N))` whose N didn't match the in-flight load count;
   the next iter's `s_barrier` then fired while a previous-iter load was
   still landing. R38B's "always emit LDS prefetch" closed the gap by
   making the actual load count match. R38C's "always emit L2-only
   prefetch" closes it the same way. Both are bandwidth-equivalent.

3. The 6 WIN→WRONG_OUTPUT and 6 CRASH→WRONG_OUTPUT demotions are NOT
   crashes — they're R37_FIX_B's stale-LDS-read condition that the R25-C
   skip was incidentally masking. Re-emitting the prefetch (in any form)
   reveals the bug; no L2-vs-LDS choice changes that.

In short: **R38B and R38C are bandwidth-equivalent fixes for the same
race; the perf cost they pay is the buffer_load bandwidth itself, which
neither can avoid while still emitting the load.**

## Recommended next steps

1. **Adopt R38C selectively** if it is preferable to R38B on any per-shape
   basis. The 3 CRASH→OK shapes (16384x4096x3072, 28672x32768x4096,
   128256x32768x4096) are the primary candidates. Numerical comparison:

   | Shape | R38B tflops | R38C tflops | better |
   |---|---|---|---|
   | 16384x4096x3072  | 3500 | 3505 | R38C (tie) |
   | 28672x32768x4096 | (re-check) | 3886 | — |
   | 128256x32768x4096| 3935 | 3883 | R38B |

2. **The right structural fix is NOT in the prefetch path** — it's in the
   R37_FIX_B step3+step4 fused KPAIR itself. The MFMAs in tail iters need
   to read FRESH LDS, but they're reading STALE LDS because the LDS write
   from the previous iter hasn't drained. Options:
   - emit an explicit `s_waitcnt vmcnt(0); s_barrier` on the tail iter
     before the MFMA reads — slow but correct;
   - drop the R25-C optimization entirely on shapes where R37_FIX_B is
     active (use STEP3_PF_N=8 / STEP4_PF_N=8 with full prefetches all
     the way through, and fix R25-C only on FUSED_STEP34=1 path);
   - port the kpair_64mfma_step34 tail iter to use the prologue/epilogue
     direct-load path that already handles the wait correctly.

3. **Variant fork for the 19+ WRONG_OUTPUT shapes**: drop `_pfoff*` from
   the BEST_VARIANTS for any shape where R37_FIX_B + R25-C combo produces
   finite < 0.995. Accept the ~3% perf loss per affected shape but get
   correct output across the full leaderboard.

## Files

- `kernel_mxfp4_gluon_cpp.cpp` — `R38C_TAIL_L2ONLY` macro, default OFF
  (set `-DR38C_TAIL_L2ONLY=1` to enable). Mutex with R38B.
- `build_R38C.py` — builder
- `bench_all_42_R38C.py` — bench harness
- `bench_all42_results_R38_optC.json` — full bench
- `R38_OPT_C_PROGRESS.md` — detailed progress log
- `R38C_BENCH_RUN.log` / `R38C_BUILD_RUN.log` — run logs
- `build_R38C/` — 30 unique .so binaries
- `R38C_smoke_one.py` — single-shape repro
- `R38C_BUILD_MANIFEST.json` — build manifest
