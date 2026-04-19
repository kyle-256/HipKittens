# R38 Opt B — VERDICT

**Status: PARTIAL WIN — CRASH eliminated structurally, but with significant perf and correctness regressions on non-CRASH shapes.**

## Headline

| Metric | R37 | R38B (Fix B2 always-on) |
|---|---|---|
| CRASH shapes (HSA fault) | 9/42 | **0/42** |
| Correct & competitive (WIN) | 14/42 | 5/42 |
| Correct but slower (LOSE) | 0/42 | 14/42 |
| WRONG_OUTPUT | 19/42 | 23/42 |

## Verdict

**Do NOT default `R38B_TAIL_FIX=1` for the whole leaderboard.** The fix correctly
eliminates GPU faults on the 9 CRASH variants — 3 of which (16384x4096x3072,
32768x6144x2048, 128256x32768x4096) are now usable for the leaderboard — but the
always-emit semantic costs 8–22% perf on shapes where R25-C tail-pf-off was
delivering net wins, demoting 9 of the 14 R37 WINs to LOSE and breaking
correctness on 3 more.

## What Fix B2 actually does

Replaces the `if (!_r25c_tail_no_pf) emit_pf_tail<0>(pf_*_p)` runtime gate (in the
`R37_FIX_B && !FUSED_STEP34` branch only) with unconditional `emit_pf_tail<0>(pf_*_p)`,
relying on the existing `pf_bt = (bt+2 < k_byte_iters) ? bt+2 : k_byte_iters-1`
clamping to keep the prefetched address in range. Param construction is moved
inside the always-emit block.

This was reached after Fix B1 (move `make_pf_params` into the original gate) DID
NOT eliminate the crash — proving the crash was about the tail-pf SKIP itself, not
the per-iter struct construction overhead. A control build with `R25C_TAIL_PF_OFF
_ITERS=0` (always-emit by definition) ran 800 reps clean, confirming always-emit
is the structural fix.

## Why crashes happened in R37

Direct evidence (gpucore on rep 1, after rep 0 already produced finite=0.984):
the R37 leaderboard's "CRASH" was **always** a "WRONG_OUTPUT then GPU fault" —
the pre-fault rep produced wrong output too. The `R37_FIX_B` fused-step34 path
needs the per-iter prefetches that R25-C drops in the last `pfoff` iterations.
When pfoff > 0 and the kernel is fused, the last few iters' MFMAs read stale
LDS data, and (critically) the next-iter scheduler emits an `s_waitcnt` with a
count that doesn't match the actual outstanding loads, so the next iter
issues an `s_barrier` while a tail-iter buffer-load-to-LDS is still in flight
into a slot that's about to be reallocated for the next K block — fault.

Fix B2 closes the wait-count mismatch by ensuring every iter actually emits
the prefetches it claims to in its scheduler footprint.

## Why some non-CRASH shapes regress under Fix B2

For pfoff = K_iters - {4..120}, the always-emit path issues 4..120 extra
prefetches per WG per K-loop. On 4096-row M shapes with K=32768, this adds
~120 wasted prefetches per iter against a kernel that only had ~2.7 iters/WG
of slack — net effect: an in-flight prefetch is still consuming the LDS slot
when the prologue/epilogue Bl direct-load reads it. finite drops from 0.66 to
0.17. These shapes need a different fix (e.g. emit only L2-only prefetches on
tail iters, never write LDS).

## Recommended next steps

1. **Adopt R38B selectively**: enable for the 3 shapes that became WIN (16384x4096
   x3072 → 3500 TFLOPS @ 100.2%, 32768x6144x2048 → 3275 @ 101.1%, and re-evaluate
   16384x4096x4096 which is the only true WIN→WIN).

2. **Variant fork for 9 CRASH shapes**: instead of changing kernel semantics,
   keep R25-C optimization but pick a non-`pfoff` variant for these 9 shapes from
   the R25 sweep (per shape, ~3% perf loss vs the CRASH variant but 0 risk).

3. **Fix B3 candidate (untested)**: emit prefetches on tail iters but route them
   through `emit_full_pf_l2only<>` (cache warm, no LDS write). Keeps the
   wait-count metadata consistent (the buffer_load issues; the LDS write is
   suppressed) without overwriting in-use double-buffer slots. This is the right
   fix in principle and should restore both R37's WINs and the 9 CRASH→OK shapes.

## Files

- `kernel_mxfp4_gluon_cpp.cpp` — Fix B2 implementation, default OFF (set
  `-DR38B_TAIL_FIX=1` to enable)
- `build_R38B.py` — builder
- `bench_all_42_R38B.py` — bench harness
- `bench_all42_results_R38_optB.json` — full bench
- `R38_OPT_B_PROGRESS.md` — detailed progress log
- `R38B_BENCH_RUN.log` — bench output
- `build_R38B/` — 30 unique .so binaries
