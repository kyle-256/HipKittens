# R38 Opt F — VERDICT

**Status: HYPOTHESIS REFUTED. Drain-based fix at the iter boundary does NOT
recover correctness, with or without R38B always-emit. Identical regression
pattern to R38B/C: 0 WIN, ~13 LOSE, ~12 WRONG_OUTPUT.**

## Headline (12-15 reps each variant on canonical CRASH shape)

| Variant            | finite range | Pass gate? |
|---|---|---|
| R37 baseline       | n/a (CRASH on rep 0) | NO (CRASH) |
| R38B alone         | 0.987–0.992 | NO |
| R38B + R38F1 (vmcnt(0))            | 0.989–0.994 | NO |
| R38B + R38F2 (vmcnt(0)+s_barrier)  | 0.989–0.991 | NO |
| R38B + R38F3 (s_waitcnt 0)         | 0.992–0.995 | NO (a few reps barely brushed) |
| R38B + R38F4 (vmcnt(0)+lgkmcnt(0)) | 0.983–0.987 | NO |

R38F WITHOUT R38B (drain alone, gated to R25-C tail iters): all four variants
CRASH (HSA aperture violation), confirming R38F alone is structurally
insufficient — the underlying CRASH must first be closed by R38B (always-emit).

## Full bench (R38B + R38F variant 3, 25 unique shapes from CRASH+WIN union)

| Outcome | Count |
|---|---|
| WIN | 0 |
| LOSE | 13 |
| WRONG_OUTPUT | 12 |
| CRASH | 0 |

This is essentially identical to R38B alone (R37_LEADERBOARD): F3 closes CRASH
(via R38B layering) but the drain at top-of-tail-iter does NOT improve the
finite ratio above the 0.995 gate.

## Root-cause re-analysis

The R38C verdict's hypothesis — "in-flight buffer_load_to_lds hasn't drained
before next iter's s_barrier releases the LDS slot" — is not the operative
bug, or at minimum cannot be fixed by inserting drains at the iter boundary.
Strong evidence:

1. F3 (`s_waitcnt 0` — drain vmcnt + lgkmcnt + expcnt, the most aggressive
   drain possible) does NOT bring finite up to 0.995. If the bug were a stale
   GMEM→LDS load racing the next-iter consumer, `s_waitcnt 0` would close it.
2. Adding the drain at TOP of tail iter (consumer-side) has the same
   regression pattern as adding it at BOTTOM of tail iter (producer-side).
3. The finite shortfall is ~0.5–2% — consistent across reps (deterministic),
   not the timing-jitter signature of a true race. This matches the project
   memo `MXFP4 17%-deterministic-wrong cells` (~17% wrong, deterministic).

Likely real root cause is **arithmetic / scale-pack mismatch** on tail iters,
not LDS race:
- `load_pq_scale_x2_async(... bt+1 ...)` continues advancing the scale index
  even on tail iters where the prefetched data tile is clamped to
  `pf_bt = k_byte_iters - 1`.
- The MFMA in iter N+1 then multiplies a stale-but-valid data tile by a
  scale that points past end-of-K → arithmetic overflow → BF16-write garbage
  → ~17% deterministic-wrong cells.
- A drain cannot fix this. The fix is either to clamp the scale index in tail
  iters or to drop R25-C entirely on shapes where R37_FIX_B is active.

## Recommended next steps for the orchestrator

1. **Drop R38F as a path**. The drain-based mechanism is structurally
   insufficient.
2. **Pivot to R39**: investigate `load_pq_scale_x2_async` clamping on tail
   iters (`R39_TAIL_SCALE_CLAMP`). Hypothesis: scale_idx clamps to
   `min(bt+1, k_byte_iters - 1)` matching the data clamp.
3. **Or**: drop `_pfoff*` from BEST_VARIANTS for any shape where R37_FIX_B +
   R25-C produces `finite < 0.995`. Accept ~3% perf regression to recover
   correctness. R38C verdict's recommendation #3.

## Files

- `kernel_mxfp4_gluon_cpp.cpp` — adds `R38F_TAIL_DRAIN` (default 0) +
  `R38F_VARIANT` (default 2). Drain inserted at the TOP of each iter inside
  the R37_FIX_B path, gated on `_r38f_in_tail`. Mutex with R38C only;
  designed to be co-enabled with R38B.
- `build_R38F.py` — builds `_R38F{1..4}` modules with R38B + R38F co-enabled.
  Default `--variant=2`; tested all four.
- `bench_all_42_R38F.py` — bench harness loading from `build_R38F3/`. Output
  JSON: `bench_all42_results_R38_optF.json`.
- `R38F_smoke_one.py` — single-shape smoke (canonical CRASH variant), takes
  variant + reps args.
- `R38F1_BUILD_MANIFEST.json` ... `R38F4_BUILD_MANIFEST.json` — per-variant
  build manifests.
- `R38F_BENCH_RUN.log` — full bench log.
- `bench_all42_results_R38_optF.json` — bench results (R38F variant 3).
- `R38_OPT_F_PROGRESS.md` — running notes during exploration.
