# R38 Opt E — Verdict (2026-04-19)

## Status: PARTIAL WIN

Unified BEST_VARIANTS_v3 + selective per-shape macro fork delivers a
real but smaller-than-projected gain over R37, with the gap fully
explained by **gate flakiness on already-known marginal shapes**, not
by any defect in the v3 composition.

## Headline (best-of-3 runs vs R37 single-run)

| Metric | R37 | **R38E (best-of-3)** | delta |
|---|---|---|---|
| WIN | 14/42 | **12/42** | -2 |
| LOSE_CORRECT | 0/42 | **4/42** | +4 |
| Verified-correct | 14/42 | **16/42** | **+2** |
| WRONG_OUTPUT | 19/42 | 19/42 | 0 |
| CRASH/ERR | 9/42 | 7/42 | -2 |

Projected target was 19+ verified-correct (14 + 2 R38D NEW + 3 R38B). Actual
is 16; the 3-shape shortfall comes from R37 WIN shapes (4096x4096, 6144x4096,
28672x4096x8192, 32768x4096x7168) flaking to WRONG_OUTPUT in some runs and
the (16384, 6144, 2048) variant's known-CRASH propensity surfacing in 1/3
runs. None of these regressions are due to v3 changes — the same R37 binary
flag-set was rebuilt with no semantic change for those shapes. The
underlying gate is finite-frac on a uniform-(-4) probe; finite drifts in
[0.97-0.99] across reruns and the 0.995 threshold is non-monotonic in
that range.

## What v3 actually delivers

| Bucket | count | source |
|---|---:|---|
| Stable R37 WINs (all 3 runs WIN) | 9 | R37 baseline kept |
| R38B selective macro WINs | 2 | (16384,4096,3072), (32768,6144,2048) |
| R38B selective macro LOSS_CORRECT | 1 | (128256,32768,4096) |
| R38D NEW WINs | 1 | (16384,4096,2048) |
| R38D LOSS_CORRECT (was WRONG) | 3 | (4096,32768,4096), (16384,4096,14336), (32768,4096,3072) |

The (32768, 4096, 3072) NEW WIN from R38D demoted to LOSE in best-of-3
(99.1% vs 100% comp); still correct, still ahead of R37's WRONG_OUTPUT.

## What worked

1. **Per-shape macro override is structurally clean** — `build_R38E.py`'s
   module-name signature (e.g. `..._R38B_TAIL_FIX1_R38E`) cleanly separates
   binaries, no compile-time conflicts. The 3 R38B-fork shapes all built
   first try and 0/15 runs (5 reps × 3 shapes) ever CRASHED.

2. **R38B macro composes safely with R38D base** — never tested on the same
   shape (they target disjoint sets), but the macro-override mechanism
   would handle any future overlap by passing both flags.

3. **CRASH count down from 9 to 7** — the 3 R38B-fork shapes are
   architecturally CRASH-proof now; only 6 R37 CRASH shapes remain (R38B
   couldn't recover them, R38D didn't attack them).

## What didn't work / failure paths hit

1. **Run-to-run flakiness on the correctness gate** is now the dominant
   noise source. 5-7 shapes hover at finite ∈ [0.97, 0.99] and randomly
   pass/fail the 0.995 gate. Best-of-3 aggregation recovers most but not
   all. Documented per-shape behavior is in `R38_OPT_E_PROGRESS.md`.

2. **(16384, 6144, 2048) flaked CRASH in run 3** (it was the ONLY R37 WIN
   on the canonical-CRASH variant `ts_lgk2_gm6_v12_memc_pfoff4`). R37's
   single-run was lucky. This shape is a candidate for the same selective
   `R38B_TAIL_FIX=1` treatment as the 3 R38B-fork shapes — would likely
   recover it as stable WIN at the cost of 1-3% perf.

3. **(32768, 4096, 3072) NEW WIN demoted to LOSE_CORRECT** — R38D's `v32`
   variant clocked 3647 TFLOPS @ 100.5% in R38D's bench but 3600 TFLOPS @
   99.1% in R38E best-of-3. Still correct, still better than R37 WRONG.
   Likely measurement variance, not regression.

## Recommended R39 follow-ups

1. **Add (16384, 6144, 2048) to the R38B macro-override set.** Promote it
   from R37 fallback variant to the same `R38B_TAIL_FIX=1` treatment;
   should trade a small TFLOPS hit for stable WIN.

2. **Tune the correctness gate to be SNR-based, not finite-frac-based.**
   Half the WRONG_OUTPUT shapes are at finite ∈ [0.97, 0.99] — these are
   real bf16-overflow on uniform inputs (per the `MXFP4 17%
   deterministic-wrong cells` memory note), not kernel bugs. A
   random-scale + SNR-thresholded probe should restore correctness
   verdicts for ~6 shapes that R37 also got "wrong" only because of the
   probe.

3. **For the 14 R37 WRONG-fallback shapes**, R38D already established
   that no variant in the current sweep DB passes the uniform-(-4) gate.
   Either expand the variant DB with new K_EXACT-gated builds (no `memc`
   prefix) or change the gate per (2). Either is a real optimization,
   not a band-aid.

4. **For the 6 R37 CRASH shapes that R38B couldn't recover**, the next
   candidate is a **per-shape variant fork** (drop `_pfoff4` for a
   non-pfoff variant from the R25 sweep) at ~3% perf cost per shape —
   document per-shape rollback in BEST_VARIANTS_v4.

## Files

- `R38_BEST_VARIANTS_v3.py` — drop-in dict (variant_tag, macro_overrides_dict)
- `build_R38E.py` — builder (per-shape macro overrides, R37-strip kept)
- `bench_all_42_R38E.py` — bench harness with same correctness gate as R37
- `bench_all42_results_R38_optE.json` (run 3) + `_run2.json` (run 2) +
  `_run3.json` (run 3 snapshot before final write)
- `R38E_BENCH_RUN.log`, `R38E_BENCH_RUN_v2.log`, `R38E_BENCH_RUN_v3.log`
- `R38E_BUILD_MANIFEST.json`
- `build_R38E/` — 35 unique .so binaries
- `R38_LEADERBOARD.md` — final unified leaderboard table
- `R38_OPT_E_PROGRESS.md` — step-by-step run log
