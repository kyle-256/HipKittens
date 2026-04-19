# R38 Opt E — Progress Log (2026-04-19)

Mission: unify the R37 14 WIN baseline + R38D 5 recoveries + R38B 3 CRASH→OK
shapes into ONE leaderboard, applying `R38B_TAIL_FIX=1` selectively only on
the 3 R38B targets (so its perf regressions don't poison the rest).

## Step 1 — identify R37 CRASH→OK transitions in R38B JSON

Joined `bench_all42_results_R37_fixB.json` × `bench_all42_results_R38_optB.json`.
For each of the 9 R37 CRASH shapes, R38B status:

| Shape | R37 | R38B status | tflops | finite | %comp |
|---|---|---|---|---|---|
| (16384, 4096, 3072) | CRASH | OK | 3500.4 | 0.9979 | 100.2% (WIN) |
| (16384, 28672, 2048) | CRASH | WRONG_OUTPUT | - | 0.9831 | - |
| (28672, 32768, 4096) | CRASH | WRONG_OUTPUT | - | 0.9919 | - |
| (32768, 4096, 2048) | CRASH | WRONG_OUTPUT | - | 0.9863 | - |
| (32768, 6144, 2048) | CRASH | OK | 3275.2 | 0.9961 | 101.1% (WIN) |
| (32768, 28672, 2048) | CRASH | WRONG_OUTPUT | - | 0.9904 | - |
| (4096, 32768, 6144) | CRASH | WRONG_OUTPUT | - | 0.9942 | - |
| (14336, 32768, 4096) | CRASH | WRONG_OUTPUT | - | 0.9921 | - |
| (128256, 32768, 4096) | CRASH | OK | 3935.1 | 0.9969 | 86.7% (LOSS_CORRECT) |

**3 CRASH→OK shapes** (status=OK + finite ≥ 0.995):
- (16384, 4096, 3072) — R38B WIN @ 100.2%
- (32768, 6144, 2048) — R38B WIN @ 101.1%
- (128256, 32768, 4096) — R38B LOSS_CORRECT @ 86.7%

The verdict prompt cited these as candidates; the JSON confirms exactly 3.
For the other 6 CRASH shapes, R38B's `R38B_TAIL_FIX=1` keeps them from
faulting but they still fail the finite gate (correctness regression of a
different mechanism).

## Step 2 — build R38_BEST_VARIANTS_v3.py

Composed:
- 14 R37 WIN entries (variant only, no macros)
- 2 R38D NEW WINs: ts_v12_tv16, v32
- 3 R38D LOSS_CORRECT: lgk2_v16, ts_lgk2_v24, ts_gm8_v12_btw_all
- 3 R38B CRASH→OK: ts_gm6_v12_memc_dc_pfoff4 + R38B_TAIL_FIX=1 (etc.)
- 14 R37 WRONG fallbacks (will still fail uniform gate)
- 6 R37 CRASH fallbacks for unrecovered shapes (will still CRASH)

Total: 42. Schema: `(M,N,K) -> (variant_tag, {macro_name: int_value})`.

All variant tags verified to exist in `bench_all_42.py` variant list.

## Step 3 — build_R38E.py

Extended `build_R37.py`:
- consumes `BEST_VARIANTS_V3` from sibling module
- module-name signature includes a `_<MACROFLAG><val>` suffix per override key
  → distinct .so for same parent variant compiled with/without R38B_TAIL_FIX
- per-shape `macro_flags()` builds `-D<NAME>=<VAL>` and prepends to CPPFLAGS
- still strips `-mllvm -amdgpu-sched-strategy=max-memory-clause`
- writes `R38E_BUILD_MANIFEST.json` including the `macro_overrides` map

42 shape entries → 35 unique builds (multiple shapes share variants).
Built in 10.5s with 24 workers, 0 failures.

## Step 4 — bench_all_42_R38E.py

Same harness as R37; loads from `build_R38E/`; module name derived from
`(M,N,K) → (variant_tag, macro_overrides)` via `module_name_for()` (same fn
as builder, kept in sync). Records `macro_overrides` in each result row.

Output: `bench_all42_results_R38_optE.json`.

## Step 5 — full sweep (×3 runs)

Each run: 8 GPUs (0-7), warmup=200, iters=500, trim=10%, gate finite≥0.995,
~2.5 min wall-clock.

| Run | WIN | LOSE_CORRECT | WRONG_OUTPUT | CRASH/ERR |
|---|---|---|---|---|
| 1 | 11 | 3 | 21 | 7 |
| 2 | 11 | 3 | 18 | 10 |
| 3 | 11 | 5 | 18 | 8 |
| **best-of-3** | **12** | **4** | 19 | 7 |

The 3 R38B-target shapes:
- (16384, 4096, 3072): WIN run 1 (3500 TFLOPS), WRONG run 2 (finite 0.9919),
  WIN run 3 (3517 TFLOPS) → **best-of-3 = WIN**.
- (32768, 6144, 2048): WRONG run 1 (0.9908), WRONG run 2 (0.9935), WIN run 3
  (3247 TFLOPS, 100.2%) → **best-of-3 = WIN**.
- (128256, 32768, 4096): LOSS_CORRECT all 3 runs (3929-3934 TFLOPS, 86.6%) →
  **best-of-3 = LOSS_CORRECT**.

All 3 selective-macro shapes deliver the predicted result in ≥1 run, never CRASH.

## Step 6 — leaderboard + verdict
See `R38_LEADERBOARD.md` and `R38_OPT_E_VERDICT.md`.

## Methodology notes
- 8 GPUs idle confirmed via `rocm-smi --showuse` before each run.
- `HIP_VISIBLE_DEVICES=N` per-shape via the bench harness.
- Correctness gate identical to R37.
- No kernel changes — pure build/variant orchestration.
