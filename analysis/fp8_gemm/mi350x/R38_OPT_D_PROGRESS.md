# R38 Opt D — Progress Log

## Plan
For each of the 19 R37 WRONG_OUTPUT shapes, find an alternate variant from
the R25 sweep DB that:
  1. Passes the fused-step34 correctness gate (kernel_finite ≥ 0.995 with R37_FIX_B=1).
  2. Comes close to R37 BEST_VARIANTS' theoretical perf.

## Steps
1. **Picked candidates** (`R38D_pick_candidates.py`, output `R38D_candidates.json`):
   loaded `bench_all42_results_R25_FINAL.json` (169 variants × 42 shapes), filtered
   per-variant TFLOPS by excluding variants whose name contained `memc`, `dc`, or `tv0`,
   selected top 10 by TFLOPS for each of the 19 WRONG shapes. All 19 had ≥1 candidate.

2. **Built 176 unique modules** (`build_R38D.py` → `build_R38D/`). Used `strip_bad_sched`
   from `build_R37.py` (drops `-mllvm -amdgpu-sched-strategy=max-memory-clause` from
   variant flags). All 176 builds succeeded in 58s. Manifest: `R38D_BUILD_MANIFEST.json`.

3. **Benched 190 candidate variants** across 8 GPUs (warmup=200, iters=500, trim=10%,
   gate kernel_finite ≥ 0.995). Elapsed 3.4 min. Result: only 4/19 shapes had ANY
   correct variant. 2 of those WIN over comp; 2 LOSS-but-correct; 15 entirely WRONG.
   Output: `bench_all42_results_R38_optD.json`, log `R38D_BENCH.log`.

   Per-shape outcome from R38D:
   - WIN: (16384, 4096, 2048) `ts_v12_tv16` 108.6%; (32768, 4096, 3072) `v32` 100.5%
   - LOSS_CORRECT: (4096, 32768, 4096) `lgk2_v16` 92.4%; (6144, 32768, 4096) `ts_lgk2_v24` 95.2%
   - 15 shapes: ALL 10 candidate variants WRONG_OUTPUT (finite ≈ 0.0 to 0.99)

4. **Diagnosed**: even the simplest variants (`default`, `ts`, `gm8`, `u16`) produce
   finite < 0.07 on K=32768 shapes. This rules out the named scheduler-aggressive flags
   as root cause. Hypothesis: R37_FIX_B / fused-step34 path itself produces bf16-saturated
   accumulators on the uniform-(-4) probe at large K.

5. **Tried alternate compile paths** (`build_R38D_v2.py` → `build_R38Dv2/`,
   `bench_all_42_R38Dv2.py` → `bench_all42_results_R38_optDv2.json`):
   - Axis `f34`: append `-DFUSED_STEP34=1` (selects FUSED_STEP34 path).
   - Axis `leg`: append `-DR37_FIX_B=0` (legacy non-fused path).

   Built 152 unique modules. Benched 170 tasks (top-5 candidates × 2 axes × ~17 shapes).
   Result: only 3/17 shapes recovered, ALL with `f34` axis, ALL LOSS_CORRECT (87.7-95.2%).
   The legacy axis recovered nothing. The `f34` axis recovered:
   - (4096, 32768, 4096) `lgk2_v16` 92.6% (already had a R38D fix)
   - (6144, 32768, 4096) `ts_v4_tv16` 95.0% (already had a R38D fix; R38D tag wins)
   - (16384, 4096, 14336) `ts_gm8_v12_btw_all` 87.7% (NEW LOSS_CORRECT)

## Final v2 selection
- 14 R37 WIN kept untouched
-  2 R38D NEW WIN: (16384, 4096, 2048), (32768, 4096, 3072)
-  3 R38D LOSS_CORRECT: (4096, 32768, 4096), (6144, 32768, 4096), (16384, 4096, 14336)
- 14 R37 WRONG_OUTPUT not recoverable — R37 BEST_VARIANTS retained as fallback.
-  9 R37 CRASH untouched (out of scope for R38 Opt D).

→ Total CORRECT under uniform gate: 19/42 (was 14/42 in R37) — net +5 correct, +2 WIN.

## Files
- `R38D_pick_candidates.py`, `R38D_candidates.json`
- `build_R38D.py`, `build_R38D/`, `R38D_BUILD_MANIFEST.json`
- `bench_all_42_R38D.py`, `bench_all42_results_R38_optD.json`, `R38D_BENCH.log`
- `build_R38D_v2.py`, `build_R38Dv2/`, `R38Dv2_BUILD_MANIFEST.json`
- `bench_all_42_R38Dv2.py`, `bench_all42_results_R38_optDv2.json`, `R38Dv2_BENCH.log`
- `R38_BEST_VARIANTS_v2.json` (full per-shape detail)
- `R38_BEST_VARIANTS_v2.py` (drop-in replacement BEST_VARIANTS dict)
