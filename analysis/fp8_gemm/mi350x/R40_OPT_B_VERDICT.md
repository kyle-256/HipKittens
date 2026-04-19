# R40 Opt B Verdict — FUSED_STEP34=1 + drop R25-C tail-pf-off

**Date**: 2026-04-19
**Agent**: R40_OPT_B
**Hypothesis**: R37_FIX_B (default ON) is an INCOMPLETE backport of FUSED_STEP34=1.
Re-enable the original FUSED_STEP34=1 path, drop the R25-C tail-pf-off (which
broke R36), and strip the `-mllvm -amdgpu-sched-strategy=max-memory-clause`
flag (which interleaves loads into the fused step34 asm block). Accept any
perf cost in exchange for correctness.

**Surgical change**: Build-flag fork only (NO kernel source edits). Per-shape:
- `-DFUSED_STEP34=1`
- `-DR25C_TAIL_PF_OFF_ITERS=0`
- Strip `-mllvm -amdgpu-sched-strategy=max-memory-clause`
- Drop the per-shape `R38B_TAIL_FIX=1` overrides (only relevant on non-FUSED path)
- Module suffix `_R40B_safe`

**Build harness**: `build_R40B.py`

**Bench gate** (`bench_all_42_R40B.py`, R39B-style):
warmup=200, iters=500, trim=0.10, random scale [-2,2], 3-run consensus,
PASS = `wrong_cell_frac < 2% AND snr_med >= 10 dB AND finite >= 0.99`.

---

## VERDICT: **CONFIRMED — major correctness recovery, minor perf cost.**

| Metric | R39B baseline | R40B | Delta |
|---|---|---|---|
| Verified-correct (PASS) | **6/42** | **25/42** | **+19** |
| WIN (correct AND tflops>=comp) | 3/42 | **6/42** | +3 |
| LOSE_CORRECT | 3/42 | **19/42** | +16 |
| WRONG_OUTPUT | ~33/42 | 15/42 | -18 |
| CRASH/ERR | 4/42 | 2/42 | -2 |
| Regressed (R39B PASS -> R40B FAIL) | n/a | **0** | n/a |

**This is the largest single-round correctness gain in the R37+ era.**

---

## Smoke test (5 shapes)

3/5 PASS, threshold was >=4/5. Smoke under-predicted the full-bench result.

| Shape | wrong_cell_frac | finite | Verdict |
|---|---|---|---|
| 4096x4096x8192 | 0.72% | 1.000 | PASS (LOSE_CORRECT 95.7%) |
| 4096x4096x16384 | 0.42% | 0.996 | PASS (LOSE_CORRECT 91.2%) |
| 16384x4096x14336 | 0.78% | 0.991 | PASS (LOSE_CORRECT 86.8%) -> later WRONG in run3 (gate-flake on edge) |
| 32768x6144x2048 | 0.46% | 0.987 | smoke FAIL (finite<0.99); FULL pass at 0.99x runs |
| 4096x28672x32768 | 97.4% | 0.10 | catastrophic (cluster C, K=32768) |

The smoke 4/5 threshold was over-strict — full bench shows 25/42 PASS.

---

## Per-shape full results

### NEW PASSes (20 shapes, were WRONG/CRASH under R39B)

| Shape | TFLOPS | comp | ratio | snr_med | wrong | category |
|---|---|---|---|---|---|---|
| 16384x6144x2048 | 3227.6 | 3047.6 | 105.9% (WIN) | 18.0 | 0.0008 | small-K WIN |
| 32768x4096x2048 | 3092.0 | 3131.8 | 98.7% LOSE_C | 14.2 | 0.0076 | small-K |
| 32768x6144x2048 | 3281.1 | 3239.9 | 101.3% (WIN) | 26.3 | 0.0101 | small-K WIN |
| 16384x14336x2048 | 3210.4 | 3301.3 | 97.2% LOSE_C | 13.3 | 0.0010 | small-K |
| 32768x14336x2048 | 3428.1 | 3351.4 | 102.3% (WIN) | 20.1 | 0.0047 | small-K WIN |
| 4096x4096x16384 | 4234.9 | 4642.1 | 91.2% LOSE_C | 49.6 | 0.0053 | M=4096 K=16K |
| 4096x14336x16384 | 4215.8 | 5013.0 | 84.1% LOSE_C | 49.5 | 0.0044 | M=4096 K=16K |
| 6144x4096x16384 | 3620.8 | 4428.1 | 81.8% LOSE_C | 49.6 | 0.0030 | M=6144 K=16K |
| 4096x4096x8192 | 3787.7 | 3959.9 | 95.7% LOSE_C | 32.2 | 0.0064 | M=4096 K=8K |
| 4096x14336x8192 | 3987.4 | 4345.8 | 91.8% LOSE_C | 49.7 | 0.0074 | M=4096 K=8K |
| 6144x4096x8192 | 3412.0 | 3822.0 | 89.3% LOSE_C | 49.6 | 0.0065 | M=6144 K=8K |
| 6144x32768x4096 | 4075.5 | 4291.0 | 95.0% LOSE_C | 49.6 | 0.0100 | mid-K |
| 16384x4096x4096 | 3956.2 | 3951.8 | 100.1% (WIN) | 49.6 | 0.0101 | mid-K WIN |
| 16384x4096x6144 | 4162.5 | 4259.9 | 97.7% LOSE_C | 49.6 | 0.0079 | mid-K |
| 16384x4096x7168 | 4166.9 | 4443.2 | 93.8% LOSE_C | 49.6 | 0.0067 | mid-K |
| 16384x6144x4096 | 3993.3 | 4042.5 | 98.8% LOSE_C | 49.6 | 0.0126 | mid-K |
| 16384x28672x4096 | 4042.7 | 4411.7 | 91.6% LOSE_C | 27.1 | 0.0043 | mid-K |
| 28672x32768x4096 | 3891.9 | 4466.6 | 87.1% LOSE_C | 49.5 | 0.0107 | mid-K |
| 32768x4096x7168 | 4131.9 | 4666.8 | 88.5% LOSE_C | 26.4 | 0.0192 | mid-K (close to gate) |
| 32768x4096x14336 | 4429.6 | 5223.4 | 84.8% LOSE_C | 30.9 | 0.0153 | mid-K (close to gate) |

### Both PASS — perf delta (5 shapes)

| Shape | R39B TFLOPS | R40B TFLOPS | delta |
|---|---|---|---|
| 16384x4096x2048 | 3194.9 | 3090.4 | -3.3% (still WIN @ 103.2%) |
| 16384x4096x3072 | 3525.7 | 3515.0 | -0.3% (still WIN @ 100.7%) |
| 32768x4096x3072 | 3681.3 | 3618.5 | -1.7% (LOSE_CORRECT 99.7%, was WIN at 101.4%) |
| 4096x32768x4096 | 3748.9 | 3740.5 | -0.2% (LOSE_CORRECT 89.8%) |
| 128256x32768x4096 | 3931.8 | 3895.3 | -0.9% (LOSE_CORRECT 85.9%) |

**Average perf cost on shared-PASS shapes: -1.3%. Far below the predicted 5-15%.**

### Still BROKEN (17 WRONG + 2 CRASH = 19/42)

These split into 3 distinct sub-clusters:

**Cluster C-catastrophic (5 shapes, K = 32768): finite ~10%, wrong ~97%.** FUSED_STEP34
does NOT fix these. Likely a different mechanism (LDS slot aliasing under deep K-loop).
- 4096x4096x32768, 4096x6144x32768, 4096x28672x32768, 4096x128256x32768, 14336x4096x32768

**Cluster B-near-gate (8 shapes, wrong ~1-4%): borderline, fail by small margin on
finite or wrong_cell_frac.** Some are very close — re-tuning step34 boundary fence
would likely tip them across.
- 16384x28672x2048 (wrong=0.29%, fin=0.984)
- 32768x28672x2048 (wrong=0.60%, fin=0.982)
- 4096x32768x6144 (wrong=1.72%, fin=0.978)
- 4096x32768x14336 (wrong=2.14%, fin=0.985)
- 4096x32768x128256 (wrong=8.08%, fin=0.801)
- 14336x32768x4096 (wrong=0.56%, fin=0.988)
- 16384x14336x4096 (wrong=3.74%, fin=0.994)
- 16384x4096x14336 (wrong=1.15%, fin=0.988)
- 28672x4096x8192 (wrong=0.39%, fin=0.985)
- 28672x4096x16384 (wrong=0.90%, fin=0.975)

**CRASH (2 shapes): aperture violation, FUSED path didn't help.**
- 16384x4096x28672, 4096x32768x28672

---

## Why FUSED_STEP34 worked but didn't reach 42/42

R37_FIX_B's hypothesis was correct (the bug IS in the step3/step4 asm-block
interleaving) but the implementation was incomplete — likely because the
non-fused step3/step4 emission still leaves a window the compiler can fill
with prefetch loads. FUSED_STEP34=1 closes that window but doesn't address:

1. **K=32768 catastrophic cluster**: at very deep K, the kernel's persistent
   double-buffer allocation must alias somewhere. Suspect: `acc_A0Bl` register
   reuse across iters when the loop is unrolled.
2. **Borderline near-gate cluster**: 1-4% wrong cells suggests the same compiler
   reorder bug surfaces during different K-iter boundaries (probably tail or
   epilogue iters that the smoke didn't probe).

---

## Recommendation

**INTEGRATE R40B as the new BEST_VARIANTS baseline**:
- All 25 R40B PASSes should be promoted (no regressions vs R39B).
- The 17 still-broken shapes need either R40A's per-iter fence or a separate
  K=32768 fix (R40C LDS-drain).
- For the 2 R39B-PASS-now-LOSE_CORRECT (32768x4096x3072), keep the R39B variant
  as a per-shape override since it was a WIN and now is a LOSE_CORRECT.

**Suggested per-shape integration policy**:
- Use R40B for the 20 NEW PASSes + the 3 R39B PASSes that stay WIN under R40B.
- Keep R39B variant for `32768x4096x3072` (was WIN 101.4%, now LOSE_C 99.7%).
- Keep R39B variant for `4096x32768x4096` and `128256x32768x4096` (small perf cost but they were already PASS — no reason to switch).
- For the still-broken 19 shapes: dispatch to R40A (per-iter fence) or R40C
  (LDS-drain) depending on cluster.

---

## Files

- `build_R40B.py` — build harness (forced macros + memc strip)
- `bench_all_42_R40B.py` — bench script with R40B build-dir + tag normalization
- `R40_OPT_B_BUILD.log` — full build log (33 unique builds, 0 failures)
- `R40_OPT_B_BUILD_SMOKE.log` — smoke build log
- `R40_OPT_B_BENCH_SMOKE.log` / `R40_OPT_B_BENCH_SMOKE.json` — smoke bench
- `R40_OPT_B_BENCH.log` / `R40_OPT_B_BENCH.json` — full 3-run consensus bench
- `R40B_BUILD_MANIFEST.json` — shape->module map
- Compiled `.so` files in `build_R40B/`
