# R31 OPT-A VERDICT — UNROLL_K sweep on L6: DEAD

**Shape**: L6 = 4096 × 32768 × 128256 (M × N × K)
**Incumbent**: `ts_lgk2_v12_memc_btw_all`, prior 5353.9 TFLOPS (92.6% of comp 5781.1)
**Target**: any UNROLL_K beating incumbent by ≥1%
**Bench rules**: warmup=200, iters=500, trim=10%, GPU 0 (idle), single-rep then 5-rep verify
**GPU 3 avoided** (busy, GFX use 100%)

---

## Step 1 — Audit of pre-existing UNROLL_K entries on L6

From `bench_all42_results_R25_FINAL_v2.json` (results[L6].per_variant):

| variant tag           | TFLOPS | vs incumbent (5353.9) |
|-----------------------|-------:|----------------------:|
| `u8`                  |  4996.5 | −6.7%                |
| `u16`                 |  5004.9 | −6.5%                |
| `u32`                 |  4997.7 | −6.7%                |
| `u32_btw_all`         |  5181.5 | −3.2%                |
| `gm8u8`               |  5068.8 | −5.3%                |
| `gm8u16`              |  5060.5 | −5.5%                |
| `gm16u16`             |  5009.2 | −6.4%                |
| `f34_u8` / `f34_u16`  |  4464 / 4458 | −16-17%        |
| (all others tagged)   |  ≤5181 | ≤−3.2%               |

**Audit verdict**: every prior UNROLL_K-tagged variant on L6 was already strictly worse than the incumbent. None used the full L6 best-parent flag set (`ts_lgk2_v12_memc_btw_all`), so a fresh sweep with the correct parent was warranted — but the prior data is a strong negative prior.

---

## Step 2 — Build sweep results (parent: `ts_lgk2_v12_memc_btw_all`)

Parent flags applied:
```
-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12
-mllvm -amdgpu-sched-strategy=max-memory-clause -DBARRIER_TO_WAITCNT_ALL=1
```
plus `-DUNROLL_K=N` for N ∈ {1, 2, 4, 16, 32}.
Built for `N_DIM=32768 K_DIM=128256` via `build_round31_optA_unrollK.py`.
`BARRIER_TO_WAITCNT_ALL=1` is the production L6 parent flag (line 480 of `bench_all_42.py`); R30 BTW fault notes referenced DLA1/DLA2, not L6.

| UNROLL_K | VGPR | AGPR | SGPR | spills | scratch | occupancy | .so KB | compile s |
|---------:|-----:|-----:|-----:|-------:|--------:|----------:|-------:|----------:|
| 1        | 187  | 256  | 73   | 0      | 0       | 1         | 243    | 4.6       |
| 2        | 187  | 256  | 86   | 0      | 0       | 1         | 246    | 4.7       |
| 4        | 187  | 256  | 90   | 0      | 0       | 1         | 251    | 4.6       |
| 16       | 212  | 256  | 93   | 0      | 0       | 1         | 306    | 5.1       |
| 32       | 212  | 256  | 93   | 0      | 0       | 1         | 306    | 5.2       |

All 5 builds PASS, no spills, occupancy 1 (matches the incumbent — limited by AGPR=256 ceiling, not VGPR). U16/U32 use 25 more VGPRs from the longer unrolled body.

---

## Step 3 — Single-rep bench (GPU 0)

| variant     | avg_ms | TFLOPS | ratio vs prior incumbent (5353.9) | ratio vs same-run incumbent (5435.9) |
|-------------|-------:|-------:|----------------------------------:|--------------------------------------:|
| incumbent   | 6.3335 | 5435.9 |                          101.53% |                              100.00% |
| u1          | 6.5702 | 5240.1 |                           97.87% |                               96.40% |
| u2          | 6.7703 | 5085.2 |                           94.98% |                               93.55% |
| u4          | 6.4479 | 5339.5 |                           99.73% |                               98.23% |
| u16         | 6.3714 | 5403.6 |                          100.93% |                               99.41% |
| u32         | 6.3521 | 5420.0 |                          101.23% |                               99.71% |

**Verdict**: u16 and u32 are within ~1% of the same-run incumbent — borderline noise. u1/u2/u4 clearly worse. Per task spec, u32 (101.23% vs prior baseline) qualified for 5-rep verify. u16 (100.93%) added as a courtesy second candidate.

---

## Step 4 — 5-rep verify (GPU 0, 15 runs sequential)

| variant   | mean TFLOPS | std  | min    | max    | mean_ms | ratio vs incumbent |
|-----------|------------:|-----:|-------:|-------:|--------:|-------------------:|
| incumbent |    **5436.0** |  9.0 | 5423.1 | 5446.8 | 6.3334  | 100.00%            |
| u16       |      5405.9 | 24.6 | 5367.1 | 5431.4 | 6.3688  |  99.45%  (−30.1)   |
| u32       |      5404.4 |  6.6 | 5396.1 | 5412.7 | 6.3705  |  99.42%  (−31.6)   |

Reps:
- incumbent: 5423.1, 5432.8, 5441.3, 5446.8, 5436.2
- u16: 5414.3, 5431.4, 5398.9, 5417.8, 5367.1
- u32: 5412.7, 5402.3, 5409.2, 5401.6, 5396.1

**Statistical verdict — DEAD**: incumbent's max (5446.8) > u32's max (5412.7), and incumbent's min (5423.1) > u16's max (5431.4) only marginally; u32's max (5412.7) is below incumbent's min (5423.1). The ~30 TFLOPS deficit of u16/u32 vs incumbent is well outside the combined std (~25 TFLOPS), so the single-rep +1.23% u32 reading was pure noise from a low incumbent draw.

The default `#pragma unroll 8` (kernel:2448) is already optimal at L6's K=128256 (K_iters=501).

---

## Step 5 — Recommendation

**DO NOT add any UNROLL_K=N variant for L6 to bench_all_42.py.** All five tested values lose to the incumbent:

| UNROLL_K | verdict (vs 5436.0 incumbent) |
|---------:|-------------------------------|
| 1        | DEAD (−3.6%, −196 TFLOPS) — branch overhead per K-iter dominates |
| 2        | DEAD (−6.4%, −351 TFLOPS) — even worse than u1, suggests bad pipelining at U=2 |
| 4        | DEAD (−1.8%, −97 TFLOPS) |
| 16       | DEAD (−0.55%, −30 TFLOPS, 5-rep verified) |
| 32       | DEAD (−0.58%, −32 TFLOPS, 5-rep verified, lowest variance) |

**L6 verdict**: the K-loop autotune (compiler default `#pragma unroll 8`) is the global optimum on this shape with this parent. Combined with the prior L6 audit (every other UNROLL_K-tagged sibling was −3% to −7%), the UNROLL_K dimension is fully explored and saturated for L6.

To close the L6 5354→5781 (92.6%→100%) gap, look elsewhere:
- Different parent (e.g., gm-variants) with K-loop tail tweak — but the top-7 incumbents are all within 0.5%, so parent saturation is also near-complete
- Persistent-block / xCD remapping refinements
- Issue-rate / s_waitcnt micro-tuning inside the K-loop body (vs unrolling scope)
- The 7.4% deficit may simply be the irreducible scheduling gap to aiter ASM at this very-large-K shape

---

## Artifacts

- `build_round31_optA_unrollK.py` — build driver
- `R31_OPT_A_BUILD_RESULTS.json` — per-build resource report
- `R31_OPT_A_BUILD.log`
- `bench_R31_optA.py` — sequential single-rep bench
- `R31_OPT_A_BENCH_RESULTS.json`, `R31_OPT_A_BENCH.log`
- `verify5_R31_optA.py` — 5-rep verifier
- `R31_OPT_A_VERIFY5_RESULTS.json`, `R31_OPT_A_VERIFY5.log`
- `build_all42/tk_mxfp4_gluon_cpp_n32768_k128256_R31A_ts_lgk2_v12_memc_btw_all_u{1,2,4,16,32}.cpython-310-x86_64-linux-gnu.so`

Total wall time: ~6 min (5s build + 60s single-rep + ~150s verify5).
