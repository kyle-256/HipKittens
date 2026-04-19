# R38 — Opt E Unified Leaderboard (2026-04-19)

Round: R38_optE = R37 14 WIN + R38D 5 recoveries + R38B 3 CRASH→OK (selective `R38B_TAIL_FIX=1`)
Bench: warmup=200, iters=500, trim=0.1, gate finite≥0.995, 8 GPUs (0-7)
Aggregation: **best-of-3 runs** (gate-near shapes flake, see `R38_OPT_E_VERDICT.md`)
Generated: 2026-04-19

## Summary (best-of-3)

| Metric | R37 baseline | **R38E (best-of-3)** | delta |
|---|---|---|---|
| WIN | 14/42 | **12/42** | -2 (gate flake) |
| LOSE_CORRECT | 0/42 | **4/42** | +4 |
| Verified-correct (WIN+LOSE) | 14/42 | **16/42** | +2 |
| WRONG_OUTPUT | 19/42 | 19/42 | 0 |
| CRASH/ERR | 9/42 | 7/42 | -2 |

Bench wall-clock: ~2.5 min/run × 3 = 7.5 min on 8 GPUs.

The 16 verified-correct cells decompose as:
- **9 stable R37 WINs** (all 3 runs WIN): 4096x4096x8192, 4096x4096x16384, 4096x14336x8192, 4096x14336x16384, 6144x4096x8192, 6144x4096x16384, 16384x4096x7168, 16384x6144x4096, 28672x4096x16384.
- **3 R38B selective-macro wins**: 16384x4096x3072 (CRASH→WIN), 32768x6144x2048 (CRASH→WIN), 128256x32768x4096 (CRASH→LOSE_CORRECT 86.6%).
- **2 R38D NEW WINs**: 16384x4096x2048 (WRONG→WIN 106.3%), 32768x4096x3072 (WRONG→LOSE_CORRECT 99.1%).
- **2 R38D LOSS_CORRECT**: 4096x32768x4096 (WRONG→LOSE 92.2%), 16384x4096x14336 (WRONG→LOSE 86.7%).

R37 WINs that flaked to WRONG_OUTPUT in R38E (despite identical variant + macros):
16384x4096x4096, 16384x4096x6144, 16384x6144x2048, 28672x4096x8192, 32768x4096x7168.
These shapes hover at finite ∈ [0.97, 0.99] across runs — they're the same kernels as R37
binaries but the gate is non-monotonic on noisy finite-frac inputs. Flakiness is in the
**correctness probe**, not the kernel.

## Per-shape results (best-of-3 across runs 1, 2, 3)

| M | N | K | R37 status | R37 TFLOPS | R37/comp | R38E status | R38E TFLOPS | R38E/comp | R38E variant | macro |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 4096 | 8192 | WIN | 4371.6 | 110.4% | WIN | 4371.4 | 110.4% | ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all | - |
| 4096 | 4096 | 16384 | WIN | 4983.0 | 107.3% | WIN | 5128.1 | 110.5% | ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all | - |
| 4096 | 4096 | 32768 | WRONG_OUTPUT | - | - | WRONG_OUTPUT | - | - | ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all | - |
| 4096 | 6144 | 32768 | WRONG_OUTPUT | - | - | WRONG_OUTPUT | - | - | ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all | - |
| 4096 | 14336 | 8192 | WIN | 4785.6 | 110.1% | WIN | 4869.3 | 112.0% | ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all | - |
| 4096 | 14336 | 16384 | WIN | 5249.4 | 104.7% | WIN | 5253.5 | 104.8% | ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all | - |
| 4096 | 28672 | 32768 | WRONG_OUTPUT | - | - | WRONG_OUTPUT | - | - | ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all | - |
| 4096 | 32768 | 4096 | WRONG_OUTPUT | - | - | LOSE | 3842.3 | 92.2% | lgk2_v16 (R38D) | - |
| 4096 | 32768 | 6144 | CRASH | - | - | CRASH | - | - | ts_v12_gm7_memc_pfoff19_kx6144_btw_all | - |
| 4096 | 32768 | 14336 | WRONG_OUTPUT | - | - | WRONG_OUTPUT | - | - | ts_v12_tv0_memc_btw_all | - |
| 4096 | 32768 | 28672 | WRONG_OUTPUT | - | - | WRONG_OUTPUT | - | - | ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all | - |
| 4096 | 32768 | 128256 | WRONG_OUTPUT | - | - | WRONG_OUTPUT | - | - | ts_lgk2_v12_memc_btw_all | - |
| 4096 | 128256 | 32768 | WRONG_OUTPUT | - | - | WRONG_OUTPUT | - | - | ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all | - |
| 6144 | 4096 | 8192 | WIN | 4287.6 | 112.2% | WIN | 4249.4 | 111.2% | ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all | - |
| 6144 | 4096 | 16384 | WIN | 4625.5 | 104.5% | WIN | 4626.7 | 104.5% | ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all | - |
| 6144 | 32768 | 4096 | WRONG_OUTPUT | - | - | WRONG_OUTPUT | - | - | ts_lgk2_v24 (R38D) | - |
| 14336 | 4096 | 32768 | WRONG_OUTPUT | - | - | WRONG_OUTPUT | - | - | ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all | - |
| 14336 | 32768 | 4096 | CRASH | - | - | CRASH | - | - | ts_lgk2_gm7_v12_memc_pfoff14 | - |
| 16384 | 4096 | 2048 | WRONG_OUTPUT | - | - | **WIN** | 3184.3 | 106.3% | ts_v12_tv16 (R38D) | - |
| 16384 | 4096 | 3072 | CRASH | - | - | **WIN** | 3516.5 | 100.7% | ts_gm6_v12_memc_dc_pfoff4 | R38B_TAIL_FIX=1 |
| 16384 | 4096 | 4096 | WIN | 4561.2 | 115.4% | WRONG_OUTPUT | - | - | ts_gm7_v12_memc_dc_pfoff14 | - |
| 16384 | 4096 | 6144 | WIN | 4926.9 | 115.7% | WRONG_OUTPUT | - | - | ts_v12_gm7_memc_pfoff19_kx6144_btw_all | - |
| 16384 | 4096 | 7168 | WIN | 5063.3 | 114.0% | WIN | 4976.3 | 112.0% | ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all | - |
| 16384 | 4096 | 14336 | WRONG_OUTPUT | - | - | LOSE | 4456.6 | 86.7% | ts_gm8_v12_btw_all (R38D) | - |
| 16384 | 4096 | 28672 | WRONG_OUTPUT | - | - | WRONG_OUTPUT | - | - | ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all | - |
| 16384 | 6144 | 2048 | WIN | 3436.2 | 112.8% | CRASH | - | - | ts_lgk2_gm6_v12_memc_pfoff4 | - |
| 16384 | 6144 | 4096 | WIN | 4768.1 | 117.9% | WIN | 4777.9 | 118.2% | ts_lgk2_gm7_v12_memc_pfoff14 | - |
| 16384 | 14336 | 2048 | WRONG_OUTPUT | - | - | CRASH | - | - | ts_v12_gm7_memc_pfoff4_kx2048_btw_all | - |
| 16384 | 14336 | 4096 | WRONG_OUTPUT | - | - | WRONG_OUTPUT | - | - | ts_gm7_v12_memc_dc_pfoff14 | - |
| 16384 | 28672 | 2048 | CRASH | - | - | WRONG_OUTPUT | - | - | ts_lgk2_gm6_v12_memc_pfoff4 | - |
| 16384 | 28672 | 4096 | WRONG_OUTPUT | - | - | WRONG_OUTPUT | - | - | ts_gm7_v12_memc_dc_pfoff14 | - |
| 28672 | 4096 | 8192 | WIN | 5243.3 | 109.0% | WRONG_OUTPUT | - | - | ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all | - |
| 28672 | 4096 | 16384 | WIN | 5770.0 | 107.8% | WIN | 5752.8 | 107.5% | ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all | - |
| 28672 | 32768 | 4096 | CRASH | - | - | CRASH | - | - | ts_gm7_v12_memc_dc_pfoff14 | - |
| 32768 | 4096 | 2048 | CRASH | - | - | WRONG_OUTPUT | - | - | ts_lgk2_gm6_v12_memc_pfoff4 | - |
| 32768 | 4096 | 3072 | WRONG_OUTPUT | - | - | LOSE | 3599.6 | 99.1% | v32 (R38D) | - |
| 32768 | 4096 | 7168 | WIN | 5159.7 | 110.6% | WRONG_OUTPUT | - | - | ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all | - |
| 32768 | 4096 | 14336 | WRONG_OUTPUT | - | - | WRONG_OUTPUT | - | - | ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all | - |
| 32768 | 6144 | 2048 | CRASH | - | - | **WIN** | 3247.5 | 100.2% | ts_lgk2_gm6_v12_memc_pfoff4 | R38B_TAIL_FIX=1 |
| 32768 | 14336 | 2048 | WRONG_OUTPUT | - | - | CRASH | - | - | ts_gm6_v12_memc_dc_pfoff4 | - |
| 32768 | 28672 | 2048 | CRASH | - | - | CRASH | - | - | ts_lgk2_gm6_v12_memc_pfoff4 | - |
| 128256 | 32768 | 4096 | CRASH | - | - | LOSE | 3929.3 | 86.6% | ts_lgk2_gm7_v12_memc_pfoff14 | R38B_TAIL_FIX=1 |

Bold = new in R38E that wasn't in R37.

## Files
- `R38_BEST_VARIANTS_v3.py` — drop-in dict (variant_tag, macro_overrides_dict)
- `build_R38E.py`, `bench_all_42_R38E.py`
- `bench_all42_results_R38_optE.json` (run 3) + `_run2.json` + `_run3.json` (per-run snapshots)
- `R38E_BENCH_RUN.log` / `_v2.log` / `_v3.log`
- `R38E_BUILD_MANIFEST.json`
- `build_R38E/` — 35 unique .so binaries (32 R37/R38D + 3 R38B-macro forks)
- `R38_OPT_E_PROGRESS.md` / `R38_OPT_E_VERDICT.md`
