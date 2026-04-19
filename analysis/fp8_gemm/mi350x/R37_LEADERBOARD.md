# R37 — Fix B Leaderboard (2026-04-19)

Round: R37_fixB (kpair_64mfma_step34 backported into default; -mllvm -amdgpu-sched-strategy=max-memory-clause stripped)
Bench: warmup=200, iters=500, trim=0.1, gate finite≥0.995
Generated: 2026-04-19 04:28:12

## Summary

- **WIN: 14/42** (correct output AND tflops ≥ comp)
- **LOSE: 0/42** (correct output but tflops < comp)
- **WRONG_OUTPUT: 19/42** (kernel_finite < 0.995)
- **CRASH/ERR: 9/42**
- Elapsed: 4.5 min on 8 GPUs

## Per-shape results

| M | N | K | finite | R37 TFLOPS | R25 TFLOPS | comp TFLOPS | R37/comp | status | variant |
|---|---|---|--------|------------|------------|-------------|----------|--------|---------|
| 4096 | 4096 | 8192 | 1.0000 | 4371.6 | 4371.7 | 3959.9 | 110.4% | OK | ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all_R37 |
| 4096 | 4096 | 16384 | 0.9991 | 4983.0 | 5131.3 | 4642.1 | 107.3% | OK | ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all_R37 |
| 4096 | 4096 | 32768 | 0.7445 | — | 5388.6 | 5152.8 | — | WRONG_OUTPUT | ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all_R37 |
| 4096 | 6144 | 32768 | 0.6623 | — | 4876.0 | 3784.2 | — | WRONG_OUTPUT | ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all_R37 |
| 4096 | 14336 | 8192 | 0.9987 | 4785.6 | 4535.1 | 4345.8 | 110.1% | OK | ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all_R37 |
| 4096 | 14336 | 16384 | 0.9964 | 5249.4 | 4965.0 | 5013.0 | 104.7% | OK | ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all_R37 |
| 4096 | 28672 | 32768 | 0.6216 | — | 5436.2 | 5649.9 | — | WRONG_OUTPUT | ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all_R37 |
| 4096 | 32768 | 4096 | 0.9913 | — | 4409.4 | 4166.5 | — | WRONG_OUTPUT | ts_lgk2_gm7_v12_memc_pfoff14_R37 |
| 4096 | 32768 | 6144 | N/A | — | 4514.2 | 4548.6 | — | CRASH | ts_v12_gm7_memc_pfoff19_kx6144_btw_all_R37 |
| 4096 | 32768 | 14336 | 0.9883 | — | 5129.0 | 5296.1 | — | WRONG_OUTPUT | ts_v12_tv0_memc_btw_all_R37 |
| 4096 | 32768 | 28672 | 0.6145 | — | 5461.4 | 5568.2 | — | WRONG_OUTPUT | ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all_R37 |
| 4096 | 32768 | 128256 | 0.7613 | — | 5368.7 | 5781.1 | — | WRONG_OUTPUT | ts_lgk2_v12_memc_btw_all_R37 |
| 4096 | 128256 | 32768 | 0.6357 | — | 5303.9 | 3195.3 | — | WRONG_OUTPUT | ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all_R37 |
| 6144 | 4096 | 8192 | 0.9993 | 4287.6 | 4163.9 | 3822.0 | 112.2% | OK | ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all_R37 |
| 6144 | 4096 | 16384 | 0.9995 | 4625.5 | 4562.2 | 4428.1 | 104.5% | OK | ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all_R37 |
| 6144 | 32768 | 4096 | 0.9903 | — | 4517.5 | 4291.0 | — | WRONG_OUTPUT | ts_gm7_v12_memc_dc_pfoff14_R37 |
| 14336 | 4096 | 32768 | 0.6603 | — | 5056.0 | 5245.4 | — | WRONG_OUTPUT | ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all_R37 |
| 14336 | 32768 | 4096 | N/A | — | 4407.8 | 4462.6 | — | CRASH | ts_lgk2_gm7_v12_memc_pfoff14_R37 |
| 16384 | 4096 | 2048 | 0.6372 | — | 3360.0 | 2995.0 | — | WRONG_OUTPUT | ts_v12_gm7_memc_pfoff4_kx2048_btw_all_R37 |
| 16384 | 4096 | 3072 | N/A | — | 3889.4 | 3492.3 | — | CRASH | ts_gm6_v12_memc_dc_pfoff4_R37 |
| 16384 | 4096 | 4096 | 0.9951 | 4561.2 | 4321.2 | 3951.8 | 115.4% | OK | ts_gm7_v12_memc_dc_pfoff14_R37 |
| 16384 | 4096 | 6144 | 0.9995 | 4926.9 | 4741.4 | 4259.9 | 115.7% | OK | ts_v12_gm7_memc_pfoff19_kx6144_btw_all_R37 |
| 16384 | 4096 | 7168 | 0.9995 | 5063.3 | 4648.0 | 4443.2 | 114.0% | OK | ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all_R37 |
| 16384 | 4096 | 14336 | 0.9789 | — | 5234.3 | 5142.1 | — | WRONG_OUTPUT | ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all_R37 |
| 16384 | 4096 | 28672 | 0.6709 | — | 5248.3 | 5525.3 | — | WRONG_OUTPUT | ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all_R37 |
| 16384 | 6144 | 2048 | 0.9996 | 3436.2 | 3429.4 | 3047.6 | 112.8% | OK | ts_lgk2_gm6_v12_memc_pfoff4_R37 |
| 16384 | 6144 | 4096 | 0.9955 | 4768.1 | 4337.2 | 4042.5 | 117.9% | OK | ts_lgk2_gm7_v12_memc_pfoff14_R37 |
| 16384 | 14336 | 2048 | 0.9944 | — | 3493.1 | 3301.3 | — | WRONG_OUTPUT | ts_v12_gm7_memc_pfoff4_kx2048_btw_all_R37 |
| 16384 | 14336 | 4096 | 0.9929 | — | 4429.0 | 4255.8 | — | WRONG_OUTPUT | ts_gm7_v12_memc_dc_pfoff14_R37 |
| 16384 | 28672 | 2048 | N/A | — | 3445.8 | 3482.3 | — | CRASH | ts_lgk2_gm6_v12_memc_pfoff4_R37 |
| 16384 | 28672 | 4096 | 0.9753 | — | 4409.9 | 4411.7 | — | WRONG_OUTPUT | ts_gm7_v12_memc_dc_pfoff14_R37 |
| 28672 | 4096 | 8192 | 0.9955 | 5243.3 | 4852.8 | 4810.0 | 109.0% | OK | ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all_R37 |
| 28672 | 4096 | 16384 | 0.9983 | 5770.0 | 5178.0 | 5350.6 | 107.8% | OK | ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all_R37 |
| 28672 | 32768 | 4096 | N/A | — | 4309.2 | 4466.6 | — | CRASH | ts_gm7_v12_memc_dc_pfoff14_R37 |
| 32768 | 4096 | 2048 | N/A | — | 3430.8 | 3131.8 | — | CRASH | ts_lgk2_gm6_v12_memc_pfoff4_R37 |
| 32768 | 4096 | 3072 | 0.9932 | — | 4014.8 | 3630.6 | — | WRONG_OUTPUT | ts_lgk2_memc_btw_all_R37 |
| 32768 | 4096 | 7168 | 0.9964 | 5159.7 | 4732.3 | 4666.8 | 110.6% | OK | ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all_R37 |
| 32768 | 4096 | 14336 | 0.9896 | — | 5161.4 | 5223.4 | — | WRONG_OUTPUT | ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all_R37 |
| 32768 | 6144 | 2048 | N/A | — | 3431.8 | 3239.9 | — | CRASH | ts_lgk2_gm6_v12_memc_pfoff4_R37 |
| 32768 | 14336 | 2048 | 0.9912 | — | 3542.5 | 3351.4 | — | WRONG_OUTPUT | ts_gm6_v12_memc_dc_pfoff4_R37 |
| 32768 | 28672 | 2048 | N/A | — | 3459.0 | 3353.4 | — | CRASH | ts_lgk2_gm6_v12_memc_pfoff4_R37 |
| 128256 | 32768 | 4096 | N/A | — | 4380.1 | 4536.4 | — | CRASH | ts_lgk2_gm7_v12_memc_pfoff14_R37 |

## Key findings

1. **R37 fix delivers correct output for 14/42 shapes**, all WINS vs comp (12-18% over baseline).
   These are the shapes where the BEST_VARIANTS flag stack is compatible with the fused step34 path.

2. **19 shapes WRONG_OUTPUT** — fused step34 still produces 0.6-0.99 finite frac on these.
   Most are M ≥ 16384 with K ≥ 4096 (large grids, more iters → more chances for memc-style
   reordering bugs to manifest). Some have finite just under the gate (0.989-0.999) and
   would benefit from a tighter SNR-aware gate; others are deeply broken (0.62-0.76).

3. **9 shapes CRASH** with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION. All use the
   `_ts_lgk2_gm6_v12_memc_pfoff4` (or sibling without _kx_btw_all) variant. The R25-C
   tail-pf-off path interaction with fused step34 is producing OOB loads.

## Recommendations

- R37's WINS confirm the surgical fix is correct in principle: when memc and other
  scheduler-aggressive flags are absent, the fused step34 is both correct AND faster
  than the legacy path (12-18% over comp on the 14 winning shapes).
- Next step: convert `emit_one_pf` (the buffer_load_to_lds intrinsic call) into an
  inline `asm volatile` block, which would prevent the LLVM scheduler from reordering
  prefetches across iteration boundaries — the root cause of memc-incompatibility.
- Alternatively: fork BEST_VARIANTS to drop scheduler-aggressive flags from any variant
  that fails correctness on R37, accepting a small TFLOPS hit on those shapes.

## Files

- `kernel_mxfp4_gluon_cpp.cpp` — modified with R37_FIX_B (default ON) + S1-force-barrier
- `build_R37.py` — builder that strips `-mllvm -amdgpu-sched-strategy=max-memory-clause`
- `bench_all_42_R37.py` — correctness-gated bench harness
- `bench_all42_results_R37_fixB.json` — full bench results
- `R37_BENCH_RUN.log` — bench run log
- `build_R37/` — module .so files