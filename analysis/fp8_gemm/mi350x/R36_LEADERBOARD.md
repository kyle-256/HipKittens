# R36 Leaderboard — _f34 Correctness-Gated Re-bench (2026-04-19)

## TL;DR — Mechanically Applying `-DFUSED_STEP34=1` to BEST_VARIANTS Stacks BREAKS Everything

- R25_FINAL_v2 reported **40/42 WIN**, but every WIN was "time to wrong output".
- R36 builds `_f34` versions of every BEST_VARIANTS module by appending `-DFUSED_STEP34=1`
  to its CPPFLAGS, then runs a correctness gate (constant scale=-4, finite_frac >= 0.995)
  before timing. Result: **0/42 WIN, 0/42 LOSE,
  25/42 WRONG_OUTPUT, 17/42 CRASH/ERR**.

- The R35 finding (_f34 produces correct output) was validated only against the
  PLAIN kernel (no R25-C tail-pf-off, no BARRIER_TO_WAITCNT, no K_EXACT). When `-DFUSED_STEP34=1`
  is composed with the production-tuned BEST_VARIANTS flags (R25C_TAIL_PF_OFF_ITERS,
  BARRIER_TO_WAITCNT_ALL, R25C_K_EXACT, etc.), every shape either crashes (memory
  aperture violation) or produces wrong output.

**Conclusion:** Fix B (root-cause repair: backport `kpair_64mfma_step34` into the
default code path while preserving R25-C tail-pf logic) is REQUIRED. Fix A (mechanical
`-DFUSED_STEP34=1` per-shape) is unworkable.

## Bench Setup

- WARMUP=200, ITERS=500, trim=0.1
- Correctness gate: kernel_finite >= 0.995 on **constant scale=-4**
  inputs (matches `snr_all_42_shapes.py`; bounds the bf16 output range so the gate
  measures *kernel correctness*, not bf16 saturation).
- Timing inputs use random scale [-2,3] (matches existing bench harness).
- GPUs: [0, 1, 2, 3, 4, 5, 6, 7], total 4.4 min.

## Per-Shape Results

| # | M | N | K | R25 TFLOPS (broken) | R36 _f34 TFLOPS (correct?) | R36 status | finite | comp | R25 vs comp | R36 vs comp | parent variant |
|---|---|---|---|---:|---:|:---|---:|---:|---:|---:|:---|
| 1 | 4096 | 4096 | 8192 | 4501.7 | — | WRONG_OUTPUT | 0.6630 | 3959.9 | 113.7% | — | `ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all` |
| 2 | 4096 | 4096 | 16384 | 5723.6 | — | WRONG_OUTPUT | 0.6090 | 4642.1 | 123.3% | — | `ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all` |
| 3 | 4096 | 4096 | 32768 | 5940.0 | — | WRONG_OUTPUT | 0.1008 | 5152.8 | 115.3% | — | `ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all` |
| 4 | 4096 | 6144 | 32768 | 5629.8 | — | WRONG_OUTPUT | 0.1266 | 3784.2 | 148.8% | — | `ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all` |
| 5 | 4096 | 14336 | 8192 | 5057.2 | — | WRONG_OUTPUT | 0.6507 | 4345.8 | 116.4% | — | `ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all` |
| 6 | 4096 | 14336 | 16384 | 5781.5 | — | CRASH | — | 5013.0 | 115.3% | — | `ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all` |
| 7 | 4096 | 28672 | 32768 | 6531.4 | — | WRONG_OUTPUT | 0.1065 | 5649.9 | 115.6% | — | `ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all` |
| 8 | 4096 | 32768 | 4096 | 4787.4 | — | CRASH | — | 4166.5 | 114.9% | — | `ts_lgk2_gm7_v12_memc_pfoff14` |
| 9 | 4096 | 32768 | 6144 | 5285.0 | — | CRASH | — | 4548.6 | 116.2% | — | `ts_v12_gm7_memc_pfoff19_kx6144_btw_all` |
| 10 | 4096 | 32768 | 14336 | 5217.0 | — | WRONG_OUTPUT | 0.5618 | 5296.1 | 98.5% | — | `ts_v12_tv0_memc_btw_all` |
| 11 | 4096 | 32768 | 28672 | 6492.0 | — | WRONG_OUTPUT | 0.4803 | 5568.2 | 116.6% | — | `ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all` |
| 12 | 4096 | 32768 | 128256 | 5353.9 | — | WRONG_OUTPUT | 0.6500 | 5781.1 | 92.6% | — | `ts_lgk2_v12_memc_btw_all` |
| 13 | 4096 | 128256 | 32768 | 6373.4 | — | WRONG_OUTPUT | 0.1128 | 3195.3 | 199.5% | — | `ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all` |
| 14 | 6144 | 4096 | 8192 | 4305.5 | — | WRONG_OUTPUT | 0.6650 | 3822.0 | 112.7% | — | `ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all` |
| 15 | 6144 | 4096 | 16384 | 5153.9 | — | WRONG_OUTPUT | 0.6103 | 4428.1 | 116.4% | — | `ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all` |
| 16 | 6144 | 32768 | 4096 | 5046.4 | — | CRASH | — | 4291.0 | 117.6% | — | `ts_gm7_v12_memc_dc_pfoff14` |
| 17 | 14336 | 4096 | 32768 | 6079.5 | — | WRONG_OUTPUT | 0.0955 | 5245.4 | 115.9% | — | `ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all` |
| 18 | 14336 | 32768 | 4096 | 5084.6 | — | CRASH | — | 4462.6 | 113.9% | — | `ts_lgk2_gm7_v12_memc_pfoff14` |
| 19 | 16384 | 4096 | 2048 | 3414.5 | — | WRONG_OUTPUT | 0.8738 | 2995.0 | 114.0% | — | `ts_v12_gm7_memc_pfoff4_kx2048_btw_all` |
| 20 | 16384 | 4096 | 3072 | 3945.4 | — | WRONG_OUTPUT | 0.8458 | 3492.3 | 113.0% | — | `ts_gm6_v12_memc_dc_pfoff4` |
| 21 | 16384 | 4096 | 4096 | 4461.5 | — | WRONG_OUTPUT | 0.7891 | 3951.8 | 112.9% | — | `ts_gm7_v12_memc_dc_pfoff14` |
| 22 | 16384 | 4096 | 6144 | 4966.9 | — | WRONG_OUTPUT | 0.7318 | 4259.9 | 116.6% | — | `ts_v12_gm7_memc_pfoff19_kx6144_btw_all` |
| 23 | 16384 | 4096 | 7168 | 5250.4 | — | WRONG_OUTPUT | 0.6921 | 4443.2 | 118.2% | — | `ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all` |
| 24 | 16384 | 4096 | 14336 | 5996.9 | — | WRONG_OUTPUT | 0.6009 | 5142.1 | 116.6% | — | `ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all` |
| 25 | 16384 | 4096 | 28672 | 6407.4 | — | WRONG_OUTPUT | 0.4546 | 5525.3 | 116.0% | — | `ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all` |
| 26 | 16384 | 6144 | 2048 | 3471.8 | — | WRONG_OUTPUT | 0.8966 | 3047.6 | 113.9% | — | `ts_lgk2_gm6_v12_memc_pfoff4` |
| 27 | 16384 | 6144 | 4096 | 4639.0 | — | WRONG_OUTPUT | 0.7747 | 4042.5 | 114.8% | — | `ts_lgk2_gm7_v12_memc_pfoff14` |
| 28 | 16384 | 14336 | 2048 | 3645.9 | — | CRASH | — | 3301.3 | 110.4% | — | `ts_v12_gm7_memc_pfoff4_kx2048_btw_all` |
| 29 | 16384 | 14336 | 4096 | 4841.2 | — | CRASH | — | 4255.8 | 113.8% | — | `ts_gm7_v12_memc_dc_pfoff14` |
| 30 | 16384 | 28672 | 2048 | 3764.1 | — | CRASH | — | 3482.3 | 108.1% | — | `ts_lgk2_gm6_v12_memc_pfoff4` |
| 31 | 16384 | 28672 | 4096 | 5059.9 | — | CRASH | — | 4411.7 | 114.7% | — | `ts_gm7_v12_memc_dc_pfoff14` |
| 32 | 28672 | 4096 | 8192 | 5480.3 | — | WRONG_OUTPUT | 0.6599 | 4810.0 | 113.9% | — | `ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all` |
| 33 | 28672 | 4096 | 16384 | 6238.2 | — | CRASH | — | 5350.6 | 116.6% | — | `ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all` |
| 34 | 28672 | 32768 | 4096 | 5048.5 | — | CRASH | — | 4466.6 | 113.0% | — | `ts_gm7_v12_memc_dc_pfoff14` |
| 35 | 32768 | 4096 | 2048 | 3472.2 | — | CRASH | — | 3131.8 | 110.9% | — | `ts_lgk2_gm6_v12_memc_pfoff4` |
| 36 | 32768 | 4096 | 3072 | 3915.3 | — | CRASH | — | 3630.6 | 107.8% | — | `ts_lgk2_memc_btw_all` |
| 37 | 32768 | 4096 | 7168 | 5342.4 | — | WRONG_OUTPUT | 0.6763 | 4666.8 | 114.5% | — | `ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all` |
| 38 | 32768 | 4096 | 14336 | 6157.3 | — | WRONG_OUTPUT | 0.5943 | 5223.4 | 117.9% | — | `ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all` |
| 39 | 32768 | 6144 | 2048 | 3728.6 | — | CRASH | — | 3239.9 | 115.1% | — | `ts_lgk2_gm6_v12_memc_pfoff4` |
| 40 | 32768 | 14336 | 2048 | 3725.5 | — | CRASH | — | 3351.4 | 111.2% | — | `ts_gm6_v12_memc_dc_pfoff4` |
| 41 | 32768 | 28672 | 2048 | 3718.1 | — | CRASH | — | 3353.4 | 110.9% | — | `ts_lgk2_gm6_v12_memc_pfoff4` |
| 42 | 128256 | 32768 | 4096 | 4943.3 | — | CRASH | — | 4536.4 | 109.0% | — | `ts_lgk2_gm7_v12_memc_pfoff14` |

## Summary Counts

- **WIN**: 0/42 (passed correctness gate AND TFLOPS >= comp)
- **LOSE**: 0/42 (passed correctness gate but TFLOPS < comp)
- **WRONG_OUTPUT**: 25/42 (kernel_finite < 0.995 even with bounded scales)
- **CRASH/ERR**: 17/42 (memory aperture violation or other failure)

## R36 Wins

(none)

## Worst Regressions

(no R36 result with TFLOPS<R25 since virtually no shape passed correctness)

## Failure Mode Breakdown

### CRASH / Memory aperture violation

These shapes' f34 builds segfault on the first run (HSA memory aperture violation).
Mechanism: when `FUSED_STEP34=1` is set, the entire R25-C tail-pf-off conditional branch
(line 2849-2945 of kernel) is bypassed. `kpair_64mfma_step34` ALWAYS issues prefetches,
which on the last K-iter read past the buffer SRD and trigger the GPU's address fault.

- `4096x14336x16384` — `ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all_f34` (CRASH)
- `4096x32768x4096` — `ts_lgk2_gm7_v12_memc_pfoff14_f34` (CRASH)
- `4096x32768x6144` — `ts_v12_gm7_memc_pfoff19_kx6144_btw_all_f34` (CRASH)
- `6144x32768x4096` — `ts_gm7_v12_memc_dc_pfoff14_f34` (CRASH)
- `14336x32768x4096` — `ts_lgk2_gm7_v12_memc_pfoff14_f34` (CRASH)
- `16384x14336x2048` — `ts_v12_gm7_memc_pfoff4_kx2048_btw_all_f34` (CRASH)
- `16384x14336x4096` — `ts_gm7_v12_memc_dc_pfoff14_f34` (CRASH)
- `16384x28672x2048` — `ts_lgk2_gm6_v12_memc_pfoff4_f34` (CRASH)
- `16384x28672x4096` — `ts_gm7_v12_memc_dc_pfoff14_f34` (CRASH)
- `28672x4096x16384` — `ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all_f34` (CRASH)
- `28672x32768x4096` — `ts_gm7_v12_memc_dc_pfoff14_f34` (CRASH)
- `32768x4096x2048` — `ts_lgk2_gm6_v12_memc_pfoff4_f34` (CRASH)
- `32768x4096x3072` — `ts_lgk2_memc_btw_all_f34` (CRASH)
- `32768x6144x2048` — `ts_lgk2_gm6_v12_memc_pfoff4_f34` (CRASH)
- `32768x14336x2048` — `ts_gm6_v12_memc_dc_pfoff4_f34` (CRASH)
- `32768x28672x2048` — `ts_lgk2_gm6_v12_memc_pfoff4_f34` (CRASH)
- `128256x32768x4096` — `ts_lgk2_gm7_v12_memc_pfoff14_f34` (CRASH)

### WRONG_OUTPUT

Shapes whose f34 build produces non-finite cells even with bounded constant scale=-4.
Distinct from CRASH because the kernel returns; the corruption manifests as inf/nan in
the output. Likely the same prefetch-past-end issue as CRASH but the OOB read happens
to land in mapped memory and produce garbage rather than fault.

- `16384x6144x2048` — `ts_lgk2_gm6_v12_memc_pfoff4_f34` (finite=0.8966)
- `16384x4096x2048` — `ts_v12_gm7_memc_pfoff4_kx2048_btw_all_f34` (finite=0.8738)
- `16384x4096x3072` — `ts_gm6_v12_memc_dc_pfoff4_f34` (finite=0.8458)
- `16384x4096x4096` — `ts_gm7_v12_memc_dc_pfoff14_f34` (finite=0.7891)
- `16384x6144x4096` — `ts_lgk2_gm7_v12_memc_pfoff14_f34` (finite=0.7747)
- `16384x4096x6144` — `ts_v12_gm7_memc_pfoff19_kx6144_btw_all_f34` (finite=0.7318)
- `16384x4096x7168` — `ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all_f34` (finite=0.6921)
- `32768x4096x7168` — `ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all_f34` (finite=0.6763)
- `6144x4096x8192` — `ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all_f34` (finite=0.6650)
- `4096x4096x8192` — `ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all_f34` (finite=0.6630)
- `28672x4096x8192` — `ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all_f34` (finite=0.6599)
- `4096x14336x8192` — `ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all_f34` (finite=0.6507)
- `4096x32768x128256` — `ts_lgk2_v12_memc_btw_all_f34` (finite=0.6500)
- `6144x4096x16384` — `ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all_f34` (finite=0.6103)
- `4096x4096x16384` — `ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all_f34` (finite=0.6090)
- `16384x4096x14336` — `ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all_f34` (finite=0.6009)
- `32768x4096x14336` — `ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all_f34` (finite=0.5943)
- `4096x32768x14336` — `ts_v12_tv0_memc_btw_all_f34` (finite=0.5618)
- `4096x32768x28672` — `ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all_f34` (finite=0.4803)
- `16384x4096x28672` — `ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all_f34` (finite=0.4546)
- `4096x6144x32768` — `ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all_f34` (finite=0.1266)
- `4096x128256x32768` — `ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all_f34` (finite=0.1128)
- `4096x28672x32768` — `ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all_f34` (finite=0.1065)
- `4096x4096x32768` — `ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all_f34` (finite=0.1008)
- `14336x4096x32768` — `ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all_f34` (finite=0.0955)

## Sanity Check: PLAIN _f34 (no R25-C/BTW/K_EXACT) DOES Work

On `32768x4096x2048` (which CRASHES with `ts_lgk2_gm6_v12_memc_pfoff4_f34`):
- Plain `tk_mxfp4_gluon_cpp_n4096_k2048_f34` (just `-DFUSED_STEP34=1`, no other flags)
  produces **finite_frac = 0.9977** at constant scale=-4, passing the correctness gate.
- This confirms the issue is in the **flag composition** (FUSED_STEP34 + R25-C tail-pf,
  + BTW, + K_EXACT), not in FUSED_STEP34 itself.

## Recommended Next Steps

1. **Fix B from R35 diagnosis (backport)**: rewrite the non-fused step3+step4 path so that
   it uses a fused single-asm-block step3+step4 INTERNALLY (matching `kpair_64mfma_step34`)
   while still respecting R25-C tail-pf-off, BTW, and K_EXACT branching. This is the only
   path that preserves both correctness AND the per-shape tuning that gave the 41/42 result.
2. **Audit `kpair_64mfma_step34` for tail-iter prefetch safety**: it should accept a
   `pf_active` template parameter and skip prefetches on the final iter (mirroring the
   `_r25c_tail_no_pf` branch that the FUSED path currently bypasses).
3. **Re-evaluate the 41/42 claim**: every R31/R32/R33 "WIN" was measuring time-to-garbage.
   The TODO/AGENT_PROMPT files should be updated to reflect that the production
   leaderboard is now empty until Fix B lands.

## Artifacts

- `bench_all_42_correct.py` — correctness-gated bench harness (constant-scale gate)
- `build_R36_f34.py` — parallel builder for `_f34` versions of every BEST_VARIANTS entry
- `bench_all42_results_R36_f34.json` — full results
- `bench_all42_results_R36_f34.log` — bench log
- `R36_F34_BUILD.log` — build log (30 unique builds, all succeeded)
- `R36_F34_BUILD_MANIFEST.json` — built module manifest

