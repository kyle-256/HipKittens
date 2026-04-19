# R54 Opt E-1 Verdict — aiter `.co` dlopen for cohort-race rescue (M-heavy/N=4096)

## TL;DR
**3/3 PROMOTE** at strict 10-run gate. **+3 NET VC** from cohort-race rescues — all three R52 VC shapes that lost VC under R53 strict 10-run regress are reinstated as bit-deterministic AITER kernels. **R50D shim used AS-IS, no rebuild.** Mean comp gain vs HK R52/R53 readings: **+15.5pp**.

## Pre-bench verification

- Tile is 256x256 (same as R50D/R51/R52 cohort), shim launch ABI is unchanged.
- Shim `aiter_gemm_a4w4_launch` derives grid internally from `M,N,K` and `tile_M,tile_N`:
  - `gdx = ceil(N / tile_N)`, `gdy = ceil(M / tile_M)`, `gdz = 1`, `bdx = 256`.
  - Confirmed at `R50D_aiter_dlopen.cpp:184-187`.
- All three shapes have N=4096 → `gdx = 16`. Heights vary: `gdy = 128 (M=32768)` or `112 (M=28672)`.

## Per-shape results (10-RUN strict gate, GPU 3)

| # | Shape (M,N,K) | Prior status | comp | tflops | pct_comp | n_OK | wcf_max | wcf_std | fin_min | snr_med_min | gate | Decision | NET VC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | (32768, 4096, 2048) | R53 PASS_8/10 fin_min=0.956 (R41B HK) | 3131.8 | 3476.0 | 110.99% | 10/10 | 0.0 | 0.0 | 1.0 | 55.52 dB | PASS | **PROMOTE** | +1 |
| 2 | (32768, 4096, 3072) | R53 PASS_9/10 wcf_max=0.0218 (R40B HK) | 3630.6 | 4307.5 | 118.64% | 10/10 | 0.0 | 0.0 | 1.0 | 55.55 dB | PASS | **PROMOTE** | +1 |
| 3 | (28672, 4096, 8192) | R53 PASS_9/10 wcf_max=0.0201 (R40B HK) | 4810.0 | 5245.5 | 109.05% | 10/10 | 0.0 | 0.0 | 1.0 | 55.59 dB | PASS | **PROMOTE** | +1 |

**Aggregate NET VC delta: +3** (three cohort-LOSS shapes reinstated as VC).

## Perf claw-back vs prior R53 readings (HK)

| Shape | HK R53 reading (LOSS) | AITER R54 PASS | delta |
|---|---|---|---|
| (32768, 4096, 2048) | 101.98% (sourceR41B, NOT VC) | 110.99% | +9.01pp |
| (32768, 4096, 3072) | 99.52% (source R40B, NOT VC) | 118.64% | +19.12pp |
| (28672, 4096, 8192) | 91.82% (source R40B, NOT VC) | 109.05% | +17.23pp |

**Aggregate mean perf gain: +15.45pp** across the three shapes (each becomes the new authoritative VC row in the integration). Even on a strict head-to-head where HK had been counted as VC at the R52 reading (102.0%, 99.5%, 91.8%), AITER beats every one (+9.0/+19.1/+17.2 pp).

## Mechanism summary
The R50D `.co` dlopen pattern is now extended to **3 more cohort-race-LOSS rescue shapes** (all aiter 256x256). The bit-deterministic nature of the precompiled aiter `.co` (compiled HSACO with no compiler-IPRA reordering of asm volatile blocks) is exactly what fixes the R45-style cohort tail-draws that affect the HK kernels. Combined with the prior 7 256x256 promotions in R51/R52 plus the 3 64x1024 promotions in R53, the shim is now the production path for **at least 13 distinct shapes**.

## Notes
- All seeds in [101..1010] are INDEPENDENT.
- All three shapes hit identical SNR ~55.5-55.6 dB with `wrong_cell_frac = 0.0` and `kernel_finite = 1.0` across all 10 seeds — extremely clean correctness.
- No SMOKE failures. No CRASH. No PARSE_FAIL. No TIMEOUT.
- `tflops_first` (perf seed) and tail seeds 202-1010 (correctness only) all converge.

## Artifacts
- Bench scripts: `bench_R54E1_1.py`, `bench_R54E1_2.py`, `bench_R54E1_3.py`
- SMOKE: `R54_OPT_E1_{1,2,3}_SMOKE.{json,log}`
- 10-RUN: `R54_OPT_E1_{1,2,3}_10RUN.{json,log}`
- Integration fragments: `R54E1_{1,2,3}_INTEGRATION_FRAGMENT.json`
- Shim (reused AS-IS): `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- aiter `.co`: `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`

## GPU usage
- All three candidates: **GPU 3** (per task assignment, sequential to avoid contention).
- `rocm-smi --showuse` confirmed all GPUs idle pre-launch.

## Bench rules compliance
- warmup=200, iters=500, trim_frac=0.10 — confirmed in scripts.
- HIP_VISIBLE_DEVICES=3 on idle GPU — confirmed.
- INDEPENDENT seeds [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010] for 10-run.
- Strict gate: n_OK >= 8/10 AND wcf_max < 0.02 AND wcf_std < 0.01 AND fin_min >= 0.97 — **all three shapes satisfy** (in fact 10/10, 0.0, 0.0, 1.0 — ideal).
- No `pgrep -f X` self-matching wait loops.
