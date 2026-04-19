# R54 Opt E-2 Verdict — aiter `.co` dlopen 256x256 cohort-race rescue (N-heavy/M=4096 + mid-K)

## TL;DR
**3/3 PROMOTE** at strict 10-run gate. **+3 NET VC** (all three cohort-race rescues from R52 VC → R53 cohort LOSS). R50D shim used **AS-IS, no rebuild**. All three exceed competitor TFLOPS.

## Per-shape results (10-RUN strict gate)

| # | Shape (M,N,K) | R52→R53 status | comp | tflops | pct_comp | n_OK | wcf_max | wcf_std | fin_min | snr_med_min | gate | Decision | NET VC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | (4096, 32768, 4096) | VC 91.8% → cohort LOSS PASS_9/10 fin_min=0.951 | 4166.5 | 4456.7 | 106.97% | 10/10 | 0.0 | 0.0 | 1.0 | 55.54 dB | PASS | **PROMOTE** | +1 |
| 2 | (4096, 32768, 6144) | VC 92.3% → cohort LOSS PASS_8/10 fin=0.926 | 4548.6 | 4990.7 | 109.72% | 10/10 | 0.0 | 0.0 | 1.0 | 55.57 dB | PASS | **PROMOTE** | +1 |
| 3 | (16384, 4096, 6144) | VC 98.5% → cohort LOSS PASS_8/10 fin_min=0.950 | 4259.9 | 4999.2 | 117.35% | 10/10 | 0.0 | 0.0 | 1.0 | 55.56 dB | PASS | **PROMOTE** | +1 |

**Aggregate NET VC delta: +3** (three cohort-race rescues).
**Aggregate perf delta vs comp: +6.97pp + +9.72pp + +17.35pp = +34.04pp aggregate above competitor.**

## Notes
- All 10 seeds in [101..1010] are INDEPENDENT (per R45 protocol).
- All three shapes hit identical SNR ~55.5–55.6 dB with `wrong_cell_frac = 0.0` and `kernel_finite = 1.0` across all 10 seeds — bit-deterministic correctness.
- E2_1 and E2_2 are recipe-exact analogs of R52D2A `(4096, 28672, 32768)` 256x256 (proven 101.78%); both delivered 106-109% comp this round.
- E2_3 (M=16384, N=4096) had the smallest grid (gdx=16, gdy=64) but produced the highest comp% (117.35%) — small-N + high-M cohort scales well on 256x256 tile.
- All three shapes were R52 VC then lost VC under R53 cohort tail-draw on UNCHANGED R40B HK `.so` files — direct evidence that the aiter `.co` deterministic path is structurally superior for these shapes.

## Mechanism summary
The R50D `.co` dlopen pattern is now proven across **at least 12 distinct shapes** (3 from R50D/R51, 4 from R52, 3 from R53D3B 64x1024, and now these 3 from R54E2 256x256). The pattern reliably converts cohort-race-flake R40B HK kernel results into bit-deterministic verified-correct status with `wcf=0.0` across all seeds — the structural rescue mechanism for shapes where HK auto-tune saturated below the deterministic frontier.

## Pre-bench verification (per R53D3B precedent)
- 256x256 .co exists: `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` (35632 bytes, mtime 2026-03-27).
- R50D shim: `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` present, used AS-IS.
- bdx=256 hardcoded in shim; matches kd metadata for 256x256 tile (proven across R50D/R51/R52).

## Artifacts
- Bench scripts: `bench_R54E2_1.py`, `bench_R54E2_2.py`, `bench_R54E2_3.py`
- SMOKE: `R54_OPT_E2_{1,2,3}_SMOKE.{json,log}`
- 10-RUN: `R54_OPT_E2_{1,2,3}_10RUN.{json,log}`
- Integration fragments: `R54E2_{1,2,3}_INTEGRATION_FRAGMENT.json`
- Shim (reused AS-IS): `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- aiter `.co`: `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`

## GPU usage
- All three candidates: GPU 4 (per worker assignment, isolated from other E-1/E-3/E-4 workers).
- rocm-smi pre-check: GPU 4 GFX use 0%, low-power state — confirmed idle before launch.

## Bench rules compliance
- warmup=200, iters=500, trim_frac=0.10 — confirmed in scripts.
- HIP_VISIBLE_DEVICES=4 on idle GPU — confirmed.
- INDEPENDENT seeds [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010] for 10-run.
- Strict gate: n_OK>=8/10 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min>=0.97 — **all three shapes satisfy with maximum margin (n_OK=10, wcf=0.0, wcf_std=0.0, fin_min=1.0)**.
- No `pgrep -f X` self-matching wait loops used (per R50C bug avoidance).
