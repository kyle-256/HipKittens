# R54 Opt D-4B Verdict — aiter `.co` dlopen, marginal HK-VC claw-back (256x256 tile)

## TL;DR
**3/3 PROMOTE** at strict 10-run gate. **+0 NET VC** (all three were already VC) but **+59.27pp aggregate perf** across the cohort (mean +19.76pp per shape vs HK). R50D shim used **AS-IS, no rebuild**. All three shapes well exceed the D-3C-style 0.5pp marginal gate — by a factor of ~40x.

## Per-shape results (10-RUN strict gate)

| # | Shape (M,N,K) | Tile | Grid (gdx,gdy,gdz,bdx) | comp | HK% | aiter tflops | aiter % | delta pp | n_OK | wcf_max | wcf_std | fin_min | snr_med_min | gate | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | (4096, 4096, 16384) | 256x256 | (16,16,1,256) | 4642.1 | 93.30% | 5250.7 | 113.11% | **+19.81** | 10/10 | 0.0 | 0.0 | 1.0 | 55.58 dB | PASS | **PROMOTE** |
| 2 | (4096, 14336, 8192) | 256x256 | (56,16,1,256) | 4345.8 | 91.92% | 4864.0 | 111.92% | **+20.00** | 10/10 | 0.0 | 0.0 | 1.0 | 55.59 dB | PASS | **PROMOTE** |
| 3 | (16384, 4096, 7168) | 256x256 | (16,64,1,256) | 4443.2 | 93.76% | 5030.5 | 113.22% | **+19.46** | 10/10 | 0.0 | 0.0 | 1.0 | 55.58 dB | PASS | **PROMOTE** |

**Aggregate perf delta: +59.27pp across 3 shapes (mean +19.76pp / shape).**
**Aggregate NET VC delta: +0** (all three were already verified-correct).

## Notes
- All 10 seeds in [101..1010] are INDEPENDENT.
- Sister-shape transfer confirmed: D4B_1 (4096,4096,16384) at 113.11% closely tracks R52D2C (4096,4096,32768) at 105.94% — same M,N geometry, different K, both >100% comp, supporting "256x256 aiter pattern is grid/M,N-generic".
- All three shapes hit identical SNR ~55.6 dB with `wrong_cell_frac = 0.0` and `kernel_finite = 1.0` across all 10 seeds — extremely clean correctness, indistinguishable from prior R50D/R51/R52/R53 256x256 promotions.
- D-3A-1 risk (SMOKE shows AITER < HK by >3pp) NOT triggered: all SMOKEs showed AITER beat HK by ~20pp, far above the 3pp DEAD threshold.
- Total wall time: ~80s SMOKE + ~240s 10-run per shape = ~16 min for all 3 candidates.

## Mechanism summary
The R50D `.co` dlopen pattern is now proven across **at least 12 distinct aiter 256x256 cohort shapes** (R50D + R51 D-1/D-2 + R52 D-2A/B/C + R53 D-3B_3 + R54 D-4B_1/2/3). The shim's `tile_M`/`tile_N` pybind kwargs and runtime `co_path`/`kernel_name` strings make it shape-generic for any aiter `.co` whose `bdx == 256` and `bpreshuffle == 1`.

This round demonstrates that the **aiter 256x256 ASM kernel beats HipKittens on this 90-95% HK cohort by ~20pp uniformly**, strongly suggesting the existing HK 256x256 path has a ~20pp headroom against the aiter reference. The D-4B candidates were specifically chosen as the "marginal claw-back" cohort, but the perf gap is anything but marginal — the cohort label was conservative.

## Artifacts
- Bench scripts: `bench_R54D4B_1.py`, `bench_R54D4B_2.py`, `bench_R54D4B_3.py`
- SMOKE: `R54_OPT_D4B_{1,2,3}_SMOKE.{json,log}`
- 10-RUN: `R54_OPT_D4B_{1,2,3}_10RUN.{json,log}`
- Integration fragments: `R54D4B_{1,2,3}_INTEGRATION_FRAGMENT.json`
- Shim (reused AS-IS): `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- aiter `.co`: `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`

## GPU usage
- All shapes: GPU 6 (verified idle via `rocm-smi` before launch).

## Bench rules compliance
- warmup=200, iters=500, trim_frac=0.10 — confirmed in scripts.
- HIP_VISIBLE_DEVICES=6 on idle GPU — confirmed.
- INDEPENDENT seeds [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010] for 10-run.
- Strict gate: n_OK>=8/10 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min>=0.97 — **all three shapes satisfy (10/10, 0.0, 0.0, 1.0)**.
- D-3C-style perf gate: aiter pct_comp > HK pct_comp + 0.5 pp — **all three exceed by ~20pp** (~40x the gate).
- No `pgrep -f X` self-matching wait loops used.
