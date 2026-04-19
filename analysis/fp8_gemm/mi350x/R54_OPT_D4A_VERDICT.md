# R54 Opt D-4A Verdict — aiter `.co` dlopen for sub-90% HK-VC rescue (256x256 tile)

## TL;DR
**3/3 PROMOTE** at strict 10-run gate. **+0 NET VC** (all three were already verified-correct on HK), but **+70.51pp aggregate perf claw-back** across the three sub-90% HK-VC shapes — largest single-round perf gain on already-VC shapes since the program began. R50D shim used **AS-IS, no rebuild**.

## Per-shape results (10-RUN strict gate, GPU 5)

| # | Shape (M,N,K) | Prior (HK pct) | comp | tflops | aiter pct | HK pct | Δpp | n_OK | wcf_max | wcf_std | fin_min | snr_med_min | gate | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | (4096, 14336, 16384) | VC R40B 84.41% | 5013.0 | 5428.3 | 108.28% | 84.41% | **+23.87** | 10/10 | 0.0 | 0.0 | 1.0 | 55.58 dB | PASS | **PROMOTE** |
| 2 | (32768, 4096, 7168)  | VC R40B 88.36% | 4666.8 | 4983.0 | 106.78% | 88.36% | **+18.42** | 10/10 | 0.0 | 0.0 | 1.0 | 55.58 dB | PASS | **PROMOTE** |
| 3 | (6144, 4096, 8192)   | VC R40B 89.53% | 3822.0 | 4500.5 | 117.75% | 89.53% | **+28.22** | 10/10 | 0.0 | 0.0 | 1.0 | 55.59 dB | PASS | **PROMOTE** |

**Aggregate perf delta: +70.51 pp** (mean +23.50 pp).
**NET VC delta: 0** (all three already VC under HK).

## Notes
- All 10 seeds in [101..1010] are INDEPENDENT.
- All three hit aiter's clean signature: `wrong_cell_frac = 0.0`, `kernel_finite = 1.0`, `snr_med_db ≈ 55.6 dB` across every seed — same correctness profile observed in R50D/R51/R52/R53D3B.
- D-3A risk (96×640-style aspect-ratio failure) did NOT materialize: 256×256 tile remains universal across all M,N,K aspect ratios tried so far.
- D4A_1 was flagged as sister of R51D1 `(14336,4096,32768)` (103.17%); actual delivery 108.28% — even stronger.
- All three perf gates clear by huge margins (>+18pp), well above the +1.0pp D-4A threshold.

## Mechanism summary
The R50D `.co` dlopen pattern is now proven across **at least 12 distinct aiter tile/shape combos** (256×256: R50D + R51 + R52 + R54 D-4A 1/2/3; 64×1024: R53D3B; 96×640: R53 partial). For sub-90% HK-VC shapes, the aiter ASM kernel substantially outperforms the best HK Gluon variant — confirming HK's HSA / scheduling overhead (rather than memory bandwidth) is the bottleneck on these mid-K shapes. The shim's `tile_M`/`tile_N` pybind kwargs and runtime `co_path`/`kernel_name` strings made all three rescues a zero-rebuild operation.

## Bench rules compliance
- warmup=200, iters=500, trim_frac=0.10 — confirmed in scripts.
- HIP_VISIBLE_DEVICES=5 on idle GPU — confirmed (rocm-smi pre-launch: 0% on GPU 5).
- INDEPENDENT seeds [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010] for 10-run.
- Strict gate: n_OK ≥ 8/10 AND wcf_max < 0.02 AND wcf_std < 0.01 AND fin_min ≥ 0.97 — **all three shapes satisfy with margin**.
- Perf gate (D-4A specific): aiter pct_comp > HK pct_comp + 1.0 pp — all three exceed by **>17.4pp**.
- No `pgrep -f X` self-matching wait loops used.

## Artifacts
- Bench scripts: `bench_R54D4A_1.py`, `bench_R54D4A_2.py`, `bench_R54D4A_3.py`
- SMOKE: `R54_OPT_D4A_{1,2,3}_SMOKE.{json,log}`
- 10-RUN: `R54_OPT_D4A_{1,2,3}_10RUN.{json,log}`
- Integration fragments: `R54D4A_{1,2,3}_INTEGRATION_FRAGMENT.json`
- Shim (reused AS-IS): `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- aiter `.co`: `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`

## GPU usage
- All 3 candidates: GPU 5 (sequential, per task spec)
- rocm-smi pre-launch confirmed GPU 5 idle (0% use)
