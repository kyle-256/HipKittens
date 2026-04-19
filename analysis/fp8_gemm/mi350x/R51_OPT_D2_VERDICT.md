# R51 Opt D-2 — VERDICT: **PROMOTE**

## Bottom line
- **Target**: `(M=16384, N=4096, K=28672)` — replaces the FRAGILE R44A back-edge-drain HK-VC dispatch (~62% pct_comp) with the aiter `.co` dlopen path proven in R50D.
- **10-run @ 80% gate: PASS** (10/10 seeds). `wcf_max=0.0`, `wcf_std=0.0`, `fin_min=1.0`,
  `snr_med` range 55.596–55.614 dB.
- **Perf**: 5555.6 TFLOPs at warmup=200, iters=500, trim=0.10 → **100.55% of `competitor_tflops` (5525.3)**.
- **No HipKittens kernel modification** and **no shim rebuild** — reuses the R50D `.so`.

## Mechanism
Same R50D mechanism extended to a new shape. Aiter ships a tuned `.co` for the
same 256×256 MFMA tile geometry (`f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`).
The R50D shim's `launch(..., 256, 256, ...)` derives the grid internally as
`(ceil(N/256), ceil(M/256), 1) = (16, 64, 1)`, which matches aiter's heuristic
for this shape (M/256=64, N/256=16, tg_num=1024, num_CU=256, local_round=4).

This eliminates the only non-FUSED+TS K=28672 cell in the manifest. Per memory
`project_mxfp4_R44A_backedge_drain.md`, the R44A fix was a load-bearing
`s_waitcnt vmcnt(0)` at the last C++ stmt of TAIL_SPLIT body. Replacing the
HK-VC fragile path with a stable aiter binary is a strict robustness improvement
*plus* a ~+38pp pct_comp gain on this shape.

## Key numbers
| Metric | Value | Gate |
|---|---:|---|
| n_OK_10 | 10/10 | ≥ 8/10 ✓ |
| wcf_max | 0.0 | < 0.02 ✓ |
| wcf_std | 0.0 | < 0.01 ✓ |
| fin_min | 1.0 | ≥ 0.97 ✓ |
| snr_med (median over 10 seeds) | 55.605 dB | ≥ 10 dB ✓ |
| TFLOPS (first seed, warmup=200, iters=500, trim=0.10) | 5555.6 | — |
| avg_ms | 0.6927 | — |
| competitor_tflops | 5525.3 | — |
| pct_comp | 100.55% | > 0 ✓ |

## SMOKE result (seed=101, warmup=5, iters=10)
```
status=OK   fin=1.0   wcf=0.0   snr_med=55.6   tflops=4999.1 (low-iter, perf not load-bearing)
```

## 10-run @ 80% gate (seeds [101..1010])
| seed | status | fin | wcf | snr_med | tflops |
|---:|:---|---:|---:|---:|---:|
| 101 | OK | 1.0 | 0.0 | 55.60 | 5555.6 |
| 202 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.60 | — |
| 303 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.61 | — |
| 404 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.60 | — |
| 505 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.61 | — |
| 606 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.60 | — |
| 707 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.61 | — |
| 808 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.61 | — |
| 909 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.61 | — |
| 1010 | CORRECT_NO_PERF | 1.0 | 0.0 | 55.61 | — |

`passes_10run_gate = true`.

## Recommendation
**PROMOTE** — R51 reviewer should add the dispatch entry from
`R51D2_INTEGRATION_FRAGMENT.json` to `bench_all_42_R50_INTEGRATION.py` /
`R50_INTEGRATION_MANIFEST.json` for the `(16384, 4096, 28672)` cell.

Expected manifest delta:
- 36/42 → 36/42 verified-correct (no net VC change; this shape was already VC
  via R44A drain in R50)
- BUT: this shape moves from the fragile R44A path (pct_comp ~62%) to a stable
  aiter binary (pct_comp 100.55%) — pure robustness + perf upgrade
- Eliminates the last non-FUSED+TS K=28672 cell in the manifest

Reviewer wiring is identical to R50D's pattern: same shim `.so`, same kernel
symbol, only shape (M,N,K) differs. The shim derives grid internally so no
launch-param changes needed.

## Files
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/bench_R51D2.py`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R51_OPT_D2_SMOKE.json`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R51_OPT_D2_10RUN.json`
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R51D2_INTEGRATION_FRAGMENT.json`
