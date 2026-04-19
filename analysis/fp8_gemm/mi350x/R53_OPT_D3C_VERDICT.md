# R53 Opt D-3C — VERDICT

## Summary
3/3 PROMOTE — all three 64×1024 medium-K cohort shapes passed strict 10-run gate via R50D aiter `.co` dlopen shim **AS-IS** (no rebuild). Net VC stable at all three shapes; perf gains are smaller than decider expectation (decider: +10-15pp; observed: +0.5-2.4pp vs HK baseline) but still positive across the cohort.

## Per-shape verdicts

| Shape (M,N,K) | TFLOPS | pct_comp | HK base | Δ pp | n_OK_10 | wcf_max | wcf_std | fin_min | Gate |
|---|---|---|---|---|---|---|---|---|---|
| (14336, 32768, 4096) | 3897.7 | 87.34 | 86.06 | **+1.28** | 10/10 | 0.0 | 0.0 | 1.0 | PASS |
| (28672, 32768, 4096) | 3923.6 | 87.84 | 87.31 | **+0.53** | 10/10 | 0.0 | 0.0 | 1.0 | PASS |
| (16384,  4096, 14336) | 4621.0 | 89.87 | 87.50 | **+2.37** | 10/10 | 0.0 | 0.0 | 1.0 | PASS |

All three pass the 80% strict gate (n_OK≥8/10, wcf_max<0.02, wcf_std<0.01, fin_min≥0.97).
SNR_med ≈ 55.5 dB across all 30 (10-seed × 3-shape) runs — clean correctness, no cohort race.

## Shim rebuild status
**NO REBUILD REQUIRED.** Verified by full read of `R50D_aiter_dlopen.cpp`:
- `tile_M`, `tile_N`, `co_path`, `kernel_name` are all runtime kwargs (lines 220-226 of cpp).
- `bdx=256` is hardcoded (line 187), but aiter's `kernel_dispatch` (`asm_gemm_a4w4.cu` line 290) uses bdx=256 for ALL tile configurations including 64×1024 (4 wv64). Match confirmed.
- KernelArgs ABI is tile-independent (372 bytes, M/N/K + ptr/stride layout only).
- aiter `.co`: `f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024.co` exists with symbol `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024E` (verified via `nm`).

R52 D-2A→C shim (256×256) is reused AS-IS for the **5th consecutive round** of the aiter `.co` dlopen pattern.

## D-3B coordination
At R53 D-3C kickoff, no `build_R53D3*/` directory existed and no `R53*D3B*` artifact existed.
D-3C did not need to rebuild (validated above), so coordination axis was moot.

## GPU isolation
Shape 1 + Shape 2: GPU 6 (HIP_VISIBLE_DEVICES=6).
Shape 3: GPU 7 (HIP_VISIBLE_DEVICES=7).
Avoided GPUs 4 and 5 per task instruction.
warmup=200, iters=500, trim=0.10, INDEPENDENT seeds [101..1010] for 10-run.

## Notes for decider
- Decider's "+10-15pp expected" overshoots reality. The 64×1024 aiter tile barely beats current HK on these N=32768, K=4096 shapes (Shape 1 and 2). On Shape 3 (N=4096, K=14336) the gain is the largest at +2.37pp.
- All three shapes are still **below** competitor (`comp` 4462-5142 TF, achieved 87-90%); the rule "all 42 shapes must beat aiter" is NOT satisfied here. The 64×1024 .co is *aiter's own choice* per its CSV but the .co itself loses to the dispatched 256×256 path on grids of this size.
- Recommend PROMOTE all 3 since they net-improve current HK production with zero correctness risk and zero shim-rebuild cost. If decider cares about strictly beating `competitor_tflops`, all three remain LOSE in the strict 42-shape sense.

## Artifact paths (all under `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/`)
- `bench_R53D3C_1.py`, `bench_R53D3C_2.py`, `bench_R53D3C_3.py`
- `R53_OPT_D3C_{1,2,3}_SMOKE.{json,log}`
- `R53_OPT_D3C_{1,2,3}_10RUN.{json,log}`
- `R53D3C_{1,2,3}_INTEGRATION_FRAGMENT.json`
- Shim: `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` (UNCHANGED)
- aiter .co: `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024.co`
