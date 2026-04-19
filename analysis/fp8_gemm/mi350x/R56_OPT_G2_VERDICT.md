# R56 Opt G-2 Verdict — 64x1024 → 256x256 swap on cluster B (L3, L4, L5)

**Date:** 2026-04-19
**Worker:** R56 G-2 (very-high-confidence 256×256 swap)
**GPUs:** 2, 3 (verified idle pre-launch)
**Shim:** `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` AS-IS (8th consecutive R50D reuse)
**Bench rules:** warmup=200, iters=500, trim=0.10; INDEPENDENT seeds [101,202,303,404,505,606,707,808,909,1010]
**.co binary:** `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`
**Kernel:** `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E`

## Result table

| Cell | Shape (MxNxK) | Grid (gdx,gdy) | Current pct_comp | Alt pct_comp (10-run perf seed) | Δpp | n_OK | wcf_max | fin_min | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| L3 | 128256x32768x4096 | (128, 501) | 86.93% | **100.53%** | **+13.60** | 10/10 | 0.0 | 1.0 | **PROMOTE** |
| L4 | 28672x32768x4096  | (128, 112) | 87.82% | **102.07%** | **+14.25** | 10/10 | 0.0 | 1.0 | **PROMOTE** |
| L5 | 14336x32768x4096  | (128, 56)  | 87.92% | **104.00%** | **+16.08** | 10/10 | 0.0 | 1.0 | **PROMOTE** |

**3/3 PROMOTE. Aggregate Δ = +43.93 pp.**

## Bit-determinism (auto-pass for AITER .co)

All 30 runs (3 cells × 10 seeds): `wcf_max=0, wcf_std=0, fin_min=1.0`. SNR_med ~55.55 dB across the board (≥48 dB gate). VC strict gate (n_OK≥8/10, wcf_max<0.02, wcf_std<0.01, fin_min≥0.97) passes trivially.

## Mechanism confirmation

R55 D-5A/D-5B / E-3 / E-4 confirmed 256×256 wins for AITER on N≥4096 shapes. G-2 extends this to N=32768 cluster: identical `local_round` count between 64×1024 and 256×256 implies aiter heuristic should have picked 256×256 by `compute2mem_efficiency` tiebreak (128 vs 60.2) — manually forcing 256×256 via R50D shim restores the heuristic-optimal pick. All three shapes saturate at ~4560–4641 TFLOPS (≈100% of competitor) where the prior 64×1024 plateau was 3923–3944 TFLOPS.

## Files

- Bench scripts: `bench_R56G2_L{3,4,5}.py`
- Smoke logs/json: `R56_OPT_G2_L{3,4,5}_SMOKE.{json,log}`
- 10-run logs/json: `R56_OPT_G2_L{3,4,5}_10RUN.{json,log}`
- Integration fragments: `R56G2_L{3,4,5}_INTEGRATION_FRAGMENT.json`

## DO-NOT-do confirmations

- No R50D shim modifications (AS-IS reuse, 8th consecutive round).
- No HK kernel rebuild.
- warmup=200 / iters=500 / trim=0.10 enforced on all 30 runs.
- GPUs 2,3 only (other GPUs untouched).
