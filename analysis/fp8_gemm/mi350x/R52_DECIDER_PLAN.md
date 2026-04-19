# R52 DECIDER PLAN — 3 parallel optimizers (post-R51 commit `1a71f8ae`)

**Date**: 2026-04-19
**Baseline**: R51 INTEGRATION 31/42 strict 10-run (37/42 mixed-protocol). Branch `mxfp4`.
**Mechanism in play**: R50D aiter `.co` dlopen shim (`build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`) — proven shape-generic for the 256×256 tile case across 4 distinct shapes (R50D, R51 D-1/2/3) with **zero rebuild**.
**Aiter heuristic verification**: ran `asm_gemm_a4w4.cu:100-145` heuristic (num_cu=256) over all sub-90% HK-VC shapes — every single one selects `256×256` tile (rounds×empty×eff tiebreak unanimous) → R50D shim is reusable AS-IS for all candidates below.

## Priority table

| pri | candidate | shape (M,N,K) | R51 baseline %comp | aiter tile | mechanism | shim rebuild? | expected pp gain | risk |
|----:|-----------|---------------|-------------------:|-----------:|-----------|:-------------:|-----------------:|:----:|
| 1   | Opt D-2A  | `4096x28672x32768`   | 61.9% (R41A, YES VC) | 256×256 | R50D shim AS-IS, per-shape dispatch entry | NO | +35 to +40 pp | **LOW** |
| 2   | Opt D-2B  | `4096x32768x128256`  | 72.3% (R40A, YES VC) | 256×256 | R50D shim AS-IS, per-shape dispatch entry | NO | +25 to +30 pp | **LOW** |
| 3   | Opt D-2C  | `4096x4096x32768`    | 77.1% (R41A, YES VC) | 256×256 | R50D shim AS-IS, per-shape dispatch entry | NO | +20 to +25 pp | **LOW** |

## Rationale

R51 proved 3/3 PROMOTE on the same mechanism. The remaining HK-VC shapes <80% comp are all 256×256-tile-optimal per the aiter heuristic. The next 3 highest-gap shapes are the highest-EV slot — each unlocks +20-40pp comp at zero kernel risk and zero shim rebuild cost. We pick **3 perf claw-backs**, NOT VC-rescue or non-256 tile work, because:
1. **Opt D-extended-2 from TODO (`32768x4096x14336`, `16384x28672x4096`)**: both are NO-VC (cohort-race tail-draws), so promoting them via aiter is +VC AND +pp — but Opt D-2A `4096x28672x32768` is a much bigger comp hole (61.9% vs ~85%) AND already-VC, so claiming it carries lower correctness risk and bigger pp delta. Reordering: Opt D-2A first.
2. **Opt D-non-256x256**: rejected for R52. Would require shim rebuild + new KernelArgs schema + per-tile heuristic table — at least 1 extra round of integration risk for shapes whose comp gap is unknown. Save for R53+ once we exhaust 256×256 candidates.
3. **Opt B (32×32×64 MFMA)**: rejected for R52. High kernel-rewrite cost, untried structural axis, unclear blast radius on the 31 currently-VC shapes. Save for the round where 256×256 pickings are exhausted.

## Per-candidate spec

### Opt D-2A — `4096x28672x32768` (BIGGEST gap)
- **Current**: HK R41A, 10/10 OK, p50=3494.4 TFLOPS = **61.9%** of competitor 5,649.9.
- **Aiter heuristic**: tile=256×256, rounds=7, empty=0, eff=128.0 (unanimous winner over 12 other tiles).
- **Aiter `.co`**: `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`
- **Per-shape grid**: gdx=ceil(N/256)=112, gdy=ceil(M/256)=16, gdz=1, bdx=256.
- **KernelArgs**: M=4096, N=28672, K=32768 (only the M/N/K differ from R51 D-1/2/3).
- **Expected**: 100-103% comp → +38 to +41 pp. Zero VC count change (already VC), pure perf claw-back.
- **Shim**: REUSE `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` AS-IS, no rebuild.
- **Worker contract**: produce `R52_OPT_D2A_VERDICT.md`, `R52D2A_INTEGRATION_FRAGMENT.json`, `bench_R52D2A.py`, `R52_OPT_D2A_{SMOKE,10RUN}.json`. 10-run @ 80% INDEPENDENT seeds [101..1010] required.

### Opt D-2B — `4096x32768x128256` (2nd biggest gap)
- **Current**: HK R40A, 10/10 OK, p50=4181.4 TFLOPS = **72.3%** of competitor 5,781.1.
- **Aiter heuristic**: tile=256×256, rounds=8, empty=0, eff=128.0.
- **Aiter `.co`**: same `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`.
- **Per-shape grid**: gdx=128, gdy=16, gdz=1, bdx=256.
- **KernelArgs**: M=4096, N=32768, K=128256.
- **Expected**: 95-103% comp → +25 to +30 pp. K=128256 is unusually large; if aiter `.co` chokes on K-iter count, fallback is HK R40A baseline (no regression).
- **Shim**: REUSE AS-IS, no rebuild.
- **Worker contract**: parallel to Opt D-2A.

### Opt D-2C — `4096x4096x32768` (3rd biggest gap, simplest shape)
- **Current**: HK R41A, 10/10 OK at wcf_max=0.0 fin_min=1.0, p50=3973.4 TFLOPS = **77.1%** of competitor 5,152.8.
- **Aiter heuristic**: tile=256×256, rounds=1, empty=0, eff=128.0 — single grid round, ideal occupancy.
- **Aiter `.co`**: same.
- **Per-shape grid**: gdx=16, gdy=16, gdz=1, bdx=256.
- **KernelArgs**: M=4096, N=4096, K=32768.
- **Expected**: 100-105% comp → +23 to +28 pp. Single-round grid means aiter's compute2mem efficiency advantage maximally exploited.
- **Shim**: REUSE AS-IS, no rebuild.
- **Worker contract**: parallel to Opt D-2A/B.

## Integration plan (decider note for reviewer)

After workers PROMOTE: merge 3 new per-shape backend dispatch entries into manifest, run R52_INTEGRATION 10-run @ 80% on all 42 shapes. Targets:
- **VC count**: 31/42 → 31/42 (no VC delta expected — all 3 candidates already VC; this is pure perf round).
- **Mean pp comp on 26 R51-shared-VC + 3 R52-promote shapes**: expect +5 to +12 pp/shape average uplift, dominated by the 3 promotions (+25-40 pp each, distributed across 29 shapes ≈ +3-4 pp/shape mean addition on top of current trend).
- **Aggregate TFLOPS**: expect +2,500 to +3,500 TFLOPS sum across the 3 promotes alone.

## Stop criteria for R52

PROMOTE iff worker delivers 10/10 OK @ wcf_max<0.02, wcf_std<0.01, fin_min≥0.97, AND p50 TFLOPS ≥ 100% × competitor (perf-positive vs aiter heuristic baseline). REVERT any worker not meeting all 4 gates — no partial promotes (lessons from R50C 5-run flake).

## Out-of-scope for R52

- **Opt B (32×32×64 MFMA)** — defer until 256×256 perf-claw-back candidates are exhausted.
- **Opt D-non-256x256** — defer; needs shim rebuild + tile_m/tile_n KernelArgs schema extension + per-tile heuristic table.
- **Cohort-race shape rescue** (`32768x4096x14336`, `16384x28672x4096`, etc.) — viable next round but lower confidence (NO-VC baselines need reviewer to confirm aiter PASSES at 10-run). Schedule for R53.
- **Any kernel modification** — R51 hit WIN with zero kernel changes; keep that streak.

## Files referenced

- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R51_INTEGRATION_VERDICT.md` (per-shape baseline)
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R51_INTEGRATION_10RUN.json` (full 10-run JSON)
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` (REUSED AS-IS)
- `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R50D_aiter_dlopen.cpp` (shim source — only KernelArgs M/N/K differ per shape)
- `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` (aiter binary)
- `/shared_nfs/kyle/test/aiter/csrc/py_itfs_cu/asm_gemm_a4w4.cu:100-145` (heuristic — verified all 3 picks)
