# R26-C: TAIL_BARRIER_VMCNT × R25 stack — ALL DEAD

Date: 2026-04-18  Branch: mxfp4  Optimizer: R26-C (claude-opus-4-7)
Vector: V4 from R26_PLAN.md (scoped to 4 highest-value shapes; full V4 is 50 builds, this was 20).

## Hypothesis
`TAIL_BARRIER_VMCNT` (kernel L70-71, defaults to `STEP3_BARRIER_VMCNT` = 8 unless overridden)
controls the s_waitcnt of the TAIL_SPLIT block. R25-F/G/H reshuffled which iterations issue
VMEM (TAIL_PF_OFF_ITERS turns off prefetch for last N iters). The optimal TAIL VMCNT may have
moved with the new VMEM-issue pattern. Per-shape sweep was never run with the R25 stack engaged.

## Method
- 4 shapes × 5 `TAIL_BARRIER_VMCNT` values {0, 4, 8, 12, 16} = 20 builds (parallel CPU build, 6.5 s p99).
- 5-rep bench on isolated GPU 6 (warmup=200 iters=500 trim=10%, hipEvent timing).
- WIN threshold: best non-default ≥ +1.5 pp vs current default with std ≤ 25 TFLOPS.
- All builds were extensions of the committed R25-F/G/H best variants for each shape:
  - DLA2 (128256×32768×4096): `_ts_gm7_v12_memc_dc_pfoff14`  default tv = STEP3_BARRIER_VMCNT = 12
  - DLA7  (28672×32768×4096): `_ts_lgk2_gm7_v12_memc_pfoff14` default tv = 12
  - SD    (16384×4096×28672): `_ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all` default tv = 8 (STEP3 not set)
  - SC    (4096×32768×6144):  `_ts_v12_gm7_memc_pfoff19_kx6144_btw_all`   default tv = 12

## Results (5-rep mean, GPU 6, %comp = mean / competitor_tflops)

| Shape | tv0 | tv4 | tv8 | tv12 | tv16 | default | Δ best vs default |
|-------|-----:|-----:|-----:|-----:|-----:|:-------:|:------------------|
| DLA2  | 4939.0 (108.87%) | 4932.2 (108.72%) | 4930.6 (108.69%) | **4926.8 (108.61%)** | 4929.3 (108.66%) | tv12 | tv0 +0.27 pp std=4.1  |
| DLA7  | 5041.4 (112.87%) | 5039.7 (112.83%) | 5043.2 (112.91%) | **5044.9 (112.95%)** | 5035.1 (112.73%) | tv12 | tv8 −0.04 pp std=2.8  |
| SD    | 6319.3 (114.37%) | 6301.8 (114.05%) | **6302.4 (114.06%)** | 6285.0 (113.75%) | 6292.3 (113.88%) | tv8  | tv0 +0.31 pp std=14.5 |
| SC    | 5229.1 (114.96%) | 5282.1 (116.13%) | 5253.5 (115.50%) | **5242.5 (115.26%)** | 5218.6 (114.73%) | tv12 | tv4 +0.87 pp std=21.2 |

## Verdict per shape

- **DLA2 — DEAD.** tv0 leads default tv12 by +0.27 pp (4939 vs 4927 TFLOPS); below 1.5 pp threshold. Std is tiny (4.1) so this is a real, but tiny, signal — not worth wiring.
- **DLA7 — DEAD.** Default tv12 is statistically the best (5044.9 vs tv8 5043.2). Sweep is flat within noise.
- **SD   — DEAD.** tv0 leads default tv8 by +0.31 pp (6319 vs 6302 TFLOPS); below threshold. Already +14.4% over competitor.
- **SC   — DEAD.** tv4 leads default tv12 by +0.87 pp (5282 vs 5243 TFLOPS); below threshold and std=21.2 is at the high end (rep 4 had a tv0 outlier 5136). Inconclusive even relative to noise.

## Mechanism takeaway

The `TAIL_BARRIER_VMCNT` axis is **flat across all 4 R25-best stacks**. The TAIL_SPLIT block
runs only ONCE per K-loop (last iteration), so its s_waitcnt vmcnt arg has tiny aggregate impact —
~1/K_iters of the runtime, even if mistuned. With R25-F/G/H already turning off most tail-iter
prefetches via `R25C_TAIL_PF_OFF_ITERS`, the *count* of inflight VMEM at the TAIL barrier is
already small, so the choice of vmcnt threshold (0 vs 16) is moot — there are not enough inflight
loads for the arg to make a difference.

## Recommendation

Do NOT extend V4 to the remaining 6 R25-G/H shapes. The mechanism is genuinely dead post-R25
because R25-F/G/H drained the tail-iter VMEM that would have made this barrier meaningful.

Next vector candidates (per R26_PLAN.md):
- V1 (R25-E peel for DLA1) — orthogonal axis (static loop split), not yet smoke-tested.
- V3 (`STEP3_PF_N` × R25-G stack on DLA2/DLA7) — different axis (steady-state, not tail).
- V2 (per-K pfoff for shapes not covered by R25-G/H) — zero-build audit first.
