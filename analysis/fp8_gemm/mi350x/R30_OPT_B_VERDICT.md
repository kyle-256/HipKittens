# R30 Optimizer B: K_EXACT Parent-Stack Audit — VERDICT

**Status: AUDIT-CLEAN (with one verified-LOSE probe). No new K_EXACT entries to add.**

## Executive summary

Audited all 11 K_EXACT entries in `bench_all_42.py` (lines 510-600) against
`bench_all42_results_R25_FINAL_v2.json`. Found one plausible cross-shape
parent-mismatch hypothesis (L7 `32768x4096x14336` might benefit from R28C u16
or R29 tv0-plain stacks since L8 same N,K does). Bench-disproved both:
both LOSE on L7 vs the current `_dc_gm7` best by 0.93% and 4.17%. No other
K_EXACT entry shows a per_variant pattern suggesting an alternate-parent win.

## Step 1: Audit table

For each shape with K ∈ {2048, 6144, 7168, 8192, 14336, 16384, 28672, 32768},
listing (a) current best variant, (b) which K_EXACT entry was picked (if any),
(c) the top non-K_EXACT competitor.

| Shape | Best TFLOPS | Best variant | Non-K_EXACT 2nd | Gap | Verdict |
|---|---|---|---|---|---|
| 16384x4096x2048 | 3414.5 | **kx2048** ts_v12_gm7_memc_pfoff4 | ts_gm6_v12_memc_dc_pfoff4 (3408.6) | +0.17% | OK — K_EXACT wins |
| 16384x6144x2048 | 3471.8 | ts_lgk2_gm6_v12_memc_pfoff4 | (kx2048 = 3452.8) | +0.55% | autotuner picks correct non-KX; no fix |
| 32768x4096x2048 | 3472.2 | ts_lgk2_gm6_v12_memc_pfoff4 | (kx2048 = 3460.9) | +0.33% | autotuner picks correct non-KX |
| 32768x6144x2048 | 3728.6 | ts_lgk2_gm6_v12_memc_pfoff4 | (kx2048 = None — build never ran) | n/a | autotuner picks correct non-KX |
| 16384x14336x2048 | 3645.9 | **kx2048** ts_v12_gm7_memc_pfoff4 | ts_lgk2_gm6_v12_memc_pfoff4 (3639.4) | +0.18% | OK — K_EXACT wins |
| 16384x28672x2048 | 3764.1 | ts_lgk2_gm6_v12_memc_pfoff4 | (kx2048 = 3754.5) | +0.26% | autotuner picks correct non-KX |
| 32768x14336x2048 | 3725.5 | ts_gm6_v12_memc_dc_pfoff4 | (kx2048 = None) | n/a | autotuner picks correct non-KX |
| 32768x28672x2048 | 3718.1 | ts_lgk2_gm6_v12_memc_pfoff4 | (kx2048 = None) | n/a | autotuner picks correct non-KX |
| 4096x4096x16384 | 5723.6 | **kx16384** ts_lgk2_gm7_pfoff56 | ts_lgk2_memc_btw_all (5065.7) | +13.0% | OK — single K_EXACT dominates |
| 4096x14336x16384 | 5781.5 | **kx16384** ts_lgk2_gm7_pfoff56 | ts_v12_tv0_memc_btw_all (5072.4) | +13.9% | OK |
| 6144x4096x16384 | 5153.9 | **kx16384** ts_lgk2_gm7_pfoff56 | v20_memc_btw_step3 (4640.8) | +11.1% | OK |
| 4096x4096x8192 | 4501.7 | **kx8192** ts_v12_tv0_gm7_pfoff28 | ts_lgk2_memc_btw_all (4350.9) | +3.5% | OK |
| 4096x4096x32768 | 5940.0 | **kx32768** ts_v12_tv0_dc_gm7_pfoff120 | ts_pf4_memc_btw_step3 (5266.5) | +12.8% | OK; alt-parent kx32768 (lgk2) = 5842.5 |
| 4096x6144x32768 | 5629.8 | **kx32768** ts_v12_tv0_dc_gm7_pfoff120 | v20_memc_btw_step3 (4896.7) | +15.0% | OK; alt-parent kx32768 lgk2 = 5449.6 |
| 4096x14336x8192 | 5057.2 | **kx8192** ts_v12_tv0_gm7_pfoff28 | ts_pf4_memc_btw_step3 (4568.0) | +10.7% | OK |
| 4096x28672x32768 | 6531.4 | **kx32768** ts_v12_tv0_dc_gm7_pfoff120 | ts_lgk2_btw_step3 (5457.6) | +19.7% | OK |
| 4096x32768x6144 | 5285.0 | **kx6144** ts_v12_gm7_pfoff19 | ts_v12_tv0_memc_btw_all (4566.8) | +15.7% | OK |
| 4096x32768x14336 (L4) | 5217.0 | ts_v12_tv0_memc_btw_all (in v2 JSON) | n/a | n/a | **R29 fix already added** to bench_all_42 — no audit |
| 4096x32768x28672 | 6492.0 | **kx28672** ts_lgk2_gm7_pfoff104 | v20_memc_btw_step3 (5334.6) | +21.7% | OK |
| 4096x128256x32768 | 6373.4 | **kx32768** ts_v12_tv0_dc_gm7_pfoff120 | ts_lgk2_btw_step3 (5213.9) | +22.2% | OK |
| 6144x4096x8192 | 4305.5 | **kx8192** ts_v12_tv0_gm7_pfoff28 | v20_memc_btw_step3 (4136.1) | +4.1% | OK |
| 14336x4096x32768 | 6079.5 | **kx32768** ts_v12_tv0_dc_gm7_pfoff120 | ts_lgk2_btw_step3 (5026.1) | +21.0% | OK |
| 16384x4096x6144 | 4966.9 | **kx6144** ts_v12_gm7_pfoff19 | ts_v12_tv0_memc_btw_all (4679.4) | +6.1% | OK |
| 16384x4096x7168 | 5250.4 | **kx7168** ts_v12_tv0_gm7_pfoff24 | ts_pf4_memc_btw_step3 (4856.4) | +8.1% | OK |
| 16384x4096x14336 (L8) | 5996.9 | kx14336 ts_v12_tv0_dc_gm7_pfoff54 (R28C u16 added later) | gm8_v12_btw_all (5250.6) | +14.2% | **R28C fix already added** |
| 16384x4096x28672 | 6407.4 | **kx28672** ts_lgk2_gm7_pfoff104 | v20_memc_btw_step3 (5330.3) | +20.2% | OK |
| 28672x4096x8192 | 5480.3 | **kx8192** ts_v12_tv0_gm7_pfoff28 | lgk2_dc_btw_step3 (4829.5) | +13.5% | OK |
| 28672x4096x16384 | 6238.2 | **kx16384** ts_lgk2_gm7_pfoff56 | ts_lgk2_memc_btw_all (5256.1) | +18.7% | OK |
| 32768x4096x7168 | 5342.4 | **kx7168** ts_v12_tv0_gm7_pfoff24 | ts_v12_tv0_memc_btw_all (4699.8) | +13.7% | OK |
| **32768x4096x14336 (L7)** | 6157.3 | **kx14336** ts_v12_tv0_memc_dc_gm7_pfoff54 | ts_lgk2_memc_btw_all (5273.8) | +16.7% | **PROBE: same N,K as L8 — does L7 also benefit from R28C u16 or R29 tv0-plain?** |

**Audit takeaways:**
- For all K ∈ {6144, 7168, 8192, 16384, 28672, 32768}: every existing K_EXACT entry
  dominates by 3-22% — no plausible parent-mismatch.
- For K=2048: K_EXACT wins on 2/9 shapes; non-K_EXACT (`ts_lgk2_gm6_v12_memc_pfoff4`,
  `ts_gm6_v12_memc_dc_pfoff4`) wins on 7/9. Gaps are 0.17-0.55%. **No 1%+ win is
  reachable** by adding a K_EXACT version of either non-K_EXACT (the K_EXACT macro
  on the same kernel just gates K==2048 — doesn't change codegen for that K).
- Three K=14336 shapes: L4 (R29), L8 (R28C) already have committed dedicated K_EXACT
  fixes. **L7 (32768x4096x14336)** is the only K=14336 shape NOT individually
  audited — current best is the OLDEST tv0-with-dc_gm7 entry. Worth probing.

## Step 2: Build status

Both L7 candidates were already built (same N=4096, K=14336 as L8/L4):

| Candidate | Source flag stack | SO file (already in build_all42/) |
|---|---|---|
| `u16_kx14336_R28C` | `-DUNROLL_K=16 -DGROUP_SIZE_M=7 -DSTEP3_BARRIER_VMCNT=12 -DR25C_TAIL_PF_OFF_ITERS=52 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=14336 -mllvm -amdgpu-sched-strategy=max-memory-clause -DBARRIER_TO_WAITCNT_ALL=1 -DTAIL_SPLIT=1` | `tk_mxfp4_gluon_cpp_n4096_k14336_ts_u16_gm7_pfoff52_kx14336_btw_all.so` |
| `tv0_btw_all_pfoff48_kx14336_R29` | `-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0 -DR25C_TAIL_PF_OFF_ITERS=48 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=14336 -mllvm -amdgpu-sched-strategy=max-memory-clause -DBARRIER_TO_WAITCNT_ALL=1` | `tk_mxfp4_gluon_cpp_n4096_k14336_ts_v12_tv0_memc_btw_all_pfoff48_kx14336.so` |

No new builds required.

## Step 3: Single-rep bench (L7 = 32768x4096x14336)

Per `benchmark-rules.md`: warmup=200, iters=500, trim=0.10. GPU 1 (idle; GPU 2
busy with Optimizer A; coordinated). Single-rep — script
`r30_optb_bench_l7.py`, JSON `R30_OPTB_BENCH_L7_reps1.json`.

| Tag | TFLOPS | Δ vs current best (6157.3) | vs comp (5223.4) | Verdict |
|---|---|---|---|---|
| u16_kx14336_R28C | 5900.67 | **−4.17%** | 112.97% | LOSE |
| tv0_btw_all_pfoff48_kx14336_R29 | 6100.21 | **−0.93%** | 116.79% | LOSE |

Both LOSE. L7's existing `ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all` is the
correct parent: the `_dc_gm7` stack benefits L7 (same as L4-vs-tv0-plain disjunction
proved L4 needs the no-dc_gm7 path). Hypothesis "L8's u16 fix or L4's no-dc_gm7
fix transfers to L7" is **disproved**.

## Step 4: 5-rep verify

Skipped — no candidate exceeded the >1% gain threshold.

## Final recommendation

**No additions to `bench_all_42.py`.**

The K_EXACT entry table is already audit-clean:
1. Each large-K K_EXACT entry (K ≥ 6144) dominates by 3-22% over all non-K_EXACT
   variants — no alternate-parent room for improvement.
2. K=14336 has three dedicated parent-specific entries (L4 tv0-plain, L7 dc_gm7,
   L8 u16) and L7's tv0-plain alternative is empirically slower (−0.93%).
3. K=2048 is correctly handled by the autotuner without K_EXACT-only entries on
   the per-shape best parent (lgk2_gm6_v12_pfoff4 / gm6_v12_dc_pfoff4) — adding
   a K_EXACT version cannot beat the same kernel by >1% at the same K.

**No commits, no edits to `bench_all_42.py`, no kernel changes.**

## Artifacts

- `r30_optb_bench_l7.py` — bench script
- `R30_OPTB_BENCH_L7_reps1.json` — single-rep results
- `R30_OPTB_BENCH_L7_singlerep.log` — bench log

## Bench environment

- MI355X (gfx950)
- GPU 1, isolated via `HIP_VISIBLE_DEVICES=1`
- warmup=200, iters=500, trim=0.10 (per `.claude/rules/benchmark-rules.md`)
- Coordinated with Optimizer A (GPU 2 busy at probe time)
