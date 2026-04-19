# R40 Reviewer Report — Validate R40B (FUSED_STEP34) as new BEST_VARIANTS baseline

**Date**: 2026-04-19
**Reviewer**: R40 reviewer agent
**Inputs**: R40B 3-run consensus (`R40_OPT_B_BENCH.json`), R40B 5-run re-bench
(`R40B_5RUN_CONSENSUS.json`), R39B baseline (`bench_all42_results_R39_optB.json`),
R40A (`agent-a9249715/R40_OPT_A_BENCH.json`), R40C (`agent-afccfa0b/.../R40_OPT_C_BENCH.json`).
**Bench params (all runs)**: warmup=200, iters=500, trim=0.10, GPUs 0-7, seed=42,
random scales [-2, 2], gate = `wrong_cell_frac<2% AND snr_med>=10dB AND finite>=0.99`.

---

## Headline

| Metric | R39B | R40B (3-run, original verdict) | R40B (5-run, this re-bench) |
|---|---|---|---|
| Verified-correct | 5/42 | 25/42 | **24/42** |
| WIN (tflops>=comp) | 3/42 | 6/42 | 7/42 |
| LOSE_CORRECT | 2/42 | 19/42 | 17/42 |
| WRONG_OUTPUT | 33/42 | 15/42 | 16/42 |
| CRASH/ERR | 4/42 | 2/42 | 2/42 |
| Regressions vs R39B | n/a | 0 | **0** |

**1 shape lost in 5-run vs 3-run**: `(32768, 4096, 2048)` — gate-flake (PASS in
runs 1-2, WRONG in runs 3-5; wcf hovers 0.006-0.008 with finite around 0.98 vs
gate 0.99). Recommend keeping R39B-baseline variant for this shape.

---

## R39B baseline preservation — VERIFIED (0 regressions)

All 5 R39B PASS shapes still PASS under R40B 5-run consensus:

| Shape | R39B | R40B 5-run | TFLOPS R39B -> R40B | Notes |
|---|---|---|---|---|
| (16384, 4096, 2048) | PASS | PASS_4/5 | 3194.9 -> 3091.6 (-3.2%, still WIN) | 1-run flake |
| (16384, 4096, 3072) | PASS | PASS_5/5 | 3525.7 -> 3524.6 (-0.0%) | stable, WIN |
| (32768, 4096, 3072) | PASS | PASS_4/5 | 3681.3 -> 3671.0 (-0.3%) | now LOSE_C 101.1% |
| (4096, 32768, 4096) | PASS | PASS_4/5 | 3748.9 -> 3772.2 (+0.6%) | stable, LOSE_C 90.5% |
| (128256, 32768, 4096) | PASS | PASS_5/5 | 3931.8 -> 3900.3 (-0.8%) | stable, LOSE_C 86.0% |

Avg perf delta on shared-PASS shapes: **-0.74%** (matches verdict's -1.3% claim within noise).

---

## Stable 5-run PASSes (22 shapes — recommended for integration)

Criteria: wcf_std across 5 runs < 1.0% AND wcf_max < 2.0%.

| Shape | consensus | wcf_mean | wcf_std | wcf_max | TFLOPS/comp |
|---|---|---|---|---|---|
| (4096, 4096, 8192) | PASS_5/5 | 0.00966 | 0.00464 | 0.0158 | 95.9% |
| (4096, 4096, 16384) | PASS_5/5 | 0.01092 | 0.00444 | 0.0162 | 91.3% |
| (4096, 14336, 8192) | PASS_5/5 | 0.00359 | 0.00233 | 0.0077 | 91.0% |
| (4096, 14336, 16384) | PASS_5/5 | 0.00647 | 0.00163 | 0.0082 | 83.8% |
| (4096, 32768, 4096) | PASS_4/5 | 0.00721 | 0.00104 | 0.0085 | 90.5% |
| (6144, 4096, 8192) | PASS_5/5 | 0.00811 | 0.00074 | 0.0089 | 89.2% |
| (6144, 4096, 16384) | PASS_5/5 | 0.01035 | 0.00486 | 0.0150 | 82.8% |
| (6144, 32768, 4096) | PASS_3/5 | 0.00823 | 0.00302 | 0.0131 | 93.9% |
| (16384, 4096, 2048) | PASS_4/5 | 0.00052 | 0.00001 | 0.0005 | 103.2% WIN |
| (16384, 4096, 3072) | PASS_5/5 | 0.01223 | 0.00121 | 0.0142 | 100.9% WIN |
| (16384, 4096, 4096) | PASS_5/5 | 0.00743 | 0.00121 | 0.0093 | 100.0% WIN |
| (16384, 4096, 6144) | PASS_5/5 | 0.00773 | 0.00289 | 0.0121 | 97.7% |
| (16384, 4096, 7168) | PASS_4/5 | 0.00699 | 0.00220 | 0.0109 | 93.9% |
| (16384, 6144, 2048) | PASS_5/5 | 0.00057 | 0.00002 | 0.0006 | 105.2% WIN |
| (16384, 14336, 2048) | PASS_4/5 | 0.00106 | 0.00024 | 0.0014 | 99.5% |
| (16384, 28672, 4096) | PASS_4/5 | 0.00771 | 0.00448 | 0.0145 | 91.4% |
| (28672, 32768, 4096) | PASS_3/5 | 0.00452 | 0.00116 | 0.0055 | 87.3% |
| (32768, 4096, 3072) | PASS_4/5 | 0.00979 | 0.00439 | 0.0174 | 101.1% WIN |
| (32768, 4096, 7168) | PASS_5/5 | 0.01047 | 0.00513 | 0.0159 | 88.4% |
| (32768, 6144, 2048) | PASS_5/5 | 0.00856 | 0.00343 | 0.0127 | 102.5% WIN |
| (32768, 14336, 2048) | PASS_3/5 | 0.00476 | 0.00091 | 0.0056 | 102.0% WIN |
| (128256, 32768, 4096) | PASS_5/5 | 0.00897 | 0.00705 | 0.0166 | 86.0% |

**WIN count among stable PASSes: 7** (all 5 ratios above 100%).
**Tflops/comp ratio range**: 82.8% (6144x4096x16384) to 105.2% (16384x6144x2048).

---

## Flake-risk 5-run PASSes (2 shapes — recommend caution / per-shape gate)

These pass overall but have wcf_max >= 2% in at least one of 5 runs:

| Shape | consensus | wcf_mean | wcf_std | wcf_max | TFLOPS/comp | Notes |
|---|---|---|---|---|---|---|
| (16384, 6144, 4096) | PASS_4/5 | 0.01644 | 0.00596 | 0.0248 | 99.1% | one run wcf>2% gate |
| (32768, 4096, 14336) | PASS_3/5 | 0.01852 | 0.00478 | 0.0264 | 84.8% | borderline (3/5 PASS, 2 WRONG) |

**Recommendation**: include in R40B integration but mark for re-verification on
each leaderboard rebuild; consider keeping R37/R39B variant as fallback.

---

## TFLOPS vs aiter (`competitor_tflops`) — perf-gap survey

- **<80% gap (serious)**: **0 shapes**.
- **80-90% gap (needs follow-up)**: **7 shapes**:
  - (4096, 14336, 16384) 83.8%
  - (6144, 4096, 8192) 89.2%
  - (6144, 4096, 16384) 82.8%
  - (28672, 32768, 4096) 87.3%
  - (32768, 4096, 7168) 88.4%
  - (32768, 4096, 14336) 84.8%
  - (128256, 32768, 4096) 86.0%

These are correctness-first wins; perf gap follow-up is a future round task.

---

## Cross-cluster: shapes R40A or R40C could rescue (NOT in R40B PASS set)

Of the 18 shapes still BROKEN under R40B 5-run, R40A and R40C 3-run consensus
results show:

### R40A (R40A_PF_FENCE) integration candidates — 1 shape

| Shape | R40A status | R40A tflops | R40A vs comp | R40B status | Recommend |
|---|---|---|---|---|---|
| (4096, 32768, 128256) | OK (LOSE_CORRECT) | 4140.5 | 71.6% | WRONG_OUTPUT | **HYBRID per-shape R40A_PF_FENCE** |

This is the K=128256 extreme shape; R40B fails (cluster B-near-gate at this K),
R40A's pre-step12 fence happens to fix it.

### R40C (R40C_LDS_DRAIN) integration candidates — 1 shape

| Shape | R40C status | R40C tflops | R40C vs comp | R40B status | Recommend |
|---|---|---|---|---|---|
| (16384, 4096, 14336) | OK (LOSE_CORRECT) | 4382.9 | 85.2% | WRONG_OUTPUT | **HYBRID per-shape R40C_LDS_DRAIN** |

This is a K=14336 mid-K shape; R40B fails, R40C's lgkmcnt drain fixes it.

### Combined hybrid potential

If both per-shape R40A and R40C overrides are added:
- R40B base: 24 stable + 2 flake-risk = 24 PASS (5-run consensus)
- + R40A per-shape: +1 (4096, 32768, 128256)
- + R40C per-shape: +1 (16384, 4096, 14336)
- **Projected: 26/42 verified-correct**

**Caveat**: both R40A and R40C have one regression each in their independent
benches (R40A regressed (128256, 32768, 4096); R40C regressed the same shape).
Per-shape gating avoids that regression by only enabling on the named target.

### Still BROKEN under all R40 variants (16 shapes — open work)

These are not rescued by R40A, R40B, or R40C:
- Cluster C-catastrophic (5 shapes, K=32768): 4096x4096x32768, 4096x6144x32768,
  4096x28672x32768, 4096x128256x32768, 14336x4096x32768
- Cluster B-near-gate (~9 shapes): 16384x28672x2048, 32768x28672x2048,
  4096x32768x6144, 4096x32768x14336, 14336x32768x4096, 16384x14336x4096,
  28672x4096x8192, 28672x4096x16384, 32768x4096x2048 (newly lost)
- CRASH (2 shapes): 16384x4096x28672, 4096x32768x28672

These need a different mechanism (likely R35 hypothesis 3 — MFMA register-file
race — which R40C tested but did not fix at root).

---

## GO / NO-GO recommendation

# **GO** — commit R40B as new BEST_VARIANTS baseline.

Rationale:
1. **0 regressions** vs R39B 5/42 baseline (all 5 R39B PASSes preserved).
2. **+19 net verified-correct** under 5-run consensus (5 -> 24).
3. **Avg perf cost on shared-PASS shapes is -0.74%** (well below the 5% gate).
4. No shape <80% comp; 7 in 80-90% range (correctness-first, perf follow-up later).
5. Build artifacts already on disk (33 .so files in `build_R40B/`); no rebuild needed.

### Conditions / caveats

- **Drop (32768, 4096, 2048) from R40B**: lost in 5-run gate (PASS in runs 1-2,
  WRONG in runs 3-5). Use R39B-baseline variant if any. Otherwise leave BROKEN.
- **Mark (16384, 6144, 4096) and (32768, 4096, 14336) as flake-risk**: include in
  R40B leaderboard but flag for re-verification each rebuild.
- **Stage R40A per-shape override** for (4096, 32768, 128256) to push to 25/42.
- **Stage R40C per-shape override** for (16384, 4096, 14336) to push to 26/42.
- **Do NOT default-on R40A or R40C globally**: both have a regression on
  (128256, 32768, 4096) when applied broadly.

---

## Files

- `R40B_5RUN_CONSENSUS.json` — full 5-run results (this re-bench)
- `R40B_5RUN_CONSENSUS.log` — bench log (5 x 42 shapes, ~7 min total on 8 GPUs)
- `R40_OPT_B_BENCH.json` — original 3-run consensus (verdict reference)
- `bench_all42_results_R39_optB.json` — R39B baseline (5/42)
- `/shared_nfs/kyle/test/HipKittens/.claude/worktrees/agent-a9249715/R40_OPT_A_BENCH.json`
- `/shared_nfs/kyle/test/HipKittens/.claude/worktrees/agent-afccfa0b/analysis/fp8_gemm/mi350x/R40_OPT_C_BENCH.json`
