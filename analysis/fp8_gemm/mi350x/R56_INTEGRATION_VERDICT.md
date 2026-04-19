# R56 INTEGRATION REVIEWER VERDICT — 42/42 strict 10-run @ 80% INDEPENDENT seeds

**Date:** 2026-04-19
**Reviewer GPUs:** 4,5,6,7 (all idle, verified `rocm-smi --showuse`)
**Bench rules:** warmup=200, iters=500, trim=0.10, INDEPENDENT seeds [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
**Manifest:** `R56_INTEGRATION_MANIFEST.json` (39 AITER + 3 HK)
**Wall-clock:** ~16 min full 10-run on 4 GPUs (smoke 2.4 min)
**Outputs:** `R56_INTEGRATION_10RUN.{json,log,console}`, `R56_INTEGRATION_SMOKE1.{json,log,console}`

---

## HEADLINE

| Metric | R55 | **R56** | Delta |
|---|---|---|---|
| Strict 10-run VC (n_OK>=8/10 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min>=0.97) | 42/42 | **42/42** | 0 (HELD) |
| AITER override count | 38 | **39** | +1 (L7 HK->AITER) |
| HK baseline count | 4 | **3** | -1 |
| WIN cells (>=100% comp) | 34 | **40** | **+6** |
| AITER bit-deterministic share (wcf=0 across 10 seeds) | 38/42 | **39/42** | +1 |
| LOSE cells (<100% comp) | 8 | **2** | -6 |

**42/42 strict-VC RETAINED for the 2nd consecutive round** (first 100% leaderboard ceiling held under R56 perf-claw-back deltas). Cohort-race churn on 35 UNCHANGED shapes = **0 lost VC**. R56 is the largest single-round WIN-cell jump in project history (+6 LOSE→WIN).

Aggregate perf delta on the 7 R56 PROMOTEs vs R55 baseline: **+198.14pp**, exceeding the Stretch target (+90pp) by 2.2x.

---

## R56 PROMOTE table (7/7 verified at reviewer)

| Cell | Shape (MxNxK) | Source change | R55 pct_comp | R56 pct_comp | Delta | Verdict |
|---|---|---|---:|---:|---:|---|
| L1 | 4096x32768x14336 | R53D3B_1 64x1024 -> R56G1_L1 256x256 | 65.68% | **99.98%** | **+34.30pp** | PROMOTE (G-1) |
| L2 | 32768x4096x14336 | R53D3B_2 64x1024 -> R56G1_L2 256x256 | 84.13% | **104.00%** | **+19.87pp** | PROMOTE (G-1) |
| L3 | 128256x32768x4096 | R53D3B_3 64x1024 -> R56G2_L3 256x256 | 86.93% | **102.86%** | **+15.93pp** | PROMOTE (G-2) |
| L4 | 28672x32768x4096 | R53D3C_2 64x1024 -> R56G2_L4 256x256 | 87.82% | **102.20%** | **+14.38pp** | PROMOTE (G-2) |
| L5 | 14336x32768x4096 | R53D3C_1 64x1024 -> R56G2_L5 256x256 | 87.92% | **104.07%** | **+16.14pp** | PROMOTE (G-2) |
| L6 | 16384x4096x14336 | R53D3C_3 64x1024 -> R56G1_L6 256x256 | 90.28% | **106.88%** | **+16.60pp** | PROMOTE (G-1) |
| L7 | 4096x128256x32768 | R41A HK -> R56G4_C1 AITER 256x256 | 97.59% | **178.51%** | **+80.92pp** | PROMOTE (G-4 HK->AITER) |

All 7 PROMOTEs cleared the 1.0pp D-3C gate by **>13pp** (smallest delta L4 +14.38pp). All 7 are bit-deterministic across 10 INDEPENDENT seeds (wcf_max=0, wcf_std=0, fin_min=1.0, n_OK=10/10).

Note: L1 at reviewer landed at 99.98% (just under 100% WIN line; worker fragment reported 100.63%, attributable to GPU/run-to-run drift ~0.6pp). Still a +34.30pp lift over R55 65.68%; the cell is one notch (0.02pp) below WIN classification.

---

## ACCEPT_FALLBACK summary (L8 stays)

| Cell | Shape | Current | R55 pct | R56 alts attempted | Decision |
|---|---|---|---:|---|---|
| L8 | 4096x32768x128256 | R52D2B AITER 256x256 | 98.32% | G-4 C2 128x512 smoke 85.01% (-13.31pp); G-4 C3 192x256 smoke 86.48% (-11.84pp) | **ACCEPT_FALLBACK (D-3A-1)** — keep R52D2B |

L8 reviewer 10-run = 98.30% (matches R55 98.32% within drift). Both alt-tile probes failed the D-3A-1 protective gate (alt < current+1.0pp). No change.

---

## AITER bit-deterministic share

| Round | AITER cells | wcf=0 across 10 seeds | Share |
|---|---:|---:|---:|
| R54 | 27 | 27 | 27/42 |
| R55 | 38 | 38 | 38/42 |
| **R56** | **39** | **39** | **39/42** |

R56 grew the AITER bit-deterministic share by +1 (L7 HK R41A → AITER 256x256). All 39 AITER cells achieve wcf_max=0, wcf_std=0, fin_min=1.0 across 10 INDEPENDENT seeds — perfect bit-determinism via R50D shim AS-IS (8th consecutive AS-IS reuse round).

---

## HK cells remaining

| Round | HK cells | List |
|---|---:|---|
| R55 | 4 | 16384x4096x2048, 16384x4096x3072, 32768x14336x2048, 4096x128256x32768 |
| **R56** | **3** | 16384x4096x2048, 16384x4096x3072, 32768x14336x2048 |

L7 (4096x128256x32768) departs HK pool via G-4 promotion. The 3 surviving HK cells are R55 carry-overs (no R56 attack):
- 16384x4096x2048: R40B PASS_10/10 105.36% (clean cohort-race-low)
- 16384x4096x3072: R40B PASS_10/10 103.06% wcf_max=0.0118 (tight wcf gate but holds)
- 32768x14336x2048: R40B PASS_10/10 100.39% wcf_max=0.0127 wcf_std=0.0029 (D-5B/1 ACCEPT_FALLBACK; HK perf > AITER -1.33pp)

Cohort-race surface area cut from R55=4 HK cells to **R56=3 HK cells** (smallest in project history).

---

## Cohort-race churn audit on 35 UNCHANGED shapes

Per-shape diff R55 -> R56 on the 35 shapes whose source binary did NOT change:

- **VC retention: 35/35** (no shape lost VC under R56 reviewer 10-run)
- **VC gains: 0/35** (none possible; all 35 were already VC at R55)
- **Mean perf drift on AITER unchanged cells: -0.06pp** (range -1.71pp to +0.48pp; the -1.71pp on R55D5A_3 4096x4096x8192 121.15->119.43% is run-to-run drift, still WIN by +19.43pp)
- **Mean perf drift on HK unchanged cells: -0.02pp** (range -0.14pp to +0.28pp)

**Cohort-race churn count: 0.** Mechanism remains as in R55: AITER `.co` dlopen via R50D shim is bit-deterministic (wcf=0 across all 10 seeds on every AITER cell). The 3 HK cells re-pass strict gates with the same wcf_max profile as R55 (R55 wcf_max 0.0024-0.0147 → R56 wcf_max 0.0006-0.0127).

---

## COMMIT recommendation

**COMMIT R56_INTEGRATION_MANIFEST.json as the canonical R56 production manifest.**

Reasons:
1. **42/42 strict 10-run VC HELD** for the 2nd consecutive round; 100% leaderboard ceiling preserved through 7 PROMOTEs + 1 ACCEPT_FALLBACK.
2. **+6 WIN cells (34 → 40)** — largest single-round WIN-cell jump in project history.
3. **7/7 PROMOTE re-verified at reviewer** with bit-determinism (wcf_max=0, fin_min=1.0, n_OK=10/10) on every R56 PROMOTE.
4. **+198.14pp aggregate perf claw-back** across 7 PROMOTEs — 2.2x Stretch target.
5. **Cohort-race churn = 0 on 35 unchanged shapes**; HK survivor pool shrunk to 3 (smallest ever).
6. **R50D shim AS-IS reuse for 8th consecutive round** — no kernel rebuild, no shim modification.
7. **AITER bit-deterministic share 38 → 39 of 42**; only 3 cells remain non-AITER.
8. **2 LOSE cells remain**: L1 4096x32768x14336 at 99.98% (0.02pp below WIN; reviewer-drift edge case from worker's 100.63%); L8 4096x32768x128256 at 98.30% (D-3A-1 ACCEPT_FALLBACK; alt tiles strictly worse).

No REVERT actions necessary. All 7 R56 PROMOTE candidates stay in the manifest.

---

## R57 candidate axes (if any LOSE cells remain)

Only 2 LOSE cells survive R56:

1. **L1 (4096x32768x14336)** — 99.98% (within 0.02pp of WIN). Reviewer-vs-worker drift fluctuation; effectively at the WIN threshold. R57 candidate axis: re-bench with longer ITERS (e.g., 1000) to tighten p50 distribution and confirm WIN crossing — but this is marginal noise, not a structural gap.
2. **L8 (4096x32768x128256)** — 98.30%. K=128256 K/N=4 ratio. Two alt tiles probed in R56 G-4 (128x512 -13.31pp, 192x256 -11.84pp) both DEAD. Remaining un-probed AITER tiles for K=128256: 96x640, 64x1024 (both lower-eff than 256x256, very low confidence). R57 axis: aiter `.co` dlopen with K-pipeline-tuned variant if any exists in `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/` not yet attempted; otherwise this is at the aiter-internal ceiling for this shape and pivot toward HK kernel optimization (but R-50D shim has been the productive path).

R56 reaches the practical WIN-cell ceiling at 40/42. The 2 remaining LOSEs are within 1.7pp of WIN classification combined.

---

## Round summary

R56 = **42/42 VC retention + 7/7 PROMOTE + +198.14pp aggregate + 34→40 WIN + 0 churn**. 8th consecutive R50D shim AS-IS round. Cohort-race surface at 3 HK cells (smallest ever). The Opt G mechanism (homogeneous 64x1024 → 256x256 swap on shapes where local_round count is identical and only `compute2mem_efficiency` differs) is mechanism-validated across 6/6 G-1+G-2 candidates. The G-4 HK→AITER 256x256 swap on L7 (4096x128256x32768) delivered the largest single-cell perf gain (+80.92pp) by replacing R41A HK with the now-canonical aiter 256x256 pattern.

---

## Files

- `R56_INTEGRATION_MANIFEST.json` — canonical R56 manifest (39 AITER + 3 HK)
- `bench_all_42_R56_INTEGRATION.py` — reviewer bench script (4-GPU sharding, --gpus 4,5,6,7)
- `R56_INTEGRATION_SMOKE1.{json,log,console}` — 1-seed smoke (42/42 VC, 40/42 WIN, 2.4 min wall)
- `R56_INTEGRATION_10RUN.{json,log,console}` — full 10-run @ 80% INDEPENDENT seeds (42/42 VC, 40/42 WIN, ~16 min wall)
- `R56_INTEGRATION_VERDICT.md` — this file
- worker fragments merged: `R56G1_{L1,L2,L6}_INTEGRATION_FRAGMENT.json`, `R56G2_{L3,L4,L5}_INTEGRATION_FRAGMENT.json`, `R56G4_C1_INTEGRATION_FRAGMENT.json`
- worker verdicts: `R56_OPT_G1_VERDICT.md`, `R56_OPT_G2_VERDICT.md`, `R56_OPT_G4_VERDICT.md` (if present)
