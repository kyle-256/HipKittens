# R58 Decider Plan — Gate Tightening Pilot (Opt N) + L8 HK Last-Mile (Opt O) + Optional Alt-Tile Closure (Opt P)

**Date:** 2026-04-19
**R57 baseline:** **42/42 strict 10-run VC HELD** (3rd consecutive 100% leaderboard round; FIRST 3-IN-A-ROW IN PROJECT HISTORY; committed `f23fd625`); **41/42 WIN** (40→41 via Opt J ITERS=1000 on L1); 39 AITER + 3 HK; 9th consecutive R50D shim AS-IS reuse round; 0 cohort-race churn on 41 unchanged-binary cells.
**Goal:** Generate methodology information (gate tightening) regardless of perf outcome; attempt the only structurally available WIN path (L8 HK rewrite); document closure of the remaining alt-tile axis if budget allows.

---

## 1. R57 baseline reminder

- **42/42 strict 10-run VC** (gate: n_OK≥8/10 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min≥0.97)
- **39 AITER overrides** (R50D shim AS-IS, 9th consecutive AS-IS reuse round) + **3 HK baselines**
- **41/42 WIN** (≥100% pct_comp); **only 1 LOSE cell remains**:
  - **L8 `(4096,32768,128256)`** — reviewer p50 **97.75%** (R52D2B AITER 256×256); aiter alt-tile axis FULLY CLOSED (128×512 -13.31pp, 192×256 -11.84pp, 224×256 -10.69pp all DEAD); at aiter-internal ceiling for K=128256 K/N=4 ratio
- **AITER bit-deterministic share 39/42 HELD** (wcf=0 across 10 INDEPENDENT seeds on every AITER cell)
- **3 surviving HK cells** (cohort-race surface, smallest in project history sustained):
  - `16384x4096x2048` — R40B 109.57% (R57 ITERS=1000 reviewer; wcf_max=0.0006, fin_min=0.9985)
  - `16384x4096x3072` — R40B 103.25% (wcf_max=0.0134, fin_min=0.9955)
  - `32768x14336x2048` — R40B 100.53% (wcf_max=0.0113, fin_min=0.9847; **near-gate**)
- Strict-VC ceiling REACHED and HELD for 3 consecutive rounds under current gate
- ITERS=1000 was a one-time R57 Opt J protocol bump for L1 noise-edge tightening; **R58 reverts to ITERS=500 default** unless an Opt N tightening applies a different protocol.

---

## 2. R58 axis selection rationale

### Selected axes (3 cohorts)

| Axis | Status | Mechanism conf | Expected value | Rationale |
|---|---|---|---|---|
| **Opt N** — gate tightening pilot (n_OK→9/10, wcf_max→0.01, fin_min→0.99) | **RUN** | METHODOLOGY (cert) | +0 NET WIN; +N reclassifications (informational) | Strict VC ceiling reached for 3 consecutive rounds; tightened gate enumerates the true cohort-race surface. **Highest methodology value remaining axis**. Cost: re-analysis of R57 10-run JSON, no new bench needed. |
| **Opt O** — L8 HK rewrite for K=128256 K/N=4 | **RUN** | LOW (high-risk high-reward) | +0 to +1 NET WIN | Aiter alt-tile axis fully closed; only path to 42/42 WIN. HK kernel `tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_btw_all_R40B_safe.so` exists in `build_R40B/`; never benched against L8 since R52D2B AITER swap. HK 256×256 path produces a different MFMA schedule than aiter 256×256. PROMOTE requires HK ≥ 100.0% (i.e., +2.25pp over current AITER 97.75%). |
| **Opt P** — non-192×256 alt-tile probe on 3 HK cells (low-eff completion) | **RUN (optional, completion)** | VERY LOW | +0 NET WIN; +N AITER share if any clears HK gate | Closes remaining alt-tile axes for HK kept-cell pool (128×256 eff=85.3, 96×640 eff=83.5, 64×1024 eff=60.2). Run only if GPU budget allows after Opt N + Opt O. SMOKE-only with hard cap 5 GPU-min/cell. |

### Deferred / skipped axes

| Axis | Status | Reason |
|---|---|---|
| **Opt Q** — revert ITERS=500 default | **NO COHORT NEEDED** | Reviewer integration plan (§5) defaults to ITERS=500. R57 H-1 ITERS=1000 was single-purpose; no axis attached. |
| Re-attempt L8 aiter alt-tiles (96×640, 64×1024, 128×384, 160×384) | **CLOSED** | Lower-eff than 224×256 which is already DEAD by 10.69pp. Mechanism axis fully closed. |
| Re-attempt 192×256 on 3 HK cells | **CLOSED** | R57 H-2 confirmed -8 to -14pp on all 3. |
| L1 4096x32768x14336 attacks | **CLOSED** | R57 J-1 ITERS=1000 already crossed WIN line (100.04% reviewer). |

---

## 3. Per-cohort candidate list with grid math

All AITER candidates use **R50D shim AS-IS** (`build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`); NO kernel rebuild. KernelArgs are `(M, N, K)`; bdx=256; gdz=1.

### Cohort I-1 (Opt N) — Gate Tightening Pilot

**Pure analysis cohort, no GPU bench, ~10 min wall-clock.** Re-analyze the existing R57 10-run JSON (`R57_INTEGRATION_10RUN.json`) under a tightened gate; emit an Opt-N reclassification report.

| ID | Action | Input file | Tightened gate (proposed) |
|---|---|---|---|
| **I-1/N-1** | Re-classify all 42 cells under tightened gate | `R57_INTEGRATION_10RUN.json` | `n_OK≥9/10 AND wcf_max<0.01 AND wcf_std<0.005 AND fin_min≥0.99 AND pct_comp≥100.0%` |
| **I-1/N-2** | Produce per-cell delta table: which cells fall out under each tightened sub-gate (n_OK, wcf_max, fin_min, perf) | same | breakdown by sub-gate |
| **I-1/N-3** | Emit `R58_OPT_N_GATE_TIGHTENING_REPORT.md` listing PASS/FAIL counts under (a) current R45+ gate, (b) Opt N tightened gate; identify the marginal cells (those that PASS R45+ but FAIL Opt N) and rank by which sub-gate kicks them out. | output | 1-page table |

**Worker output**: `R58_OPT_N_REPORT.{md,json}`. NO manifest changes (Opt N is a methodology pilot; R59+ may decide whether to adopt).

### Cohort I-2 (Opt O) — L8 HK rewrite for K=128256

**1 candidate, 1-2 GPUs, ~15 min wall (SMOKE) + conditional 30 min wall (10-run).**

| ID | Shape (current AITER pct_comp) | HK candidate | Gate to beat |
|---|---|---|---|
| **I-2/O-1** | `4096x32768x128256` (L8, AITER 256×256 R52D2B 97.75%) | `build_R40B/tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_btw_all_R40B_safe.cpython-310-x86_64-linux-gnu.so` (HK 256×256 lgk2 path; never benched on L8 since R52D2B AITER swap) | SMOKE: pct_comp ≥ **98.75%** (current + 1.0pp safety) to escalate; 10-run gate: pct_comp > **98.75%** AND VC strict |

**Mechanism hypothesis**: HK 256×256 lgk2 (latency-group K-prefetch=2) path produces a different MFMA↔ds_read schedule than aiter 256×256 .co. R56 G-4 C2/C3 falsified aiter alt-tile axis (128×512, 192×256 DEAD); HK 256×256 with the lgk2 K-pipeline tuning hasn't been compared head-to-head with R52D2B since the D-2 swap. The shape `n32768_k128256` matches L8 dimensions exactly (M=4096 implicit; HK uses gridx variable).

**Grid (HK)**: per HK wrapper `wrap_n32768_k128256_*_R40B.cpp` defaults; worker should source HK grid from `build_R40B/wrap_*` (NO hand-rolled grid).

**Fallback / completion exploration if SMOKE on R40B safe variant fails**:
- **I-2/O-1b**: token try `build_R37/tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all_R37.cpython-310-x86_64-linux-gnu.so` (R37 memc-flag variant; pre-R39 uniform-scale gate). SMOKE only; 5 GPU-min hard cap.

### Cohort I-3 (Opt P, OPTIONAL) — alt-tile completion probes on 3 HK cells

**Run ONLY if Opt N + Opt O finish under budget AND a GPU pair is idle.** Pure axis closure; very low expected value.

| ID | Shape (HK pct_comp) | Alt tile | `.co` file | SMOKE gate |
|---|---|---|---|---|
| **I-3/P-1** | `16384x4096x2048` (HK 109.57%) | 128×256 (eff=85.3) | `..._BpreShuffle_128x256.co` | SMOKE > 110.57% to escalate (HK + 1.0pp); else SMOKE-DEAD ACCEPT_FALLBACK |
| **I-3/P-2** | `16384x4096x3072` (HK 103.25%) | 128×256 (eff=85.3) | `..._BpreShuffle_128x256.co` | SMOKE > 104.25% to escalate; else ACCEPT_FALLBACK |
| **I-3/P-3** | `32768x14336x2048` (HK 100.53%, near-gate) | 128×256 (eff=85.3) | `..._BpreShuffle_128x256.co` | SMOKE > 101.53% to escalate; else ACCEPT_FALLBACK |

**Grid math**: `gdx = ceil(M/128)`, `gdy = ceil(N/256)`, `gdz = 1`, `bdx = 256`. Kernel name: `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E`.

**Hard cap**: 5 GPU-min per cell on SMOKE; if all 3 SMOKE-DEAD (expected outcome given lower-eff than 192×256 already DEAD), 128×256 alt-tile axis CLOSED for HK kept-cell pool. **Skip 96×640 and 64×1024 entirely** (eff << 192×256 which already failed).

---

## 4. Worker cohort assignments

| Cohort | Worker | GPU(s) | Candidates | Mechanism conf | Wall budget |
|---|---|---|---|---|---|
| **I-1** (Opt N) | A | NONE (analysis only) | I-1/N-1, N-2, N-3 | METHODOLOGY | 10 min |
| **I-2** (Opt O) | B | 0, 1 | I-2/O-1 (+ I-2/O-1b fallback) | LOW (high-risk) | 45 min (15 min SMOKE + cond. 30 min 10-run) |
| **I-3** (Opt P, OPTIONAL) | C | 2, 3 | I-3/P-1, P-2, P-3 | VERY LOW | 15 min total (5 min SMOKE × 3 cells, parallel pairs) |

**Total: 4-6 candidates across 3 cohorts on 4 GPUs (workers).** Reviewer integration uses GPUs 4,5,6,7 (idle pool reserved). Verify all GPUs idle via `rocm-smi --showuse` pre-launch (per benchmark-rules.md).

R58 is intentionally a small round — practical WIN ceiling at 41/42 with only L8 LOSE; methodology Opt N is the highest-information axis remaining.

---

## 5. PROMOTE gate spec per axis

All bench candidates use **bench rules MANDATORY**: warmup=200, **iters=500 (R45+ default; revert from R57 ITERS=1000)**, trim=0.10, idle GPU only. **NO kernel rebuild**; if any worker proposes shim modification, REJECT.

### Opt N (I-1) — methodology pilot

1. **No PROMOTE possible.** Opt N produces an information report only.
2. Acceptance: `R58_OPT_N_REPORT.md` exists, contains:
   - Counts under current R45+ gate (must be 42/42)
   - Counts under tightened gate (must be ≤42/42)
   - Per-cell sub-gate breakdown (n_OK, wcf_max, wcf_std, fin_min, pct_comp)
   - Ranked list of marginal cells most at risk under tightening
   - Recommendation: adopt for R59 (Y/N) with rationale

### Opt O (I-2/O-1) — L8 HK candidate PROMOTE gate

1. **SMOKE-first**: 5-seed smoke (warmup=200, iters=500, trim=0.10, seeds [101,202,303,404,505]) before 10-run.
2. **SMOKE perf gate**: pct_comp ≥ **98.75%** (current 97.75% + 1.0pp safety) to escalate to 10-run.
3. **STOP rule**: if SMOKE pct_comp < 96.75% (-1.0pp from current AITER), **STOP immediately**, ACCEPT_FALLBACK; do NOT run 10-run.
4. **10-run gates** (only if SMOKE escalates):
   - **VC strict 10-run gate**: n_OK≥8/10 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min≥0.97 (R45+ gate; not Opt N tightened)
   - **Perf gate (D-3A-1 strict)**: reviewer 10-run p50 pct_comp > **98.75%** (current 97.75% + 1.0pp)
   - **WIN gate (additional)**: pct_comp ≥ **100.0%** for "+1 WIN cell" claim
5. **Outcome**:
   - **PROMOTE** if all gates pass AND pct_comp ≥ 100.0% → L8 → WIN; AITER share 39 → 38 (HK insertion); HK cells 3 → 4. NET: +1 WIN, -1 AITER share, +1 HK.
   - **PROMOTE_PARTIAL** if all gates pass AND pct_comp ∈ [98.75%, 100.0%) → L8 stays LOSE but ≥1pp closer; manifest swap recorded; AITER share 39 → 38 reluctantly accepted ONLY if HK perf strictly beats AITER by ≥+1.0pp. Default: **prefer ACCEPT_FALLBACK** at this band (don't trade bit-determinism for perf-without-WIN).
   - **ACCEPT_FALLBACK (D-3A-1)** if perf gate fails or SMOKE STOP triggers → keep R52D2B AITER baseline; document delta. AITER share unchanged.
   - **DEAD** if VC gate fails (wcf_max ≥ 0.02 or n_OK<8/10) → close HK 256×256 axis on L8.

### Opt P (I-3) — alt-tile completion PROMOTE gate

1. **SMOKE-first**: 5-seed smoke per cell (hard cap 5 GPU-min).
2. **SMOKE perf gate**: pct_comp > current_HK_pct_comp + 1.0pp to escalate (per cell: >110.57%, >104.25%, >101.53%).
3. **STOP rule**: if SMOKE pct_comp < HK_pct - 5.0pp on any cell, ACCEPT_FALLBACK and document axis closure.
4. **10-run gates** (only if SMOKE escalates): same VC + perf gates as Opt L from R57.
5. **Expected outcome**: 0/3 PROMOTE / 3 ACCEPT_FALLBACK closes 128×256 axis; AITER share unchanged.

---

## 6. D-3A-1 protection rules (carried over from R57)

- **No swap** of an existing manifest entry unless candidate strictly wins ≥+1.0pp over current p50 pct_comp at reviewer 10-run.
- **ACCEPT_FALLBACK** preserves the current manifest entry verbatim (binary, grid, kernel name, source label).
- **SMOKE-DEAD** triggers ACCEPT_FALLBACK without 10-run cost.
- **VC LOSS** at 10-run on a swap candidate → REVERT immediately, do NOT merge fragment, treat as DEAD.
- **AITER → HK swap special clause**: PROMOTE only if HK ≥ 100.0% (WIN) AND HK strictly beats AITER by ≥+1.0pp. Trading bit-determinism for sub-WIN perf is NOT a valid PROMOTE.

---

## 7. Floor / Mode / Stretch targets

R58 is a small ceiling-tightening + closure round; targets are framed as **PROMOTE counts** and **NEW WIN cells**.

| Tier | PROMOTEs | NEW WIN cells | HK cells | AITER share | Mechanism prereq |
|---|---:|---:|---:|---:|---|
| **Floor** | 0 PROMOTE + Opt N report delivered | 41 (HELD) | 3 (HELD) | 39 (HELD) | Opt N report exists; SMOKE-DEAD on Opt O acceptable |
| **Mode** | 0-1 PROMOTE + Opt N report + ≥2 axis closures documented (Opt O DEAD or PROMOTE; Opt P DEAD on ≥2 cells) | 41-42 | 3-4 | 38-39 | Methodology + axis-closure round |
| **Stretch** | 1 PROMOTE (Opt O HK PROMOTE on L8) | 42 (FULL LEADERBOARD WIN) | 4 | 38 | HK 256×256 lgk2 beats AITER 256×256 R52D2B by ≥2.25pp on L8 (LOW probability; aiter is at internal ceiling, HK has not been competitive on K=128256 since R37) |

**Realistic bullseye: Floor.** Opt N delivers methodology information regardless of bench outcome; Opt O is the only path to 42/42 WIN but high-risk; Opt P is closure-only.

**Aggregate perf claw-back vs R57:**
- Floor: +0pp (no manifest changes)
- Mode: +0 to +2pp (Opt O fallback or partial)
- Stretch: +2.25pp on L8 (97.75 → ≥100%); aggregate ~+2.25pp

---

## 8. Reviewer integration plan (post-worker)

Each worker emits per-candidate fragments named:
- `R58I1_N_REPORT.{md,json}` (Opt N analysis)
- `R58I2_O1_INTEGRATION_FRAGMENT.json` (Opt O candidate)
- `R58I3_P{1,2,3}_INTEGRATION_FRAGMENT.json` (Opt P candidates if attempted)

Required fragment schema (Opt O / Opt P):
```json
{
  "round": "R58",
  "opt": "O | P",
  "cohort": "I-2 | I-3",
  "cell_label": "O-1 | P-1 | P-2 | P-3",
  "shape": [M, N, K],
  "current_baseline": {
    "source": "R52D2B_AITER | R40B",
    "tile": [tM_cur, tN_cur],
    "pct_comp": 97.75
  },
  "candidate": {
    "kind": "HK_so | AITER_co",
    "so_path_or_co_path": "...",
    "kernel_name": "...",
    "grid": {"gdx": ..., "gdy": ..., "gdz": 1, "bdx": 256},
    "tile": [tM_cand, tN_cand]
  },
  "shim_so": "build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so",
  "shim_rebuild_required": false,
  "smoke_5seed": {
    "seeds": [101,202,303,404,505],
    "warmup": 200,
    "iters": 500,
    "trim_frac": 0.10,
    "n_OK": ...,
    "pct_comp": ...
  },
  "smoke_decision": "ESCALATE | STOP_ACCEPT_FALLBACK | STOP_DEAD",
  "ten_run": {
    "seeds": [101,202,303,404,505,606,707,808,909,1010],
    "warmup": 200,
    "iters": 500,
    "trim_frac": 0.10,
    "n_OK": ...,
    "wcf_max": ...,
    "wcf_std": ...,
    "fin_min": ...,
    "tflops_p50": ...,
    "pct_comp": ...
  },
  "perf_delta_pp": ...,
  "verdict": "PROMOTE | PROMOTE_PARTIAL | ACCEPT_FALLBACK | DEAD",
  "verdict_reason": "..."
}
```

**Worker output files (per cohort)**:
- `bench_R58I2_O1.py` (Opt O bench script, copy from `bench_R57H3_K1.py` template, swap candidate to HK .so)
- `R58_OPT_O1_{SMOKE,10RUN}.{json,log}` (per cell)
- `R58I2_O1_INTEGRATION_FRAGMENT.json`
- `R58_OPT_I2_VERDICT.md` (cohort-level summary)
- analogous for I-3.

**Reviewer steps** (post-worker, ~25 min wall):
1. Aggregate fragments → `R58_INTEGRATION_MANIFEST.json` (delta from R57 manifest; if 0 PROMOTEs, manifest is byte-identical to R57 with version bump only).
2. Run `bench_all_42_R58_INTEGRATION.py` (copy of `bench_all_42_R57_INTEGRATION.py` with `iters=500` revert) **SMOKE 1-seed** on GPUs 4,5,6,7 (~2 min).
3. Run full 10-run @ INDEPENDENT seeds [101..1010] @ **ITERS=500 default** on GPUs 4,5,6,7 (~10 min wall, 420 runs).
4. Cohort-race churn audit on all 42 cells (must be 0 lost VC if no manifest changes; ≤1 cell churn if Opt O PROMOTE).
5. Compose `R58_INTEGRATION_VERDICT.md`:
   - Headline 42/42 strict-VC retention (must HOLD from R57 — 4th consecutive 100% leaderboard target)
   - WIN cell delta (target: 41 → 41-42)
   - HK cell delta (target: 3 → 3-4)
   - AITER bit-deterministic share delta (target: 39 → 38-39)
   - Cohort-race churn audit on UNCHANGED cells (must be 0 churn)
   - Per-axis PROMOTE / ACCEPT_FALLBACK / DEAD breakdown
   - Opt N gate-tightening report linkage and adoption recommendation for R59
6. ITERS=500 vs R57's ITERS=1000: document any pct_comp drift on the L1 cell (the only cell affected by the protocol revert); if L1 falls back below 100% under ITERS=500, document but do NOT REVERT the source-label change (R57J1_L1_AITER stays).

---

## 9. Closed-axis reminder (DO NOT propose in R58)

Carry-forward from R45-R57 closed axes; in addition for R58:

- **DO NOT modify R50D shim.** 9 consecutive AS-IS reuse rounds; treat as black box.
- **DO NOT propose VC-improving kernel changes.** Strict-VC ceiling reached for 3 consecutive rounds; no new VC available.
- **DO NOT use warmup<200 or iters<500.** Bench rules. R58 reverts to ITERS=500 default.
- **DO NOT attempt L8 alt-tiles 128×512, 192×256, 224×256, 96×640, 64×1024 on AITER.** R56 G-4 + R57 H-3 fully closed aiter alt-tile axis on L8.
- **DO NOT attempt 192×256 on the 3 HK cells.** R57 H-2 closed (-8 to -14pp).
- **DO NOT attempt 256×256 AITER on the 3 HK cells.** R55 D-5B/1 found 256×256 AITER underperforms HK on N=14336.
- **DO NOT attempt L1 4096x32768x14336 attacks.** R57 J-1 ITERS=1000 already crossed WIN line.
- **DO NOT modify the 3 HK kernel .so files.** Only AITER `.co` dlopen swaps via R50D shim are in scope for HK→AITER hardening; for L8 Opt O, only existing HK .so reuse (no rebuild).
- **DO NOT attempt 96×640 or 64×1024 on the 3 HK cells.** Lower-eff than 192×256 which already failed.
- **DO NOT bench at ITERS=1000 except for cells identified as ≤0.5pp from a classification boundary** (R57 protocol-bump validity envelope).

---

## 10. STOP rules

- **Floor reached** (Opt N report delivered, 0 PROMOTEs OK): may stop early; reviewer integration becomes a re-bench-at-ITERS=500 pass with manifest version bump only.
- **SMOKE-DEAD on Opt O cell**: ACCEPT_FALLBACK; keep R52D2B AITER 256×256 manifest entry; close HK 256×256 axis on L8.
- **SMOKE-DEAD on Opt P cell**: ACCEPT_FALLBACK; keep HK R40B; close 128×256 axis on that HK cell.
- **ANY cell loses VC at 10-run on a swap candidate**: REVERT immediately, do NOT merge fragment, treat as DEAD.
- **Reviewer integration smoke fails to reach 42/42 VC**: REVERT manifest to R57 byte-for-byte, treat round as ACCEPT_FALLBACK across the board, document the regression in `R58_INTEGRATION_VERDICT.md`.

---

## 11. Confidence rationale

- **Opt N (I-1)** is methodology-certain: it produces a deterministic report from existing 10-run JSON. The interesting result is whether R45+ gate (n_OK≥8/10, wcf_max<0.02, fin_min≥0.97) is materially looser than the proposed Opt N gate (9/10, 0.01, 0.99) on the current 42/42 manifest. The 3 HK cells are the most likely to fall out (`32768x14336x2048` has fin_min=0.9847 which fails Opt N's 0.99 threshold). Use Opt N to enumerate the cohort-race surface for R59 attention.
- **Opt O (I-2/O-1)** is high-risk: aiter R52D2B 256×256 has been the L8 baseline since R52 (D-2 swap from HK to AITER). HK kernel `tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_btw_all_R40B_safe.so` exists in `build_R40B/` and was the pre-R52 baseline (lost to AITER for perf). Re-trying HK with the lgk2 K-pipeline tuning is a re-test of a closed comparison; the prior is **DEAD** but the only open path to 42/42 WIN. Worth one SMOKE for axis-closure documentation.
- **Opt P (I-3)** is very low confidence: 128×256 (eff=85.3) is lower-eff than 192×256 (eff=109.7) which is already DEAD on all 3 HK cells. Probability of any cell PROMOTE: ≤5%. Run only for axis closure; expected outcome 0/3 PROMOTE.

---

## Summary

**3 cohorts (I-1 analysis-only, I-2 L8 HK rewrite, I-3 optional alt-tile closure), 4-6 candidates total**, on 4 GPUs (workers; GPUs 0-3) + 4 GPUs (reviewer; GPUs 4-7). Floor (0 PROMOTEs + Opt N report delivered) is the realistic bullseye and produces certain methodology value. Mode (0-1 PROMOTE + axis closures) requires Opt O SMOKE escalation. Stretch (1 PROMOTE → 42/42 WIN) requires HK 256×256 lgk2 to beat aiter 256×256 R52D2B by ≥2.25pp on L8 — historically the opposite has held since R52. R58 is intentionally a small **methodology + axis-closure round**; the strict-VC 42/42 ceiling MUST hold for the 4th consecutive 100% leaderboard. R50D shim AS-IS for the 10th consecutive round (assuming Opt O HK PROMOTE doesn't land — if it does, AITER share drops 39→38 and the 10-AS-IS streak is preserved on the remaining 38 AITER cells). ITERS=500 default reverted from R57's one-time ITERS=1000 protocol bump.
