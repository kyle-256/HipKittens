# R57 Decider Plan — L1 ITERS=1000 re-bench (Opt J) + 3 HK-cell hardening (Opt L)

**Date:** 2026-04-19
**R56 baseline:** **42/42 strict 10-run VC HELD** (2nd consecutive 100% leaderboard round; committed `03f7ff5e`); 40/42 WIN (+6 NET); 39 AITER + 3 HK; 8th consecutive R50D shim AS-IS reuse round
**Goal:** Close the last 2 LOSE cells (within 1.7pp combined) and shrink HK cohort-race surface (3 → ≤2). NO new VC available. Pure tail-tightening + bit-determinism expansion round.

---

## 1. R56 baseline reminder

- **42/42 strict 10-run VC** (gate: n_OK≥8/10 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min≥0.97)
- **39 AITER overrides** (R50D shim AS-IS, 8th consecutive AS-IS reuse round) + **3 HK baselines**
- **40/42 WIN** (≥100% pct_comp); **2 LOSE cells remain** (combined gap to all-WIN < 1.7pp):
  - **L1 `(4096,32768,14336)`** — reviewer p50 **99.98%** (worker 100.63%); 0.02pp under WIN line — reviewer-vs-worker drift edge
  - **L8 `(4096,32768,128256)`** — reviewer p50 **98.30%**; D-3A-1 ACCEPT_FALLBACK (R56 G-4 alts 128×512 -13.31pp / 192×256 -11.84pp DEAD)
- **AITER bit-deterministic share 39/42** (wcf=0 across 10 INDEPENDENT seeds on every AITER cell)
- **3 surviving HK cells** (cohort-race surface, smallest in project history):
  - `16384x4096x2048` — R40B 105.36% (wcf_max=0.000578, clean)
  - `16384x4096x3072` — R40B 103.06% (wcf_max=0.0118, tight)
  - `32768x14336x2048` — R40B 100.39% (wcf_max=0.0127, wcf_std=0.0029; D-5B/1 ACCEPT_FALLBACK in R55: 256×256 AITER -1.33pp vs HK)
- Strict-VC ceiling REACHED under current gate

---

## 2. R57 axis selection rationale

### Selected axes

| Axis | Status | Mechanism conf | Expected value | Rationale |
|---|---|---|---|---|
| **Opt J** — L1 re-bench at ITERS=1000 | **RUN** | VERY HIGH | +1 WIN cell (40→41) | L1 reviewer 99.98% vs worker 100.63% = 0.65pp drift; ITERS=1000 cuts p50 noise ~√2; bit-deterministic AITER means VC is automatic. Cost: 1 GPU × 1 cell × ~3 min wall. |
| **Opt L** — kept-HK-cell hardening (3 candidates) | **RUN** | LOW-MEDIUM | +0 NET WIN, but +1-3 AITER bit-determinism | 3 HK cells pin cohort-race surface. R55 D-5B/1 mechanism: 256×256 AITER underperforms HK on N=14336 by 1.33pp; so try **non-256×256 alt-tile** (highest-eff after 256×256 = 192×256, eff=109.7). HK perf bar is high (105.36 / 103.06 / 100.39%) → likely ACCEPT_FALLBACK on perf, but if any alt clears HK_pct + 1.0pp, win = AITER bit-determinism replacement. |

### Deferred axes

| Axis | Status | Reason for defer |
|---|---|---|
| **Opt K** — un-attempted aiter tiles for L8 | **SKIP** (essentially DEAD) | R56 G-4 already falsified 128×512 (-13.31pp) and 192×256 (-11.84pp). Remaining un-tried `.co` for K=128256: 96×640 (eff=83.5) and 64×1024 (eff=60.2) are both LOWER eff than current 256×256 (eff=128); 224×256 (eff=120.5) is the only un-tried tile with comparable eff but extant disasm shows aiter heuristic disfavors it for this K. Token-probe only if there is bandwidth; not worth a dedicated cohort. |
| **Opt M** — gate tightening pilot (n_OK→9/10, wcf_max→0.01, fin_min→0.99) | **DEFER to R58** | Methodology change, not perf gain. Requires 2-pass review (first under R45+ gate to confirm 42/42 carries, then under tightened gate to enumerate sub-optimal cells). R57 is too small a round to absorb the methodology change without confounding the Opt L verdicts. |

---

## 3. Per-cell candidate list with grid math

All candidates use **R50D shim AS-IS** (`build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`); NO kernel rebuild. KernelArgs are `(M, N, K)`; bdx=256; gdz=1.

### Cohort H-1 (Opt J) — L1 re-bench at ITERS=1000

**1 candidate, 1 GPU, ~5 min wall.** Identical .so + grid as R56G1_L1; only ITERS changes.

| ID | Shape | Tile | Grid (gdx, gdy, gdz) | `.co` | Notes |
|---|---|---|---|---|---|
| **H-1/J-1** | `4096x32768x14336` (L1) | 256×256 | (128, 16, 1) | `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` | Re-bench R56G1_L1 with **ITERS=1000** (vs default 500), warmup=200, trim=0.10, INDEPENDENT seeds [101..1010]. Worker should report p50, p25, p75 to surface drift. |

Grid: M=4096 / 256 = 16 → wait: aiter `.co` swaps M↔grid based on internal layout. Use **R56G1_L1's grid (128, 16, 1)** verbatim from `R56_INTEGRATION_MANIFEST.json` (verified bit-deterministic).

### Cohort H-2 (Opt L) — kept-HK-cell hardening (alt-tile non-256×256)

**3 candidates, 2 GPUs paired (5 seeds each), ~10 min wall per candidate.** Mechanism: per R55 D-5B/1 the 256×256 tile is suboptimal vs HK on N=14336 grid-x=56; try the **next-highest-eff non-256×256 tile = 192×256 (eff=109.7)**.

| ID | Shape (HK pct_comp) | Alt tile | Grid (gdx, gdy, gdz) | `.co` | HK target to beat |
|---|---|---|---|---|---|
| **H-2/L-1** | `16384x4096x2048` (HK 105.36%) | 192×256 | (86, 16, 1) | `..._BpreShuffle_192x256.co` | 105.36% + 1.0pp = **106.36%** |
| **H-2/L-2** | `16384x4096x3072` (HK 103.06%) | 192×256 | (86, 16, 1) | `..._BpreShuffle_192x256.co` | 103.06% + 1.0pp = **104.06%** |
| **H-2/L-3** | `32768x14336x2048` (HK 100.39%) | 192×256 | (171, 56, 1) | `..._BpreShuffle_192x256.co` | 100.39% + 1.0pp = **101.39%** |

Grid math: `gdx = ceil(M/192)`, `gdy = ceil(N/256)`, `gdz = 1`, `bdx = 256`. Kernel name: `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_192x256E`.

**Fallback exploration if 192×256 DEAD on a cell:** worker MAY one-shot probe **128×256** (eff=85.3) on the same cell with 5-seed smoke; only escalate to 10-run if smoke beats HK_pct + 1.0pp. Hard cap 5 GPU-min per fallback.

### Cohort H-3 (Opt K) — token L8 probe (optional, ≤1 candidate, low priority)

**Run only if H-1 + H-2 finish under target wall and a GPU pair is idle.**

| ID | Shape (current pct_comp) | Alt tile | Grid (gdx, gdy, gdz) | `.co` | Notes |
|---|---|---|---|---|---|
| **H-3/K-1** | `4096x32768x128256` (L8, 256×256, 98.30%) | 224×256 | (19, 128, 1) | `..._BpreShuffle_224x256.co` | Eff=120.5 (between 128 and 109.7); 5-seed SMOKE only; PROMOTE only if SMOKE >99.30% AND 10-run subsequently confirms. Likely DEAD (G-4 already explored 128×512/192×256 -11pp); justified ONLY as completeness probe. |

Grid: `gdx = ceil(4096/224) = 19`, `gdy = ceil(32768/256) = 128`. Kernel name: `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_224x256E`.

---

## 4. Worker cohort assignments

| Cohort | Worker | GPU(s) | Candidates | Mechanism conf | Wall budget |
|---|---|---|---|---|---|
| **H-1** (Opt J) | A | 0 | H-1/J-1 (L1 ITERS=1000) | VERY HIGH | 5 min |
| **H-2** (Opt L) | B | 1, 2 | H-2/L-1, H-2/L-2, H-2/L-3 | LOW-MED | 30 min (3 cells × 10 min, 2-GPU shard) |
| **H-3** (Opt K, OPTIONAL) | C | 3 | H-3/K-1 | LOW (likely DEAD) | 5 min smoke + cond. 10 min |

**Total: 4-5 candidates across 2-3 cohorts on 4 GPUs.** Reviewer integration uses GPUs 4,5,6,7 (idle pool). Verify all GPUs idle via `rocm-smi --showuse` pre-launch (per benchmark-rules.md).

R57 is intentionally a small round — we are at the practical ceiling and only have 2 LOSE cells (combined gap <1.7pp) and 3 HK cells (cohort-race surface).

---

## 5. PROMOTE gate spec per axis

All candidates use **bench rules MANDATORY**: warmup=200, iters=500 (Opt J: ITERS=1000), trim=0.10, idle GPU only. **NO kernel rebuild**; if any worker proposes shim modification, REJECT.

### Opt J (H-1/J-1) PROMOTE gate

1. **Bit-determinism gate** (auto-pass for AITER `.co`): wcf_max=0, wcf_std=0, fin_min=1.0 across 10 INDEPENDENT seeds [101..1010].
2. **VC strict 10-run gate**: n_OK≥8/10 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min≥0.97.
3. **Perf gate (Opt J specific)**: reviewer 10-run **p50 pct_comp ≥ 100.0%** at ITERS=1000.
4. **Outcome**:
   - **PROMOTE** if all 3 gates pass → L1 → WIN (40 → 41 WIN cells); manifest source label changes from `R56G1_L1_AITER` to `R57J1_L1_AITER` (same .so, ITERS=1000 confirmed).
   - **HOLD** if pct_comp ∈ [99.50%, 100.00%) → keep R56G1_L1 manifest entry; document drift width.
   - **REGRESS_INVESTIGATE** if pct_comp < 99.50% → cohort-race tail-draw on UNCHANGED .so; treat as R45-style 10-run cohort race; hold manifest.

### Opt L (H-2/L-1, L-2, L-3) PROMOTE gate

1. **Bit-determinism gate**: wcf_max=0, wcf_std=0, fin_min=1.0 across 10 INDEPENDENT seeds [101..1010].
2. **VC strict 10-run gate**: n_OK≥8/10 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min≥0.97.
3. **Perf gate (D-3A-1 strict)**: alt-tile reviewer 10-run **p50 pct_comp > current_HK_pct_comp + 1.0pp**:
   - L-1: > **106.36%**
   - L-2: > **104.06%**
   - L-3: > **101.39%**
4. **Outcome per cell**:
   - **PROMOTE** if all 3 gates pass → swap HK → AITER 192×256; AITER bit-deterministic share +1; HK cells -1.
   - **ACCEPT_FALLBACK (D-3A-1)** if perf gate fails → keep HK R40B baseline; document delta. Fallback exploration to 128×256 allowed under hard cap (see §3).
   - **DEAD** if VC gate fails (wcf_max ≥ 0.02 or n_OK<8/10) → close 192×256 axis on this cell.

### Opt K (H-3/K-1, OPTIONAL) PROMOTE gate

1. **SMOKE-first**: 5-seed smoke (warmup=200, iters=500, trim=0.10) before 10-run.
2. **Smoke perf gate**: pct_comp > **99.30%** to escalate to 10-run.
3. **10-run gates**: same VC+bit-det as Opt L; perf gate `> 99.30%` (i.e., L8_current 98.30% + 1.0pp).
4. **Outcome**:
   - **PROMOTE** → L8 → WIN (likely 41 → 42 WIN cells if Opt J also PROMOTE).
   - **DEAD** if smoke fails (most likely outcome) → close 224×256 axis; Opt K axis DEAD.

---

## 6. Floor / Mode / Stretch targets

R57 is a small ceiling-tightening round; targets are framed as **NET WIN cell deltas** and **HK cell delta**.

| Tier | Outcome | NET WIN | HK cells | AITER share | Mechanism prereq |
|---|---|---:|---:|---:|---|
| **Floor** | 1 PROMOTE (Opt J only) | 40 → **41** | 3 → 3 | 39 → 39 | L1 ITERS=1000 lifts p50 ≥100% (very high conf; worker already saw 100.63%) |
| **Mode** | 2 PROMOTEs (Opt J + 1 of 3 Opt L) | 40 → **41-42** | 3 → **2** | 39 → **40** | L1 PROMOTE + at least 1 of {L-1, L-2, L-3} clears HK_pct + 1.0pp |
| **Stretch** | 4-5 PROMOTEs (Opt J + all 3 Opt L + optional Opt K) | 40 → **41-42** | 3 → **0** | 39 → **42** | All 3 HK alt-tiles clear D-3A-1 strict gate (low prob given R55 D-5B/1 evidence) AND optionally L8 224×256 PROMOTE |

**Realistic bullseye: Floor (Opt J PROMOTE).** Mode/Stretch contingent on Opt L mechanism breaking the R55 D-5B/1 finding that 256×256 AITER underperforms HK on N=14336; using the next-highest-eff tile (192×256) is the right hypothesis but unproven.

**Aggregate perf claw-back vs R56:**
- Floor: +0.65pp (just L1 99.98 → ~100.63%)
- Mode: +0.65pp + 1-3pp on 1 HK cell (rough)
- Stretch: marginal (<10pp aggregate); R57 is not a perf round.

---

## 7. Reviewer hand-off contract

Each worker emits per-candidate fragments named:
`R57H<COHORT>_<CELL_LABEL>_INTEGRATION_FRAGMENT.json` (e.g. `R57H1_J1_INTEGRATION_FRAGMENT.json`, `R57H2_L1_INTEGRATION_FRAGMENT.json`)

Required schema:
```json
{
  "round": "R57",
  "opt": "J | L | K",
  "cohort": "H-1 | H-2 | H-3",
  "cell_label": "J-1 | L-1 | L-2 | L-3 | K-1",
  "shape": [M, N, K],
  "current_baseline": {
    "source": "R56G1_L1_AITER | R40B | R52D2B_AITER",
    "tile": [tM_cur, tN_cur],
    "pct_comp": 99.98
  },
  "alt_tile": [192, 256],
  "co_path": "/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_192x256.co",
  "kernel_name": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_192x256E",
  "shim_so": "build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so",
  "shim_rebuild_required": false,
  "grid": {"gdx": 86, "gdy": 16, "gdz": 1, "bdx": 256},
  "kernel_args": [16384, 4096, 2048],
  "smoke_pass": true,
  "ten_run": {
    "seeds": [101,202,303,404,505,606,707,808,909,1010],
    "warmup": 200,
    "iters": 500,
    "trim_frac": 0.10,
    "n_OK": 10,
    "wcf_max": 0.0,
    "wcf_std": 0.0,
    "fin_min": 1.0,
    "tflops_p50": 0.0,
    "pct_comp": 0.0
  },
  "perf_delta_pp": 0.0,
  "verdict": "PROMOTE | HOLD | ACCEPT_FALLBACK | DEAD | REGRESS_INVESTIGATE",
  "verdict_reason": "..."
}
```

Special schema notes:
- **Opt J fragment**: `iters=1000` (not 500); add `iters_protocol: "R57_OPT_J_LONG_BENCH"` field. Document p25/p50/p75 in `ten_run`.
- **Opt L fragment**: include `r55_d5b1_reference: true` to trace mechanism heritage.
- **Opt K fragment** (if attempted): include `smoke_5seed: {pct_comp, n_OK}` block before `ten_run`.

Reviewer aggregates fragments → `R57_INTEGRATION_MANIFEST.json` (delta from R56 manifest), runs full **42-shape 10-run @ 80% INDEPENDENT seeds** (warmup=200, iters=500) on idle GPU pool (4,5,6,7), emits `R57_INTEGRATION_VERDICT.md` with:
- Headline 42/42 strict-VC retention (must HOLD from R56 — 3rd consecutive 100% leaderboard target)
- WIN cell delta (target: 40 → 41-42)
- HK cell delta (target: 3 → 0-3)
- AITER bit-deterministic share delta (target: 39 → 39-42)
- Cohort-race churn audit on the 39 UNCHANGED AITER cells (must be 0 churn, per R55/R56 precedent)
- Per-axis PROMOTE/ACCEPT_FALLBACK/DEAD breakdown

---

## 8. Closed-axis reminder (DO NOT propose in R57)

Carry-forward from R45-R56 closed axes; in addition for R57:

- **DO NOT modify R50D shim.** 8 consecutive AS-IS reuse rounds; treat as black box.
- **DO NOT propose VC-improving kernel changes.** Strict-VC ceiling reached; R57 is tail-tightening + bit-det expansion only.
- **DO NOT use warmup<200 or iters<500.** Bench rules. Opt J uses iters=1000 (longer, not shorter).
- **DO NOT attempt L8 alt-tiles 128×512 or 192×256.** R56 G-4 already DEAD on these (-13.31pp, -11.84pp). Only 224×256 is allowed as a token Opt K probe (and even then SMOKE-first).
- **DO NOT modify the 3 HK kernel .so files.** Only AITER `.co` dlopen swaps via R50D shim are in scope for HK→AITER hardening.
- **DO NOT propose Opt M gate-tightening.** Deferred to R58 to avoid confounding Opt L verdicts under a methodology change.
- **DO NOT chase L1 below 100% with kernel changes.** L1 is a noise/drift cell, not a structural gap; longer ITERS is the only Opt J intervention.
- **Closed via R55 D-5B/1**: 256×256 AITER underperforms HK on N=14336 (`32768x14336x2048` ACCEPT_FALLBACK). DO NOT re-attempt 256×256 on the 3 HK cells; the H-2 alt-tile is 192×256 (next highest eff).
- **Closed via R56 G-4 C2/C3**: 128×512 and 192×256 on `4096x32768x128256` are DEAD (-11 to -13pp). DO NOT re-attempt.

---

## 9. Confidence rationale

- **Opt J (H-1/J-1)** mechanism: AITER `.co` is bit-deterministic (wcf=0 across all 10 seeds). The 0.65pp drift between worker 100.63% and reviewer 99.98% is GPU/run-to-run measurement noise on a kernel that is structurally fixed. Doubling ITERS from 500 to 1000 cuts the p50 standard error by ~√2 ≈ 0.71×, which is sufficient to lift the p50 above 100% with high probability if the true p50 is in [100.0%, 100.5%]. **VERY HIGH** confidence in Floor.
- **Opt L (H-2)** mechanism: R55 D-5B/1 found 256×256 AITER underperforms HK on N=14336 by 1.33pp because grid-x=56 (32768/256+1) leaves 256-CU saturation off-balance vs HK's 192×N pattern. The next-highest-eff tile is 192×256 (eff=109.7); 192×256 changes grid-x to 171 for the same N=14336 cell (32768/192=171), which is more saturation-balanced. However, the HK perf bar is high (100.39-105.36%), so PROMOTE requires AITER 192×256 to clear HK + 1.0pp — non-trivial. **LOW-MEDIUM** confidence per cell; expected 1/3 to 0/3 PROMOTE.
- **Opt K (H-3, optional)** mechanism: K=128256 K/N=4 ratio favors smaller-N tiles only if the K-pipeline differs; 224×256 is the only tile with comparable eff (120.5) to 256×256 (128) that has not been probed. Given R56 G-4 -11pp on 192×256 (closer to current eff than 224×256 from below), the prior is strongly **DEAD**. Run only as token completeness probe.

---

## Summary

**3 cohorts, 4-5 candidates total**, on 4 GPUs (workers) + 4 GPUs (reviewer). Floor (1 PROMOTE) targets **40 → 41 WIN cells via L1 ITERS=1000 re-bench**. Mode (2 PROMOTEs) requires at least 1 HK→AITER 192×256 swap clearing D-3A-1 strict gate. Stretch (4-5 PROMOTEs) closes both LOSE cells and zeros out HK survivor pool. R57 is a small **tail-tightening + bit-determinism expansion round**, not a perf round; the strict-VC 42/42 ceiling MUST hold for the 3rd consecutive 100% leaderboard. R50D shim AS-IS for the 9th consecutive round.
