# R59 Decider Plan — Mixed-Protocol Policy (Opt R) + L3 HK Survivor Rescue (Opt S) + L1 Noise-Edge Re-attack (Opt V)

**Date:** 2026-04-20
**R58 baseline:** **41/42 strict 10-run VC** (committed `8588b2b3`); 40/42 WIN; 40 AITER + 2 HK; 10th consecutive R50D shim AS-IS reuse round; 1 cohort-race tail-draw on UNCHANGED L3 R40B HK; AITER bit-deterministic share **40/42 (largest in project history)**; HK pool **2 (smallest in project history)**.
**Goal:** Recover the L3 (32768x14336x2048) VC drop without trading bit-determinism, decide the mixed-protocol leaderboard policy in writing, and probe one bounded-cost re-attack on L1 noise-edge — all under R45+ default ITERS=500 with 0 closed-axis re-attempts.

---

## 1. R58 baseline reminder

- **41/42 strict 10-run VC** (gate: n_OK≥8/10 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min≥0.97)
- **40 AITER overrides** (R50D shim AS-IS, 10th consecutive AS-IS reuse round) + **2 HK baselines**
- **40/42 WIN** (≥100% pct_comp); **2 LOSE cells + 1 VC-flipped cell**:
  - **L1 `(4096,32768,14336)`** — reviewer p50 **99.94%** (R57J1_L1_AITER 256×256; ITERS=500 revert pushed it back to LOSE-edge from R57's 100.04% under ITERS=1000; bit-deterministic wcf_max=0)
  - **L8 `(4096,32768,128256)`** — reviewer p50 **98.28%** (R52D2B_AITER 256×256; aiter alt-tile axis FULLY CLOSED + HK 256×256 axis CLOSED by correctness; bit-deterministic wcf_max=0)
  - **L3 `(32768,14336,2048)`** — reviewer p50 **100.49%** (R40B HK; numerically WIN but VC-flipped n_OK=9/10 wcf_max=0.0111 fin_min=0.911 under ITERS=500; predicted by R57 reviewer flag fin_min=0.9847 + R58 Opt N "worst-margin survivor")
- **AITER bit-deterministic share 40/42 HELD** (wcf=0 across 10 INDEPENDENT seeds on every AITER cell)
- **2 surviving HK cells**:
  - `16384x4096x2048` R40B — 107.48% (HOLD VC+WIN; n_OK=10/10, wcf_max=0.000586, fin_min=0.997)
  - `32768x14336x2048` R40B — 100.49% (VC-flipped; numerically WIN; lone R58 cohort-race tail-draw)
- Strict-VC ceiling REGRESSED 1 cell under ITERS=500 revert (predicted price documented in R58 verdict §"ITERS=500 revert observation").

---

## 2. R59 axis selection rationale

### Selected axes (3 cohorts)

| Axis | Status | Mechanism conf | Expected value | Rationale |
|---|---|---|---|---|
| **Opt R** — Mixed-protocol leaderboard policy decision (analysis-only, NO GPU) | **RUN** | METHODOLOGY (cert) | +0 PROMOTE; durable policy artifact for R59-R65 | ITERS=500 vs ITERS=1000 boundary needs a written disposition. R57 Opt J validated ITERS=1000 for ≤0.5pp boundary cells; R58 reverted with predicted L1 LOSE-edge re-emergence. Decide once and document; recommendation = **ITERS=500 default + L1 footnote** (no mixed-protocol). |
| **Opt S** — L3 `(32768,14336,2048)` HK survivor rescue (low-medium confidence) | **RUN** | LOW-MED | +0 to +1 NET VC | Lone VC-flipper. Try non-128×256 alt-tiles (96×640 eff=83.5, 64×1024 eff=60.2). 128×256 already SMOKE-DEAD on this cell at -25.04pp (R58 P-3); 192×256 DEAD at -14.38pp (R57 L-3); 256×256 AITER underperforms HK (R55 D-5B/1). Lower-eff probes are the only remaining alt-tile axis. Bounded-cost SMOKE-first per cell. |
| **Opt V** — L1 `(4096,32768,14336)` noise-edge re-attack (low confidence) | **RUN** | LOW | +0 to +1 NET WIN | L1 sits −0.06pp under WIN line at ITERS=500. Aiter 256×256 R57J1 is already bit-deterministic. Probe alt-tile (96×640 eff=83.5, 64×1024 eff=60.2) — both never tried on L1 since R56. Mechanism: large K=14336 may favor a wider tile; closure value if SMOKE-DEAD. |

### Deferred / skipped axes

| Axis | Status | Reason for defer/skip |
|---|---|---|
| **Opt T** — L8 from-scratch HK kernel build for K=128256 | **DEFER** | Very low confidence, ~3 R-rounds budget; needs explicit election. Not in R59 scope. |
| **Opt U** — Accept residual + documentation pivot | **DEFER** | Premature; R59 still has bounded-cost mechanism axes (Opt S + Opt V). Re-evaluate in R60+ if Opt S + Opt V both DEAD. |
| **L1 ITERS=1000 one-off bump** | **CLOSED by Opt R disposition** | R57 + R58 demonstrated mixed-protocol cost is not worth the single-cell flip. Opt R formalizes this as policy. |
| **128×256 / 192×256 / 256×256 AITER on L3** | **CLOSED** | All three DEAD per R55 D-5B/1, R57 L-3, R58 P-3. |
| **HK 256×256 lgk2 on L8 K=128256** | **CLOSED by correctness** | R58 Opt O R40B + R37 fallback both WRONG_OUTPUT. |
| **All R57+R58 closed list** | **CLOSED** | Carry-forward (see §9). |

---

## 3. Per-cohort candidate list with grid math

All AITER candidates use **R50D shim AS-IS** (`build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`); NO kernel rebuild. KernelArgs are `(M, N, K)`; bdx=256; gdz=1.

### Cohort J-1 (Opt R) — Mixed-protocol policy decision (analysis-only)

**Pure analysis cohort, no GPU bench, ~10 min wall-clock.**

| ID | Action | Input | Output |
|---|---|---|---|
| **J-1/R-1** | Tabulate per-cell pct_comp delta R57(ITERS=1000) vs R58(ITERS=500) on the 41 unchanged-binary cells | `R57_INTEGRATION_10RUN.json`, `R58_INTEGRATION_10RUN.json` | per-cell drift table |
| **J-1/R-2** | Identify all cells in [99.5%, 100.5%] band at R58 — these are the population for which ITERS choice can flip the WIN/LOSE classification. Confirm only L1 falls in the band (per R58 verdict). | same | small list (expected: L1 only) |
| **J-1/R-3** | Decide policy: (A) ITERS=500 default + L1 footnote, (B) ITERS=1000 default (cost: 2× wall per round + ~17 → ~34 min reviewer integration), (C) mixed-protocol L1-only ITERS=1000 (leaderboard inconsistency cost + per-cell protocol annotation in manifest). Recommend (A). | analysis | written disposition |
| **J-1/R-4** | Emit `R59_OPT_R_POLICY.md` with: (a) per-cell drift table; (b) cost/benefit table for each option; (c) recommendation + rationale; (d) "boundary-bump validity envelope" rule for any future one-off ITERS bumps (must be ≤0.5pp, must be bit-deterministic, must be documented). | output | 1-page policy artifact |

**Worker output**: `R59_OPT_R_POLICY.{md,json}`. NO manifest changes (Opt R is policy-only).

### Cohort J-2 (Opt S) — L3 HK survivor rescue alt-tile probe

**1 cell × 2 alt-tile candidates, 2 GPUs paired, ~15 min wall.**

| ID | Shape (HK pct_comp) | Alt tile | `.co` file | Grid (gdx, gdy, gdz) | SMOKE gate to escalate |
|---|---|---|---|---|---|
| **J-2/S-1** | `32768x14336x2048` (HK R40B 100.49%, VC-flipped) | 96×640 (eff=83.5) | `..._BpreShuffle_96x640.co` | (341, 23, 1) — `gdx=ceil(32768/96)=342? actually 32768/96=341.33→342`; **worker MUST recompute exact ceil and validate from manifest dispatch helper** | SMOKE pct_comp > **101.49%** (HK + 1.0pp) AND VC strict |
| **J-2/S-2** | `32768x14336x2048` (HK R40B 100.49%, VC-flipped) | 64×1024 (eff=60.2) | `..._BpreShuffle_64x1024.co` | (512, 14, 1) — `gdx=ceil(32768/64)=512`, `gdy=ceil(14336/1024)=14` | SMOKE pct_comp > **101.49%** (HK + 1.0pp) AND VC strict |

**Mechanism hypothesis**: 128×256 (eff=85.3) catastrophically failed (-25.04pp) because grid_x = ceil(32768/128) = 256 vs HK 256×256 grid_x = 128 — cohort-race surface roughly **doubled**. The lower-eff but **wider tiles** 96×640 and 64×1024 produce **smaller grid_x** (342 and 512 respectively — wait, BOTH grid_x larger; this is mechanism-skeptical) but **dramatically smaller grid_y** (96×640 grid_y=23 vs HK grid_y=56; 64×1024 grid_y=14 vs HK grid_y=56). The grid-y reduction may favor different XCD-load profiles. **Confidence is LOW**; this is primarily an axis-closure probe.

Kernel names: `_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_96x640E`, `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024E`.

**Hard cap**: 5 GPU-min per cell on SMOKE; if both SMOKE-DEAD, **96×640 + 64×1024 alt-tile axis CLOSED for L3** (combined with 128×256 / 192×256 / 256×256 closure, alt-tile axis fully exhausted on L3; only structural rescue path would be HK kernel rebuild with R44D FINITE_GATE 0.97 → 0.95 — defer to R60).

### Cohort J-3 (Opt V) — L1 noise-edge re-attack alt-tile probe

**1 cell × 2 alt-tile candidates, 2 GPUs paired, ~15 min wall.**

| ID | Shape (current AITER pct_comp) | Alt tile | `.co` file | Grid (gdx, gdy, gdz) | SMOKE gate to escalate |
|---|---|---|---|---|---|
| **J-3/V-1** | `4096x32768x14336` (L1, AITER 256×256 R57J1 99.94%) | 96×640 (eff=83.5) | `..._BpreShuffle_96x640.co` | (43, 52, 1) — `gdx=ceil(4096/96)=43`, `gdy=ceil(32768/640)=52` | SMOKE pct_comp > **100.94%** (current + 1.0pp) AND VC strict |
| **J-3/V-2** | `4096x32768x14336` (L1, AITER 256×256 R57J1 99.94%) | 64×1024 (eff=60.2) | `..._BpreShuffle_64x1024.co` | (64, 32, 1) — `gdx=ceil(4096/64)=64`, `gdy=ceil(32768/1024)=32` | SMOKE pct_comp > **100.94%** (current + 1.0pp) AND VC strict |

**Mechanism hypothesis**: K=14336 is intermediate-K (not the dominant K-bound regime where 256×256 wins). Wider-N tiles (96×640, 64×1024) reduce grid_y by 5-8× which may improve XCD load balance for M=4096 small-M dispatches. **Confidence is LOW**; aiter heuristic disfavors these tiles for K=14336 (both have eff < 100), but worker should empirically validate.

**Hard cap**: 5 GPU-min per cell on SMOKE; if both SMOKE-DEAD, **L1 alt-tile axis CLOSED**, L1 stays at R57J1 R45+ default ITERS=500 99.94% LOSE-edge per Opt R disposition (A).

---

## 4. Worker cohort assignments

| Cohort | Worker | GPU(s) | Candidates | Mechanism conf | Wall budget |
|---|---|---|---|---|---|
| **J-1** (Opt R) | A | NONE (analysis only) | J-1/R-1, R-2, R-3, R-4 | METHODOLOGY | 10 min |
| **J-2** (Opt S) | B | 1, 2 | J-2/S-1, J-2/S-2 | LOW-MED | 15 min (5 min SMOKE × 2 cells, parallel pair) + cond. 30 min 10-run if escalate |
| **J-3** (Opt V) | C | 5, 6 | J-3/V-1, J-3/V-2 | LOW | 15 min (5 min SMOKE × 2 cells, parallel pair) + cond. 30 min 10-run if escalate |

**Total: 4-5 candidates across 3 cohorts on 4 GPUs (workers).** GPUs 0, 3 are BUSY (other workloads); workers use ONLY GPUs 1, 2, 5, 6. Reviewer integration uses GPUs 4, 5, 6, 7 (verify `rocm-smi --showuse` idle pre-launch per benchmark-rules.md). Worker C and reviewer share GPUs 5, 6 — **worker C must release GPUs before reviewer launches** (sequential, not concurrent).

R59 is intentionally a small round — practical WIN ceiling at 41/42 with L8 structurally floored and L1 at noise-edge; methodology Opt R is the highest-information axis; Opt S + Opt V are bounded-cost mechanism probes.

---

## 5. PROMOTE gate spec per axis

All bench candidates use **bench rules MANDATORY**: warmup=200, **iters=500 (R45+ default)**, trim=0.10, idle GPU only. **NO kernel rebuild**; if any worker proposes shim modification, REJECT. **All 10-run gates use INDEPENDENT seeds [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]**.

### Opt R (J-1) — methodology pilot (no PROMOTE possible)

1. **No PROMOTE.** Opt R produces a written policy artifact only.
2. Acceptance: `R59_OPT_R_POLICY.md` exists, contains:
   - Per-cell pct_comp drift table R57(ITERS=1000) vs R58(ITERS=500) on 41 unchanged-binary cells
   - Cells in [99.5%, 100.5%] WIN-flip-sensitive band (expected: L1 only)
   - Cost/benefit table for options (A/B/C)
   - Recommendation (A: ITERS=500 default + L1 footnote)
   - Future ITERS-bump validity envelope (≤0.5pp boundary, bit-deterministic, documented)

### Opt S (J-2) — L3 HK survivor rescue PROMOTE gate

1. **SMOKE-first**: 5-seed smoke (warmup=200, iters=500, trim=0.10, seeds [101,202,303,404,505]) before 10-run.
2. **SMOKE perf gate**: pct_comp > **101.49%** (HK 100.49% + 1.0pp D-3A-1) to escalate to 10-run.
3. **STOP rule**: if SMOKE pct_comp < **95.49%** (HK_pct - 5.0pp), **STOP_DEAD immediately**, ACCEPT_FALLBACK; do NOT run 10-run; close axis.
4. **10-run gates** (only if SMOKE escalates):
   - **VC strict 10-run gate**: n_OK≥8/10 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min≥0.97 (R45+ gate)
   - **Perf gate (D-3A-1 strict)**: reviewer 10-run p50 pct_comp > **101.49%** (HK + 1.0pp)
   - **AITER → HK swap special clause**: PROMOTE valid because alt converts a VC-flipped HK cell to potentially bit-deterministic AITER (acceptable trade if perf gate passes; LOSE-direction trade rejected by D-3A-1).
5. **Outcome**:
   - **PROMOTE** if all gates pass → L3 → AITER bit-deterministic; AITER share 40 → 41; HK pool 2 → 1 (smallest ever); +1 strict VC (41→42, restoring 100% leaderboard); +1 WIN cell.
   - **ACCEPT_FALLBACK (D-3A-1)** if perf gate fails → keep R40B HK baseline; document delta; L3 stays VC-flipped.
   - **DEAD / SMOKE_DEAD** if SMOKE STOP triggers → close 96×640 / 64×1024 axis on L3.

### Opt V (J-3) — L1 noise-edge re-attack PROMOTE gate

1. **SMOKE-first**: 5-seed smoke per cell (hard cap 5 GPU-min).
2. **SMOKE perf gate**: pct_comp > **100.94%** (current AITER 99.94% + 1.0pp D-3A-1) to escalate to 10-run.
3. **STOP rule**: if SMOKE pct_comp < **94.94%** (current - 5.0pp), STOP_DEAD ACCEPT_FALLBACK.
4. **10-run gates** (only if SMOKE escalates):
   - **VC strict 10-run gate**: n_OK≥8/10 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min≥0.97 (R45+ gate)
   - **Perf gate (D-3A-1 strict)**: reviewer 10-run p50 pct_comp > **100.94%** (current + 1.0pp)
   - **WIN gate (additional)**: pct_comp ≥ **100.0%** for "+1 WIN cell" claim
5. **Outcome**:
   - **PROMOTE** if all gates pass AND pct_comp ≥ 100.0% → L1 → WIN; AITER share 40 (unchanged, AITER→AITER swap); HK pool 2 (unchanged); +1 WIN cell (40→41).
   - **PROMOTE_PARTIAL** if all gates pass AND pct_comp ∈ [100.94%, 100.0%) — impossible band, skip.
   - **ACCEPT_FALLBACK (D-3A-1)** if perf gate fails → keep R57J1_L1 baseline; document delta; L1 stays LOSE-edge per Opt R policy.
   - **DEAD / SMOKE_DEAD** if SMOKE STOP triggers → close 96×640 / 64×1024 axis on L1.

---

## 6. D-3A-1 protection rules (carried over from R57-R58)

- **No swap** of an existing manifest entry unless candidate strictly wins ≥+1.0pp over current p50 pct_comp at reviewer 10-run.
- **ACCEPT_FALLBACK** preserves the current manifest entry verbatim (binary, grid, kernel name, source label).
- **SMOKE-DEAD** triggers ACCEPT_FALLBACK without 10-run cost.
- **VC LOSS** at 10-run on a swap candidate → REVERT immediately, do NOT merge fragment, treat as DEAD.
- **HK → AITER swap special clause**: PROMOTE valid even at modest perf delta IF candidate is bit-deterministic AND existing HK is VC-flipped (Opt S J-2 case); however perf gate ≥+1.0pp still required.
- **AITER → AITER alt-tile swap** (Opt V J-3 case): standard +1.0pp gate; AITER bit-det share unchanged either way.

### D-3C (anti-streak protection)

- The R58 cohort-race tail-draw on UNCHANGED L3 binary is documented as protocol-induced (ITERS revert), NOT a swap-induced churn. R59 reviewer must audit that any R59 PROMOTE swap on L3/L1 does not introduce additional cohort-race churn on the other 40 unchanged-binary cells (≤0 lost VC tolerated; document any).

### D-5A (perf claw-back conservation)

- R59 has no D-5 perf-claw-back cohorts. Perf claw-backs are out-of-scope this round.

---

## 7. Floor / Mode / Stretch targets

| Tier | PROMOTEs | NEW WIN cells | NEW VC cells | HK cells | AITER share | Mechanism prereq |
|---|---:|---:|---:|---:|---:|---|
| **Floor** | 0 PROMOTE + Opt R policy delivered | 40 (HELD) | 41/42 (HELD) | 2 (HELD) | 40 (HELD) | Opt R artifact exists; SMOKE-DEAD on Opt S + Opt V acceptable |
| **Mode** | 1 PROMOTE (Opt S OR Opt V) + Opt R policy + ≥1 axis closure | 41 | 42/42 (Opt S) OR 41/42 (Opt V) | 1 (Opt S) or 2 (Opt V) | 41 (Opt S) or 40 (Opt V) | One alt-tile clears D-3A-1 +1.0pp |
| **Stretch** | 2 PROMOTE (Opt S AND Opt V) + Opt R policy | 42 (FULL LEADERBOARD WIN) | 42/42 | 1 | 41 | Both alt-tile probes clear D-3A-1; very low joint probability |

**Realistic bullseye: Floor + Opt R artifact.** Opt R delivers durable methodology certainty regardless of bench outcome. Opt S + Opt V are low-confidence axis-closure probes; combined expected NET PROMOTEs ≈ 0.2.

**Aggregate perf claw-back vs R58:**
- Floor: +0pp (no manifest changes)
- Mode: +1.0 to +2.5pp on the 1 PROMOTE cell
- Stretch: +1.0 to +2.5pp × 2 cells

---

## 8. Reviewer integration plan (post-worker)

Each worker emits per-candidate fragments named:
- `R59_OPT_R_POLICY.{md,json}` (Opt R analysis-only)
- `R59J2_S{1,2}_INTEGRATION_FRAGMENT.json` (Opt S candidates)
- `R59J3_V{1,2}_INTEGRATION_FRAGMENT.json` (Opt V candidates)

Required fragment schema (Opt S / Opt V):
```json
{
  "round": "R59",
  "opt": "S | V",
  "cohort": "J-2 | J-3",
  "cell_label": "S-1 | S-2 | V-1 | V-2",
  "shape": [M, N, K],
  "current_baseline": {
    "source": "R40B | R57J1_L1_AITER",
    "tile": [tM_cur, tN_cur],
    "pct_comp": 100.49
  },
  "candidate": {
    "kind": "AITER_co",
    "co_path": "...",
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
  "verdict": "PROMOTE | ACCEPT_FALLBACK | DEAD",
  "verdict_reason": "..."
}
```

**Worker output files (per cohort)**:
- `bench_R59J2_S{1,2}.py` (Opt S bench scripts; copy from `bench_R58I3_P*.py` template, swap candidate to 96×640 / 64×1024 dispatch)
- `R59_OPT_S{1,2}_{SMOKE,10RUN}.{json,log}` (per cell)
- `R59J2_S{1,2}_INTEGRATION_FRAGMENT.json`
- `R59_OPT_J2_VERDICT.md` (cohort-level summary)
- analogous for J-3 (Opt V)
- analogous report-only for J-1 (Opt R: `R59_OPT_R_POLICY.md`)

**Reviewer steps** (post-worker, ~25 min wall):
1. Aggregate fragments → `R59_INTEGRATION_MANIFEST.json` (delta from R58 manifest; if 0 PROMOTEs, manifest is byte-identical to R58 with version bump only).
2. Run `bench_all_42_R59_INTEGRATION.py` (copy of `bench_all_42_R58_INTEGRATION.py` with **iters=500 default unchanged**; add 96×640 / 64×1024 dispatch entries IF any PROMOTE) **SMOKE 1-seed** on GPUs 4,5,6,7 (~2 min).
3. Run full 10-run @ INDEPENDENT seeds [101..1010] @ **ITERS=500 default** on GPUs 4,5,6,7 (~17 min wall, 420 runs).
4. Cohort-race churn audit on all 42 cells (target: 0 lost VC on UNCHANGED cells; ≤1 cell tolerated only if explained by independent seed tail-draw, NOT swap-induced).
5. Compose `R59_INTEGRATION_VERDICT.md`:
   - Headline: strict-VC retention/recovery (target: 41/42 HELD or 42/42 RECOVERED via Opt S PROMOTE)
   - WIN cell delta (target: 40 HELD or 41-42 via Opt S or Opt V PROMOTE)
   - HK cell delta (target: 2 HELD or 1 via Opt S PROMOTE)
   - AITER bit-deterministic share delta (target: 40 HELD or 41 via Opt S PROMOTE)
   - Cohort-race churn audit on UNCHANGED cells (target: 0)
   - Per-axis PROMOTE / ACCEPT_FALLBACK / DEAD breakdown
   - Opt R policy artifact linkage (cite `R59_OPT_R_POLICY.md`); confirm adoption of recommendation (A) for R60+
6. R59 specifically tests: does the L3 cohort-race tail-draw repeat under the SAME ITERS=500 protocol with a fresh INDEPENDENT seed sweep? If reviewer 10-run on UNCHANGED L3 R40B HK shows n_OK≥8/10 (recovery), then R58's lone VC drop was a single-sweep tail-draw; if it shows n_OK<8/10 again, the cohort-race surface on this cell is intrinsic to ITERS=500 and Opt S's PROMOTE/DEAD outcome becomes the deciding factor.

---

## 9. Closed-axis reminder (DO NOT propose in R59)

Carry-forward from R45-R58 closed axes; in addition for R59:

- **DO NOT modify R50D shim.** 10 consecutive AS-IS reuse rounds; treat as black box.
- **DO NOT propose VC-improving kernel changes.** Strict-VC ceiling regressed 1 cell under ITERS=500 revert; no new kernel-side VC available without rebuild.
- **DO NOT use warmup<200 or iters<500.** Bench rules. R59 maintains ITERS=500 default per Opt R recommendation (A).
- **DO NOT bump ITERS=1000** (R57 closed; R58 reverted; Opt R formalizes the no-mixed-protocol policy).
- **DO NOT attempt L8 alt-tiles 128×512, 192×256, 224×256, 96×640, 64×1024 on AITER.** R56 G-4 + R57 H-3 + alt-tile reasoning fully closed aiter alt-tile axis on L8.
- **DO NOT attempt HK 256×256 lgk2 on L8 K=128256.** R58 Opt O closed by correctness (R40B + R37 fallback both WRONG_OUTPUT; needs from-scratch K=128256 build with R39A/R44A/R44D ports — Opt T axis, deferred).
- **DO NOT attempt 192×256 or 256×256 AITER on L3 `(32768,14336,2048)`.** R55 D-5B/1, R57 L-3, R58 P-3 all DEAD.
- **DO NOT attempt 128×256 on L3 `(32768,14336,2048)`.** R58 P-3 DEAD at -25.04pp.
- **DO NOT attempt 128×256 on `(16384,4096,2048)` HK survivor.** R58 P-1 DEAD at -3.68pp.
- **DO NOT attempt L8 from-scratch HK kernel build (Opt T)** unless R59 explicitly elects it. NOT ELECTED for R59.
- **DO NOT attempt 192×256 on L1.** R57 H-2 axis closure on HK cells; for L1 we are probing 96×640 / 64×1024 only.
- **DO NOT modify the 2 surviving HK kernel .so files.** Only AITER `.co` dlopen swaps via R50D shim are in scope for HK→AITER hardening; for L3 Opt S, only SMOKE-first AITER alt-tile probes (no HK rebuild).
- **DO NOT use GPUs 0 or 3** (currently busy with other workloads). Workers use ONLY GPUs 1, 2, 5, 6. Reviewer uses ONLY GPUs 4, 5, 6, 7.

---

## 10. STOP rules

- **Floor reached** (Opt R artifact delivered, 0 PROMOTEs OK): may stop early; reviewer integration becomes a re-bench-at-ITERS=500 pass with manifest version bump only.
- **SMOKE-DEAD on Opt S cell** (both J-2/S-1 and J-2/S-2): ACCEPT_FALLBACK; keep R40B HK manifest entry; close 96×640 + 64×1024 alt-tile axis on L3. L3 cohort-race surface stays as documented R58 attention cell; R60+ decides Opt T vs Opt U.
- **SMOKE-DEAD on Opt V cell** (both J-3/V-1 and J-3/V-2): ACCEPT_FALLBACK; keep R57J1_L1 manifest entry; close 96×640 + 64×1024 alt-tile axis on L1. L1 stays at LOSE-edge per Opt R policy (A).
- **ANY cell loses VC at 10-run on a swap candidate**: REVERT immediately, do NOT merge fragment, treat as DEAD.
- **Reviewer integration smoke fails to reach 41/42 VC**: REVERT manifest to R58 byte-for-byte, treat round as ACCEPT_FALLBACK across the board, document the regression in `R59_INTEGRATION_VERDICT.md`. If reviewer reaches **42/42 VC** (Opt S PROMOTE recovers L3), commit and celebrate (4th 100% leaderboard round though non-consecutive).

---

## 11. Risk / rollback summary

| Risk | Mitigation |
|---|---|
| Opt S PROMOTE candidate fails 10-run VC at reviewer | REVERT to R58 manifest byte-for-byte; L3 stays VC-flipped at 100.49%; log axis closure |
| Opt V PROMOTE candidate fails 10-run VC at reviewer | REVERT to R58 manifest byte-for-byte; L1 stays at LOSE-edge 99.94%; log axis closure |
| Cohort-race churn on UNCHANGED cells (additional VC drops beyond L3) | Document in verdict; do NOT REVERT non-affected swaps; flag as ITERS=500 protocol residual |
| Reviewer GPUs 4-7 not idle at integration time | Wait or shift to GPUs 1, 2 if released by workers; bench-rules.md requires idle GPU |
| Worker C (J-3 on GPUs 5, 6) overlaps reviewer (GPUs 5, 6) | Worker C must complete and release GPUs 5, 6 before reviewer integration launches; reviewer scheduler must wait |
| Opt R policy (A) recommendation rejected by user | Revisit in R60 decider plan; R59 still delivers per-cell drift table as artifact |

---

## 12. Confidence rationale

- **Opt R (J-1)** is methodology-certain: it produces a deterministic policy artifact from existing 10-run JSON. The interesting result is the per-cell drift table demonstrating that L1 is the ONLY cell in the WIN-flip-sensitive [99.5%, 100.5%] band under ITERS=500. Recommendation (A) follows directly from R57 + R58 evidence (mixed protocol = leaderboard inconsistency for 1 cell flip; not justified).
- **Opt S (J-2)** is low-medium confidence: 96×640 (eff=83.5) and 64×1024 (eff=60.2) have lower aiter heuristic scores than 192×256 (eff=109.7) which already DEAD on this cell. Mechanism hypothesis (grid_y reduction → better XCD load balance) is plausible but unverified. Probability of any cell PROMOTE: ≤15%. Run for axis closure + the small chance L3 rescue lands.
- **Opt V (J-3)** is low confidence: same lower-eff alt-tiles applied to L1 (M=4096 small-M, K=14336 intermediate-K). Aiter heuristic strongly favors 256×256 for this regime. Probability of PROMOTE: ≤10%. Run for axis closure + the small chance L1 noise-edge crosses WIN under wider tile.

---

## Summary

**3 cohorts (J-1 analysis-only, J-2 L3 HK survivor rescue, J-3 L1 noise-edge re-attack), 4-5 candidates total**, on 4 GPUs (workers; GPUs 1, 2, 5, 6) + 4 GPUs (reviewer; GPUs 4, 5, 6, 7). Floor (0 PROMOTEs + Opt R policy delivered) is the realistic bullseye and produces certain methodology value. Mode (1 PROMOTE + axis closures) requires Opt S OR Opt V SMOKE escalation. Stretch (2 PROMOTE → 42/42 WIN + 42/42 VC) requires both alt-tile probes to clear D-3A-1 — very low joint probability.

R59 is intentionally a small **methodology + bounded-mechanism-probe round**; the strict-VC 41/42 ceiling MUST hold for R59 (or recover to 42/42 via Opt S PROMOTE). R50D shim AS-IS for the **11th consecutive round**. ITERS=500 default maintained per Opt R recommendation (A); no mixed-protocol bump for L1.

The primary durable deliverables are: (1) `R59_OPT_R_POLICY.md` (mixed-protocol disposition), (2) Opt S axis closure documentation (L3 alt-tile space exhausted post-R59 if both DEAD), (3) Opt V axis closure documentation (L1 alt-tile space exhausted post-R59 if both DEAD), (4) cohort-race repeatability evidence (does the R58 L3 tail-draw repeat under fresh INDEPENDENT seed sweep at ITERS=500?). Any PROMOTE is upside.
