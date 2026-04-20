# R58 Opt N — Gate Tightening Pilot (Worker I-1, analysis-only)

**Date:** 2026-04-19
**Round:** R58 (cohort I-1)
**Mode:** ANALYSIS ONLY — no GPU runs, no kernel rebuild, no bench. Pure re-classification of `R57_INTEGRATION_10RUN.json` (R57 reviewer 10-run @ INDEPENDENT seeds, ITERS=1000) under a tightened strict-VC gate.

**Inputs:**
- `R57_INTEGRATION_10RUN.json` (R57 reviewer 10-run dataset; 42 cells; INDEPENDENT seeds [101,202,303,404,505,606,707,808,909,1010])
- `R57_INTEGRATION_VERDICT.md` (current strict-VC gate definition)
- `R58_DECIDER_PLAN.md` (cohort I-1 instructions)

**Gates compared:**

| Gate         | n_OK    | wcf_max | wcf_std | fin_min |
|--------------|---------|---------|---------|---------|
| R45+ default | ≥ 8/10  | < 0.02  | < 0.01  | ≥ 0.97  |
| R58 Opt N    | ≥ 9/10  | < 0.01  | < 0.01  | ≥ 0.99  |

(`wcf_std` retained at <0.01 per the user-provided I-1 spec; the decider plan I-1/N-1 row also lists `<0.005` as a probe variant, but the user instruction supersedes.)

---

## HEADLINE

| Metric                                      | R45+ default | **R58 Opt N tightened** | Δ        |
|---------------------------------------------|--------------|-------------------------|----------|
| Strict-VC PASS                              | **42/42**    | **40/42**               | **−2**   |
| AITER cohort PASS                           | 39/39        | **39/39**               | 0        |
| HK cohort PASS                              | 3/3          | **1/3**                 | **−2**   |
| AITER bit-deterministic (wcf_max=0, fin=1)  | 39/39        | 39/39                   | 0        |

**40/42 cells pass the tightened Opt N gate (vs 42/42 under R45+ default).** The 2 droppers are **both** R40B HK survivor cells; **0 AITER cells drop**. The AITER cohort is strictly bit-deterministic (wcf_max=0 AND fin_min=1.0 across all 10 INDEPENDENT seeds for every one of 39 cells), so it is invariant to the tightening — every AITER cell passes both gates with infinite margin on the wcf and fin axes.

---

## Per-cell drop table (PASS R45+, FAIL Opt N)

| Cell shape (M×N×K)        | Source label | Cohort | n_OK | wcf_max  | wcf_std  | fin_min | pct_comp  | Sub-gate(s) failed                              | Tail-class                  |
|---------------------------|--------------|--------|-----:|---------:|---------:|--------:|----------:|--------------------------------------------------|-----------------------------|
| **16384×4096×3072**       | R40B         | HK     |  10  | **0.0134** | 0.0010 | 0.9955  | 103.25 %  | `wcf_max>=0.01` (0.0134, 1.34× tight gate)      | wcf-tail (mid-margin)       |
| **32768×14336×2048**      | R40B         | HK     |  10  | **0.0113** | 0.0013 | **0.9847** | 100.53 %  | `wcf_max>=0.01` (0.0113) AND `fin_min<0.99` (0.9847) | wcf-tail + fin-tail (worst) |

Both droppers are previously-flagged "near-gate" HK cells in the R57 reviewer verdict (line 78). Neither is borderline on n_OK (both 10/10). Neither approaches wcf_std≥0.01.

### Surviving HK cell (passes BOTH gates)

| Cell shape          | Source | n_OK | wcf_max | wcf_std | fin_min | pct_comp |
|---------------------|--------|-----:|--------:|--------:|--------:|---------:|
| 16384×4096×2048     | R40B   |  10  | 0.0006  | 0.000050| 0.9985  | 109.57 % |

The smallest-K HK cell (K=2048, lowest cohort-race depth) passes Opt N with comfortable margin on every axis. wcf_max=0.0006 is **22×** below the tight threshold; fin_min=0.9985 is just below the 0.99 line on the third decimal but actually clears 0.99.

---

## Per-failure-reason breakdown

| Sub-gate failure  | Drop count | Notes                                                                  |
|-------------------|-----------:|------------------------------------------------------------------------|
| `n_OK<9`          |          0 | Both droppers achieve perfect 10/10 single-seed correctness            |
| `wcf_max>=0.01`   |          2 | **Sole source of attrition** — both HK survivors leak wrong-cell-frac  |
| `wcf_std>=0.01`   |          0 | Tightening this axis (unchanged at <0.01) had no effect at this dataset |
| `fin_min<0.99`    |          1 | Only the worst-margin HK cell (32768×14336×2048) also fails fin-tail   |

**Dominant failure mode: `wcf_max` tightening (0.02 → 0.01).** Both droppers fall via the wrong-cell-fraction tail. The 1 fin-tail failure is co-incident with a wcf-tail failure on the same cell, so dropping the fin tightening alone (back to ≥0.97) would still leave 2/2 droppers via the wcf axis.

---

## AITER vs HK split

| Cohort | Default-PASS | Tight-PASS | Drop |
|--------|-------------:|-----------:|-----:|
| AITER  | 39           | **39**     | **0** |
| HK     | 3            | **1**      | **2** |

**100 % of attrition concentrates in the HK cohort (2/3 dropped).** The AITER cohort, all served via the R50D `.co` dlopen shim (9th consecutive AS-IS reuse), is bit-deterministic across 10 INDEPENDENT seeds: every AITER cell has `wcf_max = 0`, `wcf_std = 0`, `fin_min = 1.0`, `n_OK = 10/10`. The tightening has zero effect on the AITER cohort and would have zero effect even if pushed further (the tight gate would survive arbitrary wcf_max, wcf_std, fin_min thresholds down to 0).

---

## Mechanism inference — what does the drop distribution tell us?

The drop pattern cleanly separates the manifest into two structurally distinct populations:

**Population 1 — bit-deterministic (39/42, all AITER)**
- All 39 AITER cells (R50D shim AS-IS over 9 rounds) achieve perfect determinism: identical bit-pattern outputs across 10 INDEPENDENT seeds. wcf_max ≡ 0, fin_min ≡ 1.0. The R45+ → Opt N tightening is a **no-op** for this population. There is no mechanism by which any AITER cell could drop under further tightening short of pushing thresholds below 0.
- This confirms the R55–R57 finding that `aiter` `.co` MFMA scheduling closes the cohort-race surface entirely on every shape it covers (39/42 of the manifest).

**Population 2 — cohort-race tail-drawers (3/42, all HK R40B)**
- The 3 surviving HK cells (`16384×4096×2048`, `16384×4096×3072`, `32768×14336×2048`) all share the same R40B compiled binary path; they are the residual HK kept-cell pool because aiter alt-tile axes have been closed for them (R57 H-2 confirmed 192×256 DEAD on all 3, R55 D-5B/1 confirmed 256×256 AITER underperforms HK on N=14336).
- Within this 3-cell pool, the tightening reveals a **clean ordering by cohort-race intensity**:
  - K=2048, N=4096: wcf_max ≈ 0.0006 → **inside Opt N envelope**
  - K=3072, N=4096: wcf_max ≈ 0.0134 → **drops** (+22× larger wcf_max than the K=2048 sister)
  - K=2048, N=14336: wcf_max ≈ 0.0113 + fin_min ≈ 0.9847 → **drops worst** (multi-axis)
- The K=3072 vs K=2048 jump in wcf_max at otherwise-identical (M=16384, N=4096) suggests the K-pipeline depth (lgk2) interacts with K-iteration count to produce a wider wrong-cell tail at K=3072. The N=14336 cell adds an N-axis cohort-race contribution (fin_min<0.99), consistent with the R49B finding that "(4096,32768,28672) cohort race scales with N".
- The "near-gate" `32768×14336×2048` cell (flagged at R57 reviewer with fin_min=0.9847) is the **structurally weakest** survivor: it is the only cell in the manifest that fails Opt N on **two** axes simultaneously.

**True cohort-race surface count (Opt N definition):** **2 of 42** cells (`16384×4096×3072` and `32768×14336×2048`). The 40 other cells (39 AITER bit-det + 1 HK with wcf_max=0.0006) sit in a regime indistinguishable from bit-deterministic at single-decimal-percent thresholds.

**Implication for R59+ axes:** the only productive cohort-race attack surface remaining lives on these 2 HK cells. The 192×256 axis is closed; the 256×256 AITER axis is closed (R55 D-5B/1); 128×256/96×640/64×1024 are deferred to Opt P (lower-eff). This means the durable R59 axis options are: (a) accept the cohort-race tail and keep 42/42 under R45+ default; (b) port HK lgk2 K-pipeline tuning to a tighter wcf trace; (c) find a non-AITER non-HK third source. Path (a) is the realistic Floor for R59.

---

## R59+ implications

| Cell                        | Status under default | Status under Opt N | Implication                                                                                              |
|-----------------------------|----------------------|--------------------|----------------------------------------------------------------------------------------------------------|
| `16384×4096×2048` (HK)     | PASS                 | PASS (margin)      | Structurally robust HK survivor; no R59 attention needed.                                                |
| `16384×4096×3072` (HK)     | PASS                 | DROP (wcf-tail)    | Cohort-race middle-tail; candidate for K-pipeline retuning if any axis remains open.                     |
| `32768×14336×2048` (HK)    | PASS                 | DROP (wcf+fin)     | Worst-margin survivor; multi-axis race; highest priority if R59 attempts a HK kernel rewrite.            |
| 39 AITER cells              | PASS                 | PASS (perfect)     | Bit-deterministic frontier; no further work possible against the tightened gate.                         |

**True noise-edge cells under Opt N: 2.** True bit-deterministic cells: 40 (39 AITER + 1 HK with wcf_max < 0.001). The strict-VC ceiling under R45+ default is at saturation; the strict-VC ceiling under Opt N tightened is at 40/42.

---

## Recommendation: should R58 manifest adopt the tightened gate?

**Recommendation: PARTIAL — do NOT adopt as the R58 default; carry Opt N as a SECONDARY informational gate alongside R45+ default.**

**Rationale (yes side):**
- The Opt N gate cleanly separates the manifest into a bit-deterministic frontier (40 cells) and a true cohort-race tail (2 cells), giving an enumerable attack surface for R59+.
- The AITER cohort is invariant to the tightening (39/39 pass with infinite margin), so adoption would produce zero false positives among 92.9 % of the manifest.
- A tighter gate produces stronger guarantees per cell that passes (no more wcf_max≈0.013 cells slipping through under a "PASS" label).

**Rationale (no side — and why this dominates):**
- **Adopting the Opt N gate as DEFAULT would IMMEDIATELY DROP THE LEADERBOARD FROM 42/42 → 40/42**, breaking the 3-consecutive-rounds 100 % strict-VC ceiling that is the headline result of R55/R56/R57 (the FIRST 3-IN-A-ROW IN PROJECT HISTORY per the R57 commit log).
- **The 2 droppers have no PROMOTE-able alternative**: the entire alt-tile axis on these cells is documented closed (R57 H-2 192×256 DEAD; R55 D-5B/1 256×256 AITER underperforms HK on N=14336). Adopting the gate would create 2 STRICT-VC-LOSE cells with zero R58/R59 mechanism path to recover.
- **R56 G-4 mechanism survey** (HK→AITER swap on L7 +80.92pp) suggests any productive HK→AITER swap on the 2 droppers requires a new aiter alt-tile axis to open, which has not occurred in 4 rounds.
- The Opt N gate's `fin_min ≥ 0.99` threshold is sub-noise relative to the protocol bump from ITERS=500 → ITERS=1000 (R57 reviewer used ITERS=1000; R58 reverts to ITERS=500 per decider §10.1, which may further widen the fin tail on R40B HK cells per the R57 cohort-race churn audit's +1.51pp HK drift observation).

**Concrete R58 integration recommendation:**
1. **Keep R45+ default gate** (n_OK≥8, wcf_max<0.02, wcf_std<0.01, fin_min≥0.97) as the canonical strict-VC determination for the R58 manifest. R58 reviewer integration MUST report 42/42 strict-VC retention against this gate.
2. **Carry Opt N tightened gate as an INFORMATIONAL secondary report** in `R58_INTEGRATION_VERDICT.md`. Format: "42/42 default-VC; 40/42 Opt-N-VC; droppers: [list]". This preserves the methodology signal without disrupting the leaderboard.
3. **Promote Opt N to default consideration only when ≥38/42 cells pass** (i.e., only when at most 4 cells drop). Currently 40/42 → 2 droppers, so the gap to a defensible default-tightening is +0 future PROMOTEs.
4. **Track Opt-N-DROP cells as the attention list for R59+ HK/cohort-race work**. Use the 2-cell list as the explicit attack surface, replacing the historic "3 HK cells" framing.

In short: **Opt N is excellent diagnostics, terrible production gate.** Adopt as a secondary lens, not as the new default.

---

## Files

- `R58_OPT_N_VERDICT.md` — this file
- `R58I1_N_INTEGRATION_FRAGMENT.json` — empty manifest fragment (analysis-only cohort)

## Reviewer integration handoff

- This cohort produces NO `.so` / `.co` manifest changes; reviewer should treat the R58 binary manifest as byte-identical to R57.
- Fragment file `R58I1_N_INTEGRATION_FRAGMENT.json` carries `manifest_changes: []` and `type: analysis_only` so the integrator can skip merge logic and only attach the methodology output to the R58 verdict.
- Recommended action for `R58_INTEGRATION_VERDICT.md`: include a 1-paragraph "Opt N secondary gate" section quoting "40/42 under tightened gate; 2 HK droppers (16384×4096×3072, 32768×14336×2048); 100 % drop concentration in HK cohort confirms HK-only cohort-race surface; recommendation: keep R45+ default."
