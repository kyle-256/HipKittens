# R43 Opt C — VERDICT: DEAD (no-promote)

**Date**: 2026-04-19
**Worker**: Opt C (perf claw-back on 14 VC_CLAWBACK shapes)
**GPUs**: 5, 6, 7
**Result**: 0/14 shapes promoted under strict gate. P-C.1 + P-C.2 + P-C.3 all unmet.

---

## Hypothesis

R41B's coarse retune REJECTED some candidate variants under the old 0.99-gate. With FINITE_GATE relaxed to 0.98 (R42 default), some of those — plus other unexplored cells in R40A/R40B/R41A/R41B — might claw back perf on the 14 VC_CLAWBACK shapes (verified-correct under 0.98 but pct_comp < 95%).

## Falsifiable predictions (all unmet)

- **P-C.1**: ≥3 shapes flip to WIN_VC (≥100% comp). **UNMET** — 0 promotions.
- **P-C.2**: ≥6 shapes improve perf by ≥5% while remaining PASS_VC. **UNMET**.
- **P-C.3** (per-shape gate: `n_OK_5 ≥ 4 AND wcf_std < 0.005 AND fin_min ≥ 0.985 AND tflops_p50 ≥ 1.05 × current`). **UNMET** for all 5 5-run candidates.

## Methodology

### Phase 0 — inventory
`R43_OPT_C/inventory_candidates.py`. Globbed each shape's `(N,K)` across `build_R40A/R40B/R41A/R41B`. Result: 147 candidates over 14 shapes (2-15 per shape). See `R43_OPT_C/candidate_inventory.json`.

### Phase 1 — smoke (1-rep at warmup=200/iters=500/trim=0.10)
`R43_OPT_C/smoke_sweep.py`. 147 jobs round-robin across GPUs 5,6,7. Elapsed 6.2 min.
Filter: `finite_frac >= 0.98 AND wrong_cell_frac < 0.02 AND tflops > current × 1.03`.

Promotable to 5-run: **5 candidates over 2 shapes** (`14336x32768x4096`: 2 cands; `4096x32768x4096`: 3 cands). The other 12 shapes had no candidate beating current by +3% under the 0.98/0.02 finite/wcf gates. See `R43_OPT_C_SWEEP_SMOKE.{json,log}`.

### Phase 2 — 5-run consensus
`R43_OPT_C/run_5run.py` (top-K=3 promoted per shape). 25 jobs (5 cands × 5 reps) on GPUs 5,6,7. Elapsed 0.8 min. Per-candidate consensus + strict promote gate. See `R43_OPT_C_5RUN.{json,log}`.

| Shape | Candidate | n_OK_5 | tflops_p50 | Δ vs cur | wcf_max | wcf_std | fin_min | Pass? |
|---|---|---|---|---|---|---|---|---|
| 14336x32768x4096 (cur 3850) | R40A `lgk2_v24_PF_FENCE1` | 4/5 | 3976.0 | +3.3% | 0.0239 | 0.0073 | 0.984 | NO (Δ<+5%, wcf_max>0.02) |
| 14336x32768x4096 (cur 3850) | R40A `lgk2_gm7_v12_memc_pfoff14_R38B_TAIL_FIX1_PF_FENCE1` | 5/5 | 3916.4 | +1.7% | 0.0139 | 0.0038 | 0.980 | NO (Δ<+5%, fin_min<0.985) |
| 4096x32768x4096 (cur 3822) | R40B `gm7_v12_dc_pfoff14_safe` | 2/5 | 3981.1 | +4.2% | 0.0214 | 0.0067 | 0.972 | NO (n_OK<4, fin_min<0.985) |
| 4096x32768x4096 (cur 3822) | R40B `lgk2_gm7_v12_pfoff14_safe` | 3/5 | 3969.4 | +3.9% | 0.0101 | 0.0022 | 0.979 | NO (n_OK<4, fin_min<0.985) |
| 4096x32768x4096 (cur 3822) | R41B `lgk2_gm7_v12_pfoff14_v2` | 3/5 | 3952.9 | +0.0%* | 0.0113 | 0.0027 | 0.975 | NO (n_OK<4, fin_min<0.985) |

*Note: tflops_p50 +3.4% but Δ shown vs cur. All 5 candidates fail the +5% perf gate AND/OR the n_OK_5≥4 stability gate.

### Phase 3 — regression probe
**Skipped**: zero promotions, so no risk of regressing the 27 already-VC shapes.

---

## Findings

1. **The variant table is near-optimal under FINITE_GATE=0.98**: 12/14 VC_CLAWBACK shapes have NO candidate that even beats current by +3% on a single rep. This is despite the 0.98 gate being more permissive than R41B's 0.99.
2. **The 2 shapes that DO have positive smoke candidates are both `N=32768, K=4096`**: the candidates are R40A/R40B `pfoff14` (k_iters=14 with offset=14 → effectively pfoff=0 on the tail), suggesting deep-K-tail prefetch tuning matters here. But variance under 5-run kills them: wrong_cell_frac drifts past 0.02 on 1-3 of 5 runs (the cohort race observed in `project_mxfp4_finite_gate_cohort_race.md`). The +5% perf gate is strict and the candidates are only +3.3% to +4.2%.
3. **The strict R43 promote gate (n_OK_5≥4 AND wcf_std<0.005 AND fin_min≥0.985 AND tflops≥+5%) was correctly designed to prevent the R41B regression mode** (R41B's loose gate let 2 candidates promote that immediately regressed). Holding it costs 0 promotions but also costs 0 regressions to baseline.

## What this means for R43 round

- Net new VC: **+0** from Opt C.
- Net perf gain: **+0** from Opt C.
- The 14 VC_CLAWBACK shapes remain at their R42 baseline values (60.5% to 94.6% comp).

## Possible next-round directions (deferred to R44)

The variant-axis is exhausted under existing macros for this cohort. To break the 14-shape ceiling, future rounds need:

1. **Kernel-axis** changes (new prefetch state machine, new MFMA cohort layout). The `project_mxfp4_aiter_binary_disasm.md` memo notes aiter has a different prefetch state machine — disasm-driven cloning is a candidate.
2. **Cohort-race fix at root** (project_mxfp4_finite_gate_cohort_race.md): closing the random-position MFMA accumulator race would tighten wcf_std and let candidates above pass the 0.985 fin_min gate. Opt B (R34 VGPR-PF + `+v` keepalive) is the in-flight attempt.
3. **Per-shape macro stack expansion** (forbidden in this round): `R44_PFOFF_TAIL_HIGHRES` sweep, `R44_TAIL_BTW_VARIANT`, `R44_LGKMCNT_AT_KBOUND`.

---

## Hard rules adherence (recap)

- warmup=200, iters=500, trim_frac=0.10: **YES** (in both smoke and 5-run harnesses).
- FINITE_GATE=0.98: **YES**.
- 5-run consensus on all final decisions: **YES**.
- No new kernel macros: **YES** (only existing R40A/R40B/R41A/R41B `.so` files).
- GPUs 5/6/7 only: **YES** (`HIP_VISIBLE_DEVICES` per worker thread).
- No regressions on the 27 already-VC shapes: **TRIVIAL** (no promotions).

## Deliverables

- `R43_OPT_C_VERDICT.md` (this file)
- `R43_OPT_C/inventory_candidates.py`, `R43_OPT_C/candidate_inventory.json`
- `R43_OPT_C/smoke_sweep.py`, `R43_OPT_C_SWEEP_SMOKE.{json,log}`
- `R43_OPT_C/run_5run.py`, `R43_OPT_C_5RUN.{json,log}`
- `R43C_BUILD_MANIFEST.json` (empty: no new builds)
- `R43C_INTEGRATION_FRAGMENT.json` (empty: no manifest changes)
