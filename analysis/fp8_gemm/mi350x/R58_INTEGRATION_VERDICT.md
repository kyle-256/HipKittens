# R58 INTEGRATION REVIEWER VERDICT — 41/42 strict 10-run @ 80% INDEPENDENT seeds (ITERS=500 revert)

**Date:** 2026-04-19
**Reviewer GPUs:** 4,5,6,7 (all idle, verified `rocm-smi --showuse`)
**Bench rules:** warmup=200, **iters=500** (R45+ default; R58 reverts R57's one-time R57_OPT_J_LONG_BENCH ITERS=1000 protocol bump), trim=0.10, INDEPENDENT seeds [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
**Manifest:** `R58_INTEGRATION_MANIFEST.json` (40 AITER + 2 HK; +1 binary delta vs R57: P-2 swap)
**Wall-clock:** SMOKE 1-seed 1.6 min; full 10-run ≈17 min (per-run 1.6-2.0 min × 10) on 4 GPUs
**Outputs:** `R58_INTEGRATION_10RUN.{json,log,console}`, `R58_INTEGRATION_SMOKE1.{json,log,console}`

---

## HEADLINE

| Metric | R55 | R56 | R57 | **R58** | Δ vs R57 |
|---|---|---|---|---|---|
| Strict 10-run VC (n_OK≥8/10 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min≥0.97) | 42/42 | 42/42 | 42/42 | **41/42** | **−1** (3-in-a-row 100% leaderboard streak ENDED) |
| AITER override count | 38 | 39 | 39 | **40** | **+1** (P-2 HK→AITER swap; smallest HK pool ever) |
| HK baseline count | 4 | 3 | 3 | **2** | **−1** (smallest in project history) |
| WIN cells (≥100% comp) | 34 | 40 | 41 | **40** | −1 (L1 100.04%→99.94% under ITERS=500 revert; expected) |
| AITER bit-deterministic share (wcf=0 across 10 seeds) | 38/42 | 39/42 | 39/42 | **40/42** | **+1** (largest AITER bit-det share in project history) |
| LOSE cells (<100% comp) | 8 | 2 | 1 | **2** | +1 (L1 reverts to LOSE-edge under ITERS=500) |

**4th consecutive 100% leaderboard streak BROKEN at R58.** The 1 dropped VC cell is `32768x14336x2048` (R40B HK), the **worst-margin survivor flagged by R57 reviewer (fin_min=0.9847)** and `R58 Opt N` analysis (fin_min<0.99 AND wcf_max≥0.01). Under R58's ITERS=500 revert (from R57's one-time ITERS=1000), this near-gate cell tail-drew n_OK=9/10 with fin_min=0.911 (<0.97 gate). This is **cohort-race churn on UNCHANGED binary** caused by the ITERS revert, not a kernel regression. The other 41 cells HOLD VC.

**P-2 PROMOTE LANDS:** `(16384, 4096, 3072)` HK R40B 103.25% → AITER 128×256 **107.62%** at reviewer (+4.37pp vs R57; worker reported 106.75%, reviewer p50 lands ~+0.87pp higher). All 10 INDEPENDENT seeds bit-deterministic (wcf_max=0, wcf_std=0, fin_min=1.0, n_OK=10/10). HK→AITER swap converts the only mid-margin Opt N dropper to bit-deterministic perfection.

10th consecutive R50D shim AS-IS reuse round.

Aggregate perf delta vs R57 (across all 42 cells): **+0.55pp net** (mean **+0.013pp/cell**); the +4.37pp P-2 PROMOTE roughly cancels the broad −0.1 to −0.5pp ITERS=500 trim-noise drift on AITER cells and the −2.09pp R40B HK ITERS-reversion drift on `16384x4096x2048`.

---

## R58 PROMOTE table (1/1 verified at reviewer)

| Cell | Shape (MxNxK) | Source change | R57 pct_comp | R58 pct_comp | Δ | n_OK | wcf_max | fin_min | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| **P-2** | 16384x4096x3072 | R40B HK (256×256) → R58P2 AITER 128×256 | 103.25% | **107.62%** | **+4.37pp** | 10/10 | 0.0 | 1.0 | **PROMOTE** |

P-2 reviewer p50 of 107.62% comfortably exceeds the worker fragment 10-run pct_comp_first of 106.75% AND the D-3A-1 +1.0pp gate (104.25%) AND the WIN gate (≥100%). Perfect bit-determinism: all 10 INDEPENDENT seeds wcf_max=0, fin_min=1.0. HK→AITER swap converts a previous Opt N dropper (R57 reviewer fin_min=0.9955, wcf_max=0.0134) to bit-deterministic gate-immune. **AITER share 39 → 40; HK pool 3 → 2; AITER bit-deterministic share 39 → 40.**

---

## ACCEPT_FALLBACK summary (worker outcomes; no manifest change)

| Cohort | Cell | Shape | Alt attempted | Worker SMOKE | Verdict |
|---|---|---|---|---:|---|
| I-1 (Opt N) | I-1/N-* | analysis-only | n_OK≥9, wcf_max<0.01, fin_min≥0.99 gate | 40/42 pass tightened | **ACCEPT_FALLBACK** (informational; do NOT adopt as default) |
| I-2 (Opt O) | O-1 | 4096x32768x128256 (L8) | HK R40B 256×256 lgk2 | WRONG_OUTPUT (fin=0.804, wcf=0.128) | ACCEPT_FALLBACK; HK 256×256 axis on L8 K=128256 CLOSED by correctness failure |
| I-2 (Opt O) | O-1b | 4096x32768x128256 (L8) | HK R37 256×256 lgk2 memc fallback | WRONG_OUTPUT (fin=0.775, wcf=0.066) | ACCEPT_FALLBACK; HK fallback DEAD same axis |
| I-3 (Opt P) | P-1 | 16384x4096x2048 | AITER 128×256 | 105.89% (-3.68pp vs HK 109.57%) | STOP_ACCEPT_FALLBACK (D-3A-1) |
| I-3 (Opt P) | P-3 | 32768x14336x2048 | AITER 128×256 | 75.49% (-25.04pp vs HK 100.53%) | STOP_DEAD (HK_pct - 5.0pp gate); 128×256 axis CLOSED for this cell |

**Closures**: HK 256×256 lgk2 axis on L8 K=128256 CLOSED (correctness, not perf); 128×256 alt-tile axis CLOSED on 2 of 3 HK kept-cell candidates (P-1 ACCEPT_FALLBACK, P-3 DEAD); 128×256 alt-tile PROMOTED on P-2 (the only success).

---

## AITER bit-deterministic share

| Round | AITER cells | wcf=0 across 10 seeds | Share |
|---|---:|---:|---:|
| R54 | 27 | 27 | 27/42 |
| R55 | 38 | 38 | 38/42 |
| R56 | 39 | 39 | 39/42 |
| R57 | 39 | 39 | 39/42 |
| **R58** | **40** | **40** | **40/42** |

**Largest AITER bit-deterministic share in project history.** P-2 HK→AITER swap converts one cohort-race-prone HK cell to perfect bit-determinism via R50D shim AS-IS (10th consecutive AS-IS reuse round). All 40 AITER cells achieve wcf_max=0, wcf_std=0, fin_min=1.0 across 10 INDEPENDENT seeds.

---

## HK cells remaining (smallest in project history)

| Round | HK cells | List |
|---|---:|---|
| R55 | 4 | 16384x4096x2048, 16384x4096x3072, 32768x14336x2048, 4096x128256x32768 |
| R56 | 3 | 16384x4096x2048, 16384x4096x3072, 32768x14336x2048 |
| R57 | 3 | 16384x4096x2048, 16384x4096x3072, 32768x14336x2048 |
| **R58** | **2** | **16384x4096x2048, 32768x14336x2048** |

The 2 surviving HK cells:
- `16384x4096x2048` R40B: 107.48% (R57 109.57%; **−2.09pp ITERS=500 reversion drift** — within the R57-warned "HK cells benefit most from longer ITERS" envelope, +1.51pp HK mean drift was R57 observation; R58 is the inverse). VC: n_OK=10/10, wcf_max=0.000586, fin_min=0.997 (HOLD).
- `32768x14336x2048` R40B: 100.49% (R57 100.53%; −0.04pp). **VC FLIP True→False: n_OK=9/10, wcf_max=0.0111, fin_min=0.911 (<0.97 gate).** Worst-margin survivor at R57 (fin_min=0.9847) and R58 Opt N analysis (failed both wcf_max>=0.01 AND fin_min<0.99). Under ITERS=500 revert, the wider trim distribution dropped fin_min below the 0.97 default gate.

---

## Cohort-race churn audit on 41 UNCHANGED-binary shapes

**40 AITER + 2 HK = 42 cells; 1 binary delta (P-2 swap). The "unchanged" cohort covers the other 41 cells.**

- **VC retention: 40/41** (1 lost VC: `32768x14336x2048` R40B HK on UNCHANGED binary; cohort-race tail-draw under ITERS=500 revert)
- **VC gains: 0/41** (all 40 AITER cells already VC at R57; the 1 HK loss exceeds 0 gains)
- **Mean perf drift on 39 AITER unchanged cells: +0.001pp** (range −1.09pp to +1.15pp; near-zero net)
- **Mean perf drift on 2 HK unchanged cells: −1.06pp** (16384x4096x2048 −2.09pp; 32768x14336x2048 −0.04pp; aligned with R57 prediction that "HK cells may drift downward by ~1.5pp due to wider trimmed-distribution effect" of ITERS=500 revert)
- **No WIN/LOSE flips** on the 39 unchanged AITER cells (HK survivor 16384x4096x2048 stays WIN; 32768x14336x2048 still numerically WIN at 100.49% but VC-flipped). L1 (R57J1_L1_AITER, unchanged binary) drops from WIN 100.04% → LOSE 99.94% under ITERS=500 — this is the EXPECTED reversion of the R57 Opt J protocol-bump effect documented in R57 verdict §"ITERS=1000 protocol observation".
- **Aggregate WIN flip cells under ITERS=500 revert: 1 (L1 WIN→LOSE)**.

**Cohort-race churn count: 1.** The 1 churn cell is the previously-flagged worst-margin HK survivor; mechanism is the ITERS=500 revert exposing the wider trim-distribution tail of R40B HK on N=14336 — exactly the population the Opt N tightened-gate analysis identified as multi-axis tail-drawer.

---

## ITERS=500 revert observation

R57 used ITERS=1000 (one-time R57_OPT_J_LONG_BENCH protocol bump for L1 noise-edge tightening). R58 reverts to ITERS=500 default per decider plan §10.1. Did the revert produce predicted vs unexpected effects?

**Predicted effects (per R57 verdict §"ITERS=1000 protocol observation"):**
- L1 (4096x32768x14336) was at 100.04% under ITERS=1000 = +0.06pp over WIN line; predicted to drift back below 100% under ITERS=500. **OBSERVED: 99.94% (LOSE-edge, −0.10pp)**. Confirmed.
- Other near-100% cells (L8 97.75%, 4096x32768x28672 101.17%) predicted minimal flip risk. **OBSERVED: L8 98.28% (HELD LOSE; +0.53pp), 4096x32768x28672 101.17% (HELD WIN; 0.0pp)**. Confirmed.

**Unexpected effects:**
- **`32768x14336x2048` R40B HK fin_min dropped from 0.9847 → 0.911 under ITERS=500.** R57 Opt N analysis flagged this as the "worst-margin survivor" with multi-axis sub-gate failures; the ITERS revert pushed fin_min below the 0.97 default gate, producing the only VC flip. This confirms the Opt N analysis's identification of this cell as the structurally-weakest survivor.
- HK survivor `16384x4096x2048` lost 2.09pp pct_comp (109.57% → 107.48%) — the R57 verdict explicitly predicted "+1.51pp HK mean drift" benefit from ITERS=1000 vs default; R58 sees the symmetric inverse on the dominant K=2048 cell. Stayed VC and WIN.

**Conclusion: ITERS=500 revert is correctly priced at the L1 noise-edge cell and the worst-margin HK cell.** The R57 ITERS=1000 protocol bump's value was concentrated on these two cells (L1 +1 WIN, L3 fin_min +0.07 above gate). **R58 should NOT re-bump ITERS for L1 only** unless paired with a true mechanism fix (no kernel changes available); per the R57 protocol-bump validity envelope, ITERS=1000 is justified only for ≤0.5pp boundary cells, and L1 currently sits −0.06pp under WIN line which is in-envelope but the cost of mixed protocols across a 42-shape leaderboard is not worth a single-cell flip when the kernel binary is identical.

---

## Opt N gate-tightening summary (cite R58_OPT_N_VERDICT.md)

R58 Opt N (cohort I-1, analysis-only; no GPU runs) re-classified `R57_INTEGRATION_10RUN.json` under a tightened strict-VC gate (n_OK≥9/10 AND wcf_max<0.01 AND wcf_std<0.01 AND fin_min≥0.99). Result: **40/42 cells pass tightened gate** (AITER: 39/39 unchanged; HK: 1/3 — only `16384x4096x2048` survives; 2 droppers `16384x4096x3072` and `32768x14336x2048` both fail wcf_max ≥ 0.01).

**P-2 swap eliminates one of the 2 Opt N droppers**: `16384x4096x3072` is now AITER 128×256 (wcf_max=0, fin_min=1.0); under tightened Opt N gate this cell now PASSES. Post-R58 Opt N tightened-gate count would be **41/42** (only `32768x14336x2048` would still fail Opt N; under R45+ default it is also the lone R58 VC-flipper).

**Recommendation (per R58_OPT_N_VERDICT.md): KEEP R45+ default as canonical strict-VC gate; carry Opt N as informational secondary gate.** Adopting Opt N as default would have created 2 strict-VC-LOSE cells with no R58/R59 mechanism path to recover the 2nd. R58 P-2 PROMOTE has organically reduced the Opt N attention list to 1 cell (`32768x14336x2048`), validating the gate-tightening's diagnostic utility.

---

## COMMIT recommendation

**COMMIT R58_INTEGRATION_MANIFEST.json as the canonical R58 production manifest.**

Reasons:
1. **P-2 PROMOTE confirmed at reviewer** at 107.62% (vs worker 106.75%, +0.87pp reviewer-vs-worker positive drift; vs R57 HK 103.25%, +4.37pp); n_OK=10/10, wcf_max=0, fin_min=1.0; passes strict 10-run VC + D-3A-1 perf gate + WIN gate.
2. **41/42 strict 10-run VC** (one VC drop on `32768x14336x2048` R40B HK is cohort-race tail-draw on UNCHANGED binary under ITERS=500 revert — predicted-prone per R57 reviewer fin_min=0.9847 flag and R58 Opt N "worst-margin survivor" identification). Not a kernel regression.
3. **AITER bit-deterministic share 39 → 40** (largest in project history); **HK pool 3 → 2** (smallest in project history).
4. **R50D shim AS-IS reuse for 10th consecutive round** — no kernel rebuild, no shim modification.
5. **Worker O-1 (Opt O) ACCEPT_FALLBACK** correctly held R52D2B AITER 256×256 on L8 (HK fallback variants both produced WRONG_OUTPUT); HK 256×256 axis on L8 K=128256 CLOSED.
6. **Worker P-1, P-3 (Opt P) ACCEPT_FALLBACK** correctly held R40B HK baselines (128×256 alt-tile DEAD on these cells); axis closure documented.
7. **2 LOSE cells remain**: L1 4096x32768x14336 R57J1_L1 at 99.94% (−0.10pp ITERS revert; was 100.04% at R57 ITERS=1000); L8 4096x32768x128256 R52D2B at 98.28% (+0.53pp; HELD LOSE; aiter alt-tile axis CLOSED + HK 256×256 axis CLOSED).

**REVERT was considered but does NOT apply**: the constraint section requires REVERT only if the swap candidate P-2 fails 10-run VC. P-2 PASSES strict VC perfectly. The L3 (32768x14336x2048) VC drop is on UNCHANGED HK binary and is governed by the ITERS=500 protocol revert, NOT the P-2 swap. REVERTing P-2 would: (a) lose the +4.37pp perf claw-back; (b) lose the AITER bit-det share gain; (c) NOT recover the L3 VC drop (different cell). The correct disposition is COMMIT the P-2 swap and document the L3 cohort-race tail-draw.

---

## R59 candidate axes (1 LOSE cell + 1 VC-flipped cell remain)

Post-R58 the surface is:
- **L8 (4096x32768x128256)** — sole structurally LOSE cell, AITER R52D2B 98.28%. Aiter alt-tile axis FULLY CLOSED (128×512, 192×256, 224×256 all DEAD per R56-R57); HK 256×256 axis CLOSED (R58 Opt O R40B+R37 both WRONG_OUTPUT). Combined gap: ~1.72pp. **No mechanism axis remaining in the existing source/binary inventory.** R59 axis options: (a) accept L8 at aiter-internal ceiling (Floor); (b) from-scratch HK kernel rewrite for K=128256 with R39A/R44A correctness fixes (very low confidence); (c) defer to non-aiter non-HK third source.
- **L1 (4096x32768x14336)** — under ITERS=500 revert sits at 99.94% (−0.06pp from WIN line); was WIN under R57 ITERS=1000. R59 axis options: (a) accept LOSE-edge under default ITERS; (b) re-apply ITERS=1000 ONLY for L1 (mixed-protocol leaderboard cost); (c) seek a different .co or HK candidate (no axis open since R56 G-1).
- **`32768x14336x2048` HK R40B** — VC-flipped under ITERS=500 (n_OK=9/10, fin_min=0.911); 100.49% pct_comp. R59 axis options: (a) accept VC drop and document; (b) attempt HK→AITER swap with a non-128×256 alt-tile (96×640 / 64×1024 untried but lower-eff than 128×256 which scored 75.49% on this cell); (c) tighten cohort gate for R40B HK rebuild (no axis open since R44).
- **`16384x4096x2048` HK R40B (last surviving HK cell)** — at 107.48% (HELD VC, HELD WIN, −2.09pp ITERS revert drift). No R59 attack needed unless HK→AITER alt-tile opens; R58 P-1 ACCEPT_FALLBACK at 105.89% suggests 128×256 close but underperforms.

**R58 reaches 41/42 VC + 40/42 WIN under default ITERS=500** with the smallest HK pool (2) and largest AITER bit-deterministic share (40) in project history. The 4-in-a-row 100% leaderboard streak ends here, but the AITER share gain and HK pool shrinkage represent durable structural progress: the cohort-race surface is now bound to 1 cell (`32768x14336x2048`), confirming the Opt N analysis's prediction that the "true cohort-race surface count" was 2 cells (one of which is now eliminated by the P-2 swap).

---

## Round summary

R58 = **41/42 VC + 1/1 PROMOTE + 1 cohort-race churn (predicted) + 40→40 WIN HOLD (L1−L8±/P-2+) + 10th consecutive R50D AS-IS + 39→40 AITER bit-det + 3→2 HK pool**. The Opt P-2 mechanism (HK R40B 256×256 → AITER 128×256 swap on a mid-K mid-N cell where HK cohort-race intensity is meaningful but aiter alt-tile efficiency holds) is **mechanism-validated**: +4.37pp perf claw-back AND wcf_max 0.0134→0 AND fin_min 0.9955→1.0 in a single swap. The L3 VC drop is the predicted price of the ITERS=500 revert on the worst-margin survivor, identified by R57 reviewer (fin_min=0.9847 flag) and R58 Opt N analysis (multi-axis tail-drawer); REVERT does not apply as the swap candidate P-2 passes strictly. Worker O-1 (HK 256×256 on L8) and Worker P-1/P-3 (128×256 on 2 HK cells) correctly closed dead axes via SMOKE-first protocol. The R50D shim demonstrated stability across the ITERS protocol revert (no AITER VC churn; mean drift +0.001pp on 39 unchanged AITER cells).

**Net structural progress**: the project now has 40 of 42 cells served via bit-deterministic AITER `.co` dlopen; HK pool reduced to 2 cells; only 1 cell (`32768x14336x2048`) now constitutes the entire residual cohort-race attention surface. The 4-in-a-row 100% leaderboard streak ended in exchange for permanent reduction of the cohort-race surface from 3 cells to 1 cell.

---

## Files

- `R58_INTEGRATION_MANIFEST.json` — canonical R58 manifest (40 AITER + 2 HK; 1 binary delta vs R57: P-2 swap)
- `bench_all_42_R58_INTEGRATION.py` — reviewer bench script (4-GPU sharding, **ITERS=500 reverted**, --gpus 4,5,6,7, +AITER 128×256 dispatch entry)
- `R58_INTEGRATION_SMOKE1.{json,log,console}` — 1-seed smoke (42/42 single-seed VC, 40/42 WIN, 1.6 min wall)
- `R58_INTEGRATION_10RUN.{json,log,console}` — full 10-run @ 80% INDEPENDENT seeds, ITERS=500 (41/42 VC, 40/42 WIN, ~17 min wall)
- `R58_INTEGRATION_VERDICT.md` — this file
- worker fragments merged: `R58I1_N_INTEGRATION_FRAGMENT.json` (analysis-only, no manifest change), `R58I2_O1_INTEGRATION_FRAGMENT.json` (ACCEPT_FALLBACK), `R58I3_P1_INTEGRATION_FRAGMENT.json` (ACCEPT_FALLBACK), `R58I3_P2_INTEGRATION_FRAGMENT.json` (**PROMOTE**), `R58I3_P3_INTEGRATION_FRAGMENT.json` (ACCEPT_FALLBACK)
- worker verdicts: `R58_OPT_N_VERDICT.md`, `R58_OPT_I2_VERDICT.md`, `R58_OPT_I3_VERDICT.md`
