# R60 DECIDER PLAN — Opt U (documentation pivot) + Opt Y (cohort-race surface monitoring)

**Date:** 2026-04-20
**Round:** R60 (post-R59)
**Decider scope:** Methodology / closure round. No new kernel work, no new .co binaries, no shim rebuilds. Manifest is byte-identical to R59 (which was byte-identical to R58 binary entries).
**GPUs available:** 8 (verified idle pre-launch via `rocm-smi --showuse`); R60 uses GPUs 4-7 only.
**R50D AS-IS reuse counter:** R60 is the **12th consecutive round** of R50D shim AS-IS reuse (no rebuild expected).

---

## 1. Round axis selection rationale

**Selected axes:** **Opt U (documentation pivot) + Opt Y (cohort-race surface monitoring)**.

This is the explicit recommendation of `R59_INTEGRATION_VERDICT.md` §"R60+ direction suggestions": **"R60 recommended axis selection: Opt U (documentation pivot) + Opt Y (single-shot cohort-race re-measurement). Total R60 wall ≈ 30 min."** The R59 recommendation is endorsed and not overridden.

Defense of the choice (against the alternatives):

| Alternative | Why rejected for R60 |
|---|---|
| **Opt T** (L8 from-scratch HK build) | ~3 R-rounds wall, very low confidence (R58 Opt O confirmed BOTH R40B and R37 fallbacks WRONG_OUTPUT for K=128256 because R39A/R44A/R44D fixes were never ported). Per R59 verdict: "Defer again unless explicit user election." No user election received. |
| **Opt W** (HK kernel rebuild, FINITE_GATE 0.97→0.95) | Breaks the 11-round R50D AS-IS streak. R59 explicitly states this option is "only justified if Opt U is rejected AND Opt Y shows L3 repeats." Opt U has not been rejected and Opt Y has not yet run for R60. |
| **Opt X** (L1 unprobed alt-tiles or grid swizzle) | L1 alt-tile space is **EXHAUSTED** per R59 (96×640 + 64×1024 closed in R59; 256×256 R57J1_L1 is best AITER tile). Per R59 verdict: "Not recommended." |
| Fresh axis (e.g. K-pipeline lgkmcnt sweep on L8) | Every reasonable bounded-cost axis on the 3 attention cells (L1, L3, L8) has been closed in R45-R59. Proposing a fresh axis here would require a mechanism hypothesis with no supporting evidence — actively prohibited by AGENT_PROMPT round-discipline. |

Positive justification for **Opt U + Opt Y**:

1. **All bounded-cost axes for the 3 attention cells are closed.** L1 alt-tile EXHAUSTED, L3 alt-tile EXHAUSTED, L8 AITER alt-tile FULLY CLOSED, L8 HK 256×256 CLOSED by correctness, L1 ITERS=1000 one-off bump CLOSED by Opt R policy. There is no remaining bounded-cost axis to attack.
2. **Opt Y is cheap (~17 min wall, 0 risk).** It runs the EXISTING R59 manifest under a fresh independent seed sweep. Result PASS or FAIL is informative either way — it answers the cohort-race-repeatability question that scopes R61-R65 axis selection.
3. **Opt U formalizes the SC/MICRO publication pivot.** The project has reached the structural ceiling reachable via the R50D shim: 42/42 strict VC + 41/42 WIN, AITER bit-deterministic share at the largest in project history (40/42), HK pool at the smallest in project history (2). The remaining 1 LOSE cell (L8) sits at the aiter-internal ceiling. Documenting this state — what is achieved, what residuals remain, what publication claims are supported — is the highest-value next deliverable.
4. **Empirical record from the last 4 rounds supports a methodology round here.** R56 was a perf claw-back round (+6 NET WIN cells). R57 was a noise-edge WIN (+1 WIN cell via Opt J). R58 was a structural WIN (+1 PROMOTE HK→AITER swap; cohort-race surface 3→1 cell). R59 was a recovery + axis-closure round (+1 VC RECOVERY, +1 WIN RECOVERY, 4 alt-tile axes closed, Opt R policy artifact). The natural successor is a methodology + monitoring round to document the ceiling and confirm cohort-race repeatability across a 3rd consecutive sweep on L3.

**Round expected outcome:** 0 PROMOTE / 0 SMOKE_DEAD / 2 POLICY_ONLY artifacts (Opt U doc + Opt Y measurement). Manifest byte-identical to R59. Net structural delta = +1 documentation artifact + 1 cohort-race repeatability data point + 12th consecutive R50D AS-IS reuse round.

---

## 2. Per-cohort plan

R60 has **2 worker cohorts** (smaller than typical because the round is methodology, not perf attack). Both cohorts are POLICY_ONLY at the worker layer (no PROMOTE candidates).

### Cohort K-1 (Opt Y) — Cohort-race surface monitoring re-bench

- **Cohort ID:** K-1
- **Worker:** A (GPUs 4-7)
- **Scope:** Re-bench `R59_INTEGRATION_MANIFEST.json` AS-IS under a fresh INDEPENDENT seed sweep. **No binary modifications, no new .co files, no shim rebuilds, no manifest changes.** This cohort's "worker" output IS the reviewer integration measurement; there is no separate SMOKE step (manifest is already known-correct from R59).
- **Manifest used:** `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R59_INTEGRATION_MANIFEST.json` (40 AITER + 2 HK; byte-identical to R58 binary entries).
- **Bench script:** Reuse `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/bench_all_42_R59_INTEGRATION.py` AS-IS, OR create `bench_all_42_R60_INTEGRATION.py` that imports the R59 manifest and only differs in the `--seeds` argument. Strongly prefer reuse: the manifest is unchanged so the script is unchanged.
- **GPU assignment:** GPUs **4, 5, 6, 7** (4-way sharding pattern established R56-R59).
- **Bench parameters (MANDATORY per `.claude/rules/benchmark-rules.md` and Opt R policy A):**
  - `WARMUP = 200`
  - `ITERS = 500` (R45+ default; per Opt R policy A — no mixed-protocol bump)
  - `TRIM_FRAC = 0.10`
  - 10-run @ 80% protocol (n_OK ≥ 8/10 AND wcf_max < 0.02 AND wcf_std < 0.01 AND fin_min ≥ 0.97)
- **Seed set (DIFFERENT from R59 [101..1010] — see §3 PASS criteria for the rationale):** **`[202, 404, 606, 808, 1010, 1212, 1414, 1616, 1818, 2020]`** (i.e. `range(202, 2021, 202)`; 10 INDEPENDENT seeds disjoint from R59's set). This is a deterministic, recorded seed list — recorded in this plan, in `R60_INTEGRATION_VERDICT.md`, and in `R60_OPT_Y_MEASUREMENT.{md,json}`.
  - **Why a different set:** R58 used `[101..1010]`. R59 used `[101..1010]` (same set; recovered the L3 VC). To make R60 a TRULY independent third measurement (not a re-rerun on the same seeds), use a disjoint deterministic set. The 202-step pattern is reproducible and avoids overlap.
- **Bench commands (for reference; the worker may issue these directly):**
  ```bash
  cd /shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x
  python bench_all_42_R59_INTEGRATION.py \
      --mode 10run \
      --gpus 4,5,6,7 \
      --seeds 202,404,606,808,1010,1212,1414,1616,1818,2020 \
      --warmup 200 --iters 500 --trim 0.10 \
      --output_json R60_INTEGRATION_10RUN.json \
      --output_log  R60_INTEGRATION_10RUN.log \
      2>&1 | tee R60_INTEGRATION_10RUN.console
  ```
  (If the existing script does not accept `--seeds`, the worker may either (a) extend it to accept the seed list, or (b) write `bench_all_42_R60_INTEGRATION.py` as a thin wrapper that imports the R59 manifest and substitutes the seed list. Both are acceptable; do NOT modify the manifest.)
- **Expected wall time:** ~17 minutes for the 10-run (10 seeds × ~1.7 min per run on 4-way GPU sharding, 420 single-cell runs total). Within 30-min round budget.
- **Acceptance gate:** No SMOKE step needed. The 10-run output IS the deliverable.
- **Decision rule:**
  - **PASS** (defined precisely in §3 below) → emit `R60_INTEGRATION_10RUN.{json,log,console}` + `R60_OPT_Y_MEASUREMENT.md` summarizing the L3 cohort-race repeatability finding. Manifest unchanged.
  - **FAIL** (defined precisely in §3 below) → emit the same artifacts but recommend Opt W as R61 priority in the verdict. Manifest unchanged. R60 still commits as a methodology round; the failure is a data point, not a regression.

### Cohort K-2 (Opt U) — Documentation pivot artifact

- **Cohort ID:** K-2
- **Worker:** B (NO GPU; analysis + writing only)
- **Scope:** Author `R60_OPT_U_DOC_PIVOT.md` formalizing the structural ceiling and the SC/MICRO publication framing.
- **Inputs:**
  - `R59_INTEGRATION_VERDICT.md` (current ceiling state)
  - `R59_INTEGRATION_MANIFEST.json` (current canonical 42-cell mapping)
  - `R59_OPT_R_POLICY.md` (the 4-criterion validity envelope)
  - `R58_INTEGRATION_VERDICT.md` (previous round; for cross-round perspective)
  - All round verdicts R55-R59 (the 100% leaderboard era)
  - `TODO.md` and `AGENT_PROMPT.md` (project-level state)
- **Required sections in `R60_OPT_U_DOC_PIVOT.md`:**
  1. **Structural ceiling reached** — Quantify the achievement: 42/42 strict 10-run VC, 41/42 WIN, AITER bit-deterministic share 40/42 (largest in project history HELD), HK pool 2/42 (smallest in project history HELD), 11+ consecutive rounds of R50D shim AS-IS reuse. Cite R59 as the validation round.
  2. **The residual surface** — Enumerate the 2 attention cells with current state and why each is at ceiling:
     - L8 `(4096, 32768, 128256)` — 98.34% LOSE; aiter alt-tile axis FULLY CLOSED (128×512, 192×256, 224×256, 96×640, 64×1024 all DEAD); HK 256×256 axis CLOSED by correctness (R58 Opt O confirmed both R40B and R37 fallbacks produce WRONG_OUTPUT because R39A/R44A/R44D fixes were never ported into the K=128256 build); only Opt T (~3 R-rounds, very low confidence) remains as a mechanism axis. Gap = 1.66pp at aiter-internal ceiling.
     - L1 `(4096, 32768, 14336)` — 100.08% WIN R59 (oscillates ±0.15pp around WIN-line under ITERS=500 between sweeps); AITER alt-tile space EXHAUSTED (96×640 + 64×1024 closed in R59); ITERS=1000 one-off bump CLOSED by Opt R policy validity envelope. **Currently WIN per Opt R policy A.**
     - L3 `(32768, 14336, 2048)` HK R40B — RECOVERED in R59 (n_OK=10/10 fin_min=0.988); near-gate (fin_min sits ~0.018 above the 0.97 default gate); empirically ~10-20% per-sweep cohort-race tail-draw probability under ITERS=500. Opt Y monitoring axis is the R60 measurement.
  3. **What the SC/MICRO publication should claim** — Concrete claim list:
     - 42/42 verified-correct on 42 production GEMM shapes spanning M ∈ [4096, 128256], N ∈ [4096, 128256], K ∈ [2048, 128256] on MI355X (gfx950), under a 10-run independent-seed validation gate.
     - Mean pct_comp vs aiter ASM baseline (compute it from R59 — a small table by cohort: Llama-3 / DeepSeek / GPT-OSS / Mixtral / etc; if the cohort labels are not in the manifest, summarize by M-N-K bucket).
     - 40 of 42 cells served via aiter `.co` dlopen (R50D shim AS-IS); 2 of 42 served via HipKittens HK kernel R40B; demonstrates the dispatch-by-shape methodology beats single-kernel HK or single-kernel aiter.
     - Cohort-race repeatability characterized: AITER cells bit-deterministic across 10+ INDEPENDENT seed sweeps (wcf_max=0); HK cells exhibit ~10-20% per-sweep tail-draw probability on the worst-margin cell (L3) which RECOVERS on re-sweep.
     - The 1 residual LOSE cell (L8 K=128256) is at the aiter-internal ceiling and requires a from-scratch HK kernel rebuild (Opt T) to potentially close — not part of the publication scope.
  4. **R61-R65 axis space** — For each remaining option, classify as DEFER / ELECT / DEPRIORITIZE:
     - Opt T (L8 from-scratch HK build): DEFER unless explicit user election; ~3 R-rounds budget required.
     - Opt W (HK kernel rebuild with FINITE_GATE 0.97→0.95): DEPRIORITIZE if Opt Y PASSES on R60 (3rd consecutive sweep); ELECT for R61 if Opt Y FAILS on R60.
     - Opt X (L1 cross-product alt-tiles): NEVER (alt-tile space exhausted).
     - Opt U continuation (further documentation rounds): ELECT for R61 if no kernel-axis work is elected (low cost, builds publication artifacts).
     - Opt Y continuation (R61, R62, ...): ELECT as a low-cost monitor every 2-3 rounds while no other work is happening; provides a longitudinal cohort-race repeatability dataset.
     - Fresh axis: NONE proposed; all R45-R59 closure documented in `R59_INTEGRATION_VERDICT.md` §"R60+ closed-axis carry-forward".
  5. **Project state snapshot table** — One-screen summary of the current 42-cell manifest with: shape, source kernel, R59 pct_comp p50, R59 n_OK, R59 wcf_max, R59 fin_min, classification (WIN/LOSE), source category (AITER/HK). Pulled directly from `R59_INTEGRATION_10RUN.json`.
  6. **Streak history** — One-line summary per round R44-R59 (VC count, WIN cell count, key axis closures, R50D AS-IS streak counter). Same format as `TODO.md` §"Round sequence sanity check".
- **GPU assignment:** None.
- **Expected wall time:** ~10-20 minutes (analysis + drafting; no compute).
- **Acceptance gate:** Artifact `R60_OPT_U_DOC_PIVOT.md` exists, contains all 6 required sections, and is internally consistent with R59 verdict numbers.
- **Decision rule:** Opt U is always POLICY_ONLY. The artifact is the deliverable; no PROMOTE / DEAD branching.

---

## 3. Reviewer integration spec

**In Opt Y mode, the reviewer integration IS the cohort K-1 measurement.** There is NO separate "merge worker fragments and re-bench" step because:
- Manifest is byte-identical to R59 (which was byte-identical to R58 binary entries).
- No PROMOTE candidates → nothing to verify.
- Cohort K-1 directly produces `R60_INTEGRATION_10RUN.{json,log,console}` from the unchanged manifest under fresh seeds.

**Reviewer protocol summary:**
- Manifest: `R59_INTEGRATION_MANIFEST.json` (UNCHANGED; copy to `R60_INTEGRATION_MANIFEST.json` with version metadata bumped to R60 and an Opt Y note added — but binary entries byte-identical).
- Bench script: `bench_all_42_R59_INTEGRATION.py` AS-IS (or thin R60 wrapper that substitutes the seed list).
- Seeds: `[202, 404, 606, 808, 1010, 1212, 1414, 1616, 1818, 2020]` (DIFFERENT from R59's `[101..1010]`).
- ITERS=500 (per Opt R policy A; no bump for L1).
- WARMUP=200, TRIM_FRAC=0.10.
- 4 GPUs (4, 5, 6, 7); idle-verified pre-launch.
- Expected wall: ~17 minutes for 10-run; 420 single-cell runs total.

### Opt Y PASS / FAIL criteria

**PASS criteria (must hold ALL of):**
1. `32768x14336x2048` R40B HK: **n_OK ≥ 8/10 AND fin_min ≥ 0.97** (the same R45+ default strict-VC gate; tail-draw not catastrophic).
2. `4096x32768x14336` R57J1_L1 AITER: **bit-determinism preserved (wcf_max = 0)** — does NOT need to be WIN; a noise-edge LOSE here at 99.x% is acceptable per Opt R policy A. The bit-determinism check ensures the AITER `.co` is producing identical outputs across the new seed set.
3. All 40 AITER cells: **bit-deterministic (wcf_max = 0)** — preservation of the 40/42 AITER bit-deterministic share.
4. All 42 cells: **strict-VC PASS** under the R45+ default gate (n_OK ≥ 8/10 AND wcf_max < 0.02 AND wcf_std < 0.01 AND fin_min ≥ 0.97).

**Interpretation if PASS:** This is the **3rd consecutive successful VC sweep** on the L3 HK cell across R58 (PASS_9/10 fin_min=0.911 — the lone VC drop), R59 (PASS_10/10 fin_min=0.988), R60 (≥ PASS_8/10 fin_min ≥ 0.97). The empirical cohort-race tail-draw rate on L3 is then ≤ 1/3 sweeps under ITERS=500 (consistent with the ~10-20% per-sweep estimate from R59). **Opt W (HK kernel rebuild with FINITE_GATE 0.97→0.95) is permanently deprioritized** — the cost (breaking R50D AS-IS streak, ~1-2 R-rounds) is not justified for a residual ~10-20% per-sweep tail-draw on a single cell that recovers on re-sweep. Project structural ceiling is empirically confirmed.

**FAIL criteria (any ONE of triggers FAIL):**
- `32768x14336x2048` R40B HK: n_OK < 8/10 OR fin_min < 0.97 — i.e. the L3 cell flips OUT of strict VC for the 2nd time in 3 sweeps (R58 was the 1st; R60 would be the 2nd).
- Any AITER cell loses bit-determinism (wcf_max > 0) — would be a brand-new AITER regression and require root-cause investigation; not expected.
- Any other previously-passing cell loses strict VC — same as above; would indicate systemic measurement issue or GPU contamination.

**Interpretation if FAIL:** L3 surface is empirically intrinsic (2 flips in 3 sweeps = ~67% per-sweep flip rate, statistically significant). **Opt W (HK kernel rebuild with R44D FINITE_GATE 0.97 → 0.95) becomes R61 priority.** Cost: ~1-2 R-rounds, breaks R50D AS-IS streak, but justified by the structural surface being non-recoverable on re-sweep.

### Required reviewer artifacts
- `R60_INTEGRATION_MANIFEST.json` — byte-identical to R59 binary entries; metadata bumped to R60.
- `bench_all_42_R60_INTEGRATION.py` — minimal wrapper around R59 script (seed list substitution only) OR direct reuse of `bench_all_42_R59_INTEGRATION.py` with `--seeds` flag.
- `R60_INTEGRATION_10RUN.{json,log,console}` — the K-1 cohort output.
- `R60_INTEGRATION_VERDICT.md` — reviewer commits with PASS or FAIL classification.
- `R60_OPT_Y_MEASUREMENT.md` + `R60_OPT_Y_MEASUREMENT.json` — Opt Y artifact (cohort-race surface data point).
- `R60_OPT_U_DOC_PIVOT.md` — Opt U artifact (documentation pivot).

### NOT required (no SMOKE 1-seed needed)
- `R60_INTEGRATION_SMOKE1.{json,log,console}` is OPTIONAL. Since the manifest is byte-identical to R59, the SMOKE 1-seed pass is not informative; skip unless the worker wants a quick sanity check (~2.2 min wall).

---

## 4. Stopping criteria

| Tier | Definition | Met if |
|---|---|---|
| **Floor** | Methodology deliverables exist | `R60_OPT_U_DOC_PIVOT.md` artifact emitted AND Opt Y K-1 cohort completes (PASS or FAIL is fine; the answer is the value). |
| **Mode** | Floor + 42/42 VC HELD on R60 sweep | Floor met AND `R60_INTEGRATION_10RUN.json` shows 42/42 strict VC under the new seed set. |
| **Stretch** | Mode + 41/42 WIN HELD (no L1 LOSE-flip on R60 sweep) | Mode met AND `4096x32768x14336` (L1) shows pct_comp ≥ 100.0% under the R60 seed set. |

If Floor is not met, the round is INCOMPLETE. If Mode is not met (i.e. Opt Y FAILS), the round is COMPLETE-with-FAIL — emit the verdict, recommend Opt W for R61, manifest unchanged. If Stretch is not met but Mode is, R60 commits as 42/42 VC + 40/42 WIN (L1 noise-edge re-sweep crossed back to LOSE-edge); per Opt R policy A, this is still production-correct.

---

## 5. R61 candidate set

**R61 axis selection branches on R60 Opt Y outcome:**

### If R60 Opt Y PASSES (3rd consecutive successful sweep on L3)
1. **R61 Opt U-2 — Continued documentation pivot** — Author `R61_PUBLICATION_DRAFT_OUTLINE.md` with section-level skeleton for the SC/MICRO submission (intro, methods, dispatch table, validation protocol, results, related work, limitations including the L8 residual). NO GPU.
2. **R61 Opt Y-2 — Cohort-race monitor (4th consecutive sweep)** — Re-bench R59/R60 manifest under another disjoint seed set (e.g. `[303, 606, 909, 1212, 1515, 1818, 2121, 2424, 2727, 3030]`). ~17 min wall. Builds longitudinal cohort-race dataset.
3. **R61 Opt Z (NEW) — Per-shape decomposition table** — Produce a per-cell mechanism summary table (which axes were tried, which closed, why current source is best) for inclusion in the publication. NO GPU. This is publication-prep work that compresses 17 rounds of round-verdicts into a single per-cell table.
4. **R61 Opt T (low priority)** — Begin L8 from-scratch HK build for K=128256. Only if user explicitly elects the ~3-round budget.
5. **R61 housekeeping** — Verify TODO.md and AGENT_PROMPT.md are aligned with the documentation-pivot framing; no benchmark cost.

### If R60 Opt Y FAILS (2nd L3 flip in 3 sweeps; surface is intrinsic)
1. **R61 Opt W — HK kernel rebuild with R44D FINITE_GATE 0.97 → 0.95** — Highest priority. Estimated cost: ~1-2 R-rounds. Breaks R50D AS-IS streak. Mechanism: relax the cohort-race finiteness gate to absorb the worst-margin tail-draw on L3. Worker scope: rebuild `tk_mxfp4_gluon_cpp_n14336_k2048_ts_gm6_v12_dc_pfoff4_R40B_safe` with the gate constant changed; SMOKE on the 2 HK cells; if PASS, propose as PROMOTE. Risk: gate relaxation can mask other correctness issues; SMOKE must include 5/5 seeds with snr_med check.
2. **R61 Opt Y-2 — Cohort-race monitor (4th consecutive sweep)** — Run alongside Opt W on a different GPU shard; gives an independent FAIL data point if Opt Y fails again, OR a cross-check if Opt Y passes (would suggest the R60 FAIL was itself a single-sweep tail-draw on the gate boundary).
3. **R61 Opt U-2 — Documentation pivot continuation** — Even if Opt W is in flight, the documentation work has independent value; assign to a no-GPU worker.
4. **R61 Opt T (low priority, unchanged)** — As above.

---

## 6. Constraints reminder (mandatory; do not violate)

- ALL bench measurements: `WARMUP=200, ITERS=500, TRIM_FRAC=0.10` (per `.claude/rules/benchmark-rules.md` and Opt R policy A).
- 10 INDEPENDENT seeds; do NOT reuse R59's `[101..1010]`. Use R60's `[202, 404, 606, 808, 1010, 1212, 1414, 1616, 1818, 2020]` (recorded above).
- 4 GPUs only (4, 5, 6, 7); idle-verified pre-launch via `rocm-smi --showuse`.
- The Opt Y cohort uses the **EXISTING `R59_INTEGRATION_MANIFEST.json` AS-IS** — no binary modifications, no new .co files, no shim rebuilds. The manifest may be COPIED to `R60_INTEGRATION_MANIFEST.json` with metadata bumped (round/timestamp/notes), but binary entries (`shapes_to_so_path`, `aiter_dispatch_template`) MUST be byte-identical to R59.
- No SMOKE step is required for K-1 (manifest is already known-correct from R59); the 10-run output IS the deliverable.
- Under no circumstances should this round propose: (a) re-attempting any closed alt-tile axis; (b) bumping ITERS=1000 outside the 4-criterion Opt R envelope; (c) editing the kernel source; (d) modifying the R50D shim.
- R60 is a **methodology round**. Net structural delta: +2 documentation/measurement artifacts + 12th consecutive R50D AS-IS reuse round. No PROMOTE candidates expected.

---

## 7. Schedule summary

| Cohort | Worker | GPU | Wall | Output |
|---|---|---|---|---|
| K-1 (Opt Y) | A | 4-7 | ~17 min | `R60_INTEGRATION_10RUN.{json,log,console}` + `R60_OPT_Y_MEASUREMENT.{md,json}` |
| K-2 (Opt U) | B | none | ~10-20 min (parallel with K-1) | `R60_OPT_U_DOC_PIVOT.md` |
| Reviewer integration | (folded into K-1) | n/a | (~5 min for verdict drafting after K-1 completes) | `R60_INTEGRATION_VERDICT.md` + `R60_INTEGRATION_MANIFEST.json` |

**Total round wall ≈ 30 minutes** (K-1 dominates; K-2 runs in parallel with no GPU).

---

## 8. Round signature

- **Round axis:** Opt U (documentation pivot) + Opt Y (cohort-race surface monitoring)
- **Cohorts:** 2 (K-1 GPU bench; K-2 no-GPU doc)
- **PROMOTE candidates:** 0 (none expected; manifest byte-identical to R59)
- **GPUs used:** 4-7
- **Seeds:** `[202, 404, 606, 808, 1010, 1212, 1414, 1616, 1818, 2020]` (disjoint from R59's `[101..1010]`)
- **R50D AS-IS reuse:** R60 = 12th consecutive round
- **Expected wall:** ~30 min (~17 min K-1 GPU + parallel ~15 min K-2 doc + ~5 min reviewer write-up)
- **Expected outcome:** 0 PROMOTE / 0 SMOKE_DEAD / 2 POLICY_ONLY (Opt U + Opt Y)
- **Branching rule for R61:** R60 Opt Y PASS → continue documentation pivot (Opt U-2 + Opt Z + Opt Y-2 monitor). R60 Opt Y FAIL → elect Opt W (HK kernel rebuild with FINITE_GATE 0.97 → 0.95) as R61 priority.
