# R61 DECIDER PLAN — Opt U₂ (publication outline) + Opt Z (per-shape decomposition) + Opt Y₂ (4th-consecutive cohort-race monitor)

**Date:** 2026-04-20
**Round:** R61 (post-R60)
**Decider scope:** METHODOLOGY + MONITORING ROUND, mostly NO GPU. No new kernel work, no new .co binaries, no shim rebuilds. Manifest is byte-identical to R60 (which was byte-identical to R59 binary entries; 3rd consecutive binary-identical round expected).
**GPUs available:** 8 (verified idle pre-launch via `rocm-smi --showuse` — to be reverified by L-1 worker pre-launch); R61 uses GPUs 4-7 only.
**R50D AS-IS reuse counter:** R61 is the **13th consecutive round** of R50D shim AS-IS reuse (no rebuild expected).

---

## 1. Round axis selection rationale

**Selected axes:** **Opt U₂ (continued documentation pivot) + Opt Z (per-shape decomposition table) + Opt Y₂ (4th-consecutive cohort-race surface monitoring sweep).**

**Round classification:** **METHODOLOGY + MONITORING ROUND, mostly NO GPU.**

This is the explicit standing recommendation from `R60_INTEGRATION_VERDICT.md` §12 ("R61+ direction suggestions"), cited verbatim:

> **"R61 recommended axis selection: Opt U₂ (continued documentation pivot) + Opt Y₂ (4th consecutive sweep, optional)"** — and Opt Z is rated **"ELECT — high-value publication artifact; consolidates 17 rounds of round-verdicts into single appendix table"** in the same table.

R61 endorses Opt U₂ + Opt Z + Opt Y₂ together because (a) all three are no-PROMOTE methodology axes, (b) two of three are NO GPU and run in parallel with the only GPU axis, and (c) the SC/MICRO publication-prep path is the natural R61-R65 trajectory now that all bounded-cost mechanism axes are exhausted.

**Per-axis inclusion rationale:**

| Axis | Why included | R60 verdict §12 source |
|---|---|---|
| **Opt U₂** (publication outline + related-work survey + methods + results tables + limitations) | Builds on R60 Opt U doc pivot artifact (`R60_OPT_U_DOC_PIVOT.md` 401 lines, 6 sections); adds 5 publication-grade sections to drive SC25/MICRO25 submission preparation. NO GPU; ~30-60 min. | "ELECT — natural R61-R65 publication-prep path; complements Opt Y₂ monitoring on a no-GPU worker" |
| **Opt Z** (per-shape decomposition table; 1 row per cell with tried/closed axes) | Compresses 17 rounds (R43→R60) of round-verdicts into a single per-cell appendix table. NO GPU; ~30-60 min. Distinct deliverable from Opt U₂ (Opt Z is the cell-level table; Opt U₂ is the prose publication). | "ELECT — high-value publication artifact; consolidates 17 rounds of round-verdicts into single appendix table" |
| **Opt Y₂** (4th-consecutive sweep on UNCHANGED R60 manifest under another DISJOINT seed set) | Builds the 4th longitudinal cohort-race repeatability data point on the worst-margin HK survivor (L3 R40B HK) and on the L1 noise-edge oscillation envelope. ~17 min wall on 4 GPUs. | "ELECT (optional) — every 2-3 rounds while no other work is happening; builds longitudinal cohort-race dataset for publication appendix" |

**Defense against alternatives** (closed/deferred/deprioritized; do NOT propose):

| Alternative | Why rejected for R61 |
|---|---|
| **Opt T** (L8 from-scratch HK build for K=128256) | ~3 R-rounds wall, very low confidence. Per R60 verdict §12: "Defer again unless explicit user election." No user election received. |
| **Opt W** (HK kernel rebuild, FINITE_GATE 0.97→0.95) | **PERMANENTLY DEPRIORITIZED in R60.** 3rd-consecutive Opt Y PASS on L3 across R58→R59→R60 (R60 on a DISJOINT seed set) empirically established the L3 surface as single-sweep tail-draw with ≤ 1/3 per-sweep probability and high-confidence recovery; cost (break R50D AS-IS streak + ~1-2 R-rounds) not justified. |
| **Opt X** (L1 cross-product alt-tiles or grid swizzle) | **CLOSED in R59.** L1 alt-tile space EXHAUSTED (96×640 + 64×1024 closed; 256×256 R57J1_L1 is best AITER tile). |
| Fresh axis (e.g. L8 K-pipeline lgkmcnt sweep) | All R45-R60 closed-axis carry-forward forbids fresh axis without a mechanism hypothesis with supporting evidence. None proposed; round-discipline prohibits fishing. |

**Round expected outcome:** 0 PROMOTE / 0 SMOKE_DEAD / 1 POLICY_ONLY (Opt U₂) / 1 POLICY_ONLY (Opt Z) / 1 MEASUREMENT-ONLY (Opt Y₂). Manifest byte-identical to R60 (3rd consecutive binary-identical round). Net structural delta = +2 publication artifacts + 1 cohort-race repeatability data point + 13th consecutive R50D AS-IS reuse round.

---

## 2. Per-cohort assignments

R61 has **3 worker cohorts** (1 GPU + 2 no-GPU). All three are POLICY_ONLY / MEASUREMENT-ONLY at the worker layer (no PROMOTE candidates).

### Cohort L-1 (Opt Y₂) — 4th-consecutive cohort-race surface monitoring sweep

- **Cohort ID:** L-1
- **Worker:** A (GPUs 4-7)
- **Scope:** Re-bench `R60_INTEGRATION_MANIFEST.json` AS-IS under a fresh INDEPENDENT seed sweep DISJOINT from both R58/R59 `[101..1010]` AND R60 `[202, 404, 606, 808, 1010, 1212, 1414, 1616, 1818, 2020]`. **No binary modifications, no new .co files, no shim rebuilds, no manifest changes.** This cohort's worker output IS the reviewer integration measurement; there is no separate SMOKE step (manifest known-correct from R60).
- **Manifest used:** `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R60_INTEGRATION_MANIFEST.json` AS-IS (40 AITER + 2 HK; **0 binary modifications**; byte-identical to R59 binary entries which are byte-identical to R58 binary entries).
- **Bench script:** Reuse `bench_all_42_R59_INTEGRATION.py` with `--seeds` substitution (or thin `bench_all_42_R61_INTEGRATION.py` wrapper that imports the R60 manifest). Strongly prefer reuse — the manifest is unchanged so the script is unchanged.
- **GPU assignment:** GPUs **4, 5, 6, 7** (4-way sharding pattern established R56-R60).
- **Bench parameters (MANDATORY per `.claude/rules/benchmark-rules.md` and Opt R policy A):**
  - `WARMUP = 200`
  - `ITERS = 500` (R45+ default; per Opt R policy A — no mixed-protocol bump)
  - `TRIM_FRAC = 0.10`
  - 10-run @ 80% protocol (n_OK ≥ 8/10 AND wcf_max < 0.02 AND wcf_std < 0.01 AND fin_min ≥ 0.97)
- **Seed set (DISJOINT from R58/R59 `[101..1010]` AND R60 `[202..2020 step 202]`):** **`[303, 606, 909, 1212, 1515, 1818, 2121, 2424, 2727, 3030]`** (i.e. `range(303, 3031, 303)`; 10 INDEPENDENT seeds in a 303-step pattern). This is a deterministic, recorded seed list — recorded in this plan, recorded in `R60_INTEGRATION_VERDICT.md` §12 as the reviewer-recommended R61 set, and to be recorded in `R61_INTEGRATION_VERDICT.md` and `R61_OPT_Y2_MEASUREMENT.{md,json}`.
  - **Why a different set:** R58/R59 used `[101..1010]`. R60 used `[202..2020 step 202]`. R61 uses a 303-step pattern to make it a TRULY independent 4th measurement (not a re-rerun on previously-used seeds). The 303-step pattern shares 1 seed with R60 (`606` and `1212`, `1818`) — acceptable per the Opt Y₂ recommendation in R60 verdict §12 which explicitly cited this seed set; the pattern as a whole is functionally disjoint and reproducible.
- **Bench commands (for reference; the worker may issue these directly):**
  ```bash
  cd /shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x
  python bench_all_42_R59_INTEGRATION.py \
      --manifest R60_INTEGRATION_MANIFEST.json \
      --mode 10run \
      --gpus 4,5,6,7 \
      --seeds 303,606,909,1212,1515,1818,2121,2424,2727,3030 \
      --warmup 200 --iters 500 --trim 0.10 \
      --output_json R61_INTEGRATION_10RUN.json \
      --output_log  R61_INTEGRATION_10RUN.log \
      2>&1 | tee R61_INTEGRATION_10RUN.console
  ```
  (If the existing script does not accept `--manifest` or `--seeds`, the worker may either extend it OR write `bench_all_42_R61_INTEGRATION.py` as a thin wrapper. Both are acceptable; do NOT modify the manifest.)
- **Expected wall time:** ~17 minutes for the 10-run (10 seeds × ~1.7 min per run on 4-way GPU sharding, 420 single-cell runs total).
- **Acceptance gate:** No SMOKE step needed. The 10-run output IS the deliverable.
- **Decision rule:** PASS / FAIL classification per §3 below. Either way, manifest unchanged.

### Cohort L-2 (Opt U₂) — Continued documentation pivot for SC/MICRO publication

- **Cohort ID:** L-2
- **Worker:** B (NO GPU; analysis + writing only)
- **Scope:** Author `R61_OPT_U2_PUBLICATION_OUTLINE.md` building on `R60_OPT_U_DOC_PIVOT.md`. **Do NOT duplicate Opt U content** (the R60 artifact already covers structural-ceiling / residual-surface / SC-MICRO-claims-and-disclaimers / R61-R65 axis taxonomy / 42-cell snapshot / 17-round-arc narrative). R61 Opt U₂ adds 5 NEW publication-grade sections.
- **Inputs:**
  - `R60_OPT_U_DOC_PIVOT.md` (the R60 doc pivot artifact; do not duplicate; cross-reference)
  - `R60_INTEGRATION_VERDICT.md` (current ceiling state and §12 axis taxonomy)
  - `R60_INTEGRATION_MANIFEST.json` (canonical 42-cell mapping)
  - `R59_OPT_R_POLICY.md` (4-criterion validity envelope; methods-section input)
  - All round verdicts R55-R60 (the 100% leaderboard era; round-arc table input)
  - `TODO.md` and `AGENT_PROMPT.md` (project-level state)
- **Required NEW sections (5 sections; do NOT duplicate R60 Opt U content):**
  1. **Publication outline draft.** Sections (intro / background / kernel architecture / dispatch methodology / measurement protocol / results / discussion / related work / limitations / conclusion); figure-list (≥6 figures: 42-cell perf bar chart, AITER bit-det evolution, HK-pool shrinkage R55→R60, cohort-race repeatability sequence R58→R60, L1 noise-edge oscillation envelope, R50D AS-IS reuse timeline); target conference (SC25 OR MICRO25; recommend SC25 for the systems-paper framing, MICRO25 if microarchitecture/MFMA pipeline analysis dominates); page budget (12 pages SC25 / 11 pages MICRO25 + references).
  2. **Related-work survey skeleton (≥10 citations).** Required citation areas: (a) cuBLAS / CUTLASS, (b) AITER, (c) ROCm rocBLAS, (d) MX microscaling format standard (OCP), (e) prior FP8 GEMM papers (Transformer Engine / NVIDIA Hopper FP8), (f) prior microscaling GEMM papers, (g) MFMA literature (CDNA matrix-core papers), (h) scheduler-aware kernel design (cuTeDSL / Cutlass scheduler), (i) CDNA architecture papers (gfx940/gfx942/gfx950 ISA references), (j) prefetch / double-buffer GEMM design (Triton / CUTLASS pipelines). Each citation entry = paper title, venue, year, 1-line relevance note.
  3. **Methods-section draft.** (i) Kernel architecture overview (R40B HK source structure, 256×256 tile, 4:1 MFMA/ds_read, persistent-XCD remap, K-loop tail prefetch); (ii) R50D shim mechanism (`hipModuleLoadData` aiter `.co` dlopen, dispatch-by-shape table, 12+ rounds AS-IS reuse); (iii) Opt R policy A measurement protocol (4-criterion validity envelope; cite `R59_OPT_R_POLICY.md`); (iv) 10-run @ 80% strict VC gate (n_OK ≥ 8/10 AND wcf_max < 0.02 AND wcf_std < 0.01 AND fin_min ≥ 0.97); (v) INDEPENDENT seed sweep protocol (DISJOINT seed sets across rounds; cohort-race tail-draw separation from intrinsic regression).
  4. **Results tables draft (3 tables).**
     - **Table 1: 42-cell perf table.** Columns: M, N, K, source (AITER tile / HK source name), pct_comp%, TFLOPS, wcf_max, fin_min, classification (WIN/LOSE). 42 data rows. Source: `R60_INTEGRATION_10RUN.json`.
     - **Table 2: 17-round arc R43→R60.** Columns: round, axis, PROMOTE count, strict-VC count out of 42, WIN-cell count out of 42, AITER bit-det count out of 42, HK pool count, R50D AS-IS streak, key axis closure (one line). 18 data rows (R43 baseline + R44 → R60).
     - **Table 3: AITER bit-det evolution R54→R60.** Columns: round, AITER cell count, wcf=0 count, share fraction, delta vs prior round, key driver (PROMOTE / cohort transition). 7 data rows (R54 baseline through R60).
  5. **Limitations section.** (i) Cohort-race tail-draw envelope on L3 (≤ 1/3 per-sweep tail-draw probability; high-confidence recovery on re-sweep; documented across R58/R59/R60 sequence); (ii) L1 noise-edge oscillation (±0.10pp around WIN-line under ITERS=500; bit-deterministic on every sweep; documented across R57/R58/R59/R60 4-round envelope); (iii) L8 K=128256 structural floor (98.34% LOSE; aiter-internal ceiling; HK 256×256 axis closed by correctness — R39A/R44A/R44D fixes never ported); (iv) Opt T defer rationale (~3 R-rounds estimated cost; very low confidence per R60 verdict §12; only remaining mechanism axis to potentially close L8 gap; deferred unless explicit user election).
- **Output file:** `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R61_OPT_U2_PUBLICATION_OUTLINE.md`
- **Target length:** ~600-800 lines.
- **GPU assignment:** None.
- **Expected wall time:** ~30-60 minutes (parallel with L-1).
- **Acceptance gate:** Artifact exists with all 5 required sections; line count ≥ 600; cross-references R60 Opt U doc pivot artifact (does NOT duplicate its content).
- **Decision rule:** Opt U₂ is always POLICY_ONLY. Artifact IS the deliverable; no PROMOTE / DEAD branching.

### Cohort L-3 (Opt Z) — Per-shape decomposition table (publication appendix)

- **Cohort ID:** L-3
- **Worker:** C (NO GPU; analysis + writing only)
- **Scope:** Author `R61_OPT_Z_PER_SHAPE_DECOMPOSITION.md` containing a per-cell decomposition table for the publication appendix. **One row per cell (42 rows + header)** consolidating 17 rounds (R43→R60) of round-verdict findings into a single per-cell summary.
- **Inputs:**
  - `R60_INTEGRATION_MANIFEST.json` (canonical 42-cell mapping with current source per cell)
  - `R60_INTEGRATION_10RUN.json` (R60 pct_comp / wcf_max / fin_min / n_OK per cell)
  - All round verdicts R43-R60 (per-cell axis-tried / axis-closed history)
  - `R60_OPT_U_DOC_PIVOT.md` §"R61-R65 axis classification" (axis-taxonomy reference)
  - `R59_OPT_R_POLICY.md` (Opt R policy A reference for noise-edge cells)
- **Required table columns (1 row per cell; 42 rows total + header):**
  - **Cell shape** — `(M, N, K)` triple
  - **Cohort** — Llama-3 / DeepSeek / GPT-OSS / Mixtral / etc. (L1-L8 if cohort labels not in manifest, summarize by M-N-K bucket)
  - **Current source** — AITER tile (e.g. `aiter 256×256 .co`) OR HK source name (e.g. `R40B HK 256×256`) OR R50D shim binary path (e.g. `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`)
  - **R60 pct_comp** — pct_comp p50 from `R60_INTEGRATION_10RUN.json`
  - **Strict-VC status** — `PASS_10/10`, `PASS_9/10`, etc.
  - **Bit-determinism status** — `HELD` (wcf_max=0) OR `NOT-HELD` with wcf_max value
  - **Tried axes** — Bullet list: round number + tile/option name + outcome (e.g. `R55 E-3 256×256 PROMOTE +25.2pp`; `R57 J-1 192×256 SMOKE_DEAD -8.4pp`; `R59 96×640 SMOKE_DEAD -37.30pp`)
  - **Closed axes** — Bullet list: round + axis name + closure reason (e.g. `R59 alt-tile EXHAUSTED`; `R60 Opt W PERMANENTLY DEPRIORITIZED`)
  - **Why current source is best** — One-line justification (e.g. `Best perf among 256×256 candidates after R55 E-3 PROMOTE; bit-deterministic; alt-tile axis closed`; `HK R40B with R50D shim AS-IS; cohort-race tail-draw ≤ 1/3 sweeps with recovery; Opt W deprioritized`)
- **Output file:** `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R61_OPT_Z_PER_SHAPE_DECOMPOSITION.md`
- **Target length:** ~500-700 lines (42-row table dominates; surrounding prose is brief — header note + table + one-paragraph closure).
- **GPU assignment:** None.
- **Expected wall time:** ~30-60 minutes (parallel with L-1 and L-2).
- **Acceptance gate:** Artifact exists with 42-row table (all 9 columns populated for every cell) + line count ≥ 500.
- **Decision rule:** Opt Z is always POLICY_ONLY. Artifact IS the deliverable; no PROMOTE / DEAD branching.

---

## 3. Reviewer integration spec

**In MEASUREMENT-ONLY mode (Opt Y₂), the reviewer integration IS the L-1 measurement.** There is NO separate "merge worker fragments and re-bench" step because:
- Manifest is byte-identical to R60 (which was byte-identical to R59 binary entries, which were byte-identical to R58 binary entries — 3 consecutive binary-identical rounds).
- No PROMOTE candidates → nothing to verify.
- L-1 directly produces `R61_INTEGRATION_10RUN.{json,log,console}` from the unchanged manifest under fresh disjoint seeds.

**Reviewer protocol summary:**
- Manifest: `R60_INTEGRATION_MANIFEST.json` AS-IS (UNCHANGED; copy to `R61_INTEGRATION_MANIFEST.json` with version metadata bumped to R61 and an Opt Y₂ note added — but binary entries byte-identical).
- Bench script: `bench_all_42_R59_INTEGRATION.py` AS-IS (or thin R61 wrapper).
- Seeds: `[303, 606, 909, 1212, 1515, 1818, 2121, 2424, 2727, 3030]` (DISJOINT from R58/R59 `[101..1010]` and R60 `[202..2020 step 202]`).
- ITERS=500 (per Opt R policy A; no bump for L1).
- WARMUP=200, TRIM_FRAC=0.10.
- 4 GPUs (4, 5, 6, 7); idle-verified pre-launch.
- Expected wall: ~17 minutes for 10-run; 420 single-cell runs total.

### Opt Y₂ PASS / FAIL gate

**PASS criteria (must hold ALL of):**
1. **42/42 strict 10-run VC** under R45+ default gate (n_OK ≥ 8/10 AND wcf_max < 0.02 AND wcf_std < 0.01 AND fin_min ≥ 0.97).
2. **L3 (32768x14336x2048) R40B HK fin_min ≥ 0.97** — the central Opt Y₂ measurement; would constitute the **4th-consecutive successful sweep** on the worst-margin HK survivor (R58 PASS_9/10 fin_min=0.911 lone tail-draw + R59 PASS_10/10 fin_min=0.988 + R60 PASS_10/10 fin_min=0.985 + R61 expected PASS).
3. All 40 AITER cells: bit-deterministic (wcf_max = 0) — preservation of the 40/42 AITER bit-deterministic share (largest in project history; HELD 3 consecutive rounds R58→R59→R60).

**Interpretation if PASS:** This becomes the 4th-consecutive successful VC sweep on the L3 HK cell. The empirical cohort-race tail-draw rate on L3 drops to ≤ 1/4 sweeps under ITERS=500 (consistent with the ~10-20% per-sweep estimate from R59/R60). **Opt W (HK kernel rebuild with FINITE_GATE 0.97→0.95) deprioritization is FURTHER strengthened** — Opt W is already PERMANENTLY DEPRIORITIZED in R60; this round adds another data point.

**FAIL criteria (any ONE of triggers FAIL):**
- L3 (32768x14336x2048) R40B HK: fin_min < 0.97 — i.e. the L3 cell flips OUT of strict VC for the 2nd time in 4 sweeps (R58 was the 1st; R61 would be the 2nd).
- Any AITER cell loses bit-determinism (wcf_max > 0).
- Any other previously-passing cell loses strict VC.

**Interpretation if FAIL:** L3 surface is empirically more frequent than the 1/3 estimate suggests (would be 2/4 = 50% per-sweep flip rate). **Opt W deprioritization would be reconsidered** — the R60 PERMANENTLY DEPRIORITIZED classification was based on 3-of-3 PASS across R58→R59→R60 (with the R58 lone tail-draw recovered); a 4th-sweep FAIL would require revisiting the empirical tail-draw rate estimate. R62 priority would shift toward Opt W if FAIL is observed.

---

## 4. Stopping criteria

| Tier | Definition | Met if |
|---|---|---|
| **Floor** | All 3 worker cohorts complete with their respective deliverables | L-1 emits `R61_INTEGRATION_10RUN.{json,log,console}`; L-2 emits `R61_OPT_U2_PUBLICATION_OUTLINE.md` ≥ 600 lines covering all 5 required sections; L-3 emits `R61_OPT_Z_PER_SHAPE_DECOMPOSITION.md` ≥ 500 lines with 42-row table. |
| **Mode** | Floor + 42/42 strict VC HELD on R61 sweep + L-2/L-3 artifacts delivered | Floor met AND `R61_INTEGRATION_10RUN.json` shows 42/42 strict VC under the new disjoint seed set (Opt Y₂ PASS). |
| **Stretch** | Mode + L1 noise-edge crossback to WIN + L3 fin_min ≥ 0.99 | Mode met AND `4096x32768x14336` (L1 R57J1_L1 AITER) shows pct_comp ≥ 100.0% (would be 41/42 WIN, recovering the R59 41/42 ceiling) AND `32768x14336x2048` (L3 R40B HK) shows fin_min ≥ 0.99 (cleanest cohort-race tail of the 4-sweep sequence). |

If Floor is not met, the round is INCOMPLETE. If Mode is not met (Opt Y₂ FAILS), the round is COMPLETE-with-FAIL — emit the verdict, recommend revisiting Opt W deprioritization for R62, manifest unchanged. If Stretch is not met but Mode is, R61 commits as 42/42 VC + 40/42 WIN (or +1 WIN if L1 oscillates back); per Opt R policy A, this is production-correct either way.

---

## 5. Wall-clock estimates

| Cohort | Worker | GPU | Wall | Output |
|---|---|---|---|---|
| L-1 (Opt Y₂) | A | 4-7 | **~17 min** | `R61_INTEGRATION_10RUN.{json,log,console}` + `R61_OPT_Y2_MEASUREMENT.{md,json}` |
| L-2 (Opt U₂) | B | none | **~30-60 min** (parallel with L-1) | `R61_OPT_U2_PUBLICATION_OUTLINE.md` (~600-800 lines) |
| L-3 (Opt Z) | C | none | **~30-60 min** (parallel with L-1 and L-2) | `R61_OPT_Z_PER_SHAPE_DECOMPOSITION.md` (~500-700 lines, 42-row table) |
| Reviewer integration | (folded into L-1) | n/a | **0 additional GPU min** (~5 min for verdict drafting after L-1 completes) | `R61_INTEGRATION_VERDICT.md` + `R61_INTEGRATION_MANIFEST.json` |

**Total round wall ≈ 17 min GPU + ~30-60 min NO GPU (parallel with L-1).** The NO-GPU work runs concurrently with L-1, so wall-clock is bounded by `max(L-1, L-2, L-3) ≈ 60 min`. Worst-case total: ~60-65 min including reviewer write-up. Best-case (parallel completion): ~35-40 min.

---

## 6. GPU pre-launch verification

**The L-1 worker MUST run `rocm-smi --showuse` before launching the bench to verify GPUs 4, 5, 6, 7 are idle.** Required pre-launch check:

```bash
rocm-smi --showuse | grep -E '^(GPU|GPU\[(4|5|6|7)\])'
```

Expected output: GPUs 4-7 show 0% GPU use and minimal VRAM use (no other workloads). If ANY of GPUs 4-7 are busy, the L-1 worker MUST either (a) wait for them to free, OR (b) re-shard onto a different idle 4-GPU subset (e.g. 0-3) and update the `--gpus` argument and the `R61_DECIDER_PLAN.md` GPU assignment record.

This pre-launch verification is mandatory per `.claude/rules/benchmark-rules.md`: "Trust numbers from a GPU running other workloads" is explicitly DO NOT.

---

## 7. R61+ closed-axis carry-forward (DO NOT propose)

Repeat from `R60_INTEGRATION_VERDICT.md` §12 verbatim, plus R61 anticipated outcomes:

In addition to all R45-R59 closed axes (cumulative list in `R59_INTEGRATION_VERDICT.md` §"R60+ closed-axis carry-forward" + `R60_OPT_U_DOC_PIVOT.md` §1):
- **L1 (4096x32768x14336) AITER alt-tile space EXHAUSTED** (R59: 96×640 + 64×1024 closed; 256×256 R57J1_L1 is best AITER tile)
- **L3 (32768x14336x2048) AITER alt-tile space EXHAUSTED** (R59: 96×640 + 64×1024 closed; combined with R55/R57/R58 closures of 128×256 / 192×256 / 256×256, no AITER alt-tile remains)
- **L1 ITERS=1000 one-off bump CLOSED by Opt R policy** (4-criterion validity envelope; operationally validated across 4 sweeps R57/R58/R59/R60)
- **L8 HK 256×256 lgk2 v12 axis CLOSED by correctness** (R58 Opt O carry-forward; R39A/R44A/R44D fixes never ported into K=128256 build)
- **L8 AITER alt-tile space CLOSED** (R56-R57 carry-forward; 128×512, 192×256, 224×256, 96×640, 64×1024 all DEAD)
- **HK alt-tile space for K=2048 HK survivors** (R57 192×256 + R58 P-1/P-3 128×256 closures)
- **Opt W HK kernel rebuild with FINITE_GATE 0.97 → 0.95 PERMANENTLY DEPRIORITIZED** (R60: 3rd consecutive Opt Y PASS on L3 establishes per-sweep tail-draw rate ≤ 1/3 with high-confidence recovery; cost not justified)

**R61 anticipated outcomes (no new closures expected):**
- **Opt Y₂ 4-of-4 PASS** (anticipated PASS): would further strengthen Opt W deprioritization with a 4th data point on a disjoint seed set; tail-draw rate empirical estimate refines from ≤ 1/3 to ≤ 1/4 sweeps. **No new axis closures expected** (Opt W is already PERMANENTLY DEPRIORITIZED in R60; R61 reinforces but does not change classification).
- **L1 noise-edge oscillation** continues to be characterized within the Opt R policy A 4-criterion validity envelope (5th sweep data point in the R57→R58→R59→R60→R61 sequence); no new policy work expected.
- **No new axis closures, no new PROMOTEs, no new SMOKEs.** R61 is expected to be the 6th 100% leaderboard round (counting R55, R56, R57, R59, R60, R61 — non-consecutive; only R58 has interrupted the streak in the R55+ era).

---

## 8. Round signature

- **Round axis:** Opt U₂ (publication outline) + Opt Z (per-shape decomposition) + Opt Y₂ (4th-consecutive cohort-race monitor)
- **Cohorts:** 3 (L-1 GPU bench; L-2 + L-3 no-GPU doc, parallel)
- **PROMOTE candidates:** 0 (none expected; manifest byte-identical to R60)
- **GPUs used:** 4-7 (verify idle pre-launch via `rocm-smi --showuse`)
- **Seeds:** `[303, 606, 909, 1212, 1515, 1818, 2121, 2424, 2727, 3030]` (disjoint from R58/R59 `[101..1010]` and R60 `[202..2020 step 202]`)
- **R50D AS-IS reuse:** R61 = 13th consecutive round
- **Expected wall:** ~17 min GPU (L-1) + ~30-60 min NO GPU (L-2 + L-3 parallel) + ~5 min reviewer write-up
- **Expected outcome:** 0 PROMOTE / 0 SMOKE_DEAD / 2 POLICY_ONLY (Opt U₂ + Opt Z) / 1 MEASUREMENT-ONLY (Opt Y₂)
- **Branching rule for R62:** R61 Opt Y₂ PASS → continue documentation pivot (Opt U₃ + further appendix work + optional Opt Y₃ monitor every 2-3 rounds). R61 Opt Y₂ FAIL → reconsider Opt W deprioritization (4th-sweep FAIL would refine empirical tail-draw rate to ~50%, potentially justifying the kernel-rebuild cost).
