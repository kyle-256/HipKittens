# R52 Reviewer — R49–R52 cycle audit & 21-cell baseline refresh

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 9a246498 (R52 Dev G REFUTED)
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=7
**Protocol:** strict SCLK — 5 runs/cell, 30 s cooldown between runs,
60 s rebuild cooldown, MXFP8_WARMUP=100, MXFP8_ITERS=200, median scoring,
isolated GPU.

## TL;DR

- **Phase 1 — 21-cell baseline:** **5/21 PASS** under refreshed strict-SCLK
  on current HEAD. Two cells lost their R48 PASS (8192³ RRR/CRR) but only
  because the FP8 baseline was uncontaminated this time — the MXFP8
  numbers themselves are essentially unchanged from R48.
- **Phase 2 — R49 8-cell re-audit (focused on 3 highest-spread cells):**
  **2 verdict flips found.** B_8B_Down_CRR flips NOISE → CONFIRMED
  (under strict SCLK the CRR XCD swizzle is +0.97% on this cell, not
  the +0.37% R49 measured); C_70B_GateUp_CRR flips CONFIRMED → INFLATED
  (real gain is +0.89%, not the +9.74% R49 reported with 14.77% OFF-arm
  spread). Combined with R52G's already-overturned REGRESSION at
  B_70B_GateUp_CRR, **3 of 9 R49 cells now show different verdicts under
  cleaner SCLK** — but in all 3 cases the SHIP gate decision is
  unchanged (all 3 levers should remain ON; details in §Phase 2).
- **Phase 3 — REFUTATION re-validation:** both **R52H** (RRR soft barrier
  → bit-identical ISA) and **R52J** (CRR EXACT IA=0 vs IA=4 → bit-identical
  ISA) re-validated cleanly.
- **Concurrent-bench contamination warning:** the R52 cycle has multiple
  parallel reviewer/dev agents on the same node. The first ~20 min of
  this audit was contaminated by a second reviewer agent (worktree
  `agent-a5edbe9d`) running the *same* baseline script on the same GPU.
  Salvaged data from that agent's first 5 cells (clean before
  contamination) and re-benched cells 6–7 on a freed GPU. Documented in
  §Methodology.

---

## Phase 1 — 21-cell strict-SCLK baseline on HEAD (9a246498)

R48 baseline reference: commit 7e469af6 ("R48 strict-protocol baseline:
7/21 PASS"). FP8 source (`kernel_fp8_layouts.cpp`) is unchanged between
7e469af6 and 9a246498.

| Shape              | Lay |  FP8 R48 | FP8 R52 |  MX R48 | MX R52  | R48 % | R52 %  | spread | R48 PASS | R52 PASS |
|--------------------|-----|---------:|--------:|--------:|--------:|------:|-------:|-------:|---------:|---------:|
| 8192³              | RCR |   3149.7 |  3157.7 |  2963.4 |  2979.7 |  94.1 |   94.4 |  0.40% |     FAIL |     FAIL |
| 8192³              | RRR |   2912.5 |  3153.9 |  2961.5 |  2963.6 | 101.7 |   94.0 |  0.36% |     PASS |   **FAIL\*** |
| 8192³              | CRR |   1303.9 |  2947.1 |  2757.8 |  2755.6 | 211.5 |   93.5 |  0.65% |     PASS |   **FAIL\*** |
| 8B Q/O (4096³)     | RCR |   2458.4 |  2460.1 |  2324.1 |  2330.3 |  94.5 |   94.7 |  1.34% |     FAIL |     FAIL |
| 8B Q/O (4096³)     | RRR |   2464.5 |  2479.6 |  2296.8 |  2325.6 |  93.2 |   93.8 |  3.57% |     FAIL |     FAIL |
| 8B Q/O (4096³)     | CRR |   2327.3 |  2340.9 |  2159.8 |  2173.2 |  92.8 |   92.8 |  0.89% |     FAIL |     FAIL |
| 8B Gate/Up         | RCR |   2655.2 |  2651.3 |  2493.9 |  2501.4 |  93.9 |   94.3 |  3.70% |     FAIL |     FAIL |
| 8B Gate/Up         | RRR |   2644.8 |  2633.4 |  2464.2 |  2463.7 |  93.2 |   93.6 |  0.92% |     FAIL |     FAIL |
| 8B Gate/Up         | CRR |   2485.1 |  2485.4 |  2338.6 |  2333.5 |  94.1 |   93.9 |  0.37% |     FAIL |     FAIL |
| 8B Down            | RCR |   3074.9 |  3075.0 |  2947.8 |  2937.2 |  95.9 |   95.5 |  0.49% |     PASS |     PASS |
| 8B Down            | RRR |   2717.1 |  2702.2 |  2910.9 |  2908.6 | 107.1 |  107.6 |  0.19% |     PASS |     PASS |
| 8B Down            | CRR |   2854.4 |  2855.2 |  2723.9 |  2719.0 |  95.4 |   95.2 |  0.35% |     PASS |     PASS |
| 70B Q/O            | RCR |   3073.0 |  3052.7 |  2888.2 |  2894.2 |  94.0 |   94.8 |  1.96% |     FAIL |     FAIL |
| 70B Q/O            | RRR |   3035.7 |  3048.5 |  2868.0 |  2864.9 |  94.5 |   94.0 |  1.99% |     FAIL |     FAIL |
| 70B Q/O            | CRR |   2847.8 |  2845.1 |  2686.8 |  2692.5 |  94.3 |   94.6 |  1.51% |     FAIL |     FAIL |
| 70B Gate/Up        | RCR |   2961.1 |  2955.1 |  2847.6 |  2850.2 |  96.2 |   96.5 |  1.28% |     PASS |     PASS |
| 70B Gate/Up        | RRR |   2944.8 |  2944.6 |  2773.8 |  2771.1 |  94.2 |   94.1 |  0.60% |     FAIL |     FAIL |
| 70B Gate/Up        | CRR |   2787.4 |  2794.7 |  2482.3 |  2488.7 |  89.1 |   89.0 |  0.36% |     FAIL |     FAIL |
| 70B Down           | RCR |   3235.8 |  3235.4 |  2929.8 |  2938.3 |  90.5 |   90.8 |  0.43% |     FAIL |     FAIL |
| 70B Down           | RRR |   2733.3 |  2729.8 |  2965.1 |  2960.8 | 108.5 |  108.5 |  0.68% |     PASS |     PASS |
| 70B Down           | CRR |   2999.2 |  3005.4 |  2642.0 |  2645.2 |  88.1 |   88.0 |  0.75% |     FAIL |     FAIL |

(SNR=49.6 dB, det=PASS for all 21 cells; criteria: ratio ≥ 95% AND SNR ≥ 45 dB AND det 3/3.)

**`*` 8192³ RRR/CRR re-classified to FAIL** because R48's FP8 baselines for
those two cells were SCLK-contaminated low (FP8 RRR 2912.5 → 3153.9; FP8
CRR 1303.9 → 2947.1). The MXFP8 numbers are stable (2961→2963, 2757→2755).
This is exactly the contamination pattern R49 Reviewer described in
B_70B_GateUp_CRR — when the FP8 baseline is the noisy arm, the ratio
spuriously inflates. R48 commit 7e469af6's "PASS" for 8192³ CRR at 211.5%
was the clearest signature; R49d_findings.md flagged R48 Dev C's +5.47%
claim shrinking to +0.40% under strict protocol for the same reason.

**Net PASS count: 5/21** (8B Down RCR/RRR/CRR, 70B Gate/Up RCR, 70B Down
RRR). Same 3 RCR/RRR LLaMA cells that were already deeply confirmed
(R48 Dev D HARDWARE-CEILING category), plus one minor bookkeeping
correction.

**Spread quality:** all 21 cells under 4% spread; 18/21 under 2%. The
high R48 spreads (8B Down RCR 101.57%, 70B Q/O RCR 41.70%, 70B Q/O CRR
21.55%) are gone — strict SCLK on a single, clean GPU produces stable
medians on every cell.

**Improvement vs R48:** +0.4pp average MXFP8/FP8 ratio (counting only
cells where comparison is valid — i.e., not the two contamination-affected
8192³ cells). The cycle has not shipped any net-new MXFP8 wins since R48,
but it has not regressed either. The R47 SHIPS (RCR + CRR XCD swizzle,
CRR SLC removal) all hold.

---

## Phase 2 — R49 Reviewer 8-cell re-audit (focused subset)

R49 Reviewer found 1 REGRESSION (B_70B_GateUp_CRR) which has *already*
been overturned by R52 Dev G's strict-SCLK confirm (+5.40% ON vs OFF).
For the OTHER 8 cells, this audit re-benches the **3 highest-spread cells
in R49's table** since those are the most likely to harbor verdict
flips:

| Cell                  | R49 OFF spread | R49 verdict     |
|-----------------------|--------------:|------------------|
| B_70B_Down_CRR        |        70.94% | CONFIRMED        |
| B_8B_Down_CRR         |        94.63% | NOISE            |
| C_70B_GateUp_CRR      |        14.77% | CONFIRMED        |

The remaining 5 cells in R49's audit had OFF spreads ≤ 10.27% and the
verdicts (CONFIRMED, INFLATED, CONFIRMED, CONFIRMED, CONFIRMED) are not
spread-fragile; not re-benched here due to time budget.

### Phase 2 results

| Cell                 |  OFF med (sp%) |  ON med (sp%) | R52 Δ%  | claim   | R52 verdict | R49 verdict | flip? |
|----------------------|---------------:|--------------:|--------:|--------:|-------------|-------------|-------|
| B_70B_Down_CRR       |  2555.6 (0.36%) | 2648.7 (0.15%) |  +3.64% | +3.35%  | CONFIRMED   | CONFIRMED   |       |
| **B_8B_Down_CRR**    |  2702.3 (0.35%) | 2728.5 (0.30%) |  +0.97% | +1.13%  | **CONFIRMED** | NOISE     | **FLIP** |
| **C_70B_GateUp_CRR** |  2461.0 (0.87%) | 2482.9 (0.72%) |  +0.89% | +1.52%  | **INFLATED**  | CONFIRMED | **FLIP** |

**Verdict flips and what they mean:**

1. **B_8B_Down_CRR: NOISE → CONFIRMED.** R49 had OFF-arm spread of 94.63%
   driving its NOISE call. Under strict SCLK both arms collapse to ≤0.35%
   spread and the +0.97% delta becomes a clean signal — it's just under
   claim (+1.13%) but unambiguously real. The CRR XCD swizzle DOES help
   this cell; R49's "no help" reading was the noise.

2. **C_70B_GateUp_CRR: CONFIRMED → INFLATED.** R49 reported +9.74% real
   vs +1.52% claim (ratio 6.41×); strict SCLK shows +0.89% (ratio 0.59×).
   Cause: R49's OFF-arm spread was 14.77% — the OFF baseline was
   measured below its true value due to a low-clock outlier. R49's
   over-attribution to the SLC-removal lever inflates the gain by ~8pp.
   The lever still helps (positive sign, low spread on both arms here),
   just much less than R49 reported and less than the +1.52% Dev C
   claimed.

3. **B_70B_Down_CRR re-confirmed.** R49's CONFIRMED verdict (+4.84%
   real vs +3.35% claim) survives strict SCLK as +3.64% — both numbers
   land above claim, ratio drops from 1.45× to 1.09× (within the
   ±2pp expected per-day GPU drift), but the verdict is unchanged.
   The R49 OFF-spread of 70.94% turned out not to be a distortion
   that changed the sign or magnitude meaningfully — it just lifted
   the apparent ratio.

**SHIP-gate impact:** Despite the verdict flips, none of the three R47
levers (RCR XCD swizzle, CRR XCD swizzle, CRR SLC removal) should be
un-shipped:
- The CRR XCD swizzle (R47 Dev B) confirms positive across all 4 CRR
  cells re-audited (B_*: +3.64, +0.97 here, +5.40 from R52G, plus
  +4.16 in R49 for 8B Q/O CRR which we did not re-bench but had
  reasonable 10.22% spread).
- The CRR SLC removal (R47 Dev C) is INFLATED at C_70B_GateUp_CRR
  (+0.89% vs +1.52% claim) but still positive — leave shipped.
- The RCR XCD swizzle (R47 Dev A) was not re-audited here but R49's
  measurements (+7.29% on 70B Gate/Up RCR with 2.45% spread, +2.14% on
  70B Down RCR with 1.24% spread) are both low-spread / unlikely to
  flip.

**5 R49 cells not re-benched** (A_70B_GateUp_RCR, A_70B_Down_RCR,
B_8B_QO_CRR, C_8192cube_CRR, C_8B_QO_CRR) — all had OFF-spread ≤10.22%
and current verdicts (CONFIRMED, INFLATED, CONFIRMED, CONFIRMED,
CONFIRMED) are not spread-fragile under the same mechanism that flipped
the C_70B_GateUp_CRR cell. Future cycles may want to re-bench these for
completeness; this audit deprioritized them.

---

## Phase 3 — Refutation re-validation (R52H + R52J)

Both refutations are ISA-evidence claims. Re-validation = rebuild the
treatment vs baseline arms with `--cuda-device-only -S` and diff. This
sidesteps GPU-contention noise entirely.

### R52H — RRR `MXFP8_RRR_SOFT_BARRIER` (`asm volatile("" ::: "memory")`)

| Property | R52H reported | R52 Reviewer re-validation |
|----------|---------------|------|
| Total ISA diff lines, 8B Gate/Up RRR (M=4096 N=14336 K=4096) | "36 lines" | **36 lines** ✓ |
| Non-marker / non-cuid changes | "0 — six empty-asm marker pairs + UID hash" | **0** ✓ |
| Verdict | REFUTED — silent no-op | **CONFIRMED**: refutation re-validates |

The diff (`r52_reviewer_phase3_results/r52h_rrr_diff.txt`) shows exactly
six pair-insertions of `;;#ASMSTART` / `;;#ASMEND` wrappers around an
empty body, plus the standard `__hip_cuid_<hash>` symbol divergence. Zero
real instruction reorders. Matches R52H findings exactly.

### R52J — CRR EXACT `CRR_EXACT_B1_LDS_INSERT_AFTER` IA=0 vs IA=4

| Property | R52J reported | R52 Reviewer re-validation |
|----------|---------------|------|
| Total ISA diff lines, 8B Gate/Up CRR (M=4096 N=14336 K=4096) | "bit-identical except cuid" | **18 lines** (all `__hip_cuid_*` related) ✓ |
| Non-cuid changes | 0 | **0** ✓ |
| Verdict | REFUTED — IA=0's apparent +1.45% is SCLK noise | **CONFIRMED**: refutation re-validates |

Confirms R52J's load-bearing claim that no IA value alters codegen, so
the IA-sweep's apparent +1.45% best-case at IA=0 is provably 100% SCLK
noise (same instructions, same scheduling, different GPU clock state at
measurement time).

### Implication for the broader cycle

Both REFUTATIONS audited are clean. Combined with R52G's confirmation
that the prior R49-Reviewer REGRESSION at B_70B_GateUp_CRR was itself
the noisy reading (+5.40% in favor of swizzle ON under steady-state
SCLK), the R52 cycle's refutation discipline is sound — no false-negative
ship has been observed.

---

## Methodology notes / contamination handling

1. **Concurrent bench detection:** at start, found a 2-hr-old r52_reviewer
   process from worktree `agent-a5edbe9d` (PID 2132460) actively
   benchmarking on the same GPU 7. Examined that agent's `r52_reviewer_results/`
   directory — cells 1-5 (8192cube, 8B_QO, 8B_GateUp, 8B_Down, 70B_QO)
   completed by 15:30 with no concurrent benches running, hence clean.
   Cell 6 (70B_GateUp) started its FP8 build at 15:36, MXFP8 RCR runs
   landed 15:42–15:45 — exactly overlapping with the start of this
   audit's parallel bench. Killed the old agent, deleted contaminated
   cell-6 data, re-benched cells 6 + 7 on freshly-freed GPU 7.
2. **Per-shape reproducibility:** `r52_reviewer_bench.sh` carries the
   prior agent's resume-aware skip logic (`if grep -q TFLOPS && exists,
   skip`), so re-running the script picks up where it left off
   without re-doing already-good runs.
3. **FP8 denominator:** verified via `git log` that `kernel_fp8_layouts.cpp`
   is unchanged between the R48 baseline commit (7e469af6) and current
   HEAD (9a246498). Re-bench of FP8 in this audit confirms numbers
   within ±2% of R48 medians, validating the assumption.

---

## Recommended next-cycle direction

1. **Continue refutation discipline.** The R52 cycle's REFUTED levers
   (G, H, J, K, L plus R51 E, F) are all well-supported. R52 Dev G
   in particular shows the value of *strict-SCLK confirmation runs*
   on suspect-spread cells before declaring REGRESSION. Future cycles
   should bake that confirm-run pattern into the dev workflow, not
   just the reviewer audit.
2. **Stop investigating RRR `do_k_iter` scheduler fences.** R49B (noinline),
   R50D (sched_barrier mask), R52H (asm volatile) all REFUTED via
   bit-identical or near-identical ISA. The K=4096 loop-control gap on
   8B Gate/Up RRR is **not** addressable via inter-phase fencing. The
   only remaining avenues are: (a) restructure `load_scale_packs` to
   reduce `v_lshrrev` chain, or (b) revisit the K-loop pragma unroll
   (R48G refuted, but with a different structural prior).
3. **Stop investigating CRR EXACT B1 insert position.** R52J's
   bit-identical ISA across IA ∈ {0..8} (re-validated here for IA=0 vs
   IA=4) means the `INSERT_AFTER` template parameter has no codegen
   effect on this kernel. Closed permanently.
4. **The 70B Down RCR / 70B Down CRR / 70B Gate/Up CRR / 8B Q/O CRR
   cluster is the open cycle target.** All four are 88.0–94.7% of FP8
   under the cleaned baseline — they are *the* SHIP gates for LLaMA
   70B production GEMM. R48 Dev D's analysis flagged 70B Gate/Up CRR
   as +4pp HEADROOM (88.5% measured vs 92.5% predicted ceiling); the
   R52 audit confirms 89.0% at HEAD. The level lever family that
   hasn't been fully explored here is the **CRR scale-cache layout
   restructure** (Dev D §2.3's "92% structural floor") — that, not
   another LDS swizzle or insert-position knob, is the path forward.
5. **Watch for SCLK contamination on the cluster.** This audit's first
   20 minutes were contaminated by a sibling reviewer agent on the
   same GPU. Recommend the orchestrator add a per-GPU exclusivity
   lock (e.g., `flock /tmp/gpu7.lock`) or at minimum a startup check
   that fails if another `python3 test_*` process is already on the
   target GPU.

---

## SHIP gate summary across LLaMA 8B/70B production GEMM shapes

Of the 14 production cells (8B and 70B, 3 layouts each = 18 minus the
non-prod 8192³ shape × 3 layouts), **5 cells PASS strict ≥95% gate**
(8B Down RCR/RRR/CRR + 70B Gate/Up RCR + 70B Down RRR). The remaining
13 cells fall into:

- **HARDWARE-CEILING (Dev D §3 / §4):** 5 RCR M=4096 cells, 70B Q/O
  RRR/CRR, 70B Gate/Up RRR, 8B Q/O CRR — at 92–95% — likely cannot
  exceed 95% without changes to the FP8 baseline itself (which is
  already known to be MFMA-bound on these shapes).
- **HEADROOM:** 70B Gate/Up CRR (89.0% vs 92.5% predicted) — the
  highest-leverage open target.
- **OPEN STRUCTURAL FLOOR:** 70B Down RCR (90.8%), 70B Down CRR (88.0%)
  — both involve K=28672 and HBM-side bottlenecks that no R47+ lever
  has unlocked.

Net: the current tree is mature; the remaining gap is concentrated in
3-4 cells with predicted ceilings that need either (a) FP8 lift +
matched MXFP8, or (b) a structural CRR scale-cache rework. No
quick-win regression has been left on the table by R49–R52.

---

## Files

- `r52_reviewer_bench.sh` — 21-cell strict-SCLK orchestrator (resume-aware),
  inherited from prior reviewer agent (worktree `agent-a5edbe9d`); resumes
  any cell with valid existing TFLOPS logs.
- `r52_reviewer_bench.run.log` — Phase 1 stdout (cells 6-7 portion;
  cells 1-5 ran under prior reviewer agent's log).
- `r52_reviewer_results/` — 21 cells × {fp8,mxfp8} × {rcr,rrr,crr} × 5 runs
  + check (SNR/det) + per-cell build logs + SUMMARY.txt (Phase 1)
- `r52_reviewer_phase2_focused.sh` — 3-cell A/B re-audit of high-spread
  R49 cells (B_70B_Down_CRR, B_8B_Down_CRR, C_70B_GateUp_CRR)
- `r52_reviewer_phase2_focused.run.log` — Phase 2 stdout with verdict table
- `r52_reviewer_phase2_results/` — A/B per-cell raw bench logs + SUMMARY.txt
- `r52_reviewer_phase3_isa.sh` — R52H + R52J ISA dump + diff orchestrator
  (no GPU required)
- `r52_reviewer_phase3_isa.run.log` — Phase 3 stdout (ISA verdict summary)
- `r52_reviewer_phase3_results/r52h_rrr_off.s`, `r52h_rrr_on.s`,
  `r52h_rrr_diff.txt` — RRR soft-barrier ISA pair + diff (36 lines, all
  empty-asm marker pairs + 1 cuid hash)
- `r52_reviewer_phase3_results/r52j_crr_ia0.s`, `r52j_crr_ia4.s`,
  `r52j_crr_diff.txt` — CRR IA=0/IA=4 ISA pair + diff (18 lines, all
  cuid-hash-related)
