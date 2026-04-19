## R48 Reviewer Audit — Cycle WIP @ 2026-04-19

**Reviewer scope:** R48 Devs A/B/D/E (4 completed commits on `feat/mxfp8-only`)
**Method:** Static (no GPU). Verify findings claims against bench logs, ISA dumps,
and source code. No kernel changes.

### Per-Dev Verdict

| Dev | Commit | Claim | Verdict | Notes |
|---|---|---|---|---|
| A | `db92bdda` | REFUTED — SCLK contamination of R47 baseline; clean re-bench 94.6% | **VERIFIED** | All 10 bench log TFLOPS values match findings tables exactly. Asymmetry argument (FP8 −0.7%, MXFP8 +3.2%) is logically sound for SCLK-vs-throttling differential sensitivity. |
| B | `80bf861e` | REFUTED — V2 layout has no inter-warp B-scale sharing; B-first reorder no-op | **VERIFIED** | Slab geometry confirmed at `kernel_mxfp8_layouts.cpp:2611` and `crr_mxfp8_exact_8wave_fastpath.inc:428`: `slab_idx_b = bc * WARPS_N + wn` — warps in same CTA load disjoint slabs. Both bench logs match findings (RCR med 2922.35 baseline / 2927.67 COOP, CRR 2638.59 / 2632.42). Δ within ±0.5%. |
| D | `7ffd083b` | HARDWARE-CEILING analysis: 11/15 cells at predicted ceiling | **VERIFIED w/ minor caveats** | Per-K-pair instruction counts match ISA dumps exactly (FP8 RCR 64/48/16, MXFP8 RCR 64/48/16/2/11/16/29 — all confirmed). Ceiling-vs-measured table shows minor ratio drift (≤1pp) vs `r47_full_baseline_SUMMARY.txt` — Dev D appears to have used slightly re-benched anchors but conclusions don't shift. |
| E | `c46227d0` | REFUTED — 392 scratch ops, 67 spill bytes, ~70% regression | **VERIFIED** | ISA verification confirmed: baseline 1711 lines / 0 scratch ops / 1 inner-loop header; unroll=32 8355 lines / **341 scratch_load + 51 scratch_store = 392 scratch ops** / 0 inner-loop header / 1024 v_mfma — exact match to findings claim. Bench mean 679.8 TFLOPS for 8B QO (= mean of 808.24, 420.00, 811.07 in run logs) confirmed. Resource log shows VGPRs=256, ScratchSize=208 bytes/lane at unroll=2 and unroll=8. |

**Verdict counts: VERIFIED 3, CAVEATS 1, DISPUTED 0.**

---

### Cross-Cutting Issues

1. **Dev D ratio drift vs R47 baseline summary.** Dev D's verdict table reports
   measurements like 8B Q/O RCR 95.3%, 70B Gate/Up RCR 95.6%, 70B Gate/Up CRR
   88.5% — but `r47_full_baseline_SUMMARY.txt` lists 95.0%, 95.9%, 87.7% for
   the same cells. Magnitudes are within ±1pp (within-GPU noise envelope), and
   the Dev A finding about SCLK contamination of the R47 baseline supports the
   view that the SUMMARY itself drifts ±1pp run-to-run. **Dev D should cite
   which run it anchored to** — currently the table's provenance is implicit.
   Does NOT change any verdict (every cell is still in the same CEILING /
   HEADROOM bucket).

2. **Dev E run2 unroll=32 outlier.** 8B QO unroll=32 run2 logged 420.00 TFLOPS
   vs run1 811.07 / run3 808.24 (~50% lower than peers). Findings reports
   3-run **mean** = 679.8, which is mathematically correct. But this is a
   ~80% intra-run spread — well beyond the ±2% SCLK envelope. This does
   *not* change the REFUTED verdict (even the 808 best-case is −66% vs
   2368 baseline), but the run2 outlier itself is likely a separate
   contamination event during Dev E's measurement window. Worth flagging
   so future investigators don't think 420 is the kernel's lower bound.

3. **Dev D ↔ Dev E linkage is sound.** Dev D §6 P3 explicitly bracketed both
   outcomes ("+1-2pp gain OR ≤0pp spill regression"). Dev E hit the
   spill-regression branch and correctly cites this. The two findings are
   non-contradictory and self-consistent.

4. **All R48 macros confirmed default-OFF in the tree:**
   - `MXFP8_RCR_COOPERATIVE_BSCALE` — `kernel_mxfp8_layouts.cpp:427` (`#define ... 0`)
   - `MXFP8_CRR_COOPERATIVE_BSCALE` — `crr_mxfp8_exact_8wave_fastpath.inc:93` (`#define ... 0`)
   - `MXFP8_RRR_MAIN_UNROLL` — `rrr_mxfp8_exact_8wave_fastpath.inc:46` (`#define ... 0`)

   One-flag repro is preserved per cycle protocol.

---

### Recommendations for Running Devs C / F / G

#### Dev C — CRR wide-N structural floor
- **Use Dev D's 92% structural floor as your prior** (§2.3 of r48d_findings.md):
  CRR has 6 `v_lshrrev_b32` scale-shift ops per K-pair from non-opsel-friendly
  scale layout. Without restructuring scale layout, the predicted ceiling is
  ~92%. **Dev D's top-priority cell — 70B Gate/Up CRR (88.5% measured vs
  92.5% predicted, +4pp gap) — is your target.** Dev D recommends
  `MXFP8_CRR_V2_SCALE_CACHEPOLICY` × `MXFP8_CRR_LDS_SWIZZLE` sweep on
  N=28672 specifically.
- **Risk of duplication:** If your hypothesis is "CRR is MFMA-bound," it is
  refuted by Dev D before you start (CRR has only 32 MFMA per K-pair, half
  RCR/RRR). Don't burn cycles on MFMA-issue-rate levers.
- **SCLK protocol:** Use Dev A's protocol (5× back-to-back, 30s cooldown,
  isolated GPU). Don't rely on R47 baseline directly for ±2pp claims.

#### Dev F — A-scale path on 70B Down RCR
- **Dev B already established:** the 70B Down RCR/CRR gap is "not in the
  B-scale loading path" (per r48b_findings.md §4) and points future work to
  A-side. Your hypothesis is well-aligned.
- **Critical prior from R31C / Dev B §1:** V2-RCR is at register-pressure
  ceiling (256 VGPRs / 312 spill / 596 bytes scratch when prefetch=1). Any
  A-scale lever that adds prefetch depth will spill. Test resource usage
  EARLY before benching.
- **Risk of duplication:** The `MXFP8_RCR_EXACT_PQ_SCALE_LDS_ENABLE` LDS
  scale-cache path was already tried (R30C) and broke correctness. Don't
  re-derive. Confirm zero `ds_write` in your variant's SASS before
  attempting.
- **Use Dev D's measured ceiling:** 70B Down RCR FP8 baseline likely has
  L2/HBM-side bottlenecks (K=28672 = largest K). Dev D didn't include 70B
  Down in the verdict table — you may want to add that cell to the ISA
  comparison.

#### Dev G — RRR partial unroll U=2/4/8/16 sweep
- **Dev E already swept u=2/4/8/32** (see r48e_findings.md §3.2 + the
  `r48e_unroll{2,4,8}_*_resource.log` files). All produced **identical 208
  bytes/lane scratch** and ~70% regression. **You will duplicate Dev E's
  sweep** unless your variant is structurally different (e.g., changing
  `do_k_iter` from `always_inline` to noinline first, or adding manual
  `__builtin_amdgcn_kill` hints, per Dev E §4.1).
- **Strong recommendation: BEFORE benching, dump device .s and check
  ScratchSize / VGPR spill bytes.** Dev E's resource logs show every unroll
  ≥ 2 spills with the current `do_k_iter` lambda. If your variant doesn't
  change `do_k_iter`'s structure, expect the same outcome.
- **If your sweep includes u=16, it is the only un-tested value** — but
  Dev E's u=2/4/8/32 monotonic regression makes u=16 nearly certain to
  also regress. If you do test it, prioritize ISA dump first.

---

### Recommendations for R48 Wrap

1. **3 SHIP-zero, 1 HARDWARE-CEILING analysis** is a defensible cycle outcome
   given the cycle started post-R47 (which already mopped up the easier
   wins). All 4 REFUTED verdicts are cleanly supported by artifacts; none
   look like premature give-up. Dev D's analysis is the keeper deliverable.

2. **Bake Dev A's SCLK protocol into the Cycle README.** "5× back-to-back, 30s
   cooldown, single GPU, no concurrent benches on the same node" — Dev A
   demonstrated a phantom 6pp regression that vanished under this protocol.
   Future cycles should adopt before any "regression hunt."

3. **Refresh `r47_full_baseline_SUMMARY.txt` (or add an `r48_*` companion)
   using Dev A's protocol.** Dev D's table already drifts ±1pp from the
   summary; the summary itself may have been captured during the same
   contamination window Dev A identified.

4. **Cite Dev D's STOP list when judging future SHIP candidates.** Specifically:
   any R49+ Dev claiming >+1pp on a CEILING cell (5 RCR M=4096 cells, 70B
   Q/O RRR/CRR, 70B Gate/Up RRR, 8B Down all layouts, 8B Gate/Up CRR) needs
   to either (a) lift FP8 baseline equally (proving non-MFMA bottleneck) or
   (b) re-benchmark per Dev A protocol to rule out SCLK noise.

---

### Top-3 Escalations

1. **Dev D verdict-table provenance.** Sub-1pp drift vs `r47_full_baseline_SUMMARY.txt`
   is harmless for the bucket assignments but unstated. Recommend Dev D
   amend §4.1/§4.2/§4.3 to cite the specific bench log per cell, OR add a
   one-line note that ratios are post-Dev-A-protocol re-bench.

2. **Dev E run2 unroll=32 outlier (420 TFLOPS).** 50% intra-run spread
   indicates concurrent contamination during Dev E's bench window. Doesn't
   change REFUTED verdict but means the "−71.30%" headline is itself
   noise-contaminated; the true regression is closer to −66% (best-case run).
   Dev E should add a footnote, or running Devs F/G should add a sanity
   sentinel run to detect they're not in the same contamination window.

3. **Running Dev G is at high duplication risk with Dev E.** Dev G's
   "RRR partial unroll U=2/4/8/16 sweep" overlaps Dev E's u=2/4/8/32 sweep
   (Dev E §3.2). Unless Dev G's hypothesis includes structural changes to
   `do_k_iter`, the result is pre-determined. **Recommend: gate Dev G with
   an ISA scratch-bytes pre-check before any benching** — saves a full GPU
   cycle on a known-failed lever family.
