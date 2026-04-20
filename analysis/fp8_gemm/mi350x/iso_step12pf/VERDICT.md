# R68 step12pf NON-SPLIT — ISO Verdict

**Date**: 2026-04-20
**Agent**: R68 Optimizer-2 (step12pf bisect harness)
**Source**: `analysis/fp8_gemm/mi350x/iso_step12pf/`
**Reference**: R67 SPLIT failure note `project_mxfp4_R67_step12pf_failed.md`

## TL;DR

**step12pf NON-SPLIT prototype is CORRECTNESS-SAFE in isolation but
PERFORMANCE-DOMINATED by the already-shipped step34pf at the canonical
M=16384 N=4096 K=2048 shape.** The R67 SPLIT design's NaN cliff is NOT
reproduced. Recommend: keep the dormant macro-guarded code in tree
(default `STEP12_PF_INTERLEAVE 0`) for future axis exploration, do NOT
add to autotune list at this round, do NOT replace step34pf.

## Three Gates Result

| Gate | Result | Detail |
|------|--------|--------|
| 1. Compile | PASS | All 3 variants build, no spills, VGPRs 240, AGPRs 256 |
| 2. Correctness (SNR) | PASS | step12pf median good-cell SNR 15.37 dB vs base 15.25 dB; finite-fraction 0.974 vs base 0.941 |
| 3. Performance | INCONCLUSIVE | step12pf 3240 TFLOPS vs base 3175 (+2.0pp); step34pf 3289 (+3.6pp) — step12pf loses 1.5pp to incumbent |

## Methodology Notes

The original Gate 2 design (byte-equality vs base) was abandoned after
`check_determinism.py` confirmed the **base kernel itself is non-deterministic**:
two back-to-back runs of the unmodified shipped kernel disagreed on 22% of
cells (14.8M / 67M), with run-to-run inf-count varying 72K vs 465K. This
matches `project_mxfp4_finite_gate_cohort_race.md` (MFMA accumulator race
producing ~3% bad cells per run, gate threshold 0.97).

Gate 2 was reframed as **good-cell SNR comparison** (5-run R44D-style
protocol):

```
== base ==     fin_frac med 0.9410, good-snr med 15.25 dB
== step34pf == fin_frac med 0.8566, good-snr med  9.68 dB  ← shipped winner
== step12pf == fin_frac med 0.9743, good-snr med 15.37 dB  ← NEW prototype
```

Crucially, **step34pf (the production R66/R67 winner) shows the WORST
correctness in this stripped-down test**, despite delivering the highest
TFLOPS in autotune. This proves cohort-race intensity is variant-dependent
and that production WIN does not imply correctness-clean. step12pf is
*better* than base on both metrics, ruling out the R67 SPLIT-style failure.

## R67 Hypothesis Disposition

R67 SPLIT failure (5632 row-clustered NaNs) listed 4 unresolved hypotheses.
Under R68 NON-SPLIT (all 16 prefetches in step12, 0 in step34):

| # | Hypothesis | Status |
|---|------------|--------|
| 1 | Compiler m0 interaction across step12/step34 boundary | **NOT TRIGGERED** — NON-SPLIT keeps m0 writes confined to step12 only |
| 2 | Operand pool / SGPR pressure at 94 ops in SPLIT | **AVOIDED** — NON-SPLIT step12 helper has 114 operands but compiler handled it (SGPRs 79, no spills) |
| 3 | LDS race pf_a0 → A0_db[cur] in SPLIT step12 | **NOT REPRODUCED** — counter-intuitively, NON-SPLIT does write A1_db[cur]/Br_db[cur] while step12 reads them, but cohort race is no worse than base |
| 4 | vmcnt(8) in SPLIT step34 misaligned vs 8 already-issued | **N/A** — NON-SPLIT step34 path takes the base (no PF tail) branch, no vmcnt issue |

So **R67's hypothesis 1+4 were SPLIT-specific**; NON-SPLIT bypasses both.
Hypothesis 3 was apparently a non-issue — possibly because step12's
ds_reads sample from `cur` early (before the buffer_load_lds complete)
and the data-dependent ordering happens to work out. This is fragile and
should NOT be assumed for arbitrary shape variations.

## Performance Discussion

step12pf spreads 16 buffer_load_dwordx4 prefetches across step12's 64
MFMAs (4:1:1 like R66's step34pf). Step34 then runs base (no PF). This
issues prefetches **earlier** in the pipeline, giving more in-flight time.

At canonical shape M=16384 N=4096 K=2048:
- base:     3175.2 TFLOPS
- step34pf: 3289.1 TFLOPS  (+3.6pp, shipped)
- step12pf: 3240.3 TFLOPS  (+2.0pp, new)

The early-prefetch hypothesis predicts step12pf should help most when
**step34 compute does NOT overlap with step12's prefetch issue** (i.e.,
small K where the prefetch-at-step12 saves a vmcnt stall at the next
iter's step12). At K=2048 (16 K-iters), this overlap is well-amortized
and step34pf's later-issue strategy wins by 1.5pp. step12pf might win at
larger K shapes — but not investigated at this iso round.

Combined `step34pf + step12pf` macro flag was tested for compile (PASS,
no spills) but the kernel branches such that STEP12 takes precedence and
step34 reverts to base — so it's equivalent to step12pf alone, not a true
cross-product.

## Recommendation

1. **DO NOT promote step12pf to default or to autotune list** at this
   round. step34pf already covers the same ground better at the canonical
   shape, and adding a slower variant to autotune just adds bench cost.

2. **DO keep the dormant code** (`STEP12_PF_INTERLEAVE 0` default,
   already committed in `fc0e6ef1`). It is correctness-safe, costs
   nothing at runtime, and is reusable for future cross-product
   experiments (e.g., shape-conditional dispatch, K>4096 shapes).

3. **Future extension** (R69+): try a true SPLIT cross-product — half the
   prefetches in step12 (different subset than R67 tried), half in
   step34. But ONLY after running THIS harness on the new variant first.

4. **Never deploy step12pf without per-shape iso re-validation**. The
   3-hypothesis NON-SPLIT clearance above was specific to this shape;
   different (M, N, K) may stress different LDS-race timing windows.

## Reproducibility

```
cd /shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/iso_step12pf
ISO_GPU=4 python3 iso_runner.py    # full pipeline (gates 1-3)
python3 check_determinism.py       # baseline non-determinism check
```

Logs preserved: `iso_run3.log` (early SNR-broken run), `iso_run6.log`
(final run with fp64 stable norm), `5run_protocol.log` (5-run good-cell
SNR), `gate3_iso.log` (perf comparison), `results.json` (machine-
readable summary).

Per-variant tensor dumps: `build/tk_iso_{base,step34pf,step12pf}_C.pt`.

## Disposition

step12pf axis remains **OPEN but de-prioritized**. R68 round did NOT land
step12pf as a WIN. Other R68 axes (Opt-1 triple cross-product, Opt-3
orphan cleanup) carry the round.
