# R49 Dev D: Per-shape GROUP_M=8 gate for 8B Gate/Up CRR — REFUTED

## Summary

R48 Dev C (commit `2d79bb04`, `r48c_findings.md`) reported **+5.47%** on 8B
Gate/Up CRR (M=4096, N=14336, K=4096) for `MXFP8_CRR_BLOCK_SWIZZLE_GROUP_M=8`
vs default GROUP_M=4 under a 3-run/20s-cooldown protocol on GPU 5.

R49 Dev D's mandate: re-validate that +5.47% on a clean isolated GPU under
strict SCLK (5 runs/30s cooldown), and if real, ship it via a per-shape
compile-time gate (`#if N_DIM == 14336`).

**Verdict: REFUTED.** Strict re-bench shows GM=8 vs GM=4 = **+0.40% median /
+0.32% mean** on 8B Gate/Up CRR — 13× smaller than R48 Dev C's reported
+5.47%, well below the +1.5% noise floor, and comparable to the within-cell
spread (0.74-1.45%). No structural lever exists. R48 Dev C's measurement was
contaminated noise.

No kernel change. No Phase 2/3/4 executed.

## Phase 1: 8B Gate/Up CRR re-bench (5 runs, strict SCLK)

GPU 7 isolated. Builds: `-DM_DIM=4096 -DN_DIM=14336 -DK_DIM=4096
-DMXFP8_CRR_BLOCK_SWIZZLE=1 -DMXFP8_CRR_BLOCK_SWIZZLE_GROUP_M=<GM>`.
30s inter-run cooldown, 60s rebuild cooldown.

| Cond.   | run1   | run2   | run3   | run4   | run5   | median  | mean    | spread |
|---------|-------:|-------:|-------:|-------:|-------:|--------:|--------:|-------:|
| GM=4    | 2277.6 | 2287.7 | 2272.5 | 2289.4 | 2274.9 | 2277.64 | 2280.43 |  0.74% |
| GM=8    | 2307.7 | 2274.6 | 2282.3 | 2287.9 | 2286.7 | 2286.66 | 2287.82 |  1.45% |

**Δ median GM=8 vs GM=4: +0.40%**
**Δ mean   GM=8 vs GM=4: +0.32%**

### Decision criteria (per mandate)

- ≥+3.0% → PROCEED to Phase 2 → **NOT MET** (+0.40%)
- +1.5% to +3.0% → MARGINAL judgment call → **NOT MET** (+0.40%)
- <+1.5% or noise-dominated → REFUTED → **MET**

Both conditions for REFUTED are met:
- (a) Δ median +0.40% < +1.5% threshold.
- (b) GM=8 spread (1.45%) > effect size (0.40%) → noise-dominated.

## Why R48 Dev C reported +5.47%

R48 Dev C's protocol: 3 runs/20s cooldown on GPU 5 (shared with other devs at
the time of the R48 cycle). Two plausible explanations for the discrepancy:

1. **Concurrent-load contamination**: GPU 5 may have been thermal/power-state
   perturbed by neighboring GPU work during the gm4 baseline runs (deflating
   them), inflating the apparent gm8 win.
2. **3-run sample variance**: With within-cell spread already ~1-2% and N=3,
   a single low gm4 outlier shifts the apparent median by ~1-2pp. Combined
   with a high gm8 outlier, the gap can balloon to ~5%.

Cross-check with R48 Dev C's own Step 2 (70B Gate/Up GROUP_M sweep, 3 runs
on the *same* GPU 5) showed GM=8 at only +0.66% there. The +5.47% on 8B
Gate/Up was an outlier within R48 Dev C's own data — strict re-bench
confirms ~+0.4% is the true effect for this lever family.

This pattern matches the broader R48 Dev A finding that GPU-5-era 3-run data
is contaminated and reproduction under strict SCLK collapses ~5pp deltas.

## Files added

- `r49d_phase1.sh` — strict-SCLK bench script (5×/30s/60s rebuild)
- `r49d_phase1.run.log` — full run log
- `r49d_phase1_results/` — per-run TFLOPS logs (10 runs total)
- `r49d_findings.md` — this document

## Files modified

None. `MXFP8_CRR_BLOCK_SWIZZLE_GROUP_M` default remains 4 in
`crr_mxfp8_exact_8wave_fastpath.inc:68-70`.

## Recommendations for R50+

1. **Do not re-test `MXFP8_CRR_BLOCK_SWIZZLE_GROUP_M` on 8B Gate/Up CRR**.
   The lever is structurally inert at this shape (≤±0.5% effect under noise
   floor). R47 Dev B's default (GM=4) is locally optimal across all 7 CRR
   shapes when measured under strict SCLK.
2. **Mark R48 Dev C's Step 3 +5.47% finding as superseded by R49 Dev D**.
   The full-sweep Δ% column in `r48c_findings.md` should be treated as
   noise-dominated; the only reliable number from that table is the
   "no-regression" pattern (no GM affects any shape by >2% in either
   direction under clean conditions).
3. **For future single-shape outlier hunts in CRR**, require ≥5 runs +
   isolated GPU before logging a delta as "real" — the wide-N CRR shapes
   appear to have ~1-2% within-cell variance that easily masquerades as
   structural lever wins under N=3 protocols.
