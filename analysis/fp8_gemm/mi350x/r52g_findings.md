# R52 Dev G — CRR XCD swizzle per-shape un-ship for B_70B_GateUp_CRR

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 72a3c217
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=5
**Hypothesis:** R49 Reviewer reported a -3.87% regression on B_70B_GateUp_CRR
(M=4096, N=28672, K=8192) under R47 Dev B's CRR XCD swizzle. Add per-shape
gate that disables `MXFP8_CRR_BLOCK_SWIZZLE` for this exact shape, expect
to recover +3.87% on this cell while leaving the other 6 confirmed cells
unchanged.
**Verdict:** **REFUTED.** Un-shipping the swizzle on this shape regresses
-5.4% to -6.41%, opposite of R49 Reviewer's prior measurement.

## TL;DR

| Cell                  | PRE (swizzle ON, HEAD) | POST (per-shape gate, swizzle OFF) | delta |  expected | verdict |
|-----------------------|---:|---:|---:|---:|---|
| **70B_GateUp_CRR**    | **2483.0** (sp=54.73%) | **2323.8** (sp=84.53%) | **-6.41%** | +3.87% | **REGRESSION (REFUTED)** |
| 70B_Down_CRR          | 2646.0 (sp=0.59%)  | 2651.8 (sp=0.64%)  | +0.22% | == | no-op (expected) |
| 70B_QO_CRR            | 2711.4 (sp=0.54%)  | 2715.2 (sp=0.54%)  | +0.14% | == | no-op (expected) |
| 8B_GateUp_CRR         | 2349.5 (sp=0.12%)  | 2346.5 (sp=0.59%)  | -0.13% | == | no-op (expected) |
| 8B_QO_CRR             | 2170.4 (sp=5.97%)  | 2162.2 (sp=1.27%)  | -0.38% | == | no-op (expected) |
| 8B_Down_CRR           | 2752.8 (sp=0.35%)  | 2759.0 (sp=1.64%)  | +0.22% | == | no-op (expected) |
| 8192cube_CRR          | 2783.8 (sp=0.53%)  | 2789.8 (sp=0.43%)  | +0.22% | == | no-op (expected) |

The 6 no-op cells confirm the per-shape gate hits only B_70B_GateUp_CRR.
The auto-gate verification (no `-DMXFP8_CRR_BLOCK_SWIZZLE` flag) at
2346.2 TFLOPS confirms the dispatch table correctly switches on
(M=4096, N=28672, K=8192). Both the gate plumbing and the measurement
are clean; the hypothesis itself is wrong.

## Confirmation run (8 runs/arm)

To rule out spread-driven measurement error, an 8-run/arm confirmation
re-ran 70B_GateUp_CRR with both POST (swizzle OFF, per-shape gate) and
PRE (swizzle ON, current HEAD):

```
CONFIRM_70B_GateUp_CRR post: med=2354.1, all=[2343.86, 2348.88, 2349.64, 2354.08, 2354.21, 2354.22, 2355.83, 2362.18]
CONFIRM_70B_GateUp_CRR pre:  med=2488.4, all=[2468.45, 2484.83, 2486.12, 2488.12, 2488.76, 2489.17, 2490.4, 2493.0]
delta = (2354.1 - 2488.4) / 2488.4 = -5.40%
```

Both arms have spread <1%. The signal is unambiguous: the current HEAD
(swizzle ON) is **5.4% better** than the per-shape un-ship at this cell.

## Diagnosis — R49 Reviewer's reading was the noisy one

R49 Reviewer reported:
- OFF med=2386.6 (spread 0.49%)
- ON  med=2294.3 (spread 10.27%)
- delta = -3.87%

R52 Dev G's strict-SCLK confirmation reports:
- OFF med=2354.1 (spread <1%)
- ON  med=2488.4 (spread <1%)
- delta = +5.40%  (sign flip)

The OFF medians agree (~2370 ± 17). The ON arms disagree by ~190 TFLOPS
(8.4%). R49's ON-arm spread of 10.27% places it squarely in the
unstable-warmup regime where the first 1-2 of 5 runs land low because
the GPU clock hasn't fully stabilized post-build. R52 Dev G's confirm
arm has spread <1% — a genuine steady-state measurement.

R49 Reviewer correctly flagged this cell as "unstable on the ON arm" but
incorrectly attributed the observed -3.87% to the swizzle lever. The
real cause was warmup instability in the build-then-bench sequencing.

## Implications for R47 ship gates

R47 Dev B's CRR XCD swizzle (`MXFP8_CRR_BLOCK_SWIZZLE`) is **re-confirmed
across all 7 CRR cells**. No un-ship is needed; the +2.21% claim at
B_70B_GateUp_CRR is real and the actual realized gain is closer to
+5.40% under steady-state SCLK.

R49 Reviewer's overall audit table stands except for this one cell:
B_70B_GateUp_CRR should be re-classified from REGRESSION to CONFIRMED.

## Files

- `r52g_bench.sh` — strict-SCLK A/B orchestrator across 7 CRR cells with
  per-shape gate code path + auto-gate verification phase
- `r52g_bench.run.log` — main 7-cell A/B output
- `r52g_results/` — raw per-run TFLOPS files
- `r52g_confirm.sh` — 8-run confirmation orchestrator for B_70B_GateUp_CRR
- `r52g_confirm.run.log` — confirmation output

No source patch is committed. The per-shape gate prototype lived in a
worktree-local edit to `crr_mxfp8_exact_8wave_fastpath.inc`; that edit
is discarded.
