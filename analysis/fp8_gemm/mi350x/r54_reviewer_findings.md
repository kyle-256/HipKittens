# R54 Reviewer Baseline 9-Cell Sweep — Findings

**Date:** 2026-04-20
**Source:** `analysis/fp8_gemm/mi350x/r54_reviewer_results/baseline_9cell/`
**Method:** Median TFLOPS over 5 runs per (shape × layout × dtype). SHIP gate = MXFP8 / FP8 >= 0.95.

## 9-cell results table

| Shape       | Layout | FP8 median (TFLOPS) | MXFP8 median (TFLOPS) | Ratio %  | pp from 95% | Verdict   |
|-------------|--------|---------------------|------------------------|----------|-------------|-----------|
| 8B_GateUp   | rcr    | 2632.66             | 2503.97                | 95.11%   | +0.11       | PASS      |
| 8B_GateUp   | rrr    | 2618.92             | 2472.05                | 94.39%   | -0.61       | HEADROOM  |
| 8B_GateUp   | crr    | 2490.21             | 2331.17                | 93.61%   | -1.39       | HEADROOM  |
| 70B_GateUp  | rcr    | 2963.26             | 2842.97                | 95.94%   | +0.94       | PASS      |
| 70B_GateUp  | rrr    | 2942.97             | 2770.94                | 94.15%   | -0.85       | HEADROOM  |
| 70B_GateUp  | crr    | 2798.39             | 2489.06                | 88.95%   | -6.05       | HEADROOM  |
| 70B_QO      | rcr    | 3049.69             | 2893.99                | 94.89%   | -0.11       | HEADROOM  |
| 70B_QO      | rrr    | 3037.57             | 2871.05                | 94.52%   | -0.48       | HEADROOM  |
| 70B_QO      | crr    | 2845.46             | 2691.51                | 94.59%   | -0.41       | HEADROOM  |

**Split:** 2/9 PASS, 7/9 HEADROOM.

## Correctness / determinism

All 9 cells: SNR 49.59-49.61 dB (threshold 45.0), pass rate 100.00%, determinism PASS over 3 runs. No build errors. No anomalies.

## Summary

The R54 reviewer baseline regresses against the prior 5/9 PASS, 4/9 HEADROOM baseline: the new sweep shows **2/9 PASS, 7/9 HEADROOM**. The two surviving PASS cells are both RCR (8B_GateUp/rcr at 95.11%, 70B_GateUp/rcr at 95.94%), which is consistent with RCR being the hottest-tuned layout in the codebase. The previously-passing 70B_QO/rcr cell has slipped just below the gate (94.89%, -0.11 pp), and the entire 70B_QO row is now uniformly tight (94.5-94.9%) — these are all marginal, sub-1pp misses likely recoverable with modest tuning. The standout regression is **70B_GateUp/crr at 88.95% (-6.05 pp)**, structurally distinct from the rest of the headroom band and the dominant single-cell loss in the sweep; this matches the CRR scale-shift floor pattern called out in prior R54 Dev E/F notes. RRR cells (8B and 70B GateUp) sit in the 94.1-94.4% band consistent with the V2 RRR 254/256 VGPR ceiling already quadruply closed (R49A/R53A/R53B/R54C).

**HEADROOM cells for R55 to attack:**
1. 8B_GateUp/rrr (-0.61 pp)
2. 8B_GateUp/crr (-1.39 pp)
3. 70B_GateUp/rrr (-0.85 pp)
4. **70B_GateUp/crr (-6.05 pp)** — largest deficit, structurally distinct
5. 70B_QO/rcr (-0.11 pp)
6. 70B_QO/rrr (-0.48 pp)
7. 70B_QO/crr (-0.41 pp)
