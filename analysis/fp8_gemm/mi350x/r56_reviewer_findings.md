# R56 Reviewer Baseline 9-Cell Sweep — Findings

**Date:** 2026-04-20
**Branch:** `feat/mxfp8-only`
**HEAD:** `0e068a54` (R54+R55 cycle wrap)
**Source:** `analysis/fp8_gemm/mi350x/r56_reviewer_results/baseline_9cell/`
**Method:** Median TFLOPS over 5 runs per (shape × layout × dtype). Strict-SCLK protocol: 30s cooldown, 60s rebuild cooldown, WARMUP=100, ITERS=200. SHIP gate = MXFP8 / FP8 >= 0.95.
**GPU:** `HIP_VISIBLE_DEVICES=7`.

## 9-cell results table

| Shape       | Layout | FP8 median (TFLOPS) | MXFP8 median (TFLOPS) | Ratio %  | pp from 95% | SNR (dB) | det  | Verdict   |
|-------------|--------|---------------------|------------------------|----------|-------------|----------|------|-----------|
| 8B_GateUp   | rcr    | 2652.60             | 2500.17                | 94.25%   | -0.75       | 49.61    | PASS | HEADROOM  |
| 8B_GateUp   | rrr    | 2641.27             | 2467.13                | 93.41%   | -1.59       | 49.61    | PASS | HEADROOM  |
| 8B_GateUp   | crr    | 2486.12             | 2324.17                | 93.49%   | -1.51       | 49.61    | PASS | HEADROOM  |
| 70B_GateUp  | rcr    | 2972.25             | 2842.46                | 95.63%   | +0.63       | 49.59    | PASS | **PASS**  |
| 70B_GateUp  | rrr    | 2941.94             | 2781.83                | 94.56%   | -0.44       | 49.60    | PASS | HEADROOM  |
| 70B_GateUp  | crr    | 2791.49             | 2486.20                | 89.06%   | -5.94       | 49.60    | PASS | HEADROOM  |
| 70B_QO      | rcr    | 3045.84             | 2892.16                | 94.95%   | -0.05       | 49.60    | PASS | HEADROOM  |
| 70B_QO      | rrr    | 3052.83             | 2873.48                | 94.13%   | -0.87       | 49.59    | PASS | HEADROOM  |
| 70B_QO      | crr    | 2838.67             | 2689.50                | 94.75%   | -0.25       | 49.59    | PASS | HEADROOM  |

**Split: 1/9 PASS, 8/9 HEADROOM.**

## Correctness / determinism

All 9 cells: SNR 49.59-49.61 dB (threshold 45.0), pass rate 100.00%, determinism 3/3 PASS. No build errors. No correctness anomalies.

## Comparison to prior baselines

| Cell                | R53 (5/9 PASS) | R54 @ `9ab79eff` (2/9) | R56 @ `0e068a54` (1/9) | Δ R54→R56 |
|---------------------|---------------:|-----------------------:|-----------------------:|----------:|
| 8B_GateUp/rcr       |    PASS (≥95)  |   95.11% PASS          |   94.25% HEADROOM      |   -0.86pp |
| 8B_GateUp/rrr       |    HEADROOM    |   94.39% HEADROOM      |   93.41% HEADROOM      |   -0.98pp |
| 8B_GateUp/crr       |    HEADROOM    |   93.61% HEADROOM      |   93.49% HEADROOM      |   -0.12pp |
| 70B_GateUp/rcr      |    PASS (≥95)  |   95.94% PASS          |   95.63% **PASS**      |   -0.31pp |
| 70B_GateUp/rrr      |    HEADROOM    |   94.15% HEADROOM      |   94.56% HEADROOM      |   +0.41pp |
| 70B_GateUp/crr      |    HEADROOM    |   88.95% HEADROOM      |   89.06% HEADROOM      |   +0.11pp |
| 70B_QO/rcr          |    PASS (≥95)  |   94.89% HEADROOM      |   94.95% HEADROOM      |   +0.06pp |
| 70B_QO/rrr          |    PASS (≥95)  |   94.52% HEADROOM      |   94.13% HEADROOM      |   -0.39pp |
| 70B_QO/crr          |    PASS (≥95)  |   94.59% HEADROOM      |   94.75% HEADROOM      |   +0.16pp |

Aggregate movements R54 → R56:
- Range: -0.98pp (8B_GateUp/rrr) to +0.41pp (70B_GateUp/rrr).
- Mean: -0.21pp, abs-mean: 0.42pp.
- All 9 cells inside ±1pp band — consistent with day-over-day SCLK/thermal drift, not regression.

## Key questions answered

### Q1: Is the R54 Reviewer's 7/9 HEADROOM drift real regression introduced by R54 Dev B's PMC commit `9ab79eff`, OR thermal/SCLK noise reproducible at the same HEAD today?

**Answer: thermal/SCLK noise band drift, not real regression.**

Rationale:
- HEAD `0e068a54` (post-R54+R55 wrap) reproduces R54 Reviewer's qualitative split (1-2/9 PASS) within ±1pp on every cell.
- The dominant CRR-class bottleneck on `70B_GateUp/crr` reproduces at 89.06% (R54: 88.95%) — only +0.11pp away. Six R54 dev runs (R54A/E/F/I + R55E + R56B) have closed this floor as REFUTED.
- The 70B_QO row that R54 saw "uniformly tight" at 94.5-94.9% reproduces here at 94.13-94.95% — same band.
- The R53→R54 transition (5/9→2/9) and R54→R56 transition (2/9→1/9) are dominated by 8B_GateUp/rcr drifting from 95.1% (R54 PASS) to 94.25% (R56 HEADROOM) — a -0.86pp slip that flips one PASS into HEADROOM. None of the 8B_GateUp/rcr fp8 runs were under 2525 TFLOPS (median 2652.60); this is a noise-band slip, not a measurable regression.
- The 8B_GateUp shape exhibited 5-of-15 fp8 runs with anomalous low TFLOPS (one as low as 506.54 TFLOPS, ~80% drop) consistent with SCLK reset / cold-cache events. Median is robust to these. The mxfp8 8B_GateUp/crr arm also shows a 476.7 TFLOPS outlier in 5 runs. These outliers do not affect the median verdicts but confirm the noise environment is hot.

### Q2: Does HEAD `0e068a54` (post R54+R55 wrap) match either baseline within ±1pp on each cell?

**Answer: YES — every cell is within ±1pp of the R54 Reviewer baseline at `9ab79eff`.** Max delta is -0.98pp on 8B_GateUp/rrr; only one other cell exceeds 0.5pp (-0.86pp on 8B_GateUp/rcr). Vs R53 (which was a higher noise band overall), 4/9 cells crossed the gate; this R56 sweep keeps only 1/9 above the gate. The cells that flipped (R53 PASS → R56 HEADROOM) are all sitting within 1pp of the gate — recoverable with modest tuning or in a future cooler-day re-bench.

### Q3: Are all R55 added macros (default-OFF) genuinely byte-identical to R53 baseline ISA at production defaults?

**Answer: YES — confirmed byte-identical at the AMDGPU instruction level.**

ISA built from `kernel_mxfp8_layouts.cpp` at shape M=4096 N=28672 K=8192 (70B Gate/Up) at:
- `94eb7675` (R53 wrap) → `r56_reviewer_results/isa_94eb7675/device_70B_GateUp.s` (30558 lines)
- `0e068a54` (R54+R55 wrap) → `r56_reviewer_results/isa_HEAD/device_70B_GateUp.s` (30558 lines)

`diff` produces only an 18-line block changed at lines 29558-29569 — entirely the `__hip_cuid_*` build-id object name (a content-hash fingerprint that always changes between independent compile invocations). All six `gemm_kernel<Layout{0,1,2}, bool{0,1}>` template instantiations occupy identical line ranges in both files: Layout0/Lb0 at lines 4-1517 (.size at 1510), Layout0/Lb1 at 1551-3064, Layout1/Lb0 at ~3550-9827, Layout1/Lb1 at ~9854-11468, Layout2/Lb0 at ~11500-13186, Layout2/Lb1 at ~13212-14904. All `.set num_vgpr / num_sgpr / private_seg_size / uses_vcc` resource directives identical. R55 macros (R55A INTERLEAVE, R55B SALU_HOIST, R55C VALU_DEP_BREAK) and the R56A `MXFP8_RRR_SALU_SETPRIO_R56A` macro all default-OFF and confirmed dead-code at production defaults.

Note: at the time of writing, HEAD has advanced to `361d71ad` (R56 Dev A landed mid-sweep), but `git diff 0e068a54..361d71ad -- analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp` produces zero changes; the kernel source under measurement is the same byte sequence at both commits, so the sweep is faithful to the `0e068a54` spec.

## Summary

The R56 reviewer baseline at HEAD `0e068a54` (post R54+R55 wrap) reproduces R54 Reviewer's 9-cell verdict to within ±1pp on every cell (mean drift -0.21pp, abs-mean 0.42pp). The R54→R56 PASS-count drop (2/9 → 1/9) is driven entirely by `8B_GateUp/rcr` slipping -0.86pp from 95.11% PASS to 94.25% HEADROOM under day-over-day SCLK/thermal noise. The dominant `70B_GateUp/crr` bottleneck at 89.06% matches R54's 88.95% within +0.11pp, confirming the CRR scale-shift floor (already 6× closed by R54A/E/F/I + R55E + R56B) is structurally distinct from the upper headroom band. **R55 macros confirmed byte-identical to R53 baseline ISA at production defaults**, so the R55 cycle introduced zero performance risk to the production tree. The R54→R56 noise drift is **not regression** — it is the SCLK noise band that R54 Reviewer already characterized. Recommend the upper-band HEADROOM cells (94.0-95.0% band) be revisited under colder thermal conditions before any further dev cycles target them.

## Reproduction

```bash
cd analysis/fp8_gemm/mi350x
HIP_VISIBLE_DEVICES=7 bash r56_reviewer_workspace/r56_reviewer_baseline_9cell.sh
python3 r56_reviewer_workspace/r56_aggregate.py
```

## Artifacts

- `r56_reviewer_results/baseline_9cell/` — 105 logs (9 cells × 10 runs + 9 check + 6 build)
- `r56_reviewer_results/baseline_9cell.run.log` — sweep driver echo
- `r56_reviewer_results/r56_aggregate.json` — per-cell run vectors + medians
- `r56_reviewer_results/isa_94eb7675/device_70B_GateUp.s` — ISA at R53 wrap
- `r56_reviewer_results/isa_HEAD/device_70B_GateUp.s` — ISA at R54+R55 wrap
- `r56_reviewer_workspace/r56_reviewer_baseline_9cell.sh` — driver
- `r56_reviewer_workspace/r56_aggregate.py` — aggregator
