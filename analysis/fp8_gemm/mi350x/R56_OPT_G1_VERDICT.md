# R56 Opt G-1 Verdict — 64x1024 -> 256x256 swap on cluster A (L1, L2, L6)

**Date:** 2026-04-19
**Worker:** R56 G-1 (very-high-confidence 256x256 swap)
**GPUs:** 0,1 (verified idle pre-launch)
**Mechanism:** R50D shim AS-IS (8th consecutive AS-IS reuse round). Force tile=256x256 (eff=128) over current 64x1024 (eff=60.2) via aiter `.co` dlopen.
**Bench params:** warmup=200, iters=500, trim=0.10, 10 INDEPENDENT seeds [101..1010].

---

## Summary table

| Cell | Shape | Current (R55) | Alt (256x256) | Delta pp | Verdict |
|---|---|---:|---:|---:|---|
| L1 | 4096x32768x14336  | 65.68% | **100.63%** | **+34.95** | **PROMOTE** |
| L2 | 32768x4096x14336  | 84.13% | **105.37%** | **+21.24** | **PROMOTE** |
| L6 | 16384x4096x14336  | 90.28% | **110.20%** | **+19.92** | **PROMOTE** |

**Aggregate G-1 perf delta: +76.11pp across 3 PROMOTEs.**

All three cells: bit-deterministic (wcf_max=0, wcf_std=0, fin_min=1.0 across 10 INDEPENDENT seeds), n_OK=10/10, snr_med~55.6 dB. Strict 10-run @ 80% gate PASSES on all candidates.

---

## Per-cell detail

### L1 (4096x32768x14336)
- Current: R53D3B_1_AITER tile=64x1024 -> 3478.5 TFLOPS @ 65.68% (worst LOSE in R55)
- Alt: 256x256 .co dlopen via R50D shim, grid (128, 16, 1), bdx=256
- Smoke: 5419.7 TFLOPS @ 102.33%
- 10-run: tflops_first=5329.7 @ 100.63%, n_OK=10/10, wcf_max=0, snr_med=55.598 dB
- Delta: +34.95pp (LOSE 65.68 -> WIN 100.63)
- Verdict: **PROMOTE**

### L2 (32768x4096x14336)
- Current: R53D3B_2_AITER tile=64x1024 -> 4394.7 TFLOPS @ 84.13%
- Alt: 256x256 .co dlopen, grid (16, 128, 1), bdx=256
- Smoke: 5386.3 TFLOPS @ 103.12%
- 10-run: tflops_first=5504.0 @ 105.37%, n_OK=10/10, wcf_max=0, snr_med=55.595 dB
- Delta: +21.24pp (LOSE 84.13 -> WIN 105.37)
- Verdict: **PROMOTE**

### L6 (16384x4096x14336)
- Current: R53D3C_3_AITER tile=64x1024 -> 4642.4 TFLOPS @ 90.28%
- Alt: 256x256 .co dlopen, grid (16, 64, 1), bdx=256
- Smoke: 5658.0 TFLOPS @ 110.03%
- 10-run: tflops_first=5666.4 @ 110.20%, n_OK=10/10, wcf_max=0, snr_med=55.598 dB
- Delta: +19.92pp (LOSE 90.28 -> WIN 110.20)
- Verdict: **PROMOTE**

---

## Mechanism confirmation

The aiter heuristic under-picked 256x256 on all three K=14336 shapes; manually forcing the 256x256 .co binary (eff=128 vs 60.2) recovers +20-35pp per cell with bit-determinism on every seed. Hypothesis from R56 Decider section 10 fully confirmed for cluster A. Sister relationship to R55 D-5A on (16384x4096x{4096,6144,7168}) holds — 256x256 on K=14336 also dominates.

## Files

- `bench_R56G1_L1.py`, `bench_R56G1_L2.py`, `bench_R56G1_L6.py`
- `R56_OPT_G1_L{1,2,6}_SMOKE.{json,log}`
- `R56_OPT_G1_L{1,2,6}_10RUN.{json,log}`
- `R56G1_L{1,2,6}_INTEGRATION_FRAGMENT.json`
