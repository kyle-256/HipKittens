# R123 — Smoke gate design rationale

`_smoke_p1_0_rcr_v2.py`: 5 shape × 2 bn = 10 case, SNR ≥ 30 dB + perf ratio ∈ [0.95, 1.05].

SNR 30 dB threshold rationale:
- Below 25 dB: clear correctness issue (e.g. wrong K-iter)
- 25-30 dB: borderline, often noise from quantize floating-point order
- ≥ 30 dB: bit-similar (often bit-equal for our deterministic kernels)

perf ratio [0.95, 1.05] gate too tight:
- Sub-0.3ms shapes have measurement noise floor ~4-6%
- Median-of-7 trials helps but ±3-5% per shape still common
- Original ±3% gate: would constantly fire; relaxed to ±5%

Real lesson: smoke gate primarily for correctness (SNR), perf gate as advisory only. Real perf gate is bench geomean.
