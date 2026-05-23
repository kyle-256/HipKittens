# R122 — Bench methodology stability

Per R71 5-trial: v2/Triton 1.024±0.003 stable when GPU not shared.
Per R67/R62 trial 3: 1.086/1.161 outliers when GPU shared (Triton+dense also run faster than baseline → relative ratio inflated).
Per R95 24-shape: thermal noise across consecutive runs causes ±5% per-shape ratio shifts.

Standard methodology:
1. Bench 5-trial, take median for geomean stability
2. Per-shape ratios accept ±5% noise floor
3. Outlier trials (geomean > 1.10 from baseline) suspect GPU state, re-run
4. Cold GPU (no other workload) for production bench

For multi-session work, always run bench at session entry + exit:
- Entry establishes per-session baseline
- Exit compares against entry baseline (not absolute)

R71 best-of-5 = methodology to lock as default in scripts/_bench_*.py.
