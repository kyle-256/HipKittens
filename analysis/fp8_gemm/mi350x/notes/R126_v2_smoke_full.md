# R126 — v2 smoke full 24-shape coverage gap

Current smoke only covers 5 shapes × 2 bn = 10 cases.
24-shape bench checks perf but uses random inputs without SNR validation.

Gap: no production-shape correctness scan.
Multi-session priority: add `_smoke_24_v2.py` mirroring 24-shape bench but with SNR validation per case. ~50 LOC. P4 prereq.
