# R129 — Skewed expert distribution test

Per Rule 11 GroupGemm: must include skewed token distribution shapes (top_k=1 cf=1.25, single-expert ≥50%).
Current bench shapes all assume uniform distribution: M_per_g = M_tot / B.

Skewed shape example: B=4, M_g lengths = [4096, 4096, 4096, 16384] (one group 4x larger).
HK group_offs scan + per-group view handles non-uniform M_g, but perf may differ.

Multi-session todo: add `_bench_skewed_v2.py` with realistic MoE distributions. Validates that R43 + future commits don't regress on imbalanced shapes.

Today's bench is uniform-only — Rule 11 partial gap.
