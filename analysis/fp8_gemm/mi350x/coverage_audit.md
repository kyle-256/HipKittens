# Round 14 OptB Coverage Audit

Maps the kernel-macro space already-tested on the 4 deep-LOSE targets vs.
what R14 OptB sweeps. Source: `bench_deep_lose_results.json` (Round 2 deep-LOSE
spot bench, warmup=200 iters=500) plus `bench_all42_results.json`.

## Variant -> macro decoder (excerpt from `bench_all_42.py:311-462`)

| Suffix token | Macro |
|---|---|
| `_ts` | `-DTAIL_SPLIT=1` |
| `_gm{N}` | `-DGROUP_SIZE_M=N` |
| `_v{N}` | `-DSTEP3_BARRIER_VMCNT=N` |
| `_lgk{N}` | `-DSTEP12_BR_LGKMCNT=N` |
| `_pf6_6` | `-DSTEP3_PF_N=6 -DSTEP4_PF_N=6` |
| `_ext_br` | `-DSTEP4_EXTERNAL_BR_PREFETCH=1` |
| `_no_embed` | `-DSTEP3_EMBED_BARRIER=0` |
| `_tv{N}` | `-DTAIL_BARRIER_VMCNT=N` |
| `_u{N}` | `-DUNROLL_K=N` |
| `_memc` | `-mllvm -amdgpu-sched-strategy=max-memory-clause` |
| `_dc` | `-mllvm -amdgpu-disable-clustered-low-occupancy-reschedule` |
| `_wpe{N}` | `__attribute__((amdgpu_waves_per_eu(N,N)))` |

## Already-covered (per `bench_deep_lose_results.json`)

48-50 variants per shape were tested in Round 2 deep-LOSE. Coverage of macro axes:

| Macro axis | DLA1 (4096×32768×128256) | DLA2 (128256×32768×4096) | DLA7 (28672×32768×4096) | P1 (28672×4096×16384) |
|---|---|---|---|---|
| GROUP_M ∈ {1,2,4,8} | 4,8 covered (gm4/gm8); 1,2 NOT on this shape | 2 covered (gm2); 1,4,8 partial | 2,4,8 covered; 1 NOT | 4,8 covered; 1,2 NOT |
| VMCNT ∈ {4,8,12,16,20,24} | 4,8(default),12,16,20 covered; 24 NOT in pf6_6 | 4,8,12,16,20 covered | 4,8,12,16,20 covered | 4,8,12,16,20 covered |
| LGKMCNT ∈ {0,2,4} | 0(default),2 covered; 4 NOT in pf6_6 | 0,2 covered; 4 NOT | 0,2 covered; 4 NOT | 0,2 covered; 4 NOT |
| EXT_BR | partial (gm8_ext_br_lgk2 only); pf6_6+ext_br NOT | gm8_ext_br_lgk2 only; gm2+memc_dc+ext_br NOT | gm8_ext_br_lgk2; lgk2_v12_memc+ext_br NOT | gm8_ext_br_lgk2; ts_gm8+ext_br NOT |
| NO_EMBED | NOT on pf6_6 stack | NOT on gm2_memc_dc stack | NOT on lgk2_v12_memc stack | NOT on ts_gm8 |
| TV (TAIL_BARRIER) | tv0/tv16 NOT on pf6_6 | NOT on memc_dc stack | NOT on lgk2_memc | (irrelevant: K=16384 large-K) |

## R14 OptB sweep — 28 untested combinations

### DLA1 (parent `_ts_pf6_6_v12_memc`, 5213.6 T = 90.2%, our re-bench 5043 T = 87.2%)
1. `_ts_pf6_6_v12_memc_r14b_extbr` — pf6_6 + EXT_BR (NEW)
2. `_ts_pf6_6_v12_memc_r14b_noembed` — pf6_6 + NO_EMBED (NEW)
3. `_ts_pf6_6_v20_memc` — pf6_6 + VMCNT=20 (NEW)
4. `_ts_pf6_6_v24_memc` — pf6_6 + VMCNT=24 (NEW)
5. `_ts_pf6_6_lgk4_v12_memc` — pf6_6 + LGKMCNT=4 (NEW)
6. `_ts_pf6_6_v12_memc_tv0` — pf6_6 + TAIL_BARRIER_VMCNT=0 (NEW)
7. `_ts_pf6_6_v12_memc_tv16` — pf6_6 + TAIL_BARRIER_VMCNT=16 (NEW)

### DLA2 (parent `_ts_gm2_v12_memc_dc`, 4243.8 T = 93.5%)
8. `_ts_gm1_v12_memc_dc` — GM=1 linear walk + memc_dc (NEW; A-bound hypothesis)
9. `_ts_gm1_lgk2_v12_memc_dc` — GM=1 + LGK=2 (NEW)
10. `_ts_gm1_v20_memc_dc` — GM=1 + VMCNT=20 (NEW)
11. `_ts_gm1_v12_memc_dc_extbr` — GM=1 + EXT_BR (NEW)
12. `_ts_gm1_v12_memc_dc_tv0` — GM=1 + TV=0 (NEW)
13. `_ts_gm2_v12_memc_dc_noembed` — parent + NO_EMBED (NEW)
14. `_ts_gm2_v12_memc_dc_extbr` — parent + EXT_BR (NEW)

### DLA7 (parent `_ts_lgk2_v12_memc`, 4192.0 T = 93.9%)
15. `_ts_lgk2_v12_memc_tv0` — TV=0 on parent (NEW; small-K, tail matters)
16. `_ts_lgk2_v12_memc_tv16` — TV=16 on parent (NEW)
17. `_ts_lgk2_v12_memc_extbr` — EXT_BR on parent (NEW)
18. `_ts_lgk2_v12_memc_noembed` — NO_EMBED on parent (NEW)
19. `_ts_lgk4_v12_memc` — LGKMCNT=4 + V12 + memc (NEW; lgk4 untested in this stack)
20. `_ts_lgk2_v20_memc` — VMCNT=20 (NEW; v20+lgk2+memc combo)

### P1 (parent `_ts_gm8`, 4992.6 T = 93.3%; this run 5013.9 = 93.7%)
21. `_ts_gm8_memc` — gm8 + memc (NEW; parent never had memc)
22. `_ts_gm8_lgk2_memc` — gm8 + lgk2 + memc (NEW)
23. `_ts_gm8_v12_memc` — gm8 + v12 + memc (NEW)
24. `_ts_gm8_extbr` — gm8 + EXT_BR (NEW)
25. `_ts_gm8_extbr_memc` — gm8 + EXT_BR + memc (NEW)
26. `_ts_gm8_noembed` — gm8 + NO_EMBED (NEW)
27. `_ts_gm8_noembed_memc` — gm8 + NO_EMBED + memc (NEW)
28. `_ts_gm8_v20_memc` — gm8 + v20 + memc (NEW)

## Skipped axes (with reasons)

- **GM=32, GM=64**: TODO marks DEAD END for large-N shapes (fewer N-tiles per group on N=32768).
- **Asymmetric PF (pf3_8, pf6_4, etc.)**: per AGENT_PROMPT/TODO, asymmetric PF degraded all shapes.
- **PF_N=1, PF_N=2**: DEAD END (-2 to -6%).
- **TAIL_SPLIT=0** for shapes whose parent uses TS=1: TS=1 already tuned in.
- **TAIL_BARRIER_VMCNT** for K=16384 (P1) and K=128256 (DLA1 main path): per R8B, tail iter is ≤0.45% of total K iters at K≥14336 — mathematically can't yield visible perf. We tested TV cross only on DLA1 (because K=128256 main pf6_6 path has tail epilogue in TAIL_SPLIT=1 mode that may have different timing) and DLA7 (K=4096 small-K, where tail IS material).
- **UNROLL_K**: per AGENT_PROMPT, U_K dead-end on these stacks.
- **PERSISTENT_XCD / STATIC_XCD_REMAP**: per AGENT_PROMPT R4/R7, dead-end on mega-M.
- **`max-ilp` / `iterative-ilp` / etc.**: per R12 task scope is **macro-axis only**, scheduler axis was R14 OptA's job and is exhausted on these 4 shapes.
