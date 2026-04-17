# Round 17 Optimizer A — Final Verdict

**Date**: 2026-04-17
**Author**: Kyle.Zhao@amd.com / kyle-256
**Time spent**: ~75 min wall (exceeded 50-min time-box; re-baseline ran longer than estimated)

---

## Deliverables

| File | Contents |
|---|---|
| `build_round17_optA_profile.{py,log}` | Profile harness (warmup=200, iters=100), achieved 5105 TFLOPS on DLA1 |
| `dla1_prof_short.py` | Short-iter driver for rocprof PMC pass (warmup=2, iters=1) |
| `rocprof_dla1_r17a/v3_stats/` | Kernel-trace + stats (rocprofv3 sqlite db) |
| `rocprof_dla1_r17a/v3_pmc/`   | 3-pass PMC counter capture |
| `rocprof_dla1_r17a/v3_pcs/`   | PC-sampling pass (0 samples — host_trap interval too coarse) |
| `rocprof_dla1_r17a/pmc_summary.json` | Aggregated PMC dump |
| `extract_pmc.py` | sqlite → JSON aggregator |
| `round17_optA_dla1_profile_findings.md` | **DLA1 bottleneck analysis + 3 source-edit proposals** |
| `bench_all42_results_r17_rebaseline.json` | **Partial 26/42 re-baseline** (parsed from log) |
| `compare_rebaseline.py` | Drift comparator |
| `round17_optA_rebaseline_summary.md` | **Drift analysis vs Round 2** |
| `round17_optA_rebaseline_compare.txt` | Per-shape comparison table |
| `parse_partial_log.py` | Partial-results extractor |

## Top-line findings

### A. DLA1 profile (the worst deep-LOSE shape)
- Achieved 5105 TFLOPS @ 6.74 ms = 88.3 % of competitor (matches existing baseline exactly)
- **VALUBusy = 49 %** — kernel sits idle half the cycles
- **MemUnitStalled ≈ 42 %** (per-SE estimate)
- **MFMA pipeline floor at K=128256 ≈ 1.8 ms** (8 cyc/MFMA-f4) or 0.9 ms (4 cyc); **wall = 6.7 ms**
- **HBM BW utilization = 264 / 5300 GB/s = 5 %** — clearly NOT memory-bandwidth-bound
- **LDS-stall = 0.04 %** — LDS is not the bottleneck

### B. Top-3 DLA1 bottleneck causes
1. **MFMA accumulator dep-stall** (single C-tile re-used; 8-cyc latency dominates) — biggest
2. **K-loop epilogue overhead** (2004 iters × s_barrier + waitcnt = ~0.6-1.2 ms wasted)
3. **Prefetch m0-hazard s_nop** (compounded 31× vs K=4096 case)

### C. Three concrete source-edit proposals (for R17C)
1. **Double C-accumulator tiling (P1)** — split `rt_C[4][4]` into `rt_C0/C1`, ping-pong MFMAs.
   Expected +2-4 pp on DLA1. Risk: register pressure forces 2-WG/CU instead of 4 (occupancy ½×) —
   but K=128256 is exactly the regime where latency-hide gain dominates.
   File: `kernel_mxfp4_gluon_cpp.cpp`.
2. **Larger M-tile for K=128256 specialization (P2)** — M=128 → M=256, halves CTA count from
   524 K to 262 K, halves all per-wave sync overhead. Expected +1-2 pp.
3. **Drop redundant inner-loop s_barrier (P3)** — replace with cheaper `s_waitcnt lgkmcnt(0)`.
   Expected up to −0.4 ms = +5-6 pp **IF** safe (must SNR-validate).

### D. Re-baseline drift (26 / 42 shapes completed)
- Baseline is broadly **STABLE** — 18 / 26 within ±2pp of Round 2
- 5 shapes improved by ≥2pp, 3 regressed by ≥2pp (net +0.4 pp on shifted shapes)
- Variant-tag instability persists (8/26 shapes pick a different best variant) — confirms R12B
  cross-GPU bias finding
- **No formerly-LOSE shape flipped to WIN** at deep-LOSE level
- 16 / 42 shapes (incl. all 4 deep-LOSE) did not finish within time-out — re-run needed

## Commits

**No commits made.** This was a profiling + analysis pass; no source files were modified.
- `kernel_mxfp4_gluon_cpp.cpp` untouched
- All artifacts are new files in `analysis/fp8_gemm/mi350x/`

## Recommendation to R17 decider

1. **Hand off to R17C with Proposal P1 (double C-accumulator)** — highest-EV source-level edit.
   Build as new variant `_dblc_pf6_6_v12_memc`; SNR-validate; bench DLA1 against current 5105 TFLOPS.
2. If P1 yields ≥ 5300 TFLOPS, transfer to DLA2/DLA7/P1 to test cross-shape applicability.
3. **Schedule a clean overnight 8-GPU re-baseline** (no parallel profile work) to get the canonical
   42-shape JSON. Current Round 2 baseline remains canonical until then.
4. **Do NOT pursue more LLVM-flag tuning on DLA1** — profile confirms VALU is half-idle, which is a
   *register-allocation/MFMA-scheduling* problem, not a backend-flag problem.
