# R26-G: STEP12_BR_LGKMCNT axis is DEAD on the R25 stack

**Date:** 2026-04-18
**Hypothesis (R26-E next-axis suggestion):** `STEP12_BR_LGKMCNT` (the
`s_waitcnt lgkmcnt(N)` between the Step-12 LDS read for the next K-pair and
the consuming MFMA) is fixed at 2 in every R25-F/G/H winner that uses the
`_ts_lgk2_*` family. Values {1, 3} were never explored on top of the R25
stack — perhaps a tighter (lgk=1) or looser (lgk=3) wait would unlock 1-2pp
on shapes where LDS-read latency vs MFMA-issue is mistuned.

**Verdict:** DEAD on all 4 representative shapes. Max delta is +0.13pp; all
within the 1-rep noise floor (best-case std 5.8 TFLOPS, worst 63.8 TFLOPS).
The kernel is insensitive to STEP12_BR_LGKMCNT in the {0,1,2,3} range when
combined with the R25 prefetch-off mechanism.

## Setup
- GPU: 4 (idle, isolated via `HIP_VISIBLE_DEVICES=4`)
- warmup=200, iters=500, trim=10%, 5 reps
- 13 builds total (3 lgk values × 3 shapes + 4 lgk values for SC)

## Per-shape results (5-rep mean ± std, % vs aiter ASM `comp`)

### DLA7 — 28672×32768×4096 — comp=4466.6 (parent: `_ts_lgk2_gm7_v12_memc_pfoff14`)
| lgk | mean (TFLOPS) | std | %comp |
|-----|--------------:|----:|------:|
| 1   | 4971.1 | 11.2 | 111.29% |
| **2 (default)** | **4967.3** | **11.5** | **111.21%** |
| 3   | 4959.2 |  6.9 | 111.03% |

Δ best vs default: **+0.08pp** → DEAD.

### SD — 16384×4096×28672 — comp=5525.3 (parent: `_ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all`)
| lgk | mean (TFLOPS) | std | %comp |
|-----|--------------:|----:|------:|
| 1   | 6248.4 | 12.0 | 113.09% |
| **2 (default)** | **6241.1** | **5.8** | **112.95%** |
| 3   | 6243.0 | 15.7 | 112.99% |

Δ best vs default: **+0.13pp** → DEAD.

### SC — 4096×32768×6144 — comp=4548.6 (parent: `_ts_v12_gm7_memc_pfoff19_kx6144_btw_all`)
SC parent has NO `STEP12_BR_LGKMCNT` macro → kernel default 0 is the baseline.
| lgk | mean (TFLOPS) | std | %comp |
|-----|--------------:|----:|------:|
| **0 (default)** | **5358.3** | **6.9** | **117.80%** |
| 1   | 5327.8 | 61.8 | 117.13% |
| 2   | 5333.3 | 63.8 | 117.25% |
| 3   | 5362.0 |  9.7 | 117.88% |

Δ best (lgk=3) vs default: **+0.08pp** → DEAD. Note lgk=1/2 had a single ~5200
TFLOPS outlier per series, suggesting these injected lgkmcnt waits sometimes
collide with the SC prefetch schedule (slight regression risk).

### SE — 28672×4096×16384 — comp=5409.1 (parent: `_ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all`)
| lgk | mean (TFLOPS) | std | %comp |
|-----|--------------:|----:|------:|
| 1   | 6075.8 | 11.6 | 112.33% |
| **2 (default)** | **6070.0** | **5.0** | **112.22%** |
| 3   | 6068.8 |  1.3 | 112.20% |

Δ best vs default: **+0.11pp** → DEAD.

## Why this axis is dead
- R25-F/G/H established that the K-loop is **bandwidth-limited** and the
  Step-12 LDS read for the next K-pair almost always hits a
  warmed-up LDS bank by the time the consumer MFMA wants it. The current
  `lgk=2` (or kernel default 0 for SC) is already small enough that the wait
  is essentially free — tightening to 1 saves at most a single fma slot,
  loosening to 3 doesn't add slack the scheduler can productively use.
- The remaining gap to comp on these 4 shapes is in the K-loop tail and
  epilogue, not in the steady-state Step-12 timing.
- For SC specifically, even the kernel-default (no waitcnt override) was the
  best mean — the lgk-injected variants showed sporadic outliers, indicating
  the explicit wait is at best neutral and can occasionally interact poorly
  with the per-K prefetch schedule.

## Implications
- Don't add `STEP12_BR_LGKMCNT` ∈ {1, 3} variants to `bench_all_42.py`.
- The R26-E axis suggestion list should mark `STEP12_BR_LGKMCNT` **EXHAUSTED**.
- Future axes worth trying (per R26-E findings still open): different
  `STEP3_BARRIER_VMCNT` values stacked on K-EXACT winners; or epilogue
  tuning (CTA-tile of C-store) on the same 4 shapes — neither was swept
  in R25 or R26-A..F.

## Artifacts
- Script: `r26g_lgkmcnt.py`
- Bench log: `r26g_bench.log`
- Results JSON: `r26g_lgkmcnt_results.json`
- Build log: `r26g_build.log`
- 13 `.so` files in `build_all42/` named `*_r26g_lgk{0,1,2,3}.cpython-310-*.so`
  (kept on disk for future reference; not wired into `bench_all_42.py` since
  no WIN was found).
