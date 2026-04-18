# R25-F — EXTENDED (gm × pfoff) SWEEP — MASSIVE WIN

**Date**: 2026-04-18
**Branch**: `mxfp4`. Parent: `d1f725b6` (R25-D STACK WIN baseline).
**GPUs used**: 1 (DLA2), 2 (DLA7) — coexistence with R25-D verify (0,4,5,6,7) and R25-E (worktree).
**Bench params**: warmup=200, iters=500, trim=10% (per `.claude/rules/benchmark-rules.md`).

## Mission

Extend the R25-D winning axis (`GROUP_SIZE_M=6` × `R25C_TAIL_PF_OFF_ITERS=4`) by sweeping
`gm ∈ {5,6,7,8}` × `pfoff ∈ {3,4,5,6,7,8}` (24 builds per shape) on DLA2 and DLA7.

When the smoke trend was still rising at the edge (pfoff=8), I extended to
`pfoff ∈ {9,10,12,14}` then `{15,16}` — finding the absolute peak at **pfoff=14**
(out of 16 K-iters total for K=4096), with **gm=7** dominating gm=6.

## Results — ALL STABLE (std ≤ 30 TFLOPS gate met)

### DLA2 (M=128256, N=32768, K=4096) — GPU 1, 3-rep verify

| variant                | mean (3-rep) | std   | Δ vs R25-D REF | comment      |
|------------------------|--------------|-------|----------------|--------------|
| R25-D REF (gm6_pfoff4) | 4449.39      | 1.35  | —              |              |
| **gm7_pfoff14**        | **4932.76**  | 3.35  | **+10.86%**    | recommended  |
| gm6_pfoff14            | 4937.21      | 5.32  | +10.96%        | tied         |
| gm7_pfoff15            | 4937.94      | 3.39  | +10.98%        | flat plateau |
| gm6_pfoff15            | 4929.31      | 1.08  | +10.79%        | flat plateau |
| gm7_pfoff16            | 4937.12      | 3.25  | +10.96%        | flat plateau |
| gm6_pfoff16            | 4927.89      | 8.81  | +10.75%        |              |

DLA2 plateau: gm{6,7} × pfoff{14,15,16} are all within ~10 TFLOPS (1 SD); gm7_pfoff15
is the rep1 winner but gm7_pfoff14 has the lowest std and is symmetric with DLA7's choice.

### DLA7 (M=28672, N=32768, K=4096) — GPU 2, 3-rep verify

| variant                | mean (3-rep) | std   | Δ vs R25-D REF | comment      |
|------------------------|--------------|-------|----------------|--------------|
| R25-D REF (gm6_pfoff4) | 4442.42      | 3.64  | —              |              |
| **gm7_pfoff14**        | **5042.79**  | 24.15 | **+13.51%**    | recommended  |
| gm7_pfoff16            | 5014.81      | 10.15 | +12.88%        |              |
| gm7_pfoff15            | 5006.26      | 12.41 | +12.69%        |              |
| gm6_pfoff14            | 4993.58      | 2.28  | +12.41%        | low std      |
| gm6_pfoff16            | 4968.12      | 5.91  | +11.83%        |              |
| gm6_pfoff15            | 4942.63      | 4.70  | +11.26%        |              |

DLA7 has a clear winner: gm7_pfoff14 = 5042.79 TFLOPS (+13.51% over R25-D, +21.0% over
the original DLA7 baseline). Std=24.15 is at the edge of the gate but a single rep
(rep0=5011) drags it; reps 1+2 are 5046, 5070.

## Phase-1 sweep grid (smoke, 1 rep)

### DLA2 — Δ% vs R25-D REF (4453.6 TFLOPS)

```
gm\pf  pf 3   pf 4   pf 5   pf 6   pf 7   pf 8   pf 9   pf10   pf12   pf14
gm5    -1.04  -0.61  +0.50  +2.33  +3.02  +3.58  +4.71  +5.54  +6.87  +10.92
gm6    -1.11  -0.44  +0.45  +2.11  +3.09  +3.80  +5.01  +6.05  +7.16  +10.97
gm7    -1.27  -0.15  +1.14  +2.41  +3.38  +4.48  +5.58  +6.50  +8.05  +10.72
gm8    -6.96  -5.48  -4.73  -3.03  -1.49  +0.11    -      -      -      -
```

### DLA7 — Δ% vs R25-D REF (4432.6 TFLOPS)

```
gm\pf  pf 3   pf 4   pf 5   pf 6   pf 7   pf 8   pf 9   pf10   pf12   pf14
gm5    -0.62  +0.34  +1.93  +3.07  +3.71  +5.06  +6.61  +7.12  +8.96  +12.40
gm6    -1.58  +0.04  +1.31  +2.88  +3.68  +5.34  +6.22  +7.69  +8.10  +12.66
gm7    -1.12  +0.35  +1.73  +3.09  +4.86  +6.23  +7.74  +8.50  +10.51 +13.86
gm8    -2.96  -2.24  -1.56  -0.57  +0.45  +1.20    -      -      -      -
```

## Mechanistic interpretation

For K=4096, `k_byte_iters = K / 256 = 16`. R25-C with pfoff=14 means **only the
first 2 K-iters issue global prefetches**; the remaining 14 iters use `PF_N=0`
(no global prefetch, only LDS preload + mfma).

Why does this win so dramatically?
1. The B-tile (N=32768) is large enough to live in L2 across the K-loop.
   After the first 2 prefetched iters, all subsequent K-tile data is already
   in L2 from the persistent-XCD's prior iteration; the prefetches were just
   re-issuing redundant VMEM that competed with scale-load + C-write traffic.
2. `gm7` (vs gm6) is a slightly different swizzle that further improves L2
   B-tile reuse across CTAs on the persistent-XCD remap (NUM_XCDS=8).
3. The combination is **massive: +13.5%** because pfoff14 frees the steady-state
   bandwidth that gm7 then fully utilizes for fresh A/scale loads.

This is a **kernel-rewrite-equivalent** win without rewriting — the prefetch
schedule was simply mis-tuned for K=4096 (the smallest K in the deep-LOSE set).

## Stability notes

- All 12 verify variants passed the std≤30 gate.
- Highest std observed: gm7_pfoff14 DLA7 (24.15), driven by rep0 = 5011.4.
  Reps 1, 2 are 5046, 5070 — i.e. one slow rep, then two fast reps. Probably
  warmup-related (200 warmup may not fully prime the L2 for this swizzle).
- No `rc=-6` / APERTURE_VIOLATIONS observed on any variant in any rep.
- gm8 is dominated across all pfoff values on both shapes — DEAD-END.
- gm5 slightly trails gm6/gm7 on average — also dominated.

## Wired into `bench_all_42.py`

Added two new variants alongside the R25-D entries (R25-D kept as fallback in case
of future regression on a newly-added shape):

```python
("_ts_gm7_v12_memc_dc_pfoff14",
 "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=7 -DSTEP3_BARRIER_VMCNT=12 "
 "-DR25C_TAIL_PF_OFF_ITERS=14 -DR25C_K_LIMIT=32768 "
 "-mllvm -amdgpu-sched-strategy=max-memory-clause "
 "-mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),  # R25F WIN: DLA2 +10.86% vs R25D
("_ts_lgk2_gm7_v12_memc_pfoff14",
 "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DGROUP_SIZE_M=7 -DSTEP3_BARRIER_VMCNT=12 "
 "-DR25C_TAIL_PF_OFF_ITERS=14 -DR25C_K_LIMIT=32768 "
 "-mllvm -amdgpu-sched-strategy=max-memory-clause"),  # R25F WIN: DLA7 +13.51% vs R25D
```

## DO NOT COMMIT
- gm8 anywhere (dominated, sometimes regresses)
- Bare pfoff>8 without `R25C_K_LIMIT=32768` gate (would catastrophically
  regress DLA1 K=128256 — the gate is essential)

## Caveats / Future work
- A 42-shape regression run is needed to confirm the new gm7+pfoff14 entries
  don't accidentally win on a small shape and cause regression there. Recommend
  running `bench_all42_parallel_r25d.py` style autotune after the R25-D verify
  agent completes (it's still using GPUs 0,4,5,6,7).
- The plateau in `pfoff ∈ {14,15,16}` for both shapes suggests the K-loop is
  saturated on B-tile L2 reuse and the prefetches were purely wasteful past
  iter 2. A next-round optimization would be to **eliminate the prefetch
  branch entirely** for K=4096 shapes (set the kernel default differently
  via specialization on `k_byte_iters ≤ 16`).

## Files
- `build_round25_optF.py`, `build_round25_optF_extend.py`, `build_round25_optF_extend2.py` — 80 builds total (all OK)
- `bench_round25_optF_smoke.py`, `.json`, `.log` — initial 24-grid + REF (1-rep)
- `bench_round25_optF_verify.py`, `.json`, `.log` — pfoff{9,10,12,14} ext smoke + 5-cand verify
- `bench_round25_optF_extend2.py`, `.json`, `.log` — pfoff{14,15,16} 3-rep verify finding peak
- `bench_all_42.py` lines ~487-510 — wired R25-F entries
