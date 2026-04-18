R31 Dev D — V2-RCR persistent-CU dispatch (Approach 1) audit
=============================================================

Goal
----
R29 Dev C and R30 Dev D both flagged dispatch-geometry as the remaining
open lever for the 4096³ V2-RCR cell (R27 Reviewer ratio 0.9229, the
largest open RCR gap). The structural diagnosis was: at BLK=256 the
4096³ grid is 16×16 = 256 blocks across 304 CUs, leaving ~16% of CUs
idle (wave-fill = 0.84). R30 Dev D ruled out per-kernel optimization
levers and pointed at two follow-ups for R31:

1. **Persistent-CU dispatch** (one block per CU, internal K-loop over
   multiple block coords) — Approach 1 in the R31D prompt.
2. **BLK=128 path resurrection** — Approach 2 (multi-day work).

R31 Dev D goal: empirically validate or refute Approach 1 in 90 min.

TL;DR — outcome: **NO SHIP (perf NULL) → AUDIT-ONLY closure of
Approach 1**. Approach 1 as scoped in the prompt cannot close the
4096³ V2-RCR gap; the hardware constraints are compounding and
mathematically prevent any wave-fill improvement at BLK=256 for
shapes with `total_tiles ≤ num_CUs * occupancy`.

Hardware / kernel facts (re-confirmed empirically in r31d_build_*.log)
---------------------------------------------------------------------
- gfx950 / MI355X / 304 CUs.
- LDS per CU = 163840 B (R30 Dev B confirmed via hipGetDeviceProperties).
- V2-RCR resource usage: **246 VGPRs / 0 spill / LDS=131072 B/block /
  occupancy=2** (4096³ build; default/persistent-on byte difference is
  only the early-exit prologue, kernel body unchanged).
- Two V2-RCR blocks need 2*131072 = 262144 B of LDS = 1.6× over the
  163840 B/CU cap → LDS, not VGPR, is the binding occupancy=2 ceiling.
  Same paradigm as V2-CRR (R30 Dev B closure #2).

Structural analysis: why Approach 1 cannot help 4096³ at BLK=256
----------------------------------------------------------------
At BLK=256 the per-tile work is fixed. Define:
- `total_tiles = (M/BLK) * (N/BLK)`
- `slots = num_CUs * occupancy = 304 * 2 = 608`

Wave-fill at occupancy=2 = `min(total_tiles, slots) / slots`.

| Shape   | total_tiles | min(tiles, 608) | slot-fill | per-CU work |
|---------|-------------|-----------------|-----------|-------------|
| 4096³   | 256         | 256             | 42%       | 0.84 / CU   |
| 8192³   | 1024        | 608 (cap)       | 100%      | 3.37 / CU   |
| LLaMA8B Gate (4096×14336×4096) | 896 | 608 | 100% | 2.95 / CU |

For 4096³ V2-RCR specifically:
- Baseline grid = 256: SPI places 256 CTAs across 304 CUs at occ=1
  (256 CUs busy, 48 idle). 48 / 304 = 15.8% idle.
- Persistent grid = 304: 256 valid CTAs + 48 early-exit; SPI places
  304 CTAs one per CU. Workers still 256, idle still 48 → identical.
- Persistent grid = 608: 256 valid CTAs + 352 early-exit; SPI places
  608 CTAs at occ=2. Workers still 256 (the early-exit CTAs do
  trivial return). Idle CUs depend on whether SPI prefers spreading
  the workers across all CUs first → in the best case 256 CUs busy at
  occ=1, 48 CUs busy with 2 early-exit blocks. Worker-CU count
  unchanged.

**The 16% idle gap is a function of `total_tiles < num_CUs`, not of
how the CTAs are dispatched. No persistent-CU scheme can synthesize
work that doesn't exist in the (br, bc) tile grid.**

The only ways to close the gap structurally are:
1. **Reduce BLK** so total_tiles increases (BLK=128 → 1024 tiles for
   4096³ → 100% wave-fill; this is Approach 2, multi-day rewrite).
2. **Split-K** so each (br, bc) tile is processed by ≥2 CUs in
   parallel with atomic accumulation (closed by R27 Dev B as
   DEAD-END for V2-CRR; same V2-RCR shape has even less arithmetic
   intensity headroom).
3. **Inner-loop persistence with WORK-TILE LARGER THAN BLK** (e.g.
   each persistent CTA processes a 2×BLK × 2×BLK super-tile by
   stitching 4 inner BLK iterations) — but this requires each CTA to
   fit 4× the LDS footprint of one tile, immediately breaking the
   163840 B/CU cap. Structurally infeasible for V2-RCR's existing
   131 KB/block.

Implementation
--------------
A minimal-surface-area macro variant was added:

```c
#ifndef MXFP8_RCR_V2_PERSISTENT
#define MXFP8_RCR_V2_PERSISTENT 0
#endif
#ifndef MXFP8_RCR_V2_PERSISTENT_GRID
#define MXFP8_RCR_V2_PERSISTENT_GRID 608
#endif
```

Wired with:
1. Early-exit at top of `rcr_exact_8wave_scaled_kernel<true,2>`:
   ```c
   constexpr int total_tiles_compile = (M_DIM / BLK) * (N_DIM / BLK);
   #if MXFP8_RCR_V2_PERSISTENT
       if (blockIdx.x >= total_tiles_compile) return;
   #endif
   ```
2. Host dispatcher override in `dispatch_rcr_exact_8wave_scaled_v2`:
   `dim3 grid(MXFP8_RCR_V2_PERSISTENT_GRID)` instead of
   `dim3 grid((g.m/BLK)*(g.n/BLK))`.
3. `PY_MODULE_NAME` macro so two .so files (base + persistent) can
   live in one Python process for paired in-process A/B benching.

This is the **simplest possible persistent variant** — it tests
whether SPI redistribution at occupancy=2 helps without paying for
the inner-loop rewrite. The full inner-loop persistence would require
wrapping the 1220-line kernel body in a `for(tile_iter=blockIdx.x;
tile_iter<total_tiles; tile_iter+=gridDim.x)` loop with per-iteration
re-zero of accumulators (cA, cB, cC, cD), reset of cached_k_pair / tic
/ toc, and re-derivation of all br/bc-dependent state (V2 SRDs at
lines 2346-2369, scale_row_bases at 2298-2306, scale staging at
2380-2395, pre-loop B/A primes at 2761-2778, etc.) — bounded but
far beyond a 90-min budget.

Default-off byte-near-identical: default build md5
`6b9991c55ae054a41beb82e3413ed27e`; persistent=1 grid=608 build md5
`3c80a9adba1913e3aabd8d1d77fa52f4`. Resource report identical
(VGPRs=246, LDS=131072, occ=2). Difference is the early-exit
prologue + larger grid.

Bench protocol
--------------
GPU3 (HIP_VISIBLE_DEVICES=3, R30D_PHYS_GPU=3 for rocm-smi).

Initial single-process build-then-bench (r30d_bench.py adapted) showed
**+8.6%, Welch t=3.43** for persistent_grid=304 vs baseline. *Suspect*
since GPU3 is DPM-throttled and cold-RUN-0 dominates 4096³ std (R29C
Hardware caveat). Confirmed by re-running baseline immediately after
which dropped to ~1500 TFLOPS — environmental drift between sequential
build/bench cells artificially inflated the delta.

To eliminate sequential drift, built **paired in-process bench**
(`r31d_paired_build.sh` + `r31d_paired_bench.py`):
- Two .so files (base, p304) with `-DPY_MODULE_NAME=...` so both
  modules co-exist in one Python process.
- 30-second sustained 16k×16k preheat (vs 8s previously) to ramp sclk
  fully and stabilize thermals.
- Interleaved BABABA pattern over N pairs, both kernels exercise the
  same C, A, B tensors with identical zero+sync.
- Welch t computed on per-pair tflops series.

Results
-------

### 4096³ V2-RCR persist_grid=304 (10 pairs, paired in-process)

| | mean | stdev | median |
|---|---:|---:|---:|
| base | 1042.00 | 6.21 | 1042.09 |
| persist_grid=304 | 1042.25 | 5.62 | 1044.18 |
| Δ | **+0.25 (+0.02%)** | — | — |

Welch t = **0.10 → NULL**. (sclk dropped to ~1900 MHz mid-bench due to
DPM throttling, hence absolute TFLOPS are lower; relative delta is
the meaningful metric since both cells are interleaved at the same
clock.) Correctness PASS for both: snr_db=49.61, det_ok.

### 4096³ V2-RCR persist_grid=512 (8 pairs)

| | mean | stdev |
|---|---:|---:|
| base | 1522.53 | 11.88 |
| persist_grid=512 | 1529.08 | 2.56 |
| Δ | **+6.55 (+0.43%)** | |

Welch t = **1.53 → NULL**. Correctness PASS for both.

### 4096³ V2-RCR persist_grid=608 (8 pairs)

| | mean | stdev |
|---|---:|---:|
| base | 2440.23 | 64.72 |
| persist_grid=608 | 2474.00 | 23.65 |
| Δ | **+33.77 (+1.38%)** | |

Welch t = **1.39 → NULL**. Correctness PASS for both. (The pair-0
cold outlier in baseline drives most of this delta; excluding pair-0
yields +0.72%, t=2.47 → still NULL.)

### 8192³ V2-RCR persist_grid=1024 (8 pairs, no-regression check)

| | mean | stdev |
|---|---:|---:|
| base | 2582.63 | 181.29 |
| persist_grid=1024 | 2586.87 | 209.05 |
| Δ | **+4.24 (+0.16%)** | |

Welch t = **0.04 → NULL**. Correctness PASS for both: snr_db=49.59.
**No-regression confirmed on 8192³** (within ±1% bench band).

### CRITICAL correctness finding: persist_grid=304 on 8192³

When persistent_grid is set BELOW total_tiles, the early-exit drops
real work: 8192³ has total_tiles=1024 > grid=304 → 720 tiles never
computed → snr collapses to **1.53 dB** (vs 49.59 baseline), output
appears valid only on the early-exit slice. **Constraint**: any
persistent-CU dispatch MUST satisfy `grid ≥ total_tiles` for the
shape, OR implement true inner-loop persistence. The macro
`MXFP8_RCR_V2_PERSISTENT_GRID` is shape-dependent and cannot be a
single compile-time constant for all shapes — would need to be
`max(num_CUs * occupancy, total_tiles)` driven from the host.

Welch summary across cells
--------------------------

| Cell | shape | persist | Δ% vs base | Welch t | Verdict |
|---|---|---|---:|---:|---|
| 4096³ | 4096×4096×4096 | grid=304 | +0.02% | +0.10 | NULL |
| 4096³ | 4096×4096×4096 | grid=512 | +0.43% | +1.53 | NULL |
| 4096³ | 4096×4096×4096 | grid=608 | +1.38% | +1.39 | NULL |
| 8192³ | 8192×8192×8192 | grid=1024 | +0.16% | +0.04 | NULL no-regress |
| 8192³ | 8192×8192×8192 | grid=304 | — | — | CORRUPT (tile dropping) |

**All cells NULL** within bench noise; SHIP gate of +5% / t > 3.0 not
met. Approach 1 (early-exit-prologue persistent dispatch) is
performance-NULL.

Why the empirical NULL matches the structural prediction
--------------------------------------------------------
The math of the structural argument is borne out exactly:
- 4096³ grid=256 (baseline) → 256 working CTAs.
- 4096³ persistent grid=304 → 256 working CTAs + 48 early-exit (~0
  cost) → SAME working CTAs.
- 4096³ persistent grid=608 → 256 working CTAs + 352 early-exit →
  SAME working CTAs (occupancy=2 doesn't help when the second
  occupancy slot is filled by trivial returns).
- 8192³ persistent grid=1024 → 1024 working CTAs (= baseline
  grid=1024) → SAME work, SAME schedule (608 simultaneous, then 416
  in second wave).

The early-exit-prologue persistent variant **adds zero useful work**;
the only effect is the prologue overhead and any subtle SPI placement
change. Both are below noise.

What about an inner-loop persistent variant?
--------------------------------------------
A "true" persistent-CU prototype with grid=608 and each CTA
processing ceil(total_tiles/608) tiles via an internal loop:
- 4096³: 608 CTAs, 256 tiles → 416 CTAs do 0 tiles, 192 do 1, 0 do 2.
  Still 256 working CTAs. **No win for 4096³.**
- 8192³: 608 CTAs, 1024 tiles → all 608 do 1 tile, then 416 do a
  second tile. Total 1024 = same work. Wave-2 partial-fill remains.

**Inner-loop persistence is NOT a wave-fill optimization.** It would
only help if there were per-launch overhead (kernel-launch latency
amortizes across multiple CUs) — but the kernel is launched once
either way; the inner loop just rebalances the workload distribution
across CUs, which the SPI dispatcher already does.

The only persistent-CU win on AMD CDNA4 would come from **persistent
across multiple GEMMs** (kernel cooperative groups, reuse warmed-up
LDS/VGPR state across distinct A/B/C operands) — which is a
batching-API change, not a kernel-internal optimization.

Approach 2 (BLK=128) lookahead audit
------------------------------------
Approach 2 (rect-V2-RCR with BLK_M=256, BLK_N=128 mirror of Dev A's
CRR work) is the only structurally viable lever for 4096³ V2-RCR. LDS
audit:
- V2-RCR baseline LDS = 131072 B (As[2][2] + Bs[2][2] + scale_stage
  if enabled).
- BLK_M=256 / BLK_N=128 halves Bs footprint (half the N-dim of B
  tile) → LDS ~ 65536 + 65536 = 131072 B (As unchanged at half_M=128
  cols×128 rows × 2 buffers; Bs at 64 rows × 128 cols × 2 buffers ×
  2 staging) — likely SAME as baseline since A is the dominant tile.
- Wave-fill: 4096³ at BLK_M=256 / BLK_N=128 → grid = (4096/256) ×
  (4096/128) = 16 × 32 = 512 tiles → 304 CUs at occ=2 = 608 slots →
  84% slot-fill. **WORSE than 100% baseline at 8192³ but BETTER than
  current 42% at 4096³.** Net: ~512/304 = 1.68 waves vs current 256/
  304 = 0.84 waves → 2× the work distributed → 2× longer run, but
  perfect wave-fill on first wave. Net potential gain ≈ (1.68 - 0.84)
  / 1.68 = 50% better wave-fill, but 2× the K-tile work per wave
  cancels — **net potential 50% / 2 = 25% better throughput at best,
  before per-tile-overhead penalties from rect.** For a target
  +120 TFLOPS on 4096³ (~5%), this is achievable in principle but
  requires the 2.5-day rewrite estimated by R30 Dev A for the rect
  CRR variant (mirrored to RCR is similar scope).

**Recommendation for R32 Dev A**: Approach 2 (rect BLK_M=256/BLK_N=128
for V2-RCR) is the only remaining lever for the 4096³ V2-RCR gap;
budget at least 2.5 days. R30 Dev A's `r30a_findings.md` line
breakdown (rect-V2-CRR Path A) provides the template; the V2-RCR
codepath has 4 MMA quadrants (vs CRR's 2) so scaffolding will be
roughly 1.5× the CRR effort.

Files
-----
- `kernel_mxfp8_layouts.cpp`:
  - +2 macros (`MXFP8_RCR_V2_PERSISTENT`, `MXFP8_RCR_V2_PERSISTENT_GRID`)
    + `PY_MODULE_NAME` for paired binaries.
  - Kernel: early-exit prologue at top of
    `rcr_exact_8wave_scaled_kernel<true,2>`.
  - Host: `dispatch_rcr_exact_8wave_scaled_v2` grid override.
  - All default-off; default build resource usage unchanged
    (246 VGPRs / 0 spill / LDS=131072 / occ=2).
- `r31d_paired_build.sh`: builds two .so files (base, p304) with
  distinct `PY_MODULE_NAME` for in-process A/B.
- `r31d_paired_bench.py`: paired in-process bench, 30-s preheat,
  BABA pattern, Welch t.
- `r31d_ab_persist.sh`: alternate orchestrator for A/B/A/B/A
  sequential build-then-bench (used to demonstrate that sequential
  bench yields false-positive +8.6% from sclk drift, confirming the
  paired protocol as the correct measurement).
- `r31d_baseline_4096.txt` / `r31d_baseline_8192.txt`: initial
  single-process baselines.
- `r31d_persist304_4096.txt` / `r31d_persist608_4096.txt`: initial
  single-process persistent variants (DRIFT-AFFECTED — see
  `r31d_paired_4096.txt` for the paired ground-truth).
- `r31d_paired_4096.txt`, `r31d_paired_4096_p512.txt`,
  `r31d_paired_4096_p304_v2.txt`: paired 4096³ A/B at three
  persistent grids — all NULL.
- `r31d_paired_8192.txt`: paired 8192³ A/B persist_grid=304 →
  CORRECTNESS FAIL (snr 1.53 dB) due to tile-dropping; the +161%
  "perf" is a measurement artifact of skipped work.
- `r31d_paired_8192_p1024.txt` / `r31d_paired_8192_p1024_v2.txt`:
  paired 8192³ A/B persist_grid=1024 (matches total_tiles) → NULL,
  no regression.
- `r31d_md5_*.txt`: per-cell .so md5 confirming fresh artifacts.

Final outcome
-------------
**NO SHIP (perf NULL) → AUDIT-ONLY closure of Approach 1.**

Approach 1 (persistent-CU dispatch via early-exit prologue + grid
override) is performance-NULL on V2-RCR for both 4096³ and 8192³,
matching the structural prediction. The 4096³ wave-fill ceiling
(0.84 = 16% idle CUs at BLK=256, total_tiles=256 < 304 CUs) cannot
be closed by any persistent-CU scheme that operates on (br, bc) tile
units of the existing geometry — the missing work simply doesn't
exist in the tile grid.

**R30 cumulative lever-closure tally extends to 15 closed levers.**
Approach 2 (rect BLK_M=256/BLK_N=128 for V2-RCR) is the only
remaining lever for the 4096³ V2-RCR gap and is multi-day work for
R32+.

The `MXFP8_RCR_V2_PERSISTENT` macro is left in tree as default-off
scaffolding documenting that the persistent-dispatch lever was
empirically tested.

Cumulative R27-R31 lever-closure list (15 closed)
-------------------------------------------------
1. s_setprio for V2-CRR (R28 Dev B)
2. s_setprio for V2-RCR (R29 Dev C)
3. sched_barrier mask relax for V2-RCR (R29 Dev C)
4. cachepolicy gate broadening beyond R28 boundary (R29 Dev C)
5. cachepolicy for V2-RCR (R27 + R29 confirm)
6. Scale LDS double-buffer / SCALE_LDS REPLACE (R23/R27)
7. Split-K-along-K (R27 Dev B)
8. V1 vs V2 runtime size-gating (R26)
9. Square BLK=128 rewrite (R27)
10. LDS bank conflicts for V2-CRR (R29)
11. Occupancy=3 via VGPR reduction (R30 Dev B — LDS-binding)
12. buffer_load_dword_lds for V2 scales (R30 Dev C — already direct-to-VGPR)
13. H7 B-tile reorder for V2-RCR (R30 Dev D)
14. (R30 cumulative entry)
15. **Persistent-CU dispatch via early-exit prologue for V2-RCR (R31 Dev D)**

Paradigm correction (R31D, extends R30D's structural-ceiling note)
-------------------------------------------------------------------
**Persistent-CU dispatch with early-exit prologue cannot improve
wave-fill when `total_tiles ≤ num_CUs`.** Mathematically: the missing
work simply doesn't exist in the (br, bc) grid. NEVER prototype
"persistent-CU at grid=608 for 4096³" again — it's structurally
identical to baseline. The only correct persistent variants for
shapes with `total_tiles > num_CUs` are (a) inner-loop persistent
with `grid = num_CUs * occupancy` and per-CTA inner loop, or (b)
work-stealing via atomic counter — both require the full inner-loop
rewrite (1220-line scope on V2-RCR). Both are multi-day; even fully
implemented, neither helps for 4096³ where `total_tiles=256 < 304`
CUs anyway.
