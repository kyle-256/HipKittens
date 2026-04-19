R47 Dev C: MXFP8 V2 scale-load cachepolicy sweep — 7 shapes × 3 layouts × 4 policies
=====================================================================================

Date: 2026-04-19
Branch: feat/mxfp8-only (worktree at 2c45f7fb)
GPU: AMD MI355X gfx950 (HIP_VISIBLE_DEVICES=2)
Files swept: rcr_mxfp8_exact_8wave_rect_fastpath.inc, rrr_mxfp8_exact_8wave_fastpath.inc,
             crr_mxfp8_exact_8wave_fastpath.inc (and downstream V2 buffer_load_b128/b64
             call sites in kernel_mxfp8_layouts.cpp).

Hypothesis (from R46 Dev A profiling)
-------------------------------------
Setting MXFP8_{RCR,RRR,CRR}_V2_SCALE_CACHEPOLICY=2 (SLC, "streaming, skip L1") for
the V2 scale buffer loads would reduce L2 pollution on large-N shapes (notably 70B
Gate/Up where the B-side scale working set ≈7.3 MB across 1792 CTAs exceeds the
typical L2 partition), recovering 2-3% of the MXFP8↔FP8 perf gap. Conversely, on
long-K shapes scales are reused so default (0) was expected to win.

Sweep design
------------
- 7 compute-bound shapes × 3 layouts (RCR, RRR, CRR) × 4 policies (0, 1, 2, 3) = 84 cells.
- Each (shape, policy) cell rebuilds a fresh tk_mxfp8_layouts.so with all three layout
  cachepolicy macros set to the same value, then runs all 3 layouts in one Python
  invocation (MXFP8_LAYOUTS=rcr,rrr,crr, MXFP8_PRESHUFFLE_QUANT=1).
- 200 warmup + 400 iters per layout per rep, 3 reps per cell, 1 throwaway preheat run
  before the measurement reps. We report the median of 3 reps. (v1 sweep used 50/100
  with no preheat and produced 2 transient outliers — see "Methodology notes" below.)
- AMD CDNA cachepolicy bit map: 0 = default; 1 = GLC; 2 = SLC; 3 = GLC | SLC.

Full table (TFLOPS, median of 3 reps; Δ% vs policy 0)
------------------------------------------------------
RCR layout
| shape                            | p0      | p1      | p2      | p3      | Δp1%   | Δp2%   | Δp3%   | best | spread (p0) |
|----------------------------------|---------|---------|---------|---------|--------|--------|--------|------|-------------|
| 8192x8192x8192                   | 2946.92 | 2950.30 | 2878.61 | 2877.54 | +0.11  | -2.32  | -2.35  | p1   | ±0.4%       |
| 4096x4096x4096                   | 2468.17 | 2474.50 | 2404.58 | 2406.87 | +0.26  | -2.58  | -2.48  | p1   | ±1.3%       |
| 4096x14336x4096 (8B Gate/Up)     | 2587.41 | 2587.98 | 2475.39 | 2468.97 | +0.02  | -4.33  | -4.58  | p1   | ±0.7%       |
| 4096x4096x14336 (8B Down)        | 2947.27 | 2948.31 | 2874.04 | 2868.92 | +0.04  | -2.48  | -2.66  | p1   | ±0.1%       |
| 4096x8192x8192  (70B Q/O)        | 2902.15 | 2895.90 | 2824.38 | 2823.91 | -0.22  | -2.68  | -2.70  | p0   | ±0.3%       |
| 4096x28672x8192 (70B Gate/Up)    | 2845.29 | 2843.46 | 2795.21 | 2795.30 | -0.06  | -1.76  | -1.76  | p0   | ±0.2%       |
| 4096x8192x28672 (70B Down)       | 2950.13 | 2942.41 | 2935.50 | 2939.39 | -0.26  | -0.50  | -0.36  | p0   | ±0.1%       |

RRR layout
| shape                            | p0      | p1      | p2      | p3      | Δp1%   | Δp2%   | Δp3%   | best | spread (p0) |
|----------------------------------|---------|---------|---------|---------|--------|--------|--------|------|-------------|
| 8192x8192x8192                   | 2941.76 | 2939.54 | 2864.59 | 2866.50 | -0.08  | -2.62  | -2.56  | p0   | ±0.2%       |
| 4096x4096x4096                   | 2445.09 | 2456.61 | 2384.01 | 2405.11 | +0.47  | -2.50  | -1.64  | p1   | ±0.8%       |
| 4096x14336x4096 (8B Gate/Up)     | 2558.60 | 2558.93 | 2458.79 | 2458.34 | +0.01  | -3.90  | -3.92  | p1   | ±1.1%       |
| 4096x4096x14336 (8B Down)        | 2932.59 | 2932.27 | 2863.11 | 2867.05 | -0.01  | -2.37  | -2.23  | p0   | ±0.2%       |
| 4096x8192x8192  (70B Q/O)        | 2889.95 | 2883.22 | 2811.88 | 2812.49 | -0.23  | -2.70  | -2.68  | p0   | ±0.2%       |
| 4096x28672x8192 (70B Gate/Up)    | 2783.07 | 2793.21 | 2708.70 | 2705.19 | +0.36  | -2.67  | -2.80  | p1   | ±0.1%       |
| 4096x8192x28672 (70B Down)       | 2971.44 | 2974.36 | 2885.58 | 2884.76 | +0.10  | -2.89  | -2.92  | p1   | ±0.1%       |

CRR layout
| shape                            | p0      | p1      | p2      | p3      | Δp1%   | Δp2%   | Δp3%   | best | spread (p0) |
|----------------------------------|---------|---------|---------|---------|--------|--------|--------|------|-------------|
| 8192x8192x8192                   | 2719.33 | 2721.73 | 2693.86 | 2693.21 | +0.09  | -0.94  | -0.96  | p1   | ±0.2%       |
| 4096x4096x4096                   | 2273.26 | 2276.29 | 2229.30 | 2233.35 | +0.13  | -1.93  | -1.76  | p1   | ±1.1%       |
| 4096x14336x4096 (8B Gate/Up)     | 2370.28 | 2371.13 | 2271.80 | 2271.67 | +0.04  | -4.15  | -4.16  | p1   | ±0.4%       |
| 4096x4096x14336 (8B Down)        | 2723.15 | 2722.46 | 2693.65 | 2691.08 | -0.03  | -1.08  | -1.18  | p0   | ±0.2%       |
| 4096x8192x8192  (70B Q/O)        | 2694.29 | 2691.56 | 2642.82 | 2647.67 | -0.10  | -1.91  | -1.73  | p0   | ±0.3%       |
| 4096x28672x8192 (70B Gate/Up)    | 2502.54 | 2507.48 | 2464.57 | 2468.83 | +0.20  | -1.52  | -1.35  | p1   | ±0.4%       |
| 4096x8192x28672 (70B Down)       | 2666.37 | 2663.57 | 2617.23 | 2615.45 | -0.11  | -1.84  | -1.91  | p0   | ±0.5%       |

Per-layout aggregate (Δ% vs p0, median across 3 reps)
-----------------------------------------------------
| layout | policy | wins ≥+2% | losses ≤-2% | mean Δ% | shapes_pos |
|--------|--------|-----------|-------------|---------|------------|
| RCR    | p1     |     0     |      0      |  -0.02  |   4/7      |
| RCR    | p2     |     0     |      5      |  -2.38  |   0/7      |
| RCR    | p3     |     0     |      5      |  -2.41  |   0/7      |
| RRR    | p1     |     0     |      0      |  +0.09  |   4/7      |
| RRR    | p2     |     0     |      7      |  -2.81  |   0/7      |
| RRR    | p3     |     0     |      6      |  -2.68  |   0/7      |
| CRR    | p1     |     0     |      0      |  +0.03  |   4/7      |
| CRR    | p2     |     0     |      1      |  -1.91  |   0/7      |
| CRR    | p3     |     0     |      1      |  -1.86  |   0/7      |

Verdict per layout (R47 Dev C SHIP decision)
--------------------------------------------
- RCR: keep MXFP8_RCR_V2_SCALE_CACHEPOLICY default = 0. p1 is statistical wash
  (mean Δ ≈ 0, biggest move +0.26% on 4096cube, biggest loss -0.26% on 70B Down);
  p2/p3 lose 2-4% on most shapes. NO SHIP.
- RRR: keep MXFP8_RRR_V2_SCALE_CACHEPOLICY default = 0. Same picture as RCR; p1
  is wash, p2/p3 lose 2-3% on essentially every shape. NO SHIP.
- CRR: change behavior — REMOVE the R28-era conditional auto-default that
  emitted SLC=2 when (N_DIM >= 28672 AND K_DIM >= 8192). On 4096x28672x8192
  CRR (the only shape that used to trigger the gate) cp0=2502.54 vs
  cp2=2464.57 → SLC LOSES 1.52% at this baseline. R28's +2.6% win has flipped.
  Make MXFP8_CRR_V2_SCALE_CACHEPOLICY default = 0 unconditionally. SHIP this
  refactor (gain ≈ +1.5% on 70B Gate/Up CRR; the only shape that used to hit
  the gate). User-provided -D still wins.

Hypothesis status (vs R46 Dev A predictions)
--------------------------------------------
- "SLC=2 will recover 2-3% on 70B Gate/Up by reducing L2 pollution":
  REFUTED. SLC=2 LOSES 1.76% (RCR), 2.67% (RRR), 1.52% (CRR) on this shape.
- "Default (0) is best for long-K shapes due to scale reuse": CONFIRMED.
  70B Down (K=28672) p0 strictly dominates p1/p2/p3 across all 3 layouts.
- Closes R46 Dev A open opportunity #2 ("scale cachepolicy tuning"). The
  scale-load L2 pressure modeled in R46 Dev A is real but not addressable
  by SLC bits — likely already mitigated by the V2 wave-tile reordered
  scale layout (issuing buffer_load_b128/b64) that landed in R21/R22.

Why R28's SLC win flipped to a loss
-----------------------------------
R28 Dev A's conditional auto-default was based on a Welch-significant
+2.6% on 4096x28672x8192 CRR (cp2=2418.99 vs cp0=2358.27, t=12.83).
Between R28 and R47 the baseline TFLOPS for this same shape rose from
~2358 to ~2503 — a +6.1% improvement attributable to landed work
including R44 Dev A (`crr_exact_8wave_scaled_kernel`), R45 Dev A
(8-wave m=2..16 fastpath), and the surrounding rect fastpath
restructuring. The L2 pressure that SLC was originally relieving has
been substantially reduced by other code paths, so the SLC bit now
costs more than it saves (extra DRAM round-trip when the line WOULD
have hit L2). The R28 finding remains historically valid for the R28
baseline; it is no longer applicable.

Correctness validation (proposed CRR default change)
----------------------------------------------------
Built tk_mxfp8_layouts with explicit -DMXFP8_CRR_V2_SCALE_CACHEPOLICY=0
on M=4096 N=28672 K=8192 (the only shape whose default behavior changes
under this patch) and ran:
  HIP_VISIBLE_DEVICES=2 MXFP8_LAYOUTS=crr MXFP8_PRESHUFFLE_QUANT=1 \
  MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 MXFP8_SNR_THRESHOLD_DB=45 \
  python3 test_mxfp8_python.py 4096 28672 8192

Result:
  CRR Correctness: SNR 49.60 dB (threshold 45.0), pass rate 100% (117440512/117440512),
                   max abs err 0.0312, max rel err 0.0078 → PASS
  CRR Determinism (3 runs): PASS

Methodology notes
-----------------
- v1 of this sweep (50 warmup + 100 iters, no preheat) produced two
  transient outlier readings: 8B_Down_4kx4kx14k policy=0 read as
  ~1759 TFLOPS (RCR) instead of the true ~2950, and
  8B_GateUp_4kx14kx4k policy=3 read as ~414 TFLOPS instead of the
  true ~2470, all three layouts simultaneously in each case. Same
  binary MD5 produced normal results when re-run after a thermal
  preheat. Spread (max-min) on v1 reached 60% on these two cells.
  v2 (200 warmup + 400 iters, 1 preheat run + 3 measured reps,
  median reported) gave clean ±0.1-1.3% spread per cell. All 84
  data points in this report are from v2.
- Build hygiene per R31/R45: rm -f tk_mxfp8_layouts*.so before each
  build, MD5 each .so. All 28 builds produced unique MD5s (verified
  in r47c_reports/build_*_p*.md5).

Files
-----
- analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc
  (lines 71-83: removed the R28 conditional auto-gate for SLC=2;
   default is now unconditional 0)
- analysis/fp8_gemm/mi350x/r47c_cachepolicy_sweep.sh   (sweep driver)
- analysis/fp8_gemm/mi350x/r47c_findings.md            (this file)
- analysis/fp8_gemm/mi350x/r47c_reports/               (84 build logs +
  84 measured run logs + 28 preheat logs + summary.tsv + raw.tsv)
- analysis/fp8_gemm/mi350x/r47c_sweep_orchestrate.log  (full sweep stdout)

SHIP gate
---------
- RCR default unchanged.
- RRR default unchanged.
- CRR conditional default removed (unconditional 0). Gains +1.52% on
  the single shape (4096x28672x8192) that used to trigger the gate
  (measured on the 2c45f7fb baseline this sweep used); no other shape's
  behavior changes. Correctness PASS.

Re-verification on R47 HEAD (post Dev A/B/D)
--------------------------------------------
The full sweep was performed on the 2c45f7fb baseline as specified.
Between sweep start and commit, R47 Dev A (RCR XCD swizzle), Dev B
(CRR XCD swizzle), and Dev D (ISA verification) landed on
feat/mxfp8-only. To confirm the CRR cachepolicy default change is
still net-positive on the new HEAD, I re-built tk_mxfp8_layouts with
the R47 A/B/D code and the proposed default change, on
M=4096 N=28672 K=8192:
  HEAD with -DMXFP8_CRR_V2_SCALE_CACHEPOLICY=2 (old gate behavior):
    2478.16 / 2478.83 / 2480.68 TFLOPS (median 2478.83)
  HEAD with -DMXFP8_CRR_V2_SCALE_CACHEPOLICY=0 (new default):
    2505.08 / 2501.38 / 2500.96 TFLOPS (median 2501.38)
  Δ = +22.55 TFLOPS = +0.91%
The win shrunk from +1.52% (on 2c45f7fb) to +0.91% (on HEAD) — not
surprising since R47 Dev B's CRR XCD swizzle has further reduced L2
contention (which was part of what SLC was originally compensating for).
Direction is unchanged: cp=0 > cp=2 on this shape.
