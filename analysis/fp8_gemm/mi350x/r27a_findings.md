R27 Dev A — gfx950 cachepolicy bits on V2 scale loads
======================================================

Hypothesis: routing V2 scale `buffer_load_b128`/`b64` through L1/L2
differently (via the `cachepolicy` operand of the LLVM raw-buffer-load
intrinsic) might free TA bus for A/B loads. Default in source today is
`coherency=0`. We swept `coherency = 0,1,2,3`, mapping (per AMDGPU
LLVM): bit0 = GLC, bit1 = SLC.

Patch (kept in tree, defaults restored to 0)
--------------------------------------------
Three new build-time macros:

  `MXFP8_CRR_V2_SCALE_CACHEPOLICY` (default 0)
  `MXFP8_RCR_V2_SCALE_CACHEPOLICY` (default 0)
  `MXFP8_RRR_V2_SCALE_CACHEPOLICY` (default 0)

Wired into the 4 V2 buffer-load sites in
`kernel_mxfp8_layouts.cpp` (RCR) and into the V2 branches of
`crr_mxfp8_exact_8wave_fastpath.inc` and
`rrr_mxfp8_exact_8wave_fastpath.inc`. With all macros = 0 the binary
is byte-identical to baseline; macros only affect the 4th argument of
`llvm_amdgcn_raw_buffer_load_b{128,64}`.

Hardware caveat
---------------
GPU0 sclk DPM-throttled (idles at ~94 MHz, ramps slowly under load).
Per-process preheat (`preheat_then_bench.py`) used. 5x runs each in a
single process; first 1-2 runs of small kernels (4096³ RCR) still show
ramp variance — see std columns. All A/B comparisons are within the
same process so relative deltas valid.

Results — H1 cachepolicy sweep
------------------------------
Each cell: 5x same-process bench, MXFP8_WARMUP=50 MXFP8_ITERS=100,
preheat ON. cp=0 is baseline.

70B Gate V2-CRR (M=4096 N=28672 K=8192) — TARGET
  cp=0 : median 2355.88, std  9.18  | snr 49.59 | det PASS
  cp=1 : median 2360.61, std  7.28  | Δ +4.7,  Welch t≈0.91   NULL
  cp=2 : median 2418.99, std  2.87  | Δ +63.1, Welch t≈+13.2  WIN +2.7%
  cp=3 : median 2429.91, std 11.72  | Δ +74.0, Welch t≈+11.0  WIN +3.1%

70B KV V2-CRR (M=4096 N=1024 K=8192) — TARGET
  cp=0 : median 785.89, std 1.68 | snr 49.60 | det PASS
  cp=1 : median 787.92, std 4.39 | Δ +2.0,  ~null
  cp=2 : median 731.67, std 2.23 | Δ -54.2, Welch t≈-46  REGRESSION -6.9%
  cp=3 : median 730.57, std 2.11 | Δ -55.3, Welch t≈-47  REGRESSION -7.0%

4096³ V2-RCR (M=N=K=4096) — TARGET
  cp=0 : median 2501.23, std 99.38 (sclk-ramp variance)
  cp=1 : median 2499.48, std 105.92 ~null
  cp=2 : median  100.59, std  0.11  CATASTROPHIC -25x
  cp=3 : median 2453.16, std 92.72  ~regression

8B Gate-up V2-CRR (M=4096 N=14336 K=4096) — extra check
  cp=0 : 2432.22 ± 33.85
  cp=1 : 2435.49 ± 47.50  ~null
  cp=2 : 2330.31 ± 43.49  -101.9 (-4.2%)
  cp=3 : 2330.86 ± 45.65  -101.4 (-4.2%)

Regression check — 8192³
  V2-CRR cp=0 2817 ± 17, cp=1 2835, cp=2 2807 (-0.4%), cp=3 2799 (-0.6%)
  V2-RCR cp=0 3091 ± 28, cp=1 3118, cp=2 3022 (-2.3%), cp=3 3038 (-1.7%)

Constraint reference (R23 baseline, non-throttled):
  V2-RCR ≥ 3180, V2-CRR ≥ 2900. cp=2 V2-RCR 8192³ would breach
  the 3180 floor by ~5% on a clean GPU, so cp=2 is NOT a safe global
  RCR ship.

Correctness
-----------
SNR and determinism PASS for all variants on all cells (snr ≥49.59 dB,
det 3/3 equal). cp=2 catastrophic slowdown on 4096³ RCR is purely a
perf collapse (likely a TA-bus deadlock-like contention pattern, since
SLC=1 forces L2-bypass and per-wave VMEM serializes); the kernel still
produces correct results.

Verdict per hypothesis
----------------------
H1 (cachepolicy bits): SHAPE-DEPENDENT. SLC=1 (cp=2 or cp=3) wins big
  on the largest K=8192, large-N CRR shape (70B Gate +63/+74 TFLOPS,
  +2.7%/+3.1%), but regresses on every smaller / RCR / KV shape we
  measured, including catastrophic regression on 4096³ RCR (cp=2).

H2 (s_setprio): NOT REACHED — H1 produced a non-NULL gain on the
  highest-impact target so we did not fall through to H2 in the
  90-min budget. Recommended next: try `s_setprio 3` on the
  scale-issue wave for 70B Gate cp=3 build to compound the win, then
  s_setprio 0 on MFMA waves for inverse pairing.

H3 (force-L1, glc=0 dlc=0): NOT REACHED. The LLVM intrinsic interprets
  cp=0 as already glc=0 dlc=0, so this is the current default.

Recommended SHIP variant
------------------------
**No global SHIP.** A by-shape gate would be required, but per project
constraints (no runtime gates added based on a single agent's data),
this does not meet the SHIP bar.

If a 70B-Gate-only build target exists in the LLaMA matrix
(`MXFP8_CRR_V2_SCALE_CACHEPOLICY=2` baked at compile time for
M=4096 N=28672 K=8192), it ships +2.7% (Welch t ≈ +13). cp=3 nudges
+3.1% but with higher variance and -1.7% regression on 8192³ RCR which
is in the same translation unit, so cp=3 cannot coexist.

To make this ship-able as a global default we would need:
  (a) per-shape autotune slot for `MXFP8_CRR_V2_SCALE_CACHEPOLICY`
      (separate kernel template instantiation per shape's tile config),
      OR
  (b) a runtime cachepolicy plumbed via SRD bits or a small
      shape-keyed dispatcher (architectural).

Files modified
--------------
  analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc
      + MXFP8_CRR_V2_SCALE_CACHEPOLICY macro (default 0)
      patched 2 buffer_load_b{128,64} sites (V2 SCALE_VERSION==2 branch)
  analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc
      + MXFP8_RRR_V2_SCALE_CACHEPOLICY macro (default 0)
      patched 2 buffer_load_b{128,64} sites
  analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp
      + MXFP8_RCR_V2_SCALE_CACHEPOLICY macro (default 0)
      patched 4 buffer_load_b{128,64} sites (V2 RCR exact + RCR PQ V2)

Tooling added (analysis/fp8_gemm/mi350x/)
  r27a_bench5x.py     — preheat-once + 5x bench in single process,
                        per layout, with correctness + determinism check
  r27a_orchestrate.sh — for cp in 0..3: build + invoke r27a_bench5x.py
  r27a_sweep.py       — alt single-process sweep driver (multi-process
                        version, kept for reference)
  r27a_*.txt          — per-cell raw run logs
  build_cp{0..3}_*.log — per-cell build logs

Default behavior unchanged: with all `*_V2_SCALE_CACHEPOLICY=0` the
emitted assembly is identical to pre-R27 baseline.
