R26 Dev A — V2 70B KV-attn CRR (M=4096 N=1024 K=8192) findings
================================================================

Goal: improve V2-CRR/FP8-CRR ratio at 4096x1024x8192 (R25 baseline 0.69).

Hardware caveat
---------------
GPU3 sclk capped at ~1700 MHz under sustained load (vs 2400 MHz spec /
2300 MHz target). All A/B comparisons remain valid since both arms run
on the same throttled GPU within the same Python process; absolute
TFLOPS numbers are NOT directly comparable to R25 figures.

Per-process preheat protocol (see preheat_then_bench.py / r26a_h3_5x.py):
  - 6-8s sustained 16k×16k fp16 matmul before benchmarking
  - keeps DPM state warm; sclk holds ~1670 MHz steady during MXFP8 bench

Baseline (preheat ON, GPU3, MXFP8_PRESHUFFLE_QUANT=1)
-----------------------------------------------------
  V2-CRR-PQ : 793 TFLOPS   (5x median 796.1, std 1.21 — see r26a_h3_5x)
  FP8-CRR   : 942 TFLOPS
  ratio     : 0.842   ←  already much better than R25-reported 0.69

The R25 0.69 figure was likely measured cold (DPM-throttled MXFP8 path)
in the multi-shape sweep, where FP8 (lighter scale traffic) suffers less
from the throttle than V2-CRR. With per-shape preheat the gap collapses.

H3 — V1 fallback (MXFP8_CRR_PRESHUFFLE_V2_RUNTIME=0)
-----------------------------------------------------
5x interleaved A/B in one process (r26a_h3_5x.py):
  V2 median = 796.09  std 1.21
  V1 median = 791.64  std 0.86
  V1 - V2   = -4.45 TFLOPS  (-0.56%)
  Welch t   = -5.87  (significant in WRONG direction)
  SNR       = 49.60 dB (V1) / 49.60 dB (V2) — both PASS
Verdict: NULL (V1 is statistically WORSE than V2 by ~0.6%).
        Do NOT add a runtime by-shape gate.

H1 — smaller N-tile (BLOCK_N = 128)
------------------------------------
Inspecting crr_mxfp8_exact_8wave_fastpath.inc:37 + crr_mxfp8_4wave_fastpath.inc:86:
  static_assert(BLK == 256, ...);
  static_assert(BK  == 128, ...);
BLK is hardcoded across BOTH the 8-wave and 4-wave CRR fastpaths plus
all RCR/RRR siblings. The kernel structure (RBM/RBN, scale pack counts,
shared-tile shapes ST_v2/ST_v2a, V2 preshuffle layout) all assume
BLK=256. Halving N-tile would require a parallel kernel template plus a
separate V2 preshuffle helper — multi-day rewrite, out of scope.
Verdict: DEAD-END (architectural).

H4 — smaller K-tile (BLOCK_K = 64 instead of 128)
--------------------------------------------------
Same static_assert(BK == 128) blocker on all paths. Smaller BK changes
RT/ST tile shapes and would invalidate the V2 b128 scale-load pattern.
Verdict: DEAD-END (architectural).

Sanity: 8192x8192x8192 V2-CRR
------------------------------
2788 TFLOPS, PASS. Below R23 baseline 2900 (GPU3 throttle), no
candidate ships, no regression risk.

Recommended next step (R26+)
----------------------------
1. Re-measure R25 LLaMA matrix on a non-throttled GPU (cold-baseline
   protocol per R24) WITH the per-shape preheat patch landed in
   run_llama_baseline.sh — likely most "70B KV-attn 0.69 gap" entries
   collapse to ratio ≥0.80 just from clean measurement.
2. The remaining N=1024 inefficiency (304 CUs / 16 blocks = 19 blocks/CU
   wave-time) is a wave-occupancy problem, not a kernel-algorithm
   problem. Fix candidates require new kernel template:
     a. Split-K accumulator (reduce M-tiles, parallelize over K)
     b. Persistent-CTA with grid-stride loop over (M,N) tiles
     c. New 128x128 BLK template (architectural rewrite)
   None achievable in a 90-min slot.

Files
-----
  preheat_then_bench.py    : preheat wrapper around test_mxfp8_python.py
  preheat_fp8_bench.py     : preheat wrapper around test_python.py (FP8)
  r26a_h3_5x.py            : H3 5x interleaved A/B in single process
  r26a_ab_h3.py            : (early dev tooling, kept for reference)
