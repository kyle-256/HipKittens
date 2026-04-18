R29 Dev B — cp=2 sweep on additional CRR shapes; gate refinement assessment
==========================================================================

TL;DR — NO SHIP
---------------
Current gate `MXFP8_CRR_V2_SCALE_CACHEPOLICY=2 iff N_DIM>=28672 AND K_DIM>=8192`
is **optimal as is**. Five new CRR shapes were swept on GPU1 (5x preheat+bench
each, cp=0 vs cp=2). cp=2 produced **zero positive-delta shapes**:

- 70B Down (M=4096 N=8192 K=28672): −0.11% (noise)
- 8B Down  (M=4096 N=4096 K=14336): −2.75%
- N=8192  K=14336 synthetic:        −3.34%
- N=14336 K=8192  synthetic:        −0.98%
- N=20480 K=8192  synthetic:        −2.53%

Even at N=20480 with K=8192 (the closest cp=0 → cp=2 region boundary),
cp=2 loses 2.53% with Welch t=−27.5 (overwhelmingly significant).
The WIN region is tightly localized to the 70B-Gate corner (N=28672, K=8192)
and does NOT extend to either neighbouring N (N=20480 LOSE) or to N=8192
even when K is huge (70B Down K=28672 is neutral, not a win).

Recommendation: **keep the R28 gate unchanged**. R28's auto-gate is correct.

---

Step 1 — Shape verification (from r27_reviewer_baseline_gpu4.json)
-------------------------------------------------------------------

Confirmed from `analysis/fp8_gemm/mi350x/r27_reviewer_baseline_gpu4.json`:

| Cell           | M    | N     | K     | Layout | R27 ratio |
|----------------|------|-------|-------|--------|-----------|
| 8b_gate_crr    | 4096 | 14336 | 4096  | crr    | 0.9216    |
| 8b_down_crr    | 4096 | 4096  | 14336 | crr    | 0.9394    |
| 70b_qo_rcr     | 4096 | 8192  | 8192  | rcr    | 0.9373    |
| 70b_kv_crr     | 4096 | 1024  | 8192  | crr    | 0.8646    |
| 70b_gate_crr   | 4096 | 28672 | 8192  | crr    | 0.8351    |
| **70b_down_crr** | **4096** | **8192** | **28672** | **crr** | **0.8459** |
| 8k_{rcr,rrr,crr} | 8192 | 8192 | 8192  | *      | ~0.93     |
| 4k_rcr           | 4096 | 4096 | 4096  | rcr    | 0.9229    |

**70B Down is M=4096 N=8192 K=28672** (large K, medium N) — confirmed.
This is the K-dominant case; N is only 8192, well below the gate's
N=28672 threshold.

No discrepancies with TODO.md or R28 docs.

Step 2 — cp=2 sweep on un-tested shapes
----------------------------------------

Bench harness: `r29b_bench.py` (mirrors `r28a_bench5x.py`, GPU1).
- Preheat: 8s sustained 16k×16k fp16 matmul
- 5 reps each, MXFP8_WARMUP=50 MXFP8_ITERS=100
- Each shape built twice: explicit `-DMXFP8_CRR_V2_SCALE_CACHEPOLICY=0`
  and `=2` so the auto-gate is bypassed and we measure the policy itself.
- Correctness: SNR ≥ 49.59 dB, pass_rate=100%, determinism 3/3 on every cell.

Results (TFLOPS, mean ± std, n=5):

| Shape (M×N×K, layout) | cp=0 mean ± std | cp=2 mean ± std | Δ TFLOPS | Δ %    | Welch t | Verdict |
|-----------------------|-----------------|-----------------|----------|--------|---------|---------|
| 70B Down 4096×8192×28672 crr   | 2528.67 ± 6.20  | 2525.78 ± 2.82  |  −2.88   | −0.11% |  −0.95  | NEUTRAL |
| 8B Down  4096×4096×14336 crr   | 2747.96 ± 62.31 | 2672.35 ± 67.90 | −75.62   | −2.75% |  −1.83  | LOSE    |
| K14336   4096×8192×14336 crr   | 2869.82 ± 17.99 | 2773.90 ± 20.70 | −95.92   | −3.34% |  −7.82  | LOSE    |
| N14336   4096×14336×8192 crr   | 2541.94 ± 12.49 | 2517.12 ± 14.60 | −24.82   | −0.98% |  −2.89  | LOSE    |
| N20480   4096×20480×8192 crr   | 2472.11 ±  2.95 | 2409.67 ±  4.13 | −62.45   | −2.53% | −27.51  | LOSE    |

Median view (more robust to RUN0 cold ramp seen in 8B Down):

| Shape                           | cp=0 median | cp=2 median | Δ med  | Δ %    |
|---------------------------------|-------------|-------------|--------|--------|
| 70B Down 4096×8192×28672        |  2527.51    |  2526.36    |  −1.15 | −0.05% |
| 8B Down  4096×4096×14336        |  2774.17    |  2704.93    | −69.24 | −2.50% |
| K14336   4096×8192×14336        |  2864.38    |  2775.93    | −88.45 | −3.09% |
| N14336   4096×14336×8192        |  2546.13    |  2521.10    | −25.03 | −0.98% |
| N20480   4096×20480×8192        |  2471.34    |  2411.91    | −59.43 | −2.40% |

(70B Q/O is RCR — `MXFP8_CRR_V2_SCALE_CACHEPOLICY` is not consumed by the
RCR codepath, so a cp=2 build there has no effect. Skipped per task guidance.)

Step 3 — Decision rationale
---------------------------

The R28 gate `(N>=28672) AND (K>=8192)` selects cp=2 for exactly one
LLaMA cell: 70B Gate (M=4096 N=28672 K=8192). R28 measured +2.61%
Welch t=+12.83 there. R29 Dev B asks: does that WIN region extend to
any other shape, or is the gate too narrow?

The five new data points uniformly answer **no**:

1. **K is not enough** to trigger the win. 70B Down has K=28672 (3.5× the
   gate threshold) but N=8192 — cp=2 is a wash (−0.11%, |t|<1).
   K14336 (N=8192, K=14336) is even worse: cp=2 loses 3.3%.
2. **N alone is not enough either.** N=14336 with K=8192 loses 1.0%.
   N=20480 with K=8192 loses 2.5% with t=−27.5 — even at 71% of the
   gate's N threshold, cp=2 is catastrophic.
3. **The K threshold is also right-on.** 8B Down (N=4096 K=14336) loses
   2.75%; even with K nearly 2× threshold, cp=2 hurts because N is small.

So the WIN is jointly N≥28672 AND K≥8192 — **and** the WIN does not
extend even to nearby (N=20480, K=8192) or (N=8192, K=28672). The gate
is tight on both axes.

Per-shape lookup table (consolidated, including R28+R29 data):

| (N, K)         | Layout | cp=0 ratio (R28/R29) | cp=2 effect    | Recommended cp |
|----------------|--------|----------------------|----------------|----------------|
| (1024, 8192)   | crr    | 0.86 (KV)            | −6.9% (R27)    | 0              |
| (4096, 14336)  | crr    | 0.94 (8B Down)       | −2.75% (R29B)  | 0              |
| (8192, 8192)   | crr    | 0.93 (8k³)           | −0.4% (R27)    | 0              |
| (8192, 14336)  | crr    | (synth)              | −3.34% (R29B)  | 0              |
| (8192, 28672)  | crr    | 0.85 (70B Down)      | −0.11% (R29B)  | 0 (no benefit) |
| (14336, 4096)  | crr    | 0.92 (8B Gate)       | −4.2% (R27)    | 0              |
| (14336, 8192)  | crr    | (synth)              | −0.98% (R29B)  | 0              |
| (20480, 8192)  | crr    | (synth)              | −2.53% (R29B)  | 0              |
| (28672, 8192)  | crr    | 0.84 (70B Gate)      | **+2.61% (R28)**| **2**         |
| (4096, 4096)   | rcr    | 0.92 (4k³)           | −96% (R27)     | 0              |

Only (28672, 8192) wins. The current `(N>=28672) AND (K>=8192)` gate
exactly matches this single WIN cell and excludes all LOSE/neutral
cells. **No refinement needed.**

Step 4 — No validation needed (no gate change)
-----------------------------------------------

Because the recommended action is "keep current gate", there is no new
binary to validate. The R28 cherry-pick already validated cells A,B,B0,C,D,E.

Files in this commit
--------------------
- `r29b_bench.py`              — preheat + 5x bench harness (GPU1 variant)
- `r29b_orchestrate.sh`        — build + bench cells {DOWN70, DOWN8, N14336,
                                  N20480, K14336} × {cp=0, cp=2}
- `r29b_cell*_cp{0,2}.txt`     — raw per-cell logs (10 files)
- `r29b_findings.md`           — this document

No source code changes. No edit to
`crr_mxfp8_exact_8wave_fastpath.inc`.

Hardware caveats
----------------
- HIP_VISIBLE_DEVICES=1 used throughout. sclk telemetry shows the device
  ramped to ~1.8 GHz under preheat for most cells, but the 8B Down cp=0
  and cp=2 runs both showed visible RUN-0 cold-ramp (e.g. cp=0 RUN 0
  2644 vs RUN 4 2804). Both cp=0 and cp=2 of 8B Down ran with similar
  cold-ramp profiles, so the −2.5% median delta is real (not a sclk
  artifact).
- The 70B Down result (cp=0=2528.67, cp=2=2525.78) sits firmly in the
  noise floor (|Δ|=2.88, σ_pooled≈4.8); cannot reject H0.
