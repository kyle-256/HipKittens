# R46 Dev D: RRR XCD-aware block swizzle

## Summary

Added **XCD-aware chiplet swizzle + grouped-M swizzle** to the MXFP8 RRR
exact 8-wave fastpath kernel (`rrr_exact_8wave_scaled_kernel`).
Macro-gated `MXFP8_RRR_BLOCK_SWIZZLE` — **default ON** based on net-positive
sweep across all 7 compute-bound shapes.

## Motivation

R46 Dev A profiling identified **L2 scale cache pressure (0-3% of gap)** as
one of three sources of the MXFP8/FP8 gap, worst for large-N shapes like
70B Gate/Up (N=28672 → 7.3 MB B-scale working set across 1792 CTAs).

Default row-major block dispatch (`br = bid / blocks_per_col`) sends adjacent
WGs to different XCDs, causing each XCD to fetch the same B-tiles into its
local L2. XCD-aware swizzle interleaves WG dispatch so adjacent WGs land on
the same XCD and share L2-resident B-tiles.

## Implementation

`analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc` lines 13-24, 60-91:

1. **Chiplet swizzle** (8 XCDs on MI355X):
   ```
   bid' = (bid % 8) * (num_wgs / 8) + (bid / 8)
   ```
2. **Grouped-M swizzle** (group of 4 M-tiles share B-tiles):
   ```
   group_id = bid' / (4 * blocks_per_col)
   br = group_id*4 + (bid' % (4*blocks_per_col)) % group_size_m
   bc = (bid' % (4*blocks_per_col)) / group_size_m
   ```
3. **constexpr promotion** of `blocks_per_row/col`, `k_iters` from runtime
   division to compile-time M_DIM/N_DIM/K_DIM macros — enables full unroll.

## Results

Sweep across 7 compute-bound shapes (R46 task `b9veo5fpr`,
`r46_swizzle_results/`):

| Shape           | RRR baseline | RRR +Swizzle |     Δ%       |
|-----------------|-------------:|-------------:|-------------:|
| 8192³           |       2913.1 |       2905.8 |       -0.25% |
| 8B Q/O          |       2298.8 |       2331.7 |       +1.43% |
| 8B Gate/Up      |       2466.4 |       2448.0 |       -0.74% |
| 8B Down         |       2840.4 |       2872.9 |       +1.14% |
| 70B Q/O         |       2821.5 |       2826.9 |       +0.19% |
| **70B Gate/Up** |   **2610.9** |   **2781.0** |   **+6.51% ★** |
| **70B Down**    |   **2871.3** |   **2957.2** |   **+2.99% ★** |

**Net positive**: 4 shapes ≥ +1.0%, 2 shapes ≤ -0.5% (worst -0.74%, within
within-GPU variance envelope established R44 = ±2%).

## Correctness + Determinism

70B Gate/Up RRR with swizzle ON:
- SNR: 49.60 dB (≥ 45 dB SHIP gate) **PASS**
- Determinism: 3/3 runs byte-identical **PASS**

8192³ all-layouts with swizzle ON:
- All RRR/RCR/CRR: SNR 49.59-49.60 dB, det 3/3 PASS

## R46 Updated Baseline (perf + corr + det vs FP8)

After enabling swizzle by default and re-benching all 7 shapes × 3 layouts:

| Shape           | Layout | MX/FP8% | Status |
|-----------------|:------:|--------:|:------:|
| 8192³           |  RCR   |  95.2%  |  PASS  |
| 8192³           |  RRR   |  97.0%  |  PASS  |
| 8192³           |  CRR   |  94.3%  |  fail  |
| 8B Q/O          |  RCR   |  91.8%  |  fail  |
| 8B Q/O          |  RRR   |  90.1%  |  fail  |
| 8B Q/O          |  CRR   |  88.4%  |  fail  |
| 8B Gate/Up      |  RCR   |  93.1%  |  fail  |
| 8B Gate/Up      |  RRR   |  91.9%  |  fail  |
| 8B Gate/Up      |  CRR   |  91.4%  |  fail  |
| 8B Down         |  RCR   |  94.4%  |  fail  |
| 8B Down         |  RRR   | 105.3%  |  PASS  |
| 8B Down         |  CRR   |  94.6%  |  fail  |
| 70B Q/O         |  RCR   |  95.1%  |  PASS  |
| 70B Q/O         |  RRR   |  94.3%  |  fail  |
| 70B Q/O         |  CRR   |  94.9%  |  fail  |
| 70B Gate/Up     |  RCR   |  90.5%  |  fail  |
| 70B Gate/Up     |  RRR   |  94.3%  |  fail  |
| 70B Gate/Up     |  CRR   |  86.3%  |  fail  |
| 70B Down        |  RCR   |  90.5%  |  fail  |
| 70B Down        |  RRR   | 107.2%  |  PASS  |
| 70B Down        |  CRR   |  85.3%  |  fail  |

**5/21 PASS** (R45 baseline was 3/21 PASS) → +2 cells via XCD swizzle.

All 21/21 SNR ≥ 45 dB + det 3/3 PASS (correctness gate).

## Refuted attempts

**R46 Dev C: CRR k-pair main loop** — REFUTED.

Hypothesis: replace per-k `fixed_phase` template with k-pair loop using
`crr_mma_scaled_from_raw_packs` to eliminate 6 scale-shift ops per odd-k
iteration. Predicted gain: +3-7%.

Sweep across all 7 shapes (R46 `r46_kpair_results/`):

| Shape       | CRR base | CRR +KPAIR |     Δ%      |
|-------------|---------:|-----------:|------------:|
| 8192³       |   2703.1 |     2436.0 |   **-9.88%** |
| 8B Q/O      |   2116.2 |     1891.8 |  **-10.60%** |
| 8B Gate/Up  |   2290.2 |     2058.3 |  **-10.12%** |
| 8B Down     |   2644.2 |     2340.1 |  **-11.50%** |
| 70B Q/O     |   2613.7 |     2328.3 |  **-10.92%** |
| 70B Gate/Up |   2427.6 |     2238.1 |   **-7.81%** |
| 70B Down    |   2563.4 |     2395.5 |   **-6.55%** |

Universal **-7% to -11% regression**. Root cause likely the runtime `k_phase`
branch in `crr_mma_scaled_from_raw_packs` defeating the predictor + register
pressure from raw-packs path.

**Code discarded** — not committed. CRR remains on the R45 fixed_phase + scale
shift baseline.

## Files modified

- `analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc`:
  - Lines 13-24: 3 new macros (default ON for `MXFP8_RRR_BLOCK_SWIZZLE`)
  - Lines 43-47: constexpr `blocks_per_row/col`, `k_iters`
  - Lines 60-91: bid swizzle + grouped-M tile mapping

## Files added

- `analysis/fp8_gemm/mi350x/r46_swizzle_sweep.sh` — 7-shape RRR baseline-vs-swizzle
- `analysis/fp8_gemm/mi350x/r46_kpair_sweep.sh` — 7-shape CRR baseline-vs-kpair
- `analysis/fp8_gemm/mi350x/r46_swizzle_results/` — sweep logs
- `analysis/fp8_gemm/mi350x/r46_kpair_results/` — sweep logs (REFUTED data)
- `analysis/fp8_gemm/mi350x/r46_full_results_SUMMARY.txt` — full 21-cell baseline

## R46 cumulative levers update

R45 closed 45 levers cumulative. R46 adds:
- **+1 closed**: RRR XCD-aware block swizzle (Dev D, +6.5% on 70B Gate/Up)
- **+1 refuted**: CRR k-pair main loop (Dev C, -10% universal)

→ **R46 cumulative: 46 closed levers, 4 refutations** since R32.
