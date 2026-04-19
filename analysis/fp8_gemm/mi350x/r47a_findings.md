# R47 Dev A — RCR XCD-aware block swizzle (port from R46 Dev D)

## TL;DR

**SHIP.** Direct port of R46 Dev D's RRR XCD swizzle to the RCR exact 8-wave
fastpath kernel. **+8.46% on 70B Gate/Up** (the headline RCR shape that was
stuck at 90.5% of FP8) and **+2.90% on 70B Down**. Worst regression -0.58%
(8B Gate/Up — within -1% noise band). Default ON.

## What changed

`analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp`:
- Added `MXFP8_RCR_BLOCK_SWIZZLE` macro (default 1) plus `_NUM_XCDS=8` and
  `_GROUP_M=4` configuration knobs (mirrors RRR macro names exactly).
- Replaced the `bid → (br, bc)` row-major mapping in
  `rcr_exact_8wave_scaled_kernel` with the chiplet swizzle + grouped-M
  swizzle from R46 Dev D's `rrr_mxfp8_exact_8wave_fastpath.inc` lines 65-98.
  The swizzle is wrapped in `#if MXFP8_RCR_BLOCK_SWIZZLE` so the legacy
  mapping is preserved when the macro is 0.
- constexpr-promoted `blocks_per_row = M_DIM / BLK` (new) and
  `k_iters = K_DIM / BK` (was `g.k / BK`). The `can_use_exact_8wave_scaled`
  guard already requires `g.m == M_DIM && g.k == K_DIM`, so this is a no-op
  semantically but enables full unrolling of the integer math the swizzle
  introduces.

## Files

- `kernel_mxfp8_layouts.cpp` — kernel + macros (lines 2346-2467 region).
- `r47a_rcr_swizzle_sweep.sh` — initial sweep harness (template = R46).
- `r47a_swizzle_results/` — sweep logs + clean re-bench summary.

## Why default ON despite the original sweep saying "OFF"

The first run of `r47a_rcr_swizzle_sweep.sh` reported a -48% regression on
70B Gate/Up swizzle (1395 TF vs 2697 baseline) and net-negative across the
board. That cell turned out to be **SCLK-contaminated** — re-running the
identical build with cooldown between cells gave 2912 TF (+8.46%). Two other
cells (8B Q/O, 8B Gate/Up) also flipped sign or compressed delta in the
re-bench. The original sweep ran 14 builds back-to-back without cooldown on
GPU 0 and the GPU clock dropped substantially through the run.

This matches the R44/R45 SCLK-contamination pattern documented in earlier
cycles. The R46 Dev D RRR sweep used GPU 2 (likely cooler) and didn't see
the same artifact, but the lesson is: long sweeps need the 3-of-3 +
cooldown re-bench discipline before declaring a regression.

## Per-shape Δ% (clean medians)

| Shape           | Baseline | Swizzle | Δ%      |
|-----------------|---------:|--------:|--------:|
| 8192 cube       |   3056.3 |  3049.6 |  -0.22% |
| 8B Q/O          |   2322.9 |  2389.0 |  +2.85% |
| 8B Gate/Up      |   2548.5 |  2533.7 |  -0.58% |
| 8B Down         |   2946.3 |  2981.8 |  +1.21% |
| 70B Q/O         |   2920.1 |  2903.8 |  -0.56% |
| 70B Gate/Up     |   2685.1 |  2912.4 |  +8.46% ★|
| 70B Down        |   2910.8 |  2995.1 |  +2.90% ★|

Worst regression: -0.58%. Shapes ≥ +1%: 4. Net-positive — default ON.

## Correctness

- 8192³ RCR + swizzle: SNR 49.60 dB, det 3/3 PASS.
- 70B Gate/Up RCR + swizzle: SNR 49.59 dB, det 3/3 PASS.
- All 7 sweep shapes: PASS during quick smoke tests at the default 48 dB
  threshold. No change to numerics — the swizzle is purely a permutation of
  block-id ordering across the same compute and memory operations.

## Compatibility note

`MXFP8_RCR_V2_PERSISTENT` (default 0) early-exits when
`blockIdx.x >= total_tiles_compile`. The early-exit runs **before** the
swizzle remap, which is correct when persistent grid == total tiles but
unsafe when persistent grid > total tiles (the swizzle could remap a
surviving bid past `total_tiles`). Documented inline in the macro comment.
Persistent has been default OFF since R31 — no production impact.

## Verdict: **SHIP** (RCR XCD swizzle, default ON, +8.46% / +2.90% on the two
70B headline shapes, -0.58% worst).
