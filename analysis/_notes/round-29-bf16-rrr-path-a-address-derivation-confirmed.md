# Round 29 — BF16 RRR K-tail fuse path A: address derivation NOT the bug

## TL;DR

Round-7 docs hypothesised (H1) that `warp_col * 2048` + `h_b * 8 + w` in the
manual `ds_read_b64_tr_b16` epilog was wrong (per-warp offset and per-h_b
stride both 2× too small). Round 29 systematically tested this hypothesis
with a `BF16_RRR_FUSE_PROBE` build flag + numerical probe
(`/tmp/probe_bf16_rrr_round29.py`) on `M=2048 N=2880 K=2880 G=1 RRR`:

| Variant                                  | SNR (dB) | allclose | Notes                                |
|------------------------------------------|----------|----------|--------------------------------------|
| Production .so (legacy `grouped_ktail_kernel_lds_rrr`) | **44.34**    | **PASS**     | RMW path; reference baseline         |
| K=2816 PROBE (no K-tail)                | 49.58    | PASS     | Main kernel correctness verified     |
| Original code (`warp_col*2048 + h_b*8`) USE_KITTENS=0 | 19.59    | FAIL     | Manual ds_read baseline              |
| Original code USE_KITTENS=1 (kittens helpers) | 19.59 | FAIL     | **Identical to manual** — confirms address derivation matches kittens |
| Attempt: `warp_col*1024 + h_b*8`        | 14.86    | FAIL     | Off-by-half on warp offset           |
| Attempt: `warp_col*2048 + h_b*16`       | NaN      | FAIL     | h_b stride OOB                       |
| Attempt: `warp_col*1024 + h_b*16`       | -671 dB  | FAIL     | Both wrong → astronomical garbage    |

## What was rederived this round

Round-7 H1 was based on the assumption that `subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[1][n_strip], {0, warp_col})` advances `data` by `warp_col * 1024` bytes (one underlying subtile width). **This was wrong** because `HALF_REG_BLOCK_N = REG_BLOCK_N / 2 = (BLOCK_SIZE / WARPS_N) / 2 = (256 / 4) / 2 = 32`, NOT 16.

Correct derivation:

* `ST_B = st_bf<K_STEP=64, HALF_BLOCK_SIZE=128, st_32x16_s>` → 16 KB total
* `underlying_subtile_rows = 32`, `underlying_subtile_cols = 16`
* `underlying_subtiles_per_row = 128 / 16 = 8` ← matches `h_b * 8 + w`
* `subtile_inplace<K_STEP=64, HALF_REG_BLOCK_N=32>(Bs, {0, warp_col})`:
  - `subtile rows = 64`, `subtile cols = 32` (NOT 16)
  - `row_offset = 0`, `col_offset = warp_col * 32`
  - `subtile_col_offset_in_underlying = (warp_col * 32) / 16 = warp_col * 2`
  - `subtile_id = 0 * 8 + warp_col * 2 = warp_col * 2`
  - `subtile_offset = warp_col * 2 * (32 * 16) = warp_col * 1024 cells = warp_col * 2048 bytes` ✓

So `warp_col * 2048` IS the correct per-warp offset. `h_b * 8 + w` is also
correct (matches `ii * underlying_subtiles_per_row + jj` from
`shared_to_register.cuh:353`). The existing round-7/8 manual code is
mathematically equivalent to the kittens `load(reg, st_subtile<col_l>)` —
verified empirically: USE_KITTENS=0 (manual) and USE_KITTENS=1 (kittens
helper) both give SNR 19.59 dB on the probe shape.

## What this means for the BF16 RRR fuse main line

The 19-25 dB SNR ceiling on path A is NOT caused by an address derivation
bug. The remaining bug is in one of the three other places round-7 H2
flagged:

1. **Cross-warp LDS visibility** — `vmcnt(0) lgkmcnt(0)` + `__syncthreads()`
   may still leave a race between the cooperative `G::load(Bs[1][n_strip])`
   ds_writes and the per-warp `ds_read_b64_tr_b16` immediately after.
   Round-8 docs claim adding `lgkmcnt(0)` lifted SNR 18.68 → 25.45, but
   round-29 PROBE shows 19.59 dB with the lgkmcnt sync already in place.
   The +7 dB round-8 claim may have been on a different shape/seed; on
   M=2048 the manual code reaches the kittens-helper baseline.

2. **Lane mapping after HW 4-lane transpose** — `ds_read_b64_tr_b16` does
   a hardware 4-lane group transpose; the resulting per-lane payload may
   not match what `mma_AB(c, A_tile, B_tile)` expects in
   `B_tile.tiles[h_b][w].data[0..3]` for `col_l rt_32x16_s`. The mapping
   in round-7 H2 (`(K1, N=L_post + {0,4,8,12})`, `(K2 = K1+4, N=...)`) is
   plausible but never empirically verified via instrumented printf.

3. **G::load destination correctness** — `G::load(Bs[1][n_strip], ...,
   b_lds_10/11)` writes 16 KB into Bs[1][n_strip] using the `b_lds_10`
   address (which has `wid * 1024` baked in for cooperative writing). If
   the WRITE coverage doesn't fully tile Bs[1][n_strip], some bytes stay
   stale from epilog 2's previous tile-43 K-data.

## Round 29 deliverables

1. **`BF16_RRR_FUSE_PROBE` macro** added at line 4137 of
   `kernel_bf16_dynamic.cpp`. Default 0 (production: RCR-only fuse).
   When `=1` (build flag `-DBF16_RRR_FUSE_PROBE=1`), RRR added to
   `fuse_ktail_eligible` for empirical testing in future rounds.

2. **`/tmp/probe_bf16_rrr_round29.py`** — reusable BF16 RRR fuse SNR
   probe; `USE_PROBE=0` loads production .so (baseline ~44 dB on K=2880),
   `USE_PROBE=1` loads `tk_bf16_layouts_probe.so` (path A SNR ~19.59 dB).
   Set `K` env var to test K%128 != 0 vs K%128 == 0 sanity.

3. **This document** — confirms round-7 H1 (address derivation) is NOT
   the bug. Path A manual code is mathematically correct. Bug is in
   sync / lane mapping / G::load destination (round-7 H2 / H3 hypotheses).

## Recommended next-round angle

Given path A architectural ceiling ~19-25 dB on this approach + 5
attempts on FP8 RRR path A all failed in round 28, and BF16 RRR address
derivation now confirmed correct, the production K-tail fuse main line
for **forward** is genuinely saturated. The remaining external launches
are all backward (BF16/FP8 RRR `grouped_ktail_kernel_lds_rrr`,
`grouped_ntail_kernel_lds_rrr`) — they don't affect the metric forward
TFLOPS but do affect bwd correctness.

The next-round agent should consider:

1. **Cleanup pivot**: instrument `Bs[1][n_strip]` post-`G::load` with
   a debug write pattern (each lane writes laneid<<24 | row<<8 | col)
   to verify cooperative WRITE coverage. Compare against expected
   layout; identify if any bytes stay stale (round-7 H3 hypothesis).
   If yes → revert `b_lds_10/11` derivation back to round-7 v0
   (`b_lds_10 - wid * 1024`) which was changed to readfirstlane in
   round-8 — the change may have left some bytes uncovered.

2. **Accept plateau on K-tail fuse** + invest the round into other
   structural cleanup (e.g., remove `H4 reroute` once BF16 RRR fuse
   passes allclose; remove `grouped_ntail_kernel_lds_rrr` once N-tail
   merge into RRR fuse; etc.). All cleanup, no metric impact, but
   advances user's "delete external K-tail kernels" main-line goal.

3. **Path B for B operand** (col_l direct HBM → register, no LDS):
   round-7 H2 sketched the lane→cell mapping but never implemented.
   Each lane needs 8 bf16 cells covering 2 K values × 4 N cols, not
   16 sequential bytes — so `raw_buffer_load_b128` doesn't directly
   apply. Path B for B requires per-cell `raw_buffer_load_b16` × 8
   per lane (or `b32` × 4 if (K,N+4) packs into 4 bytes — but
   K-stride is N bytes, so they're not adjacent). Slow but mappable.
   Estimate: ~8× slower than LDS-staged path A on a per-K-tile basis.
   Worth it only if path A truly cannot pass allclose.
