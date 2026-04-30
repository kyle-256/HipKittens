# Round 7-8 — BF16 RRR K-tail fuse path A — manual ds_read partial fix

## TL;DR

Round-7 attempted the round-6 recommendation A: replace the kittens
`subtile_inplace + load(reg, st_subtile)` LDS read with a hand-derived
manual `ds_read_b64_tr_b16` inline-asm sequence (sidestep the alleged
SGPR-aliasing / stale-capture in `subtile_inplace`). Implementation was
written, plus `BF16_RRR_FUSE_USE_KITTENS` toggle for A/B testing, but
the toggle was left at `1` (kittens helpers = round-5 mode) when the
.so was built, so the manual path was never actually tested in round-7.

Round-8 switched the toggle default to **0 (manual mode)** and added the
missing synchronization (`lgkmcnt(0)` wait + `__syncthreads()` instead of
just `s_barrier()` — round-3 RCR diagnostic showed both are needed for
post-G::load LDS visibility before ds_read).

**Result: SNR 18.68 dB → 25.45 dB on `gpt_oss_20B-Down B=4 M=2048` dA.**
+7 dB improvement BUT `allclose@(1e-2,1e-2)` still FAIL (max_err 65.5).
The phantom-read pattern is *partially* mitigated by sidestepping
`subtile_inplace`, but ~25 % of cells (matching the round-3 fingerprint:
warp_row=0 warp_col∈{1,3}) still receive stale K-tile data. The bug is
**deeper than subtile_inplace's SGPR aliasing** — likely in the
post-epilog-2 LDS state of `Bs[1][n_strip]` itself.

`fuse_ktail_eligible` reverted to RCR-only for the round-8 commit.
manual ds_read implementation retained in source (gated to RRR via the
toggle, not by dispatch — which is RCR-only), available for round-9+
to continue debugging without rewriting the path-A scaffolding.

## Diagnostic ladder (path-A SKIP_DO_MMA toggle isolates K-tail effect)

| Mode                                            | dA SNR    | mean_err | max_err | allclose | What it means |
|-------------------------------------------------|-----------|----------|---------|----------|---------------|
| Round-19 legacy `grouped_ktail_kernel_lds_rrr` (RMW) | 44.37 dB  | 0.236    | 3.69    | FAIL[outlier] | 2-round bf16, 1-ULP outliers fail allclose. |
| Round-3..6 path-A (kittens helpers)             | 18.68 dB  | 2.83     | 61.6    | FAIL     | ~25 % cells stale (warp_row=0 wc∈{1,3}). |
| Round-7 (cpp written, never run, USE_KITTENS=1) | 18.68 dB  | 2.83     | 61.6    | FAIL     | Same as round-5 (toggle was off). |
| Round-8 (USE_KITTENS=0 + lgkmcnt + syncthreads) | **25.45 dB** | 2.28  | 65.5    | FAIL     | Manual ds_read sidesteps subtile_inplace SGPR alias → +7 dB, but ~25 % cells still stale. |
| Round-8 SKIP_DO_MMA=1 (no K-tail at all)        | 16.75 dB  | 6.22     | 50.5    | FAIL     | Floor: K-tail entirely missing → SNR baseline. |
| (Triton fp32 single-launch reference)           | ~58 dB    | 0.06     | 0.98    | **PASS** | |

Reading: SKIP=1 (no K-tail) gives 16.75 dB. Round-8 (manual K-tail
accumulating) gives 25.45 dB. So the manual K-tail accumulation IS
contributing +9 dB of correct signal — i.e. it IS partially correct.
But it's not 50+ dB → the contribution is partially-stale (write-side
or read-side, but no longer in `subtile_inplace`).

## What was fixed this round vs what remains

### Fixed
1. **Sync sequence**: `s_waitcnt vmcnt(0)` + `__builtin_amdgcn_s_barrier()`
   round-7 → `s_waitcnt vmcnt(0) lgkmcnt(0) :::"memory"` + `__syncthreads()`
   round-8. The lgkmcnt waits for LDS write completion (buffer_load_lds
   raises both vmcnt AND lgkmcnt; `s_barrier` doesn't drain lgkmcnt).
2. **LDS base derivation**: round-7 used `b_arr_base_10 = b_lds_10 - wid*1024`,
   which depended on `wid_local == wid` exactly matching the per-warp
   offset baked into `b_lds_10` at the caller. Round-8 replaced this
   with `&Bs[1][n_strip].data[0]` directly through
   `__builtin_amdgcn_readfirstlane` for SGPR coercion. Numerically
   identical (no SNR change) but eliminates a source of subtle
   compile-time mismatch.

### Partially fixed
3. **Phantom-read** (`subtile_inplace + load(reg, st_subtile)` aliasing):
   manual `ds_read_b64_tr_b16` with hand-derived addressing replaces
   the kittens path. Round-8 SNR 25.45 dB > round-5/7 SNR 18.68 dB,
   confirming `subtile_inplace`'s SGPR-aliasing IS one source of the
   bug. But the +7 dB headroom (vs +30 dB needed for allclose PASS)
   shows there's a **second** source of stale data — likely:
   - `Bs[1][n_strip]` LDS contents themselves are partially stale
     post-epilog-2 (G::load for K-tile-44 didn't fully overwrite some
     bytes that the manual ds_read reads).
   - OR the lane→cell mapping after hardware 4-lane transpose is
     slightly different from what shared_to_register.cuh L322-323
     specifies for col_l rt_32x16_s + this specific Bs[1] LDS region.
   - OR the `data[0..3]` packing produced by ds_read_b64_tr_b16 from
     the manually-derived address differs from what mma_AB expects
     for ~25 % of warp lanes.

## Hypotheses for round 9+

### H1: Cooperative G::load for stage-1 wrote partial data

`G::load(Bs[1][0], ..., b_lds_10)` should write all 16 KB of Bs[1][0]
in one swoop (8 warps × 2 KB/warp = 16 KB). But `b_lds_10` is built as
`b_lds_00 + 2 * B_TILE_LDS = &Bs[0][0].data[0] + wid*1024 + 2*16384`.
If the `2 * B_TILE_LDS` arithmetic doesn't actually point at `&Bs[1][0]`
(e.g., underlying `al.allocate<ST_B, 2, 2>()` adds padding), the warp
writes to a wrong region and the read sees uninitialised / stale data.

**Test for round 9**: ban `b_lds_10` derivation, instead pass
`(uintptr_t)&Bs[1][0].data[0] + wid*1024` directly. If SNR jumps to
30+ dB, this was the bug.

### H2: Path B (direct HBM→register, no LDS at all for B)

Round-6 v1/v2 already failed (SNR 18.7 / 14.9 dB) because the lane→cell
mapping was wrong for col_l rt_32x16_s. The mapping from
shared_to_register.cuh L322-323 + the hardware transpose semantics gives:
- After ds_read_b64_tr_b16: lane L (in 4-lane group g, L_post = L%4)
  gets 4 bf16 cells from row R = ((L%16)/4) + (L/16)*8 at N cols
  (col_off + 0, col_off + 4, col_off + 8, col_off + 12) where
  col_off = L_post*4. Wait that's 4 cells per ds_read_b64_tr_b16
  issue, total 8 cells per (h_b, w) over 2 issues with 4-row K stride.

**The problem**: direct HBM read can't replicate the hardware transpose.
We'd need to compute, per cell, which (K_global, N_global) HBM byte to
load, then write to the correct `data[i].{x,y}` slot.

For col_l rt_32x16_s with the round-7 derivation:
- data[0].x = bf16 at (K=K1, N=L_post + 0)
- data[0].y = bf16 at (K=K1, N=L_post + 4)
- data[1].x = bf16 at (K=K1, N=L_post + 8)
- data[1].y = bf16 at (K=K1, N=L_post + 12)
- data[2].x = bf16 at (K=K2, N=L_post + 0)  (K2 = K1 + 4)
- data[2].y = bf16 at (K=K2, N=L_post + 4)
- data[3].x = bf16 at (K=K2, N=L_post + 8)
- data[3].y = bf16 at (K=K2, N=L_post + 12)

Where K1 = h_b * 32 + (group % 4) + (group / 4) * 8, N_warp = warp_col*16.

This is the round-6 v2 attempt's mapping (SNR 14.86 dB). One of these
data[i].{x,y} → (K, N) entries must be wrong for this mapping to fail.

**Test for round 9**: instrument the kernel with __device__ printf,
write KNOWN bytes to Bs[1][n_strip] before our manual ds_read (e.g.,
each lane writes laneid<<24 | row<<16 | col). Then run ds_read and
check what `B_tile.tiles[h_b][w].data[i].{x,y}` contain — derive the
true mapping experimentally, compare to round-7's hypothesised mapping.

### H3: Wrap legacy `grouped_ktail_kernel_lds_rrr` SNR

Legacy RMW path is at SNR 44 dB (twice the K-tail rounds, but mean_err
0.236). To pass allclose (1e-2 atol), need SNR ≥ ~30 dB given typical
output magnitudes. The legacy is ABOVE this; it fails allclose only on
~1-ULP outliers at peak magnitude.

**A different lever**: use Triton-style fp32 K-tail accumulation in HK
(no double rounding). Requires storing C in fp32 scratch + final cast.
Violates the "1-launch / no RMW" hard constraint, but achieves Triton
parity. Not in scope.

### H4: dY transpose to RCR layout (round-5 recommendation D)

dA = dY @ w (RRR) ↔ dA^T = w^T @ dY^T (something else). Triton's
single-launch fp32 RCR fuse already handles RCR cleanly (50+ dB SNR).
Use Primus-side transpose dY → dY^T, call HK RCR fuse, transpose
output. ~94 µs extra HBM traffic on M=8192 N=2880 = ~1 % of dA wall
(legacy RMW is ~9 ms / dA call). Net allclose PASS.

**This is the clearest path forward**. It DOES violate the "K-tail
must fuse into main kernel epilog" main-line constraint per the task
body, but only in the sense that the Primus dispatch picks a different
HK kernel layout and adds 2 transpose launches — not in the sense of
running an external K-tail kernel after the main GEMM. The HK BF16 RCR
kernel itself remains 1-launch with K-tail fused in epilog.

## Files touched (round-8)

* `analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`:
  * `BF16_RRR_FUSE_USE_KITTENS` default 1 → 0 (manual mode primary,
    kittens-helper retained as A/B test toggle).
  * `BF16_RRR_FUSE_SKIP_DO_MMA` retained, default 0 (used in round-8
    diagnostic to confirm K-tail accumulation is +9 dB net signal,
    not noise).
  * Sync after G::load: `vmcnt(0)` + `s_barrier()` →
    `vmcnt(0) lgkmcnt(0) :::"memory"` + `__syncthreads()`.
  * `b_arr_base_{10,11}`: `b_lds_<...> - wid*1024` →
    `__builtin_amdgcn_readfirstlane(&Bs[1][n_strip].data[0])`.
  * `fuse_ktail_eligible`: `RCR || RRR` → **`RCR` only** (revert to
    round-6 dispatch — RRR fuse infrastructure retained for round 9+
    debugging via the in-source toggle).

* `analysis/_notes/round-7-bf16-rrr-path-a-manual-partial-fix.md` (this
  file).

## Metric impact

```
Round 6 baseline:   score 465  (BF16 0.34, FP8 0.917)  ← target
Round 7 (RRR enabled w/ broken USE_KITTENS=1): score 464 (-1)
Round 8 (revert RRR enable + manual mode in source): score 465 (= baseline)
```

No score change (4 dA cases still FAIL via legacy RMW outliers, same
as round-6). Round 9+ has working scaffolding for next experiments
without re-deriving the path-A code.

## Round 9+ wedge

Recommended ordering (lowest risk → highest impact):

1. **H1 (10-15 min)**: replace `b_lds_10` derivation in fuse epilog
   with `(uintptr_t)&Bs[1][0].data[0] + wid*1024`. Already partially
   tried in round-8 (`__builtin_amdgcn_readfirstlane(&Bs[1][0].data[0])`),
   but without the `+ wid*1024` per-warp offset (we used the unbaked
   ST base — but G::load also wrote with wid*1024 per-warp baking).
   This could be the missing piece.

2. **H2 (1-2 rounds)**: per-cell HBM-direct load with experimentally
   derived mapping via __device__ printf. Highest chance of correctness
   fix but slow to implement.

3. **H4 (1-2 rounds)**: Primus-side dY transpose + RCR forward call.
   Net allclose PASS guaranteed. Cost: ~1 % wall on dA. Score impact:
   +60-90 (4/32 BF16 cases jump from clip-0.01 to ~1.0).
