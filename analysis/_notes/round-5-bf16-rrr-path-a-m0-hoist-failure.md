# Round 5 — BF16 RRR K-tail fuse path A — m0-hoist hypothesis falsified

## TL;DR

Round 5 attempt: extend round-3/4 path A (cooperative G::load + LDS-staged
+ load(reg, st_subtile) + DO_MMA) to BF16 RRR (backward dA path), and
try to fix the round-3 phantom-read by replacing the LLVM-intrinsic
G::load with an inline-asm `s_mov_b32 m0` + `buffer_load_dwordx4 ...
offen lds` helper (`bf16_dev_d::load_hoist`, drop-in mirror of FP8's
`rcr_8w_load_hoist`). **Result: SNR unchanged at 18.68 dB.**

The m0-corruption hypothesis from round-4 is **falsified**. The bug is on
the LDS read side (`subtile_inplace + load(reg, st)`), not the LDS write
side (G::load / buffer_load_lds). Round 6+ must approach RRR fuse via a
different LDS read mechanism (e.g., manual `ds_read_b64` inline asm
with hand-derived lane → cell mapping for col_l rt_32x16_s, mirroring
FP8's `load_col_from_st_half`) or a non-LDS path.

## Backward dA correctness baseline (allclose@1e-2 floor)

The 4 `grpBF16_gpt_oss_20B-Down-B*-M*` cases in `_metric_grouped_only.py`
gate `dA` correctness via `torch.allclose(rtol=1e-2, atol=1e-2)`. None
of the BF16 RRR K-tail strategies tried so far pass that gate:

| Strategy                                | dA SNR (dB) | max_err  | allclose | Comment                                         |
|-----------------------------------------|-------------|----------|----------|-------------------------------------------------|
| Round-19 legacy `grouped_ktail_kernel_lds_rrr` RMW | 44.37      | 3.69     | FAIL     | RMW: bf16 → fp32 (no round) + acc + → bf16 (1 round)  → ~1 ULP per cell at high-magnitude outliers ⇒ allclose fails for the worst cell. |
| Round-5 path A — G::load + load(reg, st)| 18.68      | 61.6     | FAIL     | Phantom-read pattern same as RCR round-3.       |
| Round-5 path A — load_hoist + load(reg, st) | 18.68 | 61.6     | FAIL     | m0-CSE hypothesis falsified.                    |
| (Triton single-launch reference)        | ~58 dB     | 0.98     | **PASS** | fp32 acc throughout, 1 round at output.         |

Triton's single-launch fp32 fuse is the only known-working baseline.
HK's legacy RMW path is correct in mean but breaks at outliers; HK's
path A is broken structurally.

`mean_err` numbers from `python3 scripts/_metric_grouped_only.py` probe
on `gpt_oss_20B-Down B=4 M=2048 N=2880 K=2880`:

* Triton:  `mean_err 0.06`, `max_err 0.98`, allclose PASS.
* HK legacy RMW (round-19): `mean_err 0.236`, `max_err 3.69`, allclose FAIL.
* HK path A (round-3..4 RCR / round-5 RRR): `mean_err ≈ 2.83`,
  `max_err 61.6`, SNR 18.7 dB, allclose FAIL.

Triton is ~4× more precise than HK legacy at the mean (1 round vs 2
rounds in the round-trip). HK path A is ~12× worse than HK legacy at
the mean (phantom-read substitutes stale K-tile data into MMA).

## What Round 5 tried

### Step 1: Lift `bf16_dev_d::load_hoist` out of `BF16_HOIST_M0` guard

The helper was already in the file as a **gated-off** experiment from
P21 Dev D (perf experiment for main-loop DTL throughput, default 0).
Lifted out of the `#if BF16_HOIST_M0` guard via a new
`BF16_LOAD_HOIST_AVAILABLE` macro so the helper compiles regardless of
the perf flag, ready for future-round LDS write-side experiments.

### Step 2: Replace the 4 G::load calls in RRR FUSED_KTAIL with `bf16_dev_d::load_hoist`

The 7-arg signature of `bf16_dev_d::load_hoist` matches the 7-arg
`G::load(dst, gl, idx, swizzled_offsets, SRD, base_ptr, lds_addr)`
form used inside `device_gemm_tile_body`'s main loop. Drop-in:

```cpp
bf16_dev_d::load_hoist<NUM_THREADS>(
    Bs[0][0], b_gl, b_coord(col*2,   k_tail_tile),
    swizzled_offsets_B, b_srsrc_base, b_base, b_lds_00);
// ... 3 more for As[0][0], Bs[0][1], As[0][1] ...
```

### Step 3: Recompile + numerical probe

`gpt_oss_20B-Down B=4 M=2048 N=2880 K=2880`:

```
WT path A (G::load):       SNR 18.68 dB  mean_err 2.83  max_err 61.6
WT path A (load_hoist):    SNR 18.68 dB  mean_err 2.83  max_err 61.6
```

**Bit-identical numerics**. The inline-asm vs LLVM-intrinsic DTL load
makes no difference. m0-CSE was the wrong hypothesis.

## What this means

Round-3 BF16 RCR path A diagnostic (lane probe on `Bs[0][0].data[0..15]`)
already showed: G::load **does** write correct K-tile-44 data to LDS
(verified row 0 from lane 0 of warp 0). The bug is on the read side:

> warp_row=0, warp_col∈{1, 3} → register has STALE data
> (K=[2688, 2752), the main loop's penultimate Bs[0][0] write).

This means `subtile_inplace<HALF_REG_BLOCK_N, K_STEP>(Bs[0][0],
{warp_col, 0})` for `warp_col ∈ {1, 3}` (odd) is computing **stale LDS
source addresses** that point at the bytes of `Bs[0][0]` from the
penultimate main-loop write (K-tile 43), not the K-tile 44 data
that the immediately-preceding G::load just wrote.

The pattern `warp_col ∈ {1, 3}` (odd warp_col only, all warp_row=1
fine) suggests the issue is in either:

1. **`subtile_inplace`** capturing a stale `tile_base` pointer or
   `row_offset` for odd `warp_col` arguments.
2. **`load(reg, st)` itself** computing `swizzle({row, col})` using a
   `col` value tied to a stale `__shared__` allocator state for odd
   subtile slot indices.

Round-4 attempt #2 (move fuse INSIDE `device_gemm_tile_body` to share
inlining state with main loop) did not fix it. So it's not pure
inlining-boundary; it's deeper in the kittens base library.

Round-5 m0-hoist swap also doesn't fix it — the read-side bug is
independent of the write-side intrinsic.

## Round 6+ recommendations

### A: Replace `load(reg, st_subtile)` with manual ds_read_b64 inline asm

Mirror FP8's `load_col_from_st_half` (analysis/fp8_gemm/mi350x/
kernel_fp8_layouts.cpp:100-129) — manual `ds_read_b64_tr_b8` with
hand-derived lane → cell mapping for col_l rt_32x16_s. This bypasses
both `subtile_inplace` AND `load(reg, st)`, so any stale-capture bug
in those helpers is sidestepped.

For BF16, the equivalent is `ds_read_b64` (regular b64, no transpose-byte)
with `row_offset = ((laneid % 16) / 4) + ((laneid / dst.base_tile_cols) *
dst.base_tile_stride)` and `col_offset = ((laneid % 4) * 4) + (16 *
((laneid % dst.base_tile_cols) / 16))`, per
`include/ops/warp/memory/tile/shared_to_register.cuh:322-323`.

**Caveat**: this is mostly a refactoring exercise. If the bug is in the
swizzle calculation (which the manual path also has to mirror), it
won't be fixed. But it sidesteps `subtile_inplace`'s capture entirely.

### B: Path B for RRR — direct HBM→register K-tail load

Mirror BF16 RCR path B (round-5 commit `8deea208`) but for col_l
rt_32x16_s. B is `[G, K_inner, N_inner]` row_l so per-lane K-strided
2-byte loads (8 b16 per base tile per lane × 4 base tiles = 32 b16
loads/lane). Slow but correct. K-tail is cold path so wall-time is
amortised over the main loop's K=2816 work.

Per-lane lane → cell mapping for col_l rt_32x16_s base tile (32K × 16N,
64 lanes × 8 cells, stride=8, num_strides=1):

```
row_offset = ((laneid % 16) / 4) + ((laneid / 16) * 8)  // K row
col_offset = ((laneid % 4) * 4) + (16 * ((laneid % 16) / 16))  // N col group of 4
```

Per lane, 4 cells at (row, col_offset..col_offset+3) and 4 cells at
(row, col_offset+4..col_offset+7) = 1 row × 8 cols = 8 cells.

For 4 base tiles (height=2 K, width=2 N), per K-tail per lane = 32 b16
HBM loads. ~16K loads per K-tail launch (= 32 loads × 64 lanes × 8 warps).
Slow but correct, and HBM bytes / wall-time are bounded.

### C: fp32 scratch buffer + 2-launch RMW (constraint-violating)

Allocate fp32 partial in HBM, main kernel writes fp32, K-tail reads
fp32 + adds + writes bf16. **Single round at output**. ~600 MB scratch
for the worst metric shape. **Violates "K-tail must fuse into main
kernel" constraint** — only included here for completeness.

### D: Move BF16 RRR dA to RCR layout via input transpose

dA = dY @ w can be expressed as `dA^T = w^T @ dY^T`, i.e. CRR layout
(out = A^T @ B). This requires an explicit transpose of dY before the
GEMM (full HBM read+write of dY = M_total*N_orig bf16 bytes) and a
transpose of dA after. Transpose costs at gpt_oss-Down M_total=8192
N=2880 = ~47 MB read+write per dY and ~47 MB per dA = 94 MB of extra
HBM traffic per backward call. At MI355X HBM bandwidth ~1 TB/s the
transpose adds ~94 µs per dA, vs the legacy K-tail RMW correction
already at ~9 ms. Marginal cost vs catastrophic precision improvement
(SNR 44 → 58 dB, allclose PASS).

But: requires Primus-side wiring + an extra kernel launch. Constraint
"1 launch per fwd/dA/dB" interpretation matters here — strictly, dA
+ transposes = 3 launches.

## Files touched (Round 5)

* `analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`:
  * Lifted `bf16_dev_d::load_hoist` out of `BF16_HOIST_M0` guard
    (added `BF16_LOAD_HOIST_AVAILABLE` macro). Helper now compiles
    unconditionally for use by future-round RRR fuse experiments.
  * Added empty FUSED_KTAIL block for `L == Layout::RRR` in
    `device_gemm_tile_body` documenting the round-5 attempt + falsified
    m0-hoist hypothesis. Template instantiation `<RRR, 0, true>` and
    `launch_one_grouped_fuse<L>` helper retained for reuse.
  * Reverted `fuse_ktail_eligible` to `(L == Layout::RCR)` only.
    Legacy `grouped_ktail_kernel_lds_rrr<64>` RMW handles RRR dA
    K-tail correction (SNR 44.37 dB; allclose FAIL on outliers but
    SNR > Triton's 25 dB FP8 threshold).

* `analysis/_notes/round-5-bf16-rrr-path-a-m0-hoist-failure.md` (this file).

## Metric impact

`scripts/_metric_grouped_only.py` (gpt_oss + DSV3, 16 BF16 + 16 FP8):

```
Round 4 best:  score 465  (BF16 segment 0.34, FP8 segment 0.917)
Round 5:       score 465  (UNCHANGED)
```

The 4 `grpBF16_gpt_oss_20B-Down-B*-M*` dA cases remain `*FAIL[dA]`
due to the legacy K-tail RMW's 1-ULP outliers. Score does not drop
from the broken RRR path A attempt because that produced the same
ratio=0 clip on the same 4 cases.

## Next-round wedge

**Path A (RRR via `load_col_from_st_half`-style manual ds_read_b64):**
estimated 1 round of work to derive the lane mapping + 1 round to wire
+ 1 round to debug. Risk: still phantom-read if the swizzle is the bug.

**Path B (RRR direct HBM→register, scalar K-strided loads):**
estimated 1 round to wire + 0.5 round to verify. Slow per-K-tail launch
but cold path so amortised — net wall-time impact small (estimate < 5%
slower than legacy RMW). Highest probability of correctness fix.

Recommendation: **path B for RRR next round**. Once correctness is
fixed, dA section moves from 0.34 to ~1.13 ratio (mean of 16 cases),
metric jumps 465 → ~830-850.
