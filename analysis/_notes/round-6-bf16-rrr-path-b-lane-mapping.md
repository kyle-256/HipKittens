# Round 6 — BF16 RRR K-tail fuse path B — lane mapping derivation failed

## TL;DR

Round 6 attempt: implement path B (direct HBM→register K-tail load) for
**RRR** by mirroring round-5's RCR path B (commit `8deea208`) but with
the col_l rt_32x16_s register layout. Two lane → cell mapping
derivations were tried; both produced sub-20 dB SNR on the dA cases.
The `st_32x16` swizzle (XOR-bank-conflict pattern, `(offset%1024)>>9<<4`,
see `include/types/shared/st_shape.cuh:173-176`) breaks the simple
linear interpretation of `(row_offset, col_offset) + ds_read_b64_tr_b16`
that worked for FP8's b8-transpose path. **fuse_ktail_eligible is
reverted to RCR-only** for the round-6 commit; RRR dA stays on the
legacy `grouped_ktail_kernel_lds_rrr` (RMW, 44 dB SNR) for now.

## Backward dA correctness baseline (allclose@1e-2 floor)

| Strategy                                     | dA SNR (dB) | max_err  | allclose | Comment                                                             |
|----------------------------------------------|-------------|----------|----------|---------------------------------------------------------------------|
| Round-19 legacy `grouped_ktail_kernel_lds_rrr` (RMW) | 44.37  | 3.69     | FAIL     | Only failing on outliers (~1 ULP at peak magnitudes).               |
| Round-5 path A (G::load + load(reg, st))     | 18.68      | 61.6     | FAIL     | Phantom-read pattern (LDS read side stale).                          |
| Round-6 v1 path B — K-major bf16_2 packing   | 18.68      | 61.6     | FAIL     | Symmetric to v0 (8 K cells × 1 N col per lane).                      |
| Round-6 v2 path B — N-major bf16_2 packing   | 14.86      | 56.8     | FAIL     | (2 K) × (4 N) per lane, swapped pair direction. Worse.               |
| (Triton single-launch reference)             | ~58 dB     | 0.98     | **PASS** | fp32 acc throughout, 1 round at output.                              |

Triton's single-launch fp32 fuse remains the only known-passing baseline
for `gpt_oss_20B-Down B=*-M=* dA` cases (K=N=2880).

## What Round 6 tried

### Attempt v1: K-major bf16_2 packing in path B

Mirrored RCR path B's per-lane HBM byte-offset structure but for B
col_l rt_32x16_s with the **MMA-semantic** lane mapping:

```cpp
const int k_quad = laneid / 16;   // ∈ [0, 4) — K-octet
const int n_col  = laneid % 16;   // ∈ [0, 16) — N-col in base
// per (h_b, w) base tile: K = h_b*32 + k_quad*8 + (kk = 0..7), N = w*16 + n_col
// data[i] = (K_cell[2*i], K_cell[2*i+1])  ← K-axis adjacent in bf16_2
```

8 b32 loads/lane/(h_b, w) (high 16 bits discarded). Per K-tail per warp:
32 b32 loads/lane × 64 lanes × 8 warps = 16K b32 = 64 KB HBM. ~0.3 µs
wall per warp.

**Result**: SNR 18.68 dB on `gpt_oss_20B-Down B=4 M=2048`. Identical to
round-5 path A (G::load + load_b_subtile), suggesting the underlying
lane mapping is structurally wrong for col_l rt_32x16_s.

### Attempt v2: N-major bf16_2 packing (2 K × 4 N per lane)

Re-derived the lane mapping from `shared_to_register.cuh:322-323`'s
prologue + the 4-lane `ds_read_b64_tr_b16` 4×4 transpose semantics:

```cpp
const int L_post  = laneid % 4;            // post-transpose group index
const int group   = laneid / 4;            // ∈ [0, 16)
const int row_off = (group % 4) + (group / 4) * 8;  // K row
// Per (h_b, w): K1 = h_b*32 + row_off, K2 = K1 + 4
// 4 N cells: N = w*16 + L_post + p*4   for p ∈ [0, 4)
// data[k_iter*2 + p/2].(x|y) = cell at (K[k_iter], N[p])  ← N-major in bf16_2
```

Reasoning: the 2nd `ds_read_b64_tr_b16` issue at `offset + 4*row_bytes`
implies the second issue reads from K = row_off + 4 (4 K rows down).
Pre-transpose 4 lanes hold same K row × 4 N col groups; post-4×4
transpose redistributes to 4 lanes × (1 K row × 4 spread N cols).
Two issues × 4 cells = 8 cells per lane = 2 K × 4 N.

**Result**: SNR 14.86 dB, `max_err 56.8`. **Worse** than v1.

## Why both fail — `st_32x16` swizzle complication

The shared tile `st_32x16` BF16 swizzle (`include/types/shared/st_shape.cuh:152-183`):

```cpp
const int swizzle = ((offset % 1024) >> 9) << 4;
const int swizzled_offset = offset ^ swizzle;
```

For `offset = 2 * (r * 16 + c)`:
* For `r < 16`: `offset < 512`, `offset % 1024 < 512`, `>> 9` = 0,
  `<< 4` = 0. `swizzle = 0`. **Identity** — no permutation.
* For `r >= 16`: `offset >= 512`, `offset % 1024 ∈ [512, 1024)`,
  `>> 9` = 1, `<< 4` = 16. `swizzle = 16`. The 4-bit XOR with `c`
  swaps cols within each 16-col K row beyond row 16.

Half of the lane mapping's `row_offset` values (`{16..19, 24..27}`) hit
the non-identity swizzle. So the simple "lane reads (row_off, col_off)
through (row_off, col_off+15)" interpretation breaks for laneid ≥ 32.

Concretely for `laneid=32` (row_off=16, col_off=0):
* Pre-swizzle byte address `2 * (16 * 16 + 0) = 512`.
* Post-swizzle: `swizzle({16, 0}) = 528` (offset 512 ^ 16 = 528).
* `ds_read_b64` at byte 528 reads bytes `[528, 535]` = matrix cells
  at swizzle inverse mapping for those bytes.
* byte 528 → matrix `(16, 0)` (since `swizzle({16, 0}) = 528`).
* byte 530 → solving `swizzle({r, c}) = 530`: `offset ^ 16 = 530`,
  `offset = 546 = 2 * (17 * 16 + 1)`, so byte 530 → matrix `(17, 1)`.

So `ds_read_b64` from byte 528 reads matrix cells `(16,0), (17,1),
(18,2), (19,3)` — interleaved across 4 K rows AND 4 N cols, not 4
contiguous cells of any single axis. Path B's "scalar HBM load + scatter
to data[]" cannot mirror this without explicit swizzle-inversion.

The natural workaround is to NOT bypass the LDS read path: write the
K-tail data to LDS first (with the matching swizzle, via per-lane
`buffer_load_b128` + `ds_write_b16`), then use the existing
`load(reg, st)` helper which already handles swizzle correctly.

## Round 7+ recommendations

### A: Manual `ds_read_b64_tr_b16` inline-asm with hand-derived addresses

Re-emit the same `ds_read_b64_tr_b16` instruction sequence as
`load(reg, st_subtile)` but with addresses computed inline (bypassing
`subtile_inplace`'s alleged stale-capture for warp_col∈{1,3}). Mirrors
FP8's `load_col_from_st_half` strategy.

The instruction sequence, with hand-derived swizzle math:

```cpp
// Per lane mapping (matches shared_to_register.cuh:322-323)
const int row_off = ((laneid % 16) / 4) + ((laneid / 16) * 8);
const int col_off = (laneid % 4) * 4;

// LDS base for warp's B subtile (B Bs[stage_1][n_strip] for warp_col).
// Compute manually instead of subtile_inplace:
const uint32_t b_subtile_lds_base = b_lds_<stage_1><n_strip> +
    warp_col * (HALF_REG_BLOCK_N * sizeof(bf16));

// For each (h_b, w) base tile:
for (h_b = 0; h_b < 2; ++h_b) {
    for (w = 0; w < 2; ++w) {
        const int row = h_b * 32 + row_off;
        const int col = w * 16 + col_off;
        const uint32_t addr = b_subtile_lds_base +
            st_32x16::swizzle<bf16>({row, col});
        // 2 ds_read_b64_tr_b16 issues for 8 K cells per lane:
        asm volatile(
            "ds_read_b64_tr_b16 %0, %1 offset:0\n"
            "ds_read_b64_tr_b16 %2, %1 offset:%3\n"
            : "=v"(*reinterpret_cast<float2*>(&B_tile.tiles[h_b][w].data[0])),
              "=v"(*reinterpret_cast<float2*>(&B_tile.tiles[h_b][w].data[2]))
            : "v"(addr), "i"(4 * /*row_bytes for st_32x16*/ 32)
            : "memory"
        );
    }
}
```

This requires:
1. The K-tail data to be written to LDS at `b_lds_<stage_1>` slots
   (G::load works for this per round-3 lane probe).
2. Manual swizzle calc to mirror what `subtile_inplace` would compute.
3. No reliance on `load(reg, st_subtile)`.

### B: Per-warp dedicated LDS scratch + `load(reg, st)` (no subtile)

Allocate fresh per-warp ST tiles in LDS (not subtiles of a shared ST),
G::load directly to each warp's own ST, then `load(reg, st)` without
`subtile_inplace`. This eliminates the subtile capture stage entirely.
Cost: extra 8 × 4 KB = 32 KB LDS, but should fit (current LDS budget is
~544 bytes/block per resource-usage analysis).

### C: K-tail in fp32 main-kernel-internal scratch buffer

Allocate a small fp32 scratch in HBM, main kernel writes K=[0, fast_k)
in fp32 to scratch, K-tail kernel does fp32 RMW (no double-rounding),
then a small "scale + cast to bf16 + store" kernel finalizes. This
violates the "no RMW" hard constraint but would achieve Triton-level
precision (fp32 throughout). Documented for future reference, NOT
currently in scope.

## Backward bench (BF16, after round-6 revert)

`bench_grouped_gemm_turbo.py --dtype bf16` (HEAD post round-6 revert):

| TestID | Case               | M    | N    | K    | Check | Fwd TFLOPS | Bwd TFLOPS |
|-------:|--------------------|-----:|-----:|-----:|-------|-----------:|-----------:|
|      1 | DeepSeek-V3-GateUP | 2048 | 4096 | 7168 | PASS  |    1402.8  |     964.0  |
|      2 | DeepSeek-V3-Down   | 2048 | 7168 | 2048 | PASS  |    1234.0  |     982.9  |
|      3 | DeepSeek-V3-GateUP | 4096 | 4096 | 7168 | PASS  |    1433.1  |    1093.0  |
|      4 | DeepSeek-V3-Down   | 4096 | 7168 | 2048 | PASS  |    1256.2  |    1122.1  |
|     10 | gpt_oss-Down       | 2048 | 2880 | 2880 | FAIL[bwd_x] |  873.7  |     435.4  |
|     12 | gpt_oss-Down       | 4096 | 2880 | 2880 | FAIL[bwd_x] | 1014.8  |     525.1  |
|     14 | gpt_oss-Down       | 2048 | 2880 | 2880 | FAIL[bwd_x] | 1059.2  |     392.9  |
|     16 | gpt_oss-Down       | 4096 | 2880 | 2880 | FAIL[bwd_x] | 1032.5  |     419.0  |

**Avg Fwd TFLOPS: 1205.55**, **Avg Bwd TFLOPS: 758.21** — same as
round-5 (legacy RMW path for RRR dA).

## Score history

* Round 5 baseline: **466**
* Round 6 v1 (K-major path B): 463 (-3)
* Round 6 v2 (N-major path B): 464 (-2)
* Round 6 revert: **465** (≈ baseline ± 1 noise)

## Attempted but DEAD code retained for round 7+

The path B v2 (N-major) implementation is retained in
`kernel_bf16_dynamic.cpp` inside `device_gemm_tile_body`'s
`else if constexpr (L == Layout::RRR)` FUSED_KTAIL block. It is
**unreachable** because `fuse_ktail_eligible` gates `FUSED_KTAIL=true`
to RCR-only. Round 7 should either:

* Replace it with the recommendation A (manual ds_read_b64_tr_b16
  inline-asm with hand-derived swizzle).
* Replace it with B (per-warp dedicated LDS).
* Delete it as a dead end and pursue a fundamentally different
  approach.
