# Round 3 — FP8 grouped K-tail path B (direct HBM→register) shipped

## TL;DR

- Round-2 path A (LDS-staged K-tail in main kernel epilog) saturated at
  SNR ≈ 20.7 dB on K=2880 gpt_oss shapes, below the 25 dB FP8
  correctness gate, blocking the metric at score 153.
  ([round-3-fp8-ktail-path-a-saturation.md](./round-3-fp8-ktail-path-a-saturation.md))
- Round-3 ships path B: each lane reads 2 × `buffer_load_b128` directly
  from HBM into `A_row_reg` / `B_row_reg` `data[]`, sidestepping LDS
  entirely and the round-3 phantom-read pattern.
- Result: all 16 FP8 grouped cases pass the 25 dB SNR gate. FP8
  geomean 0.099 → 0.917. **metric 153 → 465** (+312 = +204%).

## Lane → cell mapping for `rt_16x128_s` (FP8 `A_row_reg` / `B_row_reg`)

The FP8 register tile is `rt<fp8e4m3, RBM/RBN, BK=128, row_l, rt_16x128_s>`
where `rt_16x128 = rt_shape<16, 128, stride=16>`, giving:

```
num_elements          = 16 × 128                        = 2048
elements_per_thread   = 2048 / 64                       = 32 fp8 cells/lane
num_packed (fp8e4m3_4)= 4                                = 4 cells/pack
packed_per_thread     = 32 / 4                           = 8 packs/lane
data[8] of fp8e4m3_4  = 8 × 4 bytes                     = 32 bytes/lane
                                                        = 2 × b128
```

Per base tile (16 rows × 128 cols), 64 lanes split as 4 K-strides of 16
rows each:

```
laneid 0..15 : row = 0..15 , k_byte = 0   (K=[0,   32))
laneid 16..31: row = 0..15 , k_byte = 32  (K=[32,  64))
laneid 32..47: row = 0..15 , k_byte = 64  (K=[64,  96))
laneid 48..63: row = 0..15 , k_byte = 96  (K=[96, 128))
```

Within a lane's 32 K-cells, the layout of `data[8]` is K-contiguous:

```
data[0..3] = 16 fp8 cells = K=[k_byte,      k_byte + 16)  → b128 #1
data[4..7] = 16 fp8 cells = K=[k_byte + 16, k_byte + 32)  → b128 #2
```

Equivalent C++:

```cpp
const int row_lane    = laneid % 16;
const int k_lane_byte = (laneid / 16) * 32;

*reinterpret_cast<__uint128_t*>(&reg.tiles[h][0].data[0]) = b128_lo;
*reinterpret_cast<__uint128_t*>(&reg.tiles[h][0].data[4]) = b128_hi;
```

## K-tail K_REM=64 lane mask

For `K_REM = K_global - fast_k = 64` (gpt_oss K=2880=22*128+64):

```
laneid 0..15  : k_lane_byte=0  → both b128 in [0,32) ⊆ [0,64) → VALID
laneid 16..31 : k_lane_byte=32 → both b128 in [32,64) ⊆ [0,64) → VALID
laneid 32..47 : k_lane_byte=64 → both b128 in [64,96) ⊄ [0,64) → K-OOB
laneid 48..63 : k_lane_byte=96 → both b128 in [96,128) ⊄ [0,64) → K-OOB
```

K-OOB lanes get `voffset = SENTINEL = 0xFFFF0000u`; the SRD
`range_bytes` check rejects the load → VGPR returns 0. This relies on
documented `raw_buffer_load_b128` behaviour: zero-fill VGPR on OOB
voffset (in contrast to `raw_buffer_load_lds` which is no-op — the
exact issue that defeated path A).

The dispatcher already gates `FUSED_KTAIL=true` on `K_REM ∈ {32, 64,
96}` (32-aligned — clean per-lane mask). Mixed K_REM (e.g. 80, with
group 2 having 16 valid K cells + 16 invalid) requires partial-b128
mask; deferred to a future round if a workload demands it.

## SRD strategy

Two SRDs are constructed per K-tail invocation:

| Tile | Bound                                             | Why                                                                                                                                |
| ---- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| A    | `M_total × K` bytes (full-tensor)                 | A is `[M_total, K]` (concatenated groups along M). OOB byte > range clamps to 0 in VGPR.                                           |
| B    | `(group_idx + 1) × N × K` bytes (per-group bound) | B is `[G, N, K]`. Without per-group bound, OOB N-rows on partial last col-tile would wrap into NEXT group's data → silent garbage. |

`make_srsrc(ptr, bytes, /*row_stride_bytes=*/0)` keeps the linear range
check (no cache-swizzle), mirroring BF16 round-11's gpt_oss K=2880 fix
(non-power-of-2 row strides have UB OOB-clamp under cache swizzle).

## Address derivation (RCR)

For warp `(wm, wn)` at tile coord `(br, bc)` with `m_subtile_A =
m_start_g / HB`:

```
A row (slab s ∈ {0, 1}, base tile h ∈ [0, A_row_reg::height = 4)):
  M_warp_base = (m_subtile_A + br*2 + s) * HB + wm * RBM   (= +128 +64 stride)
  A_row_idx   = M_warp_base + h * 16 + row_lane

B row (n_strip ∈ {0, 1}, base tile h_b ∈ [0, B_row_reg::height = 2)):
  N_warp_base        = (bc*2 + n_strip) * HB + wn * RBN     (= +128 +32 stride)
  B_row_idx_in_group = N_warp_base + h_b * 16 + row_lane

A byte offset = A_row_idx * a_row_stride_bytes + K_tail_base_bytes + k_lane_byte
B byte offset = group_idx * N * K + B_row_idx_in_group * b_row_stride_bytes
              + K_tail_base_bytes + k_lane_byte
```

Constants: `HB=128, RBM=64, RBN=32, BK=128, A_row_reg::height=4,
B_row_reg::height=2, A_row_reg::width=B_row_reg::width=1`.

## Code shape (24 buffer_load_b128 per warp per K-tail)

```
M slab 0:
  load_a_kt(0)        : 4 * 2 = 8  buffer_load_b128   →  a.tiles[0..3][0]
  load_b_kt(b0, 0)    : 2 * 2 = 4  buffer_load_b128   → b0.tiles[0..1][0]
  load_b_kt(b1, 1)    : 2 * 2 = 4  buffer_load_b128   → b1.tiles[0..1][0]
  s_waitcnt vmcnt(0)
  rcr_mma(cA, a, b0); rcr_mma(cB, a, b1)

M slab 1:  (B0/B1 unchanged, share K-tail across slabs)
  load_a_kt(1)        : 4 * 2 = 8  buffer_load_b128   →  a.tiles[0..3][0]
  s_waitcnt vmcnt(0)
  rcr_mma(cC, a, b0); rcr_mma(cD, a, b1)
```

No `s_barrier`, no `s_setprio`, no LDS access. Path B is purely
register-only after the SRD construction.

## Register pressure (compiler -Rpass-analysis=kernel-resource-usage)

```
grouped_rcr_kernel<KI=0, N_MASKED=false, FUSED=true>
  TotalSGPRs: 76, VGPRs: 256, AGPRs: 0
  ScratchSize/lane: 292 bytes, VGPR Spill: 72 dwords
  Occupancy: 2 waves/SIMD

grouped_rcr_kernel<KI=0, N_MASKED=true,  FUSED=true>
  TotalSGPRs: 80, VGPRs: 256, AGPRs: 0
  ScratchSize/lane: 332 bytes, VGPR Spill: 82 dwords
  Occupancy: 2 waves/SIMD
```

For comparison the FUSED=false (no path B / no K-tail fuse) variants
spill 91/83 dwords — path B is **not** a register-pressure regression
relative to the legacy split-kernel + RMW pipeline.

## Metric impact

| Section                | round-2 ratio | round-3 ratio | Δ        |
| ---------------------- | ------------- | ------------- | -------- |
| FP8 DSV3 (8 cases)     | ~0.95         | 0.93–1.05     | similar  |
| FP8 gpt_oss (8 cases)  | 0.000 (FAIL)  | 0.82–0.88     | +0.85    |
| FP8 grouped geomean    | 0.099         | 0.917         | +0.818   |
| BF16 grouped geomean   | 0.340         | 0.340         | unchanged |
| **Score**              | **153**       | **465**       | **+312** |

All 8 FP8 gpt_oss `fwd-snr<X>` failures eliminated. The 4 remaining
correctness failures are BF16 gpt_oss-Down `dA` (different code path,
not addressed in this round).

## What's left for future rounds

1. **BF16 gpt_oss-Down `dA` correctness** (P0). Same K%128≠0 family,
   different kernel. dA uses `grouped_rrr_kernel` for backward path;
   needs its own path-B equivalent or alignment fix.
2. **FP8 gpt_oss perf headroom** (0.82–0.88 currently, target ≥1.20).
   Path B is correct but Triton still wins by ~15%. Likely K-tail
   SGPR/VGPR scheduling on top of path B; defer once BF16 dA is fixed.
3. **FP8 gpt_oss-Down `dB`** (P2). var-K backward kernel
   (`grouped_var_k_kernel_fp8`) has its own K%128 path; mirror path B
   approach there.
