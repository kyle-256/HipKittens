# Round 16 — FP8 bwd post-H4 rocprof: H4 transpose is the new 21.6% bottleneck

## Context

Post round-14 H4 reroute (FP8 dA RRR → RCR via b transpose, gated on
`K_RRR % 128 != 0`): metric forward unchanged (833 ± 1 plateau across
rounds 11-16) but bwd improved +8.4 % avg (gpt_oss-Down +28..+138 %).

Round 16 task: identify the **NEW** wedge in bwd path post-H4.

## Rocprof bwd post-H4 (gpt_oss-Down B=4 M=2048, 13 train iters)

| % wall | µs/call | n calls | Kernel |
|---|---|---|---|
| **33.4 %** | 121 | 26 | `grouped_rcr_kernel` (fwd × 13 + dA via H4 reroute × 13) |
| **25.1 %** | 183 | 13 | `grouped_var_k_kernel_fp8` (dB var-K) |
| **21.6 %** | 78 | 26 | **`elementwise_kernel_manual_unroll` (PyTorch H4 transpose)** ← NEW |
| 7.0 % | 8.5 | 78 | `reduce_row_kernel` (FP8 amax) |
| 5.9 % | 14 | 39 | `unary_kernel` (FP8 scale) |
| 2.5 % | 18 | 13 | `reduce_kernel` |
| 1.8 % | 4.4 | 39 | `compute_scale_from_amax_kernel` |
| 0.6 % | 4.2 | 14 | `vectorized_elementwise_kernel` |
| 0.6 % | 4.2 | 13 | `__amd_rocclr_fillBufferAligned` |
| 0.5 % | 3.8 | 13 | `compute_group_offs_device` |

Pre-H4 (round 14 baseline) had:
* 16.8 % `grouped_ktail_kernel_lds_rrr`  (eliminated by H4)
* 11.8 % `grouped_ntail_kernel_lds_rrr`  (eliminated by H4)
*  7.6 % `grouped_tail_kernel<RRR>` scalar (eliminated by H4)
* sum 36.2 % external-launch overhead → eliminated.

H4 paid back via:
* Replaced `grouped_rrr_kernel` (11.6 % @ 103 µs) with extra
  `grouped_rcr_kernel` launch via reroute (~13 % more grouped_rcr_kernel
  time, 33.4 % - prev forward 13.8 % = ~19.6 % new dA via reroute).
* Added `elementwise_kernel_manual_unroll` (21.6 % @ 78 µs/call,
  2 calls/iter) for the b.transpose(-2,-1).contiguous() copy.

Net: -36.2 % external + (+19.6 % dA-via-reroute - 11.6 % old dA) +
21.6 % transpose ≈ -6.6 % wall, matching the +8.4 % bwd TFLOPS we saw.

## H4 transpose cost analysis

Per call `b.transpose(-2,-1).contiguous()`:
* Shape:   `[G=4, N_orig=2880, K_in=2880]` fp8 (33.2 MB / group)
* Total copied:  `4 × 33.2 MB = 132.8 MB` rd + same wr = 265.6 MB
* PyTorch elementwise wall: 78 µs/call × 2 calls/iter = 156 µs/iter
* Effective bandwidth: 265.6 MB / 78 µs ≈ **3.4 TB/s aggregate** (across
  HBM + LDS staging in PyTorch's strided gather kernel).
* MI355X HBM peak ≈ 5.3 TB/s. Effective ≈ 65 % peak. Already well-tuned.

Conclusion: replacing PyTorch's transpose kernel with a custom HK
transpose won't significantly improve. The 78 µs is near-peak HBM
bandwidth.

The ONLY way to recover the full 21.6 % of bwd wall is to **eliminate
the transpose entirely** — i.e., make grouped_rrr_kernel handle
K-misalignment natively in its main kernel epilog (no reroute, no
transpose).

## FP8 RRR fuse path B — the architectural fix

To eliminate H4 transpose, FP8 grouped_rrr_kernel must handle K%128 != 0
in-kernel via path B (direct HBM→register K-tail load), mirroring FP8
RCR fuse path B (round-3 commit 07354791).

### BF16 RRR fuse history (rounds 4-8, all FAILED)

Per `kernel_bf16_dynamic.cpp:4070-4112`:
* Round 4: path A (LDS-staged) — phantom-read on `subtile_inplace +
  load(reg, st)` after epilog 2.
* Round 5: path B (K-major bf16_2 packing) — sub-20 dB SNR.
* Round 6: path B (N-major bf16_2 packing) — sub-20 dB SNR.
  Diagnosed: `col_l rt_32x16_s` lane→cell mapping after
  `ds_read_b64_tr_b16` is NOT a simple "K_quad × N_col"; it's mediated
  by `st_32x16`'s XOR-bank-conflict swizzle (rows >= 16 have cols
  permuted by 16 within each 16-col block — see
  `types/shared/st_shape.cuh:173`).
* Round 7: path A hybrid (A direct + B LDS-staged + manual
  ds_read_b64_tr_b16 with hand-derived swizzle). SNR 18.68 dB
  (phantom-read warp_row=0 wc∈{1,3}).
* Round 8: SNR 18.68 → 25.45 dB but **allclose still FAIL** —
  ~25 % cells stale. Bug deeper than `subtile_inplace` SGPR
  aliasing; suspected in the `ST_B[1][n_strip] / Bs[1][n_strip]`
  post-epilog-2 LDS layout itself.

### FP8 RRR fuse path B — different setup, may succeed

FP8 grouped_rrr_kernel (line 2454-2700) uses:
* `A_row_reg = rt_fp8e4m3<RBM=64, BK=128, row_l, rt_16x128_s>` ← same
  as RCR
* `B_col_reg = rt_fp8e4m3<BK=128, RBN=32, col_l, rt_128x16_s>` ← uses
  `rt_128x16_s`, NOT `rt_32x16_s` (the BF16 RRR fuse blocker)
* B's shared tile: `ST_v2 = st_fp8e4m3<HB=128, RBN_swizzle, ...>` ←
  fp8 1-byte stride, may have different (or no) XOR-bank-swizzle than
  BF16's `st_32x16`

### Three FP8 RRR fuse path B variants to attempt

**A) rt_128x16_s direct register K-tail (mirror BF16 round-5/6)**

Each lane reads K=[fast_k, K_global) = 64 bytes from B (K-axis = rows
of `rt_128x16_s`) and from A (K-axis = cols of `rt_16x128_s`) directly
into the register tile. Then call `mma_AB(cA/cB/cC/cD, a, b)` to
accumulate K-tail.

Confirmed feasibility (`include/types/register/rt_shape.cuh:38`):
* `rt_128x16` = `rt_shape<128, 16, 16>` → `num_elements = 2048`
* `elements_per_thread = 2048 / 64 = 32`  (32 fp8 elems per lane)
* `num_strides = 32 / 16 = 2`              (2 strides per lane = 2 b128
  loads per lane, matching RCR path B's load pattern)

So K-tail load per lane is 2 × `raw_buffer_load_b128` (each loads 16 fp8
bytes). Identical pattern to RCR fuse path B (round-3) which we know
works numerically.

Lane → (K, N) mapping derivable from the existing `load(b, g.b, ...)`
function template in HK headers — no need to derive from scratch
(unlike BF16 RRR's manual `ds_read_b64_tr_b16` derivation, which is the
blocker for that path). Just call `kittens::load(b, g.b, b_co(...))`
with K-tail SRD bound = K_global, and the existing load helper handles
swizzle + lane mapping correctly + zero-fills VGPR on OOB lanes via
`raw_buffer_load_b128` (which is what RCR fuse path B uses).

**B) Cooperative LDS-staged direct via raw_buffer_load_b128 (mirror
BF16 round-7+8 hybrid)**

Each warp loads K-tail bytes from HBM via `raw_buffer_load_b128` (NOT
`raw_buffer_load_lds` — the latter is no-op on OOB), writes to LDS at
swizzled offsets, then reads via `load(reg, st_subtile)`. The
`raw_buffer_load_b128` zero-fills VGPR on OOB lanes, so we get
zero-padding in the reg before LDS write — equivalent to
auto-zero-fill in LDS for the subsequent mma read.

Risk: `subtile_inplace` after epilog 2 may have phantom-read pattern as
in BF16 round 7-8. The LDS scratch `Bs[1][n_strip]` post-epilog-2 may
have stale data depending on whether epilog 2's mma reads it for the
last K-iter.

**C) Run main loop one extra iter with K_REM bytes valid + zero-pad**

Increase `ki_dyn` by 1 so the last main-loop iter is the K-tail iter.
Use `raw_buffer_load_b128` (zero-fill on OOB) into LDS via `G::load` →
mma reads zero-padded LDS tile.

Risk: `G::load` uses `raw_buffer_load_lds` internally (round-2
diagnosis), which is NO-OP on OOB voffset — leaves stale main-loop
data. So variant C requires changing the load helper, which is a
deeper kernel-template change.

## Recommendation for round 17+

1. **Variant A (direct register K-tail)** is the lowest-risk attempt
   because:
   * No LDS phantom-read (BF16 round 4/7/8 blocker).
   * `raw_buffer_load_b128` zero-fills VGPR on OOB — exactly what we
     want for K-tail OOB lanes.
   * `rt_128x16_s` lane mapping is derivable from
     `include/types/register/tile.cuh` without depending on `st_v2`'s
     swizzle layout.
   * Only register pressure cost is 1 K-block of A + B reg tiles
     (~1 a tile + 1 b tile, ≈ 32 + 16 = 48 bytes/lane ≈ 12 VGPR).
2. Numerical probe: SNR + allclose against torch fp32 reference on
   gpt_oss-Down B=4 M=2048 (smallest gpt_oss FP8 case).
3. If variant A fails (SNR < 25 dB or allclose FAIL), document with
   round-17 note and try variant B.

## Other observations

* `grouped_var_k_kernel_fp8` is 25.1 % bwd wall (183 µs/iter, dB
  path). Has 256 VGPR / 67 spill / 139 KB LDS; Round 14 noted spill
  reduction approaches saturated per task body forbidden list. Not a
  near-term wedge.
* `grouped_rcr_kernel` is 33.4 % bwd wall (now serves both fwd and
  dA-via-H4-reroute). Round 12 rocprof identified its main K-loop as
  the dominant 50 µs/iter gap vs Triton on gpt_oss; round 15 probe
  ruled out `RCR_TWO_TILE_MIN_KI` tuning. Real wedge is MFMA cell
  shape rewrite (16x16x128 → 32x32x64), 1-2 round project.

## Score plateau status

Score remains 833 ± 1 across rounds 10-16. All forward-K-tail-fuse
main-line work is shipped (BF16 RCR + FP8 RCR path B; BF16 RRR + FP8
RRR via H4 reroute). Remaining wedges are kernel-template rewrites
(MFMA cell shape, RRR fuse path B) — multi-round projects with
documented failure modes from prior rounds. Round 16 commits the
above analysis to enable round-17 to make an informed decision on
which kernel-template rewrite to invest in.
