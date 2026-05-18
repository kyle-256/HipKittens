# FP8 Blockwise GEMM — MI300X

DeepSeek-V3-style 1×128 / 128×128 block-scaled FP8 GEMM for MI300X (gfx942).
Covers fwd (RCR/NT), dgrad (RRR/NN), wgrad (CRR/TN).

`C[M,N] = sum_ki (A[M,k] * B[N,k]^T) * a_scale[m, ki] * b_scale[n_block, ki]`
- A, B: FP8 e4m3fnuz row-major
- A_scale: `[Kb, M]` fp32 (caller passes `a_scale_inv.T.contiguous()`)
- B_scale: `[Kb, Nb]` fp32 (Nb = N/128)
- C: bf16

`dispatch_micro_dgrad` routes through the fwd kernel with caller-side B-transpose;
`dispatch_micro_wgrad` routes through the fwd kernel with K-contig col-T inputs +
per-element b_scale (`B_SCALE_PER_ELEMENT=true` template).

## Tile geometry (default; per-shape overrides via `make tuned`)

| Param | Value |
|---|---|
| BLOCK_M × BLOCK_N × BLOCK_K | 256 × 128 × 128 |
| Warps | 8 (2×4), each owns 128M × 32N |
| Per-warp accumulators | 2 × `rt_fl<64,32,col>` main + 4 × `rt_fl<32,32,col>` partial |
| Pipeline | 8-cluster software pipelining + register-buffered prefetch (PR #52 pattern) |
| LDS | 48KB (32K As + 16K Bs), 1 block/CU |
| Occupancy | 2 waves/SIMD (LDS-limited) |
| VGPR / spill | 236 / 0 |

## Build flags

| Flag | Default | Effect |
|---|---|---|
| `BLOCK_M/N/K`, `NUM_WARPS`, `REG_M_BUILD` | 256/128/128/8/64 | Tile geometry |
| `BW_RAW_DRAIN` | 0 | Replace `rt_fl` partials with raw `float2[]` drain buffers |
| `BW_PRESCALE_BS` | 0 | Fold `b_s` into `svt/svb` before cluster-6 drain |
| `BW_CHIPLET_CHUNK` | 1 | XCD round-robin chunk size for `chiplet_transform_chunked` |

## Per-shape JIT autotune

`test_python.py:TUNED_REGISTRY` maps each `(M, N, K, section)` to a per-shape
`(BM, BN, BK, NW, REG_M, RAW, PRE, CHK)` tuple. The loader builds the matching
`tk_kernel_BM<…>.so` on demand via `make tuned`. Manual override via
`BW_USE_TUNED=1` + env vars `BW_BLOCK_M=…` etc.

## Build & run

```bash
make
python3 test_python.py             # default 8192³, SNR ≥ 48 dB gate

# Other shapes / sections:
BW_M=16384 BW_N=4096 BW_K=7168 BW_SECTION=fwd python3 test_python.py
```

## Files

| File | Role |
|---|---|
| `blockwise.cpp` | Kernel source (fwd / dgrad / wgrad via templated micro_tk) |
| `Makefile` | Build (default + `tuned` per-shape target) |
| `test_python.py` | Correctness + perf harness + TUNED_REGISTRY |
