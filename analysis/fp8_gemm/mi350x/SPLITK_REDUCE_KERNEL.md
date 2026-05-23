# Split-K Reduce Kernel Design (R88)

## Purpose

Per `P1_3A_SPLITK_DESIGN.md`, split-K kernel writes fp32 partials to workspace. A separate reduce kernel sums partials and writes bf16 output to C.

## Signature

```cpp
__global__ void grouped_rcr_sk_reduce_v2(
    const float* __restrict__ partials,   // [num_tiles, sk_split_n, BLK_M, BLK_N]
    const int*   __restrict__ group_offs, // tile→group lookup
    const float* sa, const float* sb,     // scale a, scale b
    bf16* __restrict__ C,                 // [M_total, N] output
    int sk_split_n, int num_tiles_total, int N);
```

## Algorithm

Per WG processes 1 output tile:
1. Read all sk_split_n partials for assigned tile from workspace
2. Sum into fp32 accumulator
3. Resolve combined_scale from group_idx
4. Convert to bf16, store to C

## Thread Layout

- BLK_M × BLK_N = 256×256 = 65536 cells per tile
- 512 threads/WG (match main kernel WG)
- Each thread handles 65536/512 = 128 cells (vectorized as float4 = 32 float4 chunks)
- Read pattern: contiguous along partials[tile, sk_split_n, m, n] — sk_split_n stride drives cache locality
- Store pattern: cell→(r, c) of output C tile, mapped to actual (M_total, N) coord via group_offs

## Workspace Sizing

```
workspace_bytes = num_tiles_total × sk_split_n × BLK_M × BLK_N × sizeof(float)
                = num_tiles × 2 × 256 × 256 × 4 (if sk_split_n=2)
                = num_tiles × 524288 bytes
```

For qwen_down (worst shape) 2048 tiles × sk_split_n=2 = 1 GB workspace. 大!  

Mitigation:
- Workspace passed from PT side (caller allocates + manages lifetime)
- Or use streaming: only keep sk_split_n partials per WG, reduce immediately after compute (not pre-allocate full grid)

## Streaming Variant (Recommended)

Each WG:
1. Computes K-chunk 0 partial → stores fp32 to LDS
2. Computes K-chunk 1 partial → adds in LDS
3. ...
4. Final reduce + scale + bf16 cast + store to C

No workspace needed; reduce in LDS. But LDS budget: 256×256×4 = 256 KB per tile, > CU LDS 160 KB.

Compromise: streaming over BLK_N subtile. Process N=64 at a time → LDS = 256×64×4 = 64 KB ≤ 160 KB. Outer loop over 4 N-subtiles.

## Per-K-chunk WG Mapping

If sk_split_n=2 and num_tiles=128 (one group's tiles), launch 2 × 128 = 256 WGs:
- WG (tile, k_chunk=0): computes K[0, ki/2]
- WG (tile, k_chunk=1): computes K[ki/2, ki]

Sync via atomic_add into LDS-shared accumulator? No, LDS not shared cross-WG. Either:
- Workspace + reduce kernel (described above)
- atomicAdd to global C (slow but no workspace)

## Decision

Go with **workspace + reduce kernel** for correctness simplicity. Workspace cost (~1 GB worst case) acceptable for B=16 shapes; for B=1 workspace negligible.

## Correctness

Partial sum = sum_{k_chunk} partials[..., k_chunk, ...]. Same as single-pass sum, just split. fp32 partials avoid precision loss vs early bf16 cast. Bit-exact if sum order preserved (need deterministic reduce order).
