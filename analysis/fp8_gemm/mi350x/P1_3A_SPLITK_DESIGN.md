# P1.3(a) Split-K Cross-Group B Share Design (R66)

## Motivation

Per `[[fp8-rrr-attempt-h14]]`: worst-shape grouped @ B=16 must stream 544 MB B vs dense 364 MB → 50% data volume diff = ~25% TFLOPS gap. **Physics-bound, kernel-internal levers cannot close.**

Split-K cross-group B share lets multiple groups reuse B tile reads from L2, reducing total B-stream bytes. This is **the only algorithmic lever** that can close the remaining 12pp gap to 1.15× Triton on grouped RCR fwd.

## Current Dispatch Model

```
gt (global tile ID) → group_idx (which expert) + br/bc (tile within group)
B [G, N, K] indexed by [group_idx, bc*BLK_N + n_warp_offset, k_iter * BLK_K]
```

Each WG processes its assigned tiles serially, reads B independently per group.

## Split-K Plan

### Concept

For shapes with small N (BPC = N/BLK_N small) but many groups, the dispatch can:

1. **K-partition** each tile into `sk_split_n` chunks along K
2. Each WG computes partial accumulation for K-chunk i ∈ [0, sk_split_n-1]
3. **Cross-group B share**: WGs in same XCD/CU group reuse B chunk via L2 keep
4. Post-kernel reduce: sum K-chunk partials → final tile output

### Infrastructure Available

PT adapter already passes `sk_split_n` and `sk_partial_buf` fields (see `hk_grouped_gemm_gfx950.cu`). v1 dispatcher has hipMallocAsync stub at line 3741-3749 but **kernel internal split-K logic does not exist**.

### Sessions

#### Session 1: Dispatcher param plumbing (~50 LOC)
- v2 dispatcher accepts sk_split_n > 0
- Allocates partial buffer [num_tiles, sk_split_n, BLK_M, BLK_N] fp32
- Launches kernel with sk_split_n × num_tiles total grid

#### Session 2: K-partition kernel body (~200 LOC)
- Modify grouped_rcr_kernel_body_pinned to accept sk_split_n
- WG index decomposes (gt, sk_idx) — sk_idx selects K-chunk
- For sk_split_n=1, behavior identical to current (no-op safety)
- For sk_split_n>1, K-iter range is [sk_idx * ki/sk, (sk_idx+1) * ki/sk]
- Store partial to sk_partial_buf, NOT direct to C
- Skip mul(scale) + store_c_tile epilog (deferred to reduce step)

#### Session 3: Reduce kernel (~100 LOC)
- New kernel `grouped_rcr_sk_reduce_v2` reads sk_partial_buf, sums partials, applies scale, stores to C
- Launches grid = num_tiles (one WG per tile)

#### Session 4: Cross-group B share heuristic (~150 LOC)
- Tile dispatch order ensures same XCD's WGs hit same B tile sequentially (L2 stays warm)
- May need new `chiplet_transform_chunked_sk` swizzle pattern

#### Session 5: Bench + autotune integration (~100 LOC)
- 24-shape bench, find shapes where sk_split_n ∈ {2, 4} gains
- Autotune integration: select sk_split_n per shape

## Expected Win

- Worst-shape (qwen_down B=16 M=2048) v2/Triton 0.937 → ~1.05-1.10 if B reuse works
- Mid-shape (gpt_oss_up B=4) ratio 1.120 → likely no change (already above target)
- Geomean v2/Triton 1.029 → ~1.10-1.15 (close to or hitting target)

## Risk

- **Workspace allocator**: hipMallocAsync per-call cost. May need pool/cache (but `[[no-cache]]` forbids result cache). Workspace tensor passed from PT is acceptable.
- **Reduce kernel overhead**: extra kernel launch + extra global write of partial. If partials are large, overhead > B-share win.
- **Correctness**: numerical equivalence: fp32 partial sum vs single accumulate — should be bit-equivalent for fp32.
- **Cross-XCD**: L2 is per-CCD. Split-K across CCD doesn't help (different L2). Need to confine sk_partition within CCD boundary.

## Estimated Effort

5 sessions × 4-6 hours each = ~25-30 hours total

## Order Dependency

P1.2 spill=0 should land first — split-K kernel inherits same body. If body still spills, perf gain may be masked.
