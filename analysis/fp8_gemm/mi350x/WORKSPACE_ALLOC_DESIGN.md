# Workspace Allocator Design (R89, split-K support)

## Constraint

`[[no-cache]]` user mandate forbids result caches in Primus-Turbo grouped GEMM. Workspace allocator must not cache between calls.

## Approach

PT-side caller allocates workspace tensor explicitly:
```python
ws = torch.empty((num_tiles_max * sk_split_n_max * BLK_M * BLK_N,), dtype=torch.float32, device='cuda')
out = primus_turbo.ops.hk_grouped_rcr_fp8_sk(af, bf, ..., workspace=ws, sk_split_n=2)
```

## Sizing

Use upper bound:
- num_tiles_max from `kittens::ceil_div(M_total, BLK_M) * kittens::ceil_div(N, BLK_N)`
- sk_split_n_max=4 (typical max useful split)
- Worst case: M_total=32768, N=4096 → num_tiles=2048; ws=2048×4×65536×4=2GB

For B=1 single group with N=2048: 8×4×65536×4=8MB

Caller can pre-allocate persistent ws (reuse across forward passes) — NOT a result cache, just a memory buffer.

## Alternative: hipMallocAsync inside dispatcher

Already stubbed in v1 dispatcher (line 3741). Cost ~5μs per call. For B=1 inference acceptable; for training step where qwen_down is hot, 5μs × thousand-calls = mS overhead.

Decision: caller-passed workspace (zero per-call alloc cost).

## API Sketch

```cpp
void hk_grouped_rcr_fp8_sk(
    const void* a_ptr, ..., 
    void* workspace_ptr,    // new
    int workspace_bytes,    // new — for OOB check
    int sk_split_n,         // new — 0=disabled, default
    ...);
```

If `sk_split_n=0` or `workspace_ptr=nullptr`: falls back to non-split-K path (current v2 dispatcher).

## Backwards Compatibility

Existing v2 entry `hk_grouped_rcr_fp8_new` unchanged. New entry `hk_grouped_rcr_fp8_sk` for split-K.

PT autotune selects between based on shape (heuristic: K < 4096 + B > 8 → try sk_split_n=2).

## Test Strategy

Unit test:
1. workspace=nullptr → bit-equal to non-SK path
2. workspace allocated + sk_split_n=2 → SNR ≥ 30 dB vs non-SK
3. workspace too small → return error (no segfault)

## Effort

~50 LOC for new API binding + dispatcher branch. Part of P1.3a Session 1.
