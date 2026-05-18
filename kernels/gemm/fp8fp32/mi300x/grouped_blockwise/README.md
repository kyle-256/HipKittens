# Grouped FP8 Blockwise GEMM — MI300X

DeepSeek-V3-style 1×128 / 128×128 blockwise FP8 grouped GEMM for MI300X (gfx942).
Single launch, persistent kernel. Covers fwd, dgrad (routed through fwd via
caller B-transpose), wgrad (CRR/TN via K-contig col-T inputs).

Per group `g ∈ [0, G)`:
```
C[offs[g]:offs[g+1], :] = A[offs[g]:offs[g+1], :] @ B[g, :, :].T
```

## API

### fwd (`grouped_blockwise.cpp` → `dispatch_grouped`)

| Tensor | Shape | Dtype |
|---|---|---|
| `A` | `[M_total, K]` | fp8 e4m3fnuz |
| `B` | `[G, N, K]` | fp8 e4m3fnuz |
| `C` | `[M_total, N]` | bf16 |
| `A_scale_T` | `[Kb, M_total]` | fp32 |
| `B_scale_T` | `[G, Kb, Nb]` | fp32 |
| `group_offs` | `[G+1]` | int32, prefix sum of group lengths |
| `cum_tiles` | `[G+1]` | int32, prefix sum of `(M_g/BM) * (N/BN)` |

### wgrad (`grouped_wgrad.cpp` → `dispatch_grouped_wgrad`)

CRR/TN via caller-side K-contig column-transposed inputs. Per-element b_scale.
Reads `A=[N, M_total]`, `B=[K, M_total]`, writes `C=[G, N, K]`.

## Persistent kernel

```
launch grid = dim3(304)               # MI300X CU count
each WG:
    for tile_id in stride(blockIdx.x, total_tiles, 304):
        tile_id = chiplet_transform_chunked(tile_id, ..., CHIPLET_CHUNK)
        # map tile_id → (group_idx, pid_m, pid_n) via group_offs/cum_tiles
        # run the same 8-cluster pipeline as the per-tensor kernel
```

## Constraints

- `N % 128 == 0`, `K % 128 == 0`
- `M_g % 256 == 0` per group (BLOCK_M); `test_python.py` provides a
  Python-side padding wrapper for arbitrary `M_g` via `BW_UNALIGNED=1`
- `G + 1 ≤ 64`

## Build & run

```bash
make            # tk_kernel.so (fwd)
make wgrad      # tk_wgrad.so

BW_G=4 BW_MPG=2048 BW_N=4096 BW_K=4096 python3 test_python.py
BW_G=4 BW_MPG=2048                       python3 test_dgrad.py
BW_G=4 BW_MPG=2048                       python3 test_wgrad.py
```

## Files

| File | Role |
|---|---|
| `grouped_blockwise.cpp` | fwd kernel |
| `grouped_wgrad.cpp` | wgrad kernel |
| `Makefile` | Build (`make`, `make wgrad`, `make tuned BLOCK_M=…`, `make wgrad-tuned BLOCK_M=…`) |
| `test_python.py` / `test_dgrad.py` / `test_wgrad.py` | Correctness (SNR ≥ 48 dB) + perf bench |
