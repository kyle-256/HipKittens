---
name: bf16-gemm-optimization
description: BF16 GEMM optimization on AMD CDNA4 (gfx950/MI355X). Covers exact-dim JIT compilation, 8-wave ping-pong scheduling, layout-specific tuning for RCR/RRR/CRR, and comparison with hipBLASLt.
---
# BF16 GEMM Optimization Status

## Current Performance (GPU4, 2026-04-10)

### With Exact-Dim JIT + Swap M↔N + CRR unroll 2 (best available)

| Layout | Geo-mean vs hipBLASLt | Notes |
|---|---|---|
| **RCR** | **~1.003x** | Beats hipBLASLt ✓ |
| **RRR** | **~1.000x** | At parity ✓ |
| **CRR** | **>>1.0x (est)** | Full benchmark pending; spot checks: +7 to +28pp |

Spot-checked CRR results (unroll 2 vs hipBLASLt, no swap):
| Shape | Before | After |
|---|---|---|
| 4096×4096×14336 | ~0.924x | **1.207x** |
| 8192×8192×4096 | ~0.997x | **1.201x** |
| 8192×8192×8192 | ~0.997x | **1.064x** |
| 8192×28672×4096 (swap) | ~0.965x | **1.005x** |
| 4096×4096×4096 | ~0.997x | 0.989x (−0.8pp) |

Key optimizations applied:
1. **Swap M↔N when N > M**: Use kernel(bM=N,bN=M) to avoid tall-N inefficiency. +3-5% for non-square shapes.
2. **mma_AtB (CRR)**: Replaced register transpose + mma_AB with direct mma_AtB_base. Saves 10 VGPRs (226→216), eliminates register shuffle. +0.3-0.5pp.
3. **CRR #pragma unroll 2**: Previously blocked by SGPR spills. With mma_AtB (fewer SGPRs), 36 spills occur but reduced barrier count far outweighs cost. Large-K shapes see +7 to +28pp improvement.
4. **DO_MMA2 (superseded)**: Pre-transposing A once per pair was tried before mma_AtB. The mma_AtB approach is cleaner and slightly better.

### Previous (JIT only, no swap)

| Layout | Geo-mean vs hipBLASLt | Wins |
|---|---|---|
| **RCR** | **0.992x** | **~16/48** |
| **RRR** | **0.973x** | **~15/48** |
| **CRR** | **0.961x** | **~8/48** |

### Baseline (Dynamic 8-wave, no JIT)

| Layout | Geo-mean | Wins |
|---|---|---|
| RCR | 0.970x | 4/48 |
| RRR | 0.957x | 9/48 |
| CRR | 0.944x | 3/48 |

**JIT improvement: +1.5-2.2% geo-mean across all layouts.**

## Key Findings

### 1. Exact-Dim JIT is the primary optimization
Compiling per-shape with `-DM_DIM=M -DN_DIM=N -DK_DIM=K` gives:
- Compile-time loop bounds → full `#pragma unroll` (RCR/RRR)
- Compile-time SRD strides → constant folding
- 2-4 fewer VGPRs

### 2. Swap M↔N for large-N shapes
When N > M, compute C^T = B^T @ A^T (swap bM=N, bN=M) to avoid tall-N grid inefficiency:
- RCR: 0.992x → 1.003x (+1.1pp geo-mean)
- RRR: 0.973x → 1.000x (+2.7pp geo-mean)
- CRR: benefits too (test: 8192×28672×4096 swap is +2.8% vs no-swap)
Note: swap uses separate compiled kernel per (bM, bN, K).

### 3. CRR: mma_AtB eliminates register transpose
CRR layout (A=K×M stored transposed, B=K×N) needs A^T @ B.
Original approach: load A in col_l → register transpose → mma_AB (+register shuffle overhead).
Better: call `mma_AtB_base` directly with col_l A and col_l B. Same hardware instruction
(`mfma_f32_16x16x32_bf16`) but no register shuffle. Result: -10 VGPRs (226→216), +0.3-0.5pp.

**Critical detail:** The outer `mma_AtB` template requires `row_layout B`, which our B_reg_t
(col_l) doesn't match. Must use `mma_AtB_base` directly in a manual tile loop:
```cpp
for(int _n=0; _n<NH; _n++) for(int _m=0; _m<NW; _m++) {
    mma_AtB_base(D.tiles[_n][_m], A.tiles[0][_n], B.tiles[0][_m], C.tiles[_n][_m]);
    for(int _k=1; _k<KH; _k++)
        mma_AtB_base(D.tiles[_n][_m], A.tiles[_k][_n], B.tiles[_k][_m], D.tiles[_n][_m]);
}
```

### 4. CRR VGPR/SGPR analysis
VGPR counts (8192x8192x8192): RCR=244, RRR=220, CRR=216, all within 256 limit.
**#pragma unroll for CRR causes SGPR spills**: Even `#pragma unroll 2` jumps SGPR from 59→106
with 36 spills (loop constant folding pressure). CRR must stay without unroll.
RCR/RRR can use full `#pragma unroll` because K_DIM is a compile-time constant.

### 5. CRR unroll 2 resolves the barrier overhead
Previous CRR gap: barriers between warps (16 s_barrier per main_loop_iter × iterations).
`#pragma unroll 2` lets compiler see 4 K-tiles at once → schedule prefetches earlier → hides barrier latency.
- Was blocked by SGPR spills with old DO_MMA2. mma_AtB approach uses fewer SGPRs (59 vs 106).
- With mma_AtB, unroll 2 causes only 36 SGPR spills — minor cost vs benefit.
- Large-K, large-M shapes: 0.924x → 1.2x+ after unroll 2.

### 6. BF16 is memory-bound → 4-wave fails
4-wave (no ping-pong) loses 30-40% vs 8-wave. The 8-wave ping-pong is essential.

## Architecture

- **Block size:** 256×256 output, K_STEP=64
- **Warps:** 2×4 = 8 warps/block, 512 threads
- **Occupancy:** 2 waves/SIMD (≤256 VGPRs)
- **Double buffering:** tic/toc ping-pong, 2×2 smem tiles per operand
- **XCD swizzle:** chiplet_transform_chunked with 64-chunk
- **Group_m:** runtime autotune per shape ∈ {1,2,4,8,16}
- **CRR smem tile:** st_bf<K_STEP, HALF_BLOCK_SIZE, st_32x16_s> (K×M row of A)

## Key Files

| File | Description |
|---|---|
| `.jit_bf16_cache/kernel_exact.cpp` | Exact-dim JIT kernel (all 3 layouts) |
| `jit_bf16_gemm.py` | JIT compilation framework |
| `bench_bf16_jit.py` | Benchmark vs torch.mm (all 3 layouts, with swap) |

## Build & Test

```bash
cd /shared_nfs/kyle/HipKittens2/analysis/bf16_gemm/mi350x

# JIT compile specific shapes
python3 -c "
from jit_bf16_gemm import warmup_shapes
shapes = [(8192,8192,8192),(8192,28672,4096)]
warmup_shapes(shapes, verbose=True)
"

# Full benchmark (subprocess per shape, compiles swap kernels too)
HIP_VISIBLE_DEVICES=4 python3 bench_bf16_jit.py
```

## IMPORTANT: Cache Invalidation
kernel_exact.cpp is the single source for ALL compiled .so files in `.jit_bf16_cache/exact_*/`.
When kernel_exact.cpp changes, delete all cached .so:
```bash
rm -rf .jit_bf16_cache/exact_*/
```
Then re-run bench_bf16_jit.py (Phase 1 recompiles everything).

## Next Steps

1. **Run full 48-shape benchmark** — clear `.jit_bf16_cache/exact_*/` and run `bench_bf16_jit.py` to confirm CRR geo-mean
2. **FP8 optimization** — see fp8-gemm-optimization skill
