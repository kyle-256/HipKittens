---
name: bf16-gemm-optimization
description: BF16 GEMM optimization on AMD CDNA4 (gfx950/MI355X). Covers exact-dim JIT compilation, 8-wave ping-pong scheduling, layout-specific tuning for RCR/RRR/CRR, and comparison with hipBLASLt.
---
# BF16 GEMM Optimization Status

## Current Performance (GPU7, 48 LLM shapes, 2026-04-09)

### With Exact-Dim JIT (best available)

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
- Compile-time loop bounds → full `#pragma unroll`
- Compile-time SRD strides → constant folding
- 2-4 fewer VGPRs (238 vs 242 for RCR)

### 2. hipBLASLt's layout gap is worse than ours
| | RRR/RCR | CRR/RCR |
|---|---|---|
| hipBLASLt | 0.974x | 0.958x |
| **HipKittens** | **0.982x** | **0.970x** |

Our kernel handles transposed layouts more efficiently than hipBLASLt.

### 3. BF16 is memory-bound → 4-wave fails
Unlike FP8, BF16 has 2x lower arithmetic intensity. 4-wave (no ping-pong) loses 30-40% vs 8-wave. The 8-wave ping-pong scheduling is essential.

### 4. All HLL-level optimizations exhausted
Tried and measured zero effect:
- VMCNT sweep (4/6/8/10/12)
- Occupancy 3 (254 spills nullify occupancy gain)
- sched_barrier / setprio removal
- K_OVERRIDE only (vs full M/N/K JIT)
- Split-K with FP32 workspace reduce
- In-kernel atomic Split-K

### 5. Remaining gap is ISA-level
hipBLASLt uses 4-wave hand-tuned ISA + in-kernel Split-K (atomic tile reduction). This scheduling precision cannot be replicated from HIP C++.

## Architecture

- **Block size:** 256×256 output, K_STEP=64
- **Warps:** 2×4 = 8 warps/block, 512 threads
- **Occupancy:** 2 waves/SIMD (≤256 VGPRs)
- **Double buffering:** tic/toc with ping-pong scheduling
- **XCD swizzle:** chiplet_transform_chunked with 64-chunk
- **Group_m:** runtime autotune per shape ∈ {1,2,4,8,16}

## Key Files

| File | Description |
|---|---|
| `kernel_bf16_dynamic.cpp` | 8-wave dynamic kernel (RCR/RRR/CRR) |
| `.jit_bf16_cache/kernel_exact.cpp` | Exact-dim JIT kernel (all 3 layouts) |
| `jit_bf16_gemm.py` | JIT compilation framework |
| `autotune_bf16.py` | Autotune dispatch with caching |
| `bench_bf16_vs_torch.py` | Benchmark vs torch.mm |

## Build & Test

```bash
cd /shared_nfs/kyle/HipKittens2/analysis/bf16_gemm/mi350x

# Dynamic kernel
THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 ROCM_PATH=/opt/rocm \
  make TARGET=tk_bf16_layouts SRC=kernel_bf16_dynamic.cpp -j4

# JIT compile all shapes
THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 python3 -c "
from jit_bf16_gemm import warmup_shapes
shapes = [(4096,4096,4096),(8192,8192,8192),(8192,28672,4096)]
warmup_shapes(shapes, verbose=True)
"

# Benchmark (subprocess mode for JIT)
HIP_VISIBLE_DEVICES=7 python3 bench_bf16_vs_torch.py
```

## Next Steps (to reach 1.0x)

1. **Inline ASM main loop** — hand-schedule ds_read/buffer_load/MFMA interleaving
2. **Tile-level atomic Split-K** — requires ISA-level atomic store bypassing kittens API
3. **32×32×16 BF16 MFMA** — CDNA4 new instruction, different latency/throughput profile
