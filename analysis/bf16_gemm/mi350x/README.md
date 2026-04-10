# BF16 GEMM on MI355X (gfx950)

HipKittens BF16 GEMM with native RCR/RRR/CRR layout support on AMD CDNA4.

## Performance vs hipBLASLt (48 LLM shapes, GPU7)

| Layout | Dynamic 8-wave | Exact-dim JIT | hipBLASLt baseline |
|---|---|---|---|
| **RCR** | 0.970x | **0.992x** | 1.0x |
| **RRR** | 0.957x | **0.973x** | 1.0x |
| **CRR** | 0.944x | **0.961x** | 1.0x |

Small/square shapes: **1.05-1.15x** win over hipBLASLt.
Wide shapes (N≥28672): ~0.93x due to hipBLASLt's ISA-level Split-K.

## Architecture

- **Tile:** 256×256, K_STEP=64
- **Warps:** 2×4 = 8 warps, 512 threads
- **Occupancy:** 2 waves/SIMD (≤256 VGPRs, 0 spills)
- **Features:** XCD-aware block swizzle, runtime group_m autotune, SRSRC buffer_load_lds, double-buffered LDS ping-pong scheduling

## Quick Start

```bash
# Build dynamic kernel
THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 ROCM_PATH=/opt/rocm \
  make TARGET=tk_bf16_layouts SRC=kernel_bf16_dynamic.cpp -j4

# Test
HIP_VISIBLE_DEVICES=7 python3 -c "
import torch, math, tk_bf16_layouts as m
M,N,K = 8192,8192,8192
A = torch.randn(M,K,dtype=torch.bfloat16,device='cuda')
B = torch.randn(N,K,dtype=torch.bfloat16,device='cuda')
C = torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
m.gemm_rcr(A,B,C,4)
ref = A.float()@B.float().T
snr = 10*math.log10((ref**2).sum()/(((C.float()-ref)**2).sum()))
print(f'SNR={snr:.1f}dB')
"
```

## JIT Compilation (per-shape optimization)

```bash
# Pre-compile shapes
THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 python3 -c "
from jit_bf16_gemm import warmup_shapes
warmup_shapes([(4096,4096,4096),(8192,8192,8192)], verbose=True)
"

# Benchmark JIT vs torch.mm
HIP_VISIBLE_DEVICES=7 python3 bench_bf16_vs_torch.py
```

## Files

| File | Description |
|---|---|
| `kernel_bf16_dynamic.cpp` | Main 8-wave dynamic kernel (RCR/RRR/CRR) |
| `.jit_bf16_cache/kernel_exact.cpp` | Exact-dim JIT kernel source |
| `jit_bf16_gemm.py` | JIT compilation framework |
| `autotune_bf16.py` | Production autotune dispatch |
| `bench_bf16_vs_torch.py` | Benchmark vs torch.mm (hipBLASLt) |
| `bench_bf16_jit.py` | JIT benchmark (subprocess mode) |
