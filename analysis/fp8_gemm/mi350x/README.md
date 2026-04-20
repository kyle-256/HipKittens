# MI350X MXFP4 GEMM Kernel

MXFP4 GEMM kernel for gfx950 (MI350X / MI355X) using ThunderKittens C++ with inline ASM.

## Current Status (2026-04-20)

**HipKittens kernel vs aiter: 16/42 WIN potential (per-shape tuned), ~95% mean.**
- 12/42 WIN with default baseline (FUSED_STEP34=1, TAIL_SPLIT=1, GM=4)
- +2 WIN with GROUP_SIZE_M=6 (32768×14336×2048, 32768×28672×2048)
- +2 WIN with GLOBAL_B=1 (fixes K=28672 CRASH shapes)
- Max: 144.3% (4096×128256×32768)

| Status | Count | Examples |
|---|---|---|
| WIN (≥100%) | 16 | 16384x4096x2048 (111%), 4096x128256x32768 (144%) |
| Close (95-100%) | 5 | 4096x4096x16384 (99.3%), 16384x4096x6144 (99%) |
| Medium (85-95%) | 15 | 16384x28672x2048 (97%), 6144x4096x8192 (92%) |
| Far (<85%) | 6 | 4096x32768x128256 (76%), 14336x4096x32768 (75%) |

### Compile flags
- `GLOBAL_B=1` — B tiles via buffer_load from global (fixes K=28672 CRASH, slower on most shapes)
- `GROUP_SIZE_M=N` — grid swizzle for L2 reuse (6 best for large-M shapes)

## Architecture

```
Tile:  256×256 (BLK=256), BK=128 bytes (256 fp4 elements)
Warps: 4 (2×2), each warp owns 128×128 of output
N-split: B tile → left (128) + right (128)
Steps/K-iter: A0×Bl → A0×Br → A1×Bl → A1×Br (4 × 32 MFMA = 128 total)
Double-buffered: tiles via buffer_load_to_lds, scales via SRD buffer_load
LDS: 128 KB (4 tiles × 2 buffers × 128×128 bytes)
Registers: 256 VGPRs + 256 AGPRs, 0 spills
```

## Key Performance Gap vs aiter

aiter uses **4:1:1 MFMA:buffer_load:ds_read interleaved scheduling**:
```
4× MFMA → 1× buffer_load_dwordx4 → 1× ds_read_b128 → repeat
```

HipKittens uses **front-loaded ds_read + pure MFMA burst**:
```
8× (MFMA + ds_read) → 24× pure MFMA → barrier
```

The aiter pattern hides memory latency inside MFMA execution, while our pattern
creates stalls waiting for ds_reads before MFMAs can start.

## Files

| File | Description |
|------|-------------|
| `kernel_mxfp4_gluon_cpp.cpp` | Main kernel source (2103 lines) |
| `Makefile` | Build system |

## Build

```bash
THUNDERKITTENS_ROOT=$(git rev-parse --show-toplevel) ROCM_PATH=/opt/rocm \
  make -B TARGET=tk_mxfp4_gluon_cpp SRC=kernel_mxfp4_gluon_cpp.cpp \
  CPPFLAGS='-DN_DIM=4096 -DK_DIM=8192 -DFUSED_STEP34=1 -DTAIL_SPLIT=1'
```

## Benchmark

```python
# Quick single-shape test
import importlib.util, torch
spec = importlib.util.spec_from_file_location("mod", "tk_mxfp4_gluon_cpp.cpython-310-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

M, N, K = 8192, 8192, 8192
a = torch.randint(0, 256, (M, K//2), dtype=torch.uint8, device='cuda')
b = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device='cuda')
sa = torch.randint(120, 136, (M, K//32), dtype=torch.uint8, device='cuda')
sb = torch.randint(120, 136, (N, K//32), dtype=torch.uint8, device='cuda')
C = torch.zeros((M, N), dtype=torch.bfloat16, device='cuda')
mod.gemm_rcr(a, b, sa, sb, C)
```
