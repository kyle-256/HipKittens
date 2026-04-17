# BF16 GEMM on MI350X (gfx950)

HipKittens BF16 GEMM with native RCR/RRR/CRR layout support on AMD CDNA4,
**built once** to a single `tk_bf16_layouts.so`. No per-shape JIT compilation.

## Performance vs hipBLASLt (48 LLM shapes, GPU1)

| Layout | Geo-mean | Wins |
|---|---|---|
| **RCR** | **0.983x** | 7 / 48 |
| **RRR** | **0.973x** | 11 / 48 |
| **CRR** | **0.948x** | 4 / 48 |

Small/square shapes win 1.05-1.15x over hipBLASLt; large tall-N (N≥22016)
shapes are the weak spot, where hipBLASLt pulls ahead ~5-10% due to
better ISA-level scheduling.

Full results in `bench_bf16_no_jit_final.json`.

## Architecture

- **Tile:** 256×256, K_STEP=64, 2×4 warps (512 threads)
- **Occupancy:** 2 waves/SIMD, ≤242 VGPRs, 0 spills (all 33 variants)
- **Scheduling:** 8-wave ping-pong with `s_waitcnt`/`s_barrier`/`s_setprio`
- **Swizzle:** XCD-aware `chiplet_transform_chunked` + dual (group-by-M / group-by-N) tile mapping
- **K-specialization:** `template<Layout, int KI_HINT>` with explicit instantiations for
  KI ∈ {56, 64, 128, 172, 224, 256, 296, 448, 462, 832} and a `KI_HINT=0`
  dynamic fallback for uncommon K.
- **Unroll:** full `#pragma unroll 2` on KI-specialized paths, except CRR KI∈{128,172,296}
  where unroll 1 avoids SGPR spills.

## Build

```bash
THUNDERKITTENS_ROOT=/workspace/code/Hipkittens_per_tensor \
ROCM_PATH=/opt/rocm \
  make -j4
```

Compiles `kernel_bf16_dynamic.cpp` → `tk_bf16_layouts.cpython-*.so` (~1MB).

## Quick Run

```bash
# Small smoke test (13 shapes × 3 layouts)
HIP_VISIBLE_DEVICES=1 PYTHONPATH=. python3 smoke.py

# Correctness (SNR >= 47 dB, bit-exact determinism)
HIP_VISIBLE_DEVICES=1 PYTHONPATH=. python3 quick_snr.py

# Full 48-shape benchmark → bench_bf16_no_jit_final.json
HIP_VISIBLE_DEVICES=1 PYTHONPATH=. python3 bench_bf16_vs_torch.py
```

## API

```python
import tk_bf16_layouts as m

# A, B, C contiguous bfloat16 CUDA tensors; group_m ∈ {1,2,4,8,16}
m.gemm_rcr(A, B, C, group_m)  # A: M×K, B: N×K, C: M×N
m.gemm_rrr(A, B, C, group_m)  # A: M×K, B: K×N, C: M×N
m.gemm_crr(A, B, C, group_m)  # A: K×M, B: K×N, C: M×N
```

Alignment constraints: `M % 256 == 0`, `N % 256 == 0`, `K % 64 == 0`.

## Files

| File | Description |
|---|---|
| `Makefile` | Builds single `.so` |
| `kernel_bf16_dynamic.cpp` | Kernel source (RCR/RRR/CRR, K-specialized, 33 variants) |
| `bench_bf16_vs_torch.py` | 48-shape benchmark with group_m autotune |
| `quick_snr.py` | Correctness gate |
| `smoke.py` | 13-shape smoke test |
| `bench_bf16_no_jit_final.json` | Benchmark results |
| `tk_bf16_layouts.cpython-*.so` | Compiled kernel |
