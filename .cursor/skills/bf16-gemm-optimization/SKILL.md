---
name: bf16-gemm-optimization
description: BF16 GEMM optimization on AMD CDNA4 (gfx950/MI350X). Single-.so dynamic kernel with runtime K-specialization via template dispatch, 8-wave ping-pong, RCR/RRR/CRR layouts. No JIT.
---
# BF16 GEMM Optimization Status

## Ground Rule — NO JIT

All optimizations must live in a single `tk_bf16_layouts.so` built from
`kernel_bf16_dynamic.cpp`. Do not re-introduce per-shape compilation with
`-DM_DIM/-DN_DIM/-DK_DIM` compile flags, do not create a `.jit_*_cache/`
folder, and do not add a Python-level driver that loads different `.so`
files per shape.

## Current Performance (GPU1, 2026-04-17)

Single `tk_bf16_layouts.so`, 48 LLM shapes, runtime group_m autotune.

| Layout | Geo-mean vs torch.mm | Wins | Status |
|---|---|---|---|
| **RCR** | **~0.98x** | ~10/48 | Close, needs +2pp |
| **RRR** | **~0.97x** | ~11/48 | Close, needs +3pp |
| **CRR** | **~0.95x** | ~3/48  | Weakest, needs +5pp |

Numbers fluctuate ±0.005 between runs depending on device thermals.

`torch.mm` under the hood uses hipBLASLt for bf16 GEMM.

## Architecture

- **Block size:** 256×256 output, K_STEP=64
- **Warps:** 2×4 = 8 warps/block, 512 threads
- **Occupancy:** 2 waves/SIMD (VGPRs ≤ 256)
- **Double buffering:** tic/toc ping-pong, 2×2 smem tiles per operand
- **XCD swizzle:** `chiplet_transform_chunked` with 64-chunk
- **Dual-direction group swizzle:** group-by-N when `bpc > bpr`, else group-by-M
- **Runtime K-specialization:** `template<Layout L, int KI_HINT>` with
  instantiations for KI ∈ {0, 56, 64, 128, 172, 224, 256, 296, 448, 462, 832}.
  Dispatch via `switch(g.ki)` at runtime. `KI_HINT>0` unlocks `#pragma unroll`.

## Key Files

| File | Description |
|---|---|
| `kernel_bf16_dynamic.cpp` | Single source compiled into `tk_bf16_layouts.so`. Contains all 3 layouts and the KI-specialization dispatch. |
| `Makefile` | Default `TARGET=tk_bf16_layouts`, `SRC=kernel_bf16_dynamic.cpp` |
| `bench_bf16_vs_torch.py` | 48-shape benchmark with runtime group_m autotune |
| `quick_snr.py` | SNR ≥ 48 dB + bit-exact determinism gate |
| `smoke.py` | Quick smoke benchmark on 13 representative shapes |

## Compile Resource Usage

- 3 layouts × 11 KI values = **33 kernel instantiations** in one .so
- VGPRs ≤ 242 on all, occupancy 2 waves/SIMD
- 0 VGPR spills
- CRR at KI ∈ {128, 172, 296} previously used `unroll 1` to dodge SGPR spill;
  this is the **primary suspected cause of the 5pp CRR gap** vs the old JIT
  path that hit 1.2x+ with `unroll 2` despite 7-26 SGPR spills.

## What Closed the JIT-vs-Dynamic Gap

1. **Runtime KI_HINT template dispatch** — replaces JIT's `constexpr int num_tiles = K_DIM / K_STEP`
2. **Group-by-N swizzle** when bpc > bpr — replaces the M↔N kernel swap the JIT
   path used (we do the equivalent inside one kernel by reversing group direction)
3. **Block-level XCD swizzle** — already present in dynamic
4. **Runtime group_m autotune** — per-shape best gm ∈ {1, 2, 4, 8, 16}, cached
   in `.autotune_bf16_cache.json`

## Build & Test

```bash
cd /workspace/code/Hipkittens_per_tensor/analysis/bf16_gemm/mi350x

# Build
THUNDERKITTENS_ROOT=/workspace/code/Hipkittens_per_tensor ROCM_PATH=/opt/rocm make clean
THUNDERKITTENS_ROOT=/workspace/code/Hipkittens_per_tensor ROCM_PATH=/opt/rocm make -j4

# Smoke (correctness)
HIP_VISIBLE_DEVICES=1 PYTHONPATH=. python3 quick_snr.py

# Quick perf (13 shapes, ~1 minute)
HIP_VISIBLE_DEVICES=1 PYTHONPATH=. python3 smoke.py

# Full 48-shape benchmark vs torch.mm
HIP_VISIBLE_DEVICES=1 PYTHONPATH=. python3 bench_bf16_vs_torch.py
```

## Correctness Gates

- **SNR ≥ 47 dB** on the `torch.mm(bf16)` reference (BF16 precision caps
  around 48 dB against a bf16 reference; use 48 dB only when comparing
  against an fp32 reference)
- **Determinism:** same inputs → bit-exact output across ≥ 3 runs
- All 3 layouts × 5 group_m values must pass

## Known Issues

- **2048×2048×2048 CRR non-determinism** — at this shape, the CRR kernel
  produces non-deterministic output (max abs diff ~15-21 bf16 ULPs across
  repeated runs). Root cause not yet diagnosed. All benchmarked shapes are
  ≥ 4096³ and pass determinism, so this is a side issue. Likely related to
  the shared-memory scheduling when `ki` is very small (2048/64 = 32).

## Remaining Gap — Why BF16 Still Below 1.0x

Most shapes sit at 0.92–0.97x. hipBLASLt has ISA-level advantages
(wave-level scheduling, tuned Split-K) that HLL frameworks cannot fully
replicate without per-shape ISA tuning. BF16 is also more memory-bound than
FP8, so schedule choices matter more.

Areas that need more work:
1. **CRR `#pragma unroll 2` on KI ∈ {128, 172, 296}** — currently falls back
   to `unroll 1` due to SGPR spill fears. Force unroll 2 and measure; the
   JIT reference showed spills are OK if throughput wins.
2. **Per-shape waitcnt tuning** — the `s_waitcnt lgkmcnt(8)` and
   `s_waitcnt vmcnt(6)` positions in the main loop were hand-tuned for 8192³
   in the JIT era; they may not be optimal for small-K large-N shapes.
3. **Swap M↔N decisively for large-N shapes** — the group-by-N swizzle
   is a partial fix; a full kernel swap (compute C^T in registers, store
   transposed) might be worth trying for N ≥ 22016.

## What NOT To Do

- Do not recreate `.jit_bf16_cache/` or `kernel_exact.cpp`
- Do not pass `-DM_DIM/-DN_DIM/-DK_DIM` to hipcc
- Do not add more than ~12 KI_HINT instantiations (compile time + .so size)
- Do not accept SNR < 48 dB or non-deterministic output
- Do not commit `tk_*.so` binaries or temp `bench_*.log` / `build_*.log`
