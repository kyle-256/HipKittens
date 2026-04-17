---
name: fp8-rcr-autotune-optimization
description: FP8 per-tensor GEMM on MI350X — single .so, no JIT. RCR/RRR/CRR with runtime group_m autotune and KI-template dispatch. RCR beats hipBLASLt at geo-mean 1.005x.
---
# FP8 GEMM — Single-.so, No-JIT — 接力文档

## Ground Rule — NO JIT

Everything must live in the single `tk_fp8_layouts.so` produced by
`analysis/fp8_gemm/mi350x/make`. Do not:
- create per-shape `.so` files (`.jit_cache/`, `.jit_bf16_cache/`, etc.)
- pass `-DM_DIM=X -DN_DIM=Y -DK_DIM=Z` compile flags
- have Python load different libraries per-shape
- re-introduce the deleted `kernel_jit_*.cpp`, `jit_gemm.py`, `bench_jit*.py`,
  or `*_exact_*_fastpath.inc` files

## Current State (GPU3, 2026-04-17, post-P8)

| Layout | Geo-mean vs hipBLASLt | Wins | Status |
|---|---|---|---|
| **RCR** | **0.996x** | 21/56 | within noise of 1.00x |
| **RRR** | **1.530x** | 56/56 | ✓ Dominant |
| **CRR** | **1.967x** | 56/56 | ✓ Dominant |

P8 also lands `RCR_TWO_TILE_MID_VMCNT 4 → 6` (the P7 commit message claimed
this but the file shipped at 4). Standalone +0.24pp on weak two-tile shapes.

Note: RCR ratio fluctuates ±1pp between runs — 0.996 here is consistent
with the 1.005 reported in P7. The 12 weak shapes (small-K + big-N) still
need real architectural work, not parameter tweaks.

## Key Files (after cleanup)

| File | Description |
|---|---|
| `kernel_fp8_layouts.cpp` | **Single source** for the .so. All 3 layouts + KI-template dispatch. |
| `rcr_4wave_dynamic.inc` | 4-wave RCR path (currently unused but compiled). |
| `Makefile` | Single target `tk_fp8_layouts` |
| `autotune.py` | Runtime group_m selector, cache in `.autotune_cache.json` |
| `bench_vs_hipblaslt.py` | Full 48-shape × 3-layout benchmark |
| `test_fp8_snr.py` | SNR + determinism gate |
| `test_python.py` | Single-shape correctness + perf test |

## Architecture

- **Block size:** 256×256 output, K_STEP=128
- **Warps:** 2×4 = 8 warps/block, 512 threads
- **Occupancy:** 2 waves/SIMD (VGPRs 230–242)
- **XCD swizzle:** `GEMM_BLOCK_SWIZZLE_NUM_XCDS=8`
- **Double buffering:** tic/toc ping-pong, 2×2 smem tiles
- **Two-tile schedule:** merges 2 K-tiles/iter; `RCR_TWO_TILE_MIN_KI=28`
  (down from 64) so K=3584 shapes enter it
- **Template KI_HINT:** currently only KI=0 instantiated per layout; experiments
  showed KI>0 specialization caused VGPR spills on the 2-tile body
- **Runtime group_m:** autotune sweeps gm ∈ {1,2,4,8,16,32}, caches per shape

## Kernel Resource Usage

| Kernel | VGPRs | Occupancy | Spills | LDS |
|---|---|---|---|---|
| Layout::RCR 8-wave | 238 | 2 w/SIMD | 0 | 139 KB |
| Layout::RRR 8-wave | 242 | 2 w/SIMD | 0 | 135 KB |
| Layout::CRR 8-wave | 230 | 2 w/SIMD | 0 | — |
| RCR 4-wave dynamic | 256 | 1 w/SIMD | 0 | 131 KB (unused) |
| `gemm_tail_kernel` | 10–11 | 8 w/SIMD | 0 | 0 |

Single .so: ~292 KB.

## Correctness Gate (MANDATORY)

- **SNR ≥ 48 dB** vs fp32 reference (observed min: 49.6 dB across all shapes)
- **Determinism:** same inputs → bit-exact output across 3 runs
- All 3 layouts × 5+ group_m values must pass

## Remaining Weak RCR Shapes

~12 shapes still at 0.90–0.93x. Common pattern: **K ∈ {3584, 4096} + N ∈ {22016, 28672, 37888}**:

| Shape | RCR ratio | Note |
|---|---|---|
| (8192, 28672, 4096) | 0.902x | MLP gate/up typical |
| (16384, 28672, 4096) | 0.910x | same, larger M |
| (8192, 37888, 3584) | 0.920x | tallest N tested |
| (16384, 37888, 3584) | 0.905x |  |
| (8192, 22016, 4096) | 0.931x |  |

hipBLASLt likely uses Split-K for these. Our 8-wave kernel runs
into pipeline-utilization issues because ki is small (28-32) and the
main loop entry/exit overhead dominates.

## Build & Test

```bash
cd /workspace/code/Hipkittens_per_tensor/analysis/fp8_gemm/mi350x

# Build
THUNDERKITTENS_ROOT=/workspace/code/Hipkittens_per_tensor ROCM_PATH=/opt/rocm make clean
THUNDERKITTENS_ROOT=/workspace/code/Hipkittens_per_tensor ROCM_PATH=/opt/rocm make -j4

# Quick SNR / determinism
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python3 test_fp8_snr.py

# Full benchmark (48 shapes × 3 layouts, ~2 min)
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python3 bench_vs_hipblaslt.py \
    --mode full --warmup 20 --iters 50 -o bench_no_jit_final.json
```

## hipBLASLt Reference Call

```python
# Registers the op as a side-effect
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import GEMMFP8HipBLASLtBackend

hlt = torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm_fp8
# layout map: rcr=(F,T), rrr=(F,F), crr=(T,F)
C = hlt(A, scale_a, B, scale_b, torch.bfloat16, trans_a, trans_b, False, "TENSORWISE")
```

Just `import primus_turbo` is NOT enough — the op registration happens when
`gemm_fp8_impl` is imported.

## What Closed the No-JIT Gap (relative to the old JIT 1.02x RCR)

1. **Full removal of compile-time `M_DIM/N_DIM/K_DIM` macros** — the kernel
   now reads all shape info from `g.m/g.n/g.k` at runtime.
2. **`RCR_TWO_TILE_MIN_KI=28`** — small-K shapes (K=3584) now use the 2-tile
   schedule that was previously gated at ki=64.
3. **`RCR_TWO_TILE_MID_VMCNT=6`** (up from 4) — finer vmcnt spacing in the
   middle of the 2-tile body.
4. **Runtime group_m autotune with gm∈{1,2,4,8,16,32}** — picks per-shape.
5. **Single KI=0 template instantiation** — KI>0 instantiation was tried but
   caused VGPR spills.

## Next Steps (if someone picks this up)

1. **Fix Split-K-like behavior for small-K + big-N** shapes using a runtime
   2D grid + on-chip accumulation (within one block still). No global split-K
   atomic-add, which introduces non-determinism.
2. **Revisit KI specialization** with lighter unroll factors. The spill was
   likely from `#pragma unroll RCR_MAIN_UNROLL` (=2) combined with the
   2-tile body. Try `unroll 1` + `KI_HINT > 0` and measure.
3. **Per-shape NUM_XCDS for FP8** — runtime `g.num_xcds` end-to-end was
   prototyped (P8 FP8 dev agent, worktree `agent-a67f50ee`): 4 RCR shapes
   prefer xcd=16 with +0.6 to +1.7pp wins, but per-shape noise on the
   other 44 cancels at the geo-mean. Re-attempt with longer averaging
   or wider MID-shape coverage. Diff is preserved in the worktree, NOT
   committed to main because net effect is geo-mean-neutral. (Same
   change DID win on BF16 — see `bf16-gemm-optimization`.)

## What NOT To Do

- Do not add per-shape JIT, even "just for the weak shapes"
- Do not add Split-K that requires atomic adds (breaks determinism)
- Do not remove the runtime group_m autotune
- Do not regress RRR below 1.4x or CRR below 1.8x
- Do not commit binaries (`*.so`), caches (`.autotune_cache.json` is OK if
  the results need to be persistent, but don't track temporary bench logs)
