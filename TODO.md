# TODO — FP8 / BF16 GEMM on MI350X

## Ground Rules

- **NO JIT per-shape compilation**. Single `.so` per target (`tk_fp8_layouts.so`,
  `tk_bf16_layouts.so`) built by `make`.
- SNR ≥ 48 dB, bit-exact determinism are hard gates.
- Never commit `*.so`, `.autotune_cache.json` is OK to keep (it's text), logs are not.

## Current Status (2026-04-17)

### FP8 — ✅ Target met on RCR

Measured on GPU0, `bench_vs_hipblaslt.py --mode full`:

| Layout | Geo-mean vs hipBLASLt | Wins | Status |
|---|---|---|---|
| RCR | 1.005x | 23/56 | ✅ ≥ 1.00x |
| RRR | 1.551x | 56/56 | ✅ ≥ 1.40x |
| CRR | 1.974x | 56/56 | ✅ ≥ 1.80x |

Last change: `RCR_TWO_TILE_MID_VMCNT 4 → 6`.

### BF16 — 🚧 Below target

Measured on GPU1, `bench_bf16_vs_torch.py`:

| Layout | Geo-mean vs torch.mm | Wins | Target |
|---|---|---|---|
| RCR | ~0.98x | ~10/48 | ≥ 1.00x |
| RRR | ~0.97x | ~11/48 | ≥ 1.00x |
| CRR | ~0.95x | ~3/48 | ≥ 1.00x |

## Open Items — High Priority

### BF16 CRR (biggest gap, -5pp)
- [ ] Force `#pragma unroll 2` on CRR KI ∈ {128, 172, 296}. Current code
      falls back to `unroll 1` due to SGPR spill fears. JIT reference showed
      spills are OK if throughput wins.
- [ ] Verify `mma_AtB(D, A, B, C)` dispatches to the direct HW form, not the
      wrapper that transposes in registers. If not, use the explicit tile-loop
      calling `mma_AtB_base`.

### BF16 RCR / RRR (small gap, -2 to -3pp)
- [ ] Try M↔N kernel swap for shapes where N > M (explicit grid swap, not
      group-by-N swizzle). Expect +3-5% on non-square shapes per JIT findings.
- [ ] Tune `s_waitcnt lgkmcnt(8)` / `vmcnt(6)` positions for small-K large-N
      shapes.
- [ ] Consider runtime 4-wave path for large-grid shapes (analogous to FP8).

### FP8 RCR (hit 1.005x, still -7-10pp on ~12 weak shapes)
- [ ] Small K + big N remain weak: (M, 28672, 4096), (M, 37888, 3584).
      hipBLASLt likely uses Split-K. Explore deterministic on-chip Split-K
      (no atomics).
- [ ] Revisit KI template specialization with `unroll 1` instead of `unroll 2`
      to avoid spills.
- [ ] Per-shape XCD swizzle tuning (currently fixed num_xcds=8).

## Open Items — Correctness

- [ ] **BF16 2048³ CRR non-determinism** — at M=N=K=2048, CRR produces
      non-deterministic output (~15-21 bf16 ULP max diff). Root cause
      unknown. All benchmarked LLM shapes are ≥ 4096³ so the 48-shape
      benchmark is unaffected, but this should be fixed before production.

## Open Items — Medium Priority

- [ ] Make `autotune.py` also autotune the 4-wave vs 8-wave path choice for
      FP8 RCR.
- [ ] Benchmark against TRITON backend in Primus-Turbo, not only hipBLASLt.
- [ ] Add a CI script that runs `test_fp8_snr.py` + `quick_snr.py` + a
      5-shape perf sanity check on both directories.

## Closed / Completed

- [x] Removed all JIT per-shape compilation (`jit_gemm.py`, `bench_jit*.py`,
      `kernel_jit_*.cpp`, `*_exact_*_fastpath.inc`, `.jit_cache/`,
      `.jit_bf16_cache/`).
- [x] Removed dead experimental kernels (`kernel_1024/2048/4096/8192/16384.cpp`,
      `kernel_bf16_128/256x128/4wave.cpp`, `kernel_crr.cpp`,
      `kernel_layouts.cpp`).
- [x] FP8 RCR geo-mean ≥ 1.00x achieved (1.005x).
- [x] BF16 migrated to single-source `kernel_bf16_dynamic.cpp` with runtime
      KI_HINT template dispatch.
- [x] Both directories use runtime group_m autotune.
- [x] Updated skill docs: `bf16-gemm-optimization`, `fp8-rcr-autotune-optimization`,
      `fp8-strict-layout-tuning`.

## How To Run

```bash
# FP8
cd analysis/fp8_gemm/mi350x
THUNDERKITTENS_ROOT=/workspace/code/Hipkittens_per_tensor ROCM_PATH=/opt/rocm make clean
THUNDERKITTENS_ROOT=/workspace/code/Hipkittens_per_tensor ROCM_PATH=/opt/rocm make -j4
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python3 bench_vs_hipblaslt.py --mode full

# BF16
cd analysis/bf16_gemm/mi350x
THUNDERKITTENS_ROOT=/workspace/code/Hipkittens_per_tensor ROCM_PATH=/opt/rocm make clean
THUNDERKITTENS_ROOT=/workspace/code/Hipkittens_per_tensor ROCM_PATH=/opt/rocm make -j4
HIP_VISIBLE_DEVICES=1 PYTHONPATH=. python3 bench_bf16_vs_torch.py
```
