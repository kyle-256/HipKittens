# FP8 GEMM on MI350X (gfx950)

HipKittens FP8 (e4m3) GEMM with native RCR/RRR/CRR layout support on AMD CDNA4,
**built once** to a single `tk_fp8_layouts.so`. No per-shape JIT compilation.

`C[bf16] = (A[fp8] op B[fp8]) * scale_a * scale_b` (per-tensor scaling).

## Performance vs hipBLASLt (56 LLM shapes, `bench_no_jit_final.json`)

| Layout | Geo-mean | Wins |
|---|---|---|
| **RCR** | **1.00x** | 22 / 56 |
| **RRR** | **1.55x** | 56 / 56 |
| **CRR** | **1.99x** | 56 / 56 |

RRR/CRR sweep hipBLASLt because hipBLASLt's own RRR/CRR fall back to a transpose+RCR
path; TK runs them natively. RCR is at parity — within ~3% on most shapes, with
hipBLASLt pulling ahead on a handful of large tall-N weak shapes.

## Architecture

- **Tile:** 256×256, K_BLOCK=128, 2×4 warps (512 threads)
- **Occupancy:** 2 blocks/CU; clean (no register spills, no scratch)
- **Swizzle:** XCD-aware tile remap (`BLOCK_SWIZZLE_NUM_XCDS=8` chiplets) +
  group-by-M pid mapping (autotune picks `group_m ∈ {1,2,4,8,16}` per shape)
- **LDS layouts:** `st_16x128_v2_s` for B-row / RRR-A / CRR-B,
  `st_16x128_v2a_s` for CRR-A, plain `st_16x128_s` for RRR-A row stream
- **Reads:** `ds_read_b64_tr_b8` cooperative col-major loads on FP8 LDS tiles
- **DTL m0-broadcast hoist** (`rcr_8w_load_hoist`) — inline-asm
  `s_mov_b32 m0, <sgpr>` + `buffer_load_dwordx4 ... offen lds`, with the
  per-pass LDS offset hoisted to SGPR via `__builtin_amdgcn_readfirstlane` to
  prevent LLVM from rematerializing it through a VGPR + `v_readfirstlane` per
  store (P19 Dev A, –16 VGPR, +0.27/+0.47pp cross-GPU)
- **RCR dispatch:** runtime picks 4-wave (2×2 warps, 4 blocks/CU) vs 8-wave (2×4)
  based on `(grid >= RCR_4WAVE_MIN_GRID=3200) && (K <= RCR_4WAVE_MAX_K=8192)`,
  overridable with `TK_RCR_FORCE_KERNEL={4,8}` env var
- **K-specialization:** `gemm_kernel<L, KI_HINT>` instantiated per layout;
  RCR uses a two-tile main-loop schedule when `ki >= RCR_TWO_TILE_MIN_KI=28`

All tuning knobs live as `constexpr int` / `#define` at the top of
`kernel_fp8_layouts.cpp` (BF16-style, no `-D` Makefile flags).

## Build

```bash
THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 \
ROCM_PATH=/opt/rocm \
  make -j
```

Compiles `kernel_fp8_layouts.cpp` → `tk_fp8_layouts.cpython-*.so` (~310 KB).

## Quick Run

```bash
# Correctness (SNR ≥ 48 dB on all 5×3 layout/shape combos, bit-exact determinism)
HIP_VISIBLE_DEVICES=2 python3 test_fp8_snr.py

# Smoke bench (28 shapes)
HIP_VISIBLE_DEVICES=2 python3 bench_vs_hipblaslt.py --mode smoke

# Full 56-shape benchmark → bench_vs_hipblaslt_full_<ts>.json
HIP_VISIBLE_DEVICES=2 python3 bench_vs_hipblaslt.py --mode full
```

`autotune.py` (`AutotunedGEMM`) caches the best `(group_m, kernel)` per shape
to `.autotune_cache.json` and is what `bench_vs_hipblaslt.py` calls under the hood.

## API

```python
import tk_fp8_layouts as m

# A, B fp8e4m3; C bf16; scale_a, scale_b float scalars (or .item()-able)
m.gemm_rcr(A, B, C, scale_a, scale_b, group_m=4)  # A: M×K, B: N×K, C: M×N
m.gemm_rrr(A, B, C, scale_a, scale_b, group_m=4)  # A: M×K, B: K×N, C: M×N
m.gemm_crr(A, B, C, scale_a, scale_b, group_m=4)  # A: K×M, B: K×N, C: M×N
```

Alignment constraints: `M % 256 == 0`, `N % 256 == 0`, `K % 128 == 0`
(misaligned tail rows/cols handled by `gemm_tail_kernel`).

## Files

| File | Description |
|---|---|
| `Makefile` | Builds single `.so` |
| `kernel_fp8_layouts.cpp` | All kernel source (RCR 4-wave + 8-wave, RRR, CRR, tail, dispatch) |
| `autotune.py` | Per-shape `(group_m, kernel)` search + cache |
| `bench_vs_hipblaslt.py` | 28- (smoke) / 56- (full) shape benchmark vs hipBLASLt |
| `test_fp8_snr.py` | SNR + determinism gate |
| `bench_no_jit_final.json` | Full-bench reference results |
| `.autotune_cache.json` | Per-shape `(group_m, kernel)` cache |
| `tk_fp8_layouts.cpython-*.so` | Compiled kernel |
