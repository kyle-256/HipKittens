# MI350X FP8 / MXFP8 / MXFP4 GEMM Tuning

This directory contains GEMM kernel tuning for `gfx950` (MI350X / MI355X).

---

## MXFP4 (Active)

### Kernel Variants

| File | Description | 8192³ TFLOPS | Notes |
|------|-------------|:---:|-------|
| `kernel_mxfp4_gluon_cpp.cpp` | **Main C++ kernel** — Gluon-architecture reimplementation in ThunderKittens C++ with inline ASM MFMAs | **4632** (89.6%) | Actively optimized. Compile-time N/K dims. |
| `kernel_mxfp4_asm_inline.cpp` + `.h` | Gluon ASM embedded — 2227-line GCN assembly from Gluon compiler, wrapped in C++ launcher | **5171** (100%) | Reference ceiling. Not shape-flexible (8192³ only). |
| `rewrite_mxfp4_gluon.py` | .s post-processor — rewrites compiler-generated assembly for better scheduling | **4764** (92.1%) | Requires manual build pipeline (compile → rewrite → link). |

**% is relative to ASM inline (5171 = 100%).**

### Architecture

```
Tile:  256×256 (BLK=256), BK=128 bytes (256 fp4 elements)
Warps: 4 (2×2), each warp owns 128×128 of the output
N-split: B tile → left (128) + right (128)
Steps/K-iter: A0×Bl → A0×Br → A1×Bl → A1×Br (4 × 32 MFMA = 128 total)
Steps 1+2: merged into kpair_64mfma_step12 (single asm block, 64 MFMAs + 16 ds_reads)
Double-buffered: tiles via buffer_load_to_lds, scales via SRD buffer_load (preshuffled)
LDS: 128 KB (4 tiles × 2 buffers × 128×128 bytes = 128K, full capacity)
Registers: 256 VGPRs + 256 AGPRs, 0 spills
```

### Scale Format

The C++ kernel uses **preshuffle_mfma16** scale format — scales are pre-arranged in memory so `buffer_load_dword` directly yields the format needed by `v_mfma_scale`. This avoids the LDS round-trip (ds_write → ds_read) that the Gluon LLIR kernel uses.

Scale SRDs are constructed once per tile group, with `soffset` advancing by 256 bytes per K iteration.

### Build

```bash
# Single shape (e.g. 8192×8192)
THUNDERKITTENS_ROOT=$(git rev-parse --show-toplevel) \
CPPFLAGS='-DK_DIM=8192 -DN_DIM=8192' \
make -B TARGET=tk_mxfp4_gluon_cpp SRC=kernel_mxfp4_gluon_cpp.cpp

# ASM inline reference
THUNDERKITTENS_ROOT=$(git rev-parse --show-toplevel) \
make -B TARGET=tk_mxfp4_asm_inline SRC=kernel_mxfp4_asm_inline.cpp
```

Compile-time defines:
- `K_DIM`, `N_DIM` — **required**, determines tile decomposition (`bpc = N_DIM / 256`)
- `GROUP_SIZE_M` — tile swizzle group size (default 4; try 1, 2 for different shapes)
- `UNROLL_K` — loop unroll factor (default auto by K size; try 16 for large K)

### Test & Benchmark

```bash
# Quick correctness + perf (default 8192³)
python3 test_mxfp4_gluon_cpp.py
python3 test_mxfp4_gluon_cpp.py 4096 4096 32768    # custom shape

# Full 42-shape competitive benchmark (auto-tunes GROUP_M × UNROLL_K variants)
python3 bench_all_42.py

# Gluon LLIR competitor benchmark (requires triton_for_gluon, see Environment below)
PYTHONPATH=/shared_nfs/kyle/test/triton_for_gluon/python:$PYTHONPATH \
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 \
python3 bench_gluon_a4w4_42.py
```

### Competitive Benchmark: 42 Shapes (LLaMA-8B + LLaMA-70B)

`bench_all_42.py` tests our C++ kernel against `competitor_tflops` (aiter numbers).
`bench_gluon_a4w4_42.py` tests Gluon LLIR against the same competitor numbers.

**Latest results (2026-04-14):**

| Kernel | vs aiter (42 shapes) | 8192³ TFLOPS |
|--------|:---:|:---:|
| Our C++ kernel | **16/42 WIN** (39%) | 4632 |
| Gluon LLIR | 12/42 WIN (29%) | 4983 |
| ASM inline | — | 5171 |
| aiter (competitor) | — | ~5090 |

Our C++ kernel already beats Gluon LLIR on most shapes.
The gap is vs aiter, mainly on **large-N shapes** (N=32768+) where we lose 10-20%.

### Where We Win / Lose

**WIN pattern**: M ≥ 16384, N = 4096–14336 (tall-M, moderate-N shapes)
**LOSE pattern**: M = 4096, N = 32768+ (wide-N shapes, 80–96%)
**ROOT CAUSE of C++ vs ASM gap**: compiler inserts ~32 `s_nop 0` per K-iter, suboptimal prefetch interleaving, SALU not dual-issued with MFMA

### .s Rewriter Pipeline

To get ~94% of ASM performance from the C++ kernel:

```bash
# 1. Generate .s from C++ (device-only)
TK_ROOT=$(git rev-parse --show-toplevel)
/opt/rocm/bin/hipcc kernel_mxfp4_gluon_cpp.cpp \
  -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
  -I/opt/rocm/include/rocrand -std=c++20 -w \
  -I${TK_ROOT}/include -I${TK_ROOT}/prototype \
  $(python3 -m pybind11 --includes) \
  -I/opt/rocm/include/hip -DK_DIM=8192 -DN_DIM=8192 \
  -S -o kernel_mxfp4_gluon_cpp_device.s --offload-device-only

# 2. Run rewriter
python3 rewrite_mxfp4_gluon.py kernel_mxfp4_gluon_cpp_device.s kernel_mxfp4_gluon_cpp_rewritten.s

# 3. Assemble + link back to .so (see commit 90e592e0 for full script)
```

### Optimization History

| Commit | TFLOPS | Description |
|--------|:---:|-------------|
| `70a7c28f` | 5366 | Pure ASM kernel (matches Gluon .s reference) |
| `97783f1c` | 5370 | ASM embedded in C++ header |
| `df5bf1eb` | 5233 | Inline ASM with SGPR remapping fix |
| `90e592e0` | 4764 | C++ + .s rewriter (94.1% of Gluon .s) |
| `53cfcf82` | 4524 | C++ with front-loaded PFs (88.6%) |
| `335d58e4` | — | C++ with merged Step12, GROUP_M=4, XCD dispatch. 13/42 WIN |
| `89259a1b` | — | Cleanup + SRD merge optimization. **Current HEAD** |

### Known Dead Ends for MXFP4

- Batching all 4 tile prefetches into single asm block: **hurts performance** — destroys prefetch/MFMA interleaving
- Moving s_waitcnt+barrier into MFMA asm tail: **hurts performance** — changes pipeline timing
- `emit_four_pf_asm` replacing `emit_one_pf` pairs: concentrates memory ops, creates MFMA bubbles

---

## Environment Setup

### Machine Paths

```
Project root:     /shared_nfs/kyle/test/HipKittens
This directory:   analysis/fp8_gemm/mi350x/
Gluon tutorials:  /shared_nfs/kyle/test/gfx9-gluon-tutorials/kernels/gemm/a4w4/
aiter:            /shared_nfs/kyle/test/aiter/
```

### Triton for Gluon Benchmark

The Gluon LLIR kernel requires a **special Triton build** from the `matmul_4waves` branch:

```
Path:   /shared_nfs/kyle/test/triton_for_gluon
Branch: matmul_4waves (ROCm/triton fork)
Build:  cd /shared_nfs/kyle/test/triton_for_gluon && pip install -e .
```

Required environment variables:
```bash
export PYTHONPATH=/shared_nfs/kyle/test/triton_for_gluon/python:$PYTHONPATH
export TRITON_ENABLE_LLIR_SCHED=1
export TRITON_ENABLE_AMDGCN_AS=1
```

**DO NOT** use the system triton (`/shared_nfs/kyle/triton`, main branch) — it lacks the LLIR scheduler and `extract_slice`.

### aiter Competitor Numbers

The `competitor_tflops` column in `bench_all_42.py` comes from aiter's tuned GEMM config:
```
/shared_nfs/kyle/test/aiter/aiter/configs/a4w4_blockscale_tuned_gemm.csv
```

These are pre-measured numbers from aiter's `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256` kernel.

### GPU

- Device: MI350X / MI355X (`gfx950`)
- 8 GPUs, use `HIP_VISIBLE_DEVICES=N` to select
- Check idle: `rocm-smi` (0% GPU = idle)
- **Run benchmarks sequentially** on an idle GPU. DVFS clock drops invalidate overlapped measurements.

---

## FP8 / MXFP8 (Stable)

### FP8 per-tensor

| Layout | TFLOPS | Status |
|--------|:---:|--------|
| RCR | 3130 | **PASS** |
| RRR | 3121 (99.7% of RCR) | **PASS** |
| CRR | 2914 (93.1%) | FAIL — hardware-limited (`ds_read_b64_tr_b8`) |

### MXFP8 (preshuffle-quant)

| Layout | TFLOPS | Status |
|--------|:---:|--------|
| RCR | 3031 (90.9% of FP8) | Best known |
| RRR | — | TODO |
| CRR | 1085 | TODO |

Build: see MXFP8 section in git history. Key files: `kernel_mxfp8_layouts.cpp`, `rcr_mxfp8_4wave_fastpath.inc`.

---

## File Index

### MXFP4 Kernels
| File | Purpose |
|------|---------|
| `kernel_mxfp4_gluon_cpp.cpp` | Main C++ kernel (actively optimized) |
| `kernel_mxfp4_asm_inline.cpp` | C++ launcher for embedded ASM kernel |
| `kernel_mxfp4_asm_inline.h` | 2227-line GCN assembly body (from Gluon compiler output) |
| `kernel_mxfp4_asm_data.h` | Data tables for ASM kernel |
| `gluon_a4w4_hsaco.h` | Compiled Gluon .hsaco binary header |

### MXFP4 Benchmarks
| File | Purpose |
|------|---------|
| `bench_all_42.py` | 42-shape benchmark vs aiter (auto-tunes GROUP_M × UNROLL_K) |
| `bench_gluon_a4w4_42.py` | 42-shape Gluon LLIR benchmark |
| `test_mxfp4_gluon_cpp.py` | Single-shape correctness + perf test |
| `mxfp4_gluon_launcher.py` | Gluon kernel Python launcher |
| `rewrite_mxfp4_gluon.py` | Assembly post-processor (.s rewriter) |

### MXFP4 Support
| File | Purpose |
|------|---------|
| `gluon_loop_asm.h` | Inline ASM helpers for Gluon loop structure |
| `interleaved_asm.h` | Interleaved MFMA+ds_read ASM blocks |

### FP8 / MXFP8
| File | Purpose |
|------|---------|
| `kernel_fp8_layouts.cpp` | FP8 per-tensor GEMM (RCR/RRR/CRR) |
| `kernel_mxfp8_layouts.cpp` | MXFP8 GEMM with preshuffle-quant |
| `rcr_mxfp8_4wave_fastpath.inc` | MXFP8 RCR 4-wave inline ASM |
| `*_fastpath.inc` | Various layout-specific inline ASM kernels |

### Build
| File | Purpose |
|------|---------|
| `Makefile` | hipcc build rules (TARGET, SRC, CPPFLAGS) |

### Generated (gitignored)
- `build_*/` — compiled .so variants for auto-tuning
- `work_*/` — temporary benchmark runner scripts
- `*.o`, `*.bc`, `*.hipi`, `*.hipfb` — compiler intermediates
- `*.cpython-*.so` — Python extension modules
- `gpucore.*` — GPU crash dumps
