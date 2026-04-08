---
name: mxfp8-mxfp4-layout-tuning
description: Tune HipKittens MXFP8 and MXFP4 microscaling GEMM kernels on gfx950/MI350X. Use when working on analysis/fp8_gemm/mi350x MXFP8/MXFP4 RCR performance, preshuffle-quant scale packing, scale-pack scheduling, buffer_load SRD, KPAIR_LOOP, mfma_scale, SNR, or determinism.
---
# MXFP8 / MXFP4 Microscaling Layout Tuning

## When To Use
- User asks to debug or optimize `analysis/fp8_gemm/mi350x` **MXFP8** or **MXFP4** GEMM.
- User mentions `mxfp8`, `mxfp4`, `microscaling`, `block-scale`, `preshuffle-quant`, `mfma_scale`, `scale-pack`, `E8M0`, `KPAIR_LOOP`, `buffer_load SRD`, or `sched_group_barrier` in the context of scaled FP8/FP4.
- For per-tensor FP8 work, use the `fp8-per-tensor-layout-tuning` skill instead.

## First Read
- Read `analysis/fp8_gemm/mi350x/README.md` for the current task state and validated numbers.
- Read this file's reference section for known dead ends and durable findings.

## Hard Rules
- Formal shape is `8192x8192x8192`.
- `preshuffle-quant` is valid for scale tensors only. `preshuffle-ab` is **not allowed**.
- Both `A_scale` and `B_scale` are preshuffled in the current PQ path.
- Success means: numerical correctness pass, `SNR > 48 dB`, deterministic output, TFLOPS within target.
- Benchmark with batch timing: warmup 100 iterations, measure 200 iterations contiguously (no per-iteration `torch.cuda.synchronize()`).
- Do not claim a win from short runs alone.
- Only create a git commit when the user explicitly asks.
- Do not commit generated artifacts (`.tmp_*.json`, ISA dumps, `.co` files, `.bak*`).

## Primary Files
- `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp` — main MXFP8 kernel (8-wave + 4-wave)
- `analysis/fp8_gemm/mi350x/rcr_mxfp8_4wave_fastpath.inc` — 4-wave MXFP8 RCR kernel
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_hybrid.cpp` — MXFP4 hybrid kernel (4-wave, best)
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_8wave.cpp` — MXFP4 8-wave kernel (slower)
- `analysis/fp8_gemm/mi350x/test_mxfp8_python.py` — MXFP8 benchmark/correctness harness
- `analysis/fp8_gemm/mi350x/test_mxfp4_hybrid.py` — MXFP4 benchmark/correctness harness
- `analysis/fp8_gemm/mi350x/Makefile` — build with CPPFLAGS macros
- `include/ops/warp/memory/util/util.cuh` — `make_srsrc`, `llvm_amdgcn_raw_buffer_load_b32`

## Build Commands
### MXFP8 current best (8-wave, buffer_load + KPAIR_LOOP)
```bash
cd analysis/fp8_gemm/mi350x
THUNDERKITTENS_ROOT=$(git rev-parse --show-toplevel) ROCM_PATH=/opt/rocm \
  CPPFLAGS='-DM_DIM=8192 -DN_DIM=8192 -DK_DIM=8192 -DMXFP8_RCR_EXACT_8WAVE_FAST_ENABLE=1 -DMXFP8_RCR_EXACT_PQ_KPAIR_LOOP_ENABLE=1 -DMXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE=1' \
  make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp
```

### MXFP4 hybrid (4-wave)
```bash
THUNDERKITTENS_ROOT=$(git rev-parse --show-toplevel) ROCM_PATH=/opt/rocm \
  CPPFLAGS='-DM_DIM=8192 -DN_DIM=8192 -DK_DIM=8192' \
  make -B TARGET=tk_mxfp4_hybrid SRC=kernel_mxfp4_hybrid.cpp
```

### Smoke test
```bash
# MXFP8
MXFP8_PRESHUFFLE_QUANT=1 MXFP8_LAYOUTS=rcr MXFP8_WARMUP=5 MXFP8_ITERS=10 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 \
  python3 test_mxfp8_python.py 256 256 256
# MXFP4
MXFP4_PRESHUFFLE_QUANT=1 MXFP4_WARMUP=5 MXFP4_ITERS=10 MXFP4_CHECK=1 MXFP4_DETERMINISM_RUNS=3 \
  python3 test_mxfp4_hybrid.py 256 256 256
```

### Formal benchmark
```bash
# MXFP8
MXFP8_PRESHUFFLE_QUANT=1 MXFP8_LAYOUTS=rcr MXFP8_WARMUP=100 MXFP8_ITERS=200 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=5 \
  python3 test_mxfp8_python.py 8192 8192 8192
# MXFP4
MXFP4_PRESHUFFLE_QUANT=1 MXFP4_WARMUP=100 MXFP4_ITERS=200 MXFP4_CHECK=1 MXFP4_DETERMINISM_RUNS=5 \
  python3 test_mxfp4_hybrid.py 8192 8192 8192
```

## Current Performance (batch timing, 8192^3)

### MXFP8
| Version | TFLOPS (with zero) | TFLOPS (pure) | Spills | SNR | vs FP8 |
| --- | ---: | ---: | ---: | --- | --- |
| FP8 per-tensor | 3335 | 3335 | 0 | - | 100% |
| MXFP8 8-wave (KPAIR+SRD) | 2848 | 3011 | 3 | 49.60 dB | 90.3% |
| MXFP8 4-wave (unoptimized) | 2118 | ~2250 | 0 | 49.60 dB | 67.5% |

### MXFP4
| Version | TFLOPS (with zero) | TFLOPS (pure) | Spills | SNR | vs Gluon |
| --- | ---: | ---: | ---: | --- | --- |
| Gluon reference | ~5400 | ~5400 | - | - | 100% |
| MXFP4 hybrid (4-wave) | 4303 | 4582 | 0 | 49.62 dB | 84.9% |
| MXFP4 V2 (4-wave, pre-hybrid) | 4032 | 4257 | 0 | 49.60 dB | 78.8% |
| MXFP4 8-wave | 3466 | 3631 | 0 | 49.60 dB | 67.2% |

## Architecture Comparison: 4-wave vs 8-wave

| Property | 4-wave | 8-wave |
| --- | --- | --- |
| Warp grid | 2×2 | 2×4 |
| Threads | 256 | 512 |
| Occupancy | 1 (or 2 if VGPRs allow) | 2 |
| LDS per workgroup | 128–139 KB (double-buffer) | 64 KB (single-buffer) |
| AGPRs per warp | 256 available | 256 available |
| Latency hiding | Double-buffer pipeline | Wave-level parallelism |
| Scale pipeline room | Yes (if VGPRs < 256) | No (VGPRs = 256) |

**Key finding**: 4-wave consistently outperforms 8-wave for MXFP4 (+26%). For MXFP8, 8-wave currently leads because 4-wave lacks KPAIR_LOOP and inline ASM optimizations.

## MXFP4 Hybrid Architecture (kernel_mxfp4_hybrid.cpp)

### Key Optimizations (validated, +6.4% over V2)
1. **Barrier repositioned**: After all tile data extracted to VGPRs (not after Phase 1). Lets 96 MFMAs overlap with tile prefetch.
2. **Intra-block MFMA + tile prefetch**: 4 asm blocks × 4 MFMA with `emit_one_pf` (compiler intrinsic `llvm_amdgcn_raw_buffer_load_lds`) between blocks. All 16 AGPRs declared `"+a"` per block to prevent hipcc AGPR reuse bug.
3. **Custom `make_pf_params`**: Pre-computes SRD/soffset/LDS addrs for prefetch, avoiding repeated `readfirstlane`.

### hipcc Compiler Limitations Discovered
1. **`#pragma unroll 1` breaks correctness**: hipcc loses VGPR liveness tracking for `asm volatile` blocks without unrolling. Must use `#pragma unroll 2`.
2. **Split asm blocks cause AGPR reuse**: 4 separate 4-MFMA blocks each declaring only their 4 AGPRs → compiler reuses same AGPRs across blocks. Fix: declare all 16 AGPRs as `"+a"` in every block.
3. **`buffer_load_dwordx4 ... lds` in inline ASM**: `"s"(int32x4_t)` constraint does not correctly generate 4-SGPR range for SRD. Fix: use compiler intrinsic `llvm_amdgcn_raw_buffer_load_lds` instead.
4. **`coord<>{}` vs brace-init**: `coord<>{0,0,r,c}` creates untyped coord where `unit_coord<2,3>()` doesn't scale by tile dimensions. Use brace-init `{0,0,r,c}` to let the compiler deduce `coord<ST>`.
5. **Scale pipeline across loop boundary**: Moving `load_scale_buffer` before/after the KPAIR loop causes 150–367 VGPR spills on the 8-wave MXFP8 kernel (256 VGPR limit, zero headroom). On 4-wave MXFP4 (14–44 VGPR headroom) this works fine.

## MXFP8 4-Wave Optimization Plan (Next Priority)

The MXFP8 4-wave kernel (`rcr_mxfp8_4wave_fastpath.inc`) is the most promising path for exceeding 8-wave performance. Current state:
- **VGPRs: 212–242, AGPRs: 0, Spills: 0** — 14–44 VGPRs headroom
- Accumulators in VGPRs (not AGPRs) → extra VGPR↔AGPR copy overhead per MFMA
- No KPAIR_LOOP, no buffer_load SRD, no inline ASM MFMAs

Planned optimizations (in priority order):
1. **Inline ASM MFMAs with AGPR accumulators**: Move acc from VGPRs to AGPRs via `"+a"` constraints. Eliminates copy overhead, frees ~64 VGPRs.
2. **KPAIR_LOOP**: Manual 2× unroll of K-pair loop (same technique as 8-wave, +13% expected).
3. **buffer_load SRD for scales**: Same SRD technique as 8-wave, reduces scale load overhead.
4. **Scale prefetch pipeline**: With AGPR accumulators freeing VGPRs, there's room to prefetch k_pair+1 scales during k_pair MFMAs (impossible on 8-wave due to 256 VGPR limit).
5. **Tile prefetch interleaving**: Same `emit_one_pf` technique from MXFP4 hybrid — issue `buffer_load_to_lds` between split MFMA asm blocks.

Expected outcome: close to or exceeding 8-wave's 3011 TFLOPS, with potential for further gains from scale pipeline.

## Durable Findings

### MXFP8
- KPAIR_LOOP (2× manual unroll) is the single biggest optimization for 8-wave (+13%).
- buffer_load with SGPR SRD reduces spills from 8 to 3 and adds +1%.
- The remaining ~9% gap to FP8 per-tensor: scale vmcnt stall (~5%), scale remap v_lshr_b32 (~2.5%), mfma_scale I-cache overhead (~1.6%).
- Software pipelining of scale loads on 8-wave causes 150–367 VGPR spills (256 limit, zero headroom). **This does NOT apply to 4-wave** (14–44 VGPRs headroom).
- LDS caching of scales is infeasible: tile double-buffers already consume 64–128 KB LDS.
- `sched_group_barrier` variants tested neutral to slightly worse vs `sched_barrier(0)`.

### MXFP4
- 4-wave with 128KB double-buffer outperforms 8-wave with occupancy 2 by 26%.
- Intra-block tile prefetch (between split 4-MFMA asm blocks) gives +3.4% over inter-block.
- Distributing prefetch loads more evenly (2+2+2+2+4+4 vs 4+4+4+4+0+0) does NOT help — more code paths hurt compiler optimization.
- N-split reordering (DOT_left/DOT_right) causes non-deterministic errors with hipcc (suspected register tracking bug across reordered asm blocks).

### General
- Per-iteration `torch.cuda.synchronize()` causes GPU DVFS clock drops; always use batch timing.
- `C.zero_()` inside the timed loop adds ~5–7% overhead; pure GEMM numbers are the fair comparison.
- hipcc `#pragma unroll 2` is critical for correctness with split asm volatile blocks.

## Known Dead Ends
- `4de0a032` buffer_load scale path: fast but invalid at large K (correctness/determinism fail).
- Shared/LDS scale cache: correctness problems, no validated win.
- `phase1` `op_sel_hi` asm variants: correctness works but long-run performance regressed.
- Scalar phase-pack caching and remap micro-optimizations: did not beat current schedule.
- Inline asm async prefetch (8-wave): achieved 2929–3052 TFLOPS but failed correctness.
- L2 prefetch with `buffer_load_dword` discard VGPR: caused 190 spills.
- `sched_group_barrier` replacing `sched_barrier(0)`: -0.8% regression.
- MXFP8 8-wave scale pipeline (any form): 256 VGPR hard limit prevents cross-iteration liveness.
- MXFP4 `buffer_load_dwordx4 ... lds` in single asm block: SRD `"s"(int32x4_t)` constraint broken.
- MXFP4 N-split DOT_left/DOT_right reorder: non-deterministic correctness failure.

## Debug Workflow
1. Read `analysis/fp8_gemm/mi350x/README.md` for current task state.
2. One change at a time. No mixed loader + schedule + waitcnt experiments.
3. Rebuild, smoke test, then formal benchmark if smoke is clean.
4. After throughput changes, inspect VGPRs, spills, occupancy, LDS in compile remarks.
5. Only keep changes that survive both correctness gate and performance gate.
6. For inline ASM MFMA blocks: always declare ALL accumulator AGPRs as `"+a"` in every split block.
7. For coord arguments to `make_pf_params`: use brace-init `{0,0,r,c}`, never `coord<>{...}`.
