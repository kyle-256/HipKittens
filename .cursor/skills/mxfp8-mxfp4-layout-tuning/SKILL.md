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
- `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp` — main kernel
- `analysis/fp8_gemm/mi350x/test_mxfp8_python.py` — benchmark/correctness harness
- `analysis/fp8_gemm/mi350x/Makefile` — build with CPPFLAGS macros
- `include/ops/warp/memory/util/util.cuh` — `make_srsrc`, `llvm_amdgcn_raw_buffer_load_b32`

## Build Commands
### Current best (buffer_load + KPAIR_LOOP)
```bash
cd analysis/fp8_gemm/mi350x
THUNDERKITTENS_ROOT=$(git rev-parse --show-toplevel) ROCM_PATH=/opt/rocm \
  CPPFLAGS='-DM_DIM=8192 -DN_DIM=8192 -DK_DIM=8192 -DMXFP8_RCR_EXACT_8WAVE_FAST_ENABLE=1 -DMXFP8_RCR_EXACT_PQ_KPAIR_LOOP_ENABLE=1 -DMXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE=1' \
  make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp
```

### Smoke test
```bash
MXFP8_PRESHUFFLE_QUANT=1 MXFP8_LAYOUTS=rcr MXFP8_WARMUP=5 MXFP8_ITERS=10 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 \
  python3 test_mxfp8_python.py 256 256 256
```

### Formal benchmark
```bash
MXFP8_PRESHUFFLE_QUANT=1 MXFP8_LAYOUTS=rcr MXFP8_WARMUP=100 MXFP8_ITERS=200 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=5 \
  python3 test_mxfp8_python.py 8192 8192 8192
```

## Macro Controls
| Macro | Purpose |
| --- | --- |
| `MXFP8_RCR_EXACT_8WAVE_FAST_ENABLE=1` | Enable exact 8-wave fast path (required) |
| `MXFP8_RCR_EXACT_PQ_KPAIR_LOOP_ENABLE=1` | 2x manual K-pair loop unrolling |
| `MXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE=1` | buffer_load with SGPR SRD for scales |

## Current Performance (batch timing, 8192^3)
| Version | TFLOPS | Spills | SNR | vs FP8 |
| --- | ---: | ---: | --- | --- |
| FP8 per-tensor | 3335 | 0 | - | 100% |
| MXFP8 buffer_load + SGPR SRD + KPAIR_LOOP | 3031 | 3 | 49.60 dB | 90.9% |
| MXFP8 KPAIR_LOOP (global_load) | 2999 | 8 | 49.60 dB | 90.0% |
| MXFP8 pre-KPAIR_LOOP baseline | ~2650 | 0 | 49.60 dB | ~79% |

## Key Optimization Details

### KPAIR_LOOP (2x manual unroll)
The K-pair inner loop is manually unrolled 2x inside a lambda, eliminating the overhead of the compiler's loop structure and enabling better instruction interleaving between successive MFMA blocks. This is the single biggest optimization (+13%).

### buffer_load with SGPR SRD
Scale loads use `buffer_load_dword` with Scalar Resource Descriptors (SRDs) instead of `global_load_dword`:
- `make_scale_srd` lambda constructs a 128-bit SRD per scale pointer using `make_buffer_resource` with config `0x00110000u`
- `__builtin_amdgcn_readfirstlane` forces each SRD component into SGPRs, preventing waterfall loops
- `soffset` (scalar uniform offset) carries `k_pair << 8`, `voffset` carries per-lane byte offset
- Reduces spills from 8 to 3 (saves ~36 B scratch)

### Remaining 9% gap to FP8
- Scale load `vmcnt` stall: ~5% (scale loads finish just before MFMAs need them)
- Scale remap `v_lshr_b32`: ~2.5% (phase-dependent scale remapping)
- `mfma_scale` 16-byte instruction overhead: ~1.6% (double the I-cache footprint vs `v_mfma_f32`)

## Durable Findings
- Software pipelining of scale loads causes 150-200 VGPR spills (256 VGPR limit).
- LDS caching of scales is infeasible: tile double-buffers already consume 64 KB LDS.
- `sched_group_barrier` variants tested neutral to slightly worse vs `sched_barrier(0)`.
- Per-iteration `torch.cuda.synchronize()` causes GPU DVFS clock drops; always use batch timing.
- `C.zero_()` inside the timed loop adds ~7% overhead; pure GEMM numbers are the fair comparison.

## Known Dead Ends
- `4de0a032` buffer_load scale path: fast but invalid at large K (correctness/determinism fail).
- Shared/LDS scale cache: correctness problems, no validated win.
- 4-wave exact PQ path: correct but slower than 8-wave.
- `phase1` `op_sel_hi` asm variants: correctness works but long-run performance regressed.
- Scalar phase-pack caching and remap micro-optimizations: did not beat current schedule.
- Inline asm async prefetch: achieved 2929-3052 TFLOPS but failed correctness (no compiler vmcnt tracking across loop boundaries).
- L2 prefetch with `buffer_load_dword` discard VGPR: caused 190 spills.
- `sched_group_barrier` replacing `sched_barrier(0)`: -0.8% regression.

## MXFP4 Notes (Pending)
- MXFP4 target throughput is `2x` MXFP8.
- Define pack and shape model before cloning FP8/MXFP8 assumptions into MXFP4.
- `mfma_scale` instruction supports FP4 via `CBSZ`/`BLGP` opsel variants.

## Debug Workflow
1. Read `analysis/fp8_gemm/mi350x/README.md` for current task state.
2. One change at a time. No mixed loader + schedule + waitcnt experiments.
3. Rebuild, smoke test, then formal benchmark if smoke is clean.
4. After throughput changes, inspect VGPRs, spills, occupancy, LDS in compile remarks.
5. Only keep changes that survive both correctness gate and performance gate.
