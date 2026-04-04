# MI350X FP8/MXFP8 Strict Layout Tuning

This directory contains the strict-layout GEMM tuning work for `gfx950` / `MI350X`.
Use this README as the task handoff before editing `kernel_mxfp8_layouts.cpp`,
`kernel_fp8_layouts.cpp`, or running the layout benchmark harnesses.

## Scope
- FP8 baseline kernels live in `kernel_fp8_layouts.cpp` with `test_python.py`.
- MXFP8 tuning lives in `kernel_mxfp8_layouts.cpp` with `test_mxfp8_python.py`.
- MXFP4 is still pending, but it inherits the same strict-layout and benchmarking rules.

## Task Requirements
- Implement `mxfp8` and `mxfp4` GEMM operators for `RCR`, `RRR`, and `CRR`.
- Formal benchmark shape is `8192x8192x8192`.
- `preshuffle-quant` is allowed for scale tensors only.
- Both `A_scale` and `B_scale` may be preshuffled or prepacked.
- `preshuffle-ab` is not allowed.
- `mxfp8` should approach `fp8` throughput, excluding quantization overhead.
- `mxfp4` target throughput is `2x` the `mxfp8` throughput.
- Re-run performance sequentially when background GPU jobs may exist. Do not trust overlapped runs.

## Performance Targets

All targets use `8192x8192x8192`, SNR > 48 dB, deterministic output.

### FP8 per-tensor
| Layout | Target | Current | Status |
| --- | ---: | ---: | --- |
| RCR | > 3100 TFLOPS | 3130 TFLOPS | **PASS** |
| RRR | >= 95% of RCR | 3121 TFLOPS (99.7%) | **PASS** |
| CRR | >= 95% of RCR | 2914 TFLOPS (93.1%) | FAIL — bottleneck: `ds_read_b64_tr_b8` 8B vs RCR's `ds_read_b128` 16B |

### MXFP8 (preshuffle-quant)
| Layout | Target | Current | Status |
| --- | ---: | ---: | --- |
| RCR | approach FP8 (~3100+) | 3031 TFLOPS (90.9% of FP8) | in progress |
| RRR | >= 95% of MXFP8 RCR | not benchmarked at 8192 | TODO |
| CRR | >= 95% of MXFP8 RCR | 1085 TFLOPS (35.8% of RCR) | TODO — not optimized |

### MXFP4
| Layout | Target | Current | Status |
| --- | ---: | ---: | --- |
| RCR | 5500 TFLOPS | — | TODO |
| RRR | 5450 TFLOPS | — | TODO |
| CRR | 5400 TFLOPS | — | TODO |

### Priority order
1. FP8 CRR → close 92.9% → 95% gap (or accept if hardware-limited)
2. MXFP8 RRR / CRR → bring to 95% of MXFP8 RCR
3. MXFP4 RCR / RRR / CRR → 5500 / 5450 / 5400 TFLOPS

## Important Files
- `kernel_fp8_layouts.cpp`
- `kernel_mxfp8_layouts.cpp`
- `rcr_mxfp8_4wave_fastpath.inc`
- `test_python.py`
- `test_mxfp8_python.py`

## Benchmark Protocol
### MXFP8 build
```bash
THUNDERKITTENS_ROOT=$(git rev-parse --show-toplevel) \
CPPFLAGS='-DM_DIM=8192 -DN_DIM=8192 -DK_DIM=8192 -DMXFP8_RCR_EXACT_8WAVE_FAST_ENABLE=1' \
make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp
```

### MXFP8 smoke test
```bash
MXFP8_PRESHUFFLE_QUANT=1 \
MXFP8_LAYOUTS=rcr \
MXFP8_WARMUP=5 \
MXFP8_ITERS=10 \
MXFP8_CHECK=1 \
MXFP8_DETERMINISM_RUNS=3 \
python3 test_mxfp8_python.py 256 256 256
```

### MXFP8 formal run
```bash
MXFP8_PRESHUFFLE_QUANT=1 \
MXFP8_LAYOUTS=rcr \
MXFP8_WARMUP=50 \
MXFP8_ITERS=200 \
MXFP8_CHECK=1 \
MXFP8_DETERMINISM_RUNS=3 \
python3 test_mxfp8_python.py 8192 8192 8192
```

### Acceptance gate
- Numerical correctness pass
- `SNR > 48 dB`
- Deterministic output across repeated runs
- Long-run TFLOPS must survive the same benchmark settings when rerun sequentially

## Current Validated Status
- The active high-value path is `MXFP8 RCR exact + preshuffle-quant`.
- In the current PQ path, both `A_scale` and `B_scale` are preshuffled.
- Best known: `buffer_load + SGPR SRD + KPAIR_LOOP` at **3031 TFLOPS** (batch timing, 90.9% of FP8).

### Current performance (batch timing: warmup 100, iters 200)

| Version | TFLOPS | Spills | Scratch | SNR | vs FP8 |
| --- | ---: | ---: | ---: | --- | --- |
| FP8 per-tensor (baseline) | 3335 | 0 | 0 | - | 100% |
| MXFP8 buffer_load + SGPR SRD | 3031 | 3 | 16 B | 49.60 dB PASS | 90.9% |
| MXFP8 KPAIR_LOOP (global_load) | 2999 | 8 | 36 B | 49.60 dB PASS | 90.0% |

Build flags for current best:
```
-DMXFP8_RCR_EXACT_8WAVE_FAST_ENABLE=1
-DMXFP8_RCR_EXACT_PQ_KPAIR_LOOP_ENABLE=1
-DMXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE=1
```

### Recent commit sweep
All rows below used the same `8192^3`, `RCR + PQ`, `warmup=50`, `iters=200`,
correctness-on, determinism-on benchmark (per-iteration timing).

| Rank | Commit | TFLOPS | Avg ms | Status | Note |
| --- | --- | ---: | ---: | --- | --- |
| 1 | `768b60fa` | 2648.81 | 0.4151 | PASS | `Tune MXFP8 exact PQ scale-pack scheduling.` |
| 2 | `7b8cf714` | 2613.61 | 0.4207 | PASS | `Tune MXFP8 exact PQ preshuffled scale-pack addressing.` |
| 3 | `62de9d9c` | 2590.51 | 0.4244 | PASS | `Tune MXFP8 exact PQ phase-pack remapping.` |
| 4 | `bc18614a` | 2237.26 | 0.4915 | PASS | `Tune MXFP8 exact PQ steady-state scheduling.` |
| 5 | `37bd7bf5` | 2141.43 | 0.5134 | PASS | `Tune MXFP8 exact PQ accumulator plumbing.` |
| 6 | `de3b84f9` | 2111.00 | 0.5208 | PASS | `Tune gated MXFP8 RCR exact 8-wave path.` |
| 7 | `e99e3809` | 1403.41 | 0.7835 | PASS | `Add gated MXFP8 RCR exact 8-wave fast path.` |
| - | `4de0a032` | 2620.01 | 0.4197 | FAIL | Fast but invalid. Correctness and determinism fail. |

## Durable Findings
- **KPAIR_LOOP** (2x manual unroll of the k-pair loop) is the single most impactful optimization, boosting from ~2650 to ~3000 TFLOPS (+13%).
- **buffer_load with SGPR SRD** (`readfirstlane` to eliminate waterfall loops, `soffset` for scalar k-pair offset) reduces spills from 8 to 3 and adds another +1%.
- The remaining ~9% gap to FP8 comes from: scale load vmcnt stall (~5%), scale remap v_lshr_b32 (~2.5%), mfma_scale 16-byte instruction overhead (~1.6%).
- Software pipelining of scale loads fails due to the 256 VGPR limit: any cross-iteration live range for scale_packs causes 150-200 spills.
- LDS caching of scales is infeasible: tile double-buffers already consume the full 64 KB LDS.
- `MXFP8_PRESHUFFLE_QUANT=1` means both `A_scale` and `B_scale` are preshuffled. Any scheduling analysis should treat both sides as part of the critical path.
- When benchmarking, use batch timing (warmup + contiguous measured iterations) to avoid GPU DVFS clock drops from per-iteration `torch.cuda.synchronize()`.

## Known Dead Ends
- `4de0a032` scale-pack `buffer_load_dword` path: invalid at large `K`; correctness and determinism fail.
- Shared/LDS scale cache experiments: produced correctness problems and did not survive validation.
- 4-wave exact PQ path: correct, but slower than the exact 8-wave path.
- `phase1` `op_sel_hi` asm variants: semantics can be made correct, but long-run performance regressed.
- Scalar phase-pack caching and similar remap micro-optimizations: did not beat the current exact 8-wave schedule on long runs.

## Next Priorities
- Extend the validated MXFP8 work beyond `RCR` while keeping the same scale-only preshuffle rule.
- Define the MXFP4 pack and shape model before cloning FP8 assumptions into `mxfp4`.
- Benchmark against FP8, `aiter`, and other backends only when the GPU is otherwise idle.
