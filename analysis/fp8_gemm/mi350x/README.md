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
- The latest valid recent-commit sweep winner is `768b60fa`.

### Recent commit sweep
All rows below used the same `8192^3`, `RCR + PQ`, `warmup=50`, `iters=200`,
correctness-on, determinism-on benchmark.

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
- The best current lever is scale-pack scheduling, not a new kernel family.
- Moving `ensure_scale_packs(k_pair)` earlier in the exact 8-wave loop is the current winning direction.
- That schedule change hides part of the scale-pack load/use chain and reduced the PQ kernel compile remark from `255 VGPRs` to `239 VGPRs` in the winning source state.
- `MXFP8_PRESHUFFLE_QUANT=1` means both `A_scale` and `B_scale` are preshuffled. Any scheduling analysis should treat both sides as part of the critical path.
- When benchmarking revisions, use isolated worktrees and identical long-run settings. Short runs are screening only.

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
