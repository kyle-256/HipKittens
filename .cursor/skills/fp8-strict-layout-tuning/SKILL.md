---
name: fp8-strict-layout-tuning
description: Debug and optimize HipKittens FP8 layout GEMM kernels on gfx950/MI350X under strict native-layout constraints. Use when working on analysis/fp8_gemm/mi350x RCR/RRR/CRR performance, SNR, determinism, bank conflicts, dynamic-shape support, or Primus-Turbo HipKittens backend integration.
---
# FP8 Strict Layout Tuning

## When To Use
- User asks to debug or optimize `analysis/fp8_gemm/mi350x` FP8 GEMM.
- User mentions `RCR`, `RRR`, `CRR`, `bank conflict`, `SNR`, `deterministic`, `GPU7`, `gfx950`, `MI350X`, or `Primus-Turbo`.
- Existing layout-kernel tuning knowledge should be reused instead of rediscovered.

## Hard Rules
- Formal acceptance runs use `HIP_VISIBLE_DEVICES=7`.
- Success means every requested layout passes numerical correctness, `SNR > 48 dB`, and determinism. Anything else is a failure.
- Keep layout handling native. No Python `.t().contiguous()` workaround. No host-side padding workaround for Primus integration.
- Treat `RRR_ROW_SHARED_TRANSPOSE=1` and `CRR_ROW_SHARED_TRANSPOSE=1` as invalid unless the user explicitly relaxes strict `no-preshuffle`.
- After every substantial kernel change: compile, run, then inspect bank conflict, MFMA utilization, and cache utilization before continuing to tune.
- Do not claim a win from short runs alone.
- Do not commit `.tmp_*.json`, `gpucore.*`, `pmc_*`, `.bak*`, generated ISA, benchmark dumps, or ad-hoc profiling scripts.
- Only create a git commit when the user explicitly asks.
- Avoid overlapping `make` and benchmark jobs that overwrite the shared `tk_fp8_layouts` extension binary.

## Primary Files
- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
- `analysis/fp8_gemm/mi350x/test_python.py`
- `analysis/fp8_gemm/mi350x/tune_mnk_yaml.py`
- `include/ops/warp/memory/tile/shared_to_register.cuh`
- `primus_turbo/pytorch/kernels/gemm/gemm_fp8_impl.py`

## Required Commands
1. Build from `analysis/fp8_gemm/mi350x`:

```bash
THUNDERKITTENS_ROOT=/workspace/code/HipKittens ROCM_PATH=/opt/rocm make -j4
```

2. Run a quick smoke test:

```bash
HIP_VISIBLE_DEVICES=7 FP8_WARMUP=1 FP8_ITERS=1 FP8_LAYOUTS=rcr,rrr,crr FP8_CHECK=1 FP8_DETERMINISM_RUNS=2 python3 test_python.py 256 256 128
```

3. Run the formal acceptance benchmark:

```bash
HIP_VISIBLE_DEVICES=7 FP8_WARMUP=50 FP8_ITERS=200 FP8_LAYOUTS=rcr,rrr,crr FP8_CHECK=1 FP8_DETERMINISM_RUNS=5 python3 test_python.py 8192 8192 8192
```

4. For Primus-Turbo benchmarking, compare the same shapes against `HIPBLASLT` and `TRITON` backends rather than looking at HipKittens in isolation.

## Debug Workflow
1. Identify the actual blocker first: correctness/SNR, determinism, absolute `RCR`, or `RRR/CRR` ratio.
2. Change one kernel idea at a time. Do not mix loader, schedule, and waitcnt experiments in the same edit.
3. Rebuild, then run a smoke test. Only run the formal `8192^3 / 50 / 200 / GPU7` benchmark after the smoke test is clean.
4. If throughput moved, inspect bank conflict, MFMA utilization, cache utilization, and compile resource remarks (`VGPRs`, spills, occupancy, LDS) before making another tweak.
5. Keep only durable source changes and reusable tuning inputs. Remove one-off artifacts.

## Durable Debugging Priors
- The FP8 col-loader bug was real. Keep the `ds_read_b64_tr_b8` path in the corrected single-address form with early-clobber `=&v` outputs.
- Dynamic shapes must remain kernel-native: runtime `m/n/k`, runtime `bpr/bpc/ki`, fast interior kernel, scalar tail kernel, and `ki >= 2` guarding the fast kernel.
- `RRR` recovered through the dual-`B` schedule plus the fixed `cD` operand lifetime. Do not replace this with Python workarounds.
- `CRR` should stay on the strict deterministic path. `CRR_BATCHED_PAIR_MMA=1` is acceptable only if SNR and determinism still pass.
- `RCR > 3100 TFLOPS` on `8192x8192x8192` with `50/200/GPU7` remains a hard gate.
- `RRR` and `CRR` still need to stay at or above `95%` of `RCR` while keeping the success gate above.

## What Not To Do
- Do not accept Python transpose, host-side padding, or scalar row-loader fallbacks as final fixes.
- Do not start with random waitcnt or SRD tweaks before operand semantics are correct.
- Do not trust a speedup that adds spills, collapses occupancy, or only wins on a short run.
- Do not keep temporary results or scratch scripts in the tree.

## Additional Reference
- For failure signatures, known-good directions, and integration rules, see [reference.md](reference.md)
