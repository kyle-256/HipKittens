---
name: fp8-strict-layout-tuning
description: Tune HipKittens FP8, MXFP8, and upcoming MXFP4 strict-layout GEMM kernels on gfx950/MI350X. Use when working on analysis/fp8_gemm/mi350x RCR/RRR/CRR performance, preshuffle-quant scale packing, scale-pack scheduling, SNR, determinism, commit sweeps, or Primus-Turbo backend comparisons.
---
# FP8 Strict Layout Tuning

## When To Use
- User asks to debug or optimize `analysis/fp8_gemm/mi350x` FP8 GEMM.
- User asks to debug or optimize `analysis/fp8_gemm/mi350x` MXFP8 or MXFP4 GEMM.
- User mentions `RCR`, `RRR`, `CRR`, `preshuffle-quant`, `scale-pack`, `block-scale`, `bank conflict`, `SNR`, `deterministic`, `gfx950`, `MI350X`, or `Primus-Turbo`.
- Existing layout-kernel tuning knowledge should be reused instead of rediscovered.

## First Read
- For the current MI350X layout task, read `analysis/fp8_gemm/mi350x/README.md` before editing `kernel_mxfp8_layouts.cpp` or running `test_mxfp8_python.py`.
- Use `reference.md` for durable priors, older FP8 strict-layout rules, and known failure signatures.

## Hard Rules
- Formal FP8 acceptance runs use `HIP_VISIBLE_DEVICES=7`.
- Formal MXFP8 and MXFP4 task runs use `8192x8192x8192`, with smoke tests first and long runs second.
- Success means every requested layout passes numerical correctness, `SNR > 48 dB`, and determinism. Anything else is a failure.
- Keep layout handling native. No Python `.t().contiguous()` workaround. No host-side padding workaround for Primus integration.
- `preshuffle-quant` is valid for both `A_scale` and `B_scale`. `preshuffle-ab` is invalid unless the user explicitly relaxes the requirement.
- Treat `RRR_ROW_SHARED_TRANSPOSE=1` and `CRR_ROW_SHARED_TRANSPOSE=1` as invalid unless the user explicitly relaxes strict `no-preshuffle`.
- After every substantial kernel change: compile, run, then inspect bank conflict, MFMA utilization, and cache utilization before continuing to tune.
- Do not claim a win from short runs alone.
- If background GPU jobs are active, treat throughput as contaminated and rerun sequentially before claiming a win.
- Do not commit `.tmp_*.json`, `gpucore.*`, `pmc_*`, `.bak*`, generated ISA, benchmark dumps, or ad-hoc profiling scripts.
- Only create a git commit when the user explicitly asks.
- Avoid overlapping `make` and benchmark jobs that overwrite shared extension binaries such as `tk_fp8_layouts` or `tk_mxfp8_layouts`.

## Primary Files
- `analysis/fp8_gemm/mi350x/README.md`
- `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp`
- `analysis/fp8_gemm/mi350x/test_mxfp8_python.py`
- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
- `analysis/fp8_gemm/mi350x/test_python.py`
- `analysis/fp8_gemm/mi350x/tune_mnk_yaml.py`
- `include/ops/warp/memory/tile/shared_to_register.cuh`
- `primus_turbo/pytorch/kernels/gemm/gemm_fp8_impl.py`

## Required Commands
1. Build the FP8 strict-layout baseline from `analysis/fp8_gemm/mi350x`:

```bash
THUNDERKITTENS_ROOT=$(git rev-parse --show-toplevel) ROCM_PATH=/opt/rocm make -j4
```

2. Run an FP8 smoke test:

```bash
HIP_VISIBLE_DEVICES=7 FP8_WARMUP=1 FP8_ITERS=1 FP8_LAYOUTS=rcr,rrr,crr FP8_CHECK=1 FP8_DETERMINISM_RUNS=2 python3 test_python.py 256 256 128
```

3. Run the FP8 formal acceptance benchmark:

```bash
HIP_VISIBLE_DEVICES=7 FP8_WARMUP=50 FP8_ITERS=200 FP8_LAYOUTS=rcr,rrr,crr FP8_CHECK=1 FP8_DETERMINISM_RUNS=5 python3 test_python.py 8192 8192 8192
```

4. Build and smoke-test the current MXFP8 path:

```bash
THUNDERKITTENS_ROOT=$(git rev-parse --show-toplevel) CPPFLAGS='-DM_DIM=8192 -DN_DIM=8192 -DK_DIM=8192 -DMXFP8_RCR_EXACT_8WAVE_FAST_ENABLE=1' make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp
MXFP8_PRESHUFFLE_QUANT=1 MXFP8_LAYOUTS=rcr MXFP8_WARMUP=5 MXFP8_ITERS=10 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 python3 test_mxfp8_python.py 256 256 256
```

5. Run the current MXFP8 long benchmark:

```bash
MXFP8_PRESHUFFLE_QUANT=1 MXFP8_LAYOUTS=rcr MXFP8_WARMUP=50 MXFP8_ITERS=200 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 python3 test_mxfp8_python.py 8192 8192 8192
```

6. For Primus-Turbo benchmarking, compare the same shapes against `HIPBLASLT` and `TRITON` backends rather than looking at HipKittens in isolation.

## Debug Workflow
1. Read `analysis/fp8_gemm/mi350x/README.md` if the task is the current MXFP8/MXFP4 effort.
2. Identify the actual blocker first: correctness/SNR, determinism, absolute `RCR`, or `RRR/CRR` ratio.
3. For revision comparisons, use isolated worktrees and the same long-run settings for every candidate.
4. Change one kernel idea at a time. Do not mix loader, schedule, and waitcnt experiments in the same edit.
5. Rebuild, then run a smoke test. Only run the formal benchmark after the smoke test is clean.
6. If throughput moved, inspect bank conflict, MFMA utilization, cache utilization, and compile resource remarks (`VGPRs`, spills, occupancy, LDS) before making another tweak.
7. Keep only durable source changes and reusable tuning inputs. Remove one-off artifacts.

## Durable Debugging Priors
- The FP8 col-loader bug was real. Keep the `ds_read_b64_tr_b8` path in the corrected single-address form with early-clobber `=&v` outputs.
- Dynamic shapes must remain kernel-native: runtime `m/n/k`, runtime `bpr/bpc/ki`, fast interior kernel, scalar tail kernel, and `ki >= 2` guarding the fast kernel.
- `RRR` recovered through the dual-`B` schedule plus the fixed `cD` operand lifetime. Do not replace this with Python workarounds.
- `CRR` should stay on the strict deterministic path. `CRR_BATCHED_PAIR_MMA=1` is acceptable only if SNR and determinism still pass.
- `RCR > 3100 TFLOPS` on `8192x8192x8192` with `50/200/GPU7` remains a hard gate for the FP8 strict-layout task.
- `RRR` and `CRR` still need to stay at or above `95%` of `RCR` while keeping the success gate above.
- For the current MXFP8 PQ task, both `A_scale` and `B_scale` are preshuffled. Do not reason about only one side of the scale path.
- The current best known committed MXFP8 `RCR + PQ` revision is `768b60fa`. Revalidate if newer commits appear.
- `4de0a032` is a known invalid fast path: scale-pack buffer loads look fast but fail correctness and determinism.
- The current winning MXFP8 lever is earlier exact-8wave scale-pack scheduling, not shared scale cache, 4-wave PQ, or `phase1` asm rewrites.

## What Not To Do
- Do not accept Python transpose, host-side padding, or scalar row-loader fallbacks as final fixes.
- Do not start with random waitcnt or SRD tweaks before operand semantics are correct.
- Do not trust a speedup that adds spills, collapses occupancy, or only wins on a short run.
- Do not keep `buffer_load_dword` scale-pack experiments from `4de0a032` as if they were valid wins.
- Do not reopen shared scale-cache, 4-wave PQ, or `phase1` asm branches without a new reason and a clean benchmark plan.
- Do not keep temporary results or scratch scripts in the tree.

## Additional Reference
- For failure signatures, known-good directions, and integration rules, see [reference.md](reference.md)
