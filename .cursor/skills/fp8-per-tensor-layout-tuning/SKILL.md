---
name: fp8-per-tensor-layout-tuning
description: Tune HipKittens FP8 per-tensor strict-layout GEMM kernels on gfx950/MI350X. Use when working on analysis/fp8_gemm/mi350x RCR/RRR/CRR FP8 performance, bank conflicts, determinism, CRR/RRR loaders, or Primus-Turbo backend comparisons.
---
# FP8 Per-Tensor Strict Layout Tuning

## When To Use
- User asks to debug or optimize `analysis/fp8_gemm/mi350x` **FP8 per-tensor** GEMM.
- User mentions FP8 `RCR`, `RRR`, `CRR`, `bank conflict`, `SNR`, `deterministic`, `gfx950`, `MI350X`, or `Primus-Turbo` in the context of per-tensor (non-microscaling) FP8.
- Existing layout-kernel tuning knowledge should be reused instead of rediscovered.

## First Read
- Read `analysis/fp8_gemm/mi350x/README.md` for the current task state.
- For MXFP8 work, use the `mxfp8-layout-tuning` skill instead.

## Hard Rules
- Formal FP8 acceptance runs use `HIP_VISIBLE_DEVICES=7`.
- Success means every requested layout passes numerical correctness, `SNR > 48 dB`, and determinism.
- Keep layout handling native. No Python `.t().contiguous()` workaround. No host-side padding workaround.
- After every substantial kernel change: compile, run, then inspect bank conflict, MFMA utilization, and cache utilization before continuing.
- Do not claim a win from short runs alone.
- Do not commit `*.s`, `*.so`, `*_layout_results_*.json`, `gpucore.*`, `pmc_*/`, `.bak*`, generated ISA, or scratch profiling scripts (`.gitignore` enforces most of this — don't weaken it).
- Only create a git commit when the user explicitly asks.

## Primary Files
- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
- `analysis/fp8_gemm/mi350x/test_python.py`
- `analysis/fp8_gemm/mi350x/rcr_exact_4wave_fastpath.inc`
- `analysis/fp8_gemm/mi350x/crr_exact_8wave_fastpath.inc`
- `analysis/fp8_gemm/mi350x/crr_exact_4wave_fastpath.inc`
- `analysis/fp8_gemm/mi350x/crr_exact_8wave_double_pump_fastpath.inc`
- `include/ops/warp/memory/tile/shared_to_register.cuh`
- `primus_turbo/pytorch/kernels/gemm/gemm_fp8_impl.py`

## Required Commands
1. Build FP8 strict-layout baseline from `analysis/fp8_gemm/mi350x`:

```bash
THUNDERKITTENS_ROOT=$(git rev-parse --show-toplevel) ROCM_PATH=/opt/rocm make -j4
```

2. Smoke test:

```bash
HIP_VISIBLE_DEVICES=7 FP8_WARMUP=1 FP8_ITERS=1 FP8_LAYOUTS=rcr,rrr,crr FP8_CHECK=1 FP8_DETERMINISM_RUNS=2 python3 test_python.py 256 256 128
```

3. Formal acceptance benchmark:

```bash
HIP_VISIBLE_DEVICES=7 FP8_WARMUP=50 FP8_ITERS=200 FP8_LAYOUTS=rcr,rrr,crr FP8_CHECK=1 FP8_DETERMINISM_RUNS=5 python3 test_python.py 8192 8192 8192
```

## Performance Gates
- `RCR > 3100 TFLOPS`
- `RRR >= 95% of RCR`
- `CRR >= 95% of RCR`

## Current Status
- FP8 RCR ~3335 TFLOPS under **batch timing** (`warmup=500, iters=100`, no per-iter `torch.cuda.synchronize()`).
- FP8 RCR ~3070.93 TFLOPS under the harness's default **per-iteration sync + `output.zero_()`** protocol (`test_python.py` as-is).
- All three layouts (RCR, RRR, CRR) pass SNR and determinism gates.
- MXFP8 RCR currently trails at 2897.66 TFLOPS under the same per-iter protocol (see `mxfp8-layout-tuning` skill).

### Protocol lock
Don't mix batch and per-iter numbers in one comparison. The per-iter protocol is the one the MXFP8 harness uses, so when comparing MXFP8 vs FP8 use the per-iter numbers.

## Durable Debugging Priors
- The FP8 col-loader bug was real. Keep the `ds_read_b64_tr_b8` path in the corrected single-address form with early-clobber `=&v` outputs.
- Dynamic shapes must remain kernel-native: runtime `m/n/k`, runtime `bpr/bpc/ki`, fast interior kernel, scalar tail kernel, and `ki >= 2` guarding the fast kernel.
- `RRR` recovered through the dual-`B` schedule plus the fixed `cD` operand lifetime.
- `CRR` should stay on the strict deterministic path.
- Keep `RRR_ROW_SHARED_TRANSPOSE=0` and `CRR_ROW_SHARED_TRANSPOSE=0` unless the user explicitly waives strict `no-preshuffle`.

## Known Dead Ends
- `CRR_A_LDS_REENCODE=1`: LDS pressure collapsed performance.
- `CRR_USE_V3_SWIZZLE=1`: worse than the strict baseline.
- Old row-shared transpose paths: fast but not valid under strict `no-preshuffle`.
- Full-tile reinterpret-cast aliasing for `RRR`: raised VGPRs too much.

## Primus-Turbo Integration
- Valid native layouts are `RCR`, `RRR`, and `CRR`.
- Backend selection via `PRIMUS_TURBO_GEMM_BACKEND=HIPKITTENS`.
- Tensorwise scale should stay fused as `a_scale_inv * b_scale_inv`.
- Compare against `HIPBLASLT` and `TRITON` on the same benchmark shape and settings.

## Commit-Time Workflow
Changes that affect FP8 performance must also refresh the running state:
1. `TODO.md` (if the work lives in an MXFP8 parity sprint, update MXFP8 baseline + FP8 regression row)
2. `agent_prompt.md` (if any rule / runbook detail changed)
3. This SKILL (only for durable findings or new dead-ends)
4. The code change itself

Commit message template:
```
FP8 <layout> <what changed>: <TFLOPS> TFLOPS (<+delta%>)

SNR: XX.XX dB, determinism: PASS (N runs)
VGPR: X, AGPR: Y, spills: Z, LDS: W KB
```
