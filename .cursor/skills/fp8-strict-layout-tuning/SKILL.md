---
name: fp8-strict-layout-tuning
description: Debug and optimize HipKittens FP8 layout GEMM kernels on gfx950/MI350X under strict native-layout constraints. Use when working on analysis/fp8_gemm/mi350x RCR/RRR/CRR performance, SNR, determinism, bank conflicts, dynamic-shape support, or Primus-Turbo HipKittens backend integration.
---
# FP8 Strict Layout Tuning

## When To Use
- User asks to debug or optimize `analysis/fp8_gemm/mi350x` FP8 GEMM.
- User mentions `RCR`, `RRR`, `CRR`, `bank conflict`, `SNR`, `deterministic`, `gfx950`, `MI350X`, or `Primus-Turbo`.
- Existing layout-kernel tuning knowledge should be reused instead of rediscovered.

## Hard Rules
- **NO JIT**: only a single compiled `tk_fp8_layouts.so` from
  `make` is acceptable. Do not create `.jit_cache/` or pass
  `-DM_DIM/-DN_DIM/-DK_DIM` compile flags.
- Success means every requested layout passes numerical correctness,
  `SNR > 48 dB`, and determinism. Anything else is a failure.
- Keep layout handling native. No Python `.t().contiguous()` workaround.
  No host-side padding workaround for Primus integration.
- Treat `RRR_ROW_SHARED_TRANSPOSE=1` and `CRR_ROW_SHARED_TRANSPOSE=1` as invalid unless the user explicitly relaxes strict `no-preshuffle`.
- After every substantial kernel change: compile, run, then inspect bank conflict, MFMA utilization, and cache utilization before continuing to tune.
- Do not claim a win from short runs alone.
- Do not commit `.tmp_*.json`, `gpucore.*`, `pmc_*`, `.bak*`, generated ISA, benchmark dumps, or ad-hoc profiling scripts.
- Only create a git commit when the user explicitly asks.
- Avoid overlapping `make` and benchmark jobs that overwrite the shared `tk_fp8_layouts` extension binary.

## Primary Files
- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` (single source)
- `analysis/fp8_gemm/mi350x/rcr_4wave_dynamic.inc` (4-wave dynamic path)
- `analysis/fp8_gemm/mi350x/test_python.py`
- `analysis/fp8_gemm/mi350x/test_fp8_snr.py`
- `analysis/fp8_gemm/mi350x/autotune.py`
- `analysis/fp8_gemm/mi350x/bench_vs_hipblaslt.py`
- `include/ops/warp/memory/tile/shared_to_register.cuh`
- `primus_turbo/pytorch/kernels/gemm/gemm_fp8_impl.py`

## Required Commands
1. Build from `analysis/fp8_gemm/mi350x`:

```bash
THUNDERKITTENS_ROOT=/workspace/code/Hipkittens_per_tensor ROCM_PATH=/opt/rocm make -j4
```

2. Run a quick smoke test:

```bash
HIP_VISIBLE_DEVICES=0 FP8_WARMUP=1 FP8_ITERS=1 FP8_LAYOUTS=rcr,rrr,crr FP8_CHECK=1 FP8_DETERMINISM_RUNS=2 python3 test_python.py 256 256 128
```

3. Run the formal acceptance benchmark (48 shapes × 3 layouts):

```bash
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python3 bench_vs_hipblaslt.py --mode full --warmup 20 --iters 50 -o bench_no_jit_final.json
```

4. For Primus-Turbo benchmarking, compare the same shapes against `HIPBLASLT` and `TRITON` backends rather than looking at HipKittens in isolation.

## Current Acceptance State (2026-04-17, post-P8)

Single `tk_fp8_layouts.so`:
- **RCR geo-mean ≥ 1.00x vs hipBLASLt** — current 0.996x (within ±1pp
  noise of 1.00x; 1.005x reported in P7), 21/56 wins
- **RRR ≥ 95% of RCR** — current 1.530x hipBLASLt
- **CRR ≥ 95% of RCR** — current 1.967x hipBLASLt
- SNR min 49.6 dB; determinism PASS across all configs.
- P8: `RCR_TWO_TILE_MID_VMCNT 4 → 6` landed (the P7 commit message
  claimed this but the file shipped at 4).

## Debug Workflow
1. Identify the actual blocker first: correctness/SNR, determinism, absolute `RCR`, or `RRR/CRR` ratio.
2. Change one kernel idea at a time. Do not mix loader, schedule, and waitcnt experiments in the same edit.
3. Rebuild, then run a smoke test. Only run the formal full benchmark after the smoke test is clean.
4. If throughput moved, inspect bank conflict, MFMA utilization, cache utilization, and compile resource remarks (`VGPRs`, spills, occupancy, LDS) before making another tweak.
5. Keep only durable source changes and reusable tuning inputs. Remove one-off artifacts.

## Durable Debugging Priors
- The FP8 col-loader bug was real. Keep the `ds_read_b64_tr_b8` path in the corrected single-address form with early-clobber `=&v` outputs.
- Dynamic shapes must remain kernel-native: runtime `m/n/k`, runtime `bpr/bpc/ki`, fast interior kernel, scalar tail kernel, and `ki >= 2` guarding the fast kernel.
- `RRR` recovered through the dual-`B` schedule plus the fixed `cD` operand lifetime. Do not replace this with Python workarounds.
- `CRR` should stay on the strict deterministic path.
- `RCR_TWO_TILE_MIN_KI=28` (not 64) is now default — enables the two-tile schedule on K=3584 shapes.
- `RCR_TWO_TILE_MID_VMCNT=6` is the tuned default (was 4).
- Runtime `group_m` autotune picks gm ∈ {1,2,4,8,16,32} per shape, cached in `.autotune_cache.json`.

## hipBLASLt Reference Call

```python
# import order matters — this registers the torch op
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import GEMMFP8HipBLASLtBackend

hlt = torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm_fp8
# layout map: rcr=(F,T), rrr=(F,F), crr=(T,F)
C = hlt(A, scale_a, B, scale_b, torch.bfloat16, trans_a, trans_b, False, "TENSORWISE")
```

## What Not To Do
- Do not re-introduce JIT per-shape compilation.
- Do not accept Python transpose, host-side padding, or scalar row-loader fallbacks as final fixes.
- Do not start with random waitcnt or SRD tweaks before operand semantics are correct.
- Do not trust a speedup that adds spills, collapses occupancy, or only wins on a short run.
- Do not keep temporary results or scratch scripts in the tree.

## Additional Reference
- For failure signatures, known-good directions, and integration rules, see [reference.md](reference.md)
