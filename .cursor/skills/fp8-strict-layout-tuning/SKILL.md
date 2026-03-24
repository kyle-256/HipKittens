---
name: fp8-strict-layout-tuning
description: Tune HipKittens FP8 layout GEMM kernels on gfx950/MI350X under strict no-preshuffle constraints. Use when optimizing or benchmarking analysis/fp8_gemm/mi350x RCR/RRR/CRR kernels, sweeping mnk.yaml shapes, comparing layout throughput, or cleaning tuning artifacts before commit.
---
# FP8 Strict Layout Tuning

## When To Use
- User asks to optimize `analysis/fp8_gemm/mi350x` FP8 GEMM.
- User mentions `RRR`, `CRR`, `RCR`, `mnk.yaml`, `gfx950`, `MI350X`, `bank conflict`, or `no-preshuffle`.
- Existing tuning work should be resumed instead of rediscovered.

## Non-Negotiable Constraints
- Default to `HIP_VISIBLE_DEVICES=0`.
- Treat `RRR_ROW_SHARED_TRANSPOSE=1` and `CRR_ROW_SHARED_TRANSPOSE=1` as invalid unless the user explicitly relaxes strict `no-preshuffle`.
- Keep `CRR_A_LDS_REENCODE=0` unless debugging that specific failed path.
- Do not commit `.tmp_*.json`, `gpucore.*`, `pmc_*`, `.bak*`, benchmark dumps, or ad-hoc profiling scripts.
- Only create a git commit when the user explicitly asks.
- Avoid parallel `make` plus benchmark workflows that overwrite the same `tk_fp8_layouts` extension binary.

## Primary Files
- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`
- `analysis/fp8_gemm/mi350x/test_python.py`
- `analysis/fp8_gemm/mi350x/tune_mnk_yaml.py`
- `include/ops/warp/memory/tile/shared_to_register.cuh`

## Quick Workflow
1. Build from `analysis/fp8_gemm/mi350x`:

```bash
export THUNDERKITTENS_ROOT="/shared_nfs/kyle/HipKittens"
HIP_VISIBLE_DEVICES=0 make -j4
```

2. Run a quick correctness and throughput smoke test:

```bash
HIP_VISIBLE_DEVICES=0 FP8_WARMUP=50 FP8_ITERS=10 FP8_LAYOUTS=RCR,RRR,CRR FP8_CHECK=1 python3 test_python.py 8192 8192 8192
```

3. Run a stable comparison before claiming a win:

```bash
HIP_VISIBLE_DEVICES=0 FP8_WARMUP=200 FP8_ITERS=50 FP8_LAYOUTS=RCR,RRR,CRR FP8_CHECK=0 python3 test_python.py 8192 8192 8192
```

4. For non-native shapes, use `FP8_BUILD_M`, `FP8_BUILD_N`, and `FP8_BUILD_K`, or let `tune_mnk_yaml.py` compute padded build sizes.

## Shape Tuning
Use the strict tuning driver:

```bash
HIP_VISIBLE_DEVICES=0 python3 tune_mnk_yaml.py --mnk-file /shared_nfs/kyle/triton_bench/mnk.yaml --output hipk_tuned_mnk.yaml
```

- The committed candidate set is strict `no-preshuffle`.
- If you edit candidate macros, keep `RRR_ROW_SHARED_TRANSPOSE=0` and `CRR_ROW_SHARED_TRANSPOSE=0` unless the user explicitly allows semantic transpose tricks.
- After any structural kernel change, rerun one representative `8192x8192x8192` benchmark before launching a full sweep.

## How To Judge Progress
- Compare `RRR` and `CRR` as `% of RCR`, not only absolute TFLOPS.
- Use long runs for regressions or wins; short runs are only for screening.
- Watch compile remarks for `VGPRs`, `VGPRs Spill`, `ScratchSize`, `LDS Size`, and occupancy.
- If a change increases spills or pushes LDS too close to the 160 KB CU budget, treat it as suspect even if a short run looks faster.

## How To Report Results
For each meaningful experiment, record:
- Macro set or code branch.
- Quick result and long result.
- Compile resource remarks if the structure changed.
- Whether the path stays strict `no-preshuffle`.
- Whether the change is promising enough to keep, revert, or move into `tune_mnk_yaml.py`.

## Current Priors
- Low-level `waitcnt`, mapping, or SRD micro-tweaks are low yield by themselves.
- `CRR_USE_V3_SWIZZLE=1` and `CRR_A_LDS_REENCODE=1` are known poor directions.
- The highest-value remaining branches are reduced-M or `192x256`-style CRR kernels, then producer-consumer if register pressure stays sane.

## Cleanup Before Commit
Use targeted cleanup and keep only source or reusable tuning inputs:

```bash
git clean -fd -- analysis/fp8_gemm/mi350x/.tmp_* analysis/fp8_gemm/mi350x/gpucore.* analysis/fp8_gemm/mi350x/fp8_layout_results_*.json analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp.bak* analysis/fp8_gemm/mi350x/pmc_crr analysis/fp8_gemm/mi350x/pmc_rrr analysis/fp8_gemm/mi350x/tmp_fp8_mnk_result.json
```

## Additional Reference
- Detailed context and known dead ends: [reference.md](reference.md)
