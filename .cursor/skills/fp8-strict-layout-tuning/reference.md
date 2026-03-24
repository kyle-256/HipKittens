# Reference

## Scope
- Repository: `HipKittens`
- Target area: `analysis/fp8_gemm/mi350x`
- Hardware assumption: `gfx950` / `MI350X`
- Primary goal: keep strict `no-preshuffle` while pushing `RRR` and `CRR` toward `RCR`

## Important File Map
- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`: main FP8 layout kernel and macro controls
- `analysis/fp8_gemm/mi350x/test_python.py`: benchmark and correctness harness, supports padded build dims
- `analysis/fp8_gemm/mi350x/tune_mnk_yaml.py`: strict shape sweep driver for `mnk.yaml`
- `include/ops/warp/memory/tile/shared_to_register.cuh`: FP8 shared-store support used by CRR experiments
- `include/types/shared/st_shape.cuh` and `include/types/shared/st.cuh`: shared-tile swizzle and padding definitions
- `/shared_nfs/kyle/triton_bench/mnk.yaml`: external shape list used for full tuning

## Important Constraints
- Strict `no-preshuffle` means no global-coordinate reinterpretation tricks for `RRR` or `CRR`.
- Keep `RRR_ROW_SHARED_TRANSPOSE=0` and `CRR_ROW_SHARED_TRANSPOSE=0` unless the user explicitly waives that rule.
- The bank-conflict fix relies on inter-subtile padding in the `v2` / `v2a` shared layouts; do not remove it casually.
- Sequential build-and-run loops are safer than concurrent sweeps because the Python extension binary is shared.

## Known Dead Ends
- `CRR_A_LDS_REENCODE=1`: compiled only after adding FP8 shared-store support, but performed very poorly because LDS usage and pressure exploded.
- `CRR_USE_V3_SWIZZLE=1`: screened and measured worse than baseline.
- Small mapping or explicit-SRD tuning alone: useful for screening, not enough to close the strict CRR gap.
- Old row-shared transpose paths: fast, but not valid under strict `no-preshuffle`.

## Benchmark Notes
- Quick screen: `FP8_WARMUP=50`, `FP8_ITERS=10`
- Stable comparison: `FP8_WARMUP=200`, `FP8_ITERS=50`
- Use `FP8_LAYOUTS=RCR,RRR,CRR` when comparing ratios.
- `test_python.py` accepts `FP8_BUILD_M`, `FP8_BUILD_N`, `FP8_BUILD_K`, `FP8_LAYOUTS`, `FP8_CHECK`, and `FP8_OUTPUT`.
- For arbitrary shapes, the benchmark harness now pads inputs and checks only valid slices.

## Promising Next Branches
- Reduced-M or `192x256`-style CRR to lower accumulator and A-register pressure.
- Producer-consumer only after checking that extra warps do not push the kernel over the VGPR cliff.
- Any structural branch should be screened on `8192x8192x8192` before a full `mnk.yaml` sweep.

## Artifact Policy
- Keep reusable source changes and reusable tuning YAML or scripts.
- Do not keep `.tmp_*.json`, `gpucore.*`, `pmc_*`, `.bak*`, ad-hoc profiling scripts, or one-off benchmark dumps in commits.
