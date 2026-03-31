# Reference

## Scope
- Repository: `HipKittens`
- Target area: `analysis/fp8_gemm/mi350x`
- Hardware assumption: `gfx950` / `MI350X`
- Primary goal: keep strict native layouts while pushing `RCR`, `RRR`, and `CRR` to the required performance gates without losing correctness

## Important File Map
- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`: main FP8 layout kernel and macro controls
- `analysis/fp8_gemm/mi350x/test_python.py`: benchmark and correctness harness with mandatory SNR and determinism gates
- `analysis/fp8_gemm/mi350x/tune_mnk_yaml.py`: strict shape sweep driver for `mnk.yaml`
- `include/ops/warp/memory/tile/shared_to_register.cuh`: FP8 shared-store support used by CRR experiments
- `include/types/shared/st_shape.cuh` and `include/types/shared/st.cuh`: shared-tile swizzle and padding definitions
- `primus_turbo/pytorch/kernels/gemm/gemm_fp8_impl.py`: Primus-Turbo integration point for the HipKittens backend

## Acceptance Gate
- Formal benchmark command:

```bash
HIP_VISIBLE_DEVICES=7 FP8_LAYOUTS=rcr,rrr,crr FP8_WARMUP=50 FP8_ITERS=200 FP8_CHECK=1 FP8_DETERMINISM_RUNS=5 python3 test_python.py 8192 8192 8192
```

- Each requested layout must satisfy all of:
  - numerical correctness pass
  - `SNR > 48 dB`
  - deterministic output across repeated runs
- `test_python.py` exits with code `1` if the success gate is not met.
- Performance gates still matter:
  - `RCR > 3100 TFLOPS`
  - `RRR >= 95% of RCR`
  - `CRR >= 95% of RCR`

## Important Constraints
- Strict native-layout support means no Python `.t().contiguous()` workaround for `RRR` or `CRR`.
- Primus integration must not rely on host-side padding to paper over unsupported shapes.
- Keep `RRR_ROW_SHARED_TRANSPOSE=0` and `CRR_ROW_SHARED_TRANSPOSE=0` unless the user explicitly waives strict `no-preshuffle`.
- The bank-conflict fix relies on shared-layout swizzle and padding behavior; do not remove those pieces casually.
- Sequential build-and-run loops are safer than concurrent sweeps because the Python extension binary is shared.

## Failure Signatures And First Checks
- `CRR` loses SNR or becomes non-deterministic after loader edits:
  - re-check `ds_read_b64_tr_b8`
  - keep the corrected single-address plus immediate-offset form
  - keep early-clobber `=&v` on asm outputs
- Irregular shapes or small `K` fail while large aligned shapes pass:
  - check the dynamic-shape path
  - verify runtime `m/n/k`, `bpr/bpc/ki`
  - verify the `ki >= 2` guard before entering the fast kernel
  - make sure the scalar tail kernel handles the remainder
- `RRR` becomes fast but numerically wrong, often in the bottom-right output region:
  - re-check the dual-`B` schedule
  - verify `cD` still consumes the correct `A` fragment
  - watch for register lifetime mistakes around prefetch overlap
- Throughput improves only after replacing the fast loader with scalar row-load logic:
  - treat this as a debug aid, not an acceptable final solution
- `RRR` or `CRR` ratios move a lot while absolute TFLOPS barely changes:
  - compare both absolute TFLOPS and `% of RCR`
  - large ratio swings can come from `RCR` moving
- A candidate looks faster but compile remarks show much higher `VGPRs`, spills, or LDS pressure:
  - treat the speedup as suspect until the formal run confirms it

## Debug Workflow
1. Decide whether the current blocker is correctness, determinism, `RCR`, or `RRR/CRR` relative performance.
2. Make one meaningful kernel change at a time.
3. Rebuild from `analysis/fp8_gemm/mi350x`.
4. Run a small smoke test first.
5. If the smoke test is clean, run the formal `GPU7 / 8192^3 / 50 / 200` benchmark.
6. After any meaningful throughput movement, inspect:
   - bank conflict
   - MFMA utilization
   - cache utilization
   - compile resource remarks (`VGPRs`, spills, occupancy, LDS)
7. Only keep a branch if it survives both the success gate and the performance gate.

## Durable Findings
- Loader correctness is foundational:
  - the FP8 col-loader bug was real
  - the corrected `ds_read_b64_tr_b8` form plus early-clobber fixed `CRR` non-determinism and numerical drift
- Dynamic shapes must be kernel-native:
  - runtime `m/n/k`
  - runtime `bpr/bpc/ki`
  - fast aligned interior path
  - scalar tail path for edge tiles and `K` tail
  - `ki >= 2` before entering the fast path
- `RRR` fast path:
  - preserve the dual-`B` schedule
  - preserve the fixed operand lifetime so `cD` sees the correct `A` fragment
- `CRR` fast path:
  - strict deterministic loader path is the safe baseline
  - `CRR_BATCHED_PAIR_MMA=1` can help, but only with validated wait/prefetch settings and full SNR/determinism checks
- Bank-conflict evaluation rule:
  - real progress means a hardware-friendly fast path that also stays correct
  - do not accept scalar-gather or other obviously throughput-killing paths as final answers

## Known Dead Ends
- `CRR_A_LDS_REENCODE=1`: compiled after adding FP8 shared-store support, but performance collapsed due to LDS pressure.
- `CRR_USE_V3_SWIZZLE=1`: measured worse than the strict baseline.
- Old row-shared transpose paths: fast, but not valid under strict `no-preshuffle`.
- Full-tile reinterpret-cast aliasing for `RRR`: raised `VGPRs` too much and regressed the useful path.
- Pure waitcnt or explicit-SRD micro-tweaks without fixing operand semantics first: low-yield and misleading.

## Primus-Turbo Integration Rules
- Valid native layouts are `RCR`, `RRR`, and `CRR`.
- Backend selection goes through `PRIMUS_TURBO_GEMM_BACKEND=HIPKITTENS`.
- Tensorwise scale should stay fused as `a_scale_inv * b_scale_inv`.
- Do not add Python transpose or host-padding workarounds to make HipKittens appear to support a shape.
- When reporting progress, compare against `HIPBLASLT` and `TRITON` on the same benchmark shape and settings.

## Artifact Policy
- Keep reusable source changes and reusable tuning YAML or scripts.
- Do not keep `.tmp_*.json`, `gpucore.*`, `pmc_*`, `.bak*`, generated ISA files, ad-hoc profiling scripts, or one-off benchmark dumps in commits.
