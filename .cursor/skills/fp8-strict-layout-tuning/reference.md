# Reference

## Scope
- Repository: `HipKittens`
- Target area: `analysis/fp8_gemm/mi350x`
- Hardware assumption: `gfx950` / `MI350X`
- Current task handoff: `analysis/fp8_gemm/mi350x/README.md`
- Primary goal: keep strict native layouts while pushing FP8, MXFP8, and later MXFP4 `RCR`, `RRR`, and `CRR` to the required performance gates without losing correctness

## Important File Map
- `analysis/fp8_gemm/mi350x/README.md`: current task summary, constraints, commit sweep, and next priorities
- `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp`: main MXFP8 layout kernel and macro controls
- `analysis/fp8_gemm/mi350x/test_mxfp8_python.py`: MXFP8 benchmark and correctness harness
- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`: main FP8 strict-layout kernel and macro controls
- `analysis/fp8_gemm/mi350x/test_python.py`: FP8 benchmark and correctness harness with mandatory SNR and determinism gates
- `analysis/fp8_gemm/mi350x/tune_mnk_yaml.py`: strict shape sweep driver for `mnk.yaml`
- `include/ops/warp/memory/tile/shared_to_register.cuh`: FP8 shared-store support used by CRR experiments
- `include/types/shared/st_shape.cuh` and `include/types/shared/st.cuh`: shared-tile swizzle and padding definitions
- `primus_turbo/pytorch/kernels/gemm/gemm_fp8_impl.py`: Primus-Turbo integration point for the HipKittens backend

## Current MXFP8 And MXFP4 Constraints
- Required layouts: `RCR`, `RRR`, `CRR`
- Formal benchmark shape: `8192x8192x8192`
- `preshuffle-quant` is valid for scale tensors only
- Both `A_scale` and `B_scale` may be preshuffled or prepacked
- `preshuffle-ab` is not allowed unless the user explicitly changes the requirement
- `mxfp8` should approach `fp8` throughput, excluding quantization overhead
- `mxfp4` target throughput is `2x` the validated `mxfp8` throughput

## Acceptance Gates
### FP8 strict-layout gate
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

### MXFP8 current-task gate
```bash
MXFP8_PRESHUFFLE_QUANT=1 MXFP8_LAYOUTS=rcr MXFP8_WARMUP=50 MXFP8_ITERS=200 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 python3 test_mxfp8_python.py 8192 8192 8192
```

- Use smoke tests before long runs.
- Long-run TFLOPS only counts after a sequential rerun under the same settings.
- If background GPU jobs are active, treat the number as contaminated and rerun later.

## Current Validated Status
- `MXFP8 RCR exact + preshuffle-quant` is the active high-value path.
- In the current PQ path, both `A_scale` and `B_scale` are preshuffled.
- **Current best**: `buffer_load + SGPR SRD + KPAIR_LOOP` at **3031 TFLOPS** (batch timing), 3 spills, SNR 49.60 dB, determinism PASS.
- The pre-KPAIR_LOOP commit sweep winner was `768b60fa` at `2648.81 TFLOPS`.
- `4de0a032` reached `2620.01 TFLOPS` but is invalid: correctness and determinism fail on `8192`.
- When the user asks which revision is best, compare revisions in isolated worktrees with identical long-run settings.
- Use batch timing (warmup 100, measure 200 contiguous iters) to avoid GPU DVFS clock drops.

## Important Constraints
- Strict native-layout support means no Python `.t().contiguous()` workaround for `RRR` or `CRR`.
- Primus integration must not rely on host-side padding to paper over unsupported shapes.
- Keep `RRR_ROW_SHARED_TRANSPOSE=0` and `CRR_ROW_SHARED_TRANSPOSE=0` unless the user explicitly waives strict `no-preshuffle`.
- The bank-conflict fix relies on shared-layout swizzle and padding behavior; do not remove those pieces casually.
- Sequential build-and-run loops are safer than concurrent sweeps because the Python extension binaries are shared.

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
- `MXFP8 RCR + PQ` looks fast but becomes wrong at large `K`:
  - re-check the scale-pack load helper first
  - do not trust `buffer_load_dword` scale-pack changes without a full `8192` correctness rerun
  - confirm both `A_scale` and `B_scale` are still handled under the same preshuffle contract
- A candidate looks faster but compile remarks show much higher `VGPRs`, spills, or LDS pressure:
  - treat the speedup as suspect until the formal run confirms it
- Throughput improves only in short runs:
  - rerun the long benchmark sequentially before keeping the branch

## Debug Workflow
1. Read `analysis/fp8_gemm/mi350x/README.md` if the task is the current MXFP8/MXFP4 effort.
2. Decide whether the current blocker is correctness, determinism, absolute `RCR`, or `RRR/CRR` relative performance.
3. Make one meaningful kernel change at a time.
4. Rebuild from `analysis/fp8_gemm/mi350x`.
5. Run a small smoke test first.
6. If the smoke test is clean, run the formal long benchmark.
7. After any meaningful throughput movement, inspect:
   - bank conflict
   - MFMA utilization
   - cache utilization
   - compile resource remarks (`VGPRs`, spills, occupancy, LDS)
8. Only keep a branch if it survives both the success gate and the relevant performance gate.

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
- MXFP8 current-task priors:
  - both `A_scale` and `B_scale` are preshuffled in the current PQ path
  - the current winning lever is earlier exact-8wave `ensure_scale_packs(k_pair)` scheduling
  - this is more valuable than reopening shared scale cache, 4-wave PQ, or `phase1` asm branches

## Known Dead Ends
- `CRR_A_LDS_REENCODE=1`: compiled after adding FP8 shared-store support, but performance collapsed due to LDS pressure.
- `CRR_USE_V3_SWIZZLE=1`: measured worse than the strict baseline.
- Old row-shared transpose paths: fast, but not valid under strict `no-preshuffle`.
- Full-tile reinterpret-cast aliasing for `RRR`: raised `VGPRs` too much and regressed the useful path.
- Pure waitcnt or explicit-SRD micro-tweaks without fixing operand semantics first: low-yield and misleading.
- `4de0a032` scale-pack `buffer_load_dword` path: invalid at large `K`.
- Shared/LDS scale cache experiments for MXFP8 PQ: correctness problems and no validated win.
- 4-wave exact PQ path: correct, but slower than the exact 8-wave path.
- `phase1` `op_sel_hi` asm variants and scalar phase-pack caching: did not survive long-run validation.

## Primus-Turbo Integration Rules
- Valid native layouts are `RCR`, `RRR`, and `CRR`.
- Backend selection goes through `PRIMUS_TURBO_GEMM_BACKEND=HIPKITTENS`.
- Tensorwise scale should stay fused as `a_scale_inv * b_scale_inv`.
- Do not add Python transpose or host-padding workarounds to make HipKittens appear to support a shape.
- When reporting progress, compare against `HIPBLASLT` and `TRITON` on the same benchmark shape and settings.

## Artifact Policy
- Keep reusable source changes and reusable tuning YAML or scripts.
- Do not keep `.tmp_*.json`, `gpucore.*`, `pmc_*`, `.bak*`, generated ISA files, ad-hoc profiling scripts, or one-off benchmark dumps in commits.
