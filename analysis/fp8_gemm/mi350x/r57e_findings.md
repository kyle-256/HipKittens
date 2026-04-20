# R57 Dev E: CRR Double-Buffer Interleaved VMEM Pipeline

## Verdict: REFUTED-EMPIRICAL-LLVM-RESCHEDULES

Porting Dev D's interleaved-load pipeline pattern from the single-buffer CRR path to
the production double-buffer CRR path produces **byte-identical ISA** across all 3
variants. The AMDGPU compiler completely ignores source-level global_load reordering
and reschedules all loads back to identical positions. Performance is within noise band
(max delta -0.24%) across all variants and both shapes.

This is the same class of refutation as R55A on the RCR path: without `sched_barrier`
to fence the scheduler, source-level VMEM placement is meaningless. And from R57B, we
know that adding `sched_barrier` to CRR breaks determinism (10/10 det FAIL).

## What was done

Added `MXFP8_CRR_DB_INTERLEAVE_R57E` macro guard (default OFF) with 3 variants for
reordering global_load instructions between MMA pairs in the production double-buffer
CRR K-loop (`kernel_mxfp8_layouts.cpp` lines 5449-5540):

- **V1** (=1): Move `global_load_b(Bs[tic][1])` from second half to between cA/cB and
  cC/cD MMA pairs. Removes it from the 3-load batch in the second half.
- **V2** (=2): Move `global_load_b(Bs[tic][0])` from end-of-iteration to between
  cA/cB and cC/cD. Removes trailing load entirely.
- **V3** (=3): Move both `global_load_b` calls to between cA/cB and cC/cD.

All variants preserve the tic/toc double-buffer structure, barrier positions, and
vmcnt values. No `sched_barrier` added (following Dev D's proven structural approach).

## ISA analysis

CRR V2 kernel (`crr_exact_8wave_scaled_kernelILb1ELi2E`):

| Variant | VGPRs | Spill | K-loop ISA |
|---------|-------|-------|------------|
| V0 (baseline) | 227 | 0 | reference |
| V1 (=1) | 227 | 0 | **byte-identical to V0** |
| V2 (=2) | 227 | 0 | **byte-identical to V0** |
| V3 (=3) | 227 | 0 | **byte-identical to V0** |

Extracted via `hipcc -S --cuda-device-only` and `diff` of the
`crr_exact_8wave_scaled_kernelILb1ELi2E` function. All 1786 lines of ISA are
identical across all 4 configs.

The only differences in the full compilation unit are minor SGPR renaming in the
`gemm_kernel<Layout2>` path (non-CRR), which is a compiler register allocator
artifact. The CRR kernel itself is unchanged.

## Performance results

### 70B Down CRR (M=4096, N=8192, K=28672) -- 5-run bench

| Variant | Runs (TFLOPS) | Median | Delta vs V0 |
|---------|--------------|--------|-------------|
| V0 (baseline) | 844.54, 845.49, 846.76, 847.04, 847.37 | 846.76 | -- |
| V1 (=1) | 843.80, 845.73, 845.94, 846.00, 846.06 | 845.94 | -0.10% |
| V2 (=2) | 844.09, 844.16, 844.75, 845.14, 845.40 | 844.75 | -0.24% |
| V3 (=3) | 844.61, 845.33, 845.60, 845.85, 848.01 | 845.60 | -0.14% |

All variants collapse within +/-0.25% noise band. ISA identity explains the result.

### 70B Gate/Up CRR (M=4096, N=28672, K=8192) -- baseline only (ISA-identical)

| Variant | Median (TFLOPS) |
|---------|----------------|
| V0 (baseline) | 836.78 |

Not benched for V1-V3 since ISA is byte-identical.

### Quality gates

| Cell | SNR (dB) | Det 3/3 | PASS |
|------|----------|---------|------|
| V1 70B Down | 49.60 | PASS | yes |
| V2 70B Down | 49.60 | PASS | yes |
| V3 70B Down | 49.60 | PASS | yes |

All variants preserve determinism and correctness.

## Why the SB path worked but DB does not

Dev D's SB_PIPELINE=4 achieved +19.8% on the single-buffer path. The single-buffer
path uses a fundamentally different code structure in `crr_mxfp8_exact_8wave_fastpath.inc`
with function-level MMA wrappers (`crr_mma_scaled_from_packs_fixed_phase`) that act as
implicit scheduling barriers for the compiler. The SB path also has fewer total
global_loads per iteration (4 instead of 4+1 for the DB next-iter prefetch).

The production double-buffer path in `kernel_mxfp8_layouts.cpp` uses inline MMA calls
via the `CRR_DO_MMA` macro, which the compiler can fully see through and reschedule.
Without `sched_barrier`, the compiler treats the entire K-loop body as one scheduling
region (modulo s_barrier calls) and is free to reorder VMEM loads to wherever it
determines is optimal.

The double-buffer path already achieves overlapping of global loads across K-iterations
via the tic/toc buffer scheme. The compiler's placement of loads is already close to
optimal for this structure, leaving no room for source-level reordering to improve.

## Closed axes

- **CRR DB source-level load interleaving**: CLOSED. Compiler reschedules regardless.
  Three distinct reordering patterns (V1/V2/V3) all produce byte-identical ISA.
- **CRR sched_barrier** (from R57B): CLOSED. Produces det FAIL.
- Together these close the "structural load interleaving" family for CRR DB: cannot
  control placement without sched_barrier, and sched_barrier breaks determinism.

## Default-path byte-identity

Macro defaults to 0. With `MXFP8_CRR_DB_INTERLEAVE_R57E=0` or undefined, the
original `#else` block compiles, producing the exact baseline ISA.
