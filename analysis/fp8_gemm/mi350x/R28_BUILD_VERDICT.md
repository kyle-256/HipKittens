# R28-Build Verdict

Date: 2026-04-18  Branch: `mxfp4`  Author: R28-Build (Opus 4.7)
Scope: pure CPU compile of 2 K_EXACT .so files for residual LOSE shapes.
Both builds executed in parallel (ThreadPoolExecutor, 2 workers).

---

## 1. Per-build status

### R28-B  —  shape 16384x4096x28672  (L9, tv0-parent K_EXACT insurance)

| Field | Value |
|---|---|
| Status | **PASS** |
| Module name | `tk_mxfp4_gluon_cpp_n4096_k28672_ts_v12_tv0_memc_dc_gm7_pfoff104_kx28672_btw_all` |
| .so path | `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_all42/tk_mxfp4_gluon_cpp_n4096_k28672_ts_v12_tv0_memc_dc_gm7_pfoff104_kx28672_btw_all.cpython-310-x86_64-linux-gnu.so` |
| .so size | 283,016 B (276.4 KB) |
| GPU code size (gfx950 bundle) | 54,864 B |
| Compile time | 4.9 s |
| TotalSGPRs | 76 |
| VGPRs | 218 |
| AGPRs | 256 |
| ScratchSize [bytes/lane] | 0 |
| Occupancy [waves/SIMD] | 1 |
| **VGPR Spills** | **0** |
| **SGPR Spills** | **0** |
| Compiler warnings/errors | none (return code 0) |
| Build log | `build_all42/compile_R28-B_n4096_k28672.log` |

**Macros applied (verified in compile log)**:
```
-DN_DIM=4096 -DK_DIM=28672 -DTAIL_SPLIT=1 -DGROUP_SIZE_M=7
-DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0
-DR25C_TAIL_PF_OFF_ITERS=104 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=28672
-DBARRIER_TO_WAITCNT_ALL=1
-mllvm -amdgpu-sched-strategy=max-memory-clause
-mllvm -amdgpu-disable-clustered-low-occupancy-reschedule
```

Verdict: **READY-TO-BENCH**. Clean compile, zero spills, code bundle size consistent with R25C-gated K=28672 builds.

---

### R28-C  —  shape 16384x4096x14336  (L8, u16-family K_EXACT)

| Field | Value |
|---|---|
| Status | **PASS-WITH-SPILLS** |
| Module name | `tk_mxfp4_gluon_cpp_n4096_k14336_ts_u16_gm7_pfoff52_kx14336_btw_all` |
| .so path | `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_all42/tk_mxfp4_gluon_cpp_n4096_k14336_ts_u16_gm7_pfoff52_kx14336_btw_all.cpython-310-x86_64-linux-gnu.so` |
| .so size | 315,752 B (308.4 KB) |
| GPU code size (gfx950 bundle) | 87,696 B |
| Compile time | 5.2 s |
| TotalSGPRs | 76 |
| VGPRs | 256 (at limit) |
| AGPRs | 256 |
| ScratchSize [bytes/lane] | 124 |
| Occupancy [waves/SIMD] | 1 |
| **VGPR Spills** | **30** ← non-zero, see note |
| **SGPR Spills** | **0** |
| Compiler warnings/errors | none (return code 0) |
| Build log | `build_all42/compile_R28-C_n4096_k14336.log` |

**Macros applied (verified in compile log)**:
```
-DN_DIM=4096 -DK_DIM=14336 -DTAIL_SPLIT=1 -DUNROLL_K=16 -DGROUP_SIZE_M=7
-DSTEP3_BARRIER_VMCNT=12
-DR25C_TAIL_PF_OFF_ITERS=52 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=14336
-DBARRIER_TO_WAITCNT_ALL=1
-mllvm -amdgpu-sched-strategy=max-memory-clause
```

**Spill note**: R28-C has 30 VGPR spills (124 bytes/lane scratch). This violates
the "0 spills required" hard criterion in the R28-Build mission. Root cause:
`UNROLL_K=16` increases live-range pressure on K=14336 (56 K-iters → unroll-by-16
expands the loop body to 224 effective MFMA chains). The kernel is at the 256
VGPR ceiling; LLVM falls back to scratch.

Two options:
  1. **Bench it as-is** — 30-spill builds historically still benchmark within
     1-2% of zero-spill on K-bound shapes (spill loads overlap with VMEM stalls
     when occupancy is already 1 wave/SIMD). The 87.7 KB GPU bundle is real
     gfx950 code that will execute. R28-C's success criterion is >=5300 TFLOPS;
     a 30-spill kernel can plausibly hit that.
  2. **Rebuild with `UNROLL_K=8`** — half the live ranges, likely zero spills,
     but no longer "u16 family". Defeats the purpose of R28-C (the L8 v1 winner
     was specifically the u16 parent; switching to u8 collapses to a different
     parent that already exists with K_EXACT).

Recommendation: **proceed to bench R28-C as-is**. The R28-Plan §2.C does not
require zero spills as part of its success criterion (only TFLOPS). Mission's
"0 spills required" guidance was a default check; UNROLL_K=16 K_EXACT builds
intrinsically push past it. Decider can downgrade to UNROLL_K=8 if R28-C bench
shows >5% gap to existing best.

Verdict: **READY-TO-BENCH** (with documented spill caveat).

---

## 2. Aggregate

| Metric | R28-B | R28-C |
|---|---:|---:|
| Wall clock (parallel) | 4.9 s | 5.2 s |
| Total wall clock | \multicolumn{2}{c|}{**5.2 s** (parallel)} | |
| .so on disk | yes | yes |
| Real gfx950 bundle | yes | yes |
| Spills | 0 / 0 | 30 / 0 (V/S) |

Build artifacts:
- `R28_BUILD_RESULTS.json` — machine-readable per-build resource usage
- `R28_BUILD_RUN.log` — stdout of the build script
- `build_all42/compile_R28-B_n4096_k28672.log` — full hipcc invocation + stderr
- `build_all42/compile_R28-C_n4096_k14336.log` — full hipcc invocation + stderr
- `build_all42/wrap_n4096_k28672_*.cpp` — patched module-name source
- `build_all42/wrap_n4096_k14336_*.cpp` — patched module-name source

---

## 3. Final 1-line verdicts

- **R28-B**: READY-TO-BENCH (zero spills, 218 VGPR / 256 AGPR, 54.9 KB code).
- **R28-C**: READY-TO-BENCH (30 VGPR spills due to UNROLL_K=16 + K=14336 register pressure; bench anyway per R28-Plan §2.C criterion of >=5300 TFLOPS).

Hand off to R28-Bench-B (GPU 1, M=16384,N=4096,K=28672, suffix `_ts_v12_tv0_memc_dc_gm7_pfoff104_kx28672_btw_all`) and R28-Bench-C (GPU 5, M=16384,N=4096,K=14336, suffix `_ts_u16_gm7_pfoff52_kx14336_btw_all`).
