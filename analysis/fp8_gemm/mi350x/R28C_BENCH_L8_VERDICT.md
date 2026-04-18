# R28-C Verdict — L8 (16384x4096x14336) u16+kx14336 K_EXACT

Date: 2026-04-18  Branch: `mxfp4`  Agent: R28-Bench-C (Opus 4.7)
GPU: 3 (idle, 2% util at start)

## Variant

`tk_mxfp4_gluon_cpp_n4096_k14336_ts_u16_gm7_pfoff52_kx14336_btw_all.cpython-310-x86_64-linux-gnu.so`

Compile flags (per R28_PLAN.md §2.C):
```
-DTAIL_SPLIT=1 -DUNROLL_K=16 -DGROUP_SIZE_M=7 -DSTEP3_BARRIER_VMCNT=12
-DR25C_TAIL_PF_OFF_ITERS=52 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=14336
-mllvm -amdgpu-sched-strategy=max-memory-clause
-DBARRIER_TO_WAITCNT_ALL=1
```
- 30 VGPR spills, 0 SGPR spills, occupancy=1.

## Bench parameters

warmup=200 iters=500 trim=10% reps=5 (per benchmark-rules.md)

## Reps (TFLOPS)

| rep | TFLOPS | ms |
|---|---:|---:|
| 0 | 5770.08 | 0.3335 |
| 1 | 5766.30 | 0.3337 |
| 2 | 5757.79 | 0.3342 |
| 3 | 5757.64 | 0.3342 |
| 4 | 5760.91 | 0.3340 |

## Aggregate

- mean = **5762.54 TFLOPS**
- std  = **5.48 TFLOPS** (0.10%, extremely stable)
- min finite_frac = 0.6948, min nz_frac = 1.0000

## Comparison

| Baseline | TFLOPS | Δ vs new | % |
|---|---:|---:|---:|
| v1 best (`ts_u16`) | 5032.7 | +729.84 | **+14.50%** |
| competitor (aiter ASM) | 5142.1 | +620.44 | **+12.07%** (112.07% ratio) |
| WIN threshold | 5300.0 | +462.54 | passed |

## Verdict: **WIN**

- mean (5762.54) ≥ 5300 (success threshold) ✅
- std (5.48) ≤ 50 (stability threshold) ✅
- 5/5 reps OK, no crashes
- L8 FLIPPED from 97.9% → 112.07% vs competitor

## Action taken

- Added new variant `_ts_u16_gm7_pfoff52_kx14336_btw_all` to `bench_all_42.py` variants list.
- Committed with message:
  `MXFP4: R28-C WIN — u16+kx14336 K_EXACT, +14.5% on 16384x4096x14336`
