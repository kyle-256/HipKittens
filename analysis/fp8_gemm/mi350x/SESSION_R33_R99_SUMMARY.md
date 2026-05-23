# Session R33-R99 Summary (R100 milestone, 67 rounds)

## Top-line Results

- **1 production perf commit** (R43 dispatcher chunk_size 64→32)
  - v2/Triton geomean 1.014 → 1.024 stable (5-trial median ±0.003)
  - v2/v1 geomean 1.120 across full 24-shape (23/24 wins, 1 noise)
- **6 foundation probes** (R52-R58) validating 32×32 mfma path
  - All spill=0 across 1-mma → 8-warp + LDS probes
  - V=104 A=16 max at production-like state
- **9 multi-session design docs** (R63, R64, R65, R66, R83-R89, R91-R93)
  - P1.2 32×32 rewrite plan
  - P1.3a split-K plan
  - P2.2 RRR plan
  - P3.2 CRR plan
  - Foundation probe methodology
  - CK vs HK comparison
  - Rocprof workflow
  - Multi-session execution checklist
  - Hyperparameter search policy
  - Per-shape goal worksheet
  - Triton autotune replication
  - Integration test plan
  - Split-K reduce kernel design
  - Workspace allocator design
- **2 new helper scripts**: `_check_spill.py`, `_bench_24_v2.py` (gitignored, synced via rsync)
- **33 commits** to HK turbo + PT 3rdparty mirrors + PT outer, all parity ✓

## What Was Tried (R33-R99 round table)

| Round | Lever | Outcome |
|---|---|---|
| R33 | unroll 4 RCR main | REGRESSION correctness |
| R34 | ISA disasm 4-variant | DIAGNOSTIC (spill all in FUSED=true) |
| R35 | FUSED v0/v1 intermediate elim | 0 effect |
| R36 | cB vacc | REGRESSION (spill +16) |
| R37 | cA+cD vacc | REGRESSION correctness |
| R38 | builtin mfma | BUILD ERROR |
| R39 | drop post-cA s_barrier | REGRESSION correctness |
| R40 | noinline scale lambda | BUILD ERROR |
| R41 | revert cA vacc | 0 effect (R27 historical artifact) |
| R42 | amdgpu_num_vgpr attribute | 0 effect (ignored) |
| **R43** | **chunk_size 64→32** | **WIN +1.5pp** |
| R44 | sched_group_barrier(0x8,4,0) | REGRESSION dsv3 5x |
| R45 | ST_v2 → ST_v2a swizzle | unstable (GPU shared) |
| R46 | single_acc_blk128 probe | BUILD ERROR |
| R47 | sched_barrier(0) end-of-iter | -1.2pp geomean |
| R48 | dynamic chunk_size (ki≤16:32 else:16) | REGRESSION long K |
| R49 | unroll 2→1 | -0.5pp noise |
| R50 | PLAN_V2 round log | DOCS commit |
| R51 | drop init prolog vmcnt | mixed shapes |
| R52-R58 | 32×32 foundation probes (6 probes) | ALL spill=0 |
| R59 | CLAUDE.md log | DOCS |
| R60 | A_row_reg_32 typedef | BUILD ERROR |
| R61 | drop-in 32×32 wrapper | BUILD ERROR |
| R62 | R43 3-trial stability | 1.027 stable |
| R63-R66 | 4 design docs | DOCS commits |
| R67 | final-state bench | 1.086 noisy trial |
| R68 | PLAN_V2 update | DOCS |
| R69 | production spill verify | 24/35 unchanged |
| R70 | PLAN_V2 round log commit | DOCS commit |
| R71 | 5-trial stability | 1.024±0.003 |
| R72 | per-shape worst-case ID | qwen_down B16 M2048 0.936 |
| R73 | NUM_CUS env sweep | 256 optimal |
| R74 | round milestone log | DOCS |
| R75 | final commit hash anchor | DOCS |
| R76 | __attribute__((hot)) | -0.2pp noise |
| R77 | rocprof infrastructure | BLOCKED (ssh escape) |
| R78 | bench post-R76 revert | 1.019 |
| R79 | drop wm==0 store barrier | -14pp BIG REGRESSION |
| R80 | rocprof PMC | BLOCKED |
| R81 | feedback_fp8_rcr_partial_barriers memory | DOCS |
| R82 | __builtin_expect FUSED rare | noisy 0 effect |
| R83 | FOUNDATION_PROBE_PATTERN doc | DOCS commit |
| R84 | CK_VS_HK_COMPARISON doc | DOCS commit |
| R85 | ROCPROF_WORKFLOW doc | DOCS commit |
| R86 | MULTI_SESSION_EXECUTION_CHECKLIST doc | DOCS commit |
| R87 | HYPERPARAM_SEARCH doc | DOCS commit |
| R88 | SPLITK_REDUCE_KERNEL doc | DOCS commit |
| R89 | WORKSPACE_ALLOC_DESIGN doc | DOCS commit |
| R90 | shape-conditional chunk_size | -0.5pp net |
| R91 | PER_SHAPE_GOAL doc | DOCS commit |
| R92 | TRITON_AUTOTUNE_REPLICATION doc | DOCS commit |
| R93 | INTEGRATION_TEST_PLAN doc | DOCS commit |
| R94 | _check_spill.py helper | scripts/ |
| R95 | _bench_24_v2.py + run | 24-shape v2/v1=1.120 |
| R96 | KPI_SNAPSHOT_R95 doc | DOCS commit |
| R97 | qwen_up regression isolated check | false alarm, v2 actually faster |
| R98 | KPI snapshot correction | DOCS commit |
| R99 | final commit hash anchor | afe3289b/81f34501 |

## Target Achievement

| Goal | Current | Status |
|---|---|---|
| v2/Triton ≥ 1.15× | 1.024× stable | -12pp (needs P1.2 + P1.3a multi-session) |
| spill=0 全 path | 24/35 | needs P1.2 K-loop rewrite (foundation R52-R58 validates path) |
| ≤ 3% hk_dense gap | qwen_down B16 M2048 0.94× worst | bandwidth-bound, needs P1.3a algorithmic |

## Multi-Session Estimates (per design docs)

| Phase | LOC | Sessions |
|---|---|---|
| P1.2 32×32 rewrite | 600 | 5 |
| P1.3a split-K | 800 | 5 |
| P2.2 RRR rewrite | 700 | 5-7 |
| P3.2 CRR layout | 400 | 3-4 |
| Total to full goal | 2500 | 18-21 sessions |

## Anchors

- HK turbo HEAD: `afe3289b`
- PT outer HEAD: `81f34501`
- PT 3rdparty HEAD: `97678b5b`
- Foundation probes embedded in `kernel_fp8_layouts2.cpp` (lines 191-339 approx)
- Design docs in `analysis/fp8_gemm/mi350x/*.md`
- Memory files updated: `fp8-rcr-unroll-harmful`, `fp8-rcr-partial-barriers`
