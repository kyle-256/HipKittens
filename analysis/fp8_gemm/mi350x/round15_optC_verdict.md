# Round 15 Optimizer C — Source-level micro-probe verdict

**Date**: 2026-04-17
**Time-box**: 35 min (used ~30 min wall-clock)
**Outcome**: ALL 3 PROBES — DEAD END. No source edits committed. Kernel.cpp reverted.

## Setup

Parent: `_ts_pf6_6_v12_memc` (`-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6
-DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm
-amdgpu-sched-strategy=max-memory-clause`)

3 NEW MACROS added to kernel_mxfp4_gluon_cpp.cpp (since reverted):

| Macro | Default | Effect when enabled |
|---|---|---|
| `TAIL_BARRIER_LGKMCNT` | -1 (off) | Adds extra `s_waitcnt lgkmcnt(N) + s_barrier` AFTER the tail vmcnt barrier (TAIL_SPLIT only). |
| `STEP4_BARRIER_VMCNT`  | -1 (off) | Inserts `s_waitcnt vmcnt(N) + s_barrier` BETWEEN `emit_pf_tail<STEP3_PF_N>` and the Step4 kpair launch. Applies to all 3 main loops (SWAP, TAIL_SPLIT=1, TAIL_SPLIT=0). |
| `PF_GROUP_OFFSET`      | 0 | Rotates `pf1` emission order in `emit_pf_tail` by ±1 (start at idx 1 wrap to 0, or last then 0..PF_MPT-1). Same loads, different scheduler-visible order. |

All 15 build variants compiled cleanly (`build_round15_optC_srclevel.log` — 9.9 s parallel).

## Single-shot smoke (warmup=20, iters=50, GPUs 6/7) — `bench_round15_optC_smoke.json`

Parent baselines: DLA1 5126.28 TFLOPS (88.67% aiter); DLA7 4202.60 TFLOPS (88.61% aiter).

| Probe | Variant | Shape | TFLOPS | Δpp vs parent | Verdict |
|---|---|---|---|---|---|
| P1 | TAIL_BARRIER_LGKMCNT=0 | DLA7 | 4167.42 | -0.84 | LOSE |
| P1 | TAIL_BARRIER_LGKMCNT=2 | DLA7 | 4162.83 | -0.95 | LOSE |
| P1 | TAIL_BARRIER_LGKMCNT=4 | DLA7 | 4193.92 | -0.21 | within noise |
| P2 | STEP4_BARRIER_VMCNT=4  | DLA1 | 4646.33 | -9.36 | hard LOSE |
| P2 | STEP4_BARRIER_VMCNT=8  | DLA1 | 4985.02 | -2.76 | LOSE |
| P2 | STEP4_BARRIER_VMCNT=12 | DLA1 | 5117.58 | -0.17 | within noise |
| P2 | STEP4_BARRIER_VMCNT=16 | DLA1 | 5104.03 | -0.43 | LOSE |
| P2 | STEP4_BARRIER_VMCNT=20 | DLA1 | 5110.99 | -0.30 | LOSE |
| P2 | STEP4_BARRIER_VMCNT=4  | DLA7 | 4006.90 | -4.66 | hard LOSE |
| P2 | STEP4_BARRIER_VMCNT=8  | DLA7 | 4094.20 | -2.58 | LOSE |
| P2 | STEP4_BARRIER_VMCNT=12 | DLA7 | 4126.85 | -1.80 | LOSE |
| P2 | STEP4_BARRIER_VMCNT=16 | DLA7 | 4134.90 | -1.61 | LOSE |
| P2 | STEP4_BARRIER_VMCNT=20 | DLA7 | 4158.17 | -1.06 | LOSE |
| P3 | PF_GROUP_OFFSET=-1     | DLA1 | 5136.44 | **+0.20** | within noise (below +0.5pp gate) |
| P3 | PF_GROUP_OFFSET=+1     | DLA1 | 5125.28 | -0.02 | flat |

**No variant cleared the +0.5pp single-shot smoke gate. None reached the +1pp commit gate.**
Best signal: P3 PF_GROUP_OFFSET=-1 at +0.20pp on DLA1 — well within run-to-run noise (±0.4pp on this shape per R12/R13 history).

## Correctness verification — `bench_round15_optC_verify.json`

Cross-variant SNR vs parent .so on identical inputs (tight scales `[0]`, low fp4 magnitudes 0..3) failed
to produce a clean SNR number because PARENT itself overflows bf16 dynamic range at K=128256/K=4096
(`finite_p` = 0.78-0.86). However:

- DLA1 variants: `nan_match` between parent and variant = **0.96-0.98** (positions of NaN/Inf agree).
- DLA7 variants: `nan_match` = 0.73-0.78 (lower, but consistent across all 3 probes — likely from
  non-deterministic atomic accumulation order which is expected with barrier reordering).

Since no variant cleared the perf gate, no commit was made and **no source edits remain in the tree**
(kernel_mxfp4_gluon_cpp.cpp git-reverted).

## Findings (registry updates)

1. **STEP4_BARRIER_VMCNT** — adding ANY second barrier between Step3 and Step4 hurts: even VMCNT=20
   (loosest) costs 0.30pp on DLA1 and 1.06pp on DLA7. The Step3 barrier (STEP3_BARRIER_VMCNT=12) is
   already sufficient; a second barrier just adds latency. **CLOSED**.

2. **TAIL_BARRIER_LGKMCNT** — ineffective on DLA7. Tail iter is too small a fraction
   (1/(K/256) = 1/16 = 6.25% for DLA7 K=4096) for tail-only barrier tweaks to register. Larger-K
   shapes already excluded by R8B's mathematical analysis. **CLOSED**.

3. **PF_GROUP_OFFSET** — ±1 rotation of pf1 emission order has no measurable effect (-0.02pp / +0.20pp
   on DLA1, both within noise). The compiler likely re-orders the buffer_load_lds dispatches anyway
   based on its scheduler — the source-level rotation doesn't survive instruction selection. **CLOSED**.

## Conclusion

The frontier remains: kernel-source rewrites at the macro/single-line level cannot move the needle on
the 4 broken deep-LOSE shapes. Confirms saturation. Multi-day kernel rewrites (different LDS layout,
different MFMA scheduling, fundamentally different prefetch strategy) are the only remaining axis.

Files: `build_round15_optC_srclevel.{py,log}`, `bench_round15_optC_smoke.{py,log,json}`,
`bench_round15_optC_verify.{py,log,json}`. No commits. No new BROKEN entries (all 15 builds clean).
