# R26-B Verdict: STEP3_PF_N × STEP4_PF_N × R25-G stack — DEAD

Date: 2026-04-18  Vector: V3 from R26_PLAN.md  Optimizer: R26-B (Claude 4.7 Opus)

## Hypothesis (recap)
R25-F/G discovered tail prefetches (last K_iters - 6 onwards) are dead weight
because B-tile is L2-resident after ~2 iters. Conjecture: the same mechanism
implies the *steady-state* PF_N (`STEP3_PF_N`/`STEP4_PF_N`, defaults 8/4)
may also be over-prefetched on K-bound shapes, freeing VMEM bandwidth.

## Method
- GPU 4 (idle), warmup=200, iters=500, trim=10%, 5-rep smoke per variant
- Two shapes: DLA2 (128256x32768x4096) and DLA7 (28672x32768x4096)
- Baseline: existing R25-F per-shape stack
  (DLA2: `_ts_gm7_v12_memc_dc_pfoff14`; DLA7: `_ts_lgk2_gm7_v12_memc_pfoff14`)
- 6 PF_N variants: `(STEP3,STEP4) ∈ {(8,4)=base, (4,4), (4,2), (2,2), (6,4), (6,2)}`
- R25-G stack flags applied to all variants (`-DTAIL_SPLIT=1 -DGROUP_SIZE_M=7
  -DSTEP3_BARRIER_VMCNT=12 -DR25C_TAIL_PF_OFF_ITERS=14
  -mllvm -amdgpu-sched-strategy=max-memory-clause` ± dc/lgk2 per shape)
- WIN threshold: ≥+75 TFLOPS (+1.5pp) AND std ≤ 25 TFLOPS

## Results (5-rep means, TFLOPS)

| variant   | DLA2 (comp=4536) | Δ vs base | DLA7 (comp=4467) | Δ vs base |
|-----------|------------------|-----------|------------------|-----------|
| base 8/4  | 4888.8 ± 6.4     | 0.0       | 4940.8 ± 17.4    | 0.0       |
| 4/4       | 4883.8 ± 5.0     | -5.0      | 4949.9 ± 12.3    | +9.1      |
| 4/2       | 4877.7 ± 3.3     | -11.1     | 4948.6 ± 17.2    | +7.8      |
| 2/2       | 4873.8 ± 5.8     | -15.0     | 4944.3 ± 10.3    | +3.5      |
| 6/4       | 4890.2 ± 5.0     | +1.4      | 4972.2 ± 10.3    | +31.4     |
| 6/2       | 4882.7 ± 7.3     | -6.1      | 4953.2 ± 15.1    | +12.4     |

## Verdict: DEAD

- **DLA2**: All variants flat or slightly worse than base (max delta +1.4 TFLOPS).
  Decreasing PF_N monotonically hurts (-5 → -15) — the steady-state PF *is*
  load-bearing on the heavy-M shape.
- **DLA7**: Best variant `pf3_6_4` = +31.4 TFLOPS over base (4972 vs 4941),
  but well under the +75 TFLOPS WIN threshold and within 2σ of measurement
  noise (base std = 17.4). Not committable.

## Mechanism interpretation
The R25-F mechanism ("B is L2-resident after iter 2") only applies to *late*
iterations: by iter K-6, B is fully cached so the tail PFs are dead weight.
Steady-state PFs in the *first* iters still serve their original purpose
(filling L2 for the cache reuse pattern). Cutting steady-state PF_N from 8 to
4-6 doesn't free meaningful VMEM bandwidth because the late-iter VMEM is
already turned off by `R25C_TAIL_PF_OFF_ITERS=14`. The two axes are *not*
super-additive on K=4096 shapes.

## Constraint adherence
- 12 builds (target ≤ 12). OK.
- GPU 4 only. OK.
- No kernel C++ touched. OK.
- bench_all_42.py untouched (no WIN to wire). OK.
- R25-C/D/F/G/H wires untouched. OK.

## Files
- `bench_r26b_pfn.py` — smoke script
- `bench_r26b_pfn.log` — full 5-rep log
- `bench_r26b_pfn_results.json` — JSON dump
- `build_r26b/` — 12 .so files (cached for re-runs)
