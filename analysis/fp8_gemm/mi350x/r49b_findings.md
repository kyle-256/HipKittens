# R49 Dev B — RRR `do_k_iter` noinline phase split — REFUTED

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ b027c06b (stale, predates R48 wrap)
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=4
**Hypothesis:** Split the `do_k_iter` lambda into two phases (load+scale
issue vs MMA dispatch) with `__attribute__((noinline))` between them, to
relieve the 256-VGPR window pressure that RRR's full unroll hits at K-tail.

## TL;DR — VERDICT: REFUTED (catastrophic)

`MXFP8_RRR_PHASE_SPLIT=1` regresses or breaks every cell except 8B Gate/Up.
The `noinline` annotation forces a real call-frame at the partition,
which the compiler resolves by spilling the entire K-loop register live
set across the call — collapsing the kernel to ~77 TFLOPS (~3% of
baseline) on 4 shapes and to 2.7 TFLOPS (a function-call-storm regime)
on 70B Down.

| Shape          | baseline med | phasesplit med | Δ% |
|----------------|-------------:|---------------:|---:|
| 8192³          | 2937.0       | 76.9           | **-97.38%** |
| 8B Q/O 4096³   | 2225.8       | 76.9           | -96.55% |
| 8B Gate/Up     | 2453.3       | 2447.6         | -0.23%  |
| 8B Down        | 2866.1       | (bench failed, empty log) | n/a |
| 70B Q/O        | 2829.9       | 79.7           | -97.18% |
| 70B Gate/Up    | 2763.0       | (bench failed, empty log) | n/a |
| 70B Down       | 2942.3       | **2.7**        | -99.91% |

The 8B Gate/Up neutral case is the structural outlier: at N=14336 the K-loop
already spills via the bimodal-spill regime (R48G), so the additional
spill from the noinline call adds nothing the compiler hadn't already
budgeted for. On the unspilled cells (N≤8192 baseline), the noinline
turns a fully-resident K-loop into a function-call-with-spills — wholly
correct kernel, completely wrong perf.

## Diagnosis

`__attribute__((noinline))` is too coarse a tool for register-window
shaping. The compiler honors the annotation by materializing a
calling-convention frame, which forces all live values across the
boundary into stack slots regardless of register pressure. There is no
HIP/clang attribute (as of ROCm 6.x) for "soft" partition that
preserves register liveness without inlining.

Two avenues remain that *could* achieve the same goal without the
function-call cost:

1. Manual `asm volatile("" ::: "memory")` barrier between the two phases
   to gate the scheduler without a call frame. Earlier R47 attempts at
   this kind of barrier showed minor (<1%) effect — within strict-SCLK
   noise.
2. Compile-time partial unroll of the lambda body (R47A/R48G already
   explored U=2/4/8/16; bimodal-spill at N≥2 is the structural ceiling).

Neither route looks promising. RRR's K-tail register pressure is treated
as structurally bounded by the bimodal-spill regime.

## Outcome

No source landed (kernel changes were guarded by `MXFP8_RRR_PHASE_SPLIT`
default-off macro in `rrr_mxfp8_exact_8wave_fastpath.inc`; the macro and
its `#if` arms are kept in tree as documented dead code so future
researchers see why this lever was tried). RRR phase split closed.

## Files

- `r49b_bench.sh` — bench driver (5 runs/cell baseline + treatment)
- `r49b_bench.run.log` — orchestrator stdout
- `r49b_results/` — per-cell baseline + phasesplit run logs
- `r49b_isa_dumps/mxfp8_4096_phasesplit_device.s` — assembly diff
  evidence of the spurious call frame
- `rrr_mxfp8_exact_8wave_fastpath.inc` — `#if MXFP8_RRR_PHASE_SPLIT`
  arms left in place (default OFF)
