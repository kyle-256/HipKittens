# R25-C — K-loop epilogue specialization (R25C_TAIL_PF_OFF_ITERS)

**Date**: 2026-04-18
**Worktree**: `/shared_nfs/kyle/test/HipKittens/.claude/worktrees/agent-adae0099`
**GPUs used**: 4, 6, 7 (idle per rocm-smi at start)
**Bench params**: warmup=200, iters=500, trim=10% (per `.claude/rules/benchmark-rules.md`)

## Hypothesis
Last 1-4 iters of the steady-state K-loop call `kpair_32mfma_with_lds_and_pf` whose
prefetch index `pf_bt = (bt+2 < k_byte_iters) ? (bt+2) : (k_byte_iters-1)` clamps
to the last K-tile and re-fetches already-cached lines, wasting saturated VMEM
slots on stale data. Switching the trailing iters to `PF_N=0` (no global pf)
should free VMEM for the actual scale loads of the final mfma block.

## Implementation
- New macro `R25C_TAIL_PF_OFF_ITERS` (default 0). When `>0`, branches the last N
  iters of the TAIL_SPLIT main loop to `kpair_32mfma_with_lds_and_pf<0,...>`
  (Step3) and `kpair_32mfma_with_lds_and_pf<0>` (Step4). Both functions already
  support `PF_N=0` via `if constexpr (PF_N>0)` — identical mfma+ds_read schedule,
  zero `emit_one_pf` calls.
- Gating macro `R25C_K_LIMIT` (default 32768) disables R25C for `K_DIM > limit`.
  Needed because the runtime branch `bt >= k_byte_iters - 1 - N` only folds to a
  compile-time constant when the K-loop is fully unrolled (`#pragma unroll` for
  `K_DIM/256 ≤ 16`). For larger K (DLA1 K=128256, `pragma unroll 8`), `bt` is a
  runtime variable inside each unrolled chunk and the compiler keeps both code
  paths live → ~10-20% regression in initial probe.
- All changes live in `kernel_mxfp4_gluon_cpp.cpp`:
  - Macro defaults: lines ~78-100
  - Step3/Step4 branch: lines ~2783-2890

## Variants & Results

### DLA1 (M=4096, N=32768, K=128256) — R25C is no-op (K_DIM > 32768 gate)

Confirmed via 2-rep re-bench that all DLA1 binaries are functionally identical
(constexpr `_r25c_tail_no_pf=false`); per-iter variation is bench noise.

| Variant | Mean TFLOPS (2 reps) | Range |
|---|---|---|
| baseline | 4943.27 | 374 (noise: rep0=4756, rep1=5130) |
| pfoff1 | 5088.14 | 19 |
| pfoff2 | 5114.14 | 45 |
| pfoff3 | 5094.76 | 53 |
| pfoff4 | 5078.68 | 31 |

**DLA1 verdict**: no real effect (gated off). Initial smoke baseline showed an
outlier 4756 TFLOPS reading; tighter re-bench shows ~5080-5114 across all
variants (within noise).

### DLA2 (M=128256, N=32768, K=4096) — WIN

3-rep tight re-bench:

| Variant | Mean TFLOPS | Δ vs baseline | Range |
|---|---|---|---|
| baseline | 4166.21 | — | 21 |
| pfoff3 | 4313.29 | **+3.53 %** | 3 |
| pfoff4 | **4414.71** | **+5.96 %** | 15 |

### DLA7 (M=28672, N=32768, K=4096) — WIN

3-rep tight re-bench:

| Variant | Mean TFLOPS | Δ vs baseline | Range |
|---|---|---|---|
| baseline | 4183.93 | — | 10 |
| pfoff3 | 4271.15 | **+2.08 %** | 4 |
| pfoff4 | **4315.26** | **+3.14 %** | 25 |

### Best N per shape
- **DLA1**: gated off (K=128256 > 32768) — no effect.
- **DLA2 (K=4096)**: best `R25C_TAIL_PF_OFF_ITERS=4` → +5.96 %.
- **DLA7 (K=4096)**: best `R25C_TAIL_PF_OFF_ITERS=4` → +3.14 %.

Initial smoke (1 rep) had ranked pfoff3 above pfoff4 for DLA2; the tighter 3-rep
bench reverses to pfoff4 best. For DLA7 pfoff4 was best in both runs.

## Verdict: WIN (limited)

- Mission gate met: ≥+1.5pp on at least 1 shape (multiple, both DLA2 +5.96% and
  DLA7 +3.14%); no other shape regressing > 1pp (DLA1 within noise once re-bench
  tightened).
- DLA1 is the deep-LOSE shape per project state; R25C does not help there
  because its k_byte_iters=501 forces partial-unroll and the runtime branch
  doesn't fold. Gating saves DLA1 from regression but leaves no improvement.

## Mechanistic interpretation
The HBM-bound short-K shapes (K=4096 → 16 K-tile iters) get a real benefit from
removing the redundant tail prefetches:
- `pf_bt` clamps to `k_byte_iters-1` for both bt=k_byte_iters-2 (final main
  loop iter) and bt=k_byte_iters-3 (because bt+2=k_byte_iters-1 is still
  in-range but already-fetched). The clamping creates 2-4 iters of duplicate
  VMEM issues that compete with the scale loads on a saturated bus.
- Removing 8 tile prefetches × 4 iters × 2 steps × number-of-CTAs frees enough
  VMEM bandwidth that the last few iters complete sooner. Tail latency
  reduction ≈ 5-6% on 16-iter loops, decreasing as 1/k_byte_iters for larger K.
- For K=128256 (501 iters), the per-loop-iteration branch overhead would
  overwhelm any tail-iter savings (4/501 ≈ 0.8% of iters affected) — so the
  K_DIM gate is correct.

## Files
- Kernel diff: `kernel_mxfp4_gluon_cpp.cpp` lines ~78-100 (macro), ~2783-2890
  (branched dispatch).
- Build: `build_round25_optC.py`
- Bench: `bench_round25_optC_smoke.py`, log `bench_round25_optC_smoke.log`,
  json `bench_round25_optC_smoke.json`

## Caveats / Future work
- DLA1 (K=128256) is untouched by R25C. The deep-LOSE on DLA1 would need a
  loop-peel approach: split the main loop into a head (`k_byte_iters-1-N`
  iters, fully prefetched) and a peeled tail (`N` iters, no pf). That requires
  duplicating the ~200-line main-loop body. Not attempted in this round.
- Even with R25C, DLA2/DLA7 are still BELOW the project's competitor TFLOPS
  baselines (DLA2 4415 vs aiter 4536; DLA7 4315 vs aiter 4467). R25C narrows
  the gap by 30-50% but does not flip the LOSE → WIN against aiter.
