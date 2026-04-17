# Round 13 Optimizer A — Alternative LLVM sched-strategies on the 4 SGPR-clobber shapes

## TL;DR
**All tested non-default sched-strategies are BROKEN on the 4 deep-LOSE
target shapes.** Every alternative iterative/greedy strategy (`iterative-minreg`,
`iterative-maxocc`, `max-ilp`) reproduces the same SGPR-clobber compiler bug
that we found with `iterative-ilp` in R11/R12. **No commits recommended.**

## Setup
- Targets: DLA1 (4096x32768x128256), DLA2 (128256x32768x4096),
  DLA7 (28672x32768x4096), WIN2 (32768x6144x2048).
- For each shape, built parent-best variant + extra `-mllvm -amdgpu-sched-strategy=<S>`
  (overriding any sched-strategy already in parent flags via cl::opt last-wins).
- Strategies: `iterative-minreg`, `iterative-maxocc`, `max-ilp`,
  plus probes `max-occupancy`, `iterative-max-occupancy-experimental`.
- Valid LLVM 20 enum names (verified by string-dump of
  `/opt/rocm/llvm/lib/libLLVMAMDGPUCodeGen.a`):
  `iterative-ilp`, `iterative-maxocc`, `iterative-minreg`,
  `max-ilp`, `max-memory-clause` (un-prefixed), all `gcn-*` (silent-noop).
  `max-occupancy` and `iterative-max-occupancy-experimental` are NOT
  in the un-prefixed list and behave as silent-noops here (verified
  by .text-section hash).

## ASM-diff probe (.text section sha256)
| Shape | iterminreg | itermaxocc | maxilp | maxocc | itermaxoccx |
|-------|-----------|-----------|-------|-------|------------|
| DLA1  | DIFF      | DIFF      | DIFF  | NOOP  | NOOP       |
| DLA2  | DIFF      | NOOP      | DIFF  | NOOP  | NOOP       |
| DLA7  | DIFF      | NOOP      | DIFF  | NOOP  | NOOP       |
| WIN2  | DIFF      | NOOP-vs-default | DIFF  | NOOP  | NOOP   |

Notes:
- "NOOP" = .text byte-identical to one of the silent-noop strategies.
- For WIN2 (parent has no sched-strategy flag at all), `itermaxocc`,
  `maxocc`, and `itermaxoccx` ALL produce the same .text as default —
  meaning iterative-maxocc converges to the default schedule on WIN2.
- For DLA2 / DLA7, `itermaxocc` matches `maxocc`/`itermaxoccx`, so it
  is also default-equivalent here (the parent's `max-memory-clause` is
  overridden but iterative-maxocc collapses to default anyway).
- Only `iterminreg` and `maxilp` consistently produce distinct schedules.

## Smoke-test verdict (warmup=20, iters=50, 3 retries, GPU 5)
| Shape | iterminreg | itermaxocc | maxilp |
|-------|-----------|-----------|-------|
| DLA1  | BROKEN (0/3, NaN finite=0.18) | BROKEN (0/3, APERTURE) | BROKEN (0/3, APERTURE) |
| DLA2  | BROKEN (0/3, NaN finite=0.91) | BROKEN (0/3, NaN finite=0.90) | BROKEN (0/3, NaN finite=0.97) |
| DLA7  | BROKEN (0/3, NaN finite=0.83) | BROKEN (0/3, NaN finite=0.81) | BROKEN (0/3, NaN finite=0.94) |
| WIN2  | BROKEN (0/3, NaN finite=0.92) | (skip: parent-equiv)   | BROKEN (0/3, NaN finite=0.97) |

**0 of 10 non-noop variants survive smoke.** No single-shot or 5-run
verify was attempted because nothing produced finite output.

## Failure modes observed
- DLA1 + iterminreg: deterministic NaN with ~18% finite output (massive
  output-pointer corruption — base pointer wrong, almost all stores went
  somewhere else but ~18% of the tile happened to land in C).
- DLA1 + itermaxocc / maxilp: deterministic HSA aperture violation
  (kernel page-fault before kernel completion).
- DLA2 / DLA7 / WIN2 + all strategies: deterministic NaN with mostly-
  finite output (80-99%). Suggests partial-tile corruption — some
  output stores OK, some clobbered. Pattern: the higher the finite
  fraction, the more localized the SGPR-clobber.

## Root-cause inference
- The bug we previously attributed to `iterative-ilp` is in fact a
  **shape-specific SGPR-clobber that triggers under ANY non-default
  AMDGPU MachineScheduler post-RA reorder** when:
  - AGPR=256 saturation (full),
  - VGPR ≥ ~420,
  - LDS = 128 KiB,
  - and the kernel uses `lds_direct` `buffer_load_dwordx4 ... offen lds`
    plus many `s_load_dwordx2` kernarg loads.
- This is a more general bug than R12 concluded. The R12 verdict said
  "real LLVM `iterative-ilp` correctness bug"; the truth is "the bug
  is triggered by `iterative-ilp` AND `iterative-minreg` AND
  `iterative-maxocc` AND `max-ilp`" — i.e. by any non-default-iteration
  scheduler. `max-memory-clause` is the lone exception that survives
  these shapes (which is why it's our default in the parent flags).

## Recommended action for these 4 shapes
1. **Do NOT commit any R13A variant.** Nothing produced correct output.
2. **The sched-strategy lever is exhausted** for DLA1/DLA2/DLA7/WIN2.
   These shapes need a non-scheduler intervention:
   - kernel-source level changes (different tile shape, different
     pipelining depth, different SLM layout), OR
   - selective LLVM compiler flags that don't change the post-RA scheduler
     (e.g. `-mllvm -amdgpu-membound-threshold=`, `-mllvm
     -amdgpu-schedule-metric-bias=`, regalloc tuning, etc.).
3. Specifically for **DLA1 (worst-shape, 88.3%)**: the input layout has
   K=128256 which is unusual. Consider a one-off K-tail variant or
   k-split/k-fission approach.
4. Consider filing an upstream LLVM AMDGPU bug with the .so + repro
   script, expanding the R12 report to include the new evidence:
   - 4 distinct sched-strategies all trigger pointer corruption.
   - max-memory-clause is the only iterative scheduler that does NOT
     trigger it.

## Files written
- `build_round13_optA_alt_sched.py`
- `build_round13_optA_alt_sched.log`
- `asm_diff_probe_r13.py`
- `asm_diff_probe_r13.log`
- `asm_diff_probe_r13_results.json`
- `bench_round13_optA_smoke.py`
- `bench_round13_optA_smoke.log`
- `bench_round13_optA_smoke.json`
- 20 .so files at `build_all42/tk_mxfp4_gluon_cpp_n*_k*_*_r13_*.cpython-310-x86_64-linux-gnu.so`

(No singleshot or verify scripts produced — gated out by smoke verdict.)
