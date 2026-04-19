# R50 Dev D — RRR `do_k_iter` inter-phase `sched_barrier` mask sweep — REFUTED

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 03c94ab3
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=4
**Hypothesis:** Insert `__builtin_amdgcn_sched_barrier(<mask>)` between the
load-issue half and the MFMA-dispatch half of the `do_k_iter` lambda body
in the MXFP8 RRR exact-8wave fastpath. The barrier should gate the LLVM
post-RA scheduler at the load→MMA boundary without materializing a call
frame (avoiding R49 Dev B's noinline catastrophe), letting the compiler
keep the K-loop body monolithic while improving load→consume scheduling
for the bimodal-spill regime (N≥2).

## TL;DR — VERDICT: REFUTED (within strict-SCLK noise)

All 4 non-default mask values (0x0000, 0x0008, 0x0010, 0x0080) deliver
≤0.15 % delta vs the OFF baseline on 8B Gate/Up RRR — far below the
+1.5 % SHIP gate and well inside the ±1 % bail band. Build remarks are
**bit-identical** across all 5 mask values (TotalSGPRs=41, VGPRs=256,
VGPRs Spill=16, ScratchSize=68 B/lane, Occupancy=2 waves/SIMD).

This empirically confirms the prior expectation in **r49b_findings.md
"Diagnosis"** that soft scheduler barriers have <1 % effect on this
kernel under strict-SCLK 5×/30s/60s. With both R49 Dev B (noinline:
catastrophic) and R50 Dev D (sched_barrier: no-op) refuted, **the
"phase-split do_k_iter" lever family is closed**.

## Phase 1 — 8B Gate/Up RRR mask sweep (5×/30s SCLK)

`M=4096 N=14336 K=4096`, MXFP8_WARMUP=100, MXFP8_ITERS=200,
MXFP8_PRESHUFFLE_QUANT=1, HIP_VISIBLE_DEVICES=4, 60s rebuild cooldown,
30s inter-run cooldown.

### Build remarks (V1-RRR preshuffled kernel `Lb1ELi1`)

All 5 builds produce identical resource usage — the sched_barrier is a
pure scheduler hint with **zero register-pressure effect**:

| MXFP8_RRR_SCHED_BARRIER | TotalSGPRs | VGPRs | SGPR Spill | VGPR Spill | Scratch (B/lane) | Occupancy |
|------------------------:|-----------:|------:|-----------:|-----------:|-----------------:|----------:|
| 0 (OFF, baseline)       | 41         | 256   | 0          | 16         | 68               | 2         |
| 1 (mask 0x0000)         | 41         | 256   | 0          | 16         | 68               | 2         |
| 2 (mask 0x0008 MFMA)    | 41         | 256   | 0          | 16         | 68               | 2         |
| 3 (mask 0x0010 VMEM)    | 41         | 256   | 0          | 16         | 68               | 2         |
| 4 (mask 0x0080 LDS)     | 41         | 256   | 0          | 16         | 68               | 2         |

(For completeness, the V2-RRR preshuffled kernel `Lb1ELi2` reports
TotalSGPRs=46, VGPRs=254, ScratchSize=0, Spill=0, also identical across
masks. The non-preshuffled V1 `Lb0ELi1` is unused at runtime.)

### Bench results (TFLOPS — sorted asc per mask, median bolded)

| Mask | Run sorted (TFLOPS)                                      | Median       | Spread % | Δ vs mask=0 |
|-----:|----------------------------------------------------------|-------------:|---------:|------------:|
| 0    | 2519.77 / 2520.08 / **2523.90** / 2524.24 / 2550.05      | **2523.90**  | +1.20 %  | —           |
| 1    | 2513.59 / 2518.72 / **2521.19** / 2523.83 / 2528.86      | **2521.19**  | +0.61 %  | **−0.11 %** |
| 2    | 2506.05 / 2520.28 / **2523.41** / 2524.22 / 2560.59      | **2523.41**  | +2.18 %  | **−0.02 %** |
| 3    | 2504.58 / 2520.21 / **2520.33** / 2523.37 / 2523.55      | **2520.33**  | +0.76 %  | **−0.14 %** |
| 4    | 2514.51 / 2515.86 / **2522.84** / 2525.35 / 2562.00      | **2522.84**  | +1.89 %  | **−0.04 %** |

All 4 mask deltas are negative-but-trivial (-0.02 % to -0.14 %), i.e.
the barrier is doing nothing useful here. The within-mask spreads
(0.61 %–2.18 %) dominate the cross-mask deltas by an order of magnitude
— there is no signal to recover.

## Bail trigger — Phase 2 SKIPPED

Per the agent prompt's bail conditions:

> "If all 4 mask values wash within ±1% of baseline at 8B Gate/Up under
> strict SCLK, REFUTE — confirms r49b_findings.md's prior expectation
> that soft barriers have <1% effect on this kernel."

The wash is unambiguous (worst delta -0.14 %), so no Phase 2 sweep was
launched.

## Correctness smoke (mask=2, MFMA-may-cross)

3 independent runs at 8B Gate/Up with `MXFP8_CHECK=1`:

| Run | TFLOPS  | Max abs | SNR (dB) | Pass rate          | Result |
|----:|--------:|--------:|---------:|-------------------:|--------|
| 1   | 2104.36 | 0.0309  | 49.61    | 58720256/58720256  | PASS   |
| 2   | 2108.87 | 0.0309  | 49.61    | 58720256/58720256  | PASS   |
| 3   | 2135.08 | 0.0309  | 49.61    | 58720256/58720256  | PASS   |

Bit-deterministic across 3 runs (max-abs/SNR/pass-rate identical).
Threshold 48.0 dB; SNR 49.61 dB clears comfortably. Macro is safe to
leave in tree (default OFF).

## Diagnosis

Two reinforcing facts:

1. **Build remarks identical across all 5 mask values.** The barrier
   does not change register pressure, scratch, occupancy, or even SGPR
   count. So whatever scheduler reordering the barrier permits or
   forbids, post-RA spill placement is unaffected. The K-loop body
   register live-range graph in this kernel is already pinned by the
   surrounding `s_barrier()` / `s_setprio()` / `s_waitcnt lgkmcnt(0)`
   sequence, which is a much stronger pipeline gate than `sched_barrier`.

2. **No measurable runtime delta.** The pre-existing
   `RRR_SCHED_BARRIER() = sched_barrier(0)` at the inter-quartet boundaries
   (lines 608, 646 of the baseline body) and the explicit `s_barrier()` /
   `s_waitcnt lgkmcnt(0)` pair before each MFMA already dominate any
   scheduler latitude that an additional inter-phase `sched_barrier`
   could constrain. The compiler's scheduling around these sites is
   already constrained tighter than any new mask we can add.

Combined, the kernel's load→MMA scheduling is already saturated by
explicit synchronization primitives (`s_barrier`, `s_waitcnt`,
`s_setprio`, `RRR_SCHED_BARRIER`); adding a pure scheduler hint between
them is below the noise floor of strict-SCLK measurement.

## Outcome — what landed

- `MXFP8_RRR_SCHED_BARRIER` macro and 4 mask arms added to
  `analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc`
  (default 0 = OFF; baseline behavior unchanged byte-for-byte). Insertion
  point: between the post-load `s_barrier()` and the
  `s_waitcnt lgkmcnt(0)` / `s_setprio(1)` pair that opens each of the 4
  MFMA dispatch windows inside the baseline `do_k_iter` body.
- Macro left in tree as documented dead code so future researchers can
  re-run the experiment in 1 line if a future toolchain changes the
  scheduling topology.

## Lever closure

With both R49 Dev B (noinline phase split — catastrophic, -97 % to -99 %)
and R50 Dev D (`sched_barrier` mask sweep — wash, ±0.15 %) refuted, the
**"split do_k_iter into a load-issue phase and an MMA-dispatch phase"**
hypothesis family is closed. The MXFP8 RRR K-loop is structurally bound
by the existing s_barrier / s_waitcnt / s_setprio pipeline; finer
scheduler hints have no leverage, and coarser tools (function call) hit
the bimodal-spill cliff R48G already mapped out.

## Files

- `r50d_phase1_bench.sh` — bench driver (5 masks × 5 runs × 1 cell)
- `r50d_results/r50d_phase1_orchestrator.log` — orchestrator stdout
- `r50d_results/r50d_mask{0,1,2,3,4}_8B_GateUp.log` — per-mask run logs
- `r50d_results/build_mask{0,1,2,3,4}_8B_GateUp.log` — per-mask build remarks
- `r50d_results/r50d_smoke_correctness_mask2.log` — 3-run correctness
- `r50d_results/build_smoke_default.log` — full resource report (default)
- `r50d_results/build_smoke_mask2.log` — full resource report (mask=2)
- `rrr_mxfp8_exact_8wave_fastpath.inc` — `MXFP8_RRR_SCHED_BARRIER` arms
  added (default OFF)
- `r50d_phase2_bench.sh` — Phase 2 driver (NOT run; kept for posterity)
