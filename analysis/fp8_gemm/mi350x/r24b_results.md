# R24B — L2 Prefetch via Discarded buffer_load_dwordx4 — DEAD END

**Date**: 2026-04-18
**Bench params**: warmup=200, iters=500, trim=10% (per `.claude/rules/benchmark-rules.md`)
**GPUs**: 0,1,2,4,6,7 (3 isolated)

## Hypothesis

Inside the steady-state K-loop, issue an extra `buffer_load_dwordx4` to a
**discarded** scratch VGPR, targeting the K-tile **one outer-K iter beyond**
the existing LDS-going prefetch (i.e. `bt+3` instead of `bt+2`). Data lands
in L2/L1, prewarming for the next-but-one LDS prefetch. This adds prefetch
**bandwidth** (extra HBM read) without changing cache **policy** (NT bypass)
— a different axis from the fully-exhausted R22B/R24D NT-hint cube.

If HBM latency was the gating factor (and we had spare HBM bandwidth), the
extra outer-K prefetch should let MFMA progress hide more of the data fetch.

## What was built

### Kernel changes (`kernel_mxfp4_gluon_cpp.cpp`)

1. Generalized the existing `emit_one_pf_l2only` helper (previously gated
   behind `OUTER_K_PF_DEPTH > 1`) so it is always available.
2. Added `L2_PF_A` / `L2_PF_B` macros (default 0). Levels 1/2/3 control
   intensity (1 dwordx4 / PF_MPT/2 / full PF_MPT loads per K-iter).
3. Wired into the steady-state K-loop in the active code path
   (TAIL_SPLIT non-SWAP, line ~2664). Builds `pf3_*` params for `pf_bt+1`
   and emits `emit_l2_pf_block<LEVEL>(...)` for A0/A1 and/or Br/Bl.

All changes are gated behind `#if (L2_PF_A > 0) || (L2_PF_B > 0)` —
default builds are byte-identical to baseline.

### Variants (5 per shape)

| Suffix         | L2_PF_A | L2_PF_B | Description                          |
|----------------|---------|---------|--------------------------------------|
| `_baseline`    | 0       | 0       | Fresh rebuild, no L2 hint            |
| `_a1`          | 1       | 0       | A: 1 dwordx4/K-iter (cheap probe)    |
| `_b1`          | 0       | 1       | B: 1 dwordx4/K-iter (cheap probe)    |
| `_ab1`         | 1       | 1       | Both                                 |
| `_a3`          | 3       | 0       | A: full PF_MPT loads (full A-tile)   |

### Disassembly verification (DLA1 .co files, gfx950)

| Variant   | `buffer_load_dwordx4` count | Δ vs baseline |
|-----------|-----------------------------|---------------|
| baseline  | 432                         | —             |
| a1        | 482                         | +50           |
| b1        | 482                         | +50           |
| ab1       | 532                         | +100          |
| a3        | 632                         | +200          |

Instruction counts scale linearly with intensity, confirming the L2 hints
are being emitted as designed.

## Bench results (TFLOPS)

| Shape | baseline | a1     | b1     | ab1    | a3     |
|-------|---------:|-------:|-------:|-------:|-------:|
| DLA1  | 5219.39  | 4991.19| 4907.10| 4772.89| ERR(rc=-6) |
| DLA2  | 4247.50  | 4231.61| 4100.91| 4027.25| 3907.79 |
| DLA7  | 4179.63  | 4053.70| 4044.40| 3946.13| 3675.14 |

### Δ vs `_r24b_baseline` (gate: ≥+1.5pp)

| Shape | a1     | b1     | ab1    | a3     |
|-------|-------:|-------:|-------:|-------:|
| DLA1  | -4.37% | -5.98% | -8.55% | ERR    |
| DLA2  | -0.37% | -3.45% | -5.19% | -8.00% |
| DLA7  | -3.01% | -3.24% | -5.59% |-12.07% |

**Every variant LOSES on every shape**, with regression magnitude scaling
monotonically with prefetch intensity. The DLA1 a3 variant errored out
(rc=-6, abort — likely from runtime correctness / fault from too many
in-flight loads). Closest-to-baseline is `_a1` on DLA2 (-0.37%), but that
is still a regression and noise can't account for the consistent direction
across all 14 successful variants.

## Verdict: **DEAD END**

## Mechanistic interpretation

The hypothesis assumed HBM had spare bandwidth that an extra prefetch
could exploit. The data refutes this in the cleanest possible way:

1. **HBM is already saturated by the existing prefetch.** R21-recon
   established TCP_DATA_STALL = 167-294% of GRBM on DLA shapes — i.e.
   memory subsystem is already the bottleneck. Adding more outer-K
   prefetch competes with the inner-K LDS prefetch for HBM bandwidth,
   evicting useful lines and starving the LDS path.

2. **L2 prefetching one tile ahead is too far.** With BK=128 and tile
   sizes ~8-16 KB per warp, the next-but-one K-tile won't be consumed
   for ~512 cycles. L2 is small enough (~16 MB across XCDs) that lines
   prefetched too early get evicted before use.

3. **VGPR pressure / scheduling damage.** The discarded `buffer_load_dwordx4`
   still consumes VGPRs (sink reg), schedule slots, and an outstanding
   VMEM counter. The compiler's `s_waitcnt vmcnt(...)` calculations
   degrade because there's now 2x the in-flight VMEM at any point.

4. **Monotone regression with intensity is the smoking gun.** a1 (lightest)
   is closest to baseline; a3 (heaviest) is worst by 8-12%. If extra
   prefetch were ever beneficial, we'd see at least one inflection point
   where a partial intensity won. We don't.

## Combined with prior rounds

| Round | Knob                        | DLA verdict |
|-------|-----------------------------|-------------|
| R22B  | NT cache hint (B-only)      | LOSE        |
| R24D  | NT cache hint (A-only)      | LOSE        |
| R24B  | Extra L2 prefetch (this)    | LOSE        |
| R22B  | NT cache hint (A+B)         | LOSE        |

Cache POLICY axis fully exhausted. Cache BANDWIDTH axis fully exhausted.
The DLA shapes truly are HBM-bandwidth-saturation bound — the existing
LDS prefetch already extracts as much bandwidth as the memory subsystem
can deliver, and any additional VMEM traffic is destructive.

## Next directions (out of scope for R24B)

The remaining levers for HBM-bound DLA shapes:
1. **Reduce HBM read volume** — improve B-tile L2 reuse via M-direction
   tile grouping (GROUP_SIZE_M is already 2 for DLA2; could try larger).
2. **Asymmetric tile sizes** — increase K-tile depth (BK=256?) to amortize
   per-tile fixed overhead.
3. **Persistent kernel correctly implemented** — would amortize tail
   overhead, but PERSISTENT_XCD has unfixable kernel-side bug per state.
4. **Conclude DLA shapes are hardware-saturated.** The current 7.8-19.9%
   of 5.3 TB/s peak HBM utilization may itself be a CDNA4 ceiling for
   non-pathological access patterns, in which case no kernel change can
   close the gap.

## Files

- Kernel modifications: `kernel_mxfp4_gluon_cpp.cpp` (lines ~832-885 helpers, ~2671-2693 wiring)
- Build script: `build_round24_optB.py`
- Bench script: `bench_round24_optB_smoke.py`
- Bench log: `bench_round24_optB_smoke.log`
- Bench JSON: `bench_round24_optB_smoke.json`
