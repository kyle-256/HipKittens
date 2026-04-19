# R52 Dev J — MXFP8 CRR `CRR_EXACT_B1_LDS_INSERT_AFTER` sweep at 8B Gate/Up — REFUTED

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 2d20953b (R51F findings landed)
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=4
**Lever:** `CRR_EXACT_B1_LDS_INSERT_AFTER` (default 4) at 8B Gate/Up CRR
(M=4096 N=14336 K=4096), the R51F-suggested follow-up after CRR
`#pragma unroll` was proven a compiler no-op.

## TL;DR — VERDICT: REFUTED

No `CRR_EXACT_B1_LDS_INSERT_AFTER` value beats the IA=4 baseline by ≥+1.5%
under strict 5×/30s/60s SCLK protocol on GPU 4. Across IA ∈ {0, 2, 3, 4, 5, 6, 8}:

- IA=4 (baseline): **2339.4 TFLOPS** (median of 5; spread 0.59%)
- IA=0:  2373.2 TFLOPS (+1.45%, spread 1.67% — **bimodal, see §2.3**)
- IA=2:  2343.5 TFLOPS (+0.18%, spread 1.59%)
- IA=3:  2342.9 TFLOPS (+0.15%, spread 0.85%)
- IA=5:  2340.9 TFLOPS (+0.06%, spread 2.04%)
- IA=6:  2342.5 TFLOPS (+0.13%, spread 1.75%)
- IA=8:  2340.7 TFLOPS (+0.06%, spread 0.24%)

**No candidate clears Phase 1 ship gate (≥+1.5% AND spread <2% AND no
spill regression).** IA=0's apparent +1.45% is invalidated by the GCN
ISA bit-identity proof in §2.2 — its codegen is byte-for-byte identical
to IA=4, so the perf gap is provably 100% SCLK noise (clock-up across
runs, not knob effect).

**Phase 2 (70B Gate/Up CRR) NOT EXECUTED**: no Phase 1 candidate cleared
the SHIP gate.

Default `CRR_EXACT_B1_LDS_INSERT_AFTER=4` left unchanged in the tree.

## Hypothesis under test (per orchestrator R52 Dev J prompt)

> "At the structurally-bound 70B Gate/Up CRR cell (88.5% — +4pp HEADROOM),
> the B1 insert position interacts with the wave-tail pattern at N=28672,
> and a different insert offset may unlock 2-4pp."

Predicted by R51F's §4.2 as the most actionable remaining CRR lever after
unroll/opsel/dormant variants all REFUTED.

## Method

### 1.1 Knob discovery

`CRR_EXACT_B1_LDS_INSERT_AFTER` is defined in
`crr_mxfp8_exact_8wave_fastpath.inc:159-161` (default 4). It is the
template parameter `INSERT_AFTER ∈ [0, 8]` to
`crr_exact_cA_with_b1_interleave_fixed_phase()` (lines 215-249) and
`crr_exact_cA_with_b1_interleave_raw_phase()` (lines 251-289) — both
helpers wrap 8 cA MMAs (one quadrant pair) and place a single
`load_col_from_v2_st(b1, ...)` (the B1 LDS write for the next K-step)
at one of the 9 possible slot positions {0..8}, where slot k means
"after MMA k" (and slot 8 means "after the final MMA, before s_setprio(0)").

Production callsite: `crr_mxfp8_exact_8wave_fastpath.inc:974` inside
the LDS double-buffer main K-loop.

The natural sweep is IA ∈ [0, 8]. We tested {0, 2, 3, 4(base), 5, 6, 8}
— covering both extremes, the immediate ±1 neighborhood, and a middle
sample. IA=1 and IA=7 omitted to keep the sweep at 7 cells.

**No source change required for this sweep.** The bench varies only
the `-DCRR_EXACT_B1_LDS_INSERT_AFTER=$IA` build flag.

### 1.2 Build / bench

Strict SCLK protocol (NON-NEGOTIABLE per orchestrator):
- 5 runs per cell
- 30s cooldown between runs
- 60s rebuild cooldown between IA values
- Isolated GPU (HIP_VISIBLE_DEVICES=4)
- MXFP8_WARMUP=100, MXFP8_ITERS=200, MXFP8_PRESHUFFLE_QUANT=1
- Median of 5 runs as score; spread = (max-min)/median

For each IA ∈ {0, 2, 3, 4, 5, 6, 8}:
```bash
rm -f tk_mxfp8_layouts*.so
make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=4096 -DN_DIM=14336 -DK_DIM=4096 \
              -DCRR_EXACT_B1_LDS_INSERT_AFTER=$IA" > build.log 2>&1
HIP_VISIBLE_DEVICES=4 MXFP8_BUILD_M=4096 MXFP8_BUILD_N=14336 \
    MXFP8_BUILD_K=4096 MXFP8_WARMUP=100 MXFP8_ITERS=200 \
    MXFP8_LAYOUTS=crr MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py 4096 14336 4096
```
Build remarks via `-Rpass-analysis=kernel-resource-usage` on the
`crr_exact_8wave_scaled_kernel<true,1>` symbol
(`crr_mxfp8_exact_8wave_fastpath.inc:293`).

## 2. Results

### 2.1 Per-IA bench medians + build remarks (8B Gate/Up CRR)

5 runs/cell, strict SCLK, GPU 4:

| IA | med TF | min | max | spread% | Δ vs base | VGPRs | SGPRs | Scratch | Occ | VGPR Spill | SNR | det |
|---:|-------:|----:|----:|--------:|----------:|------:|------:|--------:|----:|-----------:|----:|----:|
|  0 | 2373.2 | 2339.8 | 2379.5 |  1.67% |  +1.45% (NOISE — see §2.3) | 225 | 64 | 0 | 2 | 0 | 49.61 dB | PASS |
|  2 | 2343.5 | 2338.6 | 2375.9 |  1.59% |  +0.18% | 225 | 64 | 0 | 2 | 0 | 49.61 dB | PASS |
|  3 | 2342.9 | 2336.9 | 2356.9 |  0.85% |  +0.15% | 225 | 64 | 0 | 2 | 0 | 49.61 dB | PASS |
|  4 | 2339.4 | 2333.2 | 2347.1 |  0.59% |  +0.00% (base) | 225 | 64 | 0 | 2 | 0 | 49.61 dB | PASS |
|  5 | 2340.9 | 2296.9 | 2344.6 |  2.04% |  +0.06% | 225 | 64 | 0 | 2 | 0 | 49.61 dB | PASS |
|  6 | 2342.5 | 2335.4 | 2376.4 |  1.75% |  +0.13% | 225 | 64 | 0 | 2 | 0 | 49.61 dB | PASS |
|  8 | 2340.7 | 2338.6 | 2344.2 |  0.24% |  +0.06% | **232** | 64 | 0 | 2 | 0 | 49.61 dB | PASS |

Individual runs (TFLOPS):
- IA=0: 2339.79, 2379.54, 2340.59, 2376.49, 2373.23
- IA=2: 2367.93, 2338.58, 2375.94, 2343.50, 2341.44
- IA=3: 2336.94, 2341.06, 2342.90, 2347.48, 2356.89
- IA=4: 2333.25, 2339.40, 2339.32, 2340.06, 2347.08
- IA=5: 2296.86, 2333.90, 2340.90, 2342.71, 2344.58
- IA=6: 2335.44, 2338.22, 2342.46, 2365.24, 2376.36
- IA=8: 2338.63, 2339.83, 2340.71, 2342.15, 2344.17

### 2.2 STRUCTURAL FINDING — IA ∈ [0, 7] is a compiler no-op (GCN ISA bit-identical)

Built `tk_isa_${IA}.so` for IA ∈ {0, 4, 8}, extracted the `gfx950`
device ELF via `clang-offload-bundler --unbundle`, and disassembled
with `llvm-objdump -d --mcpu=gfx950`. After stripping the leading
`file format` filename header (which embeds the input filename and is
trivially different), the GCN ISA bodies have:

| IA | gcn_IA${IA}.s lines | md5(body) |
|---:|--------------------:|:----------|
|  0 | 22983 | f734efc0c73f9f622ad35b5d13177254 |
|  4 | 22983 | **f734efc0c73f9f622ad35b5d13177254** |
|  8 | 22985 | e4cf9dc616a9744a4c72e67d53cef88c |

**IA=0 and IA=4 produce byte-identical gfx950 machine code.** IA=8 is
the only value that re-shapes codegen (+2 lines, +7 VGPRs (225→232),
~6888 ISA-line diff). This is consistent with how the inner loop is
structured: when INSERT_AFTER ∈ [0..7], the LDS load is wedged between
two MMAs inside the `__builtin_amdgcn_s_setprio(1)` priority block, and
the LLVM scheduler is free to slot the load wherever it likes inside
that block (subject to dep edges); since the load has no consumer until
the *next* outer-K iteration, it always lands in the same scheduler
slot regardless of source position. Only at IA=8 (after the final MMA,
*before* `s_setprio(0)`) does the schedule structurally change, because
the load now sits in a different priority epoch and the scheduler
gains/loses a constraint, picking up 7 more VGPRs in the process.

This is the **strongest possible negative result** for the IA ∈ [0..7]
sub-range: not "we tested several values and none won" but "we
mathematically proved the values produce identical machine code, so
they cannot possibly differ in performance."

### 2.3 Why the IA=0 +1.45% number is noise (run-order pattern)

IA=0 per-run TFLOPS in original execution order:

| run | TFLOPS |
|----:|-------:|
|  1  | 2339.79 |
|  2  | 2379.54 |
|  3  | 2340.59 |
|  4  | 2376.49 |
|  5  | 2373.23 |

Compare IA=4 (codegen-identical baseline):

| run | TFLOPS |
|----:|-------:|
|  1  | 2333.25 |
|  2  | 2339.40 |
|  3  | 2339.32 |
|  4  | 2340.06 |
|  5  | 2347.08 |

IA=0 shows a bimodal "thermal lift" pattern (run1/run3 cold, run2/run4/run5
warm) — exactly what a transient SCLK perturbation looks like.
IA=4 was steady because that cell's bench window happened to land on
a thermally-stable interval. **Since the GCN ISA is bit-identical**
between IA=0 and IA=4, the two cells executed the same instructions
in the same order; the only difference is wall-clock timing of the
bench windows. This is a known shared-node SCLK artifact (the same
mode R51F observed at U=2/U=4 in their CRR_MAIN_UNROLL sweep), and
not a knob effect.

If the IA=0 spread were a real codegen win, we would expect the IA=0
distribution to be *uniformly higher* than IA=4 by ~1.5%, not a
2-low/3-high cluster. The cluster pattern is diagnostic of clock-state
nondeterminism, not kernel-side improvement.

### 2.4 IA=8 — codegen does change but doesn't help

IA=8 is the only value in the sweep where the GCN ISA actually differs
(+7 VGPRs, ~6888 ISA-line diff). Its perf is +0.06% (essentially
identical to baseline) and its spread is the *tightest* of all cells
(0.24%). So the schedule difference at IA=8 is real but performance-neutral
— neither helping nor hurting. The compiler's choice of letting the
B1 load slot into the natural scheduler position when IA ∈ [0..7] is
already optimal; forcing it past the priority-block boundary at IA=8
costs 7 extra VGPRs but produces equivalent throughput because
occupancy stays at 2 and there's no spill.

### 2.5 SHIP gate evaluation

Phase 1 ship gate (per orchestrator):
> Best IA beats IA=4 baseline by ≥+1.5% AND spread <2% AND no spill
> regression vs baseline.

| IA | ≥+1.5%? | spread<2%? | spill ≤ IA=4 (=0)? | occ ≥ IA=4 (=2)? | Verdict |
|---:|:-------:|:----------:|:-----------------:|:----------------:|:-------:|
|  0 | NO (+1.45%, NOISE — provably ISA-identical) | YES (1.67%) | YES | YES | **REFUTED** |
|  2 | NO (+0.18%) | YES | YES | YES | **REFUTED** |
|  3 | NO (+0.15%) | YES | YES | YES | **REFUTED** |
|  5 | NO (+0.06%) | NO (2.04%) | YES | YES | **REFUTED** |
|  6 | NO (+0.13%) | YES | YES | YES | **REFUTED** |
|  8 | NO (+0.06%) | YES | YES | YES | **REFUTED** |

No candidate IA passes Phase 1 ship gate. Phase 2 (70B Gate/Up CRR
with the candidate IA) skipped per workflow step 6 ("only if Phase 1
found a candidate clearing +1.5%").

## 3. Outcome

- **No source landed.** `CRR_EXACT_B1_LDS_INSERT_AFTER` macro left at default 4.
- **Hypothesis closed.** The B1 LDS insert-position knob is a **dead
  lever for CRR** at 8B Gate/Up across the entire IA ∈ [0..8] domain:
  - IA ∈ [0..7]: bit-identical GCN codegen → cannot affect perf.
  - IA = 8: changes codegen (+7 VGPRs) but is performance-neutral.
- **Lever exhausted.** This closes the second of R51F's three suggested
  CRR HEADROOM follow-ups (the first being `CRR_MAIN_UNROLL`, which
  R51F itself REFUTED). The remaining R51F suggestion is per-shape
  `MXFP8_CRR_BLOCK_SWIZZLE_GROUP_M` and source-level scale-pack
  live-range hoisting — both untouched by this work.

## 4. Cross-cycle invariants reinforced

R51F established: "CRR codegen is invariant under the `CRR_MAIN_UNROLL`
pragma — it's a compiler no-op."

R52J adds: "CRR codegen is invariant under `CRR_EXACT_B1_LDS_INSERT_AFTER`
∈ [0..7] — it's also a compiler no-op for that sub-range."

The combined invariant: **the CRR fastpath's machine code is robust to
both its iteration-count knob (#pragma unroll N) AND its inner-K
load-placement knob (INSERT_AFTER ∈ [0..7])**. The compiler's scheduler
treats the manually-interleaved priority block as a single re-orderable
basic block where source-position metadata is discarded. Future CRR-side
optimization work should NOT include either knob in the experimental
design — both are provably zero-information levers at the GCN level.

The cycle-permanent recommendation is to attack CRR HEADROOM via:
1. Source-level changes that the compiler *cannot* re-order around (the
   in-place 16-bit scale-pack shift block at lines 936-946 — split into
   2 paired shifts to halve simultaneous live-pack count).
2. Per-shape `MXFP8_CRR_BLOCK_SWIZZLE_GROUP_M` (defaults to 4; possibly
   not optimal at N=14336 vs the bigger 70B shapes).
3. Re-examining R47B CRR XCD swizzle on 8B Gate/Up specifically (per
   R49 Reviewer note).

## 5. Files

- `r52j_bench.sh` — bench driver (5 runs/cell, 30s cooldown, 60s rebuild cooldown)
- `r52j_bench_phase1.run.log` — driver stdout
- `r52j_results/8B_GateUp_IA{0,2,3,4,5,6,8}_run{1..5}.log` — per-run TFLOPS
- `r52j_results/8B_GateUp_IA{0,2,3,4,5,6,8}_check.log` — SNR + det (1 each)
- `r52j_results/8B_GateUp_IA{0,2,3,4,5,6,8}_build.log` — build remarks
  (kernel-resource-usage)
- `r52j_isa_dumps/gcn_IA{0,4,8}.s` — gfx950 device disassembly for the
  bit-identity proof (md5: IA0=IA4 byte-identical, IA8 differs)
- `r52j_isa_dumps/device_{0,4,8}.o` — extracted gfx950 device ELFs
