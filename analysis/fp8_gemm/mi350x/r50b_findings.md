# R50 Dev B — MXFP8 RRR per-shape U=2 sweet-spot at 8B Gate/Up — REFUTED

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 03c94ab3
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=2
**Lever:** `MXFP8_RRR_MAIN_UNROLL` per-shape selection at 8B Gate/Up RRR
(4096×14336×4096), the only RRR HEADROOM cell with structural spill risk
identified by R48 Dev D (90.4% vs 93.5% ceiling, +3.1pp gap).

## TL;DR — VERDICT: REFUTED

No per-shape `MXFP8_RRR_MAIN_UNROLL` value beats the U=0 baseline by
≥+1.5% with spread <2% at 8B Gate/Up RRR. The R48E/G **bimodal compiler
spill threshold at N≥2 is fully reproduced under strict 5×/30s/60s SCLK
protocol on GPU 2** (this cycle's per-agent allocation):

- U=0 (baseline, no `#pragma unroll`): **2500.8 TFLOPS** (median of 5)
- U=1 (`#pragma unroll 1` = explicit no-unroll): **2540.0 TFLOPS**
  (+1.57%, but spread 2.35% > 2% protocol gate → fails clean-win bar)
- U=2: **776.8 TFLOPS** (-68.94%, scratch 208 B/lane, 67 reg spill)
- U=4: **778.0 TFLOPS** (-68.89%, same spill profile)
- U=8: **778.2 TFLOPS** (-68.88%, same spill profile)

Phase 2 (regression sweep on the other 6 RRR shapes) **NOT EXECUTED**:
no Phase 1 candidate cleared the SHIP gate.

Default `MXFP8_RRR_MAIN_UNROLL=0` left unchanged in the tree. The
HEADROOM at 8B Gate/Up RRR remains structural spill at any unroll N≥2,
not a tunable factor — closes this lever for the R50 cycle.

## Hypothesis under test (per R49C §7.1, repeated by R50 orchestrator)

> "U=2 specifically at 8B Gate/Up may unlock a sweet spot not yet covered.
> One step above the no-spill regime (U=1) but with potential to keep
> more MMAs in flight at a tolerable spill cost."

This was based on the conjecture that **at this specific shape's
register-pressure contour** (M=4096, N=14336, K=4096 → 56 K-iterations,
B-side-heavy live ranges), the compiler scheduler might find a
register-budget local minimum at U=2 that it couldn't at U=4/8 (already
covered by R48E/G). The structural data (below) refutes this hypothesis
decisively: the spill commit at any N≥2 is exactly identical across
U∈{2,4,8} — same VGPRs (256), same scratch (208 B/lane), same 67 VGPR
spill register count. The compiler's heuristic does not have a per-shape
register-pressure tuning path through the unroll-factor knob.

## Method

### 1.1 Build / bench

Strict SCLK protocol (NON-NEGOTIABLE per orchestrator):
- 5 runs per cell
- 30s cooldown between runs
- 60s rebuild cooldown between U values
- Isolated GPU (HIP_VISIBLE_DEVICES=2, no other agent on GPU 2 this cycle)
- MXFP8_WARMUP=100, MXFP8_ITERS=200, MXFP8_PRESHUFFLE_QUANT=1
- Median of 5 runs as score; spread = (max-min)/median

For each U ∈ {0, 1, 2, 4, 8}:
```bash
rm -f tk_mxfp8_layouts*.so
make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=4096 -DN_DIM=14336 -DK_DIM=4096 \
              -DMXFP8_RRR_MAIN_UNROLL=$U" > build.log 2>&1
HIP_VISIBLE_DEVICES=2 MXFP8_BUILD_M=4096 MXFP8_BUILD_N=14336 \
    MXFP8_BUILD_K=4096 MXFP8_WARMUP=100 MXFP8_ITERS=200 \
    MXFP8_LAYOUTS=rrr MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py 4096 14336 4096
```
Build remarks via `-Rpass-analysis=kernel-resource-usage` on the
`rrr_exact_8wave_scaled_kernel` symbol (`rrr_mxfp8_exact_8wave_fastpath.inc:145`).

### 1.2 Macro path verification

Before benching, verified `rrr_mxfp8_exact_8wave_fastpath.inc` lines
45–53:
```c
#ifndef MXFP8_RRR_MAIN_UNROLL
#define MXFP8_RRR_MAIN_UNROLL 0
#endif

#if MXFP8_RRR_MAIN_UNROLL > 0
#define MXFP8_RRR_PRAGMA_UNROLL_MAIN _Pragma(TK_STRINGIFY(unroll MXFP8_RRR_MAIN_UNROLL))
#else
#define MXFP8_RRR_PRAGMA_UNROLL_MAIN
#endif
```
Macro is intact and accepts arbitrary positive integers via
`TK_STRINGIFY`. Site-of-use is line 636 (the `for (int kp=0; kp<k_pairs;
kp++)` loop). U=0 emits no pragma (compiler default). Default unchanged.

## 2. Results

### 2.1 Per-U bench medians + build remarks (8B Gate/Up RRR)

5 runs/cell, strict SCLK, GPU 2:

| U | med TF | min | max | spread% | Δ vs base | VGPRs | SGPRs | Scratch | Occ | VGPR Spill | SNR | det |
|--:|-------:|----:|----:|--------:|----------:|------:|------:|--------:|----:|-----------:|----:|----:|
|  0 | 2500.8 | 2482.2 | 2537.4 | 2.21% | +0.00% | 254 | 46 |   0 | 2 |  0 | 49.6 dB | PASS |
|  1 | 2540.0 | 2485.0 | 2544.8 | 2.35% | **+1.57%** | 254 | 46 |   0 | 2 |  0 | 49.6 dB | PASS |
|  2 |  776.8 |  775.2 |  778.0 | 0.36% | **-68.94%** | 256 | 62 | **208** | 2 | **67** | 49.6 dB | PASS |
|  4 |  778.0 |  777.7 |  778.1 | 0.05% | -68.89% | 256 | 62 | 208 | 2 | 67 | 49.6 dB | PASS |
|  8 |  778.2 |  773.3 |  780.1 | 0.88% | -68.88% | 256 | 62 | 208 | 2 | 67 | 49.6 dB | PASS |

Individual runs (TFLOPS):
- U=0: 2536.25, 2500.76, 2492.45, 2537.40, 2482.19
- U=1: 2529.68, 2540.00, 2544.78, 2542.62, 2485.02
- U=2:  778.05,  776.84,  776.43,  775.22,  777.46
- U=4:  777.74,  778.13,  778.04,  777.73,  777.97
- U=8:  778.25,  773.27,  777.55,  778.73,  780.13

### 2.2 Structural finding (cross-cycle reproduction)

R48E/G's bimodal compiler threshold is **fully reproduced** at GPU 2
under R50's stricter 5-run protocol:

- **U ∈ {0, 1}**: VGPRs=254, SGPRs=46, Scratch=0, Spill=0, Occ=2 → ~2500
  TFLOPS. The U=1 explicit `#pragma unroll 1` is a no-op equivalent to
  the absent pragma at U=0 (compiler default). Codegen is identical at
  the resource-usage level.
- **U ∈ {2, 4, 8}**: VGPRs=256, SGPRs=62, Scratch=**208 B/lane**,
  VGPR Spill=**67 reg**, Occ=2 → ~778 TFLOPS (-68.9% across all three).
  The compiler commits to spilling scale-pack live ranges to scratch
  the moment any `#pragma unroll N≥2` is present, regardless of the
  value of N. The spill profile is **shape-invariant and N-invariant**
  above the threshold.

This refutes the per-shape-tuning hypothesis: the compiler's
register-pressure heuristic at this loop body has no exploitable degree
of freedom in the unroll-factor knob.

### 2.3 SHIP gate evaluation

Phase 1 ship gate (per orchestrator):
> Candidate U gives ≥+1.5% on 8B Gate/Up RRR with spread <2% AND build
> remarks show no spill regression vs baseline U=4 (treat: spill bytes
> ≤ baseline spill bytes, occupancy ≥ baseline occupancy).

Note on baseline: orchestrator wrote "U=4 default" but the kernel's
actual production default is `MXFP8_RRR_MAIN_UNROLL=0` (no pragma).
Re-baselining against U=0 (the real production default), the gate
evaluation is:

| U | ≥+1.5%? | spread<2%? | spill ≤ U=0 (=0)? | occ ≥ U=0 (=2)? | Verdict |
|--:|:-------:|:----------:|:----------------:|:--------------:|:-------:|
| 1 | YES (+1.57%) | **NO (2.35%)** | YES (0) | YES (2) | **REFUTED** (spread fail) |
| 2 | NO (-68.94%) | YES | **NO (208)** | YES | REFUTED |
| 4 | NO (-68.89%) | YES | NO (208) | YES | REFUTED |
| 8 | NO (-68.88%) | YES | NO (208) | YES | REFUTED |

No candidate U passes Phase 1 ship gate. Phase 2 (regression sweep on
other 6 RRR shapes) skipped per workflow step 4 ("only if Phase 1 found
a candidate").

### 2.4 On the U=1 marginal-positive observation

U=1 measures +1.57% over U=0 with overlapping run distributions
(U=0 max = 2537.4, U=1 min = 2485.0). Build remarks are identical
between U=0 and U=1 (VGPRs=254, Scratch=0, no spill), confirming the
generated machine code is the same. The +1.57% delta is therefore
**SCLK noise / measurement variance** (spread 2.21%/2.35% on a
5-run median), not a compiler effect. Cannot ship as a kernel change
because there is no kernel change.

## 3. Updates to R48G's table

Filling R48G §2.1 with R50B's strict-protocol re-runs (5 runs vs
R48E/G's 3 runs, MXFP8_PRESHUFFLE_QUANT=1):

| U  | VGPRs | Scratch | 8B Gate/Up TFLOPS | Δ% | Source |
|---:|------:|--------:|------------------:|---:|--------|
|  0 | 254 |   0 | 2491.5 / **2500.8** | +0.00% | R48E base / **R50B** |
|  1 | 254 |   0 |   — / **2540.0**    | **+1.57%** (noise) | new — **R50B** |
|  2 | 256 | 208 |  777.3 / **776.8**  | -68.94% | R48E / **R50B** (matches) |
|  4 | 256 | 208 |  778.7 / **778.0**  | -68.89% | R48E / **R50B** (matches) |
|  8 | 256 | 208 |  778.2 / **778.2**  | -68.88% | R48E / **R50B** (matches) |
| 16 | 256 | 208 |  779.2 /     —      | -68.71% | R48G |
| 32 | 256 | 208 |  749.8 /     —      | -69.90% | R48E |

R50B fills the R48G-missing **U=1 at 8B Gate/Up** data point.

## 4. Outcome

- **No source landed.** `MXFP8_RRR_MAIN_UNROLL` macro left at default 0.
- **Hypothesis closed.** Per-shape `MXFP8_RRR_MAIN_UNROLL` selection at
  8B Gate/Up RRR is REFUTED. The bimodal compiler-spill threshold is a
  **structural property of the do_k_iter inlined body + scale-pack live
  ranges**, not a per-shape register-pressure inflection. R49B's
  noinline phase-split was the right *direction* (break the live-range
  graph at a function boundary so the compiler can drop live ranges
  earlier) but the wrong *implementation* (catastrophic at every other
  shape because non-spilling cells benefit from the all-inlined codegen).
- **Lever exhausted.** The MXFP8 RRR K-pair loop body cannot be unrolled
  via `#pragma unroll` without paying the 208 B/lane scratch cost. Any
  future attempt at recovering loop-control overhead at 8B Gate/Up RRR
  must change the **body** of `do_k_iter` (fewer scale packs live at
  once) rather than its **iteration count**.

## 5. Recommendations for R51+

1. **CLOSE** `MXFP8_RRR_MAIN_UNROLL` per-shape exploration. The bimodal
   threshold is now reproduced 3 times (R48E, R48G, R50B) under
   increasing protocol stringency. No experiment in the unroll-factor
   axis is going to find a sweet spot.
2. **The +3.1pp HEADROOM at 8B Gate/Up RRR remains real** but its lever
   is not RRR_MAIN_UNROLL. Candidate alternatives:
   - **B-side LDS bank conflict at N=14336** (R48D §2 candidate, never
     directly investigated). At BN=128 with 8 waves, the B-tile shared-
     memory layout may pessimize at this N specifically.
   - **Scale-pack live-range hoisting** in `do_k_iter` source: rewrite
     the lambda so b0_scale_packs and b1_scale_packs are not both live
     simultaneously for the cA/cB MFMAs (mirror what R49B's noinline
     attempted, but at the source level rather than via attribute).
   - **K-iter fission** (covered by R49B for the wrong reason — the
     live-range argument was right, the noinline split was the wrong
     mechanism): consider source-level fission of the 4-MFMA quartet
     into two 2-MFMA halves with `__restrict__`-annotated args so the
     compiler scheduler treats them as separate live-range domains
     within an inlined body. This is a deeper refactor than R49B.
3. **Per-shape compile-time gate (was step 5 in workflow): not
   implemented** because no candidate U passed Phase 1. If a future
   cycle finds a per-shape sweet spot in some other lever, the
   `MXFP8_BUILD_M/N/K` macros are already plumbed into the build
   command, so a `#if M_DIM == 4096 && N_DIM == 14336 && K_DIM == 4096`
   selector inside the kernel preamble would compile cleanly without
   touching the universal default.

## Files

- `r50b_bench.sh` — bench driver (5 runs/cell, 30s cooldown, 60s rebuild cooldown)
- `r50b_bench_phase1.run.log` — driver stdout
- `r50b_results/8B_GateUp_U{0,1,2,4,8}_run{1..5}.log` — per-run TFLOPS
- `r50b_results/8B_GateUp_U{0,1,2,4,8}_check.log` — SNR + det (1 each)
- `r50b_results/8B_GateUp_U{0,1,2,4,8}_build.log` — build remarks
  (kernel-resource-usage)
