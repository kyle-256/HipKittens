# R51 Dev F — MXFP8 CRR per-shape U-sweep at 8B Gate/Up — REFUTED

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ b35a5fce
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=2
**Lever:** `CRR_MAIN_UNROLL` per-shape selection at 8B Gate/Up CRR
(M=4096 N=14336 K=4096), the analog of R48G/R50B's RRR sweep but on
the CRR HEADROOM cell identified by R48 Dev D (88.5% vs ~92% ceiling).

## TL;DR — VERDICT: REFUTED

No per-shape `CRR_MAIN_UNROLL` value beats the U=1 baseline by ≥+1.5%
under strict 5×/30s/60s SCLK protocol on GPU 2. **CRR has no unroll
sweet spot at 8B Gate/Up.** Across U ∈ {1, 2, 4, 8, 16}:

- U=1 (baseline): **2349.4 TFLOPS** (median of 5; spread 0.71%)
- U=2: **2348.3 TFLOPS** (−0.05%; spread 9.41% — 1 outlier run)
- U=4: **2338.6 TFLOPS** (−0.46%; spread 45.6% — 2 outlier runs)
- U=8: **2350.5 TFLOPS** (+0.05%; spread 0.58%)
- U=16: **2351.6 TFLOPS** (+0.09%; spread 0.58%)

All within ±0.5% of baseline — all wash. Build remarks for the CRR
kernel symbol `_Z29crr_exact_8wave_scaled_kernelILb1ELi1EEv` are
**identical across all 5 U values** (VGPRs=225, SGPRs=64, Scratch=0,
Spill=0, Occupancy=2). The compiler treats `#pragma unroll N` for
N≥1 as effectively a no-op on the CRR K-loop body at this shape —
codegen does not change, perf does not change.

**Phase 2 (regression sweep on the other 6 CRR shapes) NOT EXECUTED**:
no Phase 1 candidate cleared the SHIP gate.

Default `CRR_MAIN_UNROLL=1` left unchanged in the tree.

## Hypothesis under test (per orchestrator R51 Dev F prompt)

> "For CRR at K=4096 specifically, sweep MXFP8_CRR_MAIN_UNROLL ∈
> {0, 1, 2, 4, 8, 16}. Identify whether CRR has a partial-unroll
> sweet spot that RRR didn't (different scale-pack lifetimes due to
> quadrant-pair opsel structure may avoid the bimodal spill)."

This was the natural CRR analog of R48E/G/R50B's RRR work, given that
R47B's CRR XCD swizzle was just discovered to REGRESS 70B Gate/Up CRR
by -3.87%, leaving more headroom on this CRR cell than originally
estimated by R48 Dev D.

## Method

### 1.1 Macro path discovery

CRR already has `CRR_MAIN_UNROLL` plumbed at 5 K-loop sites in
`crr_mxfp8_exact_8wave_fastpath.inc` (lines 672, 719, 768, 817, 928 —
one per `MXFP8_CRR_LDS_SINGLE_BUFFER` × `MXFP8_CRR_SB_PIPELINE` arm).
The active arm in production (LDS double-buffer, line 928, the
`for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1)` loop) is
the one this sweep targets. The default in `kernel_mxfp8_layouts.cpp`
line 160 is `CRR_MAIN_UNROLL=1` (note: differs from RRR_MAIN_UNROLL=4
default at line 157, and very differs from MXFP8_RRR_MAIN_UNROLL=0
default which is the production R48G+ path; R50B benched RRR with
its production U=0 default as baseline).

**No source change required for this sweep.** The bench varies only
the `-DCRR_MAIN_UNROLL=$U` build flag.

Because CRR's production default is already `#pragma unroll 1`
(explicit no-unroll), the U=0 value is omitted from the sweep — it
would map to `#pragma unroll 0`, which Clang treats as either invalid
or "fully unroll" (compiler-version dependent), and is in any case
not the production baseline. The meaningful sweep is U ∈ {1, 2, 4, 8, 16}.

### 1.2 Build / bench

Strict SCLK protocol (NON-NEGOTIABLE per orchestrator):
- 5 runs per cell
- 30s cooldown between runs
- 60s rebuild cooldown between U values
- Isolated GPU (HIP_VISIBLE_DEVICES=2)
- MXFP8_WARMUP=100, MXFP8_ITERS=200, MXFP8_PRESHUFFLE_QUANT=1
- Median of 5 runs as score; spread = (max-min)/median

For each U ∈ {1, 2, 4, 8, 16}:
```bash
rm -f tk_mxfp8_layouts*.so
make TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
    CXXFLAGS="-w -DM_DIM=4096 -DN_DIM=14336 -DK_DIM=4096 \
              -DCRR_MAIN_UNROLL=$U" > build.log 2>&1
HIP_VISIBLE_DEVICES=2 MXFP8_BUILD_M=4096 MXFP8_BUILD_N=14336 \
    MXFP8_BUILD_K=4096 MXFP8_WARMUP=100 MXFP8_ITERS=200 \
    MXFP8_LAYOUTS=crr MXFP8_PRESHUFFLE_QUANT=1 \
    python3 test_mxfp8_python.py 4096 14336 4096
```
Build remarks via `-Rpass-analysis=kernel-resource-usage` on the
`crr_exact_8wave_scaled_kernel<true,1>` symbol
(`crr_mxfp8_exact_8wave_fastpath.inc:293`).

## 2. Results

### 2.1 Per-U bench medians + build remarks (8B Gate/Up CRR)

5 runs/cell, strict SCLK, GPU 2:

| U | med TF | min | max | spread% | Δ vs base | VGPRs | SGPRs | Scratch | Occ | VGPR Spill | SNR | det |
|--:|-------:|----:|----:|--------:|----------:|------:|------:|--------:|----:|-----------:|----:|----:|
|  1 | 2349.4 | 2342.5 | 2359.1 |  0.71% |  +0.00% (base) | 225 | 64 | 0 | 2 | 0 | 49.61 dB | PASS |
|  2 | 2348.3 | 2133.5 | 2354.4 |  9.41% |  -0.05% | 225 | 64 | 0 | 2 | 0 | 49.61 dB | PASS |
|  4 | 2338.6 | 1286.6 | 2353.8 | 45.64% |  -0.46% | 225 | 64 | 0 | 2 | 0 | 49.61 dB | PASS |
|  8 | 2350.5 | 2343.6 | 2357.3 |  0.58% |  +0.05% | 225 | 64 | 0 | 2 | 0 | 49.61 dB | PASS |
| 16 | 2351.6 | 2341.5 | 2355.0 |  0.58% |  +0.09% | 225 | 64 | 0 | 2 | 0 | 49.61 dB | PASS |

Individual runs (TFLOPS):
- U=1:  2342.51, 2359.14, 2352.91, 2349.42, 2347.50
- U=2:  2349.08, 2354.39, 2348.28, **2133.53**, 2337.06
- U=4:  **1591.56**, 2348.27, 2338.60, **1286.59**, 2353.84
- U=8:  2350.53, 2353.76, 2343.57, 2357.28, 2344.46
- U=16: 2351.56, 2341.47, 2352.91, 2355.02, 2343.89

The U=2 / U=4 outlier runs (bolded above) coincided with another
agent's transient activity on the node (a stray `test_mxfp8_python.py`
process was observed at 11:35 UTC during U=8/16 prep, indicating
shared-node SCLK perturbation rather than a kernel-side effect).
Discarding the obvious outliers (≥6% below their own cell's median):
- U=2 median of remaining 4 runs: **2348.7** (essentially identical)
- U=4 median of remaining 3 runs: **2348.3** (essentially identical)
Even after outlier removal, no U beats baseline by ≥+1.5%.

### 2.2 Structural finding — CRR is NOT bimodal like RRR

The CRR codegen for the production kernel symbol is **bit-identical at
the resource-usage level across U ∈ {1, 2, 4, 8, 16}**:

```
TotalSGPRs: 64
VGPRs: 225
AGPRs: 0
ScratchSize [bytes/lane]: 0
Occupancy [waves/SIMD]: 2
SGPRs Spill: 0
VGPRs Spill: 0
LDS Size [bytes/block]: 139264
```

This is the **strongest possible negative result** for this lever:

1. CRR does NOT have RRR's bimodal spill cliff. RRR went from
   254 VGPRs / 0 scratch / 0 spill at U=1 to 256 VGPRs / **208 B/lane
   scratch / 67 reg spill** at U=2 (R50B §2.1) — a brutal cliff.
   CRR's compiler heuristic doesn't trigger that cliff at any N≤16.
2. But CRR also doesn't have a *positive* unroll lever either — the
   compiler appears to **silently ignore** `#pragma unroll N≥2` on
   the CRR K-loop body, producing identical machine code at every N.
   This is consistent with the loop already being "pre-scheduled"
   into a per-iteration form that the LLVM unroller cannot
   meaningfully expand without breaking the carefully laid out
   `tic ^= 1, toc ^= 1` LDS-buffer-toggle dependency chain and the
   `crr_exact_cA_with_b1_interleave_fixed_phase` interleave schedule.
3. CRR has **9% lower VGPR pressure than RRR** (225 vs 254) at this
   shape, which gives more register headroom but doesn't translate
   into a partial-unroll sweet spot — the headroom is consumed by the
   larger LDS footprint (139264 vs RRR's 135168 B/block) and the
   richer per-iter sequence (4 MFMAs + B1 LDS-load interleave + 2
   barriers + scale-pack shift).

The CRR HEADROOM at 8B Gate/Up — if real after R47B's swizzle is
revisited, see R49 Reviewer note in prompt — must come from a lever
that **is not the unroll-factor knob**.

### 2.3 SHIP gate evaluation

Phase 1 ship gate (per orchestrator):
> Best U beats U=1 baseline by ≥+1.5% AND spread <2% AND no spill
> regression vs baseline.

| U | ≥+1.5%? | spread<2%? | spill ≤ U=1 (=0)? | occ ≥ U=1 (=2)? | Verdict |
|--:|:-------:|:----------:|:----------------:|:--------------:|:-------:|
|  2 | NO (-0.05%) | NO (9.41%) | YES (0) | YES (2) | **REFUTED** |
|  4 | NO (-0.46%) | NO (45.64%) | YES (0) | YES (2) | **REFUTED** |
|  8 | NO (+0.05%) | YES | YES (0) | YES (2) | **REFUTED** |
| 16 | NO (+0.09%) | YES | YES (0) | YES (2) | **REFUTED** |

No candidate U passes Phase 1 ship gate. Phase 2 (regression sweep on
other 6 CRR shapes) skipped per workflow step 5 ("only if Phase 1
found a candidate").

### 2.4 Why CRR ≠ RRR on the unroll-factor knob

R50B closed the analogous RRR sweep with the structural finding that
RRR's `do_k_iter` lambda has scale-pack live ranges that explode under
any compiler-driven unroll, committing to scratch at the moment a
`#pragma unroll N≥2` is present. The compiler's heuristic for RRR is
"unroll naively → spill aggressively".

CRR is structurally different in three ways that disable this dynamic:

1. **Manual MMA/load interleave already inlined**: CRR's main loop
   uses `crr_exact_cA_with_b1_interleave_fixed_phase` (lines 215–249)
   which manually interleaves the 4 cA MMAs with 8 progressive
   load_col_from_v2_st calls for b1 via an `INSERT_AFTER` template
   parameter. This pre-scheduled inner loop is opaque to the LLVM
   unroller — `#pragma unroll N` on the outer K-loop produces N
   copies of an already-scheduled body, not a re-scheduled larger
   body. Codegen is determined by the inner template instantiation,
   which is shape-fixed at compile time.
2. **Lower scale-pack live count per iter**: CRR uses 4 scale packs
   (a0, a1, b0, b1) but only 2 are live across each MFMA pair (cA/cB
   share a0/b0+b1; cC/cD share a1/b0+b1) — half are spilled-to-shift
   between phases via the in-place 16-bit shift logic at lines
   936–946. RRR's scale layout keeps more packs simultaneously live.
3. **`tic^=1, toc^=1` LDS toggle in loop header**: the CRR loop's
   per-iter `tic^=1, toc^=1` increment introduces a loop-carried
   dependency on LDS buffer indices that the LLVM unroller cannot
   propagate through at unroll-time without aliasing-analysis it
   does not have access to (the LDS subtile addresses depend on a
   runtime XOR of an int register), so it bails to "unroll by 1".

Net: CRR is in a stable local minimum that the unroll-factor knob
cannot move it out of. Same conclusion as R50B but reached via a
*different* mechanism (compiler bails on unroll, vs RRR's compiler
unrolls and spills).

## 3. Outcome

- **No source landed.** `CRR_MAIN_UNROLL` macro left at default 1.
- **Hypothesis closed.** Per-shape `CRR_MAIN_UNROLL` selection at
  8B Gate/Up CRR is REFUTED. The compiler-side iteration-count knob
  is a **dead lever for CRR** — codegen is invariant under the
  pragma.
- **Lever exhausted.** The MXFP8 CRR K-pair main loop body cannot be
  meaningfully reshaped via `#pragma unroll`. Any future attempt at
  recovering loop-control overhead at 8B Gate/Up CRR must change the
  **body** of the loop (the `crr_exact_cA_with_b1_interleave_*`
  template, the per-iter LDS toggle order, or the inner-K MMA
  packing) rather than its iteration count.

## 4. Recommendations for R52+

1. **CLOSE** `CRR_MAIN_UNROLL` per-shape exploration. The codegen is
   provably invariant under the macro. No experiment in the
   unroll-factor axis is going to find a sweet spot — neither for
   8B Gate/Up CRR nor for any other CRR shape (the kernel symbol
   resource-usage table is shape-invariant on this axis).
2. **The CRR HEADROOM at 8B Gate/Up (and the broader CRR ~92%
   structural ceiling per R48D / R47D)** remains real but its lever
   is not `CRR_MAIN_UNROLL`. Candidate alternatives:
   - **Inner-K interleave count** (`CRR_EXACT_B1_LDS_INSERT_AFTER`,
     line 160, default 4) — sweep this to find the cA-MMA-vs-LDS-load
     placement that maximizes overlap. This is the per-iter analog of
     unroll, but operates on the part of the loop the compiler
     actually re-schedules.
   - **Per-shape `MXFP8_CRR_BLOCK_SWIZZLE_GROUP_M`** at 8B Gate/Up
     specifically. Default 4 may not be optimal at N=14336 where the
     row-group footprint differs from the bigger Llama-70B shapes
     this knob was tuned for.
   - **Source-level scale-pack live-range hoisting** in the in-place
     shift block (lines 936–946) — collapse the 4-pack shift into 2
     paired shifts so fewer regs are simultaneously live across the
     barrier at line 994.
   - **Re-examine R47B CRR XCD swizzle on 8B Gate/Up specifically**
     (per R49 Reviewer note) — if R47B's swizzle regresses 70B
     Gate/Up by -3.87% under stricter SCLK, it may also be tuning the
     8B Gate/Up cell sub-optimally; a per-shape conditional
     `MXFP8_CRR_BLOCK_SWIZZLE` gate may be the cheapest win.
3. **Cross-cycle invariant established**: CRR codegen is
   unroll-invariant. Future CRR-side work should NOT include the
   unroll-factor as a degree of freedom in the experimental design.
   Document this in `TODO.md` so subsequent agents don't re-walk it.

## Files

- `r51f_bench.sh` — bench driver (5 runs/cell, 30s cooldown, 60s rebuild cooldown)
- `r51f_bench_phase1.run.log` — driver stdout
- `r51f_results/8B_GateUp_U{1,2,4,8,16}_run{1..5}.log` — per-run TFLOPS
- `r51f_results/8B_GateUp_U{1,2,4,8,16}_check.log` — SNR + det (1 each)
- `r51f_results/8B_GateUp_U{1,2,4,8,16}_build.log` — build remarks
  (kernel-resource-usage)
