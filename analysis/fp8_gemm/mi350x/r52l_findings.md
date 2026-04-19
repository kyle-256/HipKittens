# R52 Dev L — MXFP8 RRR K-tail peeling at 8B Gate/Up — REFUTED (pre-bench)

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 02ca9018 (R52J landed)
**GPU:** MI355X (gfx950) — pre-bench static audit only; no GPU benches run
**Lever:** `MXFP8_RRR_PEEL_TAIL=1` — manually peel the final K-pair out of the
main K-pair loop body in `rrr_mxfp8_exact_8wave_fastpath.inc`. Default OFF.
**Target cell:** 8B Gate/Up RRR (M=4096 N=14336 K=4096), the only remaining
RRR HEADROOM cell at 90.4% (+3.1pp vs predicted ceiling per r48d_findings.md).

## TL;DR — VERDICT: REFUTED at pre-bench audit

Source-level peeling **does materialize** in the ISA (the production V1
PRESHUFFLED kernel grows from 128→192 MFMAs = +64 = exactly +1 K-pair body
worth of straight-line code, confirming the peel was honored by the
compiler). However, peeling crosses the **bimodal compiler spill threshold**
identical to the one R49B (noinline) and R50B (`#pragma unroll N>=2`) hit —
production V1 RRR scratch goes **68 → 132 B/lane (+94%)** and VGPR spill goes
**16 → 41 (+156%)** at 8B Gate/Up.

Per the orchestrator's gate ("If pre-bench fails on correctness or VGPRs →
REFUTE without GPU bench. Loop transformation often perturbs scheduler
liveness analysis."), no strict-SCLK A/B was executed.

Default `MXFP8_RRR_PEEL_TAIL=0` left in tree; variant header retained for
archival (so the next agent can revisit if the compiler scratch heuristic
ever changes).

## 1. Implementation

### 1.1 Source patch

`rrr_mxfp8_exact_8wave_fastpath.inc`, two added blocks:

(a) Knob declaration (after `MXFP8_RRR_SOFT_BAR()` macro, ≈ line 124):

```cpp
#ifndef MXFP8_RRR_PEEL_TAIL
#define MXFP8_RRR_PEEL_TAIL 0
#endif
```

(b) Loop body (replacing the lines `for (int kp = 0; kp < k_pairs; ...)` near
line 705 with a `#if MXFP8_RRR_PEEL_TAIL`-guarded variant). The interior
loop runs `kp = 0 .. k_pairs - 2`; the final `kp = k_pairs - 1` is emitted
as straight-line code immediately after the loop, before the existing
pre-tail / epilogue blocks. `static_assert(k_pairs >= 1)` guards the lower
bound. `tic`/`toc` semantics preserved (each `do_k_iter` flips them; net
flip across one peeled K-pair = 0, identical to one loop iteration).

The peeled body is **not** a custom hand-coded sequence — it reuses the same
`do_k_iter` lambda as the loop body, so any per-iter wait sequence, scale
load, or scheduler hint stays exactly identical between interior and tail
iterations. This was a deliberate choice to keep the diff minimal and to
test the compiler-side hypothesis (does the scheduler treat the tail
better when not bound to the loop?). A custom hand-coded tail with
"final-iteration" wait elision was deemed premature without first proving
the scheduler can use the freedom.

### 1.2 Build sanity

The variant compiles cleanly at all 3 target shapes; correctness was not
GPU-checked because the structural spill blocks ship.

## 2. Pre-bench audit data (`-Rpass-analysis=kernel-resource-usage`)

All three target RRR shapes (production-path V1 PRESHUFFLED variant —
`_Z29rrr_exact_8wave_scaled_kernelILb1ELi1EEv...`, the symbol that
`dispatch_rrr_exact_8wave_scaled` launches by default).

| Shape (M×N×K) | Build | VGPRs | Scratch (B/lane) | VGPR Spill | Occupancy | Verdict |
|---|---|---:|---:|---:|---:|---|
| 8B Gate/Up 4096×14336×4096 | OFF | 256 | 68 | 16 | 2 | baseline |
| 8B Gate/Up 4096×14336×4096 | **ON** | 256 | **132 (+94%)** | **41 (+156%)** | 2 | **SPILL REGRESSION** |
| 8B Q/O 4096×4096×4096 | OFF | 256 | 68 | 16 | 2 | baseline |
| 8B Q/O 4096×4096×4096 | **ON** | 256 | **132 (+94%)** | **41 (+156%)** | 2 | **SPILL REGRESSION** |
| 70B Q/O 4096×8192×8192 | OFF | 256 | 68 | 16 | 2 | baseline |
| 70B Q/O 4096×8192×8192 | **ON** | 256 | **148 (+118%)** | **46 (+188%)** | 2 | **SPILL REGRESSION** |

(LDS unchanged at 135168 bytes/block in all 6 builds; VGPRs stay at 256
limit but go from 254→256 because the additional K-pair body's live ranges
push past the previous boundary.)

Cross-check on the off-path V1 PRESHUFFLED=false variant (not used in
production but compiled in the same TU): scratch explodes 204 → **908**
B/lane (+345%), spill 50 → **320** (+540%). This non-production variant
amplifies the same regression and confirms it is the compiler's bimodal
threshold, not shape-specific.

The `<true,2>` (V2 PRESHUFFLED) variant remains at 0 scratch / 0 spill in
both OFF and ON builds — V2 has lower baseline register pressure (per-wave
SRD design eliminates per-lane scale-row-base array) and stays under the
threshold. This confirms the hypothesis that the threshold is on a
register-pressure contour: the additional unrolled body's live ranges
(extra VGPRs for `b0_keep`/`b1_keep`/`a` across the peeled K-pair) push
V1 across the spill cliff but leave the leaner V2 below it.

## 3. ISA evidence (8B Gate/Up RRR, `<true,1>` production symbol)

Generated via `--cuda-device-only -S` at M=4096 N=14336 K=4096:

| | OFF | ON | Δ |
|---|---:|---:|---:|
| Total lines (kernel scope) | 1724 | 2228 | +504 (+29%) |
| `v_mfma_*` count | 128 | 192 | **+64 (= 1 K-pair body — peel HONORED)** |
| `ds_read` count | 128 | 192 | +64 (matches MFMA scaling) |
| `buffer_load` count | 32 | 48 | +16 (matches +1 K-pair tile fills) |
| `s_barrier` count | 34 | 50 | +16 (matches +1 K-pair barriers) |
| `s_cbranch` count | 5 | 5 | unchanged (loop-control elsewhere) |
| `scratch_load` count | 11 | **27** | **+16 (+145%)** |
| `scratch_store` count | 11 | **27** | **+16 (+145%)** |

The peel was structurally honored — the compiler did emit the additional
K-pair body in straight-line code (visible as +64 MFMAs, +64 ds_reads, +16
buffer_loads, +16 barriers — exactly what one extra `do_k_iter` produces).
Loop-control was not eliminated (same 5 s_cbranch in the surrounding
prologue/epilogue), which is expected: the main loop branch is unchanged
because we still iterate `k_pairs - 1` times, just one fewer than before.

Cost: **+16 scratch_load / +16 scratch_store** ops in the V1 production
kernel scope. These are spill traffic that the OFF baseline did not have.
Each scratch op costs ~80-100 cycles VMEM — at 14 K-iter executions of
the loop, the per-K-pair cost adds many cycles to a body that needs to
fit inside ~620 cycles to stay at ceiling. This decisively negates the
~3-cycle/K-pair loop-control saving the peel was meant to recover.

ISA dumps preserved at `r52l_isa_dumps/mxfp8_rrr_4096_14336_4096_{OFF,ON}.s`.
Build remarks preserved at `r52l_results/{8B_GateUp,8B_QO,70B_QO}_{OFF,ON}_build.log`.

## 4. Why this matters: structural-negative finding

R52L joins R49B (noinline phase split, -97% spill), R50B (`#pragma unroll`
N>=2, -69% spill), R51F (CRR per-shape U-sweep, refuted), R52H (RRR soft
asm-volatile barrier, bit-identical ISA), and R52J (CRR INSERT_AFTER sweep,
SCLK-noise) as the **fifth lever in the R49+ cycle that targets the K=4096
RRR loop-control overhead and fails on the same compiler-side cliff**.

The structural picture is now sharp: any source-level transformation that
duplicates or replicates a `do_k_iter` body (full unroll, partial unroll
N>=2, peel-tail, noinline boundary) crosses the V1 PRESHUFFLED kernel's
bimodal spill threshold — at 256 VGPR cap with rich scale-pack live ranges,
adding even one extra body's worth of live ranges forces the compiler to
either spill or pessimize the schedule. The compiler does not have a knob
for "spill less aggressively" at a per-shape contour.

The **only successful R49+ levers on RRR have been LDS/swizzle changes
(R46D XCD swizzle, R47A) that don't touch the do_k_iter body**. Future
R49+ effort on the +3.1pp 8B Gate/Up RRR HEADROOM should pivot away from
loop-body transformations entirely. Likely productive directions per
r48d §5.1, R47C, R51E:

- B-side LDS layout / swizzle for N=14336 specifically (R51E refuted bank
  conflicts, but a different LDS access pattern may exist).
- `MXFP8_RRR_V2_SCALE_CACHEPOLICY` per-shape sweep (R47C did RCR/RRR; CRR
  was tested in R47D — but per-cell cachepolicy retest under strict SCLK
  has not been done).
- Profile with `rocprof --pmc` instead of static analysis: the +3.1pp may
  not be loop-control at all; r48d's model attributes it to ~3 loop-control
  instr/K-pair, but the actual cycle gap could be in LDS contention,
  scale-pack VMEM stalls, or the scratch already there in V1 baseline
  (`scratch=68` is non-zero!).

That last point deserves emphasis: the OFF baseline V1 PRESHUFFLED kernel
**already has 16 VGPR spill / 68 B/lane scratch** at all 3 shapes. The
8B Gate/Up RRR's +3.1pp may be tied to that pre-existing spill, not to
loop-control. A productive R53+ direction: investigate whether reducing
the scale-pack live-range footprint (e.g. by computing a0/a1 packs lazily
or interleaving them with the MMA quartet) could push V1 below the
bimodal threshold and eliminate the baseline spill, gaining headroom that
no source-level loop transformation can reach.

## 5. Falsifiable predictions

**P5L (R52L peel-tail spill):** Any future RRR experiment that adds a
straight-line copy of `do_k_iter` (peel-head, peel-multiple, fission, etc.)
will hit the same bimodal spill threshold on the V1 PRESHUFFLED kernel
at all M=4096 RRR shapes. If a future run shows scratch <100 B/lane after
adding such a copy, the compiler heuristic has changed (LLVM/HIPCC version
update suspected) and R49+ levers should be re-tested.

**P6L (V1 baseline spill blocker):** The V1 PRESHUFFLED kernel's existing
16 VGPR spill / 68 B/lane scratch at all M=4096 RRR shapes is a structural
blocker on the +3.1pp 8B Gate/Up RRR HEADROOM. If a future R53+ change
reduces this baseline spill to 0 without other regressions, expect ≥+0.5pp
on at least one of {8B Q/O RRR, 8B Gate/Up RRR}. If it lifts none, the
spill is on a non-critical path and the 3.1pp gap is elsewhere (LDS,
scale-pack VMEM, prologue/epilogue cost).

## 6. Files

- `r52l_findings.md` — this document.
- `r52l_bench.sh` — strict-SCLK A/B harness (NOT EXECUTED; archived for
  reproducibility once the compiler-side blocker is resolved).
- `r52l_results/{8B_GateUp,8B_QO,70B_QO}_{OFF,ON}_build.log` — full
  build remarks (resource usage) for all 6 builds.
- `r52l_isa_dumps/mxfp8_rrr_4096_14336_4096_{OFF,ON}.s` — full device .s
  dumps for the 8B Gate/Up shape (1 MB each, source for §3 ISA evidence).
- `rrr_mxfp8_exact_8wave_fastpath.inc` — variant header `MXFP8_RRR_PEEL_TAIL`
  added (default OFF). Patch retained for archival; no behavioral change
  to the production build.

## 7. One-line summary

**MXFP8_RRR_PEEL_TAIL=1 source-level peel of one K-pair from the main loop
materializes correctly in ISA (+64 MFMAs as expected) but crosses the V1
PRESHUFFLED kernel's bimodal compiler spill threshold (scratch +94%/+118%,
VGPR spill +156%/+188% across 3 RRR shapes). REFUTED at pre-bench audit
gate; joins R49B/R50B as the fifth do_k_iter-touching lever felled by the
same threshold. 8B Gate/Up RRR HEADROOM (+3.1pp) requires non-body-touching
levers (LDS/swizzle/cachepolicy) or a structural spill-reduction R&D arc.**
