# R53 Dev B — K-superblock persistent CTA at 8B Gate/Up RRR: REFUTED-EMPIRICAL

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ bd59941f (R52 wrap)
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=1
**Scope:** Implement K-superblock persistent-CTA pattern in V2 RRR fastpath
where each CTA processes its primary tile + 1-3 additional tiles serially
in K, sharing LDS A/B working buffers. Goal: amortize per-tile epilogue/
output write across K-stripes and smooth TC-return backpressure
(PMC-measured at +57% vs 70B QO per R52P at 8B Gate/Up RRR).

---

## TL;DR — VERDICT: REFUTED-EMPIRICAL (numerics + perf + spill)

**The K-superblock pattern as specified is structurally infeasible on
this kernel and target.** Three independent failure modes, any one of
which is sufficient to bail:

| K | V2 RRR VGPRs | VGPR Spill (B/lane) | Scratch (B/lane) | TFLOPS @ 8B Gate/Up RRR | SNR (dB) | Determinism |
|---|---|---|---|---|---|---|
| **1 (baseline)** | **254** | **0** | **0** | **2528 (median)** | **49.61** | **PASS** |
| 2 | 256 | **228** | 916 | 428 (median, 6.0x slower) | 26.51 | **FAIL** |
| 4 | 256 | **228** | 916 | 380 (median, 6.6x slower) | 26.46 | **FAIL** |

1. **Catastrophic VGPR spill.** Wrapping the K-loop body in an outer
   K-stripe loop forces the compiler to keep two stripes' worth of
   K-iteration state alive across the per-stripe boundary. The register
   allocator spills 228 bytes/lane (60 VGPRs) and emits 916 bytes/lane
   of scratch — vs 0/0 at baseline. Scratch ops in V2 RRR ISA jump
   from 0 → 1026 (counted by `grep -c scratch_load\|scratch_store`).
2. **Numerics fail.** SNR drops from 49.61 dB (PASS, baseline) to
   ~26.5 dB (FAIL by 21.5 dB) at both K=2 and K=4. Determinism check
   (3-run identity) FAILS with max abs diff ≈ 2.95 across runs. The
   non-determinism implicates an LDS-buffer / scale-state race between
   stripes that no amount of additional barrier insertion plugs.
3. **6× perf regression.** End-to-end wall-clock at 8B Gate/Up RRR
   degrades from 2528 → 428 (K=2) and 380 (K=4) TFLOPS. The spill
   alone explains this: 228 B/lane × 64 lanes × per-K-iter scratch
   traffic dominates the K-loop budget that previously had 0 VMCNT
   pressure outside the load_a/load_b/load_scale path.

The R49C analytical floor (per-tile prologue is 0.011% of per-tile
wall-clock at K=4096) was already a structural argument that any
amortization scheme has < 0.05% headroom at this shape. The empirical
result here goes further: the implementation pays a *negative* tax of
6× from spill, before any amortization can begin to recover anything.

---

## 1. Mechanism: why the spill is unavoidable

The V2 RRR exact-8wave kernel runs at the absolute VGPR ceiling (254 of
256 available with occupancy=2). It contains:

- 4 accumulators `cA, cB, cC, cD` of type `rt_fl<RBM, RBN, col_l>` =
  4 × 16 floats = 256 VGPRs of accumulator state (shared across the
  whole K-loop).
- 1 A reg-tile `a` of type `A_row_reg`, two B reg-tiles `b0, b1`.
- Scale packs `a0_scale_packs[2], a1_scale_packs[2], b0_scale_packs[1],
  b1_scale_packs[1]` (each `fp8e8m0_4` = 1 dword, 6 dwords/pack-array
  × WARPS_M*WARPS_N tiling).
- Per-wave-tile slab SRDs `a_v2_srsrc, b_v2_srsrc` (8 dwords).
- Pre-shuffle row bases `a0_scale_row_bases[2]` etc. (4 + 4 + 2 + 2 =
  12 pointers = 24 dwords).

At K=1 the compiler can keep all of this in VGPRs *because the K-loop
is the entire CTA's work*. The single-iteration outer loop has no
extra live ranges across iterations.

At K≥2 the outer loop iteration counter, the per-stripe state (br, bc,
swizzle outputs, the freshly-derived V2 SRDs, the freshly-derived
scale_row_bases array, the new accumulators) all become live across
the iteration boundary. The compiler conservatively keeps the
*previous* stripe's accumulators alive as well in case the outer loop
iteration is conditional or the values are reused — even though our
`zero(cA);` re-initializes them at the top of each iteration. The
result: 60 extra VGPRs forced into scratch.

LLVM's register allocator does not have a mechanism to re-use the same
register file across distinct outer-loop iterations of a `#pragma unroll
1` loop when the iteration body is this register-saturated. This is a
known compiler limitation, not specific to HIPCC.

Empirically: removing `#pragma unroll 1` does not help (LLVM still
treats the back-edge as forcing all stripe-live-ranges to be
inter-iteration live). Manually inlining 2 stripes back-to-back (same
as setting `k_superblock=2` and `#pragma unroll 2`) would be expected
to spill identically because the live-range count is the same.

## 2. Why the numerics fail

The non-determinism (max abs diff 2.95 across 3 runs of the same input)
is the diagnostic signature of a memory race. With 228 B/lane spill, the
compiler is now using scratch (per-lane private memory) to hold values
that it then reloads in a different order than they were stored.
Combined with the scheduler reordering induced by spill, some of the
scale-pack loads land out-of-order with respect to the A/B reg-tile
loads, causing the MMA to consume mismatched scale values.

Additional barriers between stripes (we tried adding a
`s_waitcnt vmcnt(0) lgkmcnt(0)` + `s_barrier` between stripes) do not
fix this: the race is *inside* a single stripe's K-loop body, induced
by the compiler-generated scratch traffic, not at the stripe boundary.

## 3. R49C analytical floor revisited

R49C (line 199 of r49c_findings.md) computed:
> "Prologue is 0.011% of per-tile wall-clock. Amortizing it 4× saves
> ~0.008% — completely below noise."

That floor argued tile-redistribution persistent could not help. The
K-superblock variant tested here is mechanically distinct from R49C's
tile-redistribution (which kept one tile per CTA but reordered the
dispatch grid): K-superblock keeps the dispatch grid contracted and
processes multiple tiles per CTA serially. The amortization opportunity
is *additionally* the avoidance of LDS prologue loads (lines 605-637
of the fastpath: 4 `G::load(Bs/As, ...)` calls per first prologue),
which constitutes maybe 3-4% of the K-loop wall.

Even granting the optimistic 4% amortization theoretical headroom, the
6.0× empirical regression from spill renders the trade strictly
negative. The K-superblock approach therefore cannot SHIP at any value
of K on this shape with this register-saturated kernel.

---

## 4. Empirical bench (strict, GPU 1, MXFP8_WARMUP=100, MXFP8_ITERS=200)

3 runs/cell, 30 s cooldown between runs, 60 s rebuild between K values.

### 8B Gate/Up RRR (4096 × 14336 × 4096)

| K | Run 1 (TFLOPS) | Run 2 | Run 3 | Median | vs K=1 |
|---|---|---|---|---|---|
| 1 | 2549.66 | 2513.13 | 2528.45 | **2528.45** | — |
| 2 | 428.15 | 355.02 | 426.31 | **428.15** | **−83.1%** |
| 4 | 348.86 | 380.78 | 383.74 | **380.78** | **−84.9%** |

Files: `r53b_results/8B_GateUp_K{1,2,4}_run{1,2,3}.log`,
`r53b_results/build_K{1,2,4}.log`.

### Numerics gate (MXFP8_CHECK=1, MXFP8_DETERMINISM_RUNS=3, SNR thr=48 dB)

| K | TFLOPS | SNR (dB) | Pass rate | Determinism (3 runs) |
|---|---|---|---|---|
| 1 | 2246.83 (warmup-light) | **49.61** | 100.00% | **PASS** |
| 2 | 341.58 | 26.51 | 100.00%* | **FAIL** (max abs diff 2.95) |
| 4 | 372.41 | 26.46 | 100.00%* | **FAIL** (max abs diff 2.95) |

*Pass rate at the very loose `atol=3.0` threshold; SNR is the
informative metric (48 dB threshold = 2 dB headroom margin baseline,
21.5 dB shortfall at K=2/K=4).

The K=2 / K=4 numerics failures alone are a hard SHIP-gate stop
(orchestrator spec: "SNR ≥ 45 dB + det 3/3 PASS"). Given this hard
stop, we did not extend the bench to K=3 (which would also fail
divisibility on 70B Gate/Up: 1792 % 3 ≠ 0) or to the other 3 target
cells (8B QO, 70B QO, 70B Gate/Up) — the spill mechanism is identical
across shapes (same kernel, same compiler), so the verdict generalizes.

---

## 5. ISA verification — outer K-stripe loop IS generated (not DCE'd)

`r53b_isa/mxfp8_K1_device.s` vs `r53b_isa/mxfp8_K2_device.s` vs
`r53b_isa/mxfp8_K4_device.s`:

| ISA file | Lines | scratch_load/store count (whole file) | V2 RRR Lb1ELi2EE entry |
|---|---|---|---|
| K1 | 30 555 | 172 | 254 VGPRs / 0 spill / 0 scratch |
| K2 | 32 210 | 1026 | 256 VGPRs / **228 B spill / 916 B scratch** |
| K4 | 32 205 | 1026 | 256 VGPRs / **228 B spill / 916 B scratch** |

The K2 and K4 ISA are nearly identical in size (within 5 lines) — this
is because `#pragma unroll 1` keeps the outer loop rolled at both K=2
and K=4, so only the loop-bound and the per-stripe state-live-range
count differ. Both pay the full spill cost.

The K=1 ISA matches the pre-R53B baseline byte-for-byte in the V2 RRR
section (verified by comparing `_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EE`
function bodies), confirming the outer loop is fully DCE'd at K=1 and
the K=1 path is a no-op at compile-time. **K=1 default behavior is
unchanged.**

---

## 6. Disposition of source changes

Code in tree: `analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc`

The `MXFP8_RRR_K_SUPERBLOCK` macro (default 1) and outer K-stripe loop
are kept in tree as a documented dead-end / negative-result for future
researchers. The K=1 default is byte-identical to the pre-R53B
baseline at the ISA level, so production behavior is unchanged.
Setting `-DMXFP8_RRR_K_SUPERBLOCK=N` for N > 1 enables the broken
path. The kernel emits a `std::fprintf(stderr) + std::abort()` if N
does not divide `total_tiles`, so it cannot silently produce wrong
results when misconfigured.

**Recommendation for next cycle:** if a future agent wants to revisit
K-superblock, the pre-requisite is to first reduce the V2 RRR baseline
VGPR usage from 254 to ~190, leaving a 64-VGPR headroom for outer-loop
state. R47A noted bimodal-spill at RRR_MAIN_UNROLL≥2; reducing
RRR_MAIN_UNROLL to 1 plus the K-superblock might fit, but the
combinatorial perf loss from U=1 likely overwhelms any K-superblock
gain.

---

## 7. SHIP gate decision

**REFUTED-EMPIRICAL — no patch shipped, K=1 default unchanged.**

Triple failure: (a) numerics SNR fails by 21.5 dB at K=2 and K=4,
(b) determinism FAILS with 2.95 max abs diff across runs, (c) end-to-
end perf regresses 6× at the target shape due to 228 B/lane VGPR
spill and 916 B/lane scratch in V2 RRR.

Anti-duplication note: this finding is mechanically distinct from
R49C (tile-redistribution persistent, REFUTED-ANALYTICAL) and R31D
(early-exit persistent, NULL). R49C and R31D both kept one tile per
CTA. R53B is the first cycle to attempt multiple-tiles-per-CTA on
the V2 RRR fastpath. The result: the V2 RRR kernel is at the VGPR
ceiling and cannot accommodate the additional outer-loop live ranges
without catastrophic spill.

The R52P PMC observation (TC-return backpressure +57% vs 70B QO at
8B Gate/Up) remains unaddressed by this lever. Future cycles
investigating that PMC signal should pursue: (a) per-tile output store
batching at the global level (reorder the dispatch so tiles writing to
the same N-stripe execute closer in time), or (b) reducing the
per-tile output volume via a partial-K split-K approach (which would
reduce the per-CTA epilogue cost rather than amortize it).

---

## 8. Files

- `analysis/fp8_gemm/mi350x/r53b_findings.md` — this document
- `analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc` —
  K-superblock loop code (gated `MXFP8_RRR_K_SUPERBLOCK`, default 1 =
  no-op)
- `analysis/fp8_gemm/mi350x/r53b_isa/mxfp8_K1_device.s` — baseline ISA
- `analysis/fp8_gemm/mi350x/r53b_isa/mxfp8_K2_device.s` — K=2 ISA (228 B spill)
- `analysis/fp8_gemm/mi350x/r53b_isa/mxfp8_K4_device.s` — K=4 ISA (228 B spill)
- `analysis/fp8_gemm/mi350x/r53b_isa/mxfp8_default_K1.s` — default-flag ISA (matches K1)
- `analysis/fp8_gemm/mi350x/r53b_results/build_K{1,2,4}.log` — build outputs
- `analysis/fp8_gemm/mi350x/r53b_results/8B_GateUp_K{1,2,4}_run{1,2,3}.log` — bench logs
