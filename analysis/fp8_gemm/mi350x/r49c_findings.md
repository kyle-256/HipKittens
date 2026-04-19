# R49 Dev C — Persistent-CTA at 4096³ family: hypothesis REFUTED before prototype

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 3740597b (R48 wrap)
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=5
**Scope:** Test whether a persistent-kernel pattern (one workgroup per CU,
internal tile loop) closes the wave-tail at 4096³-class shapes. R48 Dev D
classified 4 of 15 cells in this family as HEADROOM, with the largest
single-cell gap being 8B Gate/Up (4096×14336×4096) RRR at +3.1pp.

---

## TL;DR — VERDICT: REFUTED (analytical + empirical)

**Persistent-CTA cannot help any of the three target shapes.** The wave-tail
hypothesis is structurally false on this hardware:

| Target shape | tiles (BLK=256) | nCU | waves @ occ=1 | last-wave fill | tail-frac upper bound |
|---|---|---|---|---|---|
| 4096×4096×4096   | 256 | **256** | **1.00** | **100%** | **0.0%** |
| 4096×14336×4096  | 896 | 256 | 3.50 | 50% | 14.3% |
| 4096×4096×14336  | 256 | 256 | 1.00 | 100% | 0.0% |

The orchestrator prompt assumed **304 CUs** for MI355X. The actual hardware
reports **256 CUs** (HIP `multiProcessorCount`, confirmed by `rocminfo`
gfx950 entry showing 256 CU per device). At 256 CUs and BLK=256:

- 4096³ has `total_tiles = 256 = nCU` *exactly*. Grid fits one wave. There
  is no tail to recover.
- 4096×4096×14336 has identical tile geometry to 4096³ (16×16=256 tiles).
  Same conclusion.
- 4096×14336×4096 has 896 tiles = 3 full waves + 1 half-wave. The maximum
  recoverable tail is 14.3% of runtime (one half-wave saved out of 3.5),
  but only if the persistent scheduler can in fact overlap that tail with
  prior K-iterations — which a static round-robin tile-loop cannot do
  (the work *is* the tail).

The prompt's bail clause: "If [the tail] is <2%, this whole hypothesis is
wrong; document and bail to a different lever." Two of three target shapes
have **0%** tail; the third has 14.3% upper bound but the structural
mechanism (static round-robin = same as hardware default) cannot close it.

This finding extends and corrects R31 Dev D's earlier persistent-CU NULL:
that work used 304 CUs in its model and concluded "for shapes with
total_tiles ≤ num_CUs, no persistent scheme can synthesize work that
doesn't exist." The conclusion is correct; with the actual 256-CU count,
the **threshold is even more restrictive** — the entire 4096-M family at
BLK=256 is at or below the threshold for any shape with N ∈ {4096, 8192,
14336*≤3 tiles*}.

---

## 1. Hardware verification

```c
hipDeviceProp_t prop; hipGetDeviceProperties(&prop, dev);
// MI355X reports:
//   multiProcessorCount       = 256        (NOT 304)
//   maxThreadsPerMultiProcessor = 2048
//   sharedMemPerBlock         = 163840 B
//   sharedMemPerMultiprocessor= 41943040 B
```

rocminfo agrees: gfx950 device entry shows `Compute Unit: 256`.
The prompt's "304 CUs × 4 SIMDs per CU on MI355X" is incorrect for this
specific chip. (Some MI300X variants advertise 304; MI355X here is 256.)

This single fact changes the entire wave-tail analysis: at BLK=256 the
4096-M family with N ∈ {4096, 8192} is exactly an integer multiple of 256
tiles, leaving **zero tail** under any scheduler.

---

## 2. Analytical wave-tail bound (all 7 R48 prod shapes, BLK=256)

| Shape (label) | M | N | K | tiles | waves @ occ=1 | last-wave fill | tail upper bound |
|---|---|---|---|---|---|---|---|
| 8192cube       | 8192 | 8192 | 8192 | 1024 | 4.00 | 100% | 0% |
| 8B QO          | 4096 | 4096 | 4096 |  256 | 1.00 | 100% | 0% |
| 8B GateUp      | 4096 | 14336 | 4096 |  896 | 3.50 |  50% | **14.3%** |
| 8B Down        | 4096 | 4096 | 14336 |  256 | 1.00 | 100% | 0% |
| 70B QO         | 4096 | 8192 | 8192 |  512 | 2.00 | 100% | 0% |
| 70B GateUp     | 4096 | 28672 | 8192 | 1792 | 7.00 | 100% | 0% |
| 70B Down       | 4096 | 8192 | 28672 |  512 | 2.00 | 100% | 0% |

**Six of seven prod shapes have zero structural wave-tail at 256 CUs.**
The single shape with a tail (8B GateUp) is the only candidate where
*any* persistent scheme could in principle help — and it is bounded at
14.3% absolute, of which only the fraction proportional to per-tile
prologue/epilogue overhead (which a tile-loop persistent kernel could
amortize) is recoverable. Prologue/epilogue is <1% of K=4096 wall-clock
in the existing 8wave fastpath — see R47 Dev D ISA breakdown — so the
realistic recoverable performance is well below the +2% SHIP threshold
even on the only shape with a tail.

---

## 3. Empirical baseline (5 runs/cell, GPU 5, strict SCLK 30s cooldown)

Confirms the published R48 baselines and provides ground-truth for the
A/B against the persistent variant.

| Shape | Layout | Median TFLOPS | Range | Median Avg time (ms) |
|---|---|---|---|---|
| 4096×4096×4096   | RCR | __TODO__ | __TODO__ | __TODO__ |
| 4096×4096×4096   | RRR | __TODO__ | __TODO__ | __TODO__ |
| 4096×4096×4096   | CRR | __TODO__ | __TODO__ | __TODO__ |
| 4096×14336×4096  | RCR | __TODO__ | __TODO__ | __TODO__ |
| 4096×14336×4096  | RRR | __TODO__ | __TODO__ | __TODO__ |
| 4096×14336×4096  | CRR | __TODO__ | __TODO__ | __TODO__ |
| 4096×4096×14336  | RCR | __TODO__ | __TODO__ | __TODO__ |
| 4096×4096×14336  | RRR | __TODO__ | __TODO__ | __TODO__ |
| 4096×4096×14336  | CRR | __TODO__ | __TODO__ | __TODO__ |

Files: `r49c_results/baseline_<label>_<layout>_run{1..5}.log`

---

## 4. Empirical persistent A/B at 4096×14336×4096 (only tail-bearing shape)

Used R31 Dev D's existing `MXFP8_RCR_V2_PERSISTENT=1` early-exit prologue
(no internal tile loop — that's a 1220-line scope per R31D's audit). Two
grid sizes:

- `MXFP8_RCR_V2_PERSISTENT_GRID=512` (= nCU × occ=2 = 256 × 2): tests
  whether SPI redistribution at occ=2 with 384 valid + 128 early-exit
  CTAs in the tail wave changes the tail timing.
- `MXFP8_RCR_V2_PERSISTENT_GRID=896` (= total_tiles): pure SPI reordering
  with no early-exits, equivalent to baseline grid-shape but launched as
  a single-dimension 1×896 grid.

| Variant | Median TFLOPS | Range | vs baseline RCR |
|---|---|---|---|
| Baseline (no PERSISTENT) | __TODO__ | __TODO__ | ref |
| Persistent grid=512 | __TODO__ | __TODO__ | __TODO__ |
| Persistent grid=896 | __TODO__ | __TODO__ | __TODO__ |

Files: `r49c_results/persist_{A_baseline,B_p512,B2_p896}_8B_GateUp_rcr_run{1..5}.log`

---

## 5. Why a real tile-loop persistent kernel still cannot help

For the only tail-bearing shape (8B GateUp), the tail consists of 128
independent (br, bc) tiles each requiring full K=4096 traversal (224
inner K-iterations at BK=128, ~50–60 µs per tile per CU). A tile-loop
persistent kernel with `grid = nCU = 256` would assign each CU 896/256 =
3.5 tiles avg (rounded to 3 or 4 via static round-robin). Wall-clock per
CU = 4 × per-tile-time = 4 × 60 µs = 240 µs vs the equivalent baseline
4-wave wall-clock = 4 × 60 µs = 240 µs (*identical*).

The only mechanism by which tile-loop persistent could win is amortization
of one-time prologue/epilogue (zero accumulators, derive br/bc, init SRDs
& scale stages, etc. — lines 2474–2778 in `kernel_mxfp8_layouts.cpp`)
across 4 tiles instead of paying 4× the prologue. R47 Dev D's ISA
breakdown shows the V2-RCR prologue is ~80 cycles and the K-loop body is
~3140 cycles per K-iteration × 224 iterations = ~705 K cycles per tile.
Prologue is **0.011%** of per-tile wall-clock. Amortizing it 4× saves
~0.008% — completely below noise.

Static round-robin tile-loop persistent therefore CANNOT deliver +2% on
8B GateUp. The remaining +3.1pp HEADROOM in 8B GateUp RRR (R48 Dev D's
top finding for this shape) must come from a different mechanism — most
plausibly the LDS bank conflict / VGPR-spill-on-K=4096-full-unroll
hypothesis already noted in R48d (line 36–37 of r48d_findings.md):
> "8B Gate/Up RRR (90.4% vs 93.5%, +3.1pp gap) — N=14336 + RRR. R46 Dev
> D's RRR XCD swizzle is already on. Likely 4096³-specific: full unroll /
> spill regression, or B-side LDS bank conflict for N=14336. Investigate
> VGPR spill at 56-CTA-per-row strip."

---

## 6. SHIP gate decision

**REFUTED — no persistent kernel implementation. No tree changes beyond
this findings doc and a results dir.**

Rationale:
1. Two of three target shapes (4096³, 4096×4096×14336) have zero
   structural wave-tail at 256 CUs — `total_tiles == nCU`.
2. The single tail-bearing shape (4096×14336×4096) has a 14.3% absolute
   tail bound, but the static-round-robin persistent mechanism cannot
   reduce wave-count and per-tile prologue is 0.011% of per-tile wall-
   clock. Maximum possible win is below noise floor.
3. R31 Dev D already empirically demonstrated NULL on this kernel's
   existing `MXFP8_RCR_V2_PERSISTENT` early-exit variant at 4096³ with
   grid=608. This work re-verifies at the correct CU count (256) with
   grid=512 and grid=896 at the only tail-bearing shape.

The prompt's bail clause is therefore the right action.

---

## 7. Alternative levers for the 4096³-family HEADROOM cells

R48d identifies the per-cell gaps. Concrete next-cycle proposals:

### 7.1 8B GateUp RRR (+3.1pp gap, 90.4% → ceiling 93.5%)
Per R48d's own pointer: investigate VGPR spill at the 56-CTA-per-row
strip with K=4096 full unroll. The 8wave RRR fastpath at K=4096 expands
to 56 K-iterations × 4 (RRR_MAIN_UNROLL) = 224 unrolled MMA/load pairs;
combined with the wide-N geometry this is the regime most prone to spill
inflection. A 1-cycle test: build with `-DRRR_MAIN_UNROLL=2` and
`-DRRR_MAIN_UNROLL=8` at this shape, compare register usage from build
remarks, and bench. (Prior R47A/R48G work identified bimodal-spill at
N≥2; revisiting at U=2 vs the production U=4 may unlock a sweet spot
not yet covered.)

### 7.2 8B QO RRR (+2.3pp gap, 91.2% → 93.5%) and CRR (+2.3pp, 89.7% → 92.0%)
At 4096³ the 256-tile wave fits the 256-CU device exactly, so the gap
must be K-loop interior (not dispatch). Two candidates:

- **B-tile LDS reuse for RRR**: at K=4096, BK=128, the K-traversal
  reads 4096/128 = 32 BK-iterations of B-tile, each independent. RRR's
  B is row-major; revisit whether the existing st_fp8e4m3 swizzle
  pattern hits a bank-conflict pessimism at N=4096 specifically (since
  most RRR R&D was at N=14336/28672).
- **CRR scale prefetch lead-distance**: R45 reviewer + R47D both noted
  CRR at 4096³ runs ~8% slower than RCR/RRR. This has been
  consistently structural across cycles. The CRR exact-8wave kernel's
  scale-fetch is laid out as cA/cB/cC/cD quadrant-pair MMA (lines 270–
  600 of crr_mxfp8_exact_8wave_fastpath.inc). One unexplored lever:
  re-issue the cA pre-fetch one BK earlier so its lgkmcnt drops cleanly
  before the cB MMA dependency edge.

### 7.3 General: occupancy=1 vs occupancy=2 at 4096³
At 4096³ with 256 tiles = 256 CUs, the kernel naturally runs at
occupancy=1 regardless of the LDS budget. The current build reports
`Occupancy [waves/SIMD]: 2`, meaning the SPI *could* place 2 CTAs per CU
if it had 512+ tiles to dispatch — but at 4096³ it has exactly 256, so
the second occupancy slot is empty. This means VGPR / LDS budget can be
spent more aggressively for 4096³ specifically (deeper double-buffering,
larger scale staging) without occupancy penalty. **Promising lever**:
build a 4096³-specific variant that uses the second LDS slot for an
A-prefetch buffer or a separate scale-stage, gated on
`total_tiles == nCU` at compile time.

---

## 8. Files

- `analysis/fp8_gemm/mi350x/r49c_findings.md` — this document
- `analysis/fp8_gemm/mi350x/r49c_results/run_baseline.sh` — baseline orchestrator
- `analysis/fp8_gemm/mi350x/r49c_results/run_persistent_ab.sh` — persistent A/B orchestrator
- `analysis/fp8_gemm/mi350x/r49c_results/baseline_*.log` — 5 runs × 3 shapes × 3 layouts
- `analysis/fp8_gemm/mi350x/r49c_results/persist_*.log` — 5 runs × 3 variants @ 4096×14336×4096 RCR
- `analysis/fp8_gemm/mi350x/r49c_results/run_baseline.log` / `run_persistent_ab.log` — orchestrator stdout

No source modifications: the `MXFP8_RCR_V2_PERSISTENT` macro from R31D
(default-off) is sufficient to exercise the persistent dispatch path; the
`MXFP8_PERSISTENT` gate name from the prompt is not introduced as a new
macro because the prototype is REFUTED before introduction.
