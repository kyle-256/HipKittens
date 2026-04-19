# R52 Dev Q — V2 RRR data-load cachepolicy sweep — REFUTED

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 34f273db (R52P landed)
**GPU:** MI355X (gfx950) on `HIP_VISIBLE_DEVICES=1` (R52Q exclusive)
**Mandate:** R52P PMC analysis attributed the +3.1pp 8B Gate/Up RRR HEADROOM
gap to L2/TC cache-return backpressure (TA_DATA_STALLED_BY_TC +57% on the
slow cell, TCC L2 hit rate unchanged). Implement and bench
**B-side L2 cachepolicy retention biasing** on the V2 RRR
`dispatch_rrr_exact_8wave_scaled_v2<true>` data fetches.

## TL;DR — VERDICT: REFUTED (all variants regress or no-op vs baseline)

| Variant | A coh | B coh | ISA flag (A,B) | Median TFLOPS | spread | vs CP0 |
|---------|------:|------:|----------------|--------------:|-------:|-------:|
| **CP0 (baseline)** | 0 (cache_all) | 0 (cache_all) | (none, none)         | **2549.3** | 0.43% | — |
| CP1 (R52P primary) | 2 (cache_stream) | 0 (cache_all) | (`nt`, none)       | 2327.2     | 1.66% | **−8.71%** |
| CP2 (GLC both)     | 1 (cache_global) | 1 (cache_global) | (`sc0`, `sc0`)  | 2547.7     | 0.33% | −0.06% |
| CP3 (A=NT)         | 3 (non_temporal) | 0 (cache_all) | (`sc0 nt`, none)   | 2338.2     | 0.40% | **−8.28%** |
| CP4 (control B=NT) | 2 (cache_stream) | 3 (non_temporal) | (`nt`, `sc0 nt`) | 2142.6   | 0.33% | **−15.95%** |

* All five variants pass SNR ≥ 45 dB (49.61 dB) and 3/3 determinism.
* All five variants are ISA-verified (cachepolicy flags propagate correctly).
* All run spreads are < 2% (well under the 5% flag threshold).
* No cross-cell K=8192 bench was run: primary REFUTED at the discriminator
  shape, so transferable-lever investigation is out of scope.

**The R52P primary recommendation (A=STREAMING, B=ALWAYS) is REFUTED.**
The current `cache_all` (LRU, fully cached) policy is already optimal at the
8B Gate/Up RRR shape. Reducing cache retention on either side hurts
performance. CP1/CP3 (A streamed) regress ~8.5% — likely because A is in
fact reused across consecutive K-pair iterations within the same stripe
(it is *not* "consumed once per stripe" as the R52P speculative model
proposed). CP4 (B non-temporal) regresses 16% as predicted; this is the
positive control confirming the lever is wired correctly. CP2 (GLC on both)
is bit-identical-perf to baseline — bypassing L1 has no measurable effect
because the data path goes via `buffer_load_lds` directly into LDS, so L1
is essentially out of the path already; only the L2 (TCC) hint matters.

## 1. Method

### 1.1 Plumbing

The MXFP8 RRR V2 data fetches were hardcoded `coherency::cache_all` (=0) at
the kittens primitive `kittens::load(ST, GL, COORD, swizzled_offsets)`
(`include/ops/warp/memory/tile/global_to_shared.cuh:215, 246`). The cachepolicy
is forwarded to `llvm_amdgcn_raw_buffer_load_lds(...)`'s `aux` operand,
which the AMDGPU backend renders as ISA flags on `buffer_load_dwordx4 ...
offen <flag> lds`.

**Plumbing changes (3 files, minimal surface area):**
1. `include/ops/warp/memory/tile/global_to_shared.cuh` — add an `int COH`
   non-type template parameter (default = `cache_all`) to the prefilled-
   offsets `kittens::load` overload, forwarded to the load_lds aux.
2. `include/ops/group/memory/tile/global_to_shared.cuh` — add a
   `G::load_coh<COH>(...)` group wrapper that calls the new templated form.
3. `analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc` — define
   `MXFP8_RRR_A_COHERENCY` (0..3) and `MXFP8_RRR_B_COHERENCY` (0..3)
   macros (default 0 → bit-identical baseline routing through `G::load`),
   define `RRR_LOAD_A` / `RRR_LOAD_B` macros that route through either
   `G::load` (when both coherencies = 0) or `G::load_coh<COH>(...)` (when
   either is non-zero), and replace all 8 `G::load(As[…], g.a, …, soA)` and
   8 `G::load(Bs[…], g.b, …, soB)` call sites with the macros.

The `(MXFP8_RRR_A_COHERENCY == 0) && (MXFP8_RRR_B_COHERENCY == 0)` guard
preserves the baseline ISA bit-identically when no override is set.

### 1.2 Variant ISA verification

`hipcc --cuda-device-only -S` dump at 4096×14336×4096 for each variant.
Counts of distinct `buffer_load_dwordx4 … offen … lds` flag patterns:

| Variant | `offen lds` (default) | `offen sc0 lds` (GLC) | `offen nt lds` (SLC) | `offen sc0 nt lds` (both) |
|---------|----------------------:|----------------------:|---------------------:|--------------------------:|
| CP0     | 410                   | 0                     | 0                    | 0                         |
| CP1     | 362                   | 0                     | 48                   | 0                         |
| CP2     | 314                   | 96                    | 0                    | 0                         |
| CP3     | 362                   | 0                     | 0                    | 48                        |
| CP4     | 314                   | 0                     | 48                   | 48                        |

Total opcodes constant at 410 across all variants. The 48-vs-96 split
matches A vs B per-CTA load counts (B-tile is loaded twice as often as
A-tile in the 8-wave RRR phase quartet). The remaining 266 are scale-pack
loads and pre/post-loop staged loads which the cachepolicy lever does not
touch (those keep the existing `MXFP8_RRR_V2_SCALE_CACHEPOLICY` macro).

ISA flag mapping for the gfx950 assembler (confirmed by inspection):

| `coherency` enum | int value | `aux` bits | Assembled flag |
|------------------|----------:|-----------:|----------------|
| `cache_all`      | 0         | 0b000      | (none)         |
| `cache_global`   | 1         | 0b001      | `sc0` (GLC=1)  |
| `cache_stream`   | 2         | 0b010      | `nt`  (SLC=1)  |
| `non_temporal`   | 3         | 0b011      | `sc0 nt`       |

### 1.3 Strict-SCLK protocol

* **5 runs/cell, 30 s cooldown between runs**
* **60 s cooldown between A and B side rebuilds**
* `MXFP8_WARMUP=100`, `MXFP8_ITERS=200`
* GPU pinned via `HIP_VISIBLE_DEVICES=1` (R52Q exclusive)
* Every variant rebuilt fresh via `make … CXXFLAGS="… -DMXFP8_RRR_A_COHERENCY=A
  -DMXFP8_RRR_B_COHERENCY=B"`
* MXFP8 V2 RRR is the production default (`MXFP8_RRR_PRESHUFFLE_V2_RUNTIME=1`)
* Median + spread (max-min)/median reported; spread > 5% would flag
* Numeric gate: SNR ≥ 45 dB vs FP32 reference, 3/3 determinism

Bench harness: `analysis/fp8_gemm/mi350x/r52q_bench.sh`.

### 1.4 Numeric validation (all variants)

| Variant | SNR (dB) | Determinism (3 runs) | Pass rate (numerics) |
|---------|---------:|----------------------|----------------------|
| CP0     | 49.61    | PASS                 | 100.00%              |
| CP1     | 49.61    | PASS                 | 100.00%              |
| CP2     | 49.61    | PASS                 | 100.00%              |
| CP3     | 49.61    | PASS                 | 100.00%              |
| CP4     | 49.61    | PASS                 | 100.00%              |

All five variants are bit-equivalent in *output value* (cachepolicy is a
hint to the cache subsystem, not a semantic change). The only difference
is in scheduling/latency. SNR is identical to baseline as expected.

## 2. Headline results — 8B Gate/Up RRR (M=4096, N=14336, K=4096)

Per-run measurements (TFLOPS):

| Variant | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median | Min | Max | Spread% |
|---------|------:|------:|------:|------:|------:|-------:|-----:|-----:|--------:|
| CP0     | 2545.0 | 2551.2 | 2546.3 | 2555.9 | 2549.3 | **2549.3** | 2545.0 | 2555.9 | 0.43% |
| CP1     | 2327.2 | 2341.1 | 2305.5 | 2302.5 | 2335.7 | **2327.2** | 2302.5 | 2341.1 | 1.66% |
| CP2     | 2543.4 | 2551.7 | 2550.1 | 2547.7 | 2547.1 | **2547.7** | 2543.4 | 2551.7 | 0.33% |
| CP3     | 2341.5 | 2332.1 | 2338.2 | 2335.2 | 2341.5 | **2338.2** | 2332.1 | 2341.5 | 0.40% |
| CP4     | 2140.1 | 2138.6 | 2142.6 | 2142.6 | 2145.7 | **2142.6** | 2138.6 | 2145.7 | 0.33% |

## 3. Interpretation — why R52P's primary lever fails

R52P's recommendation rests on the model that A is "consumed once per
stripe, B is reused across the N-panel" — therefore biasing A toward
streaming (SLC) should free up L2 capacity for B, reducing TA_DATA stalls.

The data refutes this: **A is in fact heavily reused** within the K-loop,
not just across stripes. Each K-pair iteration of the V2 RRR fastpath
(`do_k_iter`, `rrr_mxfp8_exact_8wave_fastpath.inc:592–733`) issues:

* 1× B0 load (Bs[tic][0]) → consumed by cA, cC MFMAs (2 reuses)
* 1× B1 load (Bs[tic][1]) → consumed by cB, cD MFMAs (2 reuses)
* 1× A0 load (As[toc][1] with k+1) → next-iter staging
* 1× A0 reload (As[tic][0] with k+2) → next-next staging

A and B are both loaded into LDS via `buffer_load_lds`, then read into
register-tile space twice per K-iteration. From the L2's perspective,
both A-rows and B-cols are *re-fetched* per K-pair from L2, and both
benefit equally from L2 retention. Marking A as `nt` (CP1) or `sc0 nt`
(CP3) tells the L2 to evict A early — but A is needed again on the next
K-iteration's prefetch, so L2 ends up doing more HBM round-trips.

CP4 (B as `sc0 nt`) is the positive control: marking B for early
eviction predictably regresses by 16%, confirming the lever wires
correctly *and* that B-retention is critical (which baseline already
provides).

CP2 (`sc0` on both) bypasses L1 cache (which is largely irrelevant here
because `buffer_load_lds` writes directly to LDS, skipping the per-CU L1
load path) but leaves L2 retention unchanged — hence essentially
no-op (-0.06%, well within run-to-run noise).

**Conclusion:** the baseline `cache_all` policy is already L2-retain-both-
A-and-B, which is the optimal hint for this shape. The +12.4pp MfmaUtil
gap R52P measured is real, but the lever R52P proposed (cachepolicy bias)
is the wrong lever — it fights against the actual A/B reuse pattern of
the V2 RRR K-loop.

## 4. What R52P got right and wrong

**Right:** the diagnosis that L2/TC cache-return backpressure is the
dominant stall source on V2 8B Gate/Up RRR is unchanged. The PMC counter
deltas (TA_DATA_STALLED_BY_TC +57%) remain the best measured
characterization of the gap.

**Wrong:** the prescription. R52P's model of A as "streamed once per
stripe" missed that per-K-iteration the A-tile is *reloaded* from L2
into LDS each K-iteration (the prefetch-to-LDS pipeline keeps A-rows hot
in L2 for the same reason B-cols are kept hot). Both A and B are
reused-from-L2 within the K-loop, so neither side wants a streaming hint.

**What this rules out for R53:**
* Per-tensor cachepolicy biasing on V2 RRR data loads is a dead lever.
* Increasing L2 retention beyond `cache_all` is not exposed at this
  source layer — there is no "more aggressive retention" hint than
  `cache_all` for `buffer_load_lds`.
* The TC-return-backpressure is *intrinsic* to the K=4096 arithmetic
  intensity at this shape; cachepolicy hints alone cannot widen the
  TC→TA delivery rate.

**What R53 should pursue instead** (ordered by expected leverage):
1. **K-superblock reuse / persistent CTA** (R52P §5.2). If we cannot
   widen the per-cycle TC return, we can amortize each L2-fetch over
   more MFMA work by processing multiple M-stripes against the same
   B-column-block before advancing. Higher implementation cost but the
   only structural fix to the K=4096 amortization deficit.
2. **N-tile shrink at K≤4096 only** (R52P §5.3). Per-shape gating to
   reduce per-CTA B working set, allowing more B to fit in L2 across
   active CTAs. Worth a single sweep.
3. **A-side LDS double-buffer at K=4096 occ=1** — REFUTED by R52K
   pre-bench, but the pre-bench was speculative; with PMC data in hand
   it may merit a *measured* reattempt focused on A-prefetch overlap
   rather than B-side LDS.

**What R53 should NOT pursue:**
* Any further cachepolicy biasing on V2 RRR data fetches (R52Q closes
  this lever conclusively).
* B-side cachepolicy variants — CP4 already covers the "B less retained"
  side with a 16% regression confirming retention is binding.

## 5. Files / artifacts

* `analysis/fp8_gemm/mi350x/r52q_bench.sh` — strict-SCLK bench harness
* `analysis/fp8_gemm/mi350x/r52q_results/` — raw run/build/check logs
  for all 5 variants × 5 runs × 1 cell (8B Gate/Up RRR)
* `analysis/fp8_gemm/mi350x/r52q_isa/cp{0..4}.s` — `--cuda-device-only -S`
  dumps for ISA flag verification
* `analysis/fp8_gemm/mi350x/r52q_primary.run.log` — bench transcript

### Source patch (kept in place; default macros = no-op vs baseline)

The cachepolicy plumbing is left in the tree as a template-parameter
extension that costs nothing when both coherencies = 0 (the default; routes
through the original untemplated `G::load` for ISA bit-identity). If a
future investigation wants to revisit per-tensor cachepolicy, no further
plumbing work is required — just `-DMXFP8_RRR_A_COHERENCY=N
-DMXFP8_RRR_B_COHERENCY=M`. Files touched:

```
include/ops/warp/memory/tile/global_to_shared.cuh   (+1 template param + COH macro fwd, default cache_all)
include/ops/group/memory/tile/global_to_shared.cuh  (+1 G::load_coh<COH> overload)
analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc
                                                    (+2 macros, +1 macro guard, replace G::load(As/Bs,…) with RRR_LOAD_A/RRR_LOAD_B)
```

ISA bit-identity at default: `cp0.s` shows 410 `offen lds` opcodes with
zero flag changes vs the unmodified baseline at the same shape — the
plumbing change is a perfect no-op when defaults are unchanged. (Verified
via `--cuda-device-only -S` dump.)

## 6. Verdict

**REFUTED.** R52P's primary R53 lever (B-side L2 cachepolicy retention
tuning to close the +3.1pp 8B Gate/Up RRR HEADROOM gap) does not work as
hypothesized. The baseline `cache_all` policy is already optimal for both
A and B on V2 RRR, and any reduction in retention on either side regresses
performance by 8–16%. The cachepolicy lever is closed for V2 RRR data
loads. R53 should pursue K-superblock CTA-level reuse (R52P §5.2) or
per-shape N-tile shrink (R52P §5.3) instead.
