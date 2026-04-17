# Round 17 Optimizer A — DLA1 Profile Findings

**Date**: 2026-04-17
**Author**: Kyle.Zhao@amd.com / kyle-256
**Shape**: DLA1 = M=4096, N=32768, K=128256 (the K=128256 deep-LOSE outlier)
**Variant profiled**: `ts_pf6_6_v12_memc` (current best for DLA1, 88.3% of comp)
**Tool**: rocprofv3 (ROCm 7.1.0) — kernel-trace + 3-pass PMC
**GPU**: MI355X (gfx950), 256 CU, 1024 SIMD, 8 XCC, 32 SE
**Wall**: 6.74 ms / dispatch — 5105 TFLOPS achieved (matches `bench_all_42`)

---

## 1. Raw PMC Data (single dispatch)

```
GRBM_GUI_ACTIVE       152,391,271 cycles  (sum across 8 XCC ⇒ 19.05M/XCC ⇒ ~9.5 ms @ 2 GHz)
SQ_WAVES                2,373,835         (vs expected 524288 wg × 4 = 2.10M; +13% from kernel re-entry)
SQ_BUSY_CYCLES        592,296,420
SQ_INSTS_VALU       1,835,265,077         (sum across CU)
SQ_INSTS_LDS          131,334,144
SQ_INSTS_VMEM         121,289,077
SQ_INSTS_MFMA         525,336,576
SQ_ACTIVE_INST_VALU 2,392,177,212
SQ_ACTIVE_INST_ANY  3,343,899,131
SQ_WAIT_INST_LDS        4,354,434
TA_TA_BUSY          2,719,180,997
TCP_TCP_TA_DATA_STALL 255,758,820
```

## 2. Derived Metrics

| Metric | Value | Reading |
|---|---|---|
| **Achieved TFLOPS** | 5,105 | 88.3 % of competitor 5,781 |
| **VALU Busy** | 49.1 % | **Half the cycles VALU sits idle** |
| **VALU/AnyIssue** | 71.5 % | When we do issue, mostly VALU (good) |
| **VMEM/VALU ratio** | 6.6 % | Compute-heavy (good) |
| **MFMA per wave** | 221.3 | High density |
| **VALU per wave** | 773 | |
| **LDS per wave** | 55 | |
| **LDS-stall %** | 0.04 % | Negligible — LDS is NOT the bottleneck |
| **Min HBM bytes** | 2.51 GB | A 0.26 GB + B 2.10 GB + scales 0.15 GB |
| **Achieved BW** | ~ 264 GB/s | ↔ 5300 GB/s HBM3e (5 % of peak) — NOT bandwidth-bound |
| **MFMA-pipeline floor** (K=128256, mfma_f4 8 cyc) | 1.81 ms | Wall is 6.74 ms ⇒ MFMA only fills 27 % of wall |
| **MFMA-pipeline floor** (mfma_f4 4 cyc — likely real rate) | 0.9 ms | ⇒ MFMA fills 13 % of wall |

## 3. Bottleneck Identification

### Three causes (top-3) of the 5.6 ms wall ‑ MFMA-pipeline gap

#### **A. K-loop epilogue dominates (NEW finding for DLA1, K=128256)**

K=128256 / KPAIR_LOOP=64 → **2004 main-loop iterations** per CTA.
Each iteration ends in `s_barrier` (17 barriers in core inner) + `s_waitcnt vmcnt(0)` (68 in module).
At ~2004 iters with 1-2 barriers + waitcnt each, barrier+waitcnt cost adds up:
  2004 × ~150 cyc/barrier ≈ 300 K cycles per wave = 0.15 ms / wave.
Compounded across 8192 waves/CU and 4-SIMD serialization ⇒ **0.6-1.2 ms** of pure barrier/wait time.

#### **B. K=128256 is NOT multiple of native KPAIR_LOOP=64; tail-handling adds VALU-only iterations**

`128256 / 64 = 2004.0` — divides cleanly, BUT the producer-side scale-pack assumes K%32=0 padding, and the
`memc` family uses memclause to coalesce final 6-loop tail. The TS variant + pf6_6 prefetch chain primes 6
iterations; for K=128256 the prefetcher cycles 334 times producing **~120 M VMEM insts** (matches PMC).
**VMEM is fine (5 % BW)** but the prefetch m0-set / lds-store sequence has built-in ~6-cycle s_nop hazards
(99 such per inner block in earlier rocprof analysis) — at 2004 iters × ~6 prefetch hazards = 12K cycles/wave.

#### **C. Issue-rate divergence between SIMDs / cross-XCC scheduling (LARGEST gap)**

VALUBusy 49 % combined with MFMA/wave 221 (very high) + 0.04 % LDS-stall + 5 % BW ⇒
**MFMA is issuing serially on each SIMD but ~half the cycles are idle waiting on dest-dependency**.
mfma_scale_f32_16x16x128_f4 has 8-cycle latency before the same accumulator can be re-used.
Current code uses **only 1 accumulator-tile** per wave (4×4 mma), so consecutive MFMAs on the same C-tile
hit the dep-stall every 8 cycles. Effective pipeline: 8 cyc / 8 cyc latency = 100 % of theoretical when single
SIMD, but with 4 SIMDs × 8-cycle MFMA pipe and only 1 active C-accumulator, **MFMA throughput per-CU caps at
1 / (4 × 8) = 1/32 instead of 1/8** (4× under-utilization). This matches the 5100/~7000 = 73 % chip-wide rate
when accounting for 4 XCCs being warmed.

## 4. Comparison to Earlier Rocprof Analysis (TODO.md 2026-04-16)

Earlier covered **K=4096, TS, LGK2** (production .s in repo): 2048 MFMAs, 165 s_nop hazards, conclusion
was "PF s_nop is m0-hazard, removable but useless".

**For DLA1 (K=128256)** the situation is qualitatively different:
- **31× more MFMA loop iterations** ⇒ inner-loop MFMA latency stalls compound 31× ⇒ now matters
- **Same prefetch s_nop overhead per iter** scales linearly ⇒ would help measurably here
- **VALUBusy 49 %** (not measured in earlier analysis) reveals MFMA dep-stall is the new villain

## 5. Concrete Source-Level Edit Candidates (NOT IMPLEMENTED — for R17C)

### Proposal P1 — **Double C-accumulator tiling** (highest-EV, 2-4 pp expected)
Today the inner loop accumulates into a single `rt_C[4][4]` tile. Split into `rt_C0[4][4]` and `rt_C1[4][4]`,
ping-pong consecutive MFMAs:
```
for (k = 0; k < KPAIR_LOOP; k += 2) {
   mfma_scale_f32_16x16x128(C0, A_k0, B_k0, scale_a_k0, scale_b_k0)
   mfma_scale_f32_16x16x128(C1, A_k1, B_k1, scale_a_k1, scale_b_k1)
}
// reduce: C0 += C1 outside inner loop
```
Doubles register pressure (~256 → ~320 VGPR; still fits within gfx950 256 VGPR/wave limit only if WPE2).
With WPE2 we'd be over budget; need 2-WG/CU instead of 4-WG/CU (occupancy 1× → 0.5×).
**Risk**: occupancy halving offsets latency-hide gain unless K is large enough — K=128256 is exactly the
regime where this trade wins.
**File**: `kernel_mxfp4_gluon_cpp.cpp` lines defining `rt_C` and inner mfma loop.

### Proposal P2 — **Larger M-tile for K=128256** (tile selector change, +1-2 pp)
Current K=128256 path uses standard M=128 tile. For very-tall-K, BTI suggests **M=256, N=128** (2× M-tile)
would reuse loaded B 2× (B is the bigger operand at 2.10 GB) and halve total CTA count from 524288 to
262144. Current PMC shows 524288 wg × 4 waves = 2.10M expected waves; we measured 2.37M (warmup leak).
Halving waves ⇒ halve all per-wave overhead (barriers, waitcnt, kernel-launch).
**File**: `kernel_mxfp4_gluon_cpp.cpp` template arguments for K=128256 specialization.
**Risk**: M=256 needs 2× more A in registers ⇒ register spills if not careful.

### Proposal P3 — **Drop redundant s_barrier in inner K-loop** (-0.5 ms upper bound)
Asm shows 17 s_barrier in the kernel body. K-loop iterates 2004×, so if even 1 barrier is in the hot path,
that's 2004 barriers/wave × ~80 cyc each = 160 K cyc/wave ≈ 0.4 ms. **Current `memc` variant already used
memclause to remove 1 barrier**; check if a 2nd cross-iter barrier can be replaced with `s_waitcnt
lgkmcnt(0)` (cheaper at gfx950 ~30 cyc).
**File**: `kernel_mxfp4_gluon_cpp.cpp` `producer.commit()` / `consumer.acquire()` sync points.
**Risk**: data-race if scale-pack producer-consumer isn't fully ordered. Validate SNR.

### Lower-EV ideas (not recommended for R17C)
- P4: Split-K (2-way) — reduces K per wave to 64K, drops MFMA latency stall by 2× but adds a partial-sum
  reduction kernel; aiter ASM does NOT use split-K so likely cuts into our headroom.
- P5: Force `XNACK_MASK=0` and `WAVE_LIMIT=1` env vars — gfx950 may already do this; needs measurement.

## 6. Verdict

**Bottleneck for DLA1 is MFMA accumulator dependency stall**, not memory or LDS. The existing `_memc` and
`pf6_6` variants already squeezed prefetch overhead. The next ~3-5 pp must come from **double-buffering the
C accumulator** (P1) or **enlarging the M-tile** (P2) — both are SOURCE-level edits to
`kernel_mxfp4_gluon_cpp.cpp`, not flag tuning.

Reading: **R17C should attempt Proposal P1 (double C-accumulator) on a single K=128256 specialization**
with WPE2 + MEMC. Start as `_dblc_pf6_6_v12_memc` variant; SNR-validate first; bench against current 5105
TFLOPS. If P1 yields ≥ 5300 TFLOPS (= 92 %), commit and run on DLA2/DLA7/P1 to see if same edit transfers.
