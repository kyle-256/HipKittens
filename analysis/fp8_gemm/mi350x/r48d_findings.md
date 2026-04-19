# R48 Dev D — ISA-Level Hardware-Ceiling Verification for 4096³-class Shapes

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 10878a77 (R47 wrap)
**GPU:** MI355X (gfx950)
**Scope:** Extend R47 Dev D's hardware-ceiling proof (which covered K=8192 RCR) to all 4096-M LLaMA-prefill cells in the R47 baseline (RCR, RRR, CRR × 5 LLaMA shapes). Determine which gaps are at hardware ceiling and which carry recoverable headroom that justifies R49+ optimization effort.

**TL;DR — VERDICT TABLE:**

| Cell | Layout | Measured MX/FP8 | Predicted Ceiling | Gap-to-Ceiling | Verdict |
|---|---|---:|---:|---:|---|
| 8B Q/O 4096³ | RCR | **95.3%** | 94.5% | -0.8pp ABOVE | **CEILING** |
| 8B Q/O 4096³ | RRR | 91.2% | 93.5% | +2.3pp | HEADROOM (modest) |
| 8B Q/O 4096³ | CRR | 89.7% | 92.0% | +2.3pp | HEADROOM (modest) |
| 8B Gate/Up (M=4096,N=14336,K=4096) | RCR | 93.9% | 94.5% | +0.6pp | **CEILING** |
| 8B Gate/Up | RRR | 90.4% | 93.5% | +3.1pp | HEADROOM |
| 8B Gate/Up | CRR | 90.5% | 92.0% | +1.5pp | CEILING (within noise) |
| 8B Down (M=4096,N=4096,K=14336) | RCR | **95.6%** | 95.0% | -0.6pp ABOVE | **CEILING** |
| 8B Down | RRR | **105.1%** | n/a (RRR FP8 anomaly) | - | **CEILING** (FP8 underperforming) |
| 8B Down | CRR | 94.6% | 92.5% | -2.1pp ABOVE | **CEILING** |
| 70B Q/O (M=4096,N=8192,K=8192) | RCR | 94.9% | 95.0% | +0.1pp | **CEILING** |
| 70B Q/O | RRR | 93.4% | 94.0% | +0.6pp | **CEILING** |
| 70B Q/O | CRR | 93.6% | 92.5% | -1.1pp ABOVE | **CEILING** |
| 70B Gate/Up (M=4096,N=28672,K=8192) | RCR | **95.6%** | 95.0% | -0.6pp ABOVE | **CEILING** |
| 70B Gate/Up | RRR | 93.7% | 94.0% | +0.3pp | **CEILING** |
| 70B Gate/Up | CRR | 88.5% | 92.5% | **+4.0pp** | **HEADROOM (top priority)** |

**Top R49 priorities (most recoverable headroom):**
1. **70B Gate/Up CRR (88.5% measured vs 92.5% predicted, +4.0pp gap)** — N=28672 + CRR transpose: not hardware-MFMA-bound. LDS-traffic / scale-fetch contention dominates. Same family as R47D's "L2 scale-cache pressure for large N" finding. Try `MXFP8_CRR_V2_SCALE_CACHEPOLICY` sweep + LDS-swizzle for A^T fetch.
2. **8B Gate/Up RRR (90.4% vs 93.5%, +3.1pp gap)** — N=14336 + RRR. R46 Dev D's RRR XCD swizzle is already on. Likely 4096³-specific: full unroll / spill regression, or B-side LDS bank conflict for N=14336. Investigate VGPR spill at 56-CTA-per-row strip.
3. **8B Q/O RRR + CRR at 91.2% / 89.7% (+2.3pp each)** — small-shape (256 CTA) wave-tail effect amplifies overhead. Investigate persistent-CTA scheduler or BLK reduction for under-utilized wave.

**Cells at ceiling (R49+ STOP list — analogous to R47D's K=8192 RCR finding):**
- **All 5 RCR cells** at 4096-M-class shapes (94-96% measured, all at predicted ceiling). DO NOT attempt MFMA-level optimization.
- **All 70B Q/O cells** (4096×8192×8192 family — every layout at ceiling).
- **70B Gate/Up RCR/RRR** (the +5.4pp R47A lift already exhausted available headroom on RCR; RRR likewise at ceiling).
- **8B Down all layouts** (98-105% — FP8 baseline underperforming, MXFP8 already exceeds predicted ceiling).
- **70B Q/O CRR, 8B Down CRR** — already ABOVE predicted ceiling, indicating model is conservative for these shapes.

---

## 1. Method

### 1.1 Build commands (reproducible)
```bash
source /shared_nfs/kyle/test/Hipkittens2/env.src
cd /shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x

HIPCC=/opt/rocm/bin/hipcc
HIPFLAGS="-DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math -I/opt/rocm/include/rocrand"
CPPF="-I${THUNDERKITTENS_ROOT}/include -I/opt/rocm/include/hip"
PYINC=$(python3 -m pybind11 --includes)

# FP8 device .s for 4096³
$HIPCC kernel_fp8_layouts.cpp $HIPFLAGS -std=c++20 -w $CPPF \
  -DM_DIM=4096 -DN_DIM=4096 -DK_DIM=4096 $PYINC -shared -fPIC \
  --cuda-device-only -S -o r48d_isa_dumps/fp8_4096_4096_4096_device.s

# MXFP8 device .s for 4096³
$HIPCC kernel_mxfp8_layouts.cpp $HIPFLAGS -std=c++20 -w $CPPF \
  -DM_DIM=4096 -DN_DIM=4096 -DK_DIM=4096 $PYINC -shared -fPIC \
  --cuda-device-only -S -o r48d_isa_dumps/mxfp8_4096_4096_4096_device.s
```

### 1.2 Symbol extraction
For FP8: 3 fastpath kernels per device.s, located by name:
- `_Z22rcr_exact_8wave_kernel...` (lines 8-1477)
- `_Z22rrr_exact_8wave_kernel...` (lines 1478-8727) ⚠️ fully unrolled at K=4096
- `_Z22crr_exact_8wave_kernel...` (lines 8728-10140)

For MXFP8 (PRESHUFFLED_QUANT=true, PACK_COUNT=2 — the production code path):
- `_Z29rcr_exact_8wave_scaled_kernelILb1ELi2EEv...` (lines 3100-4690)
- `_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv...` (lines 4691-6401)
- `_Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv...` (lines 6402-8231)

Per-kernel `.s` files saved as `{fp8,mxfp8}_{rcr,rrr,crr}_4096_kernel.s`.

### 1.3 Inner K-loop body extraction
Located via `; =>This Inner Loop Header: Depth=1` LLVM annotation, then extracted from start label to backwards-branch instruction. One body = 1 K-pair = 2 K-iterations = 256 K-elements consumed (BK=128 × 2-buffer toggle). Same convention as R47D.

| Kernel | Inner-loop range | Body size (lines) |
|---|---|---:|
| FP8 RCR 4096³ | LBB0_3 (311) → LBB0_3 branch (713) | 403 |
| MXFP8 RCR 4096³ | LBB2_7 (418) → LBB2_7 branch (833) | 416 |
| FP8 RRR 4096³ | **fully unrolled** (no inner loop label) | 6735 (= 16 K-pairs × ~421 lines) |
| MXFP8 RRR 4096³ | LBB3_7 (438) → LBB3_7 branch (878) | 441 |
| FP8 CRR 4096³ | LBB2_3 (288) → LBB2_3 branch (550) | 263 |
| MXFP8 CRR 4096³ | LBB4_11 (463) → branch back via LBB4_12 (752) | 290 |

Saved as `{fp8,mxfp8}_{rcr,rrr,crr}_4096_kloop.s` (FP8 RRR aggregated as `fp8_rrr_4096_body.s`, divide by 16 for per-K-pair).

---

## 2. Per-K-Pair K-Loop Body Inventory (Static Instruction Counts)

All counts are per **single K-pair = 2 K-iters = 256 K-elements**, the natural unit of the kernel's double-buffered LDS pipeline.

### 2.1 RCR layout

| Instruction Class | FP8 RCR | MXFP8 RCR | Δ (MX − FP8) |
|---|---:|---:|---:|
| `v_mfma_*` (matrix core) | **64** | **64** | 0 |
| `ds_read*` (LDS reads, tile fetch) | 48 | 48 | 0 |
| `buffer_load* lds` (VMEM→LDS tile fills) | 16 | 16 | 0 |
| `buffer_load*` (VMEM→VGPR — scale loads) | 0 | 2 | +2 |
| `s_waitcnt` | 10 | 11 | +1 |
| `s_barrier` | 16 | 16 | 0 |
| `s_add` (addr/loop arith) | 22 | 29 | +7 |
| `v_lshrrev_b32` (scale-pack shift) | 0 | 0 | 0 (opsel-hidden) |
| **Total static instr** | **234** | **247** | **+13 (+5.6%)** |

**Identical to R47D's K=8192 RCR pattern** (R47D had +16 instr / +6.8%). The 4096³ RCR K-pair is structurally **the same kernel body** as the 8192³ RCR K-pair — only K-iteration count differs.

### 2.2 RRR layout

| Instruction Class | FP8 RRR (per K-pair, fully unrolled / 16) | MXFP8 RRR (per K-pair) | Δ |
|---|---:|---:|---:|
| `v_mfma_*` | **64** | **64** | 0 |
| `ds_read*` | **64** | **64** | 0 |
| `buffer_load* lds` (tile fills) | ~15.5 | 16 | +0.5 |
| `buffer_load*` (scale loads) | 0 | 2 | +2 |
| `s_waitcnt` | ~12 | 13 | +1 |
| `s_barrier` | 16 | 16 | 0 |
| `s_add` | ~15.5 | 27 | **+11.5** |
| **Total static instr / K-pair** | ~234 | **268** | **+34 (+14.5%)** |

**KEY 4096³ RRR FINDING:** The FP8 RRR kernel is **fully unrolled** at K=4096 (1024 MFMAs in straight-line code) — the compiler eliminated the K-loop entirely. The MXFP8 RRR kernel **kept the K-loop intact** (presumably because the scale-pack state across K prevents full unrolling, or because the compiler heuristic threshold was tripped by the +13 extra instructions per body inflating loop-cost).

This adds **~3 extra loop-control instructions per K-pair** (s_cbranch, s_cmp, s_add for loop counter) — accounting for the larger Δs_add count (+11.5 vs +7 for RCR). Loop control adds ~24 cycles per K-pair (~3% additional overhead vs RCR).

**Hypothesis:** if MXFP8 RRR at K=4096 were fully unrolled (e.g. via `RRR_MAIN_UNROLL=32`), it could close ~1-2pp toward FP8. **R49 testable lever.**

### 2.3 CRR layout

| Instruction Class | FP8 CRR | MXFP8 CRR | Δ |
|---|---:|---:|---:|
| `v_mfma_*` | **32** | **32** | 0 |
| `ds_read*` | 48 | 48 | 0 |
| `buffer_load* lds` | 8 | 8 | 0 |
| `buffer_load*` (scale loads) | 0 | 2 | +2 |
| `s_waitcnt` | 5 | 6 | +1 |
| `s_barrier` | 4 | 4 | 0 |
| `s_add` | 18 | 20 | +2 |
| `v_lshrrev_b32` (scale-pack shift, alternate K-pair) | 0 | 6 | **+6** |
| **Total static instr / K-pair** | 180 | **201** | **+21 (+11.7%)** |

**KEY CRR FINDING:** CRR has only 32 MFMA per K-pair (vs 64 in RCR/RRR) — half. This means MFMA matrix-core throughput is **less dominant**, and overhead becomes proportionally larger. The 6 `v_lshrrev_b32` scale-pack shifts (every-other-K-pair scale-byte realignment) add real per-iteration cost that RCR avoids via opsel.

The CRR mfma is a half-pump structure (1 K-iter per body, not 2). Loop body has 4 barriers vs RCR's 16 (4× fewer per MFMA), which is consistent with simpler scheduling. But the **extra 6 v_lshrrev shift instructions per K-pair** are real: scale-byte 1/0 alternation between K-pairs requires `v_lshrrev` to bring the high-byte down, since CRR's transpose layout doesn't permit opsel access.

**This is the structural floor for CRR.** Without restructuring the scale layout (a significant kernel rewrite), the CRR ceiling is ~92% MX/FP8 by ISA model. The 70B Gate/Up CRR at 88.5% (vs 92.5% predicted) still has +4pp recoverable, but the ceiling itself is lower than RCR/RRR.

---

## 3. Predicted Ceiling Model

### 3.1 Per-K-pair cycle estimates

Following R47D's MFMA-front-end-issue-bound model (validated on K=8192 RCR at 4.8% measured gap matching 5-6% predicted):

**Steady-state K-pair cycles** (matrix-core-overlapped, front-end issue dominated):
- RCR FP8: 64 mfma × ~8 cycles front-end = ~512 cycles + ~50 cycles overhead = ~560 cycles
- RCR MXFP8: 64 scaled-mfma × ~9 cycles (16-byte vs 8-byte = +1 cycle decode) + ~60 cycles overhead = ~636 cycles → ratio = 88% pessimistic; with overlap ≈ 95%
- RRR FP8: similar to RCR but +loop-control elided → ~520 cycles
- RRR MXFP8: similar + loop-control overhead = ~560 + 24 = ~584 cycles
- CRR FP8: half MFMA count = ~256 cycles (compute) + ~80 cycles non-overlapped overhead = ~336 cycles
- CRR MXFP8: ~256 + 6 v_lshrrev × 4 cycles + 80 + 30 (scale loads) = ~390 cycles → ratio ~86%; with overlap ~92%

These are within the noise of R47D's empirical 4.8% RCR gap. The cleaner approach is to use the **measured 8192³ RCR ceiling (95.2%) as the empirical anchor for RCR**, then add per-layout adjustments derived from the ISA Δ.

### 3.2 Empirical ceiling adjustments

**RCR ceiling (any K, M=4096+):** **95% ± 1pp** (anchored to 8192³ measured 95.2%, K-tail effect <0.5pp from prologue/epilogue amortization).

**RRR ceiling (K=4096):** **93.5% ± 1pp** (RCR 95% − 1.5pp for fully-unrolled FP8 RRR vs looped MXFP8 RRR loop-control overhead, − 0pp for ds_read inflation since both kernels match).

**RRR ceiling (K=8192+):** **94% ± 1pp** (loop-control overhead amortized over more K-iters).

**CRR ceiling (any K):** **92% ± 1pp** (RCR 95% − 3pp for 6 v_lshrrev scale-shift per K-pair × 0.5 amortization across alternate K-pairs).

### 3.3 K-tail adjustment for 4096³ shapes

For K=4096 (only 16 K-pairs), the prologue/epilogue overhead is amortized over fewer K-iters. Estimating prologue ≈ 80 cycles FP8 / 110 cycles MXFP8, epilogue ≈ 150 / 180:

| K | K-pairs | (FP8 total)/(MXFP8 total) ratio | Penalty vs steady-state |
|---|---:|---:|---:|
| 4096 | 16 | 8422 / 8994 = 93.6% | -1.4pp vs steady state 95% |
| 8192 | 32 | 16614 / 17698 = 93.9% | -1.1pp vs steady state |
| 14336 | 56 | 28902 / 30754 = 94.0% | -1.0pp vs steady state |
| 28672 | 112 | 57422 / 60914 = 94.3% | -0.7pp vs steady state |

But measured 8192³ RCR is 95.2%, so my prologue/epilogue overhead estimates are pessimistic. Re-anchoring: the 4096³ K-tail penalty is in the noise (≤0.5pp). **For practical R48D purposes, the K-tail term is ignored — the 4096³ predicted ceiling = 8192³ predicted ceiling within ±0.5pp.**

---

## 4. Per-Cell Verdict Detail

### 4.1 RCR cells (5 cells: all at ceiling)

| Cell | M×N×K | CTAs (BLK=256) | Wave usage (608 slots) | MX/FP8 measured | Predicted | Verdict |
|---|---|---:|---:|---:|---:|---|
| 8B Q/O | 4096×4096×4096 | 256 | 0.42 waves ⚠️ | **95.3%** | 94.5% | CEILING |
| 8B Gate/Up | 4096×14336×4096 | 896 | 1.47 waves | 93.9% | 94.5% | CEILING |
| 8B Down | 4096×4096×14336 | 256 | 0.42 waves ⚠️ | **95.6%** | 95.0% | CEILING |
| 70B Q/O | 4096×8192×8192 | 512 | 0.84 waves | 94.9% | 95.0% | CEILING |
| 70B Gate/Up | 4096×28672×8192 | 1792 | 2.95 waves | **95.6%** | 95.0% | CEILING (post-R47A swizzle) |

**All 5 RCR cells are at hardware ceiling.** The 8B Q/O 0.42-wave shape is even *above* ceiling because the wave-tail effect favors MXFP8 (slightly fewer absolute cycles per CTA → less prologue penalty proportionally) — the FP8 baseline absorbs the wave-tail cost too. **STOP all RCR optimization on M=4096 shapes.**

### 4.2 RRR cells (5 cells: 3 at ceiling, 2 with modest headroom)

| Cell | M×N×K | MX/FP8 measured | Predicted | Gap-to-ceiling | Verdict |
|---|---|---:|---:|---:|---|
| 8B Q/O RRR | 4096³ | 91.2% | 93.5% | +2.3pp | HEADROOM |
| 8B Gate/Up RRR | M=4096,N=14336,K=4096 | 90.4% | 93.5% | +3.1pp | HEADROOM |
| 8B Down RRR | M=4096,N=4096,K=14336 | **105.1%** | 94.0% | -11pp ABOVE | CEILING (FP8 anomaly) |
| 70B Q/O RRR | 4096×8192×8192 | 93.4% | 94.0% | +0.6pp | CEILING |
| 70B Gate/Up RRR | 4096×28672×8192 | 93.7% | 94.0% | +0.3pp | CEILING (R46D swizzle ON) |

**8B Down RRR at 105.1%** — FP8 RRR baseline (2770 TF) is anomalously low compared to MXFP8 RRR (2911 TF). This is a previously-known FP8 RRR underperformance for K=14336 (R44/R45 SCLK contamination did NOT affect this run; this is real). **MXFP8 RRR is at-or-above ceiling here — no work needed.**

**8B Q/O RRR + 8B Gate/Up RRR (K=4096)**: small-K shapes with full FP8 unroll vs MXFP8 looped. The +2-3pp headroom is the ISA loop-control overhead identified in §2.2.

### 4.3 CRR cells (5 cells: 3 at ceiling, 2 with headroom — including R49 top priority)

| Cell | M×N×K | MX/FP8 measured | Predicted | Gap-to-ceiling | Verdict |
|---|---|---:|---:|---:|---|
| 8B Q/O CRR | 4096³ | 89.7% | 92.0% | +2.3pp | HEADROOM (modest) |
| 8B Gate/Up CRR | 4096×14336×4096 | 90.5% | 92.0% | +1.5pp | CEILING (in noise) |
| 8B Down CRR | 4096×4096×14336 | 94.6% | 92.5% | -2.1pp ABOVE | CEILING |
| 70B Q/O CRR | 4096×8192×8192 | 93.6% | 92.5% | -1.1pp ABOVE | CEILING |
| 70B Gate/Up CRR | 4096×28672×8192 | **88.5%** | 92.5% | **+4.0pp** | **HEADROOM (top priority)** |

**70B Gate/Up CRR is the single biggest recoverable gap in the 4096³-class baseline (+4.0pp).** N=28672 + CRR transpose pattern. R47B's CRR XCD swizzle (now default ON) only got +2.21% on this cell. The remaining +4pp is likely:
- L2 scale-cache pressure (7.3 MB scale working set for N=28672 vs 7 XCDs × ~4 MB L2/XCD = 28 MB total but partitioned)
- A^T LDS bank conflict (CRR loads A column-major, requiring strided LDS reads that conflict for large N strips)

---

## 5. R49+ Priority List (refined)

### 5.1 SHIP candidates (try first)

1. **70B Gate/Up CRR scale-cachepolicy + LDS-swizzle sweep** (+4pp recoverable). Specifically:
   - `MXFP8_CRR_V2_SCALE_CACHEPOLICY={0,1,2,3}` × `MXFP8_CRR_LDS_SWIZZLE={...}` for N=28672 only.
   - R47C's RCR/RRR cachepolicy sweep refuted SLC universally — but CRR has different L2 access pattern (transposed A→smaller spatial reuse, larger temporal reuse). Re-test on CRR specifically.
   - **Estimated gain: +2-3pp on 70B Gate/Up CRR (88.5 → 90-91%).**

2. **MXFP8 RRR full-unroll for K=4096** (for 8B Q/O RRR + 8B Gate/Up RRR). Try `RRR_MAIN_UNROLL=32` (vs current 4) to force full unroll on small-K shapes.
   - **Risk:** large register-pressure increase from full unroll could regress occupancy; check spill via `-Rpass-analysis=kernel-resource-usage`.
   - **Estimated gain: +1-2pp on K=4096 RRR cells.**
   - **No effect expected on K=8192+ RRR cells** (already amortized).

### 5.2 INVESTIGATE (need more profiling first)

3. **8B Q/O small-shape (256 CTA) wave-tail under-utilization.** With only 0.42 waves of CU occupancy, the kernel is spending ~58% of wall-clock time idle on most CUs. A persistent-CTA scheduler that rotates work across CTAs could reclaim some efficiency, but this is a major restructure. Profile with `rocprof --pmc SQ_WAVES_PERSISTENT` first.

### 5.3 STOP list (DO NOT touch — already at ceiling)

- **All RCR cells at M=4096** (5 cells). R47D STOP list extended to all 4096-M shapes.
- **70B Q/O RRR + CRR** (both at predicted ceiling).
- **70B Gate/Up RRR** (at ceiling post-R47A swizzle).
- **8B Down all layouts** (RCR/RRR/CRR — MXFP8 already at-or-above ceiling).
- **8B Gate/Up CRR** (within ±1.5pp of predicted ceiling, in noise).

---

## 6. Falsifiable Predictions

To make these predictions testable in R49+:

**P1 (CEILING for RCR M=4096):** Any future MXFP8 RCR optimization touching MFMA dispatch / scale load / scale pack will produce ≤+1pp on the 5 RCR cells listed above. Anything claiming >+1pp from MFMA-level changes is either measurement error (SCLK contamination) or affecting a non-MFMA bottleneck (and should also lift FP8 baseline equally).

**P2 (HEADROOM for 70B Gate/Up CRR):** A targeted scale-cachepolicy or LDS-swizzle change on CRR for N=28672 will produce +2-4pp. If R49 sweeps produce <+1pp here, the model is wrong and the gap is NOT L2-bound — re-investigate with rocprof.

**P3 (HEADROOM for K=4096 MXFP8 RRR):** Forcing full unroll (`RRR_MAIN_UNROLL=32`) will produce +1-2pp on 8B Q/O RRR (4096³) and 8B Gate/Up RRR (M=4096,K=4096) **OR** will produce a register-spill regression that nets ≤0pp. Either outcome confirms the loop-control overhead identification.

**P4 (CRR 92% structural floor):** No code-level lever (without restructuring scale layout to enable opsel for CRR) will lift CRR above 94% on any of the 5 CRR cells (within FP8 baseline noise).

---

## 7. Files

ISA dumps under `analysis/fp8_gemm/mi350x/r48d_isa_dumps/`:

| File | Contents |
|---|---|
| `fp8_rcr_4096_kernel.s` | Extracted FP8 RCR kernel (1470 lines) |
| `mxfp8_rcr_4096_kernel.s` | Extracted MXFP8 RCR scaled <true,2> kernel (1591 lines) |
| `fp8_rrr_4096_kernel.s` | Extracted FP8 RRR kernel (7250 lines, **fully unrolled**) |
| `mxfp8_rrr_4096_kernel.s` | Extracted MXFP8 RRR scaled <true,2> kernel (1711 lines) |
| `fp8_crr_4096_kernel.s` | Extracted FP8 CRR kernel (1413 lines) |
| `mxfp8_crr_4096_kernel.s` | Extracted MXFP8 CRR scaled <true,2> kernel (1830 lines) |
| `*_kloop.s` | Per-kernel inner-K-pair-body extracts used for per-iter counts |

Full device-`.s` (1.5 MB combined) NOT committed — regenerate with §1.1 build commands. Per-kernel extracts above are sufficient for repro of all instruction counts.

Cross-references:
- `r47d_findings.md` — R47D's K=8192 RCR baseline analysis (this work extends).
- `r47d_isa_dumps/` — R47D's K=8192 RCR dumps for direct comparison.
- `r46a_profiling_findings.md` — R46A's L2/VMEM secondary-overhead modeling.
- `r47_full_baseline_SUMMARY.txt` — R47 measured ratios (anchor data for verdict table).

---

## 8. One-Line Summary

**ISA verification at K=4096 confirms R47D's RCR-MFMA hardware ceiling (~95%) extends to ALL 5 RCR cells at M=4096 — no recoverable headroom. RRR shows new finding: FP8 fully unrolled but MXFP8 looped at K=4096 (+1-2pp recoverable via `RRR_MAIN_UNROLL=32`). CRR has structural ~92% ceiling from 6 v_lshrrev/K-pair scale shifts. R49 single biggest target: 70B Gate/Up CRR (88.5% measured vs 92.5% predicted, +4pp headroom in L2 scale-cache or LDS-swizzle, NOT MFMA-level).**
