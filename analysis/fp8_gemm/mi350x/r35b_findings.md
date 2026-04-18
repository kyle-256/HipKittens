# R35 Dev B — HB shrink (BLK_M=128, HB_M=64) V2-CRR Stage A1-A5

**Date:** 2026-04-18
**Branch:** r35-b (base feat/mxfp8-only @ 758ad933, R34 cycle wrap)
**GPU:** HIP_VISIBLE_DEVICES=1 (physical GPU 1, MI355X / gfx950)
**Scope:** Stage A1-A5 of the HB shrink (per-warp M-coverage shrink) parallel template, mirroring R31 Dev A's rect-V2 CRR scaffolding pattern. Pivots from R34 Dev D's REFUTED sub-RBM hypothesis to the R34 Dev D §3.2 Option 1 alternative ("halve HB → BLK_M=128").

---

## TL;DR

- **Stage A1 (DONE):** New file `crr_mxfp8_exact_8wave_hbshrink_fastpath.inc` (450 LOC) + macro selector `MXFP8_CRR_BLK_M` (default 256). When `-DMXFP8_CRR_BLK_M=128`, a parallel template instantiates `crr_exact_8wave_scaled_hbshrink_kernel` covering HBSHRINK_BLK_M=128 × BLK_N=256 × BK=128 (1 M-half-tile per block).
- **Stage A2 (DONE):** Type bridge translation. Each warp now has 1 M-stride (HB_M=64 = single RBM=64 stride) instead of 2 (HB=128 = 2 strides). Drops cC/cD accumulators (only cA/cB remain). Uses single-buffered LDS schedule for skeleton simplicity.
- **Stage A3 (DONE):** Build sanity. Default 8192³ + 70B Gate byte-identical to base. Probe build (`-DMXFP8_CRR_BLK_M=128 -DMXFP8_CRR_HBSHRINK_PROBE=1`) compiles cleanly. **Resource report empirically validates the structural hypothesis:** VGPR/wave **234 → 168** (-66 saved, target was -64), no spills, occupancy unchanged at 2 waves/SIMD, LDS 139264 → 104448 B/block (-34816).
- **Stage A4 (DONE — STRETCH ACHIEVED):** Numerics on 70B KV (4096×1024×8192) — SNR 49.60 dB (≥ 48 dB target), pass_rate 100%, det 3/3 PASS. **Bit-equal to default V2-CRR across the full output (max abs diff = 0.0)** — exceeds spec which only required C[0,:8] bit-compare.
- **Stage A5 (DONE — NO SHIP):** BABA paired bench on 70B KV (10 paired samples, 30s preheat, sclk=2393 MHz steady throughout). Default V2-CRR median 768.79 TFLOPS, HB shrink median 675.28 TFLOPS, **delta -12.16%**, Welch t = -71.3 (highly significant). **NO SHIP** (would require ≥ +1.0%).

**Structural finding (POSITIVE):** R34 Dev D §3.2 Option 1 hypothesis CONFIRMED — per-warp M-coverage shrink actually frees VGPR (unlike sub-RBM which doubled accumulator count). Accumulator footprint dropped from 4 × 32 = 128 VGPR/wave to 2 × 32 = 64 VGPR/wave as predicted. Total VGPR drop measured at -66, very close to the predicted -64.

**Perf finding (NEGATIVE):** The single-buffered skeleton form of HB shrink is -12.2% slower than the production DB main loop with full SB pipelining + B1 LDS interleave. This is **expected** for a Stage A2 skeleton — the comparison is the simple SB form vs. a heavily optimized production kernel. The R34 Dev D hypothesis was that HB shrink would unblock SB pipelining (PIPE=3) by freeing accumulator VGPR; that hypothesis remains UNTESTED in this cycle (we did not wire SB pipelining variants into the HB shrink kernel — that's Stage A6+ work).

---

## 1. Stage A1 — Scaffolding parallel template

### 1.1 Macro selector

Added to `crr_mxfp8_exact_8wave_hbshrink_fastpath.inc`:

```cpp
#ifndef MXFP8_CRR_BLK_M
#define MXFP8_CRR_BLK_M 256
#endif
#if (MXFP8_CRR_BLK_M != 128) && (MXFP8_CRR_BLK_M != 256)
#error "MXFP8_CRR_BLK_M must be 128 or 256 (R35 Dev B scaffolding)"
#endif

#if (MXFP8_CRR_BLK_M == 128)
... // HB shrink kernel definitions
#endif
```

Default 256 → no kernel emitted, default build byte-identical to head. Independent of `MXFP8_CRR_RBM` (R33 Dev D sub-RBM macro) and `MXFP8_RECT_BLK_N` (R28 Dev D rect-V2 macro) — orthogonal progressive macros.

### 1.2 Include hook

Added to `kernel_mxfp8_layouts.cpp` after the existing sub-RBM include:

```cpp
// R35 Dev B — Stage A1: HB shrink (BLK_M=128, HB_M=64) V2-CRR fastpath
#include "crr_mxfp8_exact_8wave_hbshrink_fastpath.inc"
```

### 1.3 Dispatch wire-in

Added to the V2-CRR dispatch chain (gemm_crr_pq_v2 path) as a guarded branch BEFORE the default v2 dispatch:

```cpp
#if defined(MXFP8_CRR_BLK_M) && (MXFP8_CRR_BLK_M == 128)
        if (crr_can_use_exact_8wave_scaled_hbshrink(g)) {
            dispatch_crr_exact_8wave_scaled_v2_hbshrink<true>(g);
            return;
        }
#endif
```

This is a single edit; the default build compiles to byte-identical output because the `#if` evaluates false when `MXFP8_CRR_BLK_M` is undefined or equals 256.

---

## 2. Stage A2 — Kernel translation

### 2.1 Structural changes vs. default V2-CRR fastpath

| Element | Default V2-CRR | HB shrink (this cycle) |
|---|---|---|
| BLK_M | 256 | **128** (halved) |
| HB_M (per-warp M coverage) | 128 | **64** (halved) |
| Per-warp M-strides | 2 (top+bot half) | **1** (single stride per warp) |
| Accumulator vectors | 4: cA, cB, cC, cD | **2: cA, cB only** (no cC/cD) |
| Accumulator VGPR/wave | 4 × 32 = 128 | **2 × 32 = 64** (-64) |
| Operand A reg/wave | 32 (A_col_reg) | 32 (A_col_reg, unchanged) |
| Operand B regs/wave | 16+16 (b0, b1) | 16+16 (unchanged) |
| LDS A slots | As[2][2] (2 M-halves × 2 buffers) | **As[2]** (1 M-half × 2 buffers) |
| LDS B slots | Bs[2][2] (unchanged) | Bs[2][2] (unchanged) |
| Grid dimension | (M/BLK) × (N/BLK) | **(M/HBSHRINK_BLK_M) × (N/BLK)** = 2× more blocks in M |

### 2.2 Coordinate scheme

The HB shrink kernel mirrors the rect-V2 Path 1 trick: **keep the host preshuffle unchanged** by mapping `br` (HB-shrunk M-tile id) to `(br_orig=br>>1, mhalf=br&1)`, then using `br_orig` for the scale slab index and `mhalf` to select which M-half of the original square BLK-block we're processing.

```cpp
const int br_orig = br >> 1;     // square BLK-block id
const int mhalf   = br & 1;      // 0 = top half, 1 = bottom half

auto crr_scale_a_base = [&](int local_half) {
    (void)local_half;            // always 0 in HB shrink (single M-half per block)
    return br_orig * BLK + mhalf * (BLK / 2) + wm * RBM;
};

// Scale slab idx uses br_orig (NOT br) — host preshuffle unchanged.
const size_t slab_idx_a = static_cast<size_t>(br_orig) * WARPS_M + wm;
```

The A-side LDS load uses `(br_orig * 2 + mhalf)` as the global tile-row coord, picking up either the top-half (M-rows 0..127 of the original BLK-block) or the bottom half (rows 128..255) depending on mhalf. The square A LDS tile is 128 rows wide, so a single load covers exactly one M-half.

### 2.3 Scale-pack mhalf selection (parallel to rect-V2's nhalf)

The compiler can't constant-fold `mhalf` (it's a runtime block coord), so we build a per-iter `a_sel_packs` array selecting `a0_scale_packs` (mhalf=0) or `a1_scale_packs` (mhalf=1):

```cpp
fp8e8m0_4 a_sel_packs[crr_a_pack_count];
#pragma unroll
for (int p = 0; p < crr_a_pack_count; p++) {
    a_sel_packs[p] = (mhalf == 0) ? a0_scale_packs[p] : a1_scale_packs[p];
}
```

This is the symmetric pattern to the rect-V2 Path 1 `b_sel_packs` selector for nhalf.

### 2.4 LDS schedule

Single-buffered (mirrors SUBRBM Stage 2b skeleton). Each iter:
1. Load `b0` from `Bs[0][0]`, `b1` from `Bs[0][1]`, `a` from `As[0]`
2. `s_waitcnt lgkmcnt(0)`
3. MMA chain: `cA(a, b0)` then `cB(a, b1)` (both via `crr_mma_scaled_from_packs_fixed_phase`)
4. `s_barrier`
5. Issue next-iter VMEM->LDS for k+1
6. `s_waitcnt vmcnt(0)`, `s_barrier`

The full DB main loop with B1 LDS interleave + SB pipelining variants is deferred to R36+ (this cycle's scope was Stage A1+A2 scaffolding with stretch goals A4+A5).

---

## 3. Stage A3 — Build sanity + resource report

### 3.1 Default build byte-identity

| Shape | Pre-change md5 | Post-change md5 | Match |
|---|---|---|---|
| 8192³ | `5fd2d6745bed0071a0b9b5ddb3a7f6db` | `5fd2d6745bed0071a0b9b5ddb3a7f6db` | YES |
| 70B Gate (4096×28672×8192) | `609e390cdecfc1d74d01ad5b7a52c524` | `609e390cdecfc1d74d01ad5b7a52c524` | YES |

(Both verified by `git stash` round-trip — see logs.)

### 3.2 Probe build resource report (8192³, MXFP8_CRR_BLK_M=128, MXFP8_CRR_HBSHRINK_PROBE=1)

Build: rc=0, no errors. Both kernels co-emit; the default `crr_exact_8wave_scaled_kernel` is unchanged (still emits with VGPR=234), and the new `crr_exact_8wave_scaled_hbshrink_kernel` appears with the following resource numbers:

| Kernel | TotalSGPRs | VGPRs | AGPRs | SGPRs Spill | VGPRs Spill | LDS B/block | Occupancy |
|---|---|---|---|---|---|---|---|
| `crr_exact_8wave_scaled_kernel` (default V2-CRR baseline) | 52 | **234** | 0 | 0 | 0 | 139264 | 2 waves/SIMD |
| `crr_exact_8wave_scaled_hbshrink_kernel` (HB shrink) | 45 | **168** | 0 | 0 | **0** | **104448** | **2 waves/SIMD** |
| Δ (HB shrink − default) | -7 | **-66** | 0 | 0 | 0 | -34816 | unchanged |

**This empirically validates the R34 Dev D §3.2 Option 1 hypothesis.** The accumulator VGPR savings predicted in the task spec (-64 = 128 → 64 from dropping cC/cD) is observed at -66 net VGPR (slightly more than predicted because supporting state — coordinate registers, scale-pack registers — also shrinks slightly). VGPR target ≤ 200 ✓, no spills ✓, occ ≥ 2 ✓.

**Contrast to R34 Dev D sub-RBM result:** sub-RBM emitted 256 VGPRs (saturated) + 476 VGPR spill. HB shrink emits 168 VGPRs + 0 spill — **structurally opposite** outcomes, confirming that the lever is per-warp M-coverage, not operand-tile size.

### 3.3 LDS budget breakdown

| Region | Default | HB shrink | Δ |
|---|---|---|---|
| As[2][2] (2 buffers × 2 M-halves × 17408 B/tile) | 69632 | — | — |
| As[2] (2 buffers × 1 M-half × 17408 B/tile) | — | 34816 | — |
| Bs[2][2] (2 buffers × 2 N-halves × 17408 B/tile) | 69632 | 69632 | 0 |
| **Total** | **139264** | **104448** | **-34816 (-25%)** |

The LDS drop matches the dropped A-side slot exactly (one less M-half). Occupancy stays at 2 waves/SIMD because the smaller VGPR (168 < 256) is the binding constraint, not LDS.

---

## 4. Stage A4 — Numerics

### 4.1 70B KV (4096×1024×8192)

Built with `-DM_DIM=4096 -DN_DIM=1024 -DK_DIM=8192 -DMXFP8_CRR_BLK_M=128`. Single .so dispatched via `gemm_crr_pq_v2` (which now routes to the HB shrink kernel because the dispatch wire-in fires when `MXFP8_CRR_BLK_M=128`).

| Metric | Value | Threshold | Verdict |
|---|---|---|---|
| SNR (dB) | 49.60 | ≥ 48 dB | PASS |
| Pass rate (10% rel-tol or abs ≤ 3.0) | 100.00% | 100% | PASS |
| Determinism (3 runs bit-equal) | OK | OK | PASS |

Log: `r35b_70b_kv_hbshrink_validate.log`.

### 4.2 Bit-compare against default V2-CRR

Built two .so's (default and HB shrink) with distinct `PY_MODULE_NAME`, dispatched same inputs through `gemm_crr_pq_v2` for each. Output:

```
C_default[0,:8]   = [2.1875, -1.4765625, 0.37109375, -1.0703125, -0.318359375, 0.06787109375, -0.8359375, 0.267578125]
C_hbshrink[0,:8]  = [2.1875, -1.4765625, 0.37109375, -1.0703125, -0.318359375, 0.06787109375, -0.8359375, 0.267578125]
bit_equal_C[0,:8] = True
max_abs_diff      = 0.0          (whole output, M×N elements)
mean_abs_diff     = 0.0
```

**HB shrink and default V2-CRR produce bit-identical output across the entire 4096×1024 output.** This is stronger than the spec required (which asked for C[0,:8] bit-compare).

Log: `r35b_70b_kv_bitcompare.log`. Script: `r35b_bitcompare.py`.

---

## 5. Stage A5 — Perf bench (BABA paired)

### 5.1 Setup

- Two .so built with distinct `PY_MODULE_NAME`:
  - `tk_mxfp8_kv_default.so` — default V2-CRR (md5 `e807d8b7580bff7c3bc7bd7123bf3224`)
  - `tk_mxfp8_kv_hbshrink.so` — HB shrink active (md5 `3c702ab525d07de8f42f66248fb0193f`)
- Both dispatched via `gemm_crr_pq_v2`. The HB shrink .so routes through the new dispatch path.
- Harness: `r34c_paired_bench.py` (R34 Dev C BABA). PHYS_GPU=1, N_PAIRS=5 (10 BABA samples), 30s preheat, 2 warmup pairs.
- HIP_VISIBLE_DEVICES=1 → physical GPU 1.

### 5.2 sclk policy compliance

| Phase | sclk | Verdict |
|---|---|---|
| pre-preheat | 2395 MHz | OK |
| post-preheat (R34 rule: ≥ 2200 MHz) | **2347 MHz** | OK (≥ 2200) |
| pre-bench | 2393 MHz | OK |
| post-bench | 2393 MHz | OK |

No retry needed.

### 5.3 Results (4096×1024×8192, 70B KV)

| Variant | Median (TFLOPS) | Mean (TFLOPS) | Stdev | n |
|---|---|---|---|---|
| Default V2-CRR | **768.79** | 768.20 | 3.88 | 10 |
| HB shrink (this cycle) | **675.28** | 675.06 | 1.43 | 10 |
| **Delta (median)** | — | — | — | **-12.16%** |
| Welch t (default vs hbshrink) | -71.298 | — | — | — |

**SHIP gate fails decisively** (would require HB shrink ≥ default × 1.01, observed 0.878×). Welch t = -71.3 indicates extreme statistical significance of the regression.

Log: `r35b_70b_kv_baba_bench.log`.

### 5.4 Why the perf is worse despite the VGPR savings

The HB shrink skeleton uses the **single-buffered LDS schedule** (load → barrier → compute → barrier → next-iter VMEM → drain → barrier). This is the simplest correct schedule and was chosen to minimize Stage A2 translation work. The default V2-CRR uses the **DB main loop with all the R31/R32 pipelining tricks** (B1 LDS interleave with `INSERT_AFTER=4`, fused MMA chain, `s_setprio` priority hints, fine-grained `s_waitcnt`/`s_barrier` placement, etc.).

The fair comparison would be: HB shrink + DB main loop + SB pipelining (PIPE=2 or 3) **vs.** default V2-CRR. The R34 Dev D §3.2 hypothesis was that HB shrink frees enough VGPR to allow PIPE=3 (which previously hit a -15% structural ceiling because A_col_reg + accumulator + a_next exceeded 256-VGPR cap). With accumulator dropping 128 → 64, there's now ~64 VGPR of headroom, which **should** unblock PIPE=3.

**That hypothesis remains UNTESTED in this cycle.** The single-buffered skeleton's -12.2% perf is NOT a refutation of the HB shrink lever; it's a baseline measurement showing that pure macro-level work-per-block doubling without any pipelining recovery loses ~12%. The PIPE=2/3 variants need to be ported to the HB shrink kernel before the lever can be evaluated for SHIP-worthiness.

### 5.5 Where the regression comes from

Possible factors:
1. **Doubled grid:** 2× more blocks (each smaller) → 2× kernel launch overhead amortized, more wave-fill slack at end-of-grid.
2. **Reduced LDS reuse:** each warp only does 1 MMA pass per K-iter on the loaded `a` register (vs. 2 in default). The single `a` load amortizes over fewer FLOPs.
3. **No DB pipelining:** the single-buffered form doesn't overlap LDS reads with MMA — the production kernel does extensive overlap.
4. **No B1 LDS interleave:** default V2-CRR interleaves the b1 LDS load INTO the cA MMA chain (`crr_exact_cA_with_b1_interleave_fixed_phase`); HB shrink loads both b0 and b1 upfront with no interleave.

Items 3-4 are the ones HB shrink could fix in R36+ work. Item 2 is fundamental to the BLK_M=128 geometry (each warp has half the work per K-iter). Item 1 is also fundamental but small.

---

## 6. Files modified

| File | Change | LOC delta |
|---|---|---|
| `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_hbshrink_fastpath.inc` | NEW — full Stage A1+A2 scaffolding (macro selector + parallel kernel template + dispatch entry + probe hook) | +432 (NEW) |
| `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp` | Added `#include` for hbshrink fastpath (4 LOC) + dispatch wire-in inside V2-CRR path (8 LOC, guarded by `MXFP8_CRR_BLK_M=128`) | +12 |
| `analysis/fp8_gemm/mi350x/r35b_findings.md` | This document | NEW |
| `analysis/fp8_gemm/mi350x/r35b_bitcompare.py` | Bit-compare harness (Stage A4) | NEW |

Build/bench logs (in `analysis/fp8_gemm/mi350x/r35b_*`):
- `r35b_baseline_default_build.log` — pre-change 8192³ default (rc=0, md5 `5fd2d6745bed0071a0b9b5ddb3a7f6db`)
- `r35b_default_postchange_build.log` — post-Stage-A1 8192³ default (rc=0, md5 matches pre — byte-identical)
- `r35b_default_postwirein_build.log` — post-dispatch-wire-in 8192³ default (rc=0, md5 still matches — byte-identical)
- `r35b_70b_gate_baseline_build.log` — pre-change 70B Gate default (rc=0, md5 `609e390cdecfc1d74d01ad5b7a52c524`)
- `r35b_70b_gate_default_build.log` — post-change 70B Gate default (md5 matches — byte-identical)
- `r35b_hbshrink_probe_build.log` — Stage A3 probe build with HB shrink kernel (rc=0; emits `_Z38crr_exact_8wave_scaled_hbshrink_kernelILb1ELi2EEv...` symbol with 168 VGPRs, 0 spill, occ=2)
- `r35b_70b_kv_hbshrink_build.log` — 70B KV probe build resource report
- `r35b_70b_kv_default_build.log` — 70B KV default build (for BABA pair)
- `r35b_70b_kv_hbshrink_active_build.log` — 70B KV with HB shrink active
- `r35b_70b_kv_hbshrink_validate.log` — Stage A4 numerics: SNR 49.60 dB, det 3/3 PASS
- `r35b_70b_kv_bitcompare.log` — Bit-compare: max abs diff 0.0
- `r35b_70b_kv_baba_bench.log` — Stage A5 BABA-paired bench: -12.16% (NO SHIP)

---

## 7. Hypothesis status

| Claim | R34 Dev D status | R35 Dev B status (this cycle) | Evidence |
|---|---|---|---|
| Sub-RBM (RBM=32) frees accumulator VGPR | REFUTED (Stage 2b: 256 VGPR + 476 spill) | Unchanged — REFUTED | r34d_findings.md §3.1 |
| HB shrink (BLK_M=128) frees accumulator VGPR | Predicted (§3.2 Option 1, untested) | **CONFIRMED** | This cycle Stage A3: 234 → 168 VGPR, 0 spill |
| HB shrink unblocks SB pipelining → SHIP-worthy perf gain | Predicted | **UNTESTED** (need SB pipelining variants ported to HB shrink kernel — R36+ work) | This cycle skeleton uses single-buffered form; -12.2% on simple-vs-optimized comparison |
| HB shrink kernel produces correct numerics | — | **CONFIRMED** | This cycle Stage A4: SNR 49.60 dB, bit-equal to default across full 4096×1024 output |

---

## 8. Methodology rule compliance

| Rule | Compliance |
|---|---|
| `rm -f tk_mxfp8_layouts*.so` + per-build md5 (R29 Dev C) | YES — every build, md5s logged in §3 + §5 |
| `rocm-smi -d 1` for physical GPU 1 (R31 Reviewer) | YES — sclk readings throughout bench (§5.2) |
| In-process A/B with BABA + 30s preheat (R31 Dev D) | YES — used `r34c_paired_bench.py` with `PREHEAT_S=30 WARMUP_PAIRS=2 N_PAIRS=5` (10 paired samples) |
| sclk-post-preheat ≥ 2200 MHz (R34 NEW rule, auto-retry 3x) | YES — 2347 MHz (≥ 2200, no retry needed) |
| Closed-lever check | YES — sub-RBM (R34 Dev D REFUTED) was last cycle's lever; this cycle pivots to R34 Dev D §3.2 Option 1 (HB shrink) which was the explicitly named next-direction |
| Default build byte-identity on 2 shapes (8192³ + 70B) | YES — md5 `5fd2d6745bed0071a0b9b5ddb3a7f6db` and `609e390cdecfc1d74d01ad5b7a52c524` respectively, both unchanged from pre-change reference (latter cross-checked against base via `git stash`) |
| SHIP-claim normalization | N/A — no SHIP claim (negative perf finding on the simple skeleton form; see §5.4 for why the lever still has unevaluated potential) |
| All progress macros default-off | YES — `MXFP8_CRR_BLK_M` defaults to 256; `MXFP8_CRR_HBSHRINK_PROBE` defaults to 0; `MXFP8_CRR_HBSHRINK_8WAVE_FAST_ENABLE` only meaningful when `MXFP8_CRR_BLK_M=128` |

No closed lever was prototyped. The HB shrink scaffolding is R34 Dev D's named §3.2 Option 1 next-direction; this cycle delivers the scaffolding + Stage A4 correctness + Stage A5 baseline perf, leaving the SB pipelining test as the unrealized potential.

---

## 9. Suggested R36+ work

1. **Port SB pipelining variants** (PIPE=1, 2, 3 from R32 Dev B) to the HB shrink kernel. The accumulator VGPR drop from 128 → 64 should give ~64 VGPR of headroom for the `a_next` register that PIPE=2/3 require. Predicted: PIPE=3 may now stay under the 256-VGPR cap, unblocking the -15% structural ceiling R32 Dev B identified.
2. **Port the DB main loop** (with B1 LDS interleave, batched MMA pairs, all the R28-R32 pipelining tricks) from `crr_mxfp8_exact_8wave_fastpath.inc` to the HB shrink kernel. The single-buffered skeleton in this cycle is intentionally minimal; the production kernel uses ~500 LOC of careful pipelining that needs separate translation. Until this is done, the skeleton-vs-production -12.2% comparison underestimates the lever's potential.
3. **70B Gate (4096×28672×8192)** — current cycle did NOT bench HB shrink on 70B Gate (this is the V2-CRR cell where cachepolicy=2 wins per R28 Dev A). Worth checking whether HB shrink helps OR hurts there. Note however: per the R33 Dev A wire-in advisory printed at runtime, all the LLaMA cells where V2-CRR is the baseline are in fact also covered by V2-RRR autotune fan-out as faster alternatives — so HB shrink-vs-V2-CRR may not be the critical comparison; HB shrink-vs-V2-RRR-current-best is the cell-level question.
4. **Optional cleanup:** delete the sub-RBM kernel skeleton in `crr_mxfp8_exact_8wave_subrbm_fastpath.inc` once the HB shrink lever is fully evaluated. Keeping it preserves the R33/R34 negative finding for reproducibility but isn't structurally needed for the HB shrink path.

---

## 10. Summary

| Metric | Value |
|---|---|
| Verdict | **Stages A1-A5 DELIVERED** (no SHIP — negative perf on simple skeleton form, but with KEY POSITIVE structural finding) |
| Stage A1 (scaffolding) | DONE — parallel template (432 LOC) + macro selector + dispatch wire-in (12 LOC) |
| Stage A2 (type bridge translation) | DONE — single-buffered LDS schedule, 2 accumulators (cA/cB), HB_M=64 per-warp coverage |
| Stage A3 (build sanity + resource report) | DONE — default build byte-identical on 2 shapes; probe build VGPR 168 (-66), 0 spill, occ=2 |
| Stage A4 (numerics on 70B KV) | DONE — SNR 49.60 dB, bit-equal to default across full output, det 3/3 PASS |
| Stage A5 (perf bench BABA on 70B KV) | DONE — -12.16% vs default (NO SHIP); single-buffered skeleton vs production DB main loop |
| Default build byte-identity | PASS on 8192³ (md5 `5fd2d6745bed0071a0b9b5ddb3a7f6db`) AND 70B Gate (md5 `609e390cdecfc1d74d01ad5b7a52c524`) |
| Paradigm confirmations | **1 confirmation** — R34 Dev D §3.2 Option 1 hypothesis CONFIRMED (HB shrink frees accumulator VGPR, opposite of the REFUTED sub-RBM lever) |
| Time spent | ~75 min (within 30-90 min cycle budget) |
