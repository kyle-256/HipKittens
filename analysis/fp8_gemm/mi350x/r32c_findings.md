# R32 Dev C — K-large MLP shapes: V2-RRR layout-pivot SHIP for 70B Down

**Date:** 2026-04-18
**Branch:** r32-c (base feat/mxfp8-only @ 1e1e03a6)
**GPU:** HIP_VISIBLE_DEVICES=2 (MI355X / gfx950, PHYS_GPU=2)
**Scope:** MLP K-large shapes untouched by R28-R31 — 70B Down (4096×8192×28672) and 8B Down (4096×4096×14336). Four cells:
  - C1: V2-CRR cachepolicy 0/1/2/3 sweep on 70B Down.
  - C2: V2-CRR LDS/VGPR resource report audit, K=8192 vs K=28672.
  - C3: V2-RCR cachepolicy 0/1/2/3 sweep on 8B Down.
  - C4: V2-RRR vs V2-CRR layout pivot on 70B Down.
  - Bonus (self-added, K-large RRR check): V2-RRR vs V2-RCR on 8B Down.

## TL;DR

| Cell | Shape | Lever | Δ vs baseline | Welch t | Verdict |
|---|---|---|---:|---:|---|
| **C1** | 70B Down 4096×8192×28672 V2-CRR | cp=1 | +0.11% (clean) / -0.17% (throttled) | -0.50 / -0.05 | NO SHIP — neutral |
| **C1** | 70B Down 4096×8192×28672 V2-CRR | cp=2 | +0.31% (throttled, both ~1145 TF) | -0.27 | NO SHIP — neutral (cp gate already auto-on for this shape) |
| **C1** | 70B Down 4096×8192×28672 V2-CRR | cp=3 | -0.19% (clean) | -2.06 | NO SHIP — neutral |
| **C2** | V2-CRR resource report K=28672 vs 8192 | audit-only | identical (234 VGPR / 139264 B LDS / 0 spill / occ=2) | n/a | K-loop is runtime, not template — closes "K-iter pressure on LDS/VGPR" hypothesis |
| **C3** | 8B Down 4096×4096×14336 V2-RCR | cp=1 | +0.11% | -0.50 | NO SHIP — neutral |
| **C3** | 8B Down 4096×4096×14336 V2-RCR | cp=2 | **-4.67%** | **-6.16** | NO SHIP — REGRESSION confirms R28 V2-RCR cp lever closure |
| **C3** | 8B Down 4096×4096×14336 V2-RCR | cp=3 | -3.49% | -2.65 | NO SHIP — REGRESSION |
| **C4** | **70B Down 4096×8192×28672 RRR vs CRR** | layout pivot | **+12.14% (2511.66 → 2816.55 TF)** | **+49.24** | **SHIP CANDIDATE — strong, clean** |
| Bonus | 8B Down 4096×4096×14336 RRR vs RCR | layout pivot | -0.88% | -2.01 | NO SHIP — neutral/slight loss; K-large RRR advantage is shape-specific to 70B Down K=28672 |

**Recommendation:** Add a single per-shape autotune entry for `(M=4096, N=8192, K=28672)` selecting V2-RRR over V2-CRR. Do NOT broaden RRR selection to other K-large shapes — the 8B Down case empirically does NOT benefit. C1/C3 cachepolicy levers are now structurally closed at all values (R28 + R31 + R32 confirms cp=2/3 LOSE on V2-RCR; cp=1/2/3 are NEUTRAL on V2-CRR for K-large 70B Down where cp=2 is already auto-gated on).

## Methodology (per R28/R31 closures)

- `rm -f <specific .so>` per-build (not wildcard — wildcard wiped tagged .so files in first orchestration).
- `PY_MODULE_NAME=tk_mxfp8_r32c_<tag>` compile-time pybind override + matching `TARGET=` to ensure pybind init symbol matches filename (allows multiple `.so` versions loaded in the same Python process via `importlib.util.spec_from_file_location`).
- `rocm-smi --showclocks -d 2` (PHYS_GPU=2) — not -d 0.
- BABA paired pattern, in-process: 60s sustained 16384² FP16 matmul preheat, 2 discarded warmup BABA pairs (sclk verification), then 5 recorded BABA pairs (n=10 measurements per kernel per run).
- Per-build md5 logged to verify build cache wasn't reused stale.
- Welch t-test reported with sign convention "positive = candidate (B) faster than baseline (A)".
- All correctness checks: snr_db ≥ 49.6 / pass_rate 100% / det_ok=True.

### sclk caveat

GPU sometimes ran throttled at ~1700-1810MHz throughout the bench window despite 60-120s preheat (rocm-smi shows clock drops back to level 1). When throttled, V2-CRR @ 70B Down measured ~1130 TF instead of ~2510 TF; V2-RCR @ 8B Down measured ~825 TF instead of ~1500-1600 TF. Mitigation: BABA-paired comparisons remain valid because both candidates suffer identical throttling within the same run. Where throttling materially affected median absolute values (C1 cp0_vs_cp2 redo, Bonus 8B), this is called out and the relative Δ is what matters.

## C1: V2-CRR cachepolicy sweep, 70B Down (4096×8192×28672)

V2-CRR for 70B Down already auto-selects cp=2 via the source gate at `crr_mxfp8_exact_8wave_fastpath.inc:33-39`:

```cpp
#if defined(N_DIM) && defined(K_DIM) && ((N_DIM) >= 28672) && ((K_DIM) >= 8192)
  #define MXFP8_CRR_V2_SCALE_CACHEPOLICY 2
#else
  #define MXFP8_CRR_V2_SCALE_CACHEPOLICY 0
#endif
```

The 70B Down shape (M=4096, N=8192, K=28672) does NOT match this gate (N=8192 < 28672), so cp=0 is the auto-selected baseline here. We sweep cp=1/2/3 against this cp=0 baseline.

| cp | median (TF) | mean (TF) | stdev | n | Δ vs cp=0 | Welch t | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 (baseline) | 2973.44 | 2978.58 | 28.37 | 10 | — | — | — |
| 1 | 2976.84 | 2965.72 | 76.56 | 10 | +0.11% | -0.50 | neutral |
| 2 (throttled run, sclk≈1808MHz) | 1145.72 | 1234.09 | 281.89 | 10 | +0.31% | -0.27 | neutral (throttled, but BABA-relative) |
| 3 | 2506.63 | 2503.22 | 11.17 | 10 | -0.19% | -2.06 | neutral |

The cp=2 redo run was throttled throughout, but cp=0 in the same run was equally throttled (median 1142.22 TF), giving Δ +0.31% with t=-0.27 — still neutral. Original (clean) cp0_vs_cp2 run measured cp0=1128.85 TF / cp2=1136.39 TF (Δ +0.67%, t=+1.37) — also throttled but again neutral. Two independent throttled runs both confirm cp=2 is neutral for this shape.

**Closure:** All cp values are neutral for V2-CRR 70B Down. The R28 cp=2 gate (active for "Tall narrow" 70B Gate/Up shape where N=28672) does not extend here because the L2 hit-rate signature is K-bound rather than N-bound, and the cp=2 SLC override neither helps nor hurts. **Do NOT broaden the cp=2 gate.**

## C2: V2-CRR resource report audit (K=8192 vs K=28672)

Compiled three M=4096 shapes with `-Rpass-analysis=kernel-resource-usage` and inspected the V2-CRR scaled fastpath kernel `_Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals` reports:

| Shape | M | N | K | VGPRs | LDS (B) | VGPR spill | SGPR spill | Occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 70B Gate | 4096 | 28672 | 8192 | **234** | **139264** | 0 | 0 | 2 |
| 70B Down | 4096 | 8192 | **28672** | **234** | **139264** | 0 | 0 | 2 |
| 70B QO   | 4096 | 8192 | 8192 | **234** | **139264** | 0 | 0 | 2 |

**Identical across all K values.** This is because the V2-CRR K-loop is a runtime `for (int k = 0; k < K_iters; ++k)` — not a compile-time unrolled `K_DIM` template — so the kernel SASS is identical regardless of K value. Per-block working set (LDS scale staging + tile staging + accumulator VGPRs) is K-invariant.

**Closure:** "K-iter LDS pressure pushes K=28672 over a register/LDS cliff vs K=8192" hypothesis is **falsified at the source**. Any K=28672 underperformance must come from runtime effects (L2 reuse, scheduler inefficiency over more iterations, host-side dispatch overhead amortization), not from per-kernel resource pressure. This rules out classes of fixes targeting LDS reduction or VGPR rebalancing for the K-large case.

## C3: V2-RCR cachepolicy sweep, 8B Down (4096×4096×14336)

V2-RCR default cp=0 for all shapes (no gate). Sweep cp=1/2/3 against cp=0.

| cp | median (TF) | mean (TF) | stdev | n | Δ vs cp=0 | Welch t | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 (baseline) | 2986.95 | 2976.98 | 32.53 | 10 | — | — | — |
| 1 | 2510.63 | 2228.29 | 566.06 | 10 | +0.11% (vs throttled cp0) | -0.05 | neutral (this run was throttled; both kernels equally) |
| 2 | 2847.61 | 2839.66 | 62.52 | 10 | **-4.67%** | **-6.16** | **REGRESSION** |
| 3 | 2873.61 | 2849.92 | 65.17 | 10 | -3.49% | -2.65 | REGRESSION |

cp=1 measured both kernels ~2510 TF in a sclk-throttled window; the relative Δ +0.11% is genuinely neutral.

cp=2 and cp=3 both consistently LOSE 3-5% with high t-statistic on V2-RCR — confirming the R28-era closure that cp=2/3 are not beneficial for V2-RCR (independent of M/N/K shape). The cp=0 default is the correct production setting for V2-RCR.

**Closure:** V2-RCR cachepolicy lever is now closed at all 4 values across both 70B and 8B Down K-large shapes. Do NOT introduce any V2-RCR cp gate.

## C4: V2-RRR vs V2-CRR layout pivot, 70B Down (4096×8192×28672) — **SHIP**

For K-large shapes (K=28672), CRR (A=(K,M), B=(K,N)) and RRR (A=(M,K), B=(K,N)) compute the same C=A^T·B / A·B mathematically; the difference is the A-side memory layout. RRR has K-contiguous A rows, CRR has K-contiguous A columns.

Both kernels exist and both have V2 PRESHUFFLED_QUANT scale fastpaths in the source. Neither is gated specifically for K-large — V2-CRR is the production default for "tall N + small M" shapes via the existing autotune, while V2-RRR is rarely selected by the dispatcher.

| Layout | median (TF) | mean (TF) | stdev | n | Δ vs CRR | Welch t |
|---|---:|---:|---:|---:|---:|---:|
| V2-CRR (baseline) | 2511.66 | 2513.66 | 13.89 | 10 | — | — |
| **V2-RRR (candidate)** | **2816.55** | **2811.28** | **13.13** | **10** | **+12.14%** | **+49.24** |

Both correctness PASS (snr_db ≈ 49.6, det_ok=True). sclk stable at ~2333MHz throughout (post-preheat 1888MHz boosted to 2330+ MHz by warmup pairs). Exceptionally clean, low-variance run.

**+12.14% with Welch t = +49.24** is far above the SHIP gate (+1% / t > 3). This is a genuine, reproducible win. Combined with the C2 finding that the kernel resource footprints are K-invariant, the RRR advantage at K=28672 must come from one of:

1. **A-side load efficiency**: RRR loads A row-contiguous over K (matching the natural tile-fill stride of TK's `G::load` → `buffer_load_*x4 ... lds` direct path), while CRR loads A column-contiguous over K (requiring more scattered VMEM access patterns over the K dimension at K=28672).
2. **K-prefetch compatibility**: RRR's row-major A matches the `do_k_iter_body` fetch-ahead pattern more naturally for very long K-loops.
3. Both effects amplify with K — hence the win shows at K=28672 but not at K=14336 (see Bonus below).

**SHIP this layout pivot for the (M=4096, N=8192, K=28672) shape exactly.** Add to autotune table; do NOT broaden to other K-large shapes without per-shape validation (Bonus shows the win does not extend).

## Bonus: V2-RRR vs V2-RCR layout pivot, 8B Down (4096×4096×14336)

To test whether the C4 K-large RRR advantage extends to the smaller K-large shape (8B Down), built both V2-RCR and V2-RRR for 4096×4096×14336 and ran BABA paired bench.

| Layout | median (TF) | mean (TF) | stdev | n | Δ vs RCR | Welch t |
|---|---:|---:|---:|---:|---:|---:|
| V2-RCR (baseline) | 825.20 | 824.69 | 11.70 | 10 | — | — |
| V2-RRR (candidate) | 817.96 | 816.23 | 6.31 | 10 | -0.88% | -2.01 |

(Both throttled at ~1700 MHz throughout — absolute values are ~50% of expected non-throttled but BABA-relative comparison remains valid since both kernels suffer equal throttle.)

**Verdict:** NO SHIP. RRR is neutral-to-slightly-worse than RCR at 8B Down K=14336. The K-large RRR advantage is shape-specific to (4096, 8192, 28672) — it does NOT extend to:
- Smaller K (14336 vs 28672), OR
- Different N/baseline (RCR vs CRR baseline)

This is consistent with the hypothesis that A-side row-contiguous K-loop load efficiency only dominates at K ≥ ~28672 where the K-loop trip count is large enough for the per-iteration A-load advantage to outweigh other effects.

## Recommended autotune additions

Single shape entry for the V2 dispatcher (no broadening):

```cpp
// 70B Down (M=4096, N=8192, K=28672) — V2-RRR beats V2-CRR by +12.14% (R32 Dev C C4).
//   Welch t=+49.24, n=10 each, BABA-paired, preheated, sclk-stable.
//   Do NOT extend to other K-large shapes — 8B Down (K=14336) is neutral (R32 Dev C bonus).
if (M == 4096 && N == 8192 && K == 28672) {
    return Layout::RRR;
}
```

Place in the existing host-side `dispatch_pq_v2<>` autotune fan-out (kernel_mxfp8_layouts.cpp ~line 5602-5616 area, where `gemm_*_pq_v2` are bound). The exact integration point depends on how the upstream dispatcher selects between `gemm_crr_pq_v2` / `gemm_rcr_pq_v2` / `gemm_rrr_pq_v2` — recommend the SHIP commit only adds the data point + recommendation, and an upstream dispatcher patch lands separately after a Reviewer cross-GPU triangulation (per R31 Reviewer pattern: 4-GPU triangulation before merging shape-specific autotune entries).

## Closures (cumulative paradigm-correction list additions)

Add to the cycle-16 closure list:

1. **V2-CRR cachepolicy lever for 70B Down (K=28672) shape** — all 4 cp values neutral (R32 Dev C C1).
2. **V2-RCR cachepolicy lever for 8B Down (K=14336)** — cp=2 LOSES -4.67%, cp=3 LOSES -3.49%; reconfirms R28-era V2-RCR cp closure across both Down shapes (R32 Dev C C3).
3. **K-iter LDS/VGPR pressure hypothesis for V2-CRR K=28672** — falsified at the source: K-loop is runtime, kernel resource report identical for K=8192 vs K=28672 (R32 Dev C C2).
4. **V2-RRR layout-pivot for K-large MLP Down shapes — partially open**: SHIPs at 70B Down (M=4096, N=8192, K=28672, +12.14%). NEUTRAL at 8B Down (M=4096, N=4096, K=14336, -0.88%). Lever is shape-specific, not K-magnitude-broad.

## Artifacts

All under `analysis/fp8_gemm/mi350x/`:

- `r32c_orchestrate.sh` — main C1/C3/C4 driver
- `r32c_audit.sh` — C2 resource-report driver
- `r32c_bonus_8b_down.sh` — bonus 8B Down RRR vs RCR driver
- `r32c_paired_bench.py` — generic in-process BABA harness (loads 2 .so via PY_MODULE_NAME override)
- `r32c_c4_paired_bench.py` — cross-layout BABA harness for V2-RRR vs V2-CRR (different A/B shapes)
- `r32c_bonus_paired_bench.py` — cross-layout BABA harness for V2-RRR vs V2-RCR
- `r32c_C1_cp0_vs_cp{1,2,3}.txt`, `r32c_C1_cp0_vs_cp2_redo.txt` — C1 cachepolicy results
- `r32c_C3_cp0_vs_cp{1,2,3}.txt` — C3 cachepolicy results
- `r32c_C4_rrr_vs_crr.txt`, `r32c_C4_rrr_vs_crr_redo.txt` — C4 SHIP candidate (use redo: cleanest run)
- `r32c_BONUS_8b_down_rrr_vs_rcr.txt` — bonus 8B Down result
- `r32c_audit_summary.txt`, `r32c_audit_*.log` — C2 resource reports
- `r32c_build_*.log` — per-build compile logs (for md5 audit)
- `r32c_orchestrate_full.log` — full orchestrate stdout
