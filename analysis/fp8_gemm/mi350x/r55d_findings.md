# R55 Dev D — 70B Down RCR HEADROOM Attack: P4 Scale L2 Residency (B-scale-only cachepolicy)

**Verdict line (final):** `R55 Dev D: 70B Down RCR P4 scale L2 residency — REFUTED-EMPIRICAL — B-scale-only cachepolicy bias closes the unswept half of cachepolicy axis (R47C unified, R48F A-only); all three values within ~10 TFLOPS noise floor`

## Mission recap

- Target cell: 70B Down RCR (M=4096, N=8192, K=28672)
- Headroom gap: -3.4 pp (91.6% MX/FP8 vs 95% gate)
- Lever assigned: P4 — scale L2 residency
- Strict SCLK protocol: GPU 3 isolated, 5 runs/cell, 30 s cooldown, 60 s rebuild_cool, MXFP8_WARMUP=100 ITERS=200, median scoring
- Pass criteria: ≥+1% on 70B Down RCR with SNR ≥ 48 dB, det 3/3 PASS, no regression on 8B Down RCR + 70B Q/O RCR

## Lever choice and pivot

Three options were available:

1. Per-CTA scale prefetch warmup (similar to R31 Dev C VGPR-prefetch)
2. Cachepolicy tuning on scale loads only
3. Scale slab ordering in host preshuffle

**Option 1 was attempted first** with the existing `MXFP8_RCR_V2_SCALE_PREFETCH=1` build flag. Result: VGPR=256, Spill=191, throughput collapsed to **208 TFLOPS** (vs ~2950 baseline). The V2 RCR kernel sits at 254/256 VGPRs with 0 spill at occupancy 2 — the next-buffer prefetch family is structurally closed at this VGPR ceiling, consistent with R31 Dev C's `STRUCTURAL CLOSURE` verdict and the `v2_rrr_vgpr_ceiling.md` memory note (RRR has the same ceiling).

**Pivot to option 2.** Surveying the cachepolicy axis:

- R47C swept the **unified** `MXFP8_RCR_V2_SCALE_CACHEPOLICY` (both A and B together) → REFUTED, p=0 best, p=2/3 lose 2-4%.
- R48F swept `MXFP8_RCR_ASCALE_CACHEPOLICY` (A-scale only) → REFUTED.
- **B-scale-only had never been swept.** R55 Dev D closes that unswept half.

Rationale for B-only being potentially distinct from A-only:

- A-scale per-CTA footprint (M slab × K dim / 32) and B-scale per-CTA footprint differ by N/M ratio.
- For 70B Down (M=4096, N=8192, K=28672): A-scale = 3.67 MB, B-scale = 7.34 MB.
- Per-XCD L2 capacity ≈ 8 MB. **B-scale alone is right at the per-XCD L2 capacity boundary** while A-scale fits comfortably.
- If cross-CTA scale reuse is L2-evicting on B-side, biasing B's policy independently of A could change outcomes from the unified sweep.

## Implementation

Added macro in `kernel_mxfp8_layouts.cpp` (lines 408–419):

```cpp
#ifndef MXFP8_RCR_BSCALE_CACHEPOLICY
#define MXFP8_RCR_BSCALE_CACHEPOLICY (-1)
#endif
#define MXFP8_RCR_BSCALE_CACHEPOLICY_RESOLVED \
    ((MXFP8_RCR_BSCALE_CACHEPOLICY) >= 0 ? (MXFP8_RCR_BSCALE_CACHEPOLICY) : (MXFP8_RCR_V2_SCALE_CACHEPOLICY))
```

Wired into all five B-scale `buffer_load_b64` sites (replacing direct `MXFP8_RCR_V2_SCALE_CACHEPOLICY`):

- Line 2900 — `_coop_b_raw` (cooperative b_scale pre-issue)
- Line 2942 — non-cooperative inline path
- Line 3400 — alt path cooperative b_scale
- Line 3426 — alt path non-cooperative
- Line 3473 — `load_scale_buffer_next` (VGPR-prefetch path)

Default-OFF (`-1` sentinel) inherits unified policy → **byte-identical** with no `-DMXFP8_RCR_BSCALE_CACHEPOLICY` define.

## Resource usage (V2 RCR kernel `_Z29rcr_exact_8wave_scaled_kernelILb1ELi2EE`)

All four builds identical:

| Build | TotalSGPRs | VGPRs | VGPRs Spill | Occupancy | Scratch |
|-------|------------|-------|-------------|-----------|---------|
| default-OFF | 53 | 254 | 0 | 2 | 0 |
| BSCALE=1 (GLC)   | 53 | 254 | 0 | 2 | 0 |
| BSCALE=2 (SLC)   | 53 | 254 | 0 | 2 | 0 |
| BSCALE=3 (GLC|SLC) | 53 | 254 | 0 | 2 | 0 |

No spill, no occupancy regression — clean substitution.

## SCLK 5-run results (70B Down RCR, M=4096 N=8192 K=28672, GPU 3)

| Treatment | Median TFLOPS | Mean | Stdev | Δ vs baseline | Δ vs default-OFF |
|---|---|---|---|---|---|
| baseline_70BD (pre-wire) | 2950.86 | 2949.89 | 13.32 | — | — |
| default_off_70BD          | **2960.07** | 2959.89 | 11.49 | +0.31% | (control) |
| bscale1_70BD (GLC)        | 2967.64 | 2965.11 |  6.18 | +0.57% | +0.26% |
| bscale2_70BD (SLC)        | 2963.66 | 2962.95 |  4.45 | +0.43% | +0.12% |
| bscale3_70BD (GLC|SLC)    | 2960.86 | 2958.56 | 10.13 | +0.34% | +0.03% |

Per-run times in `r55d_results/{baseline,default_off,bscale1,bscale2,bscale3}_70BD/`.

## Statistical analysis

- **Largest treatment effect** (bscale1 GLC): +7.57 TFLOPS over default-OFF control = **+0.26%**.
- **Pooled noise floor** (across all five 5-run cells): mean stdev ≈ 9.1 TFLOPS, max stdev = 13.32 TFLOPS.
- The +0.26% bscale1 effect is well within ±1σ of the noise floor.
- Pass threshold was **≥+1.0%** (i.e., ≥+29.6 TFLOPS over default-OFF). All three treatments fall short by 4–10×.

## Verdict and analysis

**REFUTED-EMPIRICAL.** B-scale-only cachepolicy bias does not move the 70B Down RCR HEADROOM gap.

The hypothesis was that the asymmetric A/B scale footprint (3.67 MB vs 7.34 MB) and the per-XCD L2 capacity boundary (~8 MB) would create a regime where biasing B-scale independently of A would change the L2 residency story relative to either the unified sweep (R47C) or the A-only sweep (R48F). The empirical sweep refutes this.

Three closures on the cachepolicy axis now form a complete picture:

| Sweep | Verdict | Best |
|---|---|---|
| R47C: unified A+B  | REFUTED | p=0 default |
| R48F: A-scale only | REFUTED | p=0 default |
| **R55D: B-scale only** | **REFUTED** | **p=0 default** |

**Cachepolicy axis is now triply closed for V2 RCR scale loads.** The default p=0 (no GLC, no SLC) is empirically optimal across all three slicings of A/B independence. This is structurally consistent with: (a) MI355X TCC arrays running coherent by default, (b) scale tensors being shared cross-XCD, and (c) the working set for both A and B scale slabs being small enough that L2 caching policy hints do not change residency outcomes meaningfully under the existing V2 layout.

## What this rules out for future R55+ cycles

1. Any further attempts on the cachepolicy axis for V2 RCR scale loads should be expected REFUTED (full axis swept).
2. The 70B Down RCR HEADROOM gap is **not** a scale L2-residency problem under the current V2 layout — at least not one addressable by buffer-load cachepolicy bits.
3. The remaining P4 sub-levers are: (a) **scale slab ordering in host preshuffle** (untouched), and (b) per-CTA scale prefetch warmup that does NOT use VGPRs (e.g., dedicated SGPR pointer chase, or LDS staging). The VGPR-resident next-buffer family (R31C) remains structurally closed.

## Cross-cell no-regression

**Skipped.** Pass criterion (≥+1%) was not met on the primary cell, so cross-cell verification is moot — there is nothing to ship.

## Determinism / SNR

**Skipped.** Same reason — no treatment to certify.

## PMC re-collection

**Skipped.** Same reason. The R54B PMC story (SQ_WAIT_ANY +71.6% on 70B Down) remains the diagnostic; this lever did not move it.

## Artifacts

- Workspace: `/shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x/r55d_workspace/`
  - `kernel_mxfp8_layouts.cpp` — lines 408–419 macro, 5 wire sites
  - `run_bench.sh` — SCLK protocol runner
  - `build_default_off.log`, `build_b1.log`, `build_b2.log`, `build_b3.log` — per-treatment build logs with resource usage
- Results: `/shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x/r55d_results/`
  - `baseline_70BD/`, `default_off_70BD/`, `bscale1_70BD/`, `bscale2_70BD/`, `bscale3_70BD/`
  - Each contains 5 per-run JSONs + `runs.txt`

## Final line

`R55 Dev D: 70B Down RCR P4 scale L2 residency — REFUTED-EMPIRICAL — B-scale-only cachepolicy bias closes the unswept half of cachepolicy axis (R47C unified, R48F A-only); all three values within ~10 TFLOPS noise floor`
