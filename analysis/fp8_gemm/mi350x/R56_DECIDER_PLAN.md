# R56 Decider Plan — Opt G perf claw-back on 8 LOSE cells

**Date:** 2026-04-19
**R55 baseline:** **42/42 strict 10-run VC** — first 100% leaderboard round in project history (committed 099d1372)
**Goal:** Convert LOSE→WIN on the 8 sub-100% pct_comp cells via alternative aiter `.co` tile dispatch (Opt G axis). NO new VC available (ceiling reached); pure perf claw-back round.

---

## 1. R55 baseline reminder

- 42/42 strict 10-run VC (gate: n_OK≥8/10 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min≥0.97)
- 38 AITER overrides (R50D shim AS-IS, 7th consecutive AS-IS reuse round) + 4 HK baselines
- 34/42 WIN (≥100% pct_comp); **8 LOSE cells remain** (pct_comp 65.68% – 98.32%)
- Strict-VC ceiling REACHED under current gate — R56 is perf-only
- Cohort-race surface cut from 15→4 HK cells (no churn risk on remaining HK)

The 8 LOSE cells are all carried-from-prior-rounds AITER (7 cells) + 1 HK; no R55 introductions. Hypothesis: aiter heuristic `local_round` minimizer + `compute2mem_efficiency` tiebreak under-picks tiles when ties exist. Heuristic computation below shows that **5 of 7 AITER LOSE cells have a 256×256 tile with IDENTICAL `local_round` count and HIGHER `compute2mem_efficiency` (128 vs 60.2)** — i.e., 256×256 should have won the heuristic but was rejected by the aiter dispatcher (possibly a different tiebreak we did not surface). These are the prime Opt G targets.

---

## 2. The 8 LOSE cells

| # | Shape (MxNxK) | Current tile | Source | p50 TFLOPS | pct_comp |
|---:|---|---|---|---:|---:|
| L1 | 4096x32768x14336   | 64×1024  | R53D3B_1_AITER | 3478.5 | **65.68%** |
| L2 | 32768x4096x14336   | 64×1024  | R53D3B_2_AITER | 4394.7 | **84.13%** |
| L3 | 128256x32768x4096  | 64×1024  | R53D3B_3_AITER | 3943.6 | **86.93%** |
| L4 | 28672x32768x4096   | 64×1024  | R53D3C_2_AITER | 3922.5 | **87.82%** |
| L5 | 14336x32768x4096   | 64×1024  | R53D3C_1_AITER | 3923.7 | **87.92%** |
| L6 | 16384x4096x14336   | 64×1024  | R53D3C_3_AITER | 4642.4 | **90.28%** |
| L7 | 4096x128256x32768  | (HK R41A)| R41A           | 3118.3 | **97.59%** |
| L8 | 4096x32768x128256  | 256×256  | R52D2B_AITER   | 5683.7 | **98.32%** |

Cluster by current tile: 64×1024 ×6 (L1–L6), HK ×1 (L7), 256×256 ×1 (L8).

---

## 3. Available aiter `.co` binaries

`/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/`:

```
BpreShuffle: 32x{128,256,384,512,640,768,896,1024}
             64x{128,256,384,512,640,768,896,1024}
             96x{128,256,384,512,640}
            128x{128,256,384,512}
            160x{128,256,384}
            192x{128,256}    224x{128,256}
            256x{128,256}
noBpreShuffle_256x256
```

Compute-to-memory efficiency `eff = tM*tN/(tM+tN)`:
- 256×256 = 128.0 (max)  ·  128×512 = 102.4  ·  128×384 = 96.0
- 192×256 = 109.7  ·  96×640 = 83.5  ·  64×1024 = 60.2

R56 candidate tiles (tested by heuristic minimization across the 8 cells): {256×256, 128×512, 192×256, 96×640} via R50D shim AS-IS. All four `.co` files exist.

---

## 4. Per-cell alt-tile candidates

For each cell, top-3 lowest `local_round = ceil(M/tM)*ceil(N/tN)/256` with eff tiebreak shown. **All grid math: bdx=256, gdz=1. KernelArgs (M, N, K) per shape.**

### L1 — 4096x32768x14336 (current 64×1024 → 65.68%)
| alt | local_round | eff | grid (gdx, gdy) | `.co` |
|---|---:|---:|---|---|
| **256×256** | 8 | 128.0 | (128, 16) | `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` |
| 128×512  | 8 | 102.4 | (64, 32) | `..._BpreShuffle_128x512.co` |
| 64×1024 (cur) | 8 | 60.2 | (32, 64) | (current) |

256×256 ties on rounds with eff=128 vs 60.2 — strong A1 pick.

### L2 — 32768x4096x14336 (current 64×1024 → 84.13%)
| alt | local_round | eff | grid (gdx, gdy) | `.co` |
|---|---:|---:|---|---|
| **256×256** | 8 | 128.0 | (16, 128) | `..._256x256.co` |
| 128×512 | 8 | 102.4 | (8, 256) | `..._128x512.co` |
| 64×1024 (cur) | 8 | 60.2 | (4, 512) | (current) |

Identical pattern to L1; M/N swapped. Strong A1.

### L3 — 128256x32768x4096 (current 64×1024 → 86.93%)
| alt | local_round | eff | grid (gdx, gdy) | `.co` |
|---|---:|---:|---|---|
| **256×256** | 251 | 128.0 | (128, 501) | `..._256x256.co` |
| 128×512 | 251 | 102.4 | (64, 1002) | `..._128x512.co` |
| 64×1024 (cur) | 251 | 60.2 | (32, 2004) | (current) |

Round-count saturated (251 rounds across 256 CUs); eff dominates → 256×256.

### L4 — 28672x32768x4096 (current 64×1024 → 87.82%)
| alt | local_round | eff | grid (gdx, gdy) | `.co` |
|---|---:|---:|---|---|
| **256×256** | 56 | 128.0 | (128, 112) | `..._256x256.co` |
| 128×512 | 56 | 102.4 | (64, 224) | `..._128x512.co` |
| 64×1024 (cur) | 56 | 60.2 | (32, 448) | (current) |

### L5 — 14336x32768x4096 (current 64×1024 → 87.92%)
| alt | local_round | eff | grid (gdx, gdy) | `.co` |
|---|---:|---:|---|---|
| **256×256** | 28 | 128.0 | (128, 56) | `..._256x256.co` |
| 128×512 | 28 | 102.4 | (64, 112) | `..._128x512.co` |
| 64×1024 (cur) | 28 | 60.2 | (32, 224) | (current) |

### L6 — 16384x4096x14336 (current 64×1024 → 90.28%)
| alt | local_round | eff | grid (gdx, gdy) | `.co` |
|---|---:|---:|---|---|
| **256×256** | 4 | 128.0 | (16, 64) | `..._256x256.co` |
| 128×512 | 4 | 102.4 | (8, 128) | `..._128x512.co` |
| 64×1024 (cur) | 4 | 60.2 | (4, 256) | (current) |

Sister to R55 D-5A (16384x4096x{4096,6144,7168} all PROMOTE'd 256×256 with +6–11pp claw-back). Highest mechanism confidence in the 64×1024 cluster.

### L7 — 4096x128256x32768 (current HK R41A → 97.59%)
| alt | local_round | eff | grid (gdx, gdy) | `.co` |
|---|---:|---:|---|---|
| **256×256** | 32 | 128.0 | (501, 16) | `..._256x256.co` |
| 128×512  | 32 | 102.4 | (251, 32) | `..._128x512.co` |
| 96×640   | 34 | 83.5  | (201, 43) | `..._96x640.co` |

D-3A-1 RISK MEDIUM: HK R41A is at 97.59%, AITER must beat by >0.5pp. Sister R52D2B_AITER on `4096x32768x128256` (mirror N/K) hit 98.32% with 256×256, suggesting K-heavy aspect favors 256×256 marginally.

### L8 — 4096x32768x128256 (current 256×256 → 98.32%)
| alt | local_round | eff | grid (gdx, gdy) | `.co` |
|---|---:|---:|---|---|
| 256×256 (cur) | 8 | 128.0 | (128, 16) | (current) |
| **128×512** | 8 | 102.4 | (64, 32) | `..._128x512.co` |
| 192×256 | 11 | 109.7 | (128, 22) | `..._192x256.co` |

D-3A-1 RISK HIGH: 256×256 already at 98.32%; alt tiles ≥1.68pp away from baseline. Probe only — likely ACCEPT_FALLBACK. Justification for inclusion: K=128256 has unusual K/N=4 ratio; possible 128×512 K-pipeline has different prefetch vs 256×256.

---

## 5. Worker cohort assignments (4 GPUs × 2-cell each, 12 candidates total)

| Cohort | Worker | GPU(s) | Mechanism conf | Candidates | Expected outcome |
|---|---|---|---|---|---|
| **R56 Opt G-1** (homogeneous 256×256 swap on 64×1024 LOSEs cluster A) | A | 0,1 | **VERY HIGH** | L1, L2, L6 → 256×256 (3 candidates) | 3 PROMOTE; eff jump 60.2→128.0 |
| **R56 Opt G-2** (homogeneous 256×256 swap on 64×1024 LOSEs cluster B) | B | 2,3 | **VERY HIGH** | L4, L5, L3 → 256×256 (3 candidates) | 2-3 PROMOTE; large grids may saturate |
| **R56 Opt G-3** (alt-tile sweep, 128×512 falsification on cluster A) | C | 4,5 | MEDIUM | L1, L2, L6 → 128×512 (3 candidates) | 0-2 PROMOTE; eff 102.4 still > 60.2 |
| **R56 Opt G-4** (HK→AITER probe + 256×256 redundancy probe) | D | 6,7 | LOW-MED | L7 → 256×256, L8 → 128×512, L8 → 192×256 (3 candidates) | 0-1 PROMOTE; D-3A-1 dominant |

**Total: 12 candidates across 4 cohorts × 2 GPUs each.**

GPU assignment uses pairs to enable 10-run @ 80% INDEPENDENT seeds split across 2 GPUs per cohort (5 seeds each), cutting wall-clock ~2x per cohort. Verify all GPUs idle via `rocm-smi` pre-launch.

Notes:
- G-1 and G-2 collectively cover all 6 64×1024 cells with 256×256 swap (1:1). If aiter heuristic genuinely under-picks 256×256 on these shapes, expect ≥4/6 PROMOTE.
- G-3 is the **falsification cohort**: if G-1/G-2 PROMOTE on 256×256 but G-3 also PROMOTEs on 128×512, then the issue is "tile too narrow" (any wider tile wins). If only 256×256 wins, then `compute2mem_efficiency` is the dominant factor.
- G-4 covers the two non-cluster cells (L7 HK, L8 256×256-current) with low mechanism confidence; treat as exploratory.

---

## 6. PROMOTE gate spec (R56 Opt G — perf claw-back)

For each candidate (alt-tile vs current baseline):

1. **Bit-determinism gate** (auto-pass for AITER `.co` dlopen): wcf_max=0, wcf_std=0, fin_min=1.0 across 10 INDEPENDENT seeds [101..1010]. AITER `.co` is bit-deterministic so all G-1/G-2/G-3/G-4 candidates auto-clear this.
2. **VC strict 10-run gate** (R45+ standard): n_OK≥8/10 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min≥0.97. AITER bit-determinism means n_OK should be 10/10.
3. **Perf gate (Opt G specific)**: `alt_tile_pct_comp > current_pct_comp + 1.0pp` (D-3C-style strict marginal gate; tighter than R55's 0.5pp because we are swapping a known-good kernel for a probe).
4. **D-3A-1 ACCEPT_FALLBACK protection**: If alt-tile fails gate 3, **keep current baseline; do NOT swap.** Specifically:
   - L7 (HK 97.59%): keep R41A HK if AITER 256×256 < 98.59%
   - L8 (AITER 256×256 98.32%): keep current if alt < 99.32%
   - L1–L6 (AITER 64×1024): keep current if alt < current_pct_comp + 1.0pp
5. **Bench rules MANDATORY**: warmup=200, iters=500, trim=0.10, idle GPU only.
6. **NO kernel rebuild**: R50D shim AS-IS. If a worker proposes a shim modification, REJECT in review.

---

## 7. Floor / Mode / Stretch targets

Aggregate perf-claw-back delta (sum of pct_comp gains across all PROMOTEs vs R55 baseline):

- **Floor: +25pp aggregate, ≥3 PROMOTEs.**
  Mechanism floor: G-1's 3 candidates (L1, L2, L6) all PROMOTE 256×256. Conservative gain assumption: L1 65.68→85% (+20pp), L2 84→90% (+6pp), L6 90→95% (+5pp) ≈ +25–30pp. R55 hit +35.83pp aggregate via 5 D-5 PROMOTEs.

- **Mode: +60pp aggregate, ≥5 PROMOTEs.**
  G-1 (3/3) + G-2 (2/3, large-grid cells L3/L4 may saturate at fewer rounds). L1 alone may yield +20–30pp if 256×256 closes the eff gap fully on this K=14336 cell where competitor hits 5296 TFLOPS.

- **Stretch: +90pp aggregate, ≥7 PROMOTEs.**
  All 6 of L1–L6 PROMOTE on 256×256 + at least 1 G-4 success (likely L7 HK→AITER 256×256). Possible but contingent on no G-2 large-grid saturation surprise.

R55-comparable: R55 5 PROMOTEs / +35.83pp aggregate (D-5A/D-5B). R56 has 12 candidates with 6 of them in the very-high-confidence 64×1024→256×256 swap; +60pp Mode is the realistic bullseye.

---

## 8. Reviewer hand-off contract

Each worker emits per-candidate fragments named:
`R56G<COHORT_INDEX>_<CELL_LABEL>_INTEGRATION_FRAGMENT.json` (e.g. `R56G1_L1_INTEGRATION_FRAGMENT.json`)

Required schema:
```json
{
  "round": "R56",
  "opt": "G-1",            // G-1 / G-2 / G-3 / G-4
  "cell_label": "L1",      // L1..L8
  "shape": [M, N, K],
  "current_baseline": {
    "source": "R53D3B_1_AITER",
    "tile": [64, 1024],
    "pct_comp": 65.68
  },
  "alt_tile": [256, 256],
  "co_path": "/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co",
  "kernel_name": "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E",
  "shim_so": "build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so",
  "shim_rebuild_required": false,
  "grid": {"gdx": 128, "gdy": 16, "gdz": 1, "bdx": 256},
  "kernel_args": [4096, 32768, 14336],
  "smoke_pass": true,
  "ten_run": {
    "seeds": [101,202,303,404,505,606,707,808,909,1010],
    "warmup": 200, "iters": 500, "trim_frac": 0.10,
    "n_OK": 10, "wcf_max": 0.0, "wcf_std": 0.0, "fin_min": 1.0,
    "tflops_p50": 0.0, "pct_comp": 0.0
  },
  "perf_delta_pp": 0.0,           // alt_pct_comp - current_pct_comp
  "verdict": "PROMOTE | ACCEPT_FALLBACK | DEAD",
  "verdict_reason": "..."
}
```

Reviewer aggregates fragments → `R56_INTEGRATION_MANIFEST.json` (delta from R55 manifest), runs full 42-shape 10-run @ 80% INDEPENDENT seeds on idle GPU pool, emits `R56_INTEGRATION_VERDICT.md` with:
- Headline 42/42 strict-VC retention (must hold from R55)
- 8 LOSE-cell perf table: R55 pct_comp → R56 pct_comp, delta, accept/reject
- WIN count delta (target: 34 → 38–41 of 42)
- ACCEPT_FALLBACK count and cohort-race churn audit on UNCHANGED cells

---

## 9. Closed-axis reminder (DO NOT propose in R56)

Carry-forward from R55 closed axes; in addition for R56:
- **DO NOT modify R50D shim.** 7 consecutive AS-IS reuse rounds; treat as black box.
- **DO NOT touch the 4 remaining HK cells** (16384x4096x{2048,3072}, 32768x14336x2048, 4096x128256x32768) other than the L7 HK→AITER probe in G-4. Cohort-race surface is at its R55 minimum (4 cells); avoid expanding it.
- **DO NOT propose VC-improving kernel changes.** Strict-VC ceiling is reached; R56 is perf-only.
- **DO NOT use warmup<200 or iters<500.** Bench rules.

---

## 10. Confidence rationale

The G-1/G-2 mechanism is **stronger than R55 D-5** because:
- D-5 swapped HK→AITER on shapes where HK was already at 99–106% (marginal upside).
- G-1/G-2 swaps AITER 64×1024 (eff=60.2) → AITER 256×256 (eff=128) on shapes where the 256×256 tile has IDENTICAL `local_round` count. The aiter heuristic should have picked 256×256 and didn't — meaning the dispatcher applied a tiebreak we did not capture in the disasm, OR there is a runtime fallback. Either way, *manually forcing 256×256* via R50D shim bypasses any aiter-internal heuristic noise.
- 11 R55 D-5A/D-5B/E-3/E-4 PROMOTEs with 256×256 .co + bit-determinism on every seed = mechanism-validated.
- The 64×1024 .co was originally selected by R53 D-3B/D-3C in a per-shape best-of-3 sweep that did not include 256×256 vs 64×1024 head-to-head with `compute2mem_efficiency` analysis. R56 G-1/G-2 closes that gap.

The two highest-confidence cells are L1 (4096x32768x14336, 65.68%, the worst LOSE) and L6 (16384x4096x14336, 90.28%, sister to R55 D-5A 16384x4096x* cluster). Floor is gated on these two.

---

## Summary

**4 cohorts, 12 total candidates, 8 LOSE cells covered.** 6 cells (L1–L6, all currently 64×1024 AITER) get 256×256 swap (G-1+G-2; very high confidence, eff jump 60.2→128). 3 of those also get 128×512 falsification probes (G-3; medium). L7 (HK) gets HK→AITER 256×256 probe; L8 (already 256×256) gets 128×512+192×256 alt probes (G-4; low D-3A-1 risk). All use R50D shim AS-IS — zero rebuild. Floor +25pp / Mode +60pp / Stretch +90pp aggregate perf claw-back. Strict-VC ceiling 42/42 must hold. R55→R56 expected: 34→37–41 WIN cells, 8→1–4 LOSE cells.
