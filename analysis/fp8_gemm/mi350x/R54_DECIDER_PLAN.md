# R54 Decider Plan
**Date:** 2026-04-19
**R52 baseline:** 36/42 strict, **R53 outcome:** 33/42 strict (-3 cohort)
**Goal:** recover the -6 cohort losses + add new VC rescues; floor 34/42, stretch 38/42

## Context recap (do not re-derive)

R53 NET contribution = +2 NEW VC rescues (D-3B_1 `4096x32768x14336`, D-3B_2 `32768x4096x14336`)
+ 6 perf claw-backs. Apparent -3 strict net is the documented R45+ cohort-race tail-draw on
UNCHANGED HK R40B/R41B `.so` files — NOT a regression.

**R50D shim is now proven TILE-GENERIC:** handles 256×256, 96×640, AND 64×1024 aiter `.co`
files AS-IS with `bdx=256`. Validates the `asm_gemm_a4w4.cu:290` universal-bdx hypothesis.
**5 consecutive R50D-AS-IS reuse rounds; zero kernel rebuild required.**

**Aiter heuristic** (`asm_gemm_a4w4.cu:103-145`):
- `local_round = ceil(M/tile_M) * ceil(N/tile_N) / num_cu` (num_cu=256 on MI355X)
- Tiebreak: `compute2mem_efficiency = (tile_M*tile_N) / (tile_M+tile_N)` — larger wins
- All bpreshuffle=1 tiles available; `bdx=256` universal
- 256×256 has eff=128 (max among rectangular tiles); will dominate when rounds tie

**Heuristic verdict for ALL 6 cohort-race losses + remaining sub-90% HK-VC: 256×256.**
This makes R54 mechanism-confidence VERY HIGH — same dispatch already proven in 7 R50D/R51/R52
shapes (4096x32768x28672, 14336x4096x32768, 16384x4096x28672, 28672x4096x16384,
4096x28672x32768, 4096x32768x128256, 4096x4096x32768) with 100% bit-determinism.

## Worker assignments

### R54 Opt E-1 — Cohort-race rescue (M-heavy, N=4096) (worker A on GPU 0)
**Mechanism-confidence: HIGH** — direct R51-style port; identical 256×256 tile, bdx=256
- 3 candidates (R40B/R41B → AITER_SHIM 256×256):
  - `32768x4096x2048` (R41B, R52 VC pct_comp 102.0%, R53 cohort LOSS PASS_8/10 fin_min=0.956)
    - heur tile **256×256**, log2_k_split=0, grid `gdx=16, gdy=16, gdz=1, bdx=256`
    - `co_path: /shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`
    - `kernel: _ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E`
  - `32768x4096x3072` (R40B, R52 VC pct_comp 99.5%, R53 cohort LOSS PASS_9/10 wcf_max=0.0218)
    - heur tile **256×256**, grid `gdx=16, gdy=16, gdz=1, bdx=256`
  - `28672x4096x8192` (R40B, R52 VC pct_comp 91.8%, R53 cohort LOSS PASS_9/10 wcf_max=0.0201)
    - heur tile **256×256**, grid `gdx=16, gdy=14, gdz=1, bdx=256` (note: 28672 % 256 == 0; 14 wgs)
- All 3 reuse `R50D_aiter_shim` AS-IS; just add manifest `aiter_dispatch` entries
- Risk of regression: LOW — perf risk only on `32768x4096x2048` (HK was 102%);
  but D-3A-1 protection: keep HK as fallback if perf < HK; rescue-on-VC-loss still wins
- D-3A-1 caveat: `32768x4096x2048` HK-VC perf 102%; AITER perf must be ≥98% to PROMOTE,
  else accept NO_VC→VC trade only if user-policy permits cohort-race-driven swap

### R54 Opt E-2 — Cohort-race rescue (N-heavy, M=4096 + mid-K) (worker B on GPU 1)
**Mechanism-confidence: HIGH** — sister of R52D2A/D2B/D2C
- 3 candidates (R40B → AITER_SHIM 256×256):
  - `4096x32768x4096` (R40B, R52 VC pct_comp 91.8%, R53 cohort LOSS PASS_9/10 fin_min=0.951)
    - heur tile **256×256**, grid `gdx=16, gdy=128, gdz=1, bdx=256`
  - `4096x32768x6144` (R40B, R52 VC pct_comp 92.3%, R53 cohort LOSS PASS_8/10 fin=0.926)
    - heur tile **256×256**, grid `gdx=16, gdy=128, gdz=1, bdx=256`
  - `16384x4096x6144` (R40B, R52 VC pct_comp 98.5%, R53 cohort LOSS PASS_8/10 fin_min=0.950)
    - heur tile **256×256**, grid `gdx=16, gdy=64, gdz=1, bdx=256`
- All 3 are recipe-exact analogs of R52D2A `4096x28672x32768` (256×256 / shim AS-IS / N=N>>K-or-M).
- 0% kernel-rewrite risk; perf claw-back probability HIGH given R52D2A precedent (101.78%)
- Risk: `16384x4096x6144` HK was 98.5% — AITER perf likely 97-105% range; promote if ≥97%

### R54 Opt D-4A — Sub-90% HK-VC rescue (M=N≈mid-K) (worker C on GPU 2)
**Mechanism-confidence: HIGH** — same proven 256×256 dispatch
- 3 candidates (HK-VC sub-90% → AITER_SHIM 256×256):
  - `4096x14336x16384` (R40B, **84.41%** — lowest HK-VC pct on board; well below D-3C threshold)
    - heur tile **256×256**, grid `gdx=16, gdy=56, gdz=1, bdx=256`
    - Sister of R51D1 (`14336x4096x32768`) which scored 103.17% at AITER 256×256
  - `32768x4096x7168` (R40B, **88.36%**)
    - heur tile **256×256**, grid `gdx=16, gdy=128, gdz=1, bdx=256`
  - `6144x4096x8192` (R40B, **89.53%**)
    - heur tile **256×256**, grid `gdx=16, gdy=24, gdz=1, bdx=256` (note: 6144=24*256)
- All 3 use R50D shim AS-IS; PROMOTE gate: AITER pct_comp > HK pct_comp + 1.0pp
- Risk: D-3A-1-style risk LOW (all 3 are clearly sub-90% HK-VC; aiter has wide headroom)

### R54 Opt D-4B — 90-95% HK-VC perf claw-back (worker D on GPU 3)
**Mechanism-confidence: MEDIUM** — competitive HK; aiter must clearly win
- 3 candidates (HK-VC 90-95% → AITER_SHIM 256×256):
  - `4096x4096x16384` (R40B, **93.30%**)
    - heur tile **256×256**, grid `gdx=16, gdy=16, gdz=1, bdx=256`
    - Sister of R52D2C (`4096x4096x32768`) which scored 105.94% at AITER 256×256
  - `4096x14336x8192` (R40B, **91.92%**)
    - heur tile **256×256**, grid `gdx=16, gdy=56, gdz=1, bdx=256`
  - `16384x4096x7168` (R40B, **93.76%**)
    - heur tile **256×256**, grid `gdx=16, gdy=64, gdz=1, bdx=256`
- D-3C-style marginal: PROMOTE only if AITER ≥ HK + 0.5pp AND 10/10 PASS
- Risk: HK is competitive; D-3A-1 caveat applies; ACCEPT_FALLBACK if AITER < HK

## Sub-90% HK-VC shapes audit (from R53_INTEGRATION_10RUN.json)

| Shape (MxNxK) | Source | HK pct_comp | Aiter heur tile | .co available? | Worker |
|---|---|---|---|---|---|
| 4096x14336x16384 | R40B | **84.41%** | 256x256 | YES | D-4A |
| 32768x4096x7168 | R40B | **88.36%** | 256x256 | YES | D-4A |
| 6144x4096x8192 | R40B | **89.53%** | 256x256 | YES | D-4A |
| 4096x32768x14336 | AITER 64x1024 | 66.70% | (already AITER) | n/a | SKIP — already aiter VC, just re-VC'd in R53 |
| 32768x4096x14336 | AITER 64x1024 | 85.56% | (already AITER) | n/a | SKIP — already aiter VC, just re-VC'd in R53 |
| 128256x32768x4096 | AITER 64x1024 | 86.17% | (already AITER) | n/a | SKIP — already aiter VC |
| 28672x32768x4096 | AITER 64x1024 | 87.68% | (already AITER) | n/a | SKIP — already aiter VC |
| 14336x32768x4096 | AITER 64x1024 | 87.83% | (already AITER) | n/a | SKIP — already aiter VC |

90-95% HK-VC pool (broader candidate set):

| Shape | Source | HK pct_comp | Aiter heur tile | .co | Worker |
|---|---|---|---|---|---|
| 4096x4096x16384 | R40B | 93.30% | 256x256 | YES | D-4B |
| 4096x14336x8192 | R40B | 91.92% | 256x256 | YES | D-4B |
| 16384x4096x7168 | R40B | 93.76% | 256x256 | YES | D-4B |
| 6144x32768x4096 | R40B | 94.14% | 256x256 | YES | (defer; close to 95%) |

## Cohort-race losses audit (6 shapes)

| Shape | Source | R52→R53 | R53 wcf_max | R53 fin_min | Aiter heur | .co | Worker |
|---|---|---|---|---|---|---|---|
| 32768x4096x2048 | R41B | VC→PASS_8/10 NV | 0.0119 | 0.9563 | 256x256 | YES | E-1 |
| 32768x4096x3072 | R40B | VC→PASS_9/10 NV | 0.0218 | 0.9895 | 256x256 | YES | E-1 |
| 28672x4096x8192 | R40B | VC→PASS_9/10 NV | 0.0201 | 0.9771 | 256x256 | YES | E-1 |
| 4096x32768x4096 | R40B | VC→PASS_9/10 NV | 0.0098 | 0.9508 | 256x256 | YES | E-2 |
| 4096x32768x6144 | R40B | VC→PASS_8/10 NV | 0.0249 | 0.9263 | 256x256 | YES | E-2 |
| 16384x4096x6144 | R40B | VC→PASS_8/10 NV | 0.0241 | 0.9499 | 256x256 | YES | E-2 |

**100% have available 256×256 .co; 100% are R50D-shim-compatible AS-IS.**

## Closed-axis reminder (DO NOT propose)

R45-R53 closed axes (any worker proposing these should be rejected by reviewer):
- Any external/internal vmcnt fence position (R45B, R47A, R49C — 4 closures)
- MFMA reorder / interleave (R50A — ISA-verified 1:4 spread, race unmoved)
- R38A INLINE_BUFLOAD_LDS in production hot path (R47B regression)
- R39A TAIL_SCALE_CLAMP (R46A falsified on intermediate-K cohort)
- R43 Opt A prefetch-gating, R43 Opt B VGPR-PF keepalive, R43 Opt C variant table
- R48A asm-block physical split, R48B reorder+prio1+drain, R48C PF_MPT depth knob
- R49A aiter vmcnt(15) single-knob port
- R49B per-iter drain on (4096,32768,28672) cohort
- R50A MFMA↔ds_read 1:3 interleaving (axis closed)
- R50C perf claw-back via gm×lgk×pfoff sweep on UNCHANGED kernel
- **R54 Opt B (32×32×64 MFMA)** — explicitly NOT included this round; mechanism-confidence
  too low vs Opt D-4 / Opt E aiter rescues which have proven precedent

## PROMOTE/REJECT gate (apply per worker per candidate)

For aiter rescues, the standing R45+ 10-run reviewer protocol applies:
1. n_OK ≥ 8/10 INDEPENDENT seeds (ok_threshold=8)
2. wcf_max < 0.02
3. wcf_std < 0.01
4. fin_min ≥ 0.97
5. Perf gate:
   - For E-1, E-2 (cohort-race rescue): PROMOTE if AITER strict-VC, even if perf < HK
     (because HK is FAILING gate due to cohort-race). NO_VC→VC strict gain wins.
   - For D-4A: PROMOTE if AITER pct_comp > HK pct_comp + 1.0 pp
   - For D-4B: PROMOTE if AITER pct_comp > HK pct_comp + 0.5 pp (D-3C-style marginal)
6. **D-3A-1 risk**: if HK currently >92% comp AND VC, do NOT swap unless AITER strictly wins.
   In R54 only `32768x4096x2048` (HK 102%, R53 NV) carries this risk in cohort group.

## Expected R54 outcome (probability-weighted)

- Floor (highest-confidence subset): 34/42 strict — recover only the 4 cleanest cohort cells
  (`32768x4096x3072`, `28672x4096x8192`, `4096x32768x4096`, `16384x4096x6144`)
- Mode: 36/42 strict — match R52 by recovering all 6 cohort losses minus 1-2 perf-fall non-promotes
- Stretch: 38/42 strict — all 6 cohort + 2-3 sub-90% rescues PROMOTE

---

## Summary

**3 worker cohorts, 12 total candidates** assigned across **2 mechanism-confidence axes**:
(a) **Opt E cohort-race rescue** (6 candidates split E-1 / E-2; HIGH confidence; aiter
already proven on 7 sister shapes), and (b) **Opt D-4 sub-95% HK-VC rescue**
(6 candidates split D-4A sub-90% HIGH-confidence / D-4B 90-95% MEDIUM-marginal). All 12
candidates use R50D shim AS-IS with 256×256 .co; zero kernel rebuild required;
heuristic-uniform tile pick (256×256 wins on `compute2mem_efficiency` tiebreak across all
12 shapes). R54 Opt B (32×32×64 MFMA) explicitly DEFERRED — aiter rescue mining is the
mechanism-confident frontier.
