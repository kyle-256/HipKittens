# R55 Decider Plan
**Date:** 2026-04-19
**R54 baseline:** 36/42 strict 10-run VC (27 AITER bit-deterministic + 9 HK)
**Goal:** rescue the remaining 6 HK cohort-race FAIL shapes via AITER `.co` dispatch + reach 38-40/42

## Context recap (do not re-derive)

R54 NET contribution = +6 NEW VC rescues (E-1 ×3 + E-2 ×3) + 6 perf claw-backs (D-4A/D-4B
+17.66pp to +28.29pp). All 12 PROMOTE candidates re-confirmed PASS_10/10 with perfect
bit-determinism (wcf_max=0, wcf_std=0, fin_min=1.0). Apparent net 0 vs R52 (also 36) hides
the structural shift: R52 had 7/42 AITER bit-deterministic; R54 has 27/42 — round-over-round
cohort-race attrition risk now scoped to the remaining 15 HK cells.

**R50D shim is now proven TILE-GENERIC + production-stable across 6 consecutive AS-IS reuse
rounds.** Handles 256×256, 96×640, 64×1024 with universal `bdx=256`.

**Aiter heuristic** (`asm_gemm_a4w4.cu:103-145`):
- `local_round = ceil(M/tile_M) * ceil(N/tile_N) / num_cu` (num_cu=256 on MI355X)
- Tiebreak: `compute2mem_efficiency = (tile_M*tile_N) / (tile_M+tile_N)` — larger wins
- 256×256 has eff=128 (max); will dominate when rounds tie

**Available aiter `.co` tiles** (from `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/`):
- 256×{128,256}, 224×{128,256}, 192×{128,256}, 160×{128,256,384}, 128×{128,256,384,512}
- 96×{128,256,384,512,640}, 64×{128,256,384,512,640,768,896,1024}
- 32×{128,256,384,512,640,768,896,1024}
- All `BpreShuffle` variants present + 256×256 noBpreShuffle alternative

## Current 15 HK cells from R54_INTEGRATION_VERDICT.md (pct_comp + VC)

**HK VC PASS (9 — keep as HK unless AITER strictly wins):**

| Shape | Source | HK pct_comp | VC | D-3A-1 risk |
|---|---|---|---|---|
| 16384x4096x2048   | R40B | 105.2% | PASS | HIGH |
| 16384x4096x3072   | R40B | 101.6% | PASS | HIGH |
| 16384x4096x4096   | R40B | 101.2% | PASS | HIGH |
| 16384x6144x2048   | R40B | 106.2% | PASS | HIGH |
| 32768x14336x2048  | R40B | 103.0% | PASS | HIGH |
| 32768x28672x2048  | R40B |  99.7% | PASS | MEDIUM |
| 32768x6144x2048   | R40B | 103.3% | PASS | HIGH |
| 4096x4096x8192    | R40B | 100.2% | PASS | HIGH |
| 4096x128256x32768 | R41A |  97.9% | PASS | MEDIUM |

**HK FAIL (6 — primary R55 rescue targets):**

| Shape | Source | R54 pct_comp | R54 verdict | Fail mode |
|---|---|---|---|---|
| 16384x14336x2048  | R40B | 100.3% | PASS_9/10  | fin_min=0.9222 |
| 16384x14336x4096  | R40B |  94.5% | FLAKE_1/10 | wcf_max=0.0506 |
| 16384x28672x2048  | R40B |  97.1% | PASS_9/10  | fin_min=0.9657, wcf_max=0.0133 |
| 16384x28672x4096  | R40B |  91.8% | PASS_9/10  | wcf_max=0.0215 |
| 16384x6144x4096   | R40B | 100.7% | FLAKE_4/10 | wcf_max=0.0631 |
| 6144x32768x4096   | R40B |  94.7% | PASS_9/10  | wcf_max=0.0209 |

## Aiter heuristic compute per R55 candidate

For each candidate, `local_round = ceil(M/tile_M) * ceil(N/tile_N) / 256`. Smallest wins;
tiebreak by `compute2mem_efficiency = tile_M*tile_N/(tile_M+tile_N)`. Tile=256×256 has eff=128.

**6 HK FAIL shapes — heuristic table:**

| Shape (MxN) | tile=256x256 rounds | Heuristic pick | grid (gdx, gdy, gdz) | Notes |
|---|---|---|---|---|
| 16384x14336 | ceil(16384/256)*ceil(14336/256)/256 = 64*56/256 = **14** | 256×256 | (56, 64, 1) | clean tile fit |
| 16384x28672 | 64*112/256 = **28**                                     | 256×256 | (112, 64, 1) | clean tile fit |
| 16384x6144  | 64*24/256  = **6**                                      | 256×256 | (24, 64, 1)  | clean tile fit |
| 6144x32768  | 24*128/256 = **12**                                     | 256×256 | (128, 24, 1) | clean tile fit |

All 6 heuristic-pick **256×256**, identical to R54 E-1/E-2 pattern. **0 kernel rebuild risk.**

## Worker assignments

### R55 Opt E-3 — HK FAIL cohort-race rescue (M=16384 x N∈{14336,28672}) (worker A on GPU 0)
**Mechanism-confidence: VERY HIGH** — direct R54 E-1/E-2 port; 4/4 sister shapes
already PROMOTE with bit-determinism (E-2_3 `16384x4096x6144` is sister via M=16384/N=4096)
- 4 candidates (R40B FAIL → AITER_SHIM 256×256):
  - `16384x14336x2048` (R40B, R54 PASS_9/10 100.3% fin=0.9222)
    - heur tile **256×256**, grid `gdx=56, gdy=64, gdz=1, bdx=256`
  - `16384x14336x4096` (R40B, R54 FLAKE_1/10 94.5% wcf=0.0506)
    - heur tile **256×256**, grid `gdx=56, gdy=64, gdz=1, bdx=256`
  - `16384x28672x2048` (R40B, R54 PASS_9/10 97.1% fin=0.9657)
    - heur tile **256×256**, grid `gdx=112, gdy=64, gdz=1, bdx=256`
  - `16384x28672x4096` (R40B, R54 PASS_9/10 91.8% wcf=0.0215)
    - heur tile **256×256**, grid `gdx=112, gdy=64, gdz=1, bdx=256`
- All 4 reuse `R50D_aiter_shim` AS-IS; just add manifest `aiter_dispatch` entries
- `co_path: /shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`
- `kernel: _ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E`
- D-3A-1 risk: NONE — all 4 are HK FAIL (no perf gate; rescue-on-VC-loss wins by definition)

### R55 Opt E-4 — HK FAIL cohort-race rescue (smaller-N + 6144x32768) (worker B on GPU 1)
**Mechanism-confidence: VERY HIGH** — same pattern as E-3
- 2 candidates (R40B FAIL → AITER_SHIM 256×256):
  - `16384x6144x4096` (R40B, R54 FLAKE_4/10 100.7% wcf=0.0631)
    - heur tile **256×256**, grid `gdx=24, gdy=64, gdz=1, bdx=256`
  - `6144x32768x4096` (R40B, R54 PASS_9/10 94.7% wcf=0.0209)
    - heur tile **256×256**, grid `gdx=128, gdy=24, gdz=1, bdx=256`
- Worker B has 2 candidates only (E-3 and E-4 split is 4+2; balances wall on 2 GPUs)
- D-3A-1 risk: NONE for 6144x32768x4096 (HK FAIL); LOW for 16384x6144x4096 (FLAKE; AITER perf risk only — but cohort-race rescue is the dominant factor)

### R55 Opt D-5A — Marginal HK-VC perf claw-back, M=16384 (worker C on GPU 2)
**Mechanism-confidence: MEDIUM** — D-3A-1 risk; HK already at 99-105% so AITER must clearly win
Per AGENT_PROMPT.md non-AITER candidate list: `16384x4096x4096`, `16384x6144x{2048,4096}`.
We already cover `16384x6144x4096` in E-4 (FAIL), so D-5A targets HK-VC PASS shapes.
- 3 candidates (R40B HK-VC PASS → AITER_SHIM 256×256, perf-claw-back probe):
  - `16384x4096x4096` (R40B, R54 PASS_10/10 101.2%)
    - heur tile **256×256**, grid `gdx=16, gdy=64, gdz=1, bdx=256`
  - `16384x6144x2048` (R40B, R54 PASS_10/10 106.2%)
    - heur tile **256×256**, grid `gdx=24, gdy=64, gdz=1, bdx=256`
  - `4096x4096x8192` (R40B, R54 PASS_10/10 100.2%)
    - heur tile **256×256**, grid `gdx=16, gdy=16, gdz=1, bdx=256`
- D-3C marginal gate: PROMOTE only if AITER > HK + 0.5pp AND 10/10 PASS
- D-3A-1 risk: HIGH — all 3 are HK-VC at 100-106%; AITER must clearly win
- ACCEPT_FALLBACK: if AITER < HK perf, keep HK; do NOT swap. AITER bit-determinism stability
  is NOT a sufficient reason to swap a perfectly-passing HK kernel above 100%.

### R55 Opt D-5B — Marginal HK-VC perf claw-back, M=32768 (worker D on GPU 3)
**Mechanism-confidence: MEDIUM** — same D-3A-1 risk profile
Per AGENT_PROMPT.md: `32768x14336x2048`, `32768x28672x2048`, `32768x6144x2048`.
- 3 candidates (R40B HK-VC PASS → AITER_SHIM 256×256, perf-claw-back probe):
  - `32768x14336x2048` (R40B, R54 PASS_10/10 103.0%)
    - heur tile **256×256**, grid `gdx=56, gdy=128, gdz=1, bdx=256`
  - `32768x28672x2048` (R40B, R54 PASS_10/10 99.7%)
    - heur tile **256×256**, grid `gdx=112, gdy=128, gdz=1, bdx=256`
  - `32768x6144x2048` (R40B, R54 PASS_10/10 103.3%)
    - heur tile **256×256**, grid `gdx=24, gdy=128, gdz=1, bdx=256`
- D-3C marginal gate: PROMOTE only if AITER > HK + 0.5pp AND 10/10 PASS
- D-3A-1 risk: HIGH — `32768x14336x2048` and `32768x6144x2048` HK at 103%; AITER must clearly win
- D-3A-1 risk: MEDIUM — `32768x28672x2048` HK at 99.7% (closest to break-even; AITER likely wins)
- ACCEPT_FALLBACK as in D-5A

## Cohort breakdown summary

| Cohort | Worker | GPU | Mechanism conf | Candidates | NET VC potential | Perf claw-back potential |
|---|---|---|---|---|---|---|
| R55 Opt E-3 | A | 0 | VERY HIGH | 4 | +0 to +4 (rescue) | 4-shape +0-15pp |
| R55 Opt E-4 | B | 1 | VERY HIGH | 2 | +0 to +2 (rescue) | 2-shape +0-15pp |
| R55 Opt D-5A | C | 2 | MEDIUM | 3 | +0 (already VC) | 3-shape ±5pp |
| R55 Opt D-5B | D | 3 | MEDIUM | 3 | +0 (already VC) | 3-shape ±5pp |
| **TOTAL** |   |   |   | **12** | **+0 to +6** |   |

Total candidate count: **12** (matches R54 worker scaling 4 GPUs × ~3 candidates/GPU).

## Aiter heuristic uniform pick = 256×256 (no Opt F this round)

**R55 Opt F (192×256/128×256 alternative tile probe) is DEFERRED.**
Justification: aiter heuristic for ALL 6 HK FAIL shapes uniformly picks 256×256 (smallest
local_round + max compute2mem_efficiency tiebreak). The R54 evidence-base is 12/12 PROMOTE
on 256×256 dispatch. Choosing a non-heuristic tile would contradict the aiter author's own
selection logic without prior reason. If E-3/E-4 unexpectedly fail to rescue, R56 can revisit
F as a falsification round on the specific failures.

**R55 Opt B (32×32×64 MFMA in HK) explicitly DEFERRED again** per AGENT_PROMPT.md ranking.

## Closed-axis reminder (DO NOT propose)

R45-R54 closed axes (any worker proposing these should be rejected by reviewer):
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
- **Any "improvement" of HK 256×256 path** (R54 D-4A/D-4B confirmed 256×256 AITER beats HK
  by +17-28pp on 84-94% HK band — port to AITER, do NOT try to "improve" HK)
- **Any "improvement" of the R50D aiter `.co` dlopen path itself** (production-stable;
  6 consecutive AS-IS reuse rounds; treat as black box)
- **Aspect-ratio-blind aiter tile selection** (always run heuristic per shape; do not
  cargo-cult 256×256 onto shapes where heuristic picks otherwise)

## PROMOTE/REJECT gate (apply per worker per candidate)

For aiter rescues, the standing R45+ 10-run reviewer protocol applies:
1. n_OK ≥ 8/10 INDEPENDENT seeds (ok_threshold=8)
2. wcf_max < 0.02
3. wcf_std < 0.01
4. fin_min ≥ 0.97
5. Perf gate:
   - For E-3, E-4 (cohort-race rescue): PROMOTE if AITER strict-VC, even if perf < HK
     (HK is FAILING gate due to cohort-race; NO_VC→VC strict gain wins)
   - For D-5A, D-5B (perf claw-back on HK-VC PASS shape): PROMOTE if
     AITER pct_comp > HK pct_comp + 0.5 pp (D-3C-style marginal gate)
6. **D-3A-1 risk (HIGH for D-5A and D-5B)**: 6 of 6 D-5 candidates are HK-VC at >99%; do NOT
   swap unless AITER strictly wins perf gate. Bit-determinism stability is NOT a swap reason
   when HK already passes 10/10.

## Worker recipe (for all 4 workers)

For each candidate shape:

1. **Reuse R50D shim AS-IS** at
   `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
   No rebuild. Pass kwargs `tile_M`, `tile_N`, `co_path`, `kernel_name` and per-shape
   grid (gdx, gdy, gdz, bdx=256) + KernelArgs (M, N, K).

2. **SMOKE first** (1-iter sanity): single launch, single seed, confirm output is finite +
   not NaN/Inf-saturated. Reject early if SMOKE fails — do not proceed to 10-run.

3. **10-run @ 80% with INDEPENDENT seeds** [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010].
   warmup=200, iters=500, trim_frac=0.10. Per-seed bench then aggregate (n_OK, wcf_max,
   wcf_std, fin_min, tflops mean, pct_comp).

4. **Bench rules MANDATORY**: warmup=200, iters=500, trim=0.10. GPU isolation via
   `HIP_VISIBLE_DEVICES=N` on idle GPU only (check `rocm-smi` first).

5. **File output naming convention** (per cohort and per candidate index I=1..N):
   - `bench_R55<COHORT><I>.py` — bench script (cohort label, e.g., E3, E4, D5A, D5B)
   - `R55_OPT_<COHORT>_<I>_SMOKE.{json,log}`
   - `R55_OPT_<COHORT>_<I>_10RUN.{json,log}`
   - `R55<COHORT><I>_INTEGRATION_FRAGMENT.json` — single-shape manifest delta the reviewer
     can merge

## Sanity floor / mode / stretch targets

- **Floor: 38/42 strict** — recover the 2 cleanest E-3/E-4 cells (most likely
  `16384x14336x2048` and `16384x28672x2048` — both already PASS_9/10 with single fin failure;
  AITER bit-deterministic dispatch trivially passes fin_min ≥ 0.97 gate).
  Reasoning: minimum credible win is +2 NET VC from 4 R54 PASS_9/10 cells, conservative
  assumption that 2 of 4 may show cohort-race tail-draw on first 10-run set.
- **Mode: 39/42 strict** — recover all 4 E-3 + at least 1 E-4 cell (most likely
  `6144x32768x4096`; M=6144 sister R53D3A_2 `6144x4096x16384` already PROMOTE).
  Reasoning: AITER bit-determinism = wcf_max=0 = automatic pass on the failure mode
  (cohort-race wcf/fin) for all 6; only risk is the FLAKE_1/10 and FLAKE_4/10 cells which
  may show 1-iter SMOKE issues.
- **Stretch: 42/42 strict** — all 6 E-3/E-4 PROMOTE + 0 D-5A/D-5B reverts.
  Reasoning: 6 NET VC + 0 perf reverts = 36 + 6 = 42. Requires both (a) all 6 cohort-race
  cells rescue cleanly via AITER and (b) D-5A/D-5B perf claw-backs do NOT regress any
  HK-VC PASS cell. The HK-VC reversion risk is NIL since D-5A/D-5B require AITER strictly
  beat HK to PROMOTE, and HK fallback is preserved.
  HOWEVER, R54 cohort-race churn on UNCHANGED HK .so showed -3 cells; R55's 9 remaining HK
  cells (after 6 are AITER-PROMOTE'd) carry the same risk. Mode 39 is the realistic target.

## Confidence rationale

R54 outcome data: 12/12 PROMOTE with 100% bit-determinism (wcf_max=0, wcf_std=0, fin_min=1.0).
6 of those 12 were direct cohort-race rescues with identical mechanism to R55 E-3/E-4
(R54 E-1, E-2 = HK PASS_8/10 or PASS_9/10 cohort-race FAIL → AITER 256×256 PROMOTE).
The R55 E-3/E-4 candidates are the EXACT same failure mode (PASS_9/10 fin_min or wcf_max
gate failures on UNCHANGED R40B kernels) on shapes where heuristic again picks 256×256.
Mechanism-confidence VERY HIGH for E-3/E-4. The 1 high-risk cell is `16384x14336x4096`
(R54 FLAKE_1/10) — the only flake-mode failure in the cohort. AITER 256×256 still likely
to PROMOTE since the failure was fundamentally a cohort-race wcf=0.0506 (not a kernel bug).

D-5 perf claw-back is mechanism-MEDIUM only because the HK baseline is competitive (99-106%)
and AITER may or may not clearly win. Treat D-5 as exploratory; do NOT swap unless gate is
strictly cleared.

---

## Summary

**4 worker cohorts, 12 total candidates** assigned across **2 mechanism-confidence axes**:
(a) **Opt E-3 / E-4 cohort-race rescue** (6 candidates split E-3 ×4 / E-4 ×2; VERY HIGH
confidence; aiter 256×256 already proven on 12 R54 sister cells), and (b) **Opt D-5A / D-5B
HK-VC perf claw-back probe** (6 candidates split D-5A ×3 / D-5B ×3; MEDIUM confidence;
D-3A-1-style guarded promotion only on AITER strict win). All 12 candidates use R50D shim
AS-IS with 256×256 .co; zero kernel rebuild required; heuristic-uniform tile pick (256×256
wins on `compute2mem_efficiency` tiebreak across all 12 shapes). R55 Opt F (alternative
tile geometry) DEFERRED — aiter heuristic uniformly picks 256×256, no falsification reason
yet. R55 Opt B (32×32×64 MFMA) DEFERRED again. Floor 38/42, Mode 39/42, Stretch 42/42.
