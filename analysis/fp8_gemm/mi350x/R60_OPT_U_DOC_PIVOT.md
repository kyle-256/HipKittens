# R60 Opt U — Documentation Pivot Artifact

**Round**: R60
**Worker**: K-2 (cohort Opt U; NO GPU; analysis + writing only)
**Scope**: Formalize the project's structural ceiling and SC/MICRO publication framing.
**Companion artifact**: `R60K1_Y_INTEGRATION_FRAGMENT.json` (cohort K-1 Opt Y cohort-race monitoring; cross-referenced where relevant — this document does NOT claim PASS/FAIL on Opt Y).
**Inputs**:
- `R59_INTEGRATION_VERDICT.md` (current ceiling state; 4th 100% leaderboard round)
- `R59_INTEGRATION_MANIFEST.json` (canonical 42-cell mapping; 40 AITER + 2 HK)
- `R59_OPT_R_POLICY.md` (4-criterion mixed-protocol validity envelope)
- `R58_INTEGRATION_VERDICT.md` (predecessor; structural P-2 swap)
- `TODO.md` (round-sequence sanity check, R59+ closed-axis list)
- `AGENT_PROMPT.md` (round-discipline rules)
- `.claude/rules/benchmark-rules.md` (bench protocol: warmup=200, iters=500, trim=0.10)

---

## Section 1: Structural ceiling reached

The MXFP4 GEMM optimization project has reached the structural ceiling reachable under the current constraint set (R50D shim AS-IS reuse, no kernel rebuild, ITERS=500 default). The post-R59 production state is:

### Quantitative state (post-R59, byte-identical to R58 binary entries)

| Metric | Post-R59 value | Project-history extremum |
|---|---:|---|
| Strict 10-run VC (n_OK ≥ 8/10 AND wcf_max < 0.02 AND wcf_std < 0.01 AND fin_min ≥ 0.97) | **42/42** | Tied with R55, R56, R57 (4th 100% leaderboard round; non-consecutive — interrupted by R58) |
| WIN cells (≥100% comp vs aiter ASM baseline) | **41/42** | Tied with R57 (largest ever); recovered after R58 single drop |
| LOSE cells (<100% comp) | **1/42** (L8 4096x32768x128256 at 98.34%) | Smallest ever (R57 also had 1; R58 had 2) |
| AITER overrides (`.co` dlopen via R50D shim) | **40/42** | **Largest in project history** (HELD from R58) |
| HK kernel cells (in-tree HipKittens binary) | **2/42** | **Smallest in project history** (HELD from R58) |
| AITER cells bit-deterministic (wcf_max=0 across 10 INDEPENDENT seeds) | **40/40 = 100%** | All AITER cells perfect bit-determinism |
| Consecutive R50D shim AS-IS reuse rounds | **11** (R50D → R51 → R52 → R53 → R54 → R55 → R56 → R57 → R58 → R59 → R60-K2 expected 12th) | **Largest streak in project history** |
| 100% leaderboard rounds (cumulative) | **4** (R55, R56, R57, R59 — non-consecutive; interrupted by R58) | Project record |
| Cohort-race churn losses on UNCHANGED-binary cells in R59 | **0/42** | First round to demonstrate clean cohort-race repeatability of a near-gate VC pass on the worst-margin HK survivor |

(Source: `R59_INTEGRATION_VERDICT.md` headline + per-cohort tables; `R59_INTEGRATION_MANIFEST.json` `shapes_to_so_path` and `shapes_to_source` fields.)

### Argument that this is the structural ceiling under the current constraint set

**Every bounded-cost mechanism axis on the 3 attention cells (L1, L3, L8) has been definitively closed.** The closed-axis list (cumulative R45-R59) is:

1. **L1 `(4096, 32768, 14336)` AITER alt-tile space EXHAUSTED** — `R59_INTEGRATION_VERDICT.md` §"Axis-closure documentation": 96×640 SMOKE 87.42% (-12.52pp), 64×1024 SMOKE 66.05% (-33.89pp); combined with R56-R57 explorations, 256×256 R57J1_L1 is the strictly best AITER tile.
2. **L3 `(32768, 14336, 2048)` AITER alt-tile space EXHAUSTED** — `R59_INTEGRATION_VERDICT.md`: 96×640 SMOKE 63.19% (-37.30pp), 64×1024 SMOKE 91.73% (-8.76pp); combined with prior 128×256 (R58 P-3 -25.04pp), 192×256 (R57 H-2), 256×256 (R55 D-5B/1) closures, no AITER alt-tile remains.
3. **L8 `(4096, 32768, 128256)` AITER alt-tile space FULLY CLOSED** — R56-R57 carry-forward in `R59_INTEGRATION_VERDICT.md` §"R60+ closed-axis carry-forward": 128×512 -13.31pp, 192×256 -11.84pp, 224×256 -10.69pp, 96×640 + 64×1024 all DEAD; R52D2B AITER 256×256 at 98.34% is the aiter-internal ceiling.
4. **L8 HK 256×256 lgk2 v12 axis CLOSED by correctness** — `R58_INTEGRATION_VERDICT.md` Opt O: BOTH R40B and R37 HK fallbacks produce WRONG_OUTPUT (fin=0.78-0.80, wcf=0.07-0.13) for K=128256 because R39A TAIL_SCALE_CLAMP / R44A back-edge drain / R44D FINITE_GATE fixes were never ported into the K=128256 build (`project_mxfp4_correctness_17pct_wrong.md` "17% deterministic-wrong cohort").
5. **L1 ITERS=1000 one-off bump axis CLOSED by Opt R policy** — `R59_OPT_R_POLICY.md`: 4-criterion validity envelope formalized (bit-determinism, ≤0.5pp boundary proximity, dual-protocol documentation, no manifest merge); the bump option is rejected for the noise-edge use case.
6. **HK alt-tile space for 16384x4096x2048 + 32768x14336x2048 HK survivors** — R58 Opt P: 128×256 -3.68pp / -25.04pp; combined with R57 192×256 closure, alt-tile axis on K=2048 HK pool exhausted.

**Only one non-methodology mechanism axis remains open**: **Opt T — from-scratch HK kernel build for K=128256** with R39A/R44A/R44D fixes ported. Per `R59_INTEGRATION_VERDICT.md` §"R60+ direction suggestions" cost estimate: ~3 R-rounds, very low confidence; the only path to closing the L8 1.66pp gap. Per multiple round verdicts (R56, R57, R58, R59), the standing recommendation is **defer unless explicit user election** — no user election has been received as of R60.

### What "structural ceiling" means precisely

The ceiling is bounded simultaneously by:
- **Correctness**: HK 256×256 on K=128256 produces WRONG_OUTPUT, eliminating the only HK alt-source for L8.
- **Mechanism exhaustion**: every bounded-cost AITER alt-tile and every bounded-cost HK alt-tile on the 3 attention cells has been falsified.
- **Protocol stability**: ITERS=500 default delivers 40/40 AITER bit-determinism (no protocol-bump is required for the bit-deterministic surface; per Opt R policy A no protocol-bump is justified for the noise-edge surface either).
- **R50D shim invariance**: 11 consecutive rounds of AS-IS reuse demonstrates the shim is shape/tile/K-generic (proven across 256×256, 128×256, 96×640, 64×1024 tiles and across the full M/N/K dispatch range from K=2048 to K=128256 — `project_mxfp4_R52_round_win.md`, `project_mxfp4_R53_round_win.md`, `project_mxfp4_R54_round_win.md`).

The remaining 1 LOSE cell + 1 noise-edge cell + 1 near-gate HK survivor define the **residual surface** that R60+ must either (a) accept as documented residual (Opt U), or (b) attack via the only remaining bounded-cost mechanism axis (Opt T, very low confidence).

---

## Section 2: Residual surface (3 attention cells)

The post-R59 attention surface comprises 3 cells. Each is at its current state for distinct, fully-characterized reasons.

### 2.1 L8 — `(4096, 32768, 128256)` — R52D2B AITER 256×256 — 98.34% LOSE

| Field | Value | Source |
|---|---|---|
| pct_comp R59 reviewer p50 | **98.34%** (R58 was 98.28%; +0.06pp drift) | `R59_INTEGRATION_VERDICT.md` Opt R drift table; `R58_INTEGRATION_VERDICT.md` |
| n_OK / wcf_max / wcf_std / fin_min | 10/10 / 0.0 / 0.0 / 1.0 | `R59_INTEGRATION_10RUN.json` (AITER cell, perfect bit-determinism) |
| Verified-correct? | TRUE (strict-VC PASS) | Bit-deterministic |
| Classification | **LOSE** (gap = **1.66pp** vs 100% baseline) | <100% pct_comp |
| Source kernel | aiter `.co` `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim | `R59_INTEGRATION_MANIFEST.json` aiter_dispatch_template |
| Mechanism floor | **At aiter-internal ceiling for K/N=128256 / K-ratio cohort** (`project_mxfp4_R57_round_win.md`) | aiter heuristic table |

**Why L8 is at ceiling — closed axes**:
- AITER alt-tile space FULLY CLOSED: 128×512 (-13.31pp R56), 192×256 (-11.84pp R56), 224×256 (-10.69pp R57), 96×640 + 64×1024 (R59 carry-forward) — all DEAD. (`R59_INTEGRATION_VERDICT.md` §"R60+ closed-axis carry-forward".)
- HK 256×256 axis CLOSED by correctness: R58 Opt O confirmed BOTH R40B and R37 HK fallbacks WRONG_OUTPUT (fin=0.78-0.80, wcf=0.07-0.13) — the R39A/R44A/R44D correctness fixes were never ported into the K=128256 build (`project_mxfp4_correctness_17pct_wrong.md`).

**The only remaining mechanism axis**: **Opt T — from-scratch K=128256 HK build** with R39A TAIL_SCALE_CLAMP + R44A back-edge drain + R44D FINITE_GATE all ported. Cost: ~3 R-rounds. Confidence: very low (per `R59_INTEGRATION_VERDICT.md` standing rating). Only path to closing the 1.66pp gap.

**Disposition**: Production-correct (bit-deterministic, strict-VC PASS, single-cell LOSE on a 42-cell leaderboard). Deferred from R59 per `R59_INTEGRATION_VERDICT.md`: "Defer again unless explicit user election."

### 2.2 L1 — `(4096, 32768, 14336)` — R57J1_L1 AITER 256×256 — 100.08% WIN (oscillates around WIN line)

| Field | Value | Source |
|---|---|---|
| pct_comp R59 reviewer p50 | **100.08% WIN** | `R59_INTEGRATION_VERDICT.md` headline + Opt R drift table |
| pct_comp R58 reviewer p50 | 99.94% LOSE-edge | `R58_INTEGRATION_VERDICT.md` |
| pct_comp R57 reviewer p50 (ITERS=1000 one-off) | 100.04% WIN-edge | `R57_INTEGRATION_VERDICT.md` |
| Inter-sweep drift R58 → R59 | **+0.14pp** (LOSE-edge → WIN-edge) | Same UNCHANGED binary; same ITERS=500 protocol; only seed sweep differs (`R59_INTEGRATION_VERDICT.md` §"Secondary observation") |
| n_OK / wcf_max / wcf_std / fin_min (R59) | 10/10 / 0.0 / 0.0 / 1.0 | Bit-deterministic AITER cell (R59 Opt R drift table row "4096x32768x14336 R57J1_L1_AITER") |
| Source kernel | aiter `.co` 256×256 via R50D shim (R57J1 dispatch) | `R59_INTEGRATION_MANIFEST.json` |
| Classification governance | **Per-sweep value reflects production reading** (Opt R policy A) | `R59_OPT_R_POLICY.md` recommendation A |

**Why L1 is at ceiling — closed axes**:
- AITER alt-tile space EXHAUSTED: 96×640 SMOKE 87.42% (-12.52pp R59 J-3 V-1) and 64×1024 SMOKE 66.05% (-33.89pp R59 J-3 V-2); combined with R56 G-1 / R57 H-1 explorations, 256×256 is strictly best.
- ITERS=1000 one-off bump axis CLOSED: per `R59_OPT_R_POLICY.md` 4-criterion validity envelope, L1 satisfies criteria 1-2 (bit-deterministic, in [99.5%, 100.5%] band) but the cost-benefit analysis (Option A vs Option B/C in §R-3) rejects the bump because (i) the cell is bit-deterministic — 99.94% and 100.08% are functionally identical perf, only the WIN-line crosses; (ii) leaderboard consistency value > single-cell flip cosmetic; (iii) the mechanism is already characterized (R57 demonstrated the bump works) — there is no scientific reason to repeat it round-on-round.

**Disposition**: WIN per Opt R policy A. The cell oscillates ±0.15pp around the WIN-line under ITERS=500 between independent seed sweeps. Future per-sweep classifications are expected to flip occasionally. Per `R59_OPT_R_POLICY.md` §R-4: "L1 is bit-deterministic at 99.94% (wcf_max=0). The LOSE classification is purely measurement-noise, not a real perf gap."

### 2.3 L3 — `(32768, 14336, 2048)` — R40B HK 256×256 — 100.56% WIN + VC RECOVERED

| Field | Value | Source |
|---|---|---|
| pct_comp R59 reviewer p50 | **100.56% WIN** | `R59_INTEGRATION_VERDICT.md` per-cohort summary |
| pct_comp R58 reviewer p50 | 100.49% (numerically WIN but VC-flipped) | `R58_INTEGRATION_VERDICT.md` |
| n_OK R59 / R58 | **10/10** / 9/10 | R59 RECOVERED (`R59_INTEGRATION_VERDICT.md` §"Cohort-race repeatability finding") |
| fin_min R59 / R58 | **0.9880** / 0.9108 | R58 dropped below 0.97 gate; R59 recovered to 0.018 above gate |
| wcf_max R59 / R58 | 0.0120 / 0.0111 | Both within strict-VC tolerance |
| Verified-correct R59 / R58 | TRUE / FALSE | R58 was the lone R58 VC drop; R59 recovered under same ITERS=500 on UNCHANGED binary |
| Source kernel | HK in-tree `tk_mxfp4_gluon_cpp_n14336_k2048_ts_gm6_v12_dc_pfoff4_R40B_safe.so` (R40B build) | `R59_INTEGRATION_MANIFEST.json` shapes_to_so_path |
| Mechanism | R44D FINITE_GATE 0.97 governs cohort-race finiteness; this cell sits ~0.018 above the gate at fin_min=0.988 with ~10-20% per-sweep tail-draw probability under ITERS=500 | `project_mxfp4_finite_gate_cohort_race.md`, `project_mxfp4_R44D_gate_097.md`, `R59_INTEGRATION_VERDICT.md` §"Implication for R60+" |

**Why L3 is at ceiling — closed axes**:
- AITER alt-tile space EXHAUSTED: 128×256 -25.04pp (R58 P-3 catastrophic), 192×256 (R57 H-2 DEAD), 256×256 (R55 D-5B/1 already best), 96×640 -37.30pp (R59 S-1), 64×1024 -8.76pp (R59 S-2) — all closed.
- HK kernel rebuild axis (Opt W — relax R44D FINITE_GATE 0.97 → 0.95) DEFERRED per `R59_INTEGRATION_VERDICT.md` §"R60+ direction suggestions": breaks the R50D AS-IS streak; only justified if Opt Y monitoring shows L3 tail-draw repeats (cross-reference `R60K1_Y_INTEGRATION_FRAGMENT.json` for R60 Opt Y outcome).

**Disposition**: WIN + VC RECOVERED per R59. The R59 fresh INDEPENDENT seed sweep at the same ITERS=500 protocol on the same R40B HK binary returned n_OK=10/10 fin_min=0.988 PASS — empirically confirming the R58 single-VC drop was a single-sweep cohort-race tail-draw artifact, NOT an intrinsic surface (`R59_INTEGRATION_VERDICT.md` §"Cohort-race repeatability finding"). Future per-sweep tail-draw probability is estimated at ~10-20% under ITERS=500 (R58 caught one; R59 did not; R60 K-1 cohort is the next data point).

### 2.4 Summary table — residual surface

| Cell | Source | R59 pct_comp | Strict VC | Classification | Governing axis | Remaining mechanism |
|---|---|---:|---|---|---|---|
| L8 (4096,32768,128256) | R52D2B AITER 256×256 | 98.34% | PASS | LOSE (1.66pp gap) | aiter-internal ceiling; HK axis closed by correctness | Opt T (~3 R-rounds, very low confidence) |
| L1 (4096,32768,14336) | R57J1_L1 AITER 256×256 | 100.08% | PASS (bit-det) | WIN (oscillates ±0.15pp) | Noise-edge under ITERS=500 | None bounded-cost; per Opt R policy A |
| L3 (32768,14336,2048) | R40B HK 256×256 | 100.56% | PASS (RECOVERED) | WIN (near-gate fin_min=0.988) | Cohort-race tail-draw under R44D gate 0.97 | Opt W (~1-2 R-rounds, breaks R50D streak); cohort K-1 monitoring |

---

## Section 3: SC/MICRO publication claims

This section enumerates what the MXFP4 work establishes for an SC/MICRO submission and what it explicitly does NOT claim.

### 3.1 Headline claim

> **42/42 strict-VC verified-correct WIN ≥100% comp on 41 of 42 production GEMM shapes vs aiter ASM (`competitor_tflops` in `bench_all_42.py`) and Gluon LLIR baselines on MI355X (gfx950)**, under a 10-run independent-seed strict-VC validation gate (`n_OK ≥ 8/10 AND wcf_max < 0.02 AND wcf_std < 0.01 AND fin_min ≥ 0.97`).

**Decomposition** (from `R59_INTEGRATION_VERDICT.md` headline):
- 42/42 strict 10-run VC (4th 100% leaderboard round in project history)
- 41/42 WIN cells (≥100% comp); only 1 LOSE cell (L8 4096x32768x128256 at 98.34%, structurally floored at the aiter-internal ceiling)
- 40/42 cells served via aiter `.co` dlopen (R50D shim AS-IS); 2/42 served via in-tree HK kernel R40B
- AITER bit-deterministic share: 40/40 AITER cells achieve wcf_max=0 across 10 INDEPENDENT seeds — perfect bit-determinism on every AITER cell

**Coverage**: 42 production shapes spanning M ∈ {4096, 6144, 14336, 16384, 28672, 32768, 128256}, N ∈ {4096, 6144, 14336, 28672, 32768, 128256}, K ∈ {2048, 3072, 4096, 6144, 7168, 8192, 14336, 16384, 28672, 32768, 128256}. (Source: `R59_INTEGRATION_MANIFEST.json` shapes_to_source dict.)

### 3.2 Methodology claims

(a) **10-run @ 80% INDEPENDENT-seed strict-VC gate**, mandated since R45 (`project_mxfp4_R45_cohort_tail_draw.md`): 10 independent seeds per cell, 80% pass threshold, 4-component pass criteria. This protocol catches cohort-race tail-draws that 5-run measurements miss (R47C demonstrated a 5/5 → 5/10 flip caught by the 10-run protocol — `project_mxfp4_R47_round_dead.md`).

(b) **4-criterion Opt R mixed-protocol validity envelope** (`R59_OPT_R_POLICY.md` §R-4). Any one-off ITERS bump must satisfy ALL of: (1) bit-determinism (wcf_max=0 at ITERS=500); (2) boundary proximity (≤0.5pp from WIN-line or fin_min gate); (3) documented in round verdict with both numbers; (4) NOT merged into manifest as default. The envelope formalizes the protocol-noise-vs-mechanism distinction and prevents protocol band-aids from masking structural surfaces.

(c) **Cohort-race repeatability test** established R58 → R59 → R60. R58 caught a lone L3 fin_min=0.911 tail-draw on UNCHANGED R40B HK binary; R59 fresh INDEPENDENT seed sweep recovered to fin_min=0.988 (n_OK=10/10) on the SAME UNCHANGED binary at the SAME ITERS=500 protocol; R60 K-1 cohort is the third independent measurement (cross-reference `R60K1_Y_INTEGRATION_FRAGMENT.json`). The empirical per-sweep tail-draw rate on the worst-margin HK survivor is bounded ≤ 1/3 sweeps under ITERS=500 if K-1 PASSES; > 2/3 sweeps if K-1 FAILS.

(d) **Bench parameters mandate** (`.claude/rules/benchmark-rules.md`): warmup=200, iters=500, trim=0.10 (10% from each end). GPU isolation via `HIP_VISIBLE_DEVICES` on idle GPUs verified pre-launch via `rocm-smi --showuse`.

(e) **Dispatch-by-shape methodology**: 40 of 42 cells served via aiter `.co` dlopen (R50D shim; `project_mxfp4_R50D_aiter_co_dlopen_win.md`); 2 cells served via in-tree HK kernel. The split is empirically optimal — neither single-kernel HK nor single-kernel aiter strictly dominates across the 42-cell production surface. Per-cell mechanism: aiter `.co` is structurally bit-deterministic (`project_mxfp4_R51_round_win.md`); HK kernel R40B with R44D FINITE_GATE 0.97 captures the K=2048 N=14336 cohort-race surface that aiter cannot reach (lower aiter alt-tile efficiencies for this specific M-N-K tuple).

### 3.3 Mechanism claims (per major round-streak)

These are durable mechanism findings that survive across 11+ consecutive rounds of R50D AS-IS reuse, confirmed by independent seed sweeps and (in several cases) by ISA-level disassembly.

- **R44A back-edge drain for K=28672** (`project_mxfp4_R44A_backedge_drain.md`): inserting `s_waitcnt vmcnt(0)` at the last C++ statement of the TAIL_SPLIT body delivered the first ever VC for `(16384, 4096, 28672)`. Memory-ordering finding, not compiler artifact.
- **R44D FINITE_GATE 0.97 for cohort-race-bound HK cells** (`project_mxfp4_R44D_gate_097.md`): 3-phase falsification (Jaccard input-reuse, 10-run independent seed, cross-validation) supports gate relaxation 0.98 → 0.97 as a mechanism-grounded change, not a measurement band-aid. Delivered +7 of R44's +8 net VC.
- **R50D aiter-binary `.co` dlopen shim** (`project_mxfp4_R50D_aiter_co_dlopen_win.md`): proven shape-generic, tile-generic (256×256, 128×256, 96×640, 64×1024 all served from same shim AS-IS — `project_mxfp4_R53_round_win.md`), K-generic (K=2048 to K=128256 — `project_mxfp4_R52_round_win.md`), and grid-size-generic (full gdx range from R51-R52 sweeps). 11 consecutive rounds of AS-IS reuse with zero rebuild.
- **R55-R57 promote chain (HK→AITER swap discipline)** (`project_mxfp4_R55_round_win.md`, `project_mxfp4_R56_round_win.md`, `project_mxfp4_R57_round_win.md`): systematic per-cell HK→AITER swap with strict gate (10-run @ 80% n_OK + bit-determinism check). Delivered +6 NET VC (R55 = first 100% leaderboard) + +6 NET WIN cells (R56 perf claw-back largest ever) + +1 NET WIN cell (R57 first protocol-only WIN via Opt J ITERS=500→1000 boundary-bump).
- **R58 P-2 structural HK→AITER 128×256 swap** (`project_mxfp4_R58_round_win.md`): single-cell PROMOTE on `(16384, 4096, 3072)` HK R40B 103.25% → AITER 128×256 107.62% (+4.37pp). Demonstrates that the swap discipline continues to find PROMOTEs even in the late-stage when the decider rates the axis "low confidence". AITER bit-deterministic share 39 → 40 (largest in project history); HK pool 3 → 2 (smallest in project history).
- **R59 cohort-race repeatability validation** (`R59_INTEGRATION_VERDICT.md` §"Cohort-race repeatability finding"): empirically validates that R58's single-VC drop was a single-sweep cohort-race tail-draw artifact, NOT a kernel regression. Establishes the R60+ Opt Y monitoring axis.

### 3.4 What we CANNOT claim

(i) **Closing the L8 1.66pp gap**: Opt T (from-scratch K=128256 HK build) is deferred; not part of the publication scope. The L8 LOSE classification stands as a documented residual at the aiter-internal ceiling.

(ii) **Deterministic L1 WIN-line crossing across all sweeps**: per Opt R policy A (`R59_OPT_R_POLICY.md` §R-4), L1's WIN/LOSE classification is bit-deterministic measurement-noise around the 100% line. The 100.08% R59 reading is the per-sweep production value; future per-sweep readings may oscillate ±0.15pp. The publication should report L1 as "bit-deterministic at the WIN-line; classification per-sweep noise per Opt R policy."

(iii) **Deterministic L3 strict-VC PASS across all sweeps**: empirically ~10-20% per-sweep tail-draw probability under ITERS=500 on R40B HK binary. R58 caught one; R59 did not; R60 K-1 is the third sample. Publication should report L3 as "near-gate cohort-race surface; recovered to PASS on R59 fresh seed sweep; future tail-draw events expected at ≤ 1/3 sweeps if R60 K-1 PASSES (cross-reference `R60K1_Y_INTEGRATION_FRAGMENT.json`)."

(iv) **A single-protocol uniform leaderboard across all alternative ITERS values**: Opt R policy explicitly documents that ITERS=500 is the leaderboard default, with the L1 noise-edge cell footnoted. The publication should NOT claim that 42/42 holds under arbitrary ITERS choices — only under the documented ITERS=500 protocol with the 4-criterion Opt R validity envelope for any future one-off bump.

(v) **Aspirational "5084 TFLOPS gluon ASM"** (revoked 2026-04-17 per `.claude/rules/benchmark-rules.md`): the inlined `kernel_mxfp4_asm_inline.{cpp,h}` produces INCORRECT output (SNR -1.31 dB vs torch reference); the 5084-5258 TFLOPS reading is meaningless. Only the `competitor_tflops` (aiter ASM via Python dispatcher) is the comparator.

---

## Section 4: R61-R65 axis classification

For each remaining option, classify as **Closed** (do not propose), **Defer** (low priority, only on user election), or **Elect** (low-cost methodology rounds appropriate for R61+).

### 4.1 Closed (do not propose)

All R45-R59 closed-axis carry-forward (per `R59_INTEGRATION_VERDICT.md` §"R60+ closed-axis carry-forward" + the cumulative list in this document §1):

- L1 `(4096, 32768, 14336)` AITER alt-tile space EXHAUSTED — 96×640, 64×1024 closed in R59; 256×256 R57J1_L1 is best.
- L3 `(32768, 14336, 2048)` AITER alt-tile space EXHAUSTED — 96×640, 64×1024, 128×256, 192×256 all DEAD.
- L8 `(4096, 32768, 128256)` AITER alt-tile space FULLY CLOSED — 128×512, 192×256, 224×256, 96×640, 64×1024 all DEAD.
- L8 HK 256×256 lgk2 v12 axis CLOSED by correctness (R58 Opt O, both R40B and R37 fallbacks WRONG_OUTPUT for K=128256).
- L1 ITERS=1000 one-off bump axis CLOSED by Opt R policy 4-criterion validity envelope.
- 128×256 alt-tile for 16384x4096x2048 HK survivor and 32768x14336x2048 HK survivor — R58 P-1 -3.68pp / P-3 -25.04pp.
- All R45-R49 closed axes (5-DEAD-of-7 wall, fully documented in `project_mxfp4_R45-R49_*.md` series): external/internal fence positioning, MFMA↔ds_read interleave depth, M0 fresh-set propagation to FUSED+TS, asm-block split, PF_MPT depth-knob.
- All R50A-R50C closed axes (interleave axis, perf claw-back saturation).
- ANY further attempt to ITERS=1000 the L1 noise-edge cell as a one-off — formally precluded by Opt R policy.

### 4.2 Defer (only on explicit user election)

| Axis | Cost | Confidence | Rationale for deferral |
|---|---|---|---|
| **Opt T** — L8 from-scratch HK kernel build for K=128256 (port R39A TAIL_SCALE_CLAMP + R44A back-edge drain + R44D FINITE_GATE) | ~3 R-rounds | Very low | Only path to closing the L8 1.66pp gap; standing recommendation across R56-R59 verdicts is "defer unless explicit user election"; no election received |
| **Opt W** — HK kernel rebuild with R44D FINITE_GATE 0.97 → 0.95 to recover L3 cohort-race tail-draw probability | ~1-2 R-rounds | Low-medium | Breaks the 11-round R50D AS-IS streak; only justified if R60 Opt Y FAILS (i.e. shows L3 tail-draw repeats — see `R60K1_Y_INTEGRATION_FRAGMENT.json` for the R60 outcome); R59 recovery of L3 weakens this option's confidence |

### 4.3 Elect (low-cost methodology rounds)

| Axis | Cost | Output | Rationale |
|---|---|---|---|
| **Opt Y₂, Opt Y₃, ...** — Continuing cohort-race surface monitoring (re-bench UNCHANGED R59 manifest under fresh INDEPENDENT seed sweeps) | ~17 min wall per round | Longitudinal cohort-race repeatability dataset (per-sweep tail-draw rate on L1 noise-edge + L3 near-gate) | Cheap, zero risk, builds publication-grade longitudinal data; pairs naturally with continued R50D AS-IS reuse |
| **Opt U-2, Opt U-3, ...** — Continued documentation pivot (publication outline, related-work survey, methods section, results tables, limitations section) | NO GPU; ~10-20 min per round | SC/MICRO submission artifacts | All bounded-cost mechanism axes are closed; the highest-value remaining work is publication-prep |
| **Opt Z** — Per-shape decomposition table for publication appendix (one row per cell: tried axes, closed axes, why current source is best) | NO GPU; ~30-60 min | Compresses 17 rounds of round-verdicts into one publication-grade table | High-value publication artifact; consolidates evidence base |
| **Opt U-housekeeping** — Verify TODO.md and AGENT_PROMPT.md aligned with documentation-pivot framing; update closed-axis carry-forward; archive deprecated decider-plan templates | NO GPU; ~10 min | Project-state hygiene | Low-cost, prevents drift between artifacts and round verdicts |

### 4.4 Decision tree for R61

Branching rule (per `R60_DECIDER_PLAN.md` §5):

- **If R60 Opt Y PASSES** (cross-reference `R60K1_Y_INTEGRATION_FRAGMENT.json`): R61 = Opt U-2 (publication outline) + Opt Y-2 (4th sweep monitor) + Opt Z (per-shape decomposition table). Project enters publication-prep phase.
- **If R60 Opt Y FAILS**: R61 = Opt W (HK kernel rebuild, FINITE_GATE 0.97 → 0.95) becomes priority; Opt U-2 continues in parallel on a no-GPU worker; Opt Y-2 runs alongside as cross-check. Opt T remains DEFERRED at low priority.

---

## Section 5: Project state snapshot (table form)

### 5.1 Cross-round structural metrics (R44 → R59)

| Dimension | Pre-R44 (R43, 2026-04-19) | R44 | R55 (1st 100%) | R56 | R57 (3-in-row) | R58 | R59 (recovery) | Δ R44 → R59 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict 10-run VC | 27/42 | 35/42 | 42/42 | 42/42 | 42/42 | 41/42 | **42/42** | **+15 (+55.6% from pre-R44 wall)** |
| WIN cells (≥100% comp) | ? (not separately tracked pre-R55) | ? | 34/42 | 40/42 | 41/42 | 40/42 | **41/42** | trace from R55: +7 (34→41) |
| LOSE cells | ? | ? | 8/42 | 2/42 | 1/42 | 2/42 | **1/42** | trace from R55: -7 (8→1) |
| AITER bit-det share | 0/42 | 0/42 | 38/42 | 39/42 | 39/42 | 40/42 | **40/42** | **+40 (full transition from HK-only to aiter-dominant)** |
| HK kernel cells | 42/42 | 42/42 (no AITER swaps yet — R50D not invented) | 4/42 | 3/42 | 3/42 | 2/42 | **2/42** | **-40** (matched inverse of AITER expansion; mostly via R50D shim swaps R51-R58) |
| R50D AS-IS reuse streak | 0 | 0 (R44 pre-dates R50D shim) | 7 | 8 | 9 | 10 | **11** | **+11** |
| 100% leaderboard rounds (cumulative) | 0 | 0 | 1 | 2 | 3 (first 3-in-a-row) | 3 | **4** (non-consecutive) | **+4** |
| Closed axes (cumulative, approximate) | ~0 | ~6 (R44 closed crash mechanisms) | ~30 (R45-R55 fence/interleave/depth) | ~33 | ~38 | ~42 | **~46** | **+46** |
| Cohort-race attention surface (HK survivors) | n/a (HK-only) | n/a | 4 | 3 | 3 | 1 | **1** (32768x14336x2048; near-gate) | trace from R55: -3 |

(Sources: `R59_INTEGRATION_VERDICT.md` headline tables; `R58_INTEGRATION_VERDICT.md`; `project_mxfp4_R55_round_win.md`; `project_mxfp4_R56_round_win.md`; `project_mxfp4_R57_round_win.md`; `project_mxfp4_R58_round_win.md`; `project_mxfp4_R59_round_win.md`; `TODO.md` round-sequence sanity check.)

**?** = uncertain at this granularity; pre-R55 rounds tracked strict-VC count primarily, not WIN/LOSE classification (the WIN-cell metric only became meaningful once strict-VC ≥ ~36/42 made WIN-rate the relevant gating dimension).

### 5.2 Per-cell snapshot (R59 canonical manifest, 42 cells)

(Source: `R59_INTEGRATION_MANIFEST.json` shapes_to_source + R59 Opt R drift table p50 values.)

| # | shape (MxNxK) | source | R59 pct_comp | source category | classification |
|---:|---|---|---:|---|---|
| 1 | 4096x4096x8192 | R55D5A_3_AITER | ~120.49% | AITER | WIN |
| 2 | 4096x4096x16384 | R54D4B_1_AITER | ~113.26% | AITER | WIN |
| 3 | 4096x4096x32768 | R52D2C_AITER | ~108.20% | AITER | WIN |
| 4 | 4096x6144x32768 | R53D3A_3_AITER | ~130.58% | AITER | WIN |
| 5 | 4096x14336x8192 | R54D4B_2_AITER | ~115.85% | AITER | WIN |
| 6 | 4096x14336x16384 | R54D4A_1_AITER | ~106.93% | AITER | WIN |
| 7 | 4096x28672x32768 | R52D2A_AITER | ~102.50% | AITER | WIN |
| 8 | 4096x32768x4096 | R54E2_1_AITER | ~105.32% | AITER | WIN |
| 9 | 4096x32768x6144 | R54E2_2_AITER | ~105.83% | AITER | WIN |
| 10 | **4096x32768x14336** (L1) | **R57J1_L1_AITER** | **100.08%** | **AITER** | **WIN (noise-edge)** |
| 11 | 4096x32768x28672 | R50D_AITER | ~101.17% | AITER | WIN |
| 12 | **4096x32768x128256** (L8) | **R52D2B_AITER** | **98.34%** | **AITER** | **LOSE (1.66pp gap, structurally floored)** |
| 13 | 4096x128256x32768 | R56G4_C1_AITER | ~178.51% | AITER | WIN |
| 14 | 6144x4096x8192 | R54D4A_3_AITER | ~117.97% | AITER | WIN |
| 15 | 6144x4096x16384 | R53D3A_2_AITER | ~105.41% | AITER | WIN |
| 16 | 6144x32768x4096 | R55E4_2_AITER | ~106.10% | AITER | WIN |
| 17 | 14336x4096x32768 | R51D1_AITER | ~104.16% | AITER | WIN |
| 18 | 14336x32768x4096 | R56G2_L5_AITER | ~103.67% | AITER | WIN |
| 19 | **16384x4096x2048** | **R40B (HK)** | ~108.52% | **HK** | **WIN (HOLD VC)** |
| 20 | 16384x4096x3072 | R58P2_AITER | ~107.62% | AITER | WIN (R58 PROMOTE) |
| 21 | 16384x4096x4096 | R55D5A_1_AITER | ~113.20% | AITER | WIN |
| 22 | 16384x4096x6144 | R54E2_3_AITER | ~115.66% | AITER | WIN |
| 23 | 16384x4096x7168 | R54D4B_3_AITER | ~113.35% | AITER | WIN |
| 24 | 16384x4096x14336 | R56G1_L6_AITER | ~107.08% | AITER | WIN |
| 25 | 16384x4096x28672 | R51D2_AITER | ~102.98% | AITER | WIN |
| 26 | 16384x6144x2048 | R55D5A_2_AITER | ~111.87% | AITER | WIN |
| 27 | 16384x6144x4096 | R55E4_1_AITER | ~114.51% | AITER | WIN |
| 28 | 16384x14336x2048 | R55E3_1_AITER | ~106.95% | AITER | WIN |
| 29 | 16384x14336x4096 | R55E3_2_AITER | ~108.53% | AITER | WIN |
| 30 | 16384x28672x2048 | R55E3_3_AITER | ~102.18% | AITER | WIN |
| 31 | 16384x28672x4096 | R55E3_4_AITER | ~104.32% | AITER | WIN |
| 32 | 28672x4096x8192 | R54E1_3_AITER | ~105.79% | AITER | WIN |
| 33 | 28672x4096x16384 | R51D3_AITER | ~103.91% | AITER | WIN |
| 34 | 28672x32768x4096 | R56G2_L4_AITER | ~102.72% | AITER | WIN |
| 35 | 32768x4096x2048 | R54E1_1_AITER | ~106.96% | AITER | WIN |
| 36 | 32768x4096x3072 | R54E1_2_AITER | ~118.59% | AITER | WIN |
| 37 | 32768x4096x7168 | R54D4A_2_AITER | ~106.07% | AITER | WIN |
| 38 | 32768x4096x14336 | R56G1_L2_AITER | ~103.97% | AITER | WIN |
| 39 | 32768x6144x2048 | R55D5B_3_AITER | ~106.67% | AITER | WIN |
| 40 | **32768x14336x2048** (L3) | **R40B (HK)** | **100.56%** | **HK** | **WIN (RECOVERED, near-gate fin_min=0.988)** |
| 41 | 32768x28672x2048 | R55D5B_2_AITER | ~103.20% | AITER | WIN |
| 42 | 128256x32768x4096 | R56G2_L3_AITER | ~102.61% | AITER | WIN |

**Summary**: 40 AITER + 2 HK; 41 WIN + 1 LOSE; 42 strict-VC PASS (cells 19, 40 are the 2 HK survivors; cells 10, 12, 40 are the 3 attention cells per §2). pct_comp values are R59 reviewer p50 readings drawn from `R59_OPT_R_POLICY.md` Opt R drift table column "R58 @ 500" (R58 baseline) cross-referenced with R59 narrative (drift typically ≤±1.4pp per cell on UNCHANGED binaries — `R59_INTEGRATION_VERDICT.md` headline "Mean perf drift on 42 cells: -0.015pp").

### 5.3 Headline metrics one-screen

```
Round:                  R59 (post-R59, R60 K-2 documentation pivot)
Date:                   2026-04-20
Manifest:               R59_INTEGRATION_MANIFEST.json (byte-identical to R58 binary entries)
Strict 10-run VC:       42/42 (4th 100% leaderboard round; non-consecutive)
WIN cells:              41/42 (largest tied with R57)
LOSE cells:             1/42 (L8 4096x32768x128256 at 98.34%, structurally floored)
AITER overrides:        40/42 (largest in project history, HELD)
HK kernel cells:        2/42 (smallest in project history, HELD)
AITER bit-determinism:  40/40 = 100% (perfect)
R50D AS-IS streak:      11 consecutive rounds (R50D → R59)
Cohort-race churn R59:  0 lost VC on 42 unchanged-binary cells (cleanest in project history)
Mean perf drift R59:    -0.015pp/cell across 42 cells (statistically zero)
```

---

## Section 6: Round-streak history (R43 → R59)

### 6.1 Verbatim from `TODO.md` §"Round sequence sanity check (last 17 rounds)"

```
- R43: DEAD (3 axes)
- R44: WIN +8 (27 → 35/42)
- R45-R49: 5 DEAD rounds in a row
- R50: WIN +1 (35 → 36/42, aiter `.co` dlopen first PoC)
- R51: WIN +1 strict (30 → 31/42) + 2 perf claw-backs
- R52: WIN +5 strict (31 → 36/42) + 3 perf claw-backs
- R53: PARTIAL WIN +2 NET VC rescues -5 cohort net -3 strict (36 → 33/42) + 6 perf claw-backs
- R54: WIN +6 NET VC rescues -3 cohort net +3 strict (33 → 36/42) + 6 perf claw-backs +17-28pp
- R55: WIN +6 NET VC -0 cohort +6 strict (36 → 42/42) + 5 perf claw-backs +1.81-20.99pp;
       FIRST 100% LEADERBOARD ROUND IN PROJECT HISTORY
- R56: PERF-CLAW-BACK WIN +6 NET WIN CELLS (34→40 LARGEST EVER) -0 cohort +0 strict VC
       (42/42 HELD) + 7 perf claw-backs +14-81pp; 39/42 cells bit-deterministic AITER;
       7/7 PROMOTE / 1 ACCEPT_FALLBACK / 0 DEAD; 8th CONSECUTIVE R50D AS-IS REUSE;
       2ND CONSECUTIVE 100% LEADERBOARD ROUND
- R57: WIN +1 NET WIN CELL via Opt J ITERS=1000 (40→41) -0 cohort +0 strict VC (42/42 HELD)
       + 0 perf claw-backs (4 ACCEPT_FALLBACK closures); 39/42 cells bit-deterministic AITER;
       1/5 PROMOTE / 4 ACCEPT_FALLBACK / 0 DEAD; 9th CONSECUTIVE R50D AS-IS REUSE;
       3RD CONSECUTIVE 100% LEADERBOARD ROUND (FIRST 3-IN-A-ROW IN PROJECT HISTORY)
- R58: STRUCTURAL WIN +1 PROMOTE P-2 HK→AITER 128×256 swap +4.37pp -1 cohort tail-draw
       on UNCHANGED L3 + ITERS=500 revert net -1 strict VC (42→41) -1 WIN (41→40);
       40/42 cells bit-deterministic AITER (LARGEST EVER); HK pool 3→2 (SMALLEST EVER);
       1/5 PROMOTE / 4 ACCEPT_FALLBACK / 0 DEAD; 10th CONSECUTIVE R50D AS-IS REUSE;
       3-IN-A-ROW STREAK ENDS
- R59: RECOVERY ROUND +1 VC RECOVERY (41→42) +1 WIN RECOVERY (40→41) + 4 alt-tile axes
       CLOSED + Opt R policy artifact + 11th CONSECUTIVE R50D AS-IS REUSE; 40/42 AITER
       bit-det HELD (LARGEST EVER); HK pool 2 HELD (SMALLEST EVER); 0 PROMOTE / 4
       SMOKE_DEAD / 1 POLICY_ONLY; 0 cohort-race churn on 42 unchanged-binary cells;
       4TH 100% LEADERBOARD ROUND IN PROJECT HISTORY (NON-CONSECUTIVE)
       ← cohort-race repeatability test PASSED on L3; L1 noise-edge crossed back to WIN
       under same ITERS=500 protocol; alt-tile space EXHAUSTED on both L1 and L3;
       only Opt T remains for L8 1.66pp gap
```

### 6.2 One-line interpretation of the sequence

The 17-round arc tells a coherent four-act story:

**Act I (R43-R49) — The 5-DEAD-of-7 wall.** R43-R49 saw 5 consecutive DEAD rounds (R45, R46, R47, R48, R49 all net 0 VC) interrupted only by R44's +8 VC breakthrough (R44A back-edge drain + R44D FINITE_GATE 0.97). The DEAD streak documented the closure of every compiler-driven optimization axis (fence positioning internal/external, MFMA↔ds_read interleave, M0 fresh-set transfer to FUSED+TS, asm-block split, PF_MPT depth-knob) and exhausted the in-tree HK kernel mechanism space.

**Act II (R50) — The R50D dlopen breakthrough.** R50D `aiter .co` dlopen via `hipModuleLoadData` shim solved the previously perma-CRASH `(4096, 32768, 28672)` cell with 10/10 INDEPENDENT seed PASS at 100.31% comp. First non-DEAD round in 4 weeks; first +1 VC since R44; established the shape-generic dispatch-by-aiter-binary pattern that drove the next 9 rounds.

**Act III (R51-R55) — The 5-WIN streak culminating in the first 100% leaderboard.** R51 (+1 VC + 2 perf claw-backs), R52 (+5 VC), R53 (+2 net rescue, -3 cohort tail-draw, partial), R54 (+6 net rescue, +3 net strict), R55 (+6 NET VC = 36 → 42/42, **FIRST 100% LEADERBOARD IN PROJECT HISTORY**). R55 also delivered +35.83pp aggregate perf claw-back across 5 D-5 PROMOTEs and cut the cohort-race surface from 15 HK cells to 4. The R50D shim was reused AS-IS for 7 consecutive rounds across this arc.

**Act IV (R56-R57) — The 100% leaderboard era and the first 3-in-a-row.** R56 (+6 NET WIN cells = 34 → 40, +198.14pp aggregate perf, 2.2× Stretch target; 7/7 PROMOTE; G-1+G-2 6/6 64×1024→256×256 swap; G-4 C1 first kept-HK cell HK→AITER swap success +80.92pp on L7), R57 (+1 NET WIN cell via Opt J ITERS=1000 noise-edge; first protocol-only WIN; 4 ACCEPT_FALLBACK closures including 192×256 axis closed for HK pool and 224×256 closure completing L8 aiter alt-tile axis — first 3-in-a-row 100% leaderboard streak). HK pool shrank from 4 → 3 across these rounds; AITER bit-deterministic share grew from 38 → 39.

**Coda (R58-R59) — The structural P-2 swap, the lone VC drop, and the recovery.** R58 delivered a +1 PROMOTE STRUCTURAL HK→AITER 128×256 swap on `(16384, 4096, 3072)` (+4.37pp); AITER bit-deterministic share 39 → 40 (largest ever); HK pool 3 → 2 (smallest ever); 10th consecutive R50D AS-IS reuse. The 3-in-a-row 100% streak ended via a single ITERS=500-revert cohort-race tail-draw on the UNCHANGED L3 R40B HK binary (predicted by R57 fin_min=0.9847 + R58 Opt N analysis as worst-margin survivor). R59 fresh INDEPENDENT seed sweep on the same UNCHANGED binary at the same ITERS=500 protocol RECOVERED the L3 VC (n_OK=10/10 fin_min=0.988); L1 noise-edge ALSO crossed back to WIN at 100.08% under the same protocol; 4 alt-tile axes CLOSED (96×640 + 64×1024 on both L1 and L3); Opt R policy delivered with 4-criterion validity envelope; **4th 100% leaderboard round in project history (non-consecutive — interrupted by R58)**. R59 is the **first round to demonstrate clean cohort-race repeatability of a near-gate VC pass on the worst-margin HK survivor without any binary modification.**

The sequence shows the project trajectory: **HK-only mechanism exhaustion → aiter dispatch breakthrough → systematic HK→AITER swap discipline → 100% leaderboard era → structural ceiling reached → empirical validation of cohort-race repeatability.** R60 K-2 (this document) formalizes the structural ceiling as the SC/MICRO publication framing; R60 K-1 (Opt Y) provides the third independent cohort-race repeatability data point (cross-reference `R60K1_Y_INTEGRATION_FRAGMENT.json`).

---

## Appendix A: Open questions for the R61 author / SC/MICRO submission preparer

The following items are NOT inconsistencies in the R59 record, but represent decisions the publication preparer should resolve before submission:

1. **Cohort labels for the per-cell decomposition table** (Opt Z work). The 42 cells span Llama-3, DeepSeek, GPT-OSS, Mixtral, and other production model shapes. The current `R59_INTEGRATION_MANIFEST.json` does NOT record cohort labels per cell; only M-N-K. The publication appendix should add a `model_cohort` annotation per cell (or a separate cohort-mapping table).
2. **Mean pct_comp aggregate by cohort** (Opt U-2 publication outline). A single "mean pct_comp" metric is misleading without cohort decomposition (the 178.51% C1 cell on L7 dominates a naive mean). Recommend reporting (a) median pct_comp per cohort with min/max per cohort; (b) WIN-rate per cohort.
3. **Treatment of the L1 oscillation in publication results table**. Three options: (a) report R59 reading 100.08% WIN; (b) report a per-sweep range (R57 100.04% / R58 99.94% / R59 100.08%) with the Opt R policy footnote; (c) report mean across R57+R58+R59 sweeps (~100.02%) with confidence interval. Option (b) is most honest but requires reader-side interpretation; option (a) is cleanest but loses information. Recommend option (b) with a clear explanatory footnote.
4. **Decision on Opt T inclusion in the publication scope**. Currently DEFERRED. If the SC/MICRO submission deadline accommodates ~3 R-rounds of L8 from-scratch HK build work, Opt T could potentially close the L8 1.66pp gap; the publication would then claim 42/42 WIN. If deadline does not accommodate, the publication should explicitly document L8 as a residual at the aiter-internal ceiling with the structural reasons enumerated in §2.1. **This is a user-election decision, not an R61 author decision.**
5. **Reference comparator clarification**. Per `.claude/rules/benchmark-rules.md`, only `competitor_tflops` (aiter ASM via Python dispatcher) is the comparator; the inlined "5084 TFLOPS" gluon ASM reference was REVOKED 2026-04-17 (produces incorrect output, SNR -1.31 dB). Publication should explicitly state the comparator as "aiter ASM (`competitor_tflops` baseline) + Gluon LLIR baselines" and disclaim the revoked inline ASM number to prevent reviewer confusion.
6. **Opt Y₂+ longitudinal dataset format**. R60 K-1 (and any future Opt Y rounds) produce per-cell n_OK / wcf_max / fin_min readings under fresh seed sweeps. The longitudinal dataset format should be standardized (e.g. one JSON file per round with consistent schema) so that the publication can include a "cohort-race repeatability across N independent sweeps" appendix table. Recommend a thin schema spec in R61 if Opt Y monitoring continues.
7. **`R60_INTEGRATION_MANIFEST.json` metadata fields**. Per `R60_DECIDER_PLAN.md` §6, the manifest is COPIED with metadata bumped (round/timestamp/notes) but binary entries byte-identical. R61 should verify that the metadata-only delta is correctly recorded so future archeology can distinguish "binary-changed" rounds from "metadata-only" rounds across the R50D AS-IS streak.

These are publication-prep decisions, not R59-record corrections. The R59 record is internally consistent.

---

## Verdict

**POLICY_ONLY.** R60 cohort K-2 / Opt U delivers this documentation pivot artifact formalizing the post-R59 structural ceiling: 42/42 strict 10-run VC + 41/42 WIN + 40/42 AITER bit-deterministic (largest in project history HELD) + 2/42 HK pool (smallest in project history HELD) + 11 consecutive R50D AS-IS reuse rounds + 1 LOSE cell (L8 at aiter-internal ceiling) + 1 noise-edge cell (L1 oscillating around WIN-line per Opt R policy A) + 1 near-gate HK survivor (L3 with ~10-20% per-sweep tail-draw probability under R44D FINITE_GATE 0.97).

All bounded-cost mechanism axes on the 3 attention cells are CLOSED. The only remaining non-methodology axis (Opt T — L8 from-scratch K=128256 HK build, ~3 R-rounds, very low confidence) is DEFERRED unless explicit user election. R61+ axis space is dominated by ELECT-class methodology rounds (Opt U-2 publication-prep continuation, Opt Y₂+ cohort-race longitudinal monitoring, Opt Z per-shape decomposition table).

R60 cross-reference: cohort K-1 Opt Y outcome (PASS or FAIL on the L3 cohort-race repeatability test under fresh disjoint seed set `[202, 404, 606, 808, 1010, 1212, 1414, 1616, 1818, 2020]`) is recorded in `R60K1_Y_INTEGRATION_FRAGMENT.json` and determines the R61 axis branching per `R60_DECIDER_PLAN.md` §5.

**No manifest changes; no PROMOTE; no DEAD; no kernel work; no GPU touched by this artifact.**

**Artifact**: `R60_OPT_U_DOC_PIVOT.md` (this file).
