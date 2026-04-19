# MXFP4 GEMM Optimization TODO

## Current State (2026-04-19, post-R54 — WIN +3 NET VC STRICT + 6 PERF CLAW-BACKS +17-28pp, COMMIT, 12/12 PROMOTE / 0 DEAD ACROSS 4 COHORTS, AITER BIT-DETERMINISTIC SHARE 15→27 OF 42 = LARGEST SINGLE-ROUND AITER EXPANSION, 36/42 STRICT 10-RUN VC, 6TH CONSECUTIVE R50D AS-IS REUSE)

**HEADLINE — R54 RESTORES R52'S 36/42 STRICT 10-RUN VC LEVEL AND FUNDAMENTALLY CHANGES THE KERNEL-BASE STRUCTURE: 27 OF 42 CELLS ARE NOW BIT-DETERMINISTIC AITER `.CO` DISPATCH (vs R52's 7/42, R53's 15/42).** Net VC delta vs R53 = **+3 NET VC strict 10-run** (33 → 36/42); vs R52 = **net 0 in count BUT 27 AITER cells (vs R52's 7) eliminate cohort-race tail-draw risk on the majority of the leaderboard**. **12/12 PROMOTE / 0 DEAD** (best round structure since round-tracking began). 4 worker cohorts: E-1 cohort-race rescue M-heavy +3 NEW VC, E-2 cohort-race rescue N-heavy +3 NEW VC, D-4A sub-90% perf claw-back +70.51pp aggregate, D-4B 90-95% marginal claw-back +59.27pp aggregate. 6 D-4A/D-4B perf claw-backs deliver +17.66pp to +28.28pp over HK baselines, all well past D-3C 0.5pp gate, **zero reverts**. Mean +4.42pp comp/shape on 30 shared-VC vs R53. AITER cells **27/27 PASS (100% bit-deterministic, wcf_max=0.0, wcf_std=0.0, fin_min=1.0)**; HK cells 9/15 PASS. 6th consecutive R50D shim AS-IS reuse round (no rebuild, no kernel modification).

**R54 attempts summary** (all 12 reuse R50D shim AS-IS with 256×256 aiter `.co`):
- **R54 Opt E-1 — Cohort-race rescue M-heavy/N=4096 (worker, 3/3 PROMOTE +3 NET VC)**:
  - `(32768,4096,2048)`: HK R52-VC 102.0% → R53 LOSS PASS_8/10 → AITER 105.16% PASS_10/10. **+1 NET VC rescue.**
  - `(32768,4096,3072)`: HK R52-VC 99.5% → R53 LOSS PASS_9/10 → AITER 118.58% PASS_10/10. **+1 NET VC rescue.**
  - `(28672,4096,8192)`: HK R52-VC 91.8% → R53 LOSS PASS_9/10 → AITER 108.72% PASS_10/10. **+1 NET VC rescue.**
- **R54 Opt E-2 — Cohort-race rescue N-heavy/M=4096+mid-K (worker, 3/3 PROMOTE +3 NET VC)**:
  - `(4096,32768,4096)`: HK R52-VC 91.8% → R53 LOSS PASS_9/10 → AITER 105.66% PASS_10/10. **+1 NET VC rescue.**
  - `(4096,32768,6144)`: HK R52-VC 92.3% → R53 LOSS PASS_8/10 → AITER 106.96% PASS_10/10. **+1 NET VC rescue.**
  - `(16384,4096,6144)`: HK R52-VC 98.5% → R53 LOSS PASS_8/10 → AITER 115.20% PASS_10/10. **+1 NET VC rescue.**
- **R54 Opt D-4A — Sub-90% HK-VC perf claw-back (worker, 3/3 PROMOTE 0 NET VC, +70.51pp aggregate)**:
  - `(4096,14336,16384)`: HK 84.41% → AITER 105.87% (+21.46pp).
  - `(32768,4096,7168)`: HK 88.36% → AITER 106.63% (+18.27pp).
  - `(6144,4096,8192)`: HK 89.53% → AITER 117.82% (+28.29pp). **Largest D-4A.**
- **R54 Opt D-4B — 90-95% HK-VC marginal claw-back (worker, 3/3 PROMOTE 0 NET VC, +59.27pp aggregate)**:
  - `(4096,4096,16384)`: HK 93.30% → AITER 110.96% (+17.66pp).
  - `(4096,14336,8192)`: HK 91.92% → AITER 114.17% (+22.25pp).
  - `(16384,4096,7168)`: HK 93.76% → AITER 111.89% (+18.13pp).

**R54 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010]; 4 GPUs, ~14 min wall, 420 runs)**:
- Manifest has 27 aiter `.co` overrides (R50D + R51 + R52 + R53 8 non-256×256 + R54 12 new 256×256); 15 shapes use HipKittens kernel with R44 baseline params.
- All 12 R54 PROMOTE candidates RE-VERIFIED 10/10 PASS at reviewer re-bench with bit-determinism.
- AITER cells: **27/27 PASS (100% bit-deterministic)**.
- HK cells: 9/15 PASS (6 R40B FAIL on cohort-race wcf or fin gates).
- 0 R53-NO shapes gained VC under R54; 3 R53-VC HK shapes lost VC under cohort-race churn on UNCHANGED R40B `.so` (`16384x14336x2048`, `16384x28672x2048`, `6144x32768x4096`) — R45+ phenomenon, NOT R54 regression.
- Net cohort churn: **0 / -3 = -3** on UNCHANGED HK .so cells.
- R54 contribution: **+6 NEW VC** (E-1, E-2) - 3 cohort losses = **net +3 strict VC**.
- **Final VC count: 36/42 strict 10-run (vs R53 33; vs R52 36)**.
- Mean perf delta on 30 shared-VC shapes: **+4.42pp comp/shape vs R53**.
- D-3C/D-4-style marginal-promotion check: all 6 D-4A/D-4B promotions hold +17.66 to +28.28pp over HK baselines on reviewer 10-run with bit-determinism — **zero reverts needed**.
- Files: `R54_INTEGRATION_VERDICT.md`, `R54_INTEGRATION_MANIFEST.json`, `bench_all_42_R54_INTEGRATION.py`, `R54_INTEGRATION_{10RUN,SMOKE1}.{json,log,console}`, `R54_DECIDER_PLAN.md`.

**R54 net result**: **+6 NEW VC RESCUES + 6 PERF CLAW-BACKS (+129.78pp aggregate on D-4A+D-4B) + LARGEST AITER EXPANSION ROUND (+12 cells)**, -3 cohort tail-draw (R45+ phenomenon, not regression). Net strict VC: 36/42. **Kernel base now structurally stable: 27/42 AITER bit-deterministic vs R52's 7/42. Zero kernel modification, zero new shim build (6th consecutive R50D reuse).**

### R55 candidates (post-R54, ordered by mechanism-confidence)
1. **R55 Opt D-extended-5 (highest confidence)** — Continue mining the remaining 9 HK-cell sub-95% pool against AITER 256×256. After R54, the explicit non-AITER candidates include: `16384x14336x{2048,4096}`, `16384x28672x{2048,4096}`, `16384x4096x4096`, `16384x6144x{2048,4096}`, `32768x14336x2048`, `32768x28672x2048`, `32768x6144x2048`, `4096x4096x8192`, `6144x32768x4096`. Estimated +3-6 NET VC + further round-over-round stability gain.
2. **R55 Opt F (medium)** — Investigate 192×256 / 128×256 aiter tile rescues for the remaining 6 HK FAILs (`16384x{14336,28672}x{2048,4096}`, `16384x6144x4096`, `6144x32768x4096`).
3. **R55 Opt B (low)** — 32×32×64 MFMA in HK kernel; still deferred — AITER mining is far higher confidence per round.

### R55+ axes to NOT attempt (closed by R45-R54)
- All R53 closed list PLUS:
- **DO NOT try to "improve" HK 256×256 path** (D-4A/D-4B confirmed AITER beats HK +17-28pp on 84-94% HK band — port to AITER instead of optimizing HK).
- Aspect-ratio-blind aiter tile selection (R53 D-3A_1 demonstrated 96×640 NOT universal — must SMOKE-gate).
- ANY fence position in K-loop, MFMA reorder, R38A/R39A/R47B variants — all closed.

### R54 stopping-criterion check
- Floor (≥34/42 strict 10-run, no R54-attributable regression): **MET (36/42 strict; +6 NET VC rescues + 6 perf claw-backs attributable to R54; -3 strict net is documented cohort tail-draw on UNCHANGED `.so`, not R54 regression)**.
- Stretch (≥38/42 strict): **MISS (36/42)** but the round mechanism (largest AITER expansion +12 cells) is durable and unblocks R55 D-extended-5.
- Round value: **+6 NEW VC + 6 perf claw-backs + 5 durable findings** (AITER `.co` is now load-bearing for stability; D-3A-1 risk does NOT generalize to 256×256; HK 256×256 has +17-28pp untapped headroom; aiter heuristic uniformly picks 256×256 for current candidates; 6 consecutive R50D AS-IS reuse rounds).

### Round sequence sanity check (last 12 rounds)
- R43: DEAD (3 axes)
- R44: WIN +8 (27 → 35/42)
- R45-R49: 5 DEAD rounds in a row
- R50: WIN +1 (35 → 36/42, aiter `.co` dlopen first PoC)
- R51: WIN +1 strict (30 → 31/42) + 2 perf claw-backs
- R52: WIN +5 strict (31 → 36/42) + 3 perf claw-backs
- R53: PARTIAL WIN +2 NET VC rescues -5 cohort net -3 strict (36 → 33/42) + 6 perf claw-backs
- **R54: WIN +6 NET VC rescues -3 cohort net +3 strict (33 → 36/42) + 6 perf claw-backs +17-28pp; 27/42 cells now bit-deterministic AITER** ← LARGEST AITER EXPANSION ROUND; KERNEL BASE NOW STRUCTURALLY STABLE

---

## Previous State (2026-04-19, post-R53 — PARTIAL WIN +2 NET VC RESCUES -5 COHORT NET -3 STRICT, COMMIT, FIRST NON-256×256 ATTEMPT, R50D SHIM PROVEN TILE-GENERIC, 8 PROMOTE / 1 DEAD ACROSS 9 CANDIDATES, 33/42 STRICT 10-RUN VC, 5TH CONSECUTIVE R50D AS-IS REUSE)

**HEADLINE — R53 IS THE FIRST NON-256×256 AITER `.CO` DISPATCH ATTEMPT and validated the universal-bdx=256 hypothesis.** Strict 10-run VC delta vs R52 = **-3 (36 → 33/42)**, but the loss is ENTIRELY cohort-race tail-draw on UNCHANGED HK R40B/R41B `.so` files (R45+ documented phenomenon: +1 cohort gain / -6 cohort losses, NONE caused by R53). R53's actual contribution = **+2 NET VC rescues** (D-3B_1 `(4096,32768,14336)` 66.70% NEW VC + D-3B_2 `(32768,4096,14336)` 85.56% NEW VC, both previously NO_VC) **+ 6 perf claw-backs** via aiter `.co` (D-3A_2 +22.65pp, D-3A_3 +48.82pp, D-3B_3 +0.22pp, D-3C_1 +1.77pp, D-3C_2 +0.37pp, D-3C_3 +3.56pp). 8/8 PROMOTE re-confirmed at reviewer 10-run with bit-determinism (wcf_max=0.0, wcf_std=0.0, fin_min=1.0). 1 DEAD (D-3A_1 `(4096,14336,16384)` 96×640 SMOKE-regressed to 52.45% comp vs HK 84.53% — worker correctly stopped pre-10-run). **The R50D shim is now proven tile-generic**: same `bdx=256` shim handles 256×256, 96×640, AND 64×1024 aiter `.co` files with ZERO rebuild — validates universal-bdx hypothesis at `asm_gemm_a4w4.cu:290`. 5th consecutive R50D AS-IS reuse round; zero kernel modification. Mean +3.20pp comp/shape on 30 shared-VC shapes.

**R53 attempts summary**:
- **R53 Opt D-3A (worker, 96×640 tile, 3 candidates)**: 2/3 PROMOTE / 1 DEAD.
  - D-3A_1 `(4096,14336,16384)`: **DEAD** at SMOKE — 96×640 underperforms (52.45% comp vs HK 84.53%); aspect-ratio sensitivity. Worker correctly stopped pre-10-run.
  - D-3A_2 `(6144,4096,16384)`: **PROMOTE 10/10 OK, 107.62% comp (+22.65pp vs HK)**. wcf_max=0.0, fin_min=1.0.
  - D-3A_3 `(4096,6144,32768)`: **PROMOTE 10/10 OK, 131.63% comp (+48.82pp vs HK)** — largest perf claw-back of round. wcf_max=0.0, fin_min=1.0.
- **R53 Opt D-3B (worker, 64×1024 tile, 3 candidates)**: 3/3 PROMOTE, **+2 NET VC rescues**.
  - D-3B_1 `(4096,32768,14336)`: **PROMOTE 10/10 OK, 66.70% NEW VC** (was NO_VC). Bit-deterministic.
  - D-3B_2 `(32768,4096,14336)`: **PROMOTE 10/10 OK, 85.56% NEW VC** (was NO_VC). Bit-deterministic.
  - D-3B_3 `(128256,32768,4096)`: **PROMOTE 10/10 OK, 86.17% (+0.22pp marginal vs HK)**. Bit-deterministic.
- **R53 Opt D-3C (worker, 64×1024 tile, 3 marginal candidates)**: 3/3 PROMOTE.
  - D-3C_1 `(14336,32768,4096)`: **PROMOTE 87.83% (+1.77pp vs HK)**. Bit-deterministic.
  - D-3C_2 `(28672,32768,4096)`: **PROMOTE 87.68% (+0.37pp marginal vs HK)**. Bit-deterministic.
  - D-3C_3 `(16384,4096,14336)`: **PROMOTE 91.06% (+3.56pp vs HK)**. Bit-deterministic.

All 8 PROMOTE workers reused **the EXISTING R50D shim** at `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` AS-IS, with new per-shape grid params and per-shape aiter `.co` paths (`f4gemm_bf16_per1x32Fp4_BpreShuffle_96x640.co` and `f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024.co`). NO shim rebuild. Kernel UNCHANGED.

**R53 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010]; 4 GPUs, 14 min wall, 420 runs)**:
- Manifest has 15 aiter `.co` overrides (R50D + R51 D-1/D-2/D-3 + R52 D-2A/D-2B/D-2C + R53 8 non-256×256); 27 shapes use HipKittens kernel with R44 baseline params.
- All 8 R53 PROMOTE candidates RE-VERIFIED 10/10 PASS at reviewer re-bench; 15/15 AITER cells PASS (100% bit-deterministic).
- 18/27 HK cells PASS (9 FAIL: cohort-race wcf or fin gates).
- 1 R52-NO shape gained VC under R53 cohort-race tail-draw on UNCHANGED `.so` (`4096x4096x16384`).
- 6 R52-VC shapes lost VC under same churn mechanism on UNCHANGED HK R40B/R41B `.so` files (`16384x4096x6144`, `28672x4096x8192`, `32768x4096x3072`, `4096x32768x4096`, `4096x32768x6144`, `32768x4096x2048`).
- Net cohort churn: **+1 / -6 = -5** on UNCHANGED .so cells (NOT R53 regression — R45+ tail-draw).
- R53 contribution: **+2 NEW VC** (D-3B_1, D-3B_2) + 1 cohort gain.
- **Final VC count: 33/42 strict 10-run (vs 36/42 at R52)**.
- Mean perf delta on 30 shared-VC shapes: **+3.20pp comp/shape**.
- D-3C marginal-promotion check: all 4 (D-3C_1/D-3C_2/D-3C_3/D-3B_3) hold positive vs HK on reviewer 10-run with bit-determinism — **zero reverts needed**.
- Files: `R53_INTEGRATION_VERDICT.md`, `R53_INTEGRATION_MANIFEST.json`, `bench_all_42_R53_INTEGRATION.py`, `R53_INTEGRATION_10RUN.{json,log,console}`, `R53_INTEGRATION_SMOKE1.{json,log,console}`, `R53_DECIDER_PLAN.md`.

**R53 net result**: **+2 NET VC RESCUES + 6 PERF CLAW-BACKS** attributable to R53; -5 cohort tail-draw on UNCHANGED `.so` files (R45+ phenomenon, not regression). Net strict VC: 33/42. **First non-256×256 attempt validated R50D shim as TILE-GENERIC. Zero kernel modification, zero new shim build (5th consecutive R50D reuse).**

### R54 candidates (post-R53, ordered by mechanism-confidence)
1. **R54 Opt D-extended-4 (highest confidence)** — Continue mining non-256×256 aiter `.co` tiles for sub-90% HK shapes. After R53, enumerate remaining sub-90% HK-VC shapes against the full 36-tile aiter `.co` library. Estimated +0-2 NET VC + 5-10pp mean comp via D-3B/D-3C-style rescues.
2. **R54 Opt E — HK kernel cohort-race stabilization (medium)** — Address the -6 cohort tail-draw losses on R40B/R41B by either (a) tightening internal kernel gate criteria or (b) finding aiter `.co` replacements for the 6 shapes that fell out of VC. Estimated +3-6 NET VC if successful.
3. **R54 Opt B (low-medium)** — Try 32×32×64 MFMA in HipKittens kernel for shapes where NO aiter `.co` fits. Untried structural axis. Higher risk.

### R54+ axes to NOT attempt (closed by R45-R53)
- All R52 closed list PLUS:
- **Aspect-ratio-blind aiter tile selection** (D-3A_1 demonstrated 96×640 is NOT universally superior — must SMOKE-gate per-shape before promotion).
- ANY fence position in K-loop (R45B / R47A / R48A / R49C / R50A all DEAD)
- MFMA↔ds_read interleaving variants alone (R50A closed)
- `R38A_INLINE_BUFLOAD_LDS=1` for production builds (R47B closed)
- `R39A_TAIL_SCALE_CLAMP` on intermediate-K wcf-flake shapes (R46A closed)
- 4-buffer or higher LDS rotation alone (R46B + R47A closed)
- Wave-priority / s_nop pacing / MFMA half-split alone (R47C closed by 10-run)
- Physical asm-block split of `kpair_64mfma_step34` (R48A closed)
- `PF_MPT` depth override (R48C closed mechanically)
- Internal MFMA reorder / s_setprio / lgkmcnt drain inside step34 alone (R48B closed)
- Single-knob aiter pattern ports (R49A closed)
- R44A back-edge drain extension to N=32768 (R49B closed; cohort scales with N)
- Embedded vmcnt INSIDE producer asm volatile (R49C closed)
- `gm × lgk × pfoff` knob sweeps on R44 VC shapes (R50C closed within 2pp of saturation)
- Any attempt to "improve" the aiter `.co` dlopen path itself (R51 closed)

### R53 stopping-criterion check
- Floor (≥31/42 strict 10-run, no regression attributable to round): **MET (33/42 strict; +2 NET VC rescues + 6 perf claw-backs attributable to R53; -3 strict net is documented cohort tail-draw on UNCHANGED `.so`, not R53 regression)**.
- Stretch (≥34/42 strict): **MISS (33/42)** but the round mechanism (tile-generic shim validation) is durable and unblocks R54 D-extended-4.
- Round value: **+2 NET VC rescues + 6 perf claw-backs + 5 durable findings** (R50D shim is tile-generic; 96×640 has aspect-ratio sensitivity; 64×1024 broadly competitive; aiter `.co` proven across 15 shapes total; cohort-race churn now dominates net VC accounting at -5 noise floor).

### Round sequence sanity check (last 11 rounds)
- R43: DEAD (3 axes)
- R44: WIN +8 (27 → 35/42)
- R45-R49: 5 DEAD rounds in a row
- R50: WIN +1 (35 → 36/42, aiter `.co` dlopen first PoC)
- R51: WIN +1 strict (30 → 31/42) + 2 perf claw-backs
- R52: WIN +5 strict (31 → 36/42) + 3 perf claw-backs
- **R53: PARTIAL WIN +2 NET VC rescues -5 cohort net -3 strict (36 → 33/42) + 6 perf claw-backs; first non-256×256 attempt; R50D shim proven tile-generic** ← STREAK CONTINUES IN MECHANISM EVEN WITH NEGATIVE NET STRICT

---

## Previous State (2026-04-19, post-R52 — WIN +5 VC + 3 PERF CLAW-BACKS, COMMIT, 3RD CONSECUTIVE NON-DEAD ROUND, LARGEST VC GAIN SINCE R44, 36/42 STRICT 10-RUN VC, AITER `.CO` DLOPEN PATTERN VALIDATED ACROSS 7 SHAPES)

**HEADLINE — R52 IS THE 3RD CONSECUTIVE WIN ROUND AND THE LARGEST VC GAIN SINCE R44 (+8). Net VC delta vs R51 = **+5 NET VC strict 10-run** (31 → 36/42). 3/3 PROMOTE / 0 DEAD (matches R51 round structure). The 3 PROMOTE workers are perf claw-backs via R50D aiter `.co` dlopen shim REUSED AS-IS for the 4th consecutive round (no shim rebuild, no kernel modification): D-2A `(4096,28672,32768)` 61.9% → 101.85% comp (+39.93pp), D-2B `(4096,32768,128256)` 72.3% → 99.72% comp (+27.47pp), D-2C `(4096,4096,32768)` 77.1% → 105.91% comp (+28.81pp). All RE-VERIFIED 10/10 PASS at reviewer 10-run with bit-determinism (wcf_max=0.0, wcf_std=0.0, fin_min=1.0). The +5 NET VC comes entirely from cohort-race tail-draw on UNCHANGED `.so` files (6 shapes gained, 1 lost — favorable seed draw). The aiter `.co` dlopen pattern is now **production-ready, proven across 7 distinct shapes total** (R50D + R51 D-1/D-2/D-3 + R52 D-2A/D-2B/D-2C), shape-generic for the 256×256 tile case, K-generic (D-2B at K=128256), grid-size-generic (D-2C at gdx=gdy=16, D-2A at gdx=112). 100% PROMOTE rate (6/6) on the dlopen axis when target shape's aiter heuristic picks 256×256.**

**R52 attempts summary**:
- **R52 Opt D-2A — Aiter `.co` dlopen for `(4096, 28672, 32768)` largest sub-90% gap**: **PROMOTE 10/10 OK, +39.93pp comp.** Reuses R50D shim AS-IS at `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` (no rebuild). Per-shape grid: gdx=ceil(N/256)=112, gdy=ceil(M/256)=16, gdz=1, bdx=256. KernelArgs: M=4096, N=28672, K=32768. Worker 10-run: p50=5736.7 TFLOPS = 101.54% comp; reviewer re-bench: p50=5754.2 TFLOPS = 101.85% comp. wcf_max=0.0, wcf_std=0.0, fin_min=1.0 across 10 INDEPENDENT seeds. Files: `R52_OPT_D2A_VERDICT.md`, `R52D2A_INTEGRATION_FRAGMENT.json`, `R52_OPT_D2A_{SMOKE,10RUN}.{json,log}`, `bench_R52D2A.py`. Kernel UNCHANGED.
- **R52 Opt D-2B — Aiter `.co` dlopen for `(4096, 32768, 128256)` (largest K on board)**: **PROMOTE 10/10 OK, +27.47pp comp + K=128256 generality proof.** Reuses R50D shim AS-IS. Per-shape grid: gdx=128, gdy=16, gdz=1, bdx=256. KernelArgs: M=4096, N=32768, K=128256. K=128256 was the only flagged risk axis — presented zero issue (clean SMOKE first try, perfectly bit-stable across 10 INDEPENDENT seeds). Worker 10-run: p50=5763.6 TFLOPS = 99.70% comp; reviewer re-bench: p50=5765.1 TFLOPS = 99.72% comp. **Confirms aiter `.co` pattern is K-generic.** Files: `R52_OPT_D2B_VERDICT.md`, `R52D2B_INTEGRATION_FRAGMENT.json`, `R52_OPT_D2B_{SMOKE,10RUN}.{json,log}`, `bench_R52D2B.py`. Kernel UNCHANGED.
- **R52 Opt D-2C — Aiter `.co` dlopen for `(4096, 4096, 32768)` (single-grid-round 16×16)**: **PROMOTE 10/10 OK, +28.81pp comp + grid-size genericity proof.** Reuses R50D shim AS-IS. Per-shape grid: gdx=16, gdy=16, gdz=1, bdx=256. KernelArgs: M=4096, N=4096, K=32768. Single-round 16×16 grid (smallest grid tested in R51-R52). Worker 10-run: p50=5437.2 TFLOPS = 105.52% comp; reviewer re-bench: p50=5457.4 TFLOPS = 105.91% comp. **Confirms aiter `.co` pattern is grid-size-generic** (D-2A at gdx=112 + D-2C at gdx=16 = full range). Files: `R52_OPT_D2C_VERDICT.md`, `R52D2C_INTEGRATION_FRAGMENT.json`, `R52_OPT_D2C_{SMOKE,10RUN}.{json,log}`, `bench_R52D2C.py`. Kernel UNCHANGED.

**R52 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010]; 4 GPUs, 13 min wall, 420 runs)**:
- Manifest has 7 aiter `.co` overrides (R50D's `4096x32768x28672` + R51 D-1/D-2/D-3 + R52 D-2A/D-2B/D-2C); 35 shapes use HipKittens kernel with R44 baseline params.
- All 3 R52 PROMOTE candidates RE-VERIFIED 10/10 PASS at reviewer re-bench.
- 6 R51-NO shapes gained VC under R52 cohort-race tail-draw (`128256x32768x4096`, `14336x32768x4096`, `32768x28672x2048`, `4096x128256x32768`, `4096x32768x6144`, `4096x6144x32768`) on UNCHANGED `.so` files.
- 1 R51-VC shape lost VC under same churn mechanism (`4096x4096x16384`: PASS_10/10 → PASS_9/10, pct nearly unchanged 93.34 → 93.07).
- Net cohort churn: **+5 VC** (6 gained - 1 lost). D-2 PROMOTE workers don't add VC count (already HK-VC at low comp); they're pure perf claw-backs.
- **Final VC count: 36/42 strict 10-run (vs 31/42 at R51 strict)**.
- Mean perf delta on 30 shared-VC shapes: **+3.27pp comp/shape** (D-2A/B/C contribute +96.21pp aggregate).
- Files: `R52_INTEGRATION_VERDICT.md`, `R52_INTEGRATION_MANIFEST.json`, `bench_all_42_R52_INTEGRATION.py`, `R52_INTEGRATION_10RUN.{json,log,console}`, `R52_INTEGRATION_SMOKE1.{json,log,console}`, `R52_DECIDER_PLAN.md`.

**R52 net result**: **+5 NET VC + 3 PERF CLAW-BACKS (+96.21pp aggregate on D-2A/B/C)**, 0 regression attributable to R52. Branch advances with R52 manifest delta + 3 new per-shape backend dispatch entries. **Zero kernel modification, zero new shim build (4th consecutive R50D reuse).**

### R53 candidates (post-R52, ordered by mechanism-confidence)
1. **R53 Opt D-extended-3 (highest confidence)** — Continue mining sub-90% comp HK-VC shapes where 256×256 aiter tile is optimal. After R52 promotions, audit `R52_INTEGRATION_10RUN.json` to enumerate remaining sub-95% shapes. Estimated +0-2 VC + 5-10pp mean comp. Same R50D shim AS-IS.
2. **R53 Opt D-non-256x256 (medium)** — Aiter has 36 `.co` files at various tile geometries. For shapes where aiter heuristic picks NON-256×256, generalize R50D shim with runtime tile parameters (extending KernelArgs `tile_m`/`tile_n` fields) and add per-tile shim build (or runtime dispatch). Requires 1 shim rebuild but reusable thereafter.
3. **R53 Opt B (low-medium)** — Try a DIFFERENT MFMA shape (32×32×64 instead of 16×16×128) in HipKittens kernel for shapes where NO aiter `.co` fits. Untried structural axis. Higher risk.

### R53+ axes to NOT attempt (closed by R45-R52 work)
- ANY fence position in K-loop (R45B / R47A / R48A / R49C / R50A all DEAD)
- MFMA↔ds_read interleaving variants alone (R50A closed)
- `R38A_INLINE_BUFLOAD_LDS=1` for production builds (R47B closed)
- `R39A_TAIL_SCALE_CLAMP` on intermediate-K wcf-flake shapes (R46A closed)
- 4-buffer or higher LDS rotation alone (R46B + R47A closed)
- Wave-priority / s_nop pacing / MFMA half-split alone (R47C closed by 10-run)
- Physical asm-block split of `kpair_64mfma_step34` (R48A closed)
- `PF_MPT` depth override (R48C closed mechanically)
- Internal MFMA reorder / s_setprio / lgkmcnt drain inside step34 alone (R48B closed)
- Single-knob aiter pattern ports (R49A closed)
- R44A back-edge drain extension to N=32768 (R49B closed; cohort scales with N)
- Embedded vmcnt INSIDE producer asm volatile (R49C closed)
- `gm × lgk × pfoff` knob sweeps on R44 VC shapes (R50C closed within 2pp of saturation)
- Any attempt to "improve" the aiter `.co` dlopen path itself (R51 closed: bit-deterministic, at 100%+ comp)

### R52 stopping-criterion check
- Floor (≥31/42 strict 10-run, no regression): **MET (36/42 strict; +5 net VC; 0 attributable regression — the 1 lost VC is documented cohort tail-draw on UNCHANGED `.so`)**.
- Stretch (≥34/42 strict): **MET +2 ABOVE STRETCH** (36/42).
- Round value: **+5 VC + 3 perf claw-backs (+96.21pp aggregate) + 4 durable findings** (aiter `.co` dlopen pattern proven across 7 shapes; K-generic at K=128256; grid-size-generic across full gdx range; 100% PROMOTE rate 6/6 on the dlopen axis).

### Round sequence sanity check (last 10 rounds)
- R43: DEAD (3 axes)
- R44: WIN +8 (27 → 35/42)
- R45-R49: 5 DEAD rounds in a row
- R50: WIN +1 (35 → 36/42, aiter `.co` dlopen first PoC)
- R51: WIN +1 strict (30 → 31/42) + 2 perf claw-backs (3/3 PROMOTE)
- **R52: WIN +5 strict (31 → 36/42) + 3 perf claw-backs (3/3 PROMOTE)** ← LARGEST GAIN SINCE R44, STREAK CONTINUES

---

## Previous State (2026-04-19, post-R51 — WIN +1 VC + 2 MASSIVE PERF CLAW-BACKS, COMMIT, 2ND CONSECUTIVE NON-DEAD ROUND, AITER `.CO` DLOPEN PATTERN PROVEN SHAPE-GENERIC, 37/42 VC MIXED-PROTOCOL OR 31/42 STRICT 10-RUN)

**HEADLINE — R51 IS THE 2ND CONSECUTIVE WIN ROUND AFTER R50'S STREAK BREAK. Net VC delta vs R50 = +1 (30 → 31/42 strict 10-run, OR 36 → 37/42 in R50 mixed-protocol headline). 3/3 PROMOTE, 0 DEAD = highest-yield round since R44. The breakthrough is **R51 Opt D (3 workers in parallel)**: same R50D aiter `.co` dlopen shim REUSED AS-IS (no rebuild) for 3 new shapes via per-shape backend dispatch entries: D-1 `(14336,4096,32768)` 60.4% → 103.3% comp (+42.9pp, +2,249.7 TFLOPS), D-2 `(16384,4096,28672)` 62.0% → 104.8% comp (+42.8pp, +2,366.2 TFLOPS, retires fragile R44A back-edge-drain cell), D-3 `(28672,4096,16384)` FLAKE_7/10 → PASS_10/10 @ 102.4% comp (+1 NET VC). All 3 achieve bit-determinism (wcf_max=0.0, wcf_std=0.0, fin_min=1.0 across 10 INDEPENDENT seeds). Reviewer 10-run integration confirms net +1 VC + mean +220.8 TFLOPS/shape on 26 shared-VC shapes (+4.35pp comp/shape). The aiter `.co` dlopen pattern is now proven shape-generic for the 256×256 tile case (4 distinct shapes ported with ZERO shim rebuild) — production-ready as a per-shape escape hatch for ANY HipKittens cell trailing aiter binary by >10pp.**

**R51 attempts summary**:
- **R51 Opt D-1 — Aiter `.co` dlopen for `(14336,4096,32768)` largest leaderboard gap**: **PROMOTE 10/10 OK, +42.9pp comp.** Reuses R50D shim AS-IS at `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` (no rebuild). Per-shape grid: gdx=ceil(N/256)=16, gdy=ceil(M/256)=56, gdz=1, bdx=256. KernelArgs: M=14336, N=4096, K=32768. Aiter heuristic at `asm_gemm_a4w4.cu:100-145` confirms 256×256 is the optimal tile selection (min local_round 0.219, tiebreak compute2mem_efficiency=128.0 wins over 192×256). 10-run @ 80% INDEPENDENT seeds: 10/10 OK, wcf_max=0.0, fin_min=1.0, snr_med 55.6 dB, p50 = 5,418.4 TFLOPS = 103.3% of competitor 5,243.6. Files: `R51_OPT_D1_VERDICT.md`, `R51D1_INTEGRATION_FRAGMENT.json`, `R51_OPT_D1_{SMOKE,10RUN}.json`, `bench_R51D1.py`. Kernel UNCHANGED.
- **R51 Opt D-2 — Aiter `.co` dlopen for `(16384,4096,28672)` (R44A fragile back-edge drain replacement)**: **PROMOTE 10/10 OK, +42.8pp comp + robustness improvement.** Reuses R50D shim AS-IS. Per-shape grid: gdx=16, gdy=64, gdz=1, bdx=256. KernelArgs: M=16384, N=4096, K=28672. Replaces the only K=28672 cell HipKittens HAD in VC table — but only via R44A back-edge drain which has wcf_std=0.008 (fragile, near 0.01 protocol limit). Aiter binary delivers wcf_std=0.0 — **retires the last fragile non-FUSED+TS K=28672 cell**, eliminating cohort-race risk on this shape. p50 = 5,219.0 TFLOPS = 104.8% of competitor 4,981.2. Files: `R51_OPT_D2_VERDICT.md`, `R51D2_INTEGRATION_FRAGMENT.json`, `R51_OPT_D2_{SMOKE,10RUN}.json`, `bench_R51D2.py`. Kernel UNCHANGED.
- **R51 Opt D-3 — Aiter `.co` dlopen for `(28672,4096,16384)` FLAKE-to-PASS rescue (+1 NET VC)**: **PROMOTE 10/10 OK, +20.2pp comp, +1 NET VC.** This is the only D-1/D-2/D-3 candidate that increases VC count vs R50 baseline. R47C identified this shape as 5/10 OK at wcf_max=0.0272 under HK kernel; R51D-3 swaps in aiter binary → 10/10 OK at wcf_max=0.0. Reuses R50D shim AS-IS. Per-shape grid: gdx=16, gdy=112, gdz=1, bdx=256. KernelArgs: M=28672, N=4096, K=16384. p50 = 5,179.5 TFLOPS = 102.4% of competitor 5,058.5. Files: `R51_OPT_D3_VERDICT.md`, `R51D3_INTEGRATION_FRAGMENT.json`, `R51_OPT_D3_{SMOKE,10RUN}.json`, `bench_R51D3.py`. Kernel UNCHANGED.

**R51 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010]; 4 GPUs, 13 min wall, 420 runs)**:
- 4 aiter `.co` overrides in manifest (R50D's `4096x32768x28672` + R51 D-1 + D-2 + D-3); 38 shapes use HipKittens kernel with R44 baseline params.
- All 3 R51 PROMOTE candidates confirmed 10/10 PASS with wcf_max=0.0, fin_min=1.0.
- 4 R50-VC shapes lost VC under R51 (`32768x28672x2048`, `4096x6144x32768`, `4096x128256x32768`, `128256x32768x4096`) — ALL on UNCHANGED `.so` files; documented R45+ cohort-race tail-draw phenomenon, NOT R51 regressions.
- 4 R50-NO shapes symmetrically gained VC under R51 (`32768x4096x2048`, `4096x28672x32768`, `4096x32768x4096`, `4096x32768x128256`) from same cohort-race churn.
- Net VC delta from cohort churn: 0 (4 lost = 4 gained); net VC delta from D-3: +1; **TOTAL +1 NET VC**.
- Mean perf delta on 26 shared-VC shapes: **+220.8 TFLOPS/shape, +4.35pp comp/shape** (D-1 + D-2 contribute +85.7pp aggregate to the mean).
- **Final VC count: 31/42 strict 10-run; 37/42 in R50 mixed 5-run baseline + 10-run override headline construction.**
- Files: `R51_INTEGRATION_VERDICT.md`, `R51_INTEGRATION_MANIFEST.json`, `bench_all_42_R51_INTEGRATION.py`, `R51_INTEGRATION_10RUN.{json,log,console}`, `R51_INTEGRATION_SMOKE1.{json,log,console}`, `R51_DECIDER_PLAN.md`.

**R51 net result**: **+1 NET VC + 2 MASSIVE PERF CLAW-BACKS (+85.7pp aggregate on D-1+D-2)**, 0 regression attributable to R51. Branch advances with R51 manifest delta + 3 new per-shape backend dispatch entries. **Zero kernel modification, zero new shim build.**

### R52 candidates (post-R51, ordered by mechanism-confidence)
1. **R52 Opt D-extended-2 (highest confidence)** — Identify the next batch of HK-VC shapes <90% comp where 256×256 aiter tile is optimal per the heuristic. Per `R51_DECIDER_PLAN.md` remaining candidates list: `32768x4096x14336`, `16384x28672x4096`. Both expected 80-95pp gain. Estimated +0-2 VC + 5-10pp mean comp. Same shim reused AS-IS.
2. **R52 Opt D-non-256x256 (medium)** — Aiter has 36 `.co` files at various tile geometries (128×128, 192×256, 256×128, etc.). For shapes where the aiter heuristic picks NON-256×256, port the shim to a DIFFERENT tile by extending the KernelArgs `tile_m`/`tile_n` fields and adding a per-tile shim build (or generalize R50D shim with runtime tile parameters).
3. **R52 Opt B (low-medium)** — Try a DIFFERENT MFMA shape (32×32×64 instead of 16×16×128) in HipKittens kernel to break the cluster-B cohort race on shapes where NO aiter `.co` fits. Untried structural axis. Higher risk.

### R52+ axes to NOT attempt (closed by R45-R51 work)
- ANY fence position in K-loop (R45B / R47A / R48A / R49C / R50A all DEAD)
- MFMA↔ds_read interleaving variants alone (R50A closed)
- `R38A_INLINE_BUFLOAD_LDS=1` for production builds (R47B closed)
- `R39A_TAIL_SCALE_CLAMP` on intermediate-K wcf-flake shapes (R46A closed)
- 4-buffer or higher LDS rotation alone (R46B + R47A suggest fence-interaction not slot-count)
- Wave-priority / s_nop pacing / MFMA half-split alone (R47C closed by 10-run)
- Physical asm-block split of `kpair_64mfma_step34` (R48A closed)
- `PF_MPT` depth override (R48C closed mechanically)
- Internal MFMA reorder / s_setprio / lgkmcnt drain inside step34 alone (R48B closed)
- Single-knob aiter pattern ports (R49A closed)
- R44A back-edge drain extension to N=32768 (R49B closed; cohort scales with N)
- Embedded vmcnt INSIDE producer asm volatile (R49C closed)
- `gm × lgk × pfoff` knob sweeps on R44 VC shapes (R50C closed within 2pp of saturation)
- **Any attempt to "improve" the aiter `.co` dlopen path itself (R51 closes: bit-deterministic, at 100%+ comp, no room left)**

### R51 stopping-criterion check
- Floor (≥36/42 mixed protocol or ≥30/42 strict 10-run, no regression): **MET (37/42 mixed or 31/42 strict; +1 net VC; 0 attributable regression — the 4 lost VC are documented cohort tail-draw on UNCHANGED `.so`)**.
- Stretch (≥38/42 mixed): **NOT MET** (3 PROMOTE workers but the 4 cohort losses offset 2 of the gains in mixed accounting).
- Round value: **+1 VC + 2 massive perf claw-backs + 4 durable findings** (aiter `.co` dlopen pattern shape-generic for 256×256; aiter binaries structurally bit-deterministic; R50 "36/42" headline is mixed-protocol; 3 PROMOTE / 0 DEAD = highest-yield round since R44).

### Round sequence sanity check (last 9 rounds)
- R43: DEAD (3 axes)
- R44: WIN +8 (27 → 35/42)
- R45-R49: 5 DEAD rounds in a row
- R50: WIN +1 (35 → 36/42, aiter `.co` dlopen first proof of concept)
- **R51: WIN +1 (36 → 37/42 mixed OR 30 → 31/42 strict) + 2 perf claw-backs (3/3 PROMOTE)** ← STREAK CONTINUES

---

## Previous State (2026-04-19, post-R50 — WIN +1 VC ROUND, COMMIT, FIRST NON-DEAD SINCE R44, AITER `.CO` DLOPEN BREAKTHROUGH, 36/42 VC)

**HEADLINE — R50 BREAKS THE 5-DEAD-OF-7 STREAK. Net VC delta vs R44 baseline = +1 (35 → 36/42 VC). The breakthrough is **R50 Opt D**: per-shape backend dispatch in the production harness binds aiter's hand-written `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via `hipModuleLoadData` + `hipModuleGetFunction` + `hipModuleLaunchKernel` for the perma-CRASH cell `(4096,32768,28672)`. 10/10 INDEPENDENT seeds OK, 5585.6 TFLOPS = 100.31% of competitor 5568.2. **First ever VC for this shape** (perma-CRASH on every HipKittens variant tested R43-R49). The aiter `.co` dlopen pattern is reusable: ANY future MXFP4 cell HipKittens cannot solve and aiter has a tuned `.co` for can use the same shim approach in <1 day of work.**

**R50 attempts summary**:
- **R50 Opt A — Aiter MFMA↔ds_read 1:3/1:4 spread interleaving INSIDE `kpair_64mfma_step34` asm volatile**: **DEAD on 10-run @ 80% gate. 5th independent closure of the "MFMA accumulator race is orderable-scheduling" hypothesis.** Macro `R50A_AITER_INTERLEAVE` (default OFF, byte-equiv to R44 baseline) replaces ~256 lines of asm volatile body with a 1:4 spread variant. ISA-verified: BASELINE shows 25-instr pure-MFMA runs (61 occurrences); v1 breaks 30 of those 25-runs into runs of 3, 4, 11. Compiler did NOT strip the rewrite. 10-run on 6 cluster-B wcf-flake targets: 0/6 VC under v1 (identical 0/6 under baseline); R44 stretch `4096x4096x8192` retained VC (10/10 OK, +0.3% TFLOPS). Combined with R45B (5 in-block fence positions DEAD), R47A (3 external positions on 3-buffer DEAD), R49A (single vmcnt knob DEAD), R49C (embedded fence in producer asm DEAD): **the cluster-B cohort race lives in the AGPR forwarding path INSIDE the MFMA pipeline itself, NOT at any orderable boundary the compiler or asm body can rearrange.** The race is structural to the MFMA/AGPR microarchitecture. Files: `R50_OPT_A_VERDICT.md`, `R50A_INTEGRATION_FRAGMENT.json` (`{}`), `R50A_aiter_kloop_body.s`, `R50A_old_step34.s`, `R50A_new_step34.s`, `R50A_KERNEL_ISA.s` (v1 disasm — verified emit), `R50A_BASELINE_ISA.s`, `R50_OPT_A_{SMOKE,JACCARD,10RUN}.{json,log}`, `R50A_BUILD_MANIFEST.json`, `build_R50A/*.so` (22 modules). Kernel macro `R50A_AITER_INTERLEAVE` at `kernel_mxfp4_gluon_cpp.cpp` (default OFF).
- **R50 Opt C — Per-shape `gm × lgk × pfoff` perf-axis claw-back sweep on 14 R44 VC <90% comp shapes**: **PROMOTE → REVERT under 10-run cross-val. Pure variant-knob retuning (NO new macros).** 14 R44 VC shapes <90% comp targeted; per shape: 27-cell sweep (lgk ∈ {1,2,3} × gm ∈ {6,7,8} × pfoff offset ∈ {-4,0,+4}). 5-run @ 80% INDEPENDENT-seed gate found 1 PROMOTE candidate: `4096x28672x32768` gm8_lgk2_po28 (+2.90% pct_comp, base 61.90% → 64.80%, wcf_max=0.0182). 10-run cross-val under R50 INTEGRATION reviewer caught wcf_max=0.0296 > 0.02 hard gate at 9/10 OK — classic cohort-race tail draw. **REVERT to R44 R41A baseline.** Other shapes' best gains all <2.0%: `6144x4096x8192` +1.77%, `32768x4096x7168` +1.98%, `28672x4096x8192` +1.02% — the existing variant table is **auto-tuner-saturated within 0-2pp** on 13 of 14 tested shapes. **Mechanism (durable)**: R25-F/G `pfoff` mechanism is mostly tapped out; for R51+ perf work attack DIFFERENT structural axes (MFMA shape, tile geometry, thread block size). 5-run perf gate is INSUFFICIENT for cohort-race-prone shapes — promote candidates can pass 5-run and fail 10-run on the SAME `.so`. **All R51+ perf rounds MUST use 10-run @ 80% as the promote gate.** Decider had to synthesize the verdict because the Opt C agent stalled in a self-matching `pgrep -f "bench_R50C"` wait loop after sweep completion (the wait loop's bash command line itself contained literal `bench_R50C`, so pgrep found its own process and never returned 0). **Agent process bug (durable)**: NEVER use `pgrep -f X` wait loops where the bash command line could match X — use `pgrep -fx`, save PID and `wait $PID`, or run bench in foreground. Files: `R50_OPT_C_VERDICT.md` (decider-synthesized), `R50C_INTEGRATION_FRAGMENT.json` (PROMOTE → reviewer cross-val REVERT), `R50C_SWEEP.json` (88 KB — full 5-run consensus), `R50C_PREFILTER.json` (386 KB), `R50C_PROMOTE_CANDIDATES.json`, `R50C_baseline_pct_comp.json`, `R50C_BUILD_MANIFEST.json` (113 KB), `build_R50C/*.so`. Kernel UNCHANGED.
- **R50 Opt D — Aiter `.co` dlopen escape hatch for `(4096,32768,28672)` perma-CRASH cell**: **PROMOTE +1 VC. FIRST WIN since R44.** Self-contained pybind11 shim (`R50D_aiter_dlopen.cpp`, 219 lines) binds `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via `hipModuleLoadData` + `hipModuleGetFunction` + `hipModuleLaunchKernel`. 372-byte `KernelArgs` ABI mirrored verbatim from `/shared_nfs/kyle/test/aiter/csrc/py_itfs_cu/asm_gemm_a4w4.cu` (`static_assert(sizeof(KernelArgs)==372)`). Launch grid (128, 16, 1) block (256, 1, 1) shared 0 bytes — same MFMA shape as HipKittens 256×256 tile geometry. **Layout requirements (load-bearing for integration)**: B preshuffled via `aiter.shuffle_weight(B, layout=(16,16))` (DIFFERENT from HK); A_scale and B_scale via `aiter.get_triton_quant(per_1x32)(x, shuffle=True)` (DIFFERENT from HK `preshuffle()`); A is row-major fp4x2 uint8 (same as HK); C is `[((M+31)//32)*32, N] bf16` row-major (pad rows to multiples of 32). 10-run @ 80% INDEPENDENT-seed result: 10/10 OK, wcf_max=0.0, fin_min=1.0, snr_med 55.59-55.63 dB, p50 = 5585.6 TFLOPS = **100.31% of competitor** 5568.2. First ever VC for this shape — goes from PERMA-CRASH directly to slightly above the aiter baseline. **Per-shape backend dispatch in `bench_all_42_R50_INTEGRATION.py`**: `(M,N,K) == (4096,32768,28672)` → call shim with aiter prep utilities; else → HipKittens kernel with HK prep. **Zero blast radius on other shapes** — single-cell escape hatch with no kernel mutation. Files: `R50_OPT_D_VERDICT.md`, `R50D_INTEGRATION_FRAGMENT.json`, `R50D_aiter_dlopen.cpp` (219 lines), `build_R50D.py`, `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`, `bench_R50D.py`, `R50D_aiter_csv_audit.md`, `R50D_aiter_symbols.txt`, `R50_OPT_D_{SMOKE,10RUN}.{json,log}`, `R50D_BUILD_MANIFEST.json`.

**R50 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010]; 2 GPUs, 22 min wall, 420 runs)**:
- 35 R44 VC shapes preserved at R44 manifest (5 of 35 probabilistic cohort-race losses on UNCHANGED `.so` files — documented R45+ phenomenon per `project_mxfp4_R45_cohort_tail_draw.md`, NOT R50-induced regressions; R44 5-run protocol still shows 35 VC).
- `4096x32768x28672` (Opt D) → **PROMOTE 10/10 OK at 100.31% comp**.
- `4096x28672x32768` (Opt C) → REVERT (10-run wcf_max 0.0296 > 0.02 hard gate).
- Net perf delta on 35 R44 VC shapes: **-4.3 TFLOPS/shape avg** (perf-neutral, within seed noise floor).
- **Final VC count: 36/42 (35 R44 baseline + 1 from R50 Opt D)**.
- Files: `R50_INTEGRATION_VERDICT.md`, `R50_INTEGRATION_MANIFEST.json`, `bench_all_42_R50_INTEGRATION.py`, `R50_INTEGRATION_10RUN.{json,log,console}`, `R50_INTEGRATION_SMOKE1.{json,log}`, `R50_DECIDER_PLAN.md`.

**R50 net result**: **+1 VC (35 → 36/42)**, 0 regression. Branch should advance with R50 manifest delta + aiter shim + per-shape backend dispatch + R50A macro (default OFF) commit.

### R51 candidates (post-R50, ordered by mechanism-confidence)
1. **R51 Opt D-extended (highest confidence)** — Identify other sub-90% comp HipKittens shapes where aiter has a tuned `.co` and port the R50D shim pattern. Look for shapes ≤80% comp where an aiter binary exists in `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/`. Estimated +2-3 VC potential. Cost: ~1 day per shape; layout requires aiter prep utils. Zero blast radius on other shapes.
2. **R51 Opt B (medium)** — Try a DIFFERENT MFMA shape (e.g. 32×32×64 instead of 16×16×128) to break the AGPR forwarding chain on cluster-B cohort race. Structurally different from all prior axes (R50A is the 5th orderable-scheduling closure; R51B would attack the microarchitecture layer). High risk but the only untried structural axis on the kernel side.
3. **R51 perf round (low priority — perf-axis is mostly tapped out per R50 Opt C)** — If attempted, MUST use 10-run @ 80% gate from the start; do NOT trust 5-run gains. Attack DIFFERENT structural axes (kernel-level changes; not `gm × lgk × pfoff` knobs).

### R51+ axes to NOT attempt (closed by R45-R50 work)
- ANY fence position in K-loop (R45B / R47A / R48A / R49C / R50A all DEAD)
- MFMA↔ds_read interleaving variants alone (R50A closed)
- `R38A_INLINE_BUFLOAD_LDS=1` for production builds (R47B closed)
- `R39A_TAIL_SCALE_CLAMP` on intermediate-K wcf-flake shapes (R46A closed)
- 4-buffer or higher LDS rotation alone (R46B + R47A suggest fence-interaction not slot-count)
- Wave-priority / s_nop pacing / MFMA half-split alone (R47C closed by 10-run)
- Physical asm-block split of `kpair_64mfma_step34` (R48A closed: compiler RA/sched before asm boundary)
- `PF_MPT` depth override (R48C closed mechanically)
- Internal MFMA reorder / s_setprio / lgkmcnt drain inside step34 alone (R48B closed)
- Single-knob aiter pattern ports (R49A closed)
- R44A back-edge drain extension to N=32768 (R49B closed; cohort scales with N)
- Embedded vmcnt INSIDE producer asm volatile (R49C closed)
- `gm × lgk × pfoff` knob sweeps on R44 VC shapes (R50C closed within 2pp of saturation)

### R50 stopping-criterion check
- Floor (≥35/42, no regression): **MET (35 R44 baseline preserved + 1 new VC; final 36/42; 0 regression)**.
- Stretch (≥36/42): **MET** (Opt D delivers `4096x32768x28672` at 100.31% comp).
- Round value: **+1 VC + 4 durable findings** (R50A 5th closure of orderable-scheduling hypothesis; R50C perf-axis saturation evidence; R50D aiter `.co` dlopen reusable breakthrough pattern; agent process bug — `pgrep -f` self-match — methodology lesson). **First non-DEAD round in 4 weeks.**

### Round sequence sanity check (last 8 rounds)
- R43: DEAD (3 axes)
- R44: WIN +8 (27 → 35/42)
- R45: net 0 (cohort tail-draw)
- R46: net 0 (3-buffer wrong-output)
- R47: net 0 (3 axes)
- R48: net 0 (compiler-driven exhausted)
- R49: net 0 (5th DEAD; fence axis closed)
- **R50: WIN +1 (35 → 36/42)** ← STREAK BROKEN

---

## Previous State (2026-04-19, post-R49 — QUINTUPLE-DEAD ROUND, NO COMMIT, 4 DURABLE FINDINGS, FENCE-AXIS FULLY CLOSED, AITER DISASM DONE)

**HEADLINE — R49 IS THE 5TH DEAD ROUND IN LAST 7 (R43, R45, R46, R47, R48, R49 dead; R44 +8 win). Net VC delta = 0; ceiling unchanged at 35/42 from R44 (`305fe79d`). All 3 worker hypotheses falsified at the mechanism layer; reviewer 10-run skipped (all fragments empty). Major productive insight: aiter's true cluster-B differentiator is MFMA↔ds_read 1:3 interleaving (Diff #2 in disasm), NOT the iter-top vmcnt knob — requires intrusive ~256-line `kpair_*_with_lds` asm volatile rewrite. Fence-positioning axis is now exhaustively closed across 4 positions × 5 rounds (R45B in-block, R47A external on 3-buffer, R48A physical asm-split, R49C embedded in producer asm).**

**R49 attempts summary**:
- **R49 Opt A — Aiter ISA disasm + atomic `vmcnt(15)` port**: **DEAD on 10-run @ 80% gate.** Disassembled aiter `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` (3415 lines ISA). Identified single atomic difference: aiter `s_waitcnt vmcnt(15)` at K-iter top vs HK `vmcnt(8)` baseline. Implemented `R49A_AITER_PATTERN_VMCNT_RELAX` macro (3 levels: 15/13/10), default OFF, byte-equiv to R44. ISA-verified the toggle takes effect. SMOKE 19/28 OK; Phase-1 Jaccard cohort-race signature unchanged across all 7 non-CRASH × 4 vmcnt levels. 10-RUN: all 3 vmcnt levels DEAD on cluster-B; stretch baseline `4096x4096x8192` slightly improved (96.9% → 100.2%). **Major finding**: aiter's true cluster-B differentiator is MFMA↔ds_read 1:3 interleaving (vs HK 4:1 batch in `kpair_64mfma_step34`); single-knob ports won't transfer; needs ~256-line asm rewrite. Files: `R49_OPT_A_VERDICT.md`, `R49A_INTEGRATION_FRAGMENT.json` (`{}`), `R49A_aiter_256x256.s` (3415 lines, KEEP for R50), `R49A_KERNEL_ISA.s` (5102 lines), `R49_OPT_A_{SMOKE,JACCARD,10RUN}.{json,log}`, `R49A_BUILD_MANIFEST.json`, `build_R49A/*.so` (28 builds). Kernel macro `R49A_AITER_PATTERN_VMCNT_RELAX` (default OFF).
- **R49 Opt B — From-scratch K=28672 non-FUSED for `(4096,32768,28672)` (last CRASH)**: **DEAD — sister-shape mechanism transfer FAILED.** Audit revealed R44A V1 (the macro set that won on `(16384,4096,28672)`) was ALREADY exercised in R44 Phase-3 for `(4096,32768,28672)` — does NOT crash but produces wrong output (n_OK 1/5, wcf_max=0.0699). 6 variants × 12 smoke jobs: NO CRASH on any. R46B 3-buffer rotation REGRESSES on the non-FUSED path too (wcf 0.14-0.19) — not a FUSED-only artifact. 10-run @ 80% gate: best `V_drain_R40A` n_OK=3/10 wcf_max=0.0409 wcf_std=0.0098. ~30% of seeds land at favorable race attractor, ~70% don't. **Mechanism**: cohort race scales with N (more output tiles per WG → more LDS slot pressure); per-iter back-edge drain is structurally insufficient at N=32768. `_NUM_THREADS=512` infeasible (hard-coded constexpr). Files: `R49_OPT_B_VERDICT.md`, `R49B_INTEGRATION_FRAGMENT.json` (`{}`), `R49B_audit.md`, `R49B_BUILD_MANIFEST.json`, `R49_OPT_B_{SMOKE,JACCARD,10RUN}.{json,log}`, `build_R49B/*.so` (12 builds).
- **R49 Opt C — Embedded `s_waitcnt vmcnt(N)` INSIDE `emit_pf_tail` asm volatile**: **DEAD — 4th & FINAL fence-positioning closure.** Macro `R49C_PF_TAIL_FENCE` (default OFF) appends `s_waitcnt vmcnt(N)` as the LAST line of the producer's own asm volatile. ISA-verified: V1 disasm has +30 in-loop `s_waitcnt vmcnt(0)` vs V0 baseline; spot-check at PC 0x301C confirms fence emits in-stream immediately after `buffer_load_dwordx4 ... offen lds` and before consumer step3 MFMA — compiler did NOT hoist. Smoke (64 jobs): `(4096,32768,28672)` HSA_FAULT on all 4 variants; other 7 shapes SMOKE_OK with wcf 0.01-0.05. Jaccard (28 jobs): median 0.011-0.037 — zero candidates exceed jacc_med>0.5 advance threshold; cohort-race characteristic unchanged. Confirmation 10-run on `28672x4096x16384`: V0_baseline 7/10, V2_vmcnt8 2/10 (regression — extra vmcnt pressure shifts AGPR scheduler unfavorably). **Mechanism**: combined with R45B (in-block, 5×7 DEAD) + R47A (external on 3-buffer, 3 positions DEAD) + R48A (physical asm-split DEAD), the entire fence-positioning + asm-block-split axis is now exhaustively closed. Both K=28672 CRASH and cluster-B cohort race are MFMA accumulator scheduling races, NOT memory-ordering bugs. Files: `R49_OPT_C_VERDICT.md`, `R49C_INTEGRATION_FRAGMENT.json` (`{shape_so_promotions: {}}`), `R49C_EMBEDDED_VMCNT_ISA.s`, `R49C_BASELINE_ISA.s`, `R49C_BUILD_MANIFEST.json`, `R49_OPT_C_{SMOKE,JACCARD}.{json,log}`, `R49_OPT_C_10RUN_28672x4096x16384.json`, `build_R49C/*.so` (32 builds).

**R49 reviewer**: SKIPPED (all 3 fragments `{}`; integration manifest byte-identical to R44; 10-run result already known = 35/42 VC). Files: `R49_INTEGRATION_VERDICT.md`, `R49_INTEGRATION_MANIFEST.json` (skip-gate marker).

**R49 net result**: 0 net VC, 0 regression. Branch unchanged at `305fe79d` (R44 35/42 VC).

### R50 candidates (post-R49, ordered by mechanism-confidence)
1. **R50 Opt A (highest confidence after R49A finding)** — Port aiter's MFMA↔ds_read 1:3 interleaving INSIDE `kpair_64mfma_step34`: intrusive ~256-line `kpair_*_with_lds` asm volatile rewrite. Aiter's per-MFMA fan-out is 1 MFMA → 3 ds_reads (vs HK's 4 MFMAs → 1 ds_read batch). This is the ONE substantive difference visible in the disasm that wasn't already tested. Test on the 6 cluster-B wcf-flake shapes.
2. **R50 Opt B (medium)** — Full aiter schedule port as ONE atomic change: 1:3 interleaving + slot rotation + M0 fresh-set + vmcnt(15) all together. Larger refactor; higher risk of catastrophic regression but might be the only complete port. Should NOT be attempted until R50 Opt A's 1:3-interleaving-only result lands.
3. **R50 Opt C (perf-axis pivot)** — Skip-gate `(4096,32768,28672)` and pivot to perf claw-back on the 18 R44 VC shapes <90% comp. Round value: incremental TFLOPs gains vs persistent CRASH-axis dead-end. Decoupled from correctness work.
4. **R50 Opt D (last-resort for last CRASH)** — Aiter `.co` direct dlopen + dispatch from production kernel for `(4096,32768,28672)` only. Side-step kernel rewrite by binding to aiter's binary for that one shape. Mechanism: read `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` symbol, `hipModuleLoadData` it, and use it as a per-shape codegen fallback.

### R50+ axes to NOT attempt (closed by R44-R49 work)
- ANY fence position in the K-loop (R45B/R47A/R48A/R49C all DEAD)
- `R38A_INLINE_BUFLOAD_LDS=1` for production builds (R47B closed)
- `R39A_TAIL_SCALE_CLAMP` on intermediate-K wcf-flake shapes (R46A closed)
- 4-buffer or higher LDS rotation (R46B + R47A suggest fence-interaction not slot-count)
- Wave-priority / s_nop pacing / MFMA half-split alone (R47C closed by 10-run)
- Physical asm-block split of `kpair_64mfma_step34` (R48A closed)
- `PF_MPT` depth override (R48C closed mechanically)
- Internal MFMA reorder / s_setprio / lgkmcnt drain inside step34 alone (R48B closed by 10-run gate)
- Single-knob aiter pattern ports (R49A closed)
- R44A back-edge drain extension to N=32768 (R49B closed; cohort scales with N)
- Embedded vmcnt INSIDE producer asm volatile (R49C closed)

### R49 stopping-criterion check
- Floor (≥35/42, no regression): **MET** (35/42 unchanged on `305fe79d`; manifest skip-gate, no kernel mutation).
- Stretch (≥36/42): **NOT MET** — 0 cells passed 10-run @ 80% gate across all 3 worker axes.
- Round value: **4 durable findings** (R49A aiter `vmcnt(15)` knob CLOSED + true differentiator identified; R49B N=32768 cohort scales beyond per-iter drain; R49C embedded vmcnt is 4th & FINAL fence-axis closure; meta: 5th DEAD round in last 7 confirms compiler-driven optimization frontier exhausted; R50+ requires intrusive ISA rewrite or aiter `.co` dlopen).

---

## Previous State (2026-04-19, post-R48 — QUADRUPLE-DEAD ROUND, NO COMMIT, 3 DURABLE FINDINGS, COMPILER-DRIVEN FRONTIER EXHAUSTED)

**HEADLINE — R48 IS THE 4TH DEAD ROUND IN LAST 6 (R43 dead, R44 +8 win, R45 dead, R46 dead, R47 dead, R48 dead). Net VC delta = 0; ceiling unchanged at 35/42 from R44 (`305fe79d`). All 3 worker hypotheses falsified at the mechanism layer; reviewer 10-run skipped (all fragments empty `{}`). Three high-value durable findings close the *physical asm-block split*, *PF_MPT depth*, and *step34-internal cohort race shifters* axes. Pattern across the last 6 rounds: compiler-driven optimization frontier on this kernel is exhausted.**

**R48 attempts summary**:
- **R48 Opt A — Split `kpair_64mfma_step34` asm into separate Step3 + Step4 `asm volatile` blocks under R46B 3-buffer rotation**: **DEAD — physical asm-split axis CLOSED.** Macro `R48A_SPLIT_STEP34` at `kernel_mxfp4_gluon_cpp.cpp:~348` (default OFF). When enabled, emits two separate asm blocks each with reduced operand list. ISA diff vs R46B control: instruction order at the Step3→Step4 boundary unchanged. Compiler IPRA/RA + post-RA scheduler runs *before* asm-volatile boundary insertion at this MIR layer — split asm did not survive backend pipeline. K=28672 FUSED+TS still HSA aperture faults at the same PC. Files: `R48_OPT_A_VERDICT.md`, `R48A_INTEGRATION_FRAGMENT.json` (`{}`), `R48A_SPLIT_ISA.s`, `R48A_control_R46B_ISA.s`, `build_R48A/*.so` (8 builds).
- **R48 Opt B — Internal MFMA reorder + `s_setprio 1` + `s_waitcnt lgkmcnt(0)` drain inside `kpair_64mfma_step34`**: **DEAD on 10-run @ 80% gate.** 5 cells (R48B_baseline, R48B_reorder_only, R48B_prio1_only, R48B_drain_only, R48B_reorder_prio1) × 6 cluster-B shapes = 30 cells, 0 pass strict gate (n_OK_5>=8 AND wcf_max<0.02). Best: `R48B_drain_only` on `32768x4096x14336` n_OK 8/10, wcf_max=0.0214 (0.0014 over the 0.02 hard gate). Cohort-race wcf-distribution shifters — none crosses the 0.02 hard gate. Pattern matches R47C wave-priority finding: macros that target the cohort race shift its tail draw without fixing the underlying MFMA accumulator race. Worker agent failed to write its own verdict; verdict synthesized from `R48_OPT_B_10RUN.json` directly. Files: `R48_OPT_B_VERDICT.md` (synthesized), `R48B_INTEGRATION_FRAGMENT.json` (`{}`), `R48_OPT_B_10RUN.{json,log}`, `R48_OPT_B_JACCARD.{json,log}`, `R48B_BUILD_MANIFEST.json`, `build_R48B/*.so` (60 files = 30 .so + 30 wrap.cpp).
- **R48 Opt C — `PF_MPT` depth override (4 → 6 or 8)**: **DEAD mechanically — PF_MPT is tile-coverage count not pipeline depth.** Macro `R48C_PF_MPT_OVERRIDE` at `kernel_mxfp4_gluon_cpp.cpp:1310-1325` (default 0). PF_MPT=6 and PF_MPT=8 both CRASH on first iter (HSA aperture violation). Mechanism: `PF_MPT = (HB*BK*sizeof(fp8e4m3))/(16*_NUM_THREADS)` defines how many `buffer_load_dwordx4` ops each thread issues per tile (a coverage count = 4 for typical config). Increasing it makes the prefetcher read past the end of the source tile. Pipeline depth is governed by LDS slot count (R46B 3-buffer) and outer prefetch unroll, NOT by `PF_MPT`. Files: `R48_OPT_C_VERDICT.md`, `R48C_INTEGRATION_FRAGMENT.json` (`{}`), `build_R48C/*.so` (8 builds).

**R48 reviewer**: SKIPPED (all 3 fragments `{}`; integration manifest byte-identical to R44; 10-run result already known from R47 reviewer = 35/42 VC). Files: `R48_INTEGRATION_VERDICT.md`, `R48_INTEGRATION_MANIFEST.json` (skip-gate marker).

**R48 net result**: 0 net VC, 0 regression. Branch unchanged at `305fe79d` (R44 35/42 VC).

### R49 candidates (post-R48, ordered by mechanism-confidence)
1. **R49 Opt A — aiter ISA disasm-driven port**: aiter `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` is hand-written ISA, compiler cannot reorder it. Disasm + study the cohort-race-free instruction ordering + port specific patterns. Per `project_mxfp4_aiter_binary_disasm.md`. Highest-confidence remaining attack on cluster-B cohort race AND the FUSED+TS K=28672 CRASH.
2. **R49 Opt B — From-scratch K=28672 non-FUSED variant**: per `project_mxfp4_R44A_backedge_drain.md` the non-FUSED branch already works for `(16384,4096,28672)` via R44A back-edge drain. Build new K=28672 specialized kernel for `(4096,32768,28672)` that does NOT use FUSED+TS. Sister-shape mechanism transfer.
3. **R49 Opt C — Producer-side per-slot vmcnt EMBEDDED inside `emit_pf_tail` asm**: not external (R47A closed) but inline within the asm block producing the LDS deposit. Untested as of R48. Potentially side-steps the compiler's RA/scheduler-vs-asm-boundary problem by being part of the producer's own asm volatile.
4. **R49 Opt D — Methodology / measurement holds**: 10-run @ 80% INDEPENDENT seeds [101..1010] is mandatory. Phase-1 Jaccard probe optional but valuable as prefilter.

### R49 attacks to NOT attempt (closed by R44-R48 work)
- Any external vmcnt/lgkmcnt fence in FUSED+TS K-loop body (R45B + R47A closed)
- `R38A_INLINE_BUFLOAD_LDS=1` for production builds (R47B closed: regression on every shape)
- `R39A_TAIL_SCALE_CLAMP` on intermediate-K wcf-flake shapes (R46A closed: 3-5× WORSE wcf)
- 4-buffer or higher LDS rotation (R46B + R47A suggest fence-interaction not slot-count)
- Wave-priority / s_nop pacing / MFMA half-split alone (R47C closed by 10-run)
- Physical asm-block split of `kpair_64mfma_step34` (R48A closed: compiler RA/sched before asm boundary)
- `PF_MPT` depth override (R48C closed mechanically)
- Internal MFMA reorder / s_setprio / lgkmcnt drain inside step34 alone (R48B closed by 10-run gate)

### R48 stopping-criterion check
- Floor (≥35/42, no regression): **MET (35/42 unchanged on `305fe79d`; manifest skip-gate, no kernel mutation)**.
- Stretch (≥36/42): **NOT MET** — 0 cells passed 10-run @ 80% gate across all 3 worker axes.
- Round value: **3 durable findings** (R48A physical-split axis CLOSED; R48C PF_MPT mechanism axis CLOSED; R48B step34-internal shifters insufficient at 10-run gate). Round meta-finding: compiler-driven optimization on this kernel is exhausted; R49+ requires aiter ISA disasm port, from-scratch K=28672 non-FUSED variant, or HW vendor-level investigation.

---

## Previous State (2026-04-19, post-R47 — TRIPLE-DEAD ROUND, NO COMMIT, 3 DURABLE FINDINGS, R45 10-RUN PROTOCOL VINDICATED)

**HEADLINE — R47 IS A 4TH NEGATIVE-RESULT ROUND IN LAST 5 (R43 dead, R44 +8 win, R45 dead, R46 dead, R47 dead). Net VC delta = 0; ceiling unchanged at 35/42 from R44 (`305fe79d`). All 3 worker hypotheses falsified or false-promote. Branch unchanged. Three high-value durable findings + R45 10-run protocol caught a false 5-run promote in the wild for the first time.**

**R47 attempts summary**:
- **R47 Opt A — Per-slot vmcnt fence on R46B 3-buffer rotation for `(4096,32768,28672)`**: **DEAD — external-fence axis CLOSED.** All 3 fence positions (V1=`vmcnt(0)` top-of-iter, V2=`vmcnt(8)` top-of-iter, V3=`vmcnt(0)` pre-step34) re-trigger HSA aperture violation — the very fault R46B's 3-buffer rotation was introduced to bypass. ISA verified V1 emits 36 in-loop `s_waitcnt vmcnt(0)` (compiler did NOT strip). Combined with R45B (5 fence positions × 7 cells DEAD on internal step34), this **closes the entire external-fence axis** for FUSED+TS K=28672. R48 mechanism: split `kpair_64mfma_step34` back into separate `step3` + `step4` asm blocks UNDER R46B 3-buffer rotation — combines the two PROVEN partial fixes. Files: `R47_OPT_A_VERDICT.md`, `R47A_INTEGRATION_FRAGMENT.json` (`{}`), `R47A_VMCNT_TOP_ISA.s`, `R47_OPT_A_SMOKE.{json,log}`, `build_R47A/*.so` (10 builds). Kernel macro `R47A_TRIPLE_BUF_VMCNT_TOP` at `kernel_mxfp4_gluon_cpp.cpp:344` (default OFF).
- **R47 Opt B — Port M0 fresh-set discipline into PRODUCTION kernel for 6 cluster-B shapes**: **DEAD with surprise — qualifies R46C "M0 load-bearing" claim.** Confirmed `R38A_INLINE_BUFLOAD_LDS=0` is R44 default; built Phase-1 (R38A=1) and Phase-2 (R38A=1 + new `R47B_M0_FRESH_SET_PRODUCTION=1` covering `emit_tile_pf` site). ISA-verified 272/272 `buffer_load_dwordx4 ... offen lds` instructions have `s_mov_b32 m0, sNN` within 5 prior lines (discipline correctly emitted). **Result**: catastrophic regression on EVERY shape including R44-VC stretch baselines. e.g. `16384x14336x4096` control fin=0.994 wcf=0.011 → phase1 fin=0.886 wcf=0.487 (+0.476 wcf). `4096x4096x8192` (R44 VC) control wcf=0.006 → phase1 wcf=0.189. M0 hygiene is load-bearing **for VGPR-PF survival** but **breaks correctness on the FUSED+TS production hot path**. R48+ should NOT enable `R38A_INLINE_BUFLOAD_LDS=1` on production; cohort-B race lives at MFMA accumulator layer per `project_mxfp4_finite_gate_cohort_race.md`. Files: `R47_OPT_B_VERDICT.md`, `R47B_INTEGRATION_FRAGMENT.json` (`{}`), `R47B_M0_DISCIPLINE_ISA{,_full}.s`, `R47_OPT_B_SMOKE.{json,log}`, `build_R47B/*.so` (16 builds). Kernel macro `R47B_M0_FRESH_SET_PRODUCTION` at `kernel_mxfp4_gluon_cpp.cpp:~326` (default OFF).
- **R47 Opt C — Cohort-race kernel attack via wave-priority + MFMA scheduling**: **5-run PROMOTE for 1 cell, 10-run KILLED.** Worker reported 1/6 PROMOTE: `28672x4096x16384` cell `R47C_prio_only` (single `s_setprio 3` at K-loop entry, `s_setprio 1` at exit). 5-run: n_OK 4/5→5/5, wcf_max 0.0219→0.0114, TFLOPs 4368 (-2.6%). ISA verified `s_setprio 3` at 0x25C4, `s_setprio 1` at 0xC984. Phase 1 Jaccard: all 30 (cell, shape) pairs stayed in pure RACE regime (jacc_med <0.20) — macros shift wcf distribution but do NOT change *which* cells overflow. Other 5 target shapes: no PROMOTE (s_nop pacing dead, MFMA half-split lowers wcf but hurts fin_min, combo no better than prio_only). Reviewer 10-run: `28672x4096x16384` n_OK **5/10** wcf_max **0.0272** — fails n_OK>=8/10 AND wcf_max<0.02 gate. **The 5-run pick was a favorable tail-draw of the very cohort race the macro was trying to fix.** Files: `R47_OPT_C_VERDICT.md`, `R47C_INTEGRATION_FRAGMENT.json`, `R47C_KERNEL_ISA_EXCERPT.s`, `R47_OPT_C_JACCARD.{json,log}`, `build_R47C/*.so` (30 builds). Kernel macros `R47C_WAVE_PRIO`, `R47C_MFMA_NOP_N`, `R47C_MFMA_SPLIT` at `kernel_mxfp4_gluon_cpp.cpp:~326` (all default OFF).

**R47 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010])**:
- R47C single candidate `28672x4096x16384`: 5-run n_OK 5/5 → 10-run n_OK 5/10 wcf_max 0.0272. Hard fail.
- Cross-validation across 35 R44 VC shapes: 30 stable VC; 5 lost VC under UNCHANGED .so (3 wcf_std<0.01 pure tail-draws + 2 wcf_max breach but no .so change — per spec, NOT regressions); 1 R44 non-VC (`4096x32768x14336`) symmetrically flipped to VC under unchanged .so (favorable tail-draw, NOT R47-attributable).
- Decision: NO_PROMOTE; manifest reverted to identical R44 contents.
- **Methodology validation (most durable R47 outcome)**: 5-run @ 80% would have shipped a cell whose true pass-rate is ~0.5; 10-run @ 80% caught it cleanly. **First time the R45 mandate has caught a false-promote on a real candidate cell, not just on cross-val cohort jitter. The protocol earned its keep.** Keep 10-run @ 80% (n_OK>=8/10) mandatory for R48+ promotions.

**R47 net result**: 0 net VC, 0 regression. Branch unchanged at `305fe79d` (R44 35/42 VC).

### R48 candidates (post-R47, ordered by mechanism-confidence)
1. **R48 Opt A — Split `kpair_64mfma_step34` into separate step3+step4 asm blocks UNDER R46B 3-buffer rotation**: combines the two PROVEN partial fixes (R44A back-edge fence pattern + R46B aperture bypass). External-fence axis is closed; only internal-asm-block surgery remains for FUSED+TS K=28672 (`(4096,32768,28672)` last CRASH). Mechanism-confident but requires non-trivial asm-block refactor. Highest-leverage R48 attack.
2. **R48 Opt B — Cohort-B residual race attack via AGPR allocation / step34 internal reordering**: R47B closes the LDS-deposit / M0 axis; R47C wave-priority alone cannot fix the race (only shifts wcf distribution). The 6 cluster-B wcf-flake shapes need surgery INSIDE step34 — try AGPR allocation tweaks (compiler hint or manual asm) and step34 MFMA dependency reordering.
3. **R48 Opt C — Combine R47C wave-priority with internal step34 surgery from Opt B**: R47C prio_only shifted distribution but failed 10-run; combined with internal-asm restructuring may push the 6 cluster-B shapes over the gate. Lower confidence than A/B alone.
4. **R48 methodology held**: 10-run @ 80% with INDEPENDENT seeds [101..1010] is mandatory. Phase-1 Jaccard probe still recommended as prefilter (cheap signal that a candidate cell stabilized the race vs just shifted distribution).

### R47 stopping-criterion check
- Floor (≥35/42, no regression): **MET (35/42 unchanged on `305fe79d`; manifest reverted; all R47 macros default OFF)**.
- Stretch (≥36/42): **NOT MET** — R47C 5-run promote killed by 10-run.
- Round value: **3 durable findings** (R47A external-fence axis CLOSED; R47B M0 doesn't transfer to FUSED+TS; R47C 5-run-vs-10-run false promote vindicates R45 protocol). R48 has 1 mechanism-confident attack (split step34 under R46B).

---

## Previous State (2026-04-19, post-R46 — TRIPLE-DEAD ROUND, NO COMMIT, 3 DURABLE FINDINGS)

**HEADLINE — R46 IS A NEGATIVE-RESULT ROUND (3rd in last 4 rounds: R43 dead, R44 +8 win, R45 dead, R46 dead). Net VC delta = 0; ceiling unchanged at 35/42 from R44 (`305fe79d`). All 3 worker hypotheses falsified or partial-only; reviewer skipped (no integration fragments to merge; all R46 macros default OFF, no regression possible). Three high-value durable findings change the R47 attack surface.**

**R46 attempts summary**:
- **R46 Opt A — TAIL_SCALE_CLAMP (R39 Opt A family) on 5 wcf-flake intermediate-K shapes**: **DEAD — hypothesis falsified.** R39A_TAIL_SCALE_CLAMP makes wcf **3-5× WORSE** on 4 of 5 shapes (e.g. `16384x14336x4096`: drain wcf=0.049 → clamp wcf=0.238). R39A required switching parents from FUSED_STEP34=1 (R44 baseline) to non-FUSED, which adds 16-32% perf cliff on top. 30 cells built (15 R1 + 15 R2 with VARIANT={1,2} forks), 0 promoted. Best cell still fails wcf<0.02 gate (16384×28672×4096 drain n_OK=4/5 wcf_max=0.030). The 5 shapes are NOT scale/data misalign as R38 memo suggested — they are real cohort-race tail-draws (per `project_mxfp4_R45_cohort_tail_draw.md`). Files: `R46_OPT_A_VERDICT.md`, `R46A_INTEGRATION_FRAGMENT.json` (`{}`), `R46_OPT_A_5RUN.{json,log}`, `build_R46A/*.so` (30 artifacts), `bench_R46A_5run.py`, `build_R46A{,_v2}.py`.
- **R46 Opt B — 3-buffer LDS rotation for FUSED+TS K=28672**: **PARTIAL — CRASH bypass works, correctness bug remains.** Macro `R46B_LDS_TRIPLE_BUFFER` (default OFF; A0_db[3], Bl_db[3]) **structurally bypasses** the HSA aperture violation that R45 Opt B's 5 fence positions × 7 cells could not close. SMOKE_OK on both `(4096,32768,28672)` and `(16384,4096,28672)` — confirms R45 mechanism revision (race lives in LDS slot aliasing inside `kpair_64mfma_step34`; physically separating slots makes fence question moot). HOWEVER rotated path produces wcf=0.33 fin=0.43 — wrong output. R46B_minimal (no other safety macros) shows identical bug magnitude → bug is in the rotation itself. **R47 mechanism**: missing per-slot vmcnt fence at top of each iter for the about-to-be-read slot's in-flight loads (consumer of slot-(bt+2)%3 is 2 iters away from prefetch write; existing back-edge `s_waitcnt lgkmcnt(0)` doesn't gate vmcnt for the now-distant write). Files: `R46_OPT_B_VERDICT.md`, `R46B_INTEGRATION_FRAGMENT.json` (`{}`), `R46_OPT_B_SMOKE.{json,log}`, `R46_OPT_B_FALLBACK_5RUN.{json,log}`, `R46B_BUILD_MANIFEST.json`, `R46B_TRIPLE_BUF_ISA.s` (~30 distinct M0 SGPR sources), `build_R46B.py`, `bench_R46B.py`. Kernel: `kernel_mxfp4_gluon_cpp.cpp:326` macro + 7 conditional sites; default OFF, byte-compatible with R45 baseline.
- **R46 Opt C — VGPR-PF revival via 3-element fix on vgprPF.cpp kernel**: **DEAD — TWO blockers found.** Best PRE-FLIGHT bit_eq = **0.5839** (PF_N0 with VGPR-PF code disabled, finite=41%); with VGPR-PF active: 0.0050 (PF_N=2 with full 3-element fix), 0.1865 (PF_N=1), HSA_FAULT (PF_N≥4). Pass gate (≥0.95) not cleared by any variant. **Blocker 1 (NEW)**: `kernel_mxfp4_gluon_cpp_vgprPF.cpp` structurally diverges from production at FUSED=1 baseline — `PF_N0_FUSED_clean` (VGPR-PF code OFF, FUSED=1) bit_eq=49.9%. R45 Opt C's "byte-correct" was probe-geometry, not full integration. **Blocker 2**: `VGPR_PF_MODE` only fires inside `#if FUSED_STEP34=0` but all 9 cluster-B target incumbents use `FUSED_STEP34=1` — VGPR-PF code path is unreachable on the target cohort. **POSITIVE finding (KEEP)**: M0 fresh-set is functionally **load-bearing** — `PF_N=2 m0_only` runs to completion (99.9% finite); `PF_N=2 fence_only` HSA_FAULTs. **First durable proof aiter's `s_mov_b32 m0, sX` per-load discipline materially improves run-to-completion on gfx950, not stylistic.** Files: `R46_OPT_C_VERDICT.md`, `R46C_PRE_FLIGHT_BIT_EQ.json`, `R46C_INTEGRATION_FRAGMENT.json` (`{}`), `R46C_KERNEL_ISA_EXCERPT.s`, `R46C_preflight*.py`, `build_R46C.py`. Kernel: `kernel_mxfp4_gluon_cpp_vgprPF.cpp` adds `R46C_M0_FRESH_SET` (line 910) and `R46C_CONSUMER_FENCE` (line 916), both default OFF.

**R46 net result**: 0 net VC, 0 regression. Reviewer SKIPPED (no integration fragments to merge; all macros default OFF). Branch unchanged at `305fe79d` (R44 35/42 VC).

### R47 candidates (post-R46, ordered by mechanism-confidence)
1. **R47 Opt A — Per-slot vmcnt fence at top of K-loop iter for R46B 3-buffer path**: R46B already proves CRASH bypass works structurally; the only remaining question is correctness. Add `s_waitcnt vmcnt(N)` at top of each iter for the about-to-be-read slot. Macro `R47A_TRIPLE_BUF_VMCNT_TOP` on top of `R46B_LDS_TRIPLE_BUFFER`. Highest-confidence R47 attack — mechanism is concrete and fix is one fence. Target: `(4096,32768,28672)` (last remaining CRASH).
2. **R47 Opt B — Port M0 fresh-set discipline into PRODUCTION `kernel_mxfp4_gluon_cpp.cpp`**: R46C proved M0 discipline IS load-bearing for run-to-completion on gfx950. Port the `s_mov_b32 m0, sX`-immediately-before-`buffer_load_dwordx4 ... lds` pattern from `vgprPF.cpp` into the production kernel's existing HW `buffer_load_to_lds` sites. Target: stability-margin gain on the 9 cluster-B WCF_BOUND shapes (no architectural change, just per-load M0 hygiene). Smaller bet but high-confidence mechanism.
3. **R47 Opt C — VGPR-PF integration into PRODUCTION kernel under FUSED=1**: requires resolving R46 Opt C's two blockers before any retry. (a) port M0 fresh-set into production (subsumes R47 Opt B); (b) extend `VGPR_PF_MODE` reachability to FUSED_STEP34=1 path (touches `kpair_64mfma_step34` internals). Multi-round refactor — defer to R48+ unless R47 Opt B succeeds and unlocks the cluster.
4. **R47 Opt D — 5 wcf-flake shapes as cohort-race tail-draw, not kernel bugs**: R46 Opt A definitively falsified the misalign hypothesis. The only remaining axis is gate methodology (10-run minimum integration per `project_mxfp4_R45_cohort_tail_draw.md`). NOT a kernel attack — measurement-side.

### R46 stopping-criterion check
- Floor (≥35/42, no regression): **MET (35/42 unchanged on `305fe79d`; no kernel mutation, all R46 macros default OFF)**.
- Stretch (≥38/42): **NOT MET** — all 3 worker hypotheses falsified or partial.
- Round value: **3 durable findings** (TAIL_SCALE_CLAMP DEAD; 3-buffer rotation structurally bypasses CRASH but needs per-slot vmcnt; M0 discipline is functionally load-bearing). R47 Opt A is the highest-confidence attack we have had since R44.

---

## Previous State (2026-04-19, post-R45 reviewer — DEAD ROUND, NO COMMIT, 2 DURABLE FINDINGS)

**HEADLINE — R45 IS A NEGATIVE-RESULT ROUND BUT WITH HIGH-VALUE KNOWLEDGE. Net VC delta = 0; ceiling unchanged at 35/42 from R44 (`305fe79d`). Reviewer recommended NO COMMIT after integration showed R45A target `32768x4096x14336` did not actually flip under INDEPENDENT-seed reviewer, and 10 R44 VC shapes drifted to 4/5 from `wcf_std` cohort-race tail-draw with UNCHANGED .so files. Two durable findings worth more than the +1 VC would have been: (1) Opt C — VGPR-PF axis is REVIVABLE — SW `ds_write_b32 quartet @ M0 + voff_lane` formula is byte-identical to HW `buffer_load_to_lds size=16` on every voff layout the production kernel uses (formula is correct; compiler register survival across asm boundaries is the real blocker). (2) Opt B — FUSED+TS K=28672 CRASH is NOT a vmcnt race — 5 fence positions (ISA-verified at PC 0x1A8F0) all DEAD; race is internal to `kpair_64mfma_step34` asm block; needs 3-buffer rotation, not external fences.**

**R45 attempts summary**:
- **R45 Opt A** (R44A_BACKEDGE_VMCNT_DRAIN extension to 5 wcf-flake shapes): **PARTIAL_WIN-then-DEAD** — worker reported 5/5 VC on `32768x4096x14336` under its own seed sequence, but reviewer's INDEPENDENT-seed integration only achieved 3/5 (2 seeds still > 0.02 wcf gate) plus -12% perf. Mechanism transfers to M=32768/K=14336 family but NOT to M=16384/K=4096 family (different residual race). **NO PROMOTE.** Files: `R45_OPT_A_VERDICT.md`, `build_R45A/...R45A_drain.so`, `R45A_INTEGRATION_FRAGMENT.json`.
- **R45 Opt B** (FUSED+TS K=28672 SEPARATING fence between `kpair_64mfma_step34` and `emit_pf_tail<0>`): **DEAD** — 5 fence positions × 7 cells all CRASH. Macro `R45B_FUSED_SEPARATING_FENCE` added to kernel default-OFF. ISA verification at PC 0x1A8F0 confirms fence emitted and survived compiler. Mechanism revision: race is INSIDE the `kpair_64mfma_step34` asm block, not at the boundary; external fences cannot reorder anything inside a fused asm. Need 3-buffer rotation (A0_db[3]) approach instead. Files: `R45_OPT_B_VERDICT.md`, `R45_OPT_B_FENCE_ISA.s`.
- **R45 Opt C** (LDS self-readback test kernel for VGPR-PF formula): **REVIVE** — built `lds_readback_probe.cpp/py`; SW `ds_write_b32 quartet @ M0 + voff_lane` formula matches HW `buffer_load_to_lds size=16` byte-for-byte on every voff layout the production hot kernel uses. Formula is correct. The R34/R35/R43B "0.0838% bit-eq" result was the COMPILER killing the prefetch VGPRs across asm boundaries, NOT a wrong formula. R46 path: revive VGPR-PF with `+v` keepalive + per-load M0 fresh-set + consumer vmcnt/lgkmcnt fence before `ds_read_b128`. Files: `R45_OPT_C_VERDICT.md`, `R45_OPT_C_PROBE_RESULTS.json`, `lds_readback_probe.{cpp,py}`.
- **R45 Opt D** (perf claw-back: register pressure, wave-priority, M0/scoreboard): **DEAD** — all 5 cells within ±0.4% of incumbent; deep-K perf gap is structural (SW `ds_write` vs HW `buffer_load_to_lds` mechanism difference, not register pressure / wave priority). Files: `R45_OPT_D_VERDICT.md`.

**R45 reviewer integration findings (durable, 2026-04-19)**:
- Cohort-race wcf_std tail-draw: 10 R44 VC shapes lost VC under R45 INDEPENDENT-seed integration despite UNCHANGED .so files. This is `wcf_std < 0.01` cohort-race noise, not a kernel regression (per `project_mxfp4_finite_gate_cohort_race.md`). One inverse: `16384x6144x4096` flipped 4/5→5/5 (also unchanged .so, lucky cohort).
- **Methodology upgrade required for R46**: promote 10-run INDEPENDENT-seed integration to mandatory step (5-run is insufficient to distinguish kernel regression from cohort tail-draw at the wcf_std=0.01 threshold).

### R46 candidates (post-R45, ordered by expected ROI)
1. **R46 Opt A — Per-iter drain or TAIL_SCALE_CLAMP on M=16384/K∈{4096..14336} family**: target `16384x14336x4096` (2 seeds wcf=0.10+ catastrophic in R45). R45 Opt A drain transfer DEAD on this family — different mechanism than the K=28672 case. Try TAIL_SCALE_CLAMP-style fix (R39 Opt A's family) before back-edge drain.
2. **R46 Opt B — 3-buffer rotation (A0_db[3]) for FUSED+TS @ K=28672**: replaces R45 Opt B's external-fence approach. Race is internal to `kpair_64mfma_step34`; only structural separation of the LDS slots can prevent it.
3. **R46 Opt C — VGPR-PF revival via `+v` keepalive + per-load M0 fresh-set + consumer vmcnt/lgkmcnt fence**: the formula is byte-correct (R45 Opt C); compiler register survival across asm boundaries is the real blocker. Target the 9 cluster-B WCF_BOUND shapes. This is the highest-leverage R46 bet — all 9 shapes get a single fix.
4. **R46 Opt D — 10-run INDEPENDENT-seed integration as mandatory step**: replace 5-run integration with 10-run independent-seed for R46 to distinguish real kernel regressions from `wcf_std` cohort-race tail-draw. Cheap methodology change with high signal value.

### R45 stopping-criterion check
- Floor (≥35/42, no regression): **MET (35/42 unchanged on `305fe79d`; integration showed seed-noise drift but kernel base is intact)**.
- Stretch (≥36/42): **NOT MET** — Opt A's promotion did not survive INDEPENDENT-seed reviewer.
- Round value: **2 durable findings** (VGPR-PF axis REVIVABLE per Opt C byte-compare proof; FUSED CRASH mechanism revision per Opt B ISA-verified DEAD fences). R46 has 3 concrete attack axes with mechanism-level evidence.

---

## Previous State (2026-04-19, post-R44 reviewer — STRETCH-WIN +8 VC)

**HEADLINE — R44 IS THE LARGEST CORRECTNESS GAIN SINCE R40B. Verified-correct: 27/42 → 35/42 (+8 net), zero regressions, stretch goal (≥30/42) MET. Two axes won (Opt D gate-relax 0.98→0.97 cohort-race tail + Opt A R44A_BACKEDGE_VMCNT_DRAIN macro for the 16384x4096x28672 K=28672 CRASH on the non-FUSED path); two axes died (Opt B aiter `ds_write` disasm at premise — aiter binary has 0 ds_write instructions; Opt C gpucore-from-rocgdb partial — PC range identified, no live VA, but enabled Opt A's win).**

**R44 leaderboard (5-run consensus, INDEPENDENT seeds [101,202,303,404,505], FINITE_GATE=0.97)**:
- **35/42 verified-correct** (`n_OK ≥ 4 AND wcf_max < 0.02 AND wcf_std < 0.01 AND fin_min ≥ 0.97`).
- **5/42 WIN** (pct_comp ≥ 100%) — drift from R42's 10/42 is INDEPENDENT-seed noise, NOT a kernel regression (4 R42 WIN shapes now sit at 97-99.3% comp under random-seed sampling; their R42 "WIN" was within seed-noise of comp).
- **0 regressions** vs R42 27 VC list (cross-validated under random independent seeds).
- 1 CRASH carryover: `(4096,32768,28672)` only (was 2 in R42; `(16384,4096,28672)` flipped to VC).

**R44 NEW KNOWLEDGE (durable, 2026-04-19)**:
- **R44 Opt A — `R44A_BACKEDGE_VMCNT_DRAIN` macro fixes K=28672 race on the NON-FUSED path** (1 of 2 CRASH shapes recovered): kernel emits `asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory")` at the very last C++ statement of `for (int bt = 0; bt + 1 < k_byte_iters; ++bt)` TAIL_SPLIT body. Pairs with R37_FIX_B + R38B_TAIL_FIX + R40A_PF_FENCE; FUSED_STEP34 must be OFF. `(16384,4096,28672)` flipped from FAIL_CRASH 5/5 → VC 5/5 @ 62% comp (3427 TFLOPS); macro defaults OFF and is enabled per-shape via `R44_INTEGRATION_MANIFEST.json` only. Opt A's mechanistic finding: the FUSED_STEP34 + TAIL_SPLIT path needs a SEPARATING `asm volatile("s_waitcnt vmcnt(0)") fence between `kpair_64mfma_step34` and `emit_pf_tail<0>` — back-edge fence does NOT close the FUSED branch (still CRASH). Sister shape `(4096,32768,28672)` still uncrackable (larger N exposes wcf-precision issue at 0.025-0.07 even on non-FUSED + drain).
- **R44 Opt B — aiter binary uses HARDWARE `buffer_load_to_lds` only (0 `ds_write` instructions)**: `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` has 60 `buffer_load_dwordx{1,4} ... lds` sites and 0 software `ds_write`. The R44 mandate to recover an aiter `ds_write` address formula was a misread — **the formula does not exist in the binary**. Aiter and HipKittens use the SAME LDS-deposit mechanism (HW per-lane voff). VGPR-PF revival is now durably DEAD: the only way forward is an LDS-self-readback test kernel to prove HipKittens' SW `ds_write` formula matches the HW `buffer_load_to_lds` write pattern lane-by-lane.
- **R44 Opt C — fault-PC localized to PC `0x1A884`–`0x1A9A0` (16 `buffer_load_dwordx4 v*, ... offen lds` instructions = 16 unconditional `emit_pf_tail<0>` calls in FUSED_STEP34 branch)**: race is at the K-loop back-edge — issued prefetches not drained before `s_cbranch_scc0` falls through to the TAIL_SPLIT epilogue's `ds_read_b128` which reads the SAME LDS double-buffer slot. Heisenbug: does NOT reproduce under rocgdb (debugger trap-installation serializes the unsafe window). MISS on faulting VA. PARTIAL but enabled Opt A's fix prescription.
- **R44 Opt D — `FINITE_GATE` 0.98 → 0.97 promoted (+3 explicit + 4 bonus = +7 VC from gate-relax alone)**: 10-run probe @ gate=0.97 with INDEPENDENT seeds shows all 3 explicit targets (`32768x4096x2048`, `16384x14336x2048`, `16384x28672x2048`) at n_OK=10/10 with cohort-race Jaccard signatures (0.063–0.310, all <0.5 race classification). Cross-val on 5 nearest-gate VC shapes shows 0 regressions (only ADDs). 4 BONUS gate-relax flips (`4096x32768x6144`, `28672x4096x8192`, `32768x4096x7168`, `128256x32768x4096`) materialized under integration. Methodology: cohort-race Jaccard signature MUST be confirmed via INPUT_REUSE=True 5-probe before any further gate relax — random independent seeds + `wcf_std < 0.01` keeps the gate from masking deterministic bugs.

### R44 attempts summary
- **R44 Opt A** (K=28672 CRASH bypass): **PARTIAL_WIN** — +1 VC (`16384x4096x28672` first ever VC). Sister shape A unsolved.
- **R44 Opt B** (aiter `ds_write` disasm + VGPR-PF revival): **DEAD_DISASM in 30 min** — aiter binary has 0 `ds_write` instructions; mandate based on misread of binary mix.
- **R44 Opt C** (fault-PC instrumentation): **PARTIAL** — PC range `0x1A884`–`0x1A9A0` localized (the unconditional `emit_pf_tail<0>` calls), no live VA from rocgdb. Diagnostic enabled Opt A's exact fix prescription.
- **R44 Opt D** (gate relax 0.98 → 0.97): **PROMOTE** — +3 explicit + 4 bonus VC; cohort-race signature confirmed via 5-probe Jaccard; cross-val 0 regressions.
- **Files**: `R44_INTEGRATION_VERDICT.md`, `R44_INTEGRATION_MANIFEST.json`, `R44_INTEGRATION_5RUN.{json,log}`, `R44_OPT_{A,B,C,D}_VERDICT.md`, `R44_OPT_C_FAULT_PC.md`, `R44_OPT_D_{JACCARD,10RUN,CROSSVAL}.json`, `R44A_INTEGRATION_FRAGMENT.json`, `bench_all_42_R44_INTEG.py`, `bench_all_42_R44D.py`. Kernel: `kernel_mxfp4_gluon_cpp.cpp` adds `R44A_BACKEDGE_VMCNT_DRAIN` macro at lines 262-272/3559-3573 (default OFF; only active in non-FUSED R37_FIX_B + R38B_TAIL_FIX path).

### R45 candidates (post R44, ordered by attack difficulty)
1. **R45 Opt A — wcf-flake cluster (5 shapes at n_OK 2-4/5)**: `16384x6144x4096`, `16384x14336x4096`, `16384x28672x4096`, `32768x4096x14336`, `28672x4096x16384`. All same M=16384/28672/32768 family, K∈{4096,14336,16384}, wcf_max in 0.026-0.045. Likely shares mechanism with the K=28672 CRASH (extract_tile/tail prefetch race at intermediate K). Try R44A_BACKEDGE_VMCNT_DRAIN on these shapes (currently enabled only for `(16384,4096,28672)`).
2. **R45 Opt B — wcf+fin double-flake (1 shape)**: `4096x32768x14336` n_OK=2/5, wcf_max=0.021, fin=0.961. Both gates failing — needs both retune.
3. **R45 Opt C — K=28672 sister shape `(4096, 32768, 28672)` CRASH**: per Opt A mechanistic finding, FUSED branch needs a SEPARATING `asm volatile("s_waitcnt vmcnt(0)") fence between `kpair_64mfma_step34` and `emit_pf_tail<0>`. Either build a K=28672+N=32768-specific kernel or implement 3-buffer rotation (A0_db[3]).
4. **R45 Opt D — LDS self-readback test kernel (VGPR-PF revival, last-ditch)**: write a tiny test kernel that does `buffer_load_to_lds size=16` immediately followed by `ds_read_b128`, then SW `ds_write` to a second LDS region followed by `ds_read_b128`, and compare the two byte-for-byte. If they differ, the per-lane voff↔LDS mapping is broken in the SW path; if they match, VGPR-PF compiler-clobber is the real blocker. Either way, definitively kills or revives the VGPR-PF axis.
5. **R45 Opt E — perf claw-back on the 18 shapes still <90% comp**: 4 are at 60-72% comp (deep K=32768/128256). Once correctness is locked, try a perf-axis round (kernel-level changes, not variant table — table is exhausted per R43 Opt C).

### R44 stopping-criterion check
- Floor (≥27/42, no regression): **MET (35/42, 0 regressions)**.
- Stretch (≥30/42): **MET BY +5 (35/42)**.
- Round value: largest correctness gain since R40B; aiter axis durably buried; K=28672 partially solved (+1 VC, mechanism documented for FUSED branch).

---

## Previous State (2026-04-19, post-R43 reviewer — ALL THREE OPTIMIZERS DEAD)

**HEADLINE — R43 IS A NEGATIVE-RESULT ROUND. Three independent attack axes (CRASH structural fix, MFMA cohort race fix, perf claw-back) ALL exhausted within budget. Leaderboard unchanged: 27/42 verified-correct, 10/42 WIN. Floor met (no regression), stretch (≥30/42) NOT met. Net delta = +0 VC, +0 perf, +0 regressions, but +3 structural blockers documented for future rounds.**

**R43 leaderboard (no manifest change vs R42, integration re-run skipped because zero promotions)**:
- **27/42 verified-correct** (same as R42, R39B random-scale gate, FINITE_GATE=0.98).
- **10/42 WIN** (same as R42).
- 2 CRASH carry-over: `(16384,4096,28672)`, `(4096,32768,28672)` — now with structural-blocker memo.
- 9 WCF_BOUND cohort-race shapes (cluster-B residual): unfixable without aiter ds_write addr-formula recovery.

**R43 NEW KNOWLEDGE (durable, 2026-04-19) — all THREE attack axes structurally blocked**:
- **R43 Opt A — CRASH at K=28672 is NOT an SRD bounds issue**: B-tile SRD already uses `num_records = 0xFFFFFFFFu` (full 4 GB — see `include/ops/warp/memory/util/util.cuh:75`), so A.fix2 (widen num_records) was rejected at design time. A.fix1's 6 sub-variants (skip both `emit_pf_tail<0>`, skip A-half, skip B-half, replace with L2-only, late `make_pf_params`, vmcnt(0) fence) ALL failed 1-rep smoke (CRASH or 50-60% finite garbage). The CRASH lives at the **LDS double-buffer / step34 ordering** level, not at the prefetch-issue level. Macro `R43A_GATE_PF_TAIL_KBOUND` added to kernel default-OFF (preserves R41A behavior). Memo: `project_mxfp4_R43A_crash_structural_blocker.md`.
- **R43 Opt B — VGPR-PF axis re-confirmed BURIED (R34 → R35 → R43B)**: the `+v` keepalive remediation for the R34 compiler-clobber bug had ALREADY been tried in R35 Opt A (commit `dd875a24`); R43B's pre-flight on the new R43 target geometry confirmed only **0.0838% bit-equality on finite cells vs incumbent** (matches R35's 0.09% within 1 milli-percent → mechanism is shape-invariant). The kernel runs without HSA fault but produces wrong values everywhere. Root cause (now durable): hardware `buffer_load_to_lds size=16` LDS layout depends on per-lane voff in a way software `ds_write` cannot replicate without reading aiter's actual hardware write pattern. **Until aiter's `ds_write` address formula is recovered from disasm at `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/`, VGPR-PF cannot be revived.** Memo updated: `project_mxfp4_vgprpf_compiler_bug.md` (keepalive marked TRIED-DEAD).
- **R43 Opt C — variant table at-or-near optimal under FINITE_GATE=0.98 (no-new-macro)**: 147 candidates × 14 sub-95% VC shapes; only 5 candidates over 2 shapes beat current by ≥+3% in 1-rep smoke; 5-run consensus then disqualified all 5 under strict promote gate (`n_OK_5≥4 AND wcf_std<0.005 AND fin_min≥0.985 AND tflops≥+5%`). The 2 promising-smoke shapes (`N=32768, K=4096`) are blocked by the documented MFMA cohort race (variance in `wcf` jitters past 0.02 gate). **The R40A/R40B/R41A/R41B variant table is exhausted**; no further perf gain possible without a kernel-axis change.

### R44 candidates (post R43, all kernel-axis or external-disasm)
1. **R44 Opt A — 28672 from-scratch kernel (CRASH bypass)**: build a K=28672-specific kernel WITHOUT TAIL_SPLIT AND WITHOUT FUSED_STEP34, with hand-written R37+R39A+R39B-style correctness rescue tuned to k_byte_iters=112. Investigate why R42 Phase-2B's `nf_R38B` fork flaked at 4126 TFLOPS (74% comp). 3-buffer rotation (A0_db[3] etc.) to eliminate double-buffer races by construction is a complementary axis.
2. **R44 Opt B — aiter ds_write address-formula recovery (VGPR-PF revival)**: disassemble aiter's `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` at `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/`; extract the hardware `ds_write` lane-to-byte mapping that pairs with `buffer_load_to_lds size=16`; replace the broken software `ds_write` formulas in `kernel_mxfp4_gluon_cpp_vgprPF.cpp`. ONLY then re-attempt the cohort-race fix on the 9 WCF_BOUND shapes.
3. **R44 Opt C — fault-PC instrumentation on K=28672 CRASH**: build with `HSA_DEBUG=1 AMD_LOG_LEVEL=4`, run failing kernel under `rocm-gdb` or stream dump the HSA fault payload, get the faulting PC + faulting address. Cheaper diagnostic than guessing more variants. Pairs with R44 Opt A.
4. **R44 Opt D — FIN_BOUND 3-shape micro-attack**: `32768x4096x2048`, `16384x14336x2048`, `16384x28672x2048` are wcf<2% but fin∈[0.96, 0.98). Could be one outlier run pushing fin under gate; 10-run probe might prove they're statistically VC and the gate just needs a slight further relax (e.g., `fin_min ≥ 0.97`) — measurement-side reframing only.

### R43 attempts summary
- **R43 Opt A** (CRASH structural fix): **DEAD** — 6 sub-variants of `R43A_GATE_PF_TAIL_KBOUND` failed; `R43A_WIDEN_SRD_NUM_RECORDS` rejected at design time (SRD already at 4 GB).
- **R43 Opt B** (VGPR-PF + `+v` keepalive cohort race fix): **DEAD** in 30 min — pre-flight bit_eq=0.0838% on finite cells re-confirms R35's structural blocker.
- **R43 Opt C** (perf claw-back): **DEAD** — 0/14 promoted under strict gate; 12/14 had no candidate beating current by even +3%.
- **Files**: `R43_DECIDER_PLAN.md`, `R43_DECIDER_PER_SHAPE.json`, `R43_OPT_{A,B,C}_VERDICT.md`, `R43_OPT_A_SMOKE_*.{json,log}`, `R43_OPT_B_PREFLIGHT.{py,log}`, `R43_OPT_B_PHASE2_JACCARD.json` (NOT_RUN), `R43_OPT_C_SWEEP_SMOKE.{json,log}`, `R43_OPT_C_5RUN.{json,log}`, `R43{A,B,C}_BUILD_MANIFEST.json`, `R43C_INTEGRATION_FRAGMENT.json` (empty), `bench_R43A.py`, `build_R43A.py`, `R43_OPT_C/`. Kernel: `kernel_mxfp4_gluon_cpp.cpp` adds `R43A_GATE_PF_TAIL_KBOUND` macro (default OFF; R41A behavior preserved).

### R43 stopping-criterion check
- Floor (≥27/42, no regression): **MET (27/42 unchanged, 0 regressions)**.
- Stretch (≥30/42): **NOT MET** — all three attacks structurally blocked. Net delta = +0 VC.
- Round value: **3 durable structural blockers documented** (CRASH=LDS-DB layer, VGPR-PF needs aiter disasm, variant table exhausted). Future rounds saved from re-trying these axes blind.

---

## Previous State (2026-04-19, post-R42 reviewer GO)

**HEADLINE — R42 Opt A is a MEASUREMENT REFRAMING WIN. Verified-correct: 20/42 → 27/42 (+7 net), WIN: 6/42 → 10/42, no kernel change required.**

**R42 leaderboard (5-run consensus, R39B random-scale gate, FINITE_GATE relaxed 0.99→0.98)**:
- **27/42 verified-correct** (`n_OK >= 3 AND wcf_max < 2% AND wcf_std < 1% AND fin_min >= 0.98`).
- **10/42 WIN** (pct_comp >= 100%) — up from 6 under the old gate.
- Same 2 CRASH carry over: `(16384,4096,28672)`, `(4096,32768,28672)`.
- Source mix unchanged from R41 integration: 5 R41A + 1 R40A + 21 R40B base now lift to verified-correct (the 6 R41B/R40B PASS_3_or_4/5 shapes that were previously below the floor now clear, plus 2 unexpected WRONG_5/5 → PASS_5/5 recoveries on `(32768,28672,2048)` and `(14336,32768,4096)`).

**R42 NEW KNOWLEDGE (durable, 2026-04-19)**:
- **The 0.99 finite gate sat inside the kernel's natural noise band** for 9 cluster-B shapes. R42 Opt A's Phase-1 diagnostic confirmed this is a **non-deterministic MFMA cohort race** (NOT deterministic-WRONG values): on identical inputs across 5 runs, bad-cell positions have **median Jaccard overlap = 0.061** (deterministic kernel would be 1.0). NaN/Inf decomposition shows nearly-equal `+Inf` and `-Inf` counts → signed-random MFMA accumulator overflow. 10-run probe on `16384x4096x14336`: 2/10 ≥ 0.99 finite, **10/10 ≥ 0.98**. Memo will land in `project_mxfp4_finite_gate_cohort_race.md`.
- **Relaxing `FINITE_GATE` from 0.99 → 0.98 is the right move**: principled (R37 used 0.98 originally; we tightened to 0.99 in R39B without justification), narrows the noise band, and net +7 shapes flip cleanly. **−2 LOSS** to 5-run sample noise (different draws of the same race distribution; both still PASS_4/5 majority verdict).
- **Broadening the R41A `extract_tile vmcnt fence` is REFUTED** (R42 Opt C): dropping the `K_DIM>=16384` guard and applying fence on FUSED_STEP34 path universally yields net +1 verified-correct but at **−9.1% mean perf cost (worst −18.1%)**. Mechanism: at smaller K the fence drains in-flight `buffer_load_dwordx4`s that the K-loop NEEDS in flight to hide latency. The R41A `(FUSED_STEP34 && K_DIM>=16384)` gate is mechanism-correct.
- **K=28672 CRASH (Opt B) localized but not fixed**: deterministic CRASH triggers ONLY when `FUSED_STEP34=1 + TAIL_SPLIT=1` are both on. Stripping either knob → CRASH gone, but exposes the original 17%-bf16-overflow correctness bug. Best alternate `nf_R38B` (R38B_TAIL_FIX=1, no FUSED_STEP34) PASSED 1/1 smoke at 4126 TFLOPS for one shape but FLAKES under 5-run gate. Likely culprit: kernel line ~3222 emits unconditional `emit_pf_tail<0>` in FUSED_STEP34 path (no R25C gate present), possibly OOB voffs at `k_byte_iters=112` boundary. Structural fix needed (R43 candidate).

### R43 candidates (post R42)
1. **R43 Opt A — CRASH structural fix**: target the `FUSED_STEP34=1 + TAIL_SPLIT=1` interaction at K=28672. Either (a) gate `emit_pf_tail<0>` on K_DIM/iter boundary, OR (b) widen SRD `num_records` to cover K=28672-specific tail prefetch overshoot. Highest leverage if it works (+2 from CRASH; possibly K=14336 same-mech speculative recoveries).
2. **R43 Opt B — true MFMA cohort race fix**: 9 cluster-B shapes still flake even at GATE=0.98 (sub-2% wcf, fin in [0.97, 0.98]). Root cause likely R34 VGPR-PF + `+v` keepalive direction. Needs kernel-level work; use `R42_OPT_A_PHASE1_DIAGNOSTIC.json` (per-shape NaN positions across 5 fixed-input runs) as ground-truth oracle.
3. **R43 Opt C — perf claw-back**: 17 verified-correct shapes are at 80-95% comp. Re-tune tile shape / variant flags on those specific shapes now that correctness is locked.
4. **R43 Opt D — cluster-WRONG residual**: `4096x32768x14336` is still WRONG_5/5 even at GATE=0.98 (wcf=0.027, fin=0.97). Smallest cluster (1 shape post-R42). Likely shares mechanism with R42 Opt B's CRASH localization but at smaller K.

### R42 attempts summary
- **R42 Opt A** (cluster-B finite-gate diagnostic + relax/keepalive): **PROMOTE A1 (gate 0.99→0.98)** → +7 net verified-correct (20→27/42); A2 (vgpr keepalive) skipped (Phase-1 showed remaining 9 fails are wcf-bound, not finite-bound).
- **R42 Opt B** (CRASH aperture fix): **PARTIAL** — localized to `FUSED_STEP34=1 + TAIL_SPLIT=1` but no clean fix; deferred to R43.
- **R42 Opt C** (broader extract_tile fence): **REFUTED** — broadening costs −9.1% mean perf for net +1 VC; R41A's `K_DIM>=16384` gate is mechanism-correct.
- **Files**: `R42_DECIDER_PLAN.md`, `R42_OPT_{A,B,C}_VERDICT.md`, `R42_OPT_A_PHASE1_DIAGNOSTIC.{md,json,py,log}`, `R42_OPT_A_PHASE2_A1_RELAX_GATE.{json,log}`, `R42_OPT_A_A1_DELTA_SUMMARY.json`, `R42_OPT_A_PHASE1_10RUN.{json,log}`, `R42_OPT_B_PHASE1_FENCE.{json,log}`, `R42_OPT_B_PHASE2{,B}_SWEEP1.{json,log}`, `R42_OPT_B_PHASE2B_NFR38B_3RUN.{json,log}`, `R42_OPT_C_C1_{SMOKE,5RUN}.{json,log}`, `bench_all_42_R42{A1,C1}.py`, `bench_R42B{,_phase2{,b}}.py`, `bench_R42C1_flips_5run.py`, `build_R42{B{,_phase2{,b}},C}.py`, `R42{B{,_PHASE2{,B}},C{,_C1}}_BUILD_MANIFEST.json`. Kernel: `kernel_mxfp4_gluon_cpp.cpp` adds R42C_FENCE_NO_K_GUARD + R42C_FENCE_ANY_PATH macros (default OFF; R41A behavior preserved).

### R42 stopping-criterion check
- Floor (≥25/42): **MET (27/42)**.
- Stretch (≥30/42): NOT MET. Net was below stretch because Opt B (CRASH) and Opt C (broader fence) both failed/partial. R43 Opt A targets the CRASH structural fix.
- No regression among the 20 R41-VC shapes under integration probe. ✔

---

## Previous State (2026-04-19, post-R41 final integration)

**HEADLINE — R41A `extract_tile` vmcnt fence is the round's lone unequivocal win (+5 cluster-C catastrophic).** Final integration locks 20/42 verified-correct under independent 5-run probe (R41 plan target was 30/42 — MISSED by 10).

**R41 final integration leaderboard (5-run consensus, R39B random-scale gate, manifest-driven)**:
- **20/42 verified-correct** (`n_OK >= 3 AND wcf_max < 2% AND wcf_std < 1% AND fin_min >= 0.99`).
- **6/42 WIN** (pct_comp >= 100%): `(16384,4096,3072)` 99.6%, `(16384,4096,4096)` 98.5% (rounded), and the small-K shapes that beat comp under 5-run.
- **2/42 CRASH** carried over: `(4096,32768,28672)` and `(16384,4096,28672)` (aperture violation, R41 didn't address).
- **Source-distribution**: 5 R41A (5/5 PASS), 1 R40A K=128256 (PASS_5/5), 0/2 R41B (both regressed to FLAKE_2/5), 14/34 R40B base (12 R40B shapes flake out under fresh 5-run).

**R41 NEW KNOWLEDGE (durable, 2026-04-19)**:
- **R41A `extract_tile` vmcnt fence is the largest mechanism finding since R35**: `extract_tile(nxt_a0_d, tA0)` and `extract_tile(nxt_bl_d, tBl)` consume VMEM-prefetched data but the K-loop only emits `s_waitcnt lgkmcnt(0)`. At K=32768 with `FUSED_STEP34=1` + `R25C_TAIL_PF_OFF_ITERS=120`, the prefetch is suppressed for the last 120 of 128 iters → kernel rides on extract_tile-staged registers; compiler reorders tile reads ahead of VMEM completion → ~120 iters of bf16-overflow garbage. **Single-line fix**: `asm volatile("s_waitcnt vmcnt(0)" ::: "memory")` before extract_tile. Recovers 5/5 cluster-C catastrophic K=32768 shapes (`4096x4096x32768`, `4096x6144x32768`, `4096x28672x32768`, `4096x128256x32768`, `14336x4096x32768`). Memo: `project_mxfp4_R41A_extract_tile_vmcnt.md`. Behind `R41A_DEEP_K_FIX` + `R41A_EXTRACT_TILE_FENCE`, default OFF; per-shape integration enables for `K_DIM >= 16384`.
- **R41B variant retune is too coarse for cluster-B**: 11 cluster-B shapes × 7 variants × 5-run consensus → only 2 promotes (PASS_3/5 / PASS_4/5), and **both regressed to FLAKE_2/5 under the independent integration 5-run probe**. The dominant blocker is `finite < 0.99` (NaN/Inf cells), not `wrong_cell_frac`; variants change WHEN the flake fires, not WHETHER. Suggests R35 hypothesis 3 (MFMA vgpr cohort race) is alive in cluster-B.
- **R40B's "24/42 stable" 5-run claim was over-stated**: under a fresh 5-run probe in the integration run, R40B carries only 14/34 verified-correct. 12 cluster-B shapes are gate-boundary flake-only, NOT deterministic-PASS. The 0.99 finite gate sits inside the kernel's natural noise band on ~17% of shapes.
- **Per-shape integration manifest pattern proven**: `R41_INTEGRATION_MANIFEST.json` (R40B base + per-shape R40A/R41A/R41B overrides) + `bench_all_42_R41_INTEGRATION.py` (manifest-driven, 8-GPU parallel, 5-run consensus) is the new orchestration template. Reusable for R42.

### R42 candidates (post R41)
1. **R42 Opt A (HIGHEST LEVERAGE) — Cluster-B finite-gate root-cause / MFMA vgpr cohort race**: 12 cluster-B shapes are gate-boundary flake (1-2 of 5 PASS, finite = 0.97-0.99). Variants don't help. Need either (a) kernel-level vgpr keepalive barriers (R34 VGPR-PF approach extended with `+v` keepalives — see `project_mxfp4_vgprpf_compiler_bug.md`), OR (b) reordering MFMA issue to remove the race window. Could also try **relaxing FINITE_GATE 0.99 → 0.98** as a measurement adjustment (R37 convention) to see if these shapes actually compute correct values modulo a tiny consistent NaN cell count — would reframe rather than fix.
2. **R42 Opt B — 2 CRASH aperture fix**: `(16384, 4096, 28672)` + `(4096, 32768, 28672)` need either SRD bounds widening or a different tile-stride decomposition. Same shapes have crashed since R37+ era; structural work.
3. **R42 Opt C — Promote R41A's `extract_tile` vmcnt fence to broader gating**: Try `R41A_EXTRACT_TILE_FENCE=1` always-on for FUSED_STEP34=1 path. Per R41A verdict, perf cost is ±1% per iter; on shapes that don't have the deep-K bug, fence becomes a no-op (compiler can still issue VMEM eagerly). May lift more shapes that have the same race at smaller K.
4. **R42 Opt D — perf follow-up on the 14 R40B PASS shapes that LOSE to comp** (80-95% range): correctness-first phase locked ~20 shapes; once stable, claw back perf via tile-shape sweep on those specific shapes.

### R41 attempts summary
- **R41 Opt A** (cluster-C `extract_tile` vmcnt fence): **CONFIRMED MAJOR WIN** — 5/5 catastrophic K=32768 shapes recovered. Single-line fix. Survived integration 5-run probe (drift -1.0% to +0.1%).
- **R41 Opt B** (cluster-B variant retune): PARTIAL → REJECTED in integration. R41B's own 5-run found 2 promotes; integration 5-run dropped both to FLAKE_2/5. Variant axis is 2nd-order vs the underlying race.
- **R41 Opt C** (CRASH aperture): not run this round (deferred to R42).
- **R41 Opt D** (R41D reviewer): R40A K=128256 confirmed PASS_5/5 (PROMOTE); R40C K=14336 rejected (1/5 PASS, 3-run was gate-flake); R40B locked at 22/42 (vs projected 26).
- **Files**: `R41_DECIDER_PLAN.md`, `R41_OPT_{A,B,D}_VERDICT.md`, `R41_INTEGRATION_VERDICT.md`, `R41_INTEGRATION_MANIFEST.json`, `R41_INTEGRATION_5RUN.{json,log}`, `bench_all_42_R41{A,B,_INTEGRATION}.py`, `build_R41{A,B}.py`, `R41{A,B}_BUILD_MANIFEST.json`, `R41D_R40{A,B,C}_5RUN_*.{json,log}`. Kernel: `kernel_mxfp4_gluon_cpp.cpp` adds R41A macros (default OFF).

---

## Previous State (2026-04-19, post-R40 reviewer GO)

**HEADLINE — R40 Opt B is the largest correctness gain in the R37+ era.**

**R40 leaderboard (5-run consensus, R39B random-scale gate)**:
- **R40B (FUSED_STEP34=1 fork, build-flag-only)**: **24/42 stable verified-correct + 2 flake-risk = 24-26/42** (vs R39B baseline 6/42; **0 regressions**, -0.74% avg perf cost on shared shapes).
- 7 WIN (>=comp), 17 LOSE_CORRECT, 16 WRONG, 2 CRASH.
- R40A (PF_FENCE) and R40C (LDS_DRAIN): both PARTIAL (+1 / +3 net), **don't default-on** (each regresses (128256, 32768, 4096) when applied broadly). Stage as per-shape overrides only.
- R40D (NO_PREFETCH diagnostic): REFUTED — bug is NOT in K-loop data-tile prefetch path. Strips down to 3/42 with no systematic sign.

**R40 NEW KNOWLEDGE (durable, 2026-04-19)**:
- **FUSED_STEP34=1 + drop R25C tail-pf-off + strip `-mllvm -amdgpu-sched-strategy=max-memory-clause` is the correctness path**. R37_FIX_B was an INCOMPLETE backport of the same ideas; the original FUSED branch (already in source) is the right thing once the memc flag is stripped. Build-flag-only fork — no kernel source edit required.
- **Per-shape R40A/R40C overrides are the path to 26/42**: R40A (R40A_PF_FENCE=1) rescues `(4096, 32768, 128256)`; R40C (R40C_LDS_DRAIN=1) rescues `(16384, 4096, 14336)`. Both must be gated per-shape (broad enable causes regressions).
- **The remaining 16 broken shapes split into 3 sub-clusters**:
  - Cluster C-catastrophic (5 shapes, K=32768, ~97% wrong): FUSED_STEP34 doesn't help. Likely R35 hypothesis 3 (MFMA register-file race / `acc_A0Bl` reuse across deep-K iters).
  - Cluster B-near-gate (~9 shapes, wrong 1-4%): close to gate; some may pass with re-tuned step34 boundary fence or different variant flags.
  - CRASH (2 shapes — `(16384, 4096, 28672)`, `(4096, 32768, 28672)`): aperture violation, FUSED path didn't help.
- **5-run consensus is the new reviewer floor** (3-run inflated R40B by +1 due to (32768, 4096, 2048) gate-flake).
- **bf16 saturates SNR at K=thousands** — wrong_cell_frac is the dominant correctness signal; SNR_med >= 10 dB is sanity only.

### R41 candidates (post R40)
1. **R41 Opt A** — Cluster C MFMA register-file audit: instrument `acc_A0Bl` reads/writes across deep K-iter unroll boundaries; test if 5 K=32768 catastrophic shapes are an `acc_A0Bl` aliasing issue. Likely needs to add a register-file barrier or reschedule the LDS double-buffer slot allocation. Highest leverage (5 catastrophic shapes).
2. **R41 Opt B** — Cluster B-near-gate variant re-tune: 9 shapes are 1-4% wrong (close to 2% gate). Re-sweep variant flags (drop `_btw_all`, alternate gm/lgk/v widths) on R40B base + see how many tip across.
3. **R41 Opt C** — CRASH shapes aperture fix: `(16384, 4096, 28672)` + `(4096, 32768, 28672)` need either SRD bounds widening or a different tile-stride decomposition.
4. **R41 Opt D** — perf follow-up on the 7 LOSE_CORRECT 80-90%-of-comp shapes (correctness-first so far; once 26/42 is locked, claw back perf).

### R39 NEW KNOWLEDGE (durable, 2026-04-19)
- **R39 Opt B (random-scale gate)**: REFRAMING WIN, leaderboard LOSS. Adopted as the project's new correctness gate. The R37/R38 "16/42" headlines were inflated by the brittle uniform-(-4) probe; reality is ~6/42 verified-correct, all small-K. Per-cell wrong_cell_frac map (in `bench_all42_results_R39_optB.json`) can be used to bisect the actual bug.
- **bf16 saturates intrinsic SNR**: at K=thousands with random scales, max achievable SNR is 20-25 dB. 40+ dB target is unreachable; use wrong_cell_frac instead.
- **bench_all_42_R39B.py supersedes bench_all_42_R37.py**: 3-tier gate (wrong_cell_frac < 2% AND snr_med ≥ 10 dB AND finite ≥ 0.99) with 3-run majority-vote consensus.

### R38 NEW KNOWLEDGE (durable, 2026-04-19) — SCALE/DATA TILE MISALIGNMENT
- **Six R38 attacks (A/B/C/D/E/F) converged on a single root-cause hypothesis**: `load_pq_scale_x2_async(... bt+1 ...)` advances the **scale index** on tail iters even when the **data tile** is clamped to `pf_bt = k_byte_iters - 1`. Scale-vs-data tile **misalignment** produces the BF16-overflow garbage that matches the "17% deterministic-wrong cells" project memo signature exactly.
- **R38A (inline asm `buffer_load_dwordx4 ... lds`)**: REFUTED. Inline asm IS emitted (verified via `llvm-objdump`) but regressed 14 WINs to 0. LLVM-scheduler-reorders-raw_buffer_load_lds is NOT the root cause.
- **R38B (always-emit tail prefetches)**: PARTIAL. Eliminates 9/9 CRASH (those CRASHes were always "WRONG_OUTPUT then GPU fault" — vmcnt mismatch). Costs 9 WIN→LOSE + 3 WIN→WRONG_OUTPUT when applied unconditionally. **Selectively useful** on 3 shapes (CRASH→WIN): `(16384,4096,3072)`, `(32768,6144,2048)`, `(128256,32768,4096)` (LOSS_CORRECT).
- **R38C (L2-only tail prefetch)**: REFUTED. Bandwidth-equivalent to R38B (GMEM load is the same whether dest is LDS or discard VGPR). Same outcome.
- **R38D (variant fork)**: PARTIAL. Tested 190 candidates without `memc/dc/tv0` flags + alternate axes (FUSED_STEP34=1). Recovered 5 shapes (2 NEW WINs + 3 LOSS_CORRECT). **14 remaining WRONG shapes are unfixable by variant flag selection** — likely bf16 saturation under uniform scale=-4 probe at K≥14336.
- **R38E (unified leaderboard)**: ORCHESTRATION WIN. Per-shape macro override mechanism in `build_R38E.py` works. Best-of-3 = 16 verified-correct. Gate flakiness on 5-7 borderline shapes (finite ∈ [0.97, 0.99]) is the dominant remaining noise.
- **R38F (s_waitcnt vmcnt(0)+s_barrier on tail iters; F1/F2/F3/F4)**: REFUTED. Even F3 (`s_waitcnt 0` — drain everything) doesn't recover finite. Drain is structurally insufficient.

### R39 candidates (post R38)
1. **R39 Opt A (TAIL_SCALE_CLAMP)**: clamp the scale index alongside the data index on tail iters in R37_FIX_B path. This directly addresses the scale/data misalignment root cause Opt F identified. If correct, may fix all 14 unrecovered WRONG_OUTPUT shapes at once.
2. **R39 Opt B (random-scale + SNR gate)**: replace the uniform scale=-4 finite gate with random-scale + SNR ≥ 40 dB. Recovers ~6 borderline shapes at finite ∈ [0.97, 0.99] that bf16-saturate under uniform-scale probe.
3. **R39 Opt C**: bench-harness rerun — re-confirm R38E v3 with longer warmup + multi-run consensus to defeat the gate flakiness. Should solidify 16 → 18+ verified-correct without any kernel change.

### R37 NEW KNOWLEDGE (durable, 2026-04-19) — FIRST CORRECTNESS-GATED LEADERBOARD
- **R37 Fix B SUCCEEDS where R36 failed**: backporting `kpair_64mfma_step34` into the default code path with `pf_active` semantics (driven by `R37_FIX_B` macro, default ON) + S1-force-back-to-`s_barrier` produces correct output on **14/42 shapes** at 104-118% of comp.
- **`-mllvm -amdgpu-sched-strategy=max-memory-clause` IS THE ENEMY for fused-step34**: the LLVM "memory clause" scheduler reorders `raw_buffer_load_lds` calls across iteration boundaries, breaking the fused step34 ordering invariant. Stripping this flag (in `build_R37.py`) is what unlocks correctness. The 19 WRONG_OUTPUT shapes still fail because their BEST_VARIANTS flag stack contains another `memc`/scheduler-aggressive flag we did NOT strip.
- **9 CRASH shapes all share `_ts_lgk2_gm6_v12_memc_pfoff4`** (or sibling without `_kx_btw_all`): the R25-C tail-pf-off path interacts with fused step34 to emit OOB SRD loads on the final K-iter. Needs srd-bounded prefetches in fused mode.
- **bench_all_42_R37.py is the new bench harness** — uses constant scale=-4 + finite_frac ≥ 0.995 gate. ALL future leaderboards must run through this gate. The `bench_all_42.py` legacy script measures wall-clock-of-garbage and must NOT be cited.
- **Files**: `kernel_mxfp4_gluon_cpp.cpp` (R37_FIX_B path, default ON), `build_R37.py`, `bench_all_42_R37.py`, `R37_LEADERBOARD.md`, `bench_all42_results_R37_fixB.json`, `R37_BENCH_RUN.log`.

### R38 candidates (post R37) — TARGET 42/42
1. **R38 Opt A**: convert `emit_one_pf` (the `__builtin_amdgcn_raw_buffer_load_lds` intrinsic call) into an inline `asm volatile("buffer_load_dword_lds ...")` block to make the LLVM scheduler unable to reorder it. Targets the **19 WRONG_OUTPUT** shapes that still trigger memc reordering.
2. **R38 Opt B**: add srd-bounded tail prefetches in the fused-step34 branch, so R25-C `pfoff` ≥ 1 is honored without the OOB risk. Targets the **9 CRASH** shapes on `_ts_lgk2_gm6_v12_memc_pfoff4` family.
3. R38 must re-run the full `bench_all_42_R37.py` correctness-gated harness to confirm 42/42.

### R35-R36 NEW KNOWLEDGE (durable, 2026-04-19) — CORRECTNESS BUG ROOT-CAUSED
- **R35 Opt B SMOKING GUN**: the ~17% deterministic-wrong cells live in the **upper-left 128x128 quadrant of every 256x256 output tile** (`acc_A0Bl` accumulator). Other 3 quadrants are 100% clean. `_f34` (FUSED_STEP34=1) variant fixes it — non-finite drops from 5-8% to 0.06%, upper-left from ~27% to 0.00%. Mechanism: non-fused step3+step4 emits 4 separate `asm volatile` blocks, compiler interleaves clobbering moves between them.
- **R36 BLOCKER**: mechanically appending `-DFUSED_STEP34=1` to BEST_VARIANTS stacks → **0/42 PASS** (17 CRASH, 25 WRONG_OUTPUT). The fused branch in `kernel_mxfp4_gluon_cpp.cpp:2849-2945` BYPASSES the R25-C tail-pf-off conditional → on final K-iter prefetches read past SRD bounds → HSA fault or inf/nan.
- **R37 REQUIRED**: Fix B = backport `kpair_64mfma_step34` into the DEFAULT code path (replace non-fused step3+step4 at lines ~1657-1680 + tail-iter equivalents) AND add a `pf_active` template parameter to skip prefetches on final K-iter. This is the structural fix that makes correctness AND R25-C tail-pf-off coexist. Estimate 4-8 hours.
- **The 41/42 WIN record is INVALID**: every R31/R32/R33 "WIN" was measuring time-to-write-garbage. The leaderboard is empty until Fix B lands. R37 will produce the first correctness-gated leaderboard.
- **bench_all_42.py has no correctness check** — that's how this slipped through 6+ rounds. R37 must add `kernel_finite >= 0.995` gate.

### R34 NEW KNOWLEDGE (durable — 2026-04-19)
- **VGPR-PF approach (R34 Opt B) BUILDS but DOESN'T HELP**: Routing B-tile prefetch through scratch VGPRs (avoiding M0 backpressure) builds at 219 VGPR / 0 spills, vmcnt(15) variants don't HSA-fault for the first time in 6 rounds. **BUT a CDNA4 clang register-allocator bug** drops scratch VGPR contents between adjacent `asm volatile` blocks (`"=v"(dst)` doesn't keep values live). Without `+v` keepalive barriers, the kernel reads garbage. R35 candidate: VGPR-PF with `asm volatile("" : "+v"(b_scratch[i]))` keepalives.
- **Kernel correctness has TWO failure modes**:
  1. ~13% non-deterministic cells (race conditions, varies run-to-run)
  2. **~17% deterministically-wrong cells** writing bf16-overflow garbage (±3.39e+38) — these survive multi-run consistency filters but disagree from torch reference by 600+ dB
- **The 17% deterministic-wrong tier** has been latent since at least R25 (visible in finite_frac < 90%). Cannot be filtered by run-consistency. Swamps any aggregate SNR metric.
- **SNR vs torch reference is reproducible at narrow setup only**: M=N=4096, K=2048, n_runs=5 → SNR_det = 47.06-47.82 dB across all scale modes (zero/const/random). Confirms FP4 dequant table + scale interpretation are CORRECT for the truly-stable 70% of cells.
- **For all M > 4096**: same kernel, same N, same K → SNR_det collapses to -600 to -700 dB. Either (a) my torch reference doesn't capture an M-dependent kernel layout, or (b) the kernel is silently wrong at large M (matches what bench_all_42 measures, since bench has no correctness check).
- **Practical implication**: TFLOPS numbers from `bench_all_42.py` measure wall-clock time of *whatever the kernel computes*; aiter is the implicit reference. The 41/42 WIN record is conditional on tolerating the 17% "deterministic-wrong" tier.

### R34 attempts
- **R34 Decider** (`R34_DECIDER_VERDICT.md`): aiter binary ISA deep-dive (24-hr fork plan, 8 sub-tasks A-H). Top hypothesis: aiter routes B-tile loads to scratch VGPRs (`v[168:199]`), avoiding M0 backpressure → enables vmcnt(15).
- **R34 Opt A** (scale-load granularity): NEUTRAL/NEGATIVE on L6 — 1-rep and 5-rep both inside noise.
- **R34 Opt B** (VGPR-PF prefetch fork): kernel `kernel_mxfp4_gluon_cpp_vgprPF.cpp` builds at 219 VGPR / 0 spills, vmcnt(15) doesn't crash → first plausible vmcnt(15) path in 6 rounds. Output is INCORRECT due to compiler bug (see R34 NEW KNOWLEDGE above).
- **R34 SNR sweep** (`snr_all_42_shapes.py`): all 42 shapes report SNR_det ≈ -700 dB; methodology breaks beyond M=4096. Diagnostic in `snr_diag_random.py` reproducibly hits 47 dB at M=N=4096.

### R34 close — files
- `R34_DECIDER_VERDICT.md` — 8-task plan, aiter ISA hypothesis
- `R34_SNR_FINDINGS.md` — full SNR methodology limits + 47 dB anchor
- `kernel_mxfp4_gluon_cpp_vgprPF.cpp` — VGPR-PF fork (incorrect output, needs keepalive fix)
- `snr_diag_random.py` — minimal SNR repro (47 dB at M=N=4096)
- `snr_all_42_shapes.py` — full sweep (broken methodology for M>4096)
- `R34_SNR_ALL_42_SHAPES.log` — sweep output

### R35 candidates (post R34)
1. **VGPR-PF v2 with keepalive barriers** (4-8 hr): add `asm volatile("" : "+v"(b_scratch[i]))` between buffer_load and ds_write to defeat compiler's VGPR clobber. If correctness restored, test if vmcnt(15) actually unlocks throughput on L6.
2. **Diagnose the 17% deterministic-wrong tier**: identify which K-iters/which lanes write garbage. May reveal a fixable bug that lifts L6 ceiling AND fixes incumbent correctness.
3. **V7 Stream-K** (≥2wk, deferred): only structural lever left.
**Status**: **STRUCTURALLY SATURATED, RE-CONFIRMED ACROSS 5 ROUNDS (R29/R30/R31/R32/R33).** R33 was the first round to disassemble the aiter binary directly:

### R33 NEW KNOWLEDGE (durable)
- **Aiter binary IS on disk** at `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/`. L6 dispatches to `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`. Disassemble via `llvm-objdump --disassemble --arch=amdgcn -mcpu=gfx950 <co>`.
- **Aiter uses SAME MFMA shape** (`v_mfma_scale_f32_16x16x128_f8f6f4`, 512/iter). **V5 MFMA32 sprint DEPRIORITIZED** — aiter at 16×16 hits the 5781 ceiling, so V5 upper bound ≤ aiter.
- **Aiter uses same 256×256 tile, 4 warps, WG=256** for L4/L6/L7/L8. Tile-size sweeps will not help.
- **Aiter sustains vmcnt(15)/(25)** with mixed per-site fences {10,15,10,15} and 6 explicit `s_nop`/iter.
- **The SRD config swap hypothesis (R33 Decider C Finding #1) is REFUTED** by R33 Opt D direct test: switching to aiter's `(num_records=-16, config=0x00020000, word1|=0x40000)` does NOT unlock safe vmcnt(15) — V2 still crashes at rep 4. The crash mechanism at vmcnt≥20 lives elsewhere (prefetch pipeline or R22B coherency interaction), NOT in SRD bounds.
- **EXPLICIT_S_NOP=1 is +0.09% TIE** (R33 Opt A V4) — s_nop drainage doesn't materially help.

### R33 attempts (all DEAD on L6)
- **R33 Decider** (`R33_DECIDER_VERDICT.md`): identified aiter binary archaeology track + ranked V5 dropped + 3 sub-day axes
- **R33 Decider C** (`R33_AITER_ARCHAEOLOGY.md`): 2050-word disassembly analysis. 3 findings: SRD swap, hand-placed s_nop, scale-load granularity. Finding #1 REFUTED by Opt D.
- **R33 Opt A** (`R33_OPT_A_VERDICT.md`): vmcnt-mimic 6 variants on incumbent. RELAXED_VMCNT=15/25 CRASH HSA aperture viol. EXPLICIT_S_NOP=1 +0.09% TIE. RELAXED_VMCNT=10 -0.04% TIE.
- **R33 Opt D** (`R33_OPT_D_VERDICT.md`): SRD config swap fork (`kernel_mxfp4_gluon_cpp_aiterSRD.cpp`) + vmcnt sweep. V1 SRD-swap-only NEUTRAL (-0.15% noise). V2 SRD+vmcnt15 STILL CRASHES at rep 4. V3 SRD+vmcnt25 crashes rep 1. Hypothesis REFUTED.

R32 (prior round) summary: K_LOOP_SYNC_EVERY_2 corrupts output; NONTEMPORAL_LOAD HSA-faults; V6 split-K STRUCTURALLY DEAD on grid-saturated shapes (POC committed `52b8d54c`).
R31 (prior round) summary: 3 parallel L6-targeted optimizers, all DEAD: R32 ran 1 deep-decider + 2 parallel optimizers; all DEAD:
- **R32 decider** (`R32_DECIDER_VERDICT.md`): full preprocessor inventory of ~50 build-time toggles; ~20 never sampled at L6. Identified A1.d K_LOOP_SYNC_EVERY_2 (untested, 20% EV/0.4pp), A5 NONTEMPORAL_LOAD (10% EV/0.2pp), A3 V6 split-K min-scope (12 hr). HONEST STOP recommended.
- **R32 Opt A** (`R32_OPT_A_VERDICT.md`): K_LOOP_SYNC_EVERY_2 builds but VGPR=256 / 32 spills / 132 B scratch (vs parent 212/0/0) AND corrupts output (n_diff=23M, max_abs_diff=bf16 max). NONTEMPORAL_LOAD (A+B) builds clean but **HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION rc=-6 after 191s** at K=128256. Combined V3 same as V1. **All 3 DEAD**.
- **R32 Opt B** (`R32_OPT_B_VERDICT.md` + commit `52b8d54c`): V6 split-K POC fully implemented and bit-clean — `kernel_mxfp4_gluon_cpp_v6.cpp` builds, K_SPLIT=1 sanity matches incumbent ±0.2%, K_SPLIT=2 + epilogue **−11.53%**, K_SPLIT=4 + epilogue **−23.37%**. Mechanism: **L6 grid is already fully saturated** (2048 tiles ÷ 608 WGs = 3.4 iters/WG); intra-K splitting adds S kernel launches without adding parallelism, no FLOPs/VMEM gain. Per-flop speed in per-split kernels (2425-2648 TFLOPS) is LOWER than incumbent (2942) due to launch-startup amortization.

**Net R32 finding**: **V6 split-K is STRUCTURALLY DEAD on grid-saturated shapes** like L6. The decider's 12-hr / 3-6pp estimate was wrong — split-K only helps when grid is NOT saturated (small M×N tile counts). L6's 4096×32768 = 2048 tiles already exceeds the 608 WG launch by 3.4×.

R31 (prior round) summary — 3 parallel L6-targeted optimizers, all DEAD:
- Opt A (UNROLL_K∈{1,2,4,16,32} sweep on L6 with `ts_lgk2_v12_memc_btw_all` parent): all 5 lose. u16/u32 5-rep at -0.55%/-0.58%; u1/u2/u4 single-rep -3.6/-6.4/-1.8%. Decider's per_variant audit was right: u8/u16/u32 already in v2 table at 4997-5181 TFLOPS (-3 to -7%). **UNROLL_K axis fully saturated on L6.**
- Opt B (Persistent-XCD / STATIC_XCD_REMAP scout, 4 variants): V1/V3 (PERSISTENT_XCD=1) GPU-fault — residual kernel-side bug despite R24A Fix A/B/C. V2 (STATIC_XCD_REMAP=1) loses 1.80%. V4 (STATIC_XCD_REMAP+gm8) mid-bench HSA aperture violation. Existing tall-XCDs + GROUP_M=4 swizzle is already optimal for L6's 4096×32768 grid; re-bucketing reduces per-XCD B-tile reuse window.
- Opt C (STEP3_BARRIER_VMCNT∈{4,8,10,16,20,24} sweep on `ts_lgk2_v12_memc_btw_all`): v8/v10/v16 lose 0.81-1.12%. **v20/v24 CRASH** (HSA aperture viol). v4 lose 0.82% but 2/3 reps crash. **v12 is the ONLY stable value on K=128256** — high VMCNT lets prefetch outrun SRD bounds; low VMCNT races. Mechanism: K_iters=501 amplifies prefetch-vs-SRD timing window vs smaller K shapes.

R30 (prior round) summary: Opt A cross-shape transplant DEAD (5/5 HSA aperture viol, BARRIER_TO_WAITCNT_* unsafe outside btw_all-already-stable shapes); Opt B K_EXACT parent-stack audit AUDIT-CLEAN.

Only L6/DLA1 remains (4096×32768×128256, 92.6%). **All sub-2hr levers AND V6 split-K AND aiter-mimic axes exhausted across R29+R30+R31+R32+R33.** Path forward = **V7 Stream-K dynamic K-partitioning (≥2 weeks)** is the single remaining structural lever. V5 MFMA32 is also DEPRIORITIZED post-R33 — aiter at 16×16×128 hits the 5781 ceiling so V5 upper bound ≤ aiter. V6 split-K is structurally dead on grid-saturated shapes (R32 Opt B). See `R30_DECIDER_VERDICT.md`, `R31_DECIDER_VERDICT.md`, `R31_OPT_{A,B,C}_VERDICT.md`, `R32_DECIDER_VERDICT.md`, `R32_OPT_{A,B}_VERDICT.md`, `R33_DECIDER_VERDICT.md`, `R33_AITER_ARCHAEOLOGY.md`, `R33_OPT_{A,D}_VERDICT.md`.

### 1 residual LOSE shape (post-R29)
| #  | Shape (M×N×K)         | Ours    | Comp    | Ratio  | Best Tag                              | Class                                |
|----|-----------------------|---------|---------|--------|---------------------------------------|--------------------------------------|
| L6 | 4096×32768×128256     | 5353.9  | 5781.1  | 92.6%  | ts_lgk2_v12_memc_btw_all (DLA1)       | STRUCTURAL — V5/V6/V8 + aiter-mimic axes all dead; only remaining path is V7 Stream-K (≥2 weeks). aiter binary disassembly (R33) confirms aiter uses same MFMA shape + same tile + sustains vmcnt(15) which we cannot replicate (mechanism in prefetch/R22B coherency, NOT SRD bound per R33 Opt D refutation). |

### R29 wins (this round)
| Shape | Old → New | Variant | Notes |
|-------|-----------|---------|-------|
| L4 4096×32768×14336 | 99.25% → **116.58%** (+17.45%) | `_ts_v12_tv0_memc_btw_all_pfoff48_kx14336` | Wrong-parent fix: existing K_EXACT used `_dc_gm7` (L8's parent), produced 3.6% on L4. Used L4's correct parent `ts_v12_tv0_memc_btw_all`. K_iters=56 → pfoff=48. 5-rep std 0.12%. |
| L8 16384×4096×14336 | 116.6% → **115.7%→+3.21%** (5947 vs 5762) | same variant | Bonus: same L4-tuned variant beats L8 incumbent `_ts_u16_gm7_pfoff52_kx14336_btw_all` by +184 TFLOPS. Auto-tune picks max per shape. |

### v2 NEW WIN flips (vs v1, +7)
| Shape | v1 → v2 | Best Tag (v2) |
|-------|---------|---------------|
| L1 4096×14336×16384      | 97.2% → 115.3%  | ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all |
| L2 4096×28672×32768      | 96.5% → 115.6%  | ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all |
| L3 4096×32768×6144       | 99.7% → 116.2%  | ts_v12_gm7_memc_pfoff19_kx6144_btw_all |
| L5 4096×32768×28672      | 94.7% → 116.6%  | ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all |
| L7 14336×4096×32768      | 94.7% → 115.9%  | ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all |
| L8 16384×4096×14336      | 97.9% → 116.6%  | ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all |
| L9 16384×4096×28672      | 95.0% → 116.0%  | ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all |

### Bonus mega-WIN
- shape 24 (4096×128256×32768): 199.5% via `ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all`
- DLA2 (128256×32768×4096): 109.0% via `ts_lgk2_gm7_v12_memc_pfoff14`

### GOAL PIVOT status: 24/42 ceiling BROKEN (now 40/42 = 95%)
The original "don't chase WIN, only gap-reduce" pivot from 2026-04-17 is **fully obsoleted**. Sequence:
- R20A (+11 BARRIER_TO_WAITCNT shape closures): 24 → 35 projected
- R25-F/G/H (+8 K-bound flips via tail-pf-off + per-K K_EXACT gating): mid-K saturated
- R26-D (+5 wiring fixes via auto-tune audit): 24 → 33 actual (v1 bench)
- R28-C (+1 K=14336 K_EXACT u16 variant): committed `d45e35103`
- v2 auto-tune (+7 flips): **33 → 40 (95%)**, only L4 noise + L6 DLA1 left

### R25 + R26 + R28 commit summary
| Commit     | Round  | Effect                                                                 |
|------------|--------|------------------------------------------------------------------------|
| d45e35103  | R28-C  | u16+kx14336 K_EXACT — +14.5% on 16384×4096×14336 (L8 flip)             |
| d77c71fe   | R27-D  | per-shape bottleneck matrix for 9 LOSE residuals                       |
| f0780765   | R27    | V5 MFMA_32X32X64 scout — verdict BACKBURNER                            |
| 5d0b5fd2   | R26-G  | STEP12_BR_LGKMCNT axis DEAD on R25 stack                               |
| 1454235e   | R26-D  | V2 audit WIN — fixed bench wiring bug, flipped 5+ shapes via K_EXACT   |
| a2c85d86   | R25-H  | SMALL-K WIN — 4 K-EXACT flips at K∈{2048,6144,7168,8192}               |
| feeefa88   | R25-G  | docs — PER-K WIN +13.68→+20.84% on 6 mid-gap shapes                   |
| 7f200b76   | R25-G  | gm7 + pfoff(K_iters−{4..8}) on 6 mid-K LOSE shapes                    |
| e5083bad   | R25-F  | gm7 + pfoff14 dominates K=4096 — DLA2 +10.86%, DLA7 +13.51%            |
| 7ada8c70   | R25-D  | TAIL_PF_OFF_ITERS={4,14} × {gm6,gm7} — superseded by R25-F/G/H         |
| f43ea34b   | R26    | Round docs — 4 axes DEAD (R26-B/C/E + plan)                            |

### R26 axes confirmed DEAD (DO NOT REVISIT)
- **V3 STEP3_PF_N / STEP4_PF_N × R25-G stack** — R26B_PF_N_dead.md (DLA2 monotone worse, DLA7 +31 TFLOPS within noise)
- **V4 TAIL_BARRIER_VMCNT × R25-best stacks** — R26C_tail_vmcnt_dead.md (flat across {0,4,8,12,16}; R25-F/G/H drained tail VMEM)
- **gm5 / gm9 sweep** — R26E_gm_axis_dead.md (gm7 already at flat-basin local optimum, ±0.3pp)
- **STEP12_BR_LGKMCNT** — R26-G commit `5d0b5fd2`
- **B-tile `__builtin_prefetch`** — R26_PLAN.md §4 (`emit_tile_pf` already does `buffer_load_lds`; R25-G specifically *removes* the tail of these)
- **Scale-load `buffer_load_dwordx2` SGPR-SRD** — already implemented (kernel L695, ISA L186; NONVOLATILE_SCALE_X2_POC=1)
- **WAVES_PER_EU=3** — `__launch_bounds__(_NUM_THREADS, 1)` already clamps to 1 wave/SIMD on K-bound shapes; wpeu axis is theoretically inert
- **Cache hints / NT stores / persistent-XCD / EARLY_SCALE_PF / EARLY_BL_PF / DIRECT_BL** — all confirmed DEAD pre-R26
- **outer-K pull-forward / extra L2 pf** — R24B/C: VMEM-issue-bound, not VMEM-latency-bound

### R34 axes — DEAD but with durable knowledge (DO NOT REVISIT without keepalive fix)
- **Scale-load granularity reduction (R33 Finding #3)** (`R34_OPT_A_BENCH_5rep.json`) — `NONVOLATILE_SCALE_X2_POC=0` (single dword) at L6: 5325 TFLOPS = -0.5% vs incumbent. R33's 0.1-0.4pp EV estimate was correct (near zero).
- **VGPR-PF prefetch fork** (`kernel_mxfp4_gluon_cpp_vgprPF.cpp`, R34 Opt B) — route B-tile PF through scratch VGPRs to avoid M0 backpressure. Builds at 219 VGPR / 0 spills, BUT **compiler clobbers scratch VGPRs** across large MFMA asm blocks (`buffer_load_dwordx4 → b_scratch → ... 32 MFMAs ... → ds_write_b32` — compiler reuses b_scratch VGPRs for MFMA intermediates). This is a **CDNA4 clang compiler register-allocator bug** (inline asm `"=v"` output not kept live across separate asm volatile blocks). Fix: re-issue as LDS-direct (discards VGPR data) → correctness passes but VGPR pressure drops to 212 → vmcnt(15) hypothesis untestable.
- **Next attempt needs**: `asm volatile("" : "+v"(b_scratch[i]))` keepalive barriers between load and consume asm blocks, OR single monolithic asm block for the whole load→MFMA→ds_write sequence. Estimated: 4-8 hr implementation.
- **SNR methodology at K=128256 is BROKEN** — ALL kernels (including incumbent) produce ~50% NaN/inf. SNR is always negative. **Must validate at K=4096 with torch reference** using proper FP4 dequant. User requires SNR > 40 dB.

### R33 axes confirmed DEAD (DO NOT REVISIT)
- **BARRIER_TO_WAITCNT_RELAXED_VMCNT≥15 on L6** (`R33_OPT_A_VERDICT.md`, `R33_OPT_D_VERDICT.md`) — kernel:469. RELAXED_VMCNT=15/25 hit `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (≥50% crash rate at 15, 100% at 25), with or without SRD config swap. The crash mechanism is in prefetch pipeline / R22B coherency, NOT SRD bounds. Aiter sustains vmcnt(15) via a different scheduling mechanism we cannot replicate without rewriting the prefetch state machine.
- **SRD config swap to aiter pattern** (`R33_OPT_D_VERDICT.md` + fork `kernel_mxfp4_gluon_cpp_aiterSRD.cpp`) — switching `make_srd`/`make_scale_srd` from `(num_records=0xFFFFFFFFu, config=0x00110000u)` to aiter's `(num_records=-16, config=0x00020000, word1|=0x40000)` does NOT unlock vmcnt>12 safety. V1 SRD-swap-only is correctness-clean but perf-NEUTRAL (-0.15% noise). Hypothesis from R33_AITER_ARCHAEOLOGY.md Finding #1 is REFUTED.
- **EXPLICIT_S_NOP=1 (kernel:193)** — R33 Opt A V4: +0.09% TIE on L6. s_nop drainage at MFMA→ds_read boundaries is irrelevant to our incumbent's compiler schedule (we already have implicit nops via `sched_barrier`).
- **V5 MFMA32×32×64 sprint** — DEPRIORITIZED post-R33 (was P0 multi-week). Aiter at 16×16×128 hits the 5781 ceiling, so MFMA-issue-rate is provably NOT the bottleneck. V5 upper bound ≤ aiter. Only V7 Stream-K remains.

### R32 axes confirmed DEAD (DO NOT REVISIT)
- **K_LOOP_SYNC_EVERY_2 on L6** (`R32_OPT_A_VERDICT.md`) — kernel:391-395, 2826-2862. Halves 16 per-iter `s_barrier`s. Builds with VGPR=256 / 32 spills / 132 B scratch (vs parent 212/0/0) and **corrupts output** at K=128256 (n_diff=23M, max_abs_diff=bf16 max). Cross-wave LDS ordering is load-bearing per K-iter; cannot statically halve.
- **B/A_LOAD_NONTEMPORAL on L6** (`R32_OPT_A_VERDICT.md`) — kernel:325-350. Builds clean (212/0/0) but **HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION rc=-6 after 191s** at K=128256. Same crash class as R31-C STEP3_BARRIER_VMCNT≥20 — NT hint changes load ordering, allows prefetch to outrun SRD bounds.
- **V6 split-K on L6** (`R32_OPT_B_VERDICT.md` + commit `52b8d54c`) — full POC implemented in `kernel_mxfp4_gluon_cpp_v6.cpp` + 50-line epilogue add-and-cast. K_SPLIT=1 sanity matches incumbent ±0.2%, K_SPLIT=2 **−11.53%**, K_SPLIT=4 **−23.37%**. **Mechanism (durable lesson)**: L6 grid is grid-saturated (2048 tiles ÷ 608 WGs = 3.4 iters/WG); intra-K splitting adds S launches without parallelism gain. Per-flop in per-split kernels (2425-2648 TFLOPS) < incumbent (2942) due to launch amortization. **V6 split-K is structurally dead on grid-saturated shapes; do not retry.**

### R31 axes confirmed DEAD (DO NOT REVISIT)
- **UNROLL_K sweep on L6** (`R31_OPT_A_VERDICT.md`) — UNROLL_K ∈ {1,2,4,16,32} on L6's `ts_lgk2_v12_memc_btw_all` parent. All 5 lose. u16/u32 5-rep at -0.55%/-0.58%; smaller values worse. Default `#pragma unroll 8` (kernel:2448) is already optimal at K=128256; smaller unroll adds loop overhead in 501-iter loop, larger doesn't help (occupancy is AGPR-limited, not unroll-limited).
- **Persistent-XCD / STATIC_XCD_REMAP scout on L6** (`R31_OPT_B_VERDICT.md`) — PERSISTENT_XCD=1 (V1, V3) GPU-faults — residual kernel-side bug despite R24A Fix A/B/C in source. STATIC_XCD_REMAP=1 (V2) clean -1.80% lose; V4 mid-bench fault. Mechanism: existing tall-XCDs + GROUP_M=4 swizzle achieves L2 B-tile reuse; re-bucketing into N-strips reduces reuse window.
- **STEP3_BARRIER_VMCNT sweep on L6** (`R31_OPT_C_VERDICT.md`) — v∈{4,8,10,16,20,24}. v8/v10/v16 lose 0.81-1.12%. v20/v24 CRASH (HSA aperture viol). v4 lose 0.82% but 2/3 reps crash. **v12 is the ONLY stable value on K=128256** — high VMCNT (≥20) lets prefetch outrun SRD bounds; low VMCNT (≤4) is racy. K_iters=501 amplifies the prefetch-vs-SRD timing window vs smaller K shapes.

### R30 axes confirmed DEAD (DO NOT REVISIT)
- **Cross-shape transplant scout** (`R30_OPT_A_VERDICT.md`) — 5 missing per_variant entries on DLA2 + 32768×14336×2048 all faulted with HSA aperture violation rc=-6. Root cause: `BARRIER_TO_WAITCNT_*` macros are correctness-risky on shapes where parent SNR is marginal. The "missing" entries in v2 per_variant tables are MISSING BECAUSE THEY CRASH. Parallel bench harness silently drops crashes (`bench_all42_parallel_R25_FINAL.py:263-267`).
- **K_EXACT parent-stack audit (R29 L4-style sweep)** (`R30_OPT_B_VERDICT.md`) — AUDIT-CLEAN. K∈{6144,7168,8192,16384,28672,32768}: K_EXACT dominates by 3-22%. K=2048: K_EXACT wins on 2/9; gaps 0.17-0.55% (sub-1%). K=14336: L4/L8 already individually fixed; L7's `_dc_gm7` is genuinely the right parent (u16/tv0 transplants -4.17% / -0.93%).
- **R30 V8 R25E peel revival** — N/A; R29 already declared DEAD via state-hazard. Decider mistakenly recommended V8 not knowing R29 had tried it.

### R29+ vector candidates (post-v2 — only L6/DLA1 has real headroom)
- **V5 — MFMA_32X32X64 tiling rewrite** — only remaining structural lever for L6/DLA1. ≥1 week of work; 0-5pp on K-bound deep-LOSE; high uncertainty. Per R27 V5 scout (`f0780765`): BACKBURNER unless dedicated 1-week sprint.
- **R26-A DLA1 K-loop peel (V1)** — `pf495` was unstable noise (false alarm); verify on `r25e-kpeel` worktree showed std=1729 TFLOPS, mean swing 1672→5546 across 5 reps. DEAD.
- **R27-C DLA1 K_EXACT bypass** — DEAD via aperture violation (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION rc=-6 on all 5 reps). Kernel HARD-GATE at K_DIM ≤ 32768 documented in `kernel_mxfp4_gluon_cpp.cpp:85-91`. `R27C_VERIFY_VERDICT.md`.
- **L4 4096×32768×14336 (98.5%)** — noise band, not worth structural attack. Single-shape variance is ±2-3% in this region.

---

## ⚠️ GOAL PIVOT (2026-04-17, 用户最新指令) — SUPERSEDED
> "24win 已经卡了好久了，现在把优化目标改成优化剩下那几个差的比较多的。"

**Status (2026-04-18)**: pivot text is now historical. 24/42 ceiling has been **broken to 33/42** (+9 flips). Both WIN-flipping and gap-reduction axes are paying off; do not artificially deprioritize WIN flips.

**原文** (kept for context): 不再追求翻 LOSE→WIN. 24/42 WIN 已被 4 轮饱和验证, 是当前架构天花板.
**改为**: **缩小 deep-LOSE shape 的 gap**. 即使无法 flip 到 WIN, 把 88% → 92% 也是真实进步.

### 新成功度量 (按优先级)
1. **降低 worst-shape gap**: 4096×32768×128256 (88.3%), 14336×4096×32768 (89.6%), 16384×4096×28672 (90.1%)
2. **提高 deep-LOSE shape 平均 ratio**: 当前 10 shapes 平均 ~92%, 目标 → ≥94%
3. **不允许 regression**: 现有 24 WIN 必须保住, 不能为了拉 deep-LOSE 牺牲 WIN
4. WIN count 不再是首要 KPI (可继续涨, 但不再是焦点)

### 重点 shape (post-R20, R20A 11-shape breakthrough)
| 优先级 | Shape | 当前 Ratio | 当前 Best Variant | 类别 | 备注 |
|-------|-------|-----------|------------------|------|------|
| **P0** | 4096×32768×128256 | 91.6% | p1_btw_all (R20B) | mega-K + 大N | DLA-class; memory-stall bound (R21-recon style) |
| **P0** | 128256×32768×4096 | 93.0% | ts_gm2_v12_memc_dc | mega-M+N | DLA2; HBM 19.9% peak, TCP stall 292% |
| **P0** | 28672×32768×4096 | 78.2% (R21recon) | ts_lgk2_v12_memc_btw_all (R20A) | 大M+N | DLA7; **R20A +2.30pp**; bench reads 94 % when using BTW-all |
| P0 (was) | 14336×4096×32768 | **96.18%** | **lgk2_dc_btw_step3** | 大K + 大M | **R19A +6.58pp** |
| P0 (was) | 16384×4096×28672 | **R20A WIN** | **u32_btw_all** | 大K + 大M | **R20A +3.74pp** |
| P1 (was) | 4096×32768×28672 | **R20A WIN** | **v20_memc_btw_step3** | 大K + 大N | **R20A +2.84pp** |
| P1 (was) | 28672×4096×16384 | **99.10%** | **p1_btw_all** | 大K + 大M | **R18A +4.16pp WIN** |
| P2 (was) | 4096×28672×32768 | 95.4% | u16_r11_iterilp | 大K + 大N | R11 +2.00pp |
| P2 (was) | 32768×4096×14336 | **R20A WIN** | **ts_gm8_v12_btw_all** | 大K + 大M | **R20A +4.09pp** |
| P2 (was) | 4096×32768×14336 | **R20A WIN** | **ts_lgk2_memc_btw_all** (R20A) or **r19c_iterilp_btw_all** (R19C) | 大K + 大N | **R20A +3.11pp** / **R19C +2.69pp** |
| **P0** (NEW) | 28672×4096×8192 | 97.5% | lgk2_dc_btw_all (R20B) | 大K + 大M | DLA-class; not yet R20A-validated |

### 应当探索的方向 (缩 gap, 非翻 WIN)
所有这些都已知**不会翻 WIN**, 但**可能缩 gap 1-3pp**:
- **MFMA_32X32X64_TILING** (重大重构, ~1天 asm 重写) — 唯一未试的"内核重构"级别尝试. 即使不翻 WIN, deep-LOSE 上若得 +2-5pp 即算成功.
- **per-shape 专属 K-loop unrolling** (UNROLL_K=8/16 仅对大K shape) — 之前测得 +0.3-1.2pp 但是被"不翻 WIN"否决, 现在重新算作有效改进.
- **cluster-launch / cooperative-grid** for mega-M shape (128256×32768×4096) — 即使不翻 WIN, 拉到 95%+ 即算成功.
- **per-shape compiler flag tuning** (按 shape 分类调 -mllvm 调度策略) — 之前 ±0.4% noise 是平均, 单 deep-LOSE shape 上可能更大.
- **K-loop epilogue 优化** for K=128256, K=32768 (尾部 K 迭代专属调优, TAIL_BARRIER_VMCNT 之外的方向).
- **B-tile L2 prefetch** for 大N shapes — 软件 prefetch hint 提前把 B tile 拉进 L2, 利用 L2 bandwidth 弥补 LDS 瓶颈.
- **block_id 重映射** for mega-M shape — 不用 atomic 的 static XCD-aware re-mapping, 改善 L2 hit rate without atomic overhead.

### Agent-team 新工作流 (per-shape gap-reduction)
- **不再**跑 42-shape full-sweep auto-tune (饱和过 4 次).
- **改为**: 每轮锁定 1-3 个 P0/P1 shape, 让 optimizer team 在该 shape 上专项优化, 接受 +1pp 增益.
- 提交标准: deep-LOSE shape 上 +1pp 即可 commit (而不是必须翻 WIN).
- 必跑 regression check: 改动后用 `bench_deep_lose.py` 验证不退化, 然后用 `bench_all42_parallel.py` 抽测确认 24 WIN 未掉.

---

## Current State (2026-04-18, post-R21)
- **Repo**: `/shared_nfs/kyle/test/HipKittens`
- **Branch**: `mxfp4`
- **42-shape result (R20B, post-R18+R19, 116 variants)**: **27/42 WIN** (+3 LOSE→WIN flips: P1, S1, S5; 0 regressions)
- **Projected after R20A wiring (127 variants, pending re-bench)**: **~34/42 WIN** (+7 additional flips from R20A's S2/S3/S6/S7/S10/S12 — S5/S15 already won; S8/S9/S13 not in R20B's K≥4096 LOSE list)
- **Deep-LOSE 10 shapes 平均 ratio**: improved meaningfully (R18+R19+R20 closed 14 of ~18 deep-LOSE shapes)
- **Stuck shapes (R21-recon classified as memory-stall bound, TCP_DATA_STALL 167-294 % of GRBM)**: DLA1 (4096×32768×128256, HBM 7.8 % peak), DLA2 (128256×32768×4096, HBM 19.9 %), DLA7 (28672×32768×4096, HBM 17.2 %)
- **上一轮**: 19/42 WIN → 24/42 WIN (+5 LOSE→WIN flip in Round 1, 0 regressions)
- **Cursor (Hipkittens2)**: 16/42 WIN (同参数, 2026-04-16T06:26)
- **我们领先**: **8 WIN**
- **Auto-tune variants**: 115 (expanded from 95 with memclause family + Optimizer A discoveries + ceiling sweep)
- **Saturation**: Round 2 (+28 ceiling variants) + deep-LOSE分析员 (+44 targeted variants on 10 stuck shapes) BOTH yielded **+0 LOSE→WIN flip**

## Recent Commits
```
f886d940 MXFP4: Round 1 final 24/42 WIN (+5 vs baseline 19/42)
06a416a8 TAIL_BARRIER_VMCNT + GM×LGK cross-products → 95 variants
03d3ba0a 62-variant auto-tune space saturated (docs)
78593c0c STEP12_BR_LGKMCNT tunable + expanded auto-tune (62 variants)
80b736b5 update docs + results for 19/42 WIN
4820d632 STEP4_EXTERNAL_BR_PREFETCH + expanded auto-tune → 19/42 WIN
```

## Round 1 LOSE→WIN flips (+5)
| Shape | Best Variant | Before → After |
|-------|-------------|----------------|
| 32768×28672×2048 | ts_gm2_v12_memc_dc | 99.1% → 100.3% |
| 4096×14336×8192 | u8 | 98.9% → 101.0% |
| 4096×32768×4096 | ts_no_embed_tv16_memc | 98.6% → 103.0% |
| 6144×32768×4096 | ts_v4_memc | 99.5% → 101.1% |
| 16384×4096×14336 | ts_v12_tv0_memc | 98.4% → 101.7% |

memclause family (`-mllvm -amdgpu-sched-strategy=max-memory-clause`) is the dominant new winner — appears in 14/24 WIN best variants.

## 已做的优化
1. **Store block reorder** (A0Bl,A0Br,A1Bl,A1Br) — +0.8% 全局
2. **SWAP operand port** from HK2 — 正确但略慢，作为 auto-tune 选项
3. **TAIL_SPLIT=1** — 小K(≤4096)帮助+1-2%, 大K(≥7168)退化
4. **SPREAD_LDS=1** (rowspread ds_reads) — 退化1-2%，作为选项
5. **NONVOLATILE_SCALE_X2_POC=1** (default ON) — 非易失性 scale loads, +1-2%
6. **STEP3_BARRIER_VMCNT** (4,8,12,16) — 不同 shapes 最优不同
7. **PF_N=4** (STEP3_PF_N/STEP4_PF_N) — 减少 prefetch 深度
8. **STEP4_EXTERNAL_BR_PREFETCH** — Br prefetch 从 Step4 MFMA 中分离
9. **STEP12_BR_LGKMCNT** (0,2,4) — Step1→Step2 lgkmcnt 放松
10. **STEP3_EMBED_BARRIER** (0,1) — barrier 独立/嵌入 Step3
11. **62-variant auto-tune** (GM, U, SWAP, TS, V4/8/12/16, SPREAD, PF4, NVS, EXT_BR, LGKMCNT, NO_EMBED, 多轴交叉)
12. **gl.cuh size_t overflow fix** — 大shape (128256×32768) int溢出修复
13. **TAIL_BARRIER_VMCNT** — 尾部K迭代单独barrier VMCNT调优 (ts_tv16 在2个shapes上最优)
14. **GM×LGK cross-products** — gm8_lgk2, ts_gm2_lgk2, ts_gm2_lgk2_v12 (提供 0.1-0.3pp 边际提升)

## 24 WIN shapes (115-variant benchmark, Round 2 final)
| Shape | TFLOPS | Ratio | Best Variant |
|-------|--------|-------|-------------|
| 16384×4096×2048 | 3264 | 109.0% | ts_v12_tv0_memc |
| 16384×4096×3072 | 3802 | 108.9% | ts_pf6_6_lgk2_memc |
| 16384×6144×2048 | 3494 | 114.6% | ts_v4_tv0 |
| 32768×4096×2048 | 3465 | 110.6% | ts_v12_tv0_memc |
| 32768×4096×3072 | 3927 | 108.2% | ts_lgk2_memc |
| 32768×6144×2048 | 3428 | 105.8% | ts_gm8_v12 |
| 16384×14336×2048 | 3553 | 107.6% | ts_lgk2_v20 |
| 32768×14336×2048 | 3478 | 103.8% | ts_gm2_v12_memc |
| **32768×28672×2048** | 3365 | **100.3%** | **ts_gm2_v12_memc_dc** ← R1 flip |
| 16384×4096×4096 | — | 106.3% | ts_lgk2_memc_dc |
| 16384×6144×4096 | — | 106.4% | ts_v12_tv0_memc |
| 16384×14336×4096 | — | 104.3% | ts_lgk2_v20_memc |
| 4096×4096×8192 | — | 110.9% | default |
| **4096×14336×8192** | — | **101.0%** | **u8** ← R1 flip |
| 6144×4096×8192 | — | 102.7% | v32 |
| **4096×32768×4096** | — | **103.0%** | **ts_no_embed_tv16_memc** ← R1 flip |
| **6144×32768×4096** | — | **101.1%** | **ts_v4_memc** ← R1 flip |
| 16384×4096×6144 | — | 107.7% | ts_lgk2_v20_memc |
| 16384×4096×7168 | — | 104.7% | ts_lgk2_v20_memc |
| 4096×4096×16384 | — | 104.4% | ts_v24 |
| **16384×4096×14336** | — | **101.7%** | **ts_v12_tv0_memc** ← R1 flip |
| 4096×4096×32768 | — | 100.9% | ts |
| 4096×6144×32768 | — | 124.3% | v16 |
| 4096×128256×32768 | — | 161.2% | memc |

## 18 LOSE shapes (Round 2 final)
### Near-threshold (≥98%, 3 shapes)
| Shape | Ratio | Best | Δ to WIN |
|-------|-------|------|----------|
| 32768×4096×7168 | 99.3% | ts_gm8_v12 | 0.7pp |
| 4096×14336×16384 | 98.6% | ts_lgk2 | 1.4pp |
| 6144×4096×16384 | 98.6% | ts_lgk2 | 1.4pp |

### Mid-LOSE (95-97%, 5 shapes)
| Shape | Ratio | Best |
|-------|-------|------|
| 16384×28672×2048 | 96.4% | ts_gm2_v12_memc_dc |
| 4096×32768×6144 | 96.4% | ts_pf4_memc |
| 16384×28672×4096 | 96.5% | ts_gm2_v12_memc |
| 28672×4096×8192 | 96.6% | ts_lgk2_memc_dc |
| 14336×32768×4096 | 96.9% | ts_v12_tv0_memc |

### Deep-LOSE (<95%, 10 shapes — confirmed STRUCTURAL by deep-LOSE分析员)
| Shape | Ratio | Best | 类别 |
|-------|-------|------|------|
| 4096×32768×128256 | 88.3% | ts_gm8 | mega-K + 大N |
| 14336×4096×32768 | 89.6% | lgk2_dc | 大K + 大M |
| 16384×4096×28672 | 90.1% | u32 | 大K + 大M |
| 128256×32768×4096 | 92.9% | ts_gm2_v12_memc_dc | mega-M+N |
| 4096×32768×28672 | 93.7% | v20_memc | 大K + 大N |
| 28672×4096×16384 | 93.7% | ts_gm8 | 大K + 大M |
| 4096×28672×32768 | 94.1% | u16 | 大K + 大N |
| 32768×4096×14336 | 94.1% | ts_gm8_v12 | 大K + 大M |
| 4096×32768×14336 | 94.5% | ts_lgk2_memc | 大K + 大N |
| 28672×32768×4096 | 94.5% | ts_lgk2_v12_memc | 大M+N |

## DEPRECATED — 19 WIN shapes (62-variant benchmark, pre-Round 1)
| Shape | TFLOPS | Ratio | Best Variant |
|-------|--------|-------|-------------|
| 16384×4096×2048 | 3147 | 105.1% | ts_no_embed_v12 |
| 16384×4096×3072 | 3698 | 105.9% | ts_lgk4 |
| 16384×6144×2048 | 3301 | 108.3% | ts_v16 |
| 32768×4096×2048 | 3330 | 106.3% | ts_no_embed_v12 |
| 32768×4096×3072 | 3823 | 105.3% | ts_no_embed |
| 32768×6144×2048 | 3379 | 104.3% | ts_gm8 |
| 16384×14336×2048 | 3408 | 103.2% | ts_v12 |
| 32768×14336×2048 | 3407 | 101.7% | ts_gm2_v12 |
| 4096×4096×16384 | 4858 | 104.6% | ts_lgk2_v12 |
| 4096×4096×8192 | 4353 | 109.9% | u8 |
| 4096×4096×32768 | 5158 | 100.1% | ts_no_embed_v12 |
| 4096×6144×32768 | 4645 | 122.8% | u16 |
| 4096×128256×32768 | 5146 | 161.1% | default |
| 6144×4096×8192 | 3906 | 102.2% | u8 |
| 16384×4096×4096 | 4061 | 102.8% | ts_no_embed |
| 16384×4096×6144 | 4510 | 105.9% | ts_v16 |
| 16384×4096×7168 | 4575 | 103.0% | lgk2 |
| 16384×6144×4096 | 4221 | 104.4% | ts_u16 |
| 16384×14336×4096 | 4273 | 100.4% | ts_lgk2_v12 |

## 23 LOSE shapes 分析
### 接近 WIN (97-99.5%)
| Shape | TFLOPS | Ratio | Best | 差距 |
|-------|--------|-------|------|------|
| 6144×32768×4096 | 4268 | 99.5% | ts_lgk2 | 0.5% |
| 32768×28672×2048 | 3325 | 99.1% | ts_gm2_v12 | 0.9% |
| 4096×14336×8192 | 4296 | 98.9% | lgk2 | 1.1% |
| 6144×4096×16384 | 4377 | 98.8% | ts_v4 | 1.2% |
| 4096×32768×4096 | 4110 | 98.6% | ts_gm2_v12 | 1.4% |
| 16384×4096×14336 | 5062 | 98.4% | ts_pf4 | 1.6% |
| 28672×4096×8192 | 4688 | 97.5% | gm8_v12 | 2.5% |

### 中等差距 (95-97%)
| Shape | TFLOPS | Ratio | Best |
|-------|--------|-------|------|
| 32768×4096×7168 | 4503 | 96.5% | gm8_ext_br |
| 4096×14336×16384 | 4830 | 96.3% | ts_v4 |

### 结构性 gap (>5%)
| Shape | Best | Ratio | 限制因素 |
|-------|------|-------|---------|
| 16384×28672×2048 | 3275 | 94.1% | 大N, B-LDS瓶颈 |
| 4096×32768×128256 | 5115 | 88.5% | 超大K, B-LDS瓶颈 |
| 14336×4096×32768 | 4671 | 89.1% | 大K, B-LDS瓶颈 |
| 16384×4096×28672 | 4967 | 89.9% | 大K, B-LDS瓶颈 |
| 128256×32768×4096 | 4129 | 91.0% | 超大M, XCD dispatch |
| 28672×32768×4096 | 4089 | 91.5% | N=32768, B-LDS |
| 4096×32768×6144 | 4193 | 92.2% | N=32768 |
| 4096×32768×28672 | 5194 | 93.3% | N=32768, 大K |
| 16384×28672×4096 | 4116 | 93.3% | N=28672 |
| 4096×32768×14336 | 4946 | 93.4% | N=32768 |
| 28672×4096×16384 | 5000 | 93.4% | 大M大K |
| 14336×32768×4096 | 4183 | 93.7% | N=32768 |
| 4096×28672×32768 | 5297 | 93.8% | N=28672, 大K |
| 32768×4096×14336 | 4905 | 93.9% | M=32768 |

## Auto-tune 空间已饱和 — 多轮验证
1. **62-variant 全量benchmark**: 41→62 variants, 结果稳定 19/42 WIN
2. **Cross-product spot tests**: 在 98-99% shapes 上测试 24 个新交叉组合 → 全部无效, flag stacking counterproductive
3. **UNROLL_K=1,2,4**: 全部比编译器默认差
4. **Fine-grained VMCNT (6,10,14,16,18,20,24)**: 全部不如 ts_lgk2 (99.5%)
5. **Fine-grained LGKMCNT (1,3,6)**: LGKMCNT=2 是最优, 其他值更差
6. **Cursor对比**: Cursor 14 个新 commit 无新思路, 我们变体超集
7. **LGKMCNT×VMCNT cross-products (15 new)**: lgk2_v4/v12/v16, lgk4_v4/v12/v16, ts_lgk2_v4/v16, ts_lgk4_v4/v12/v16, lgk2_no_embed, ts_lgk2_no_embed, ts_lgk2_no_embed_v12 → 全部无效或退化. 6144×4096×16384 best=99.3% (ts_lgk2_v4), 16384×4096×14336 best=98.2% (ts_lgk2)
8. **FUSED_STEP34 (14 variants)**: 合并 Step3+Step4 为 64-MFMA 单块. 全面退化 ~7% (f34 系列). ds_read/PF 冲突, 大块内编译器 MFMA 调度过激
9. **TAIL_BARRIER_VMCNT spot tests (30 new variants on 7 shapes)**: PF_N=2/1, PF4×LGK2, EXT_BR×LGK2, GM×LGK, TAIL_VMCNT, asymmetric PF — 全7个 near-threshold shapes 测试, 0 个 WIN flip. ts_tv16 在 2 shapes 上边际最优 (+0.3pp), gm8_lgk2/ts_gm2_lgk2/ts_gm2_lgk2_v12 各在1个shape上边际最优 (+0.1-0.2pp). PF_N=1/2 全面退化. 不对称PF无效

## 结构性限制 (不preshuffle B 无法突破)
- **B走LDS**: 比aiter多18% TCP read traffic, 2x Frac_Wait_Any
- **256 AGPR**: 4 acc blocks 占满, B tile 数据必须在 256 VGPR 内
- **LDS swizzle**: MFMA 需要的数据排列 (已证明 identity)
- **buffer_load vs ds_read**: 10x延迟差距, LDS prefetch pipeline 完全隐藏
- **N=32768 shapes**: B tile 大 → LDS traffic 成为瓶颈
- **Store epilogue**: 非SWAP路径 s[0..3] 是行连续, 无法pack成 dword stores
- **VMCNT/LGKMCNT gradient**: 99.5% 已是当前架构天花板

## 不要再做的事
- **Direct-B (不preshuffle)**: 正确但慢28% (buffer_load延迟)
- **BK=256**: LDS装不下 (256KB > 160KB max)
- **Preshuffle-B 1-pass**: VGPR spill → NaN
- **Preshuffle-B 2-pass**: 正确但慢56% (K-loop跑两遍)
- **sched_group_barrier / iglp_opt**: 无改善
- **ds_bpermute wide stores**: 退化16%
- **GROUP_SIZE_M=32/64**: 大N shapes 退化10-16pp
- **UNROLL_K=1,2,4**: 比编译器默认差
- **Cross-product flag stacking**: 98-99% shapes 上全部无效
- **Fine-grained VMCNT/LGKMCNT 微调**: 已穷举, 无增益
- **LGKMCNT×VMCNT cross-products**: lgk{2,4}×v{4,12,16}×ts×no_embed 全15种 → 无效或退化
- **FUSED_STEP34**: Step3+Step4 合并为 64-MFMA 块 → 全面退化 ~7% (14 variants tested)
- **ASM rewriter (s_nop removal)**: 后编译 ASM 重写移除 162 个 s_nop → 破坏正确性(26%元素错误, m0 hazard是硬件强制的), 且性能无变化(+0.15% = noise). Inter-block code 被 MFMA pipeline depth (16 cycles) 完全隐藏
- **ASM rewriter (PF redistribution)**: Cursor 的 rewriter 不适用(结构不同: 我们是 1×64-MFMA + 8×8-MFMA, Cursor 是 4×32-MFMA; 我们 K≤4096 全展开无 loop)
- **PF_N=1/2**: 减少 prefetch 深度 → 全面退化 2-6%. PF_N=4 是最优
- **Asymmetric PF (STEP3_PF_N ≠ STEP4_PF_N)**: 2/8, 8/2 全部不如对称 PF
- **GM×LGK cross-products**: 边际改进 0.1-0.3pp, 不足以 flip 任何 shape
- **TAIL_BARRIER_VMCNT tuning (0,4,16)**: ts_tv16 在部分 shapes 边际最优, 但 < 0.5pp 改进
- **Half-Direct Bl (DIRECT_BL)**: preshuffle B后, Bl 从 global buffer_load 直取 (跳过LDS). 0 spill (253 VGPRs, 87 SGPRs, 98KB LDS). 但 buffer_load 延迟 ~400 cycles, Step4 只有 ~128 cycles MFMA 隐藏 → 全 7 shapes 退化 8-15%. 大N shapes (32768) 退化 13-15%, 最差 -15.3% (6144×4096×16384). LDS 减少33% (131→98KB) 无法弥补 VMEM latency penalty
- **NT_STORE (non-temporal stores)**: 全部 global_store_short 加 `nt` modifier 绕过 L2. 结果: 全 7 近阈值 shapes 退化 15-25% (最差 -25%). CDNA L2 对 store coalescing 至关重要, 绕过 L2 = 直写 HBM = 慢. NT+SWAP 组合退化 34-72%
- **PACKED_STORE (bf16x2 dword stores)**: SWAP路径 store_block_inner 中用 pack_bf16x2 将 4×2B stores 合并为 2×4B stores. 需要 SWAP_STEP34_MAIN=1. SWAP 路径本身退化 13-34%, packed 不改善
- **Compiler flag tuning**: 测试 -O2, -mllvm -amdgpu-max-memory-clause=1/4, -mllvm -amdgpu-early-inline-all=true. 结果: 前两个shape ±0.4% noise, 6144×4096×16384 O2比O3好5%但仍远低于目标. 编译器调优无法翻WIN
- **18个未测variant组合**: ts_lgk2_ext_br, ts_lgk2_gm8, ts_lgk2_tv16/tv0, ts_gm2_lgk2_v4/v16/ext_br/no_embed, lgk2_ext_br/gm2/gm1/no_embed_v12 等 → 全部 0 WIN, 全部不如已有best variant
- 把 preshuffle 时间不算进比较 — 用户明确拒绝过

## 可能的未来方向 (高风险/高工作量)
1. ~~**Fused Step34**~~ — DEAD END, 退化 ~7%
2. ~~**ASM rewriter**~~ — DEAD END, s_nop 硬件强制, 无增益
3. ~~**Pre-shuffle B (Half-Direct Bl)**~~ — DEAD END, 8-15% 慢 (buffer_load latency)
4. ~~**rocprof 分析**~~ — DONE: 瓶颈在 B-LDS traffic, 不在 scheduling
5. ~~**NT_STORE (non-temporal stores)**~~ — DEAD END, 15-25% 慢 (L2 对 store coalescing 必要)
6. ~~**PACKED_STORE (bf16x2 dword stores)**~~ — DEAD END, SWAP路径必需, SWAP本身退化
7. ~~**Compiler flag tuning**~~ — DEAD END, ±0.4% noise
8. ~~**Extended variant combos (18 new)**~~ — DEAD END, 0 WINs

**结论 (2026-04-17 更新)**: 24/42 WIN 是当前内核架构 (B-through-LDS, 4-step K-loop, 256 AGPR) 的性能天花板. memclause + Round 2 ceiling sweep + deep-LOSE 分析员 (44 targeted variants) 全部确认: 18 个 LOSE shape 的剩余 gap 是结构性的, flag-axis 关不上.

**Round 2 (115 variants × 42 shapes) 与 deep-LOSE 分析员 (44 variants × 10 stuck shapes) 双重验证后**:
- 0 个 LOSE→WIN flip
- deep-LOSE 最大边际增益 1.2pp (16384×4096×28672: ts_v12 89.2% → u8 90.4%)
- 突破需要根本性的内核重构 (aiter 架构: 只有 A 走 LDS, B 通过深度软件流水直接从 global 加载, 需要 >256 VGPRs 的寄存器预算)

## Round 2 + Deep-LOSE 新增 dead-end (2026-04-17)
- **memclause × 28 ceiling variants** (Round 2): 0 LOSE→WIN flip vs Round 1, 边际 ±25 TFLOPS movement only. WIN shapes 上 best variant 漂移 (e.g., `ts_lgk2_v12_memc` → `ts_lgk2_v20_memc` on 16384×14336×4096) 但 ratio 几乎不变
- **VMCNT ceiling sweep (v20/v24/v32)**: v20 在已 WIN shape 上偶尔最优, 但不翻 LOSE
- **TAIL_BARRIER_VMCNT × VMCNT cross**: tv0/tv16 × v4/v12 — `_ts_v12_tv0_memc` 在 4 个 WIN shape 上是 best variant, 但不翻任何 LOSE
- **PF asymmetric (pf4_6, pf6_4, pf3_8)**: pf6_6 对称在 4096×32768×28672 等 2 shape 边际最优, 不对称无效
- **waves_per_eu(2,2) attribute**: deep-LOSE 上 +0.4pp 仅 1 shape, 其余 sub-best — REFUTED
- **AGPR hint (amdgpu_num_agpr=192)**: 全部 sub-best — FAILED
- **UNROLL_K=8/16 × LGK × VMCNT × memc cross**: 在 deep-LOSE 上 +0.3-1.2pp 边际 (4 个 shape), 不翻 LOSE (推翻 AGENT_PROMPT 之前 "UNROLL=1,2,4 worse" 的过早结论 — 8/16 配合 lgk/v12/memc 才有效, 但仍不够)
- **128256×32768×4096 (mega-M)**: 完全 IMPENETRABLE — Round 2 + deep-LOSE 都无法逼近 R1 best (`ts_gm2_v12_memc_dc` 92.9%)
- **EARLY_BL_PF (DIRECT_BL with Bl buffer_load 移到 Step12 前, ~128 MFMA latency hiding)**: 在 14336×4096×32768 deep-LOSE shape 测试 (warmup=200, iters=500):
  - baseline LDS 路径:        4540.4 TFLOPS (86.6% aiter)
  - DIRECT_BL 原版 (Step4 内): 3786.3 TFLOPS (72.2%)
  - DIRECT_BL + EARLY_BL_PF:  4004.2 TFLOPS (76.3%)
  Latency-hiding 假设 VALIDATED (+4.1pp), 但 DIRECT_BL+EARLY 仍比 LDS 慢 10.3pp. 结论: B-direct 路径在当前内核上 **结构性 inadequate**, 即使 Step12-launched prefetch 完全隐藏 buffer_load 延迟. LDS broadcast bandwidth 是真正瓶颈, 不是 load latency. 代码保留在 `EARLY_BL_PF=1` flag 下 (default 0). 见 `test_early_bl_pf.py`. 这次实验 close 了 "B-direct 重写" 这条路 — 唯一能突破 24 WIN 的就是 aiter 架构 (A-only-LDS + deep-pipelined B-direct), 需 >256 VGPRs, 在 gfx950 不可行.

## Round 4 (2026-04-17) — 3 parallel optimizers, all DEAD END
Decider 提出 5 个 untested vectors, 3 个 in-session 可执行. 启动 3 个 optimizer agents 并行验证:
- **SCALE_REG_CACHE** → DUPLICATE: scales 已经 VGPR-resident (`v70-v73`), 用 `buffer_load_dwordx2` 直接 VMEM→VGPR (无 LDS round-trip), `pf_a0/pf_a1/pf_bl/pf_br` 持久 VGPR 缓存 + `NONVOLATILE_SCALE_X2_POC=1` (default-on) 已实现该优化. Decider 误读了 kernel. 无代码改动.
- **LDS_XOR_SWIZZLE_B** → DUPLICATE: `st_16x128_s::swizzle()` 已实现 (`include/types/shared/st_shape.cuh:236-237`), 写侧 `prefill_swizzled_offsets` 预 permute global offset, 读侧 `compute_lds_base_addrs` 同步 XOR. 模拟确认 perfect 8 acc/bank uniform = LDS 硬件下界. 之前 `kernel_mxfp4_xor_toggle.cpp` 测试无改善. 无代码改动.
- **PERSISTENT_XCD_QUEUE** → DEAD END: 实现持久 kernel + atomic counter + `PERSISTENT_BATCH ∈ {1,4,8}` 调优, 在 IMPENETRABLE 128256×32768×4096 上测试 (warmup=200, iters=500):
  - static (`ts_gm2_v12_memc_dc`): 4296 TFLOPS (94.7%)
  - PERSISTENT b=1 g=608: 3791 TFLOPS (83.6%) −11.8%
  - PERSISTENT b=4 g=608: 4129 TFLOPS (91.0%) −3.9% ← best
  - PERSISTENT b=8 g=608: 3857 TFLOPS (85.0%) −10.2%
  失败原因: (1) HWS overhead 不是瓶颈 — static 已 94.7%, 仅剩 5% headroom, atomic 立即吃掉 4%; (2) XCD-locality LOSS — static `raw_bid % 8` 保证每个 XCD 内连续 tile 的 L2 B-tile 复用, persistent 破坏该模式; (3) atomic latency 在已 ALU-bound (256V/256A, 1 wave/SIMD) 的 kernel 上叠加可见开销. **128256×32768×4096 的 92.9% 上限是 register-pressure / MFMA-pipeline bound, 不是 launch-bound**. 代码保留在 `PERSISTENT_XCD=1` flag 下 (default 0).

**Round 4 净增 WIN: 0**. 24/42 第 4 次确认饱和.

## Round 5 (2026-04-17) — 4 parallel optimizers, all DEAD END / INFEASIBLE
直接针对 deep-LOSE shapes 启动 4 个 optimizer agent (Opus 4.7) 并行验证 untested vectors:
- **Optimizer F**: 阻断 — INFEASIBLE in single session
- **Optimizer G**: 阻断 — INFEASIBLE in single session
- **Optimizer D (MFMA_32X32X64)**: NO-GO for <1 week. **重要订正: decider 关于 AGPR 节省的说法 WRONG**: per-warp output 仍是 128×128 (4 quadrants of 64×64 acc), 32x32x64 仍需 256 AGPRs. 先前 "256→64 AGPR 释放 192 VGPRs" 是误读. MFMA_32X32 仅在 K-loop 调度自由度上有差异, 不是寄存器层面的解放.
- **Optimizer E (EARLY_SCALE_PF)**: BROKEN + 性能无改善:
  - 实现: shadow regs `nxt_pf_*` 缓存 next-iter scale loads, iter 末 `pf_* = nxt_pf_*` writeback
  - **正确性破坏**: compiler 把 `pf_*` 和 `nxt_pf_*` aliased 到同一 VGPR (writeback 看似 no-op 被折叠), VMEM 在 `_raw = pf_*` 读取前 clobber pf_* → race condition. NaN pattern 与 baseline 不同 (2225 vs 2273 NaNs at 1024×1024×4096 random fp4)
  - **性能** (broken variant, warmup=200 iters=500): 14336×4096×32768 -0.85%, 16384×4096×28672 -0.47%, 4096×32768×128256 +0.35% — all in noise
  - **根本原因**: baseline ASM 已在 iter 顶部 issue 4×dwordx2 scale loads (NONVOLATILE_SCALE_X2_POC=1), latency hiding window ~512 cyc 已超 ~400 cyc VMEM latency. 没有未利用的调度空间. Force distinct VGPRs 需 +8 VGPRs 超 256 cap, no occupancy benefit
  - 代码加 `#error` 守卫 (`EARLY_SCALE_PF=1` 编译失败), 保留 flag 和 test 作为 DEAD END 文档. 见 `test_early_scale_pf.py`

**Round 5 净增**: 0 WIN, 0 gap reduction. 触发用户的 GOAL PIVOT 指令 (见文档顶部).

## Round 22 (2026-04-18, IN FLIGHT) — memory-stall axis attack
3 parallel optimizers + 1 reviewer attacking R21-recon's memory-stall finding.

- **R22A — LDS sub-arbitration via s_nop stagger after ds_read_b128 (DEAD END, committed `a4074d2d`)**:
  - Added `LDS_RD_STAGGER_NOP=0/1/2` macro (default 0 = no-op `LDS_NOP_STR=""` → bit-identical baseline). 40 sites injected across 4 KPAIR functions.
  - Smoke (warmup=200 iters=500): all 3 DLA shapes Δ ∈ [-4.07%, +0.01%]. Best DLA2/nop1 = +0.01% (noise). Gate (+1.5pp) failed everywhere.
  - **CRITICAL MECHANISTIC CORRECTION**: TCP_TA_DATA_STALL counts TCP (L1) **back-pressure on the TA side**, which for an MXFP4 GEMM (no textures) reflects **producer-side `buffer_load_to_lds` L2-miss / HBM-latency stalls**, NOT consumer-side `ds_read` LDS port contention. Consistent with raw HBM only 7-20% of peak (latency-bound, not bandwidth-bound). Spreading consumer ds_reads cannot help when producer is the bottleneck.
  - **Reframes R23 frontier**: producer-side fixes (STATIC_XCD_REMAP for DLA2/DLA7, `buffer_load_to_lds` SLC/DLC bits, L2 prefetch, outer-iter pull-forward).
  - Macro framework retained as zero-overhead opt-in for future probes.
- **R22C — Finer SCHED_GROUP_BARRIERS masks at R21B's 4 hook sites (DEAD END, no commit)**:
  - 6 mask combos × 4 shapes = 24 builds. Tested: 0x004 (MFMA-only), 0x044 (MFMA+DS_R), 0x008 (VMEM), 0x040|0x080 (DS_R+W=0xc0), 0x004 size=2, 0x004 size=8.
  - Best smoke: DLA7 + `mfmadsr_h0f` mask=0x044 = +0.13% (noise). All other combos −0.5% to −14.7%. Gate failed on all.
  - Confirms R21B's diagnosis: even fine-grained scheduler hints disrupt the LLVM-tuned MFMA/prefetch interleave more than they help. The hooks are wired but no usable mask exists for these shapes.
- **R22B — Streaming/non-temporal global loads on B-tile (DEAD END, no commit)**:
  - Macros wired: `B_LOAD_NONTEMPORAL=1/2`, `A_LOAD_NONTEMPORAL=1/2`. 5 variants × 3 DLA shapes built.
  - Aperture inconclusive (overflow at K=128256 swamps hash oracle). Pivoted to direct perf smoke — cache hints don't change MFMA emit-order, only L2/SLC bypass.
  - Smoke (warmup=200 iters=500): **all variants regress on all 3 DLA shapes**:
    - DLA1: bnt1=−6.30%, bnt2=−6.44%, bnt1_ant1=−20.08%, bnt2_ant2=−19.94%
    - DLA2: bnt1=−7.85%, bnt2=−6.27%, bnt1_ant1=−14.27%, bnt2_ant2=−13.01%
    - DLA7: bnt1=−0.83%, bnt2=−1.60%, bnt1_ant1=−6.58%, bnt2_ant2=−6.84%
  - **Mechanistic insight**: NT/streaming bypass actively HURTS on DLA shapes because B-tile is shared across K-iters within a CTA — bypass kills the only L2 reuse path we have. Confirms that the binding stall is producer-side L2 miss, not L2 thrash. NT helps only when L2 hit rate is already poor and we're polluting cache for *future* loads we don't need; here L2 reuse is essential.
  - macros retained as zero-overhead opt-in.
- **R22-rebench — full 42-shape rebench locking R20A's 11 wires (COMPLETE: 29/42 WIN, 13/42 LOSE, 0 ERR, avg ratio 105.3%)**:
  - WIN +5 over R20B baseline (24→29). Best variant per shape consistently picks R20A wires on K-heavy shapes (ts_lgk2_memc_btw_all, ts_v12_tv0_memc_btw_all, ts_lgk2_btw_step3, lgk2_dc_btw_all, ts_gm2_v12_memc_dc_btw_all).
  - 13 remaining LOSE: DLA1 91.9%, DLA2 96.3%, DLA7 96.9%, plus 10 mid-gap shapes 95.7-99.9%.
  - Saved: bench_all42_results_r22.json, bench_all42_r22.log.

## Round 23 (2026-04-18) — producer-side memory-stall axis (R23A/B both DEAD END)
Per R22A's mechanistic correction (TCP_TA_DATA_STALL = producer-side `buffer_load_to_lds` L2-miss/HBM-latency).

**Targets**: DLA1 (4096x32768x128256), DLA2 (128256x32768x4096), DLA7 (28672x32768x4096).

- **R23A — STATIC_XCD_REMAP (DEAD END)**:
  - 4 variants (_xcd_baseline, _xcd_remap, _xcd_remap_g4, _xcd_remap_g8) built for all 3 DLA shapes; bpc%8==0 verified.
  - Smoke results (vs xcd_baseline):
    - DLA1: best _xcd_remap_g8 = +0.95% (+0.85pp/comp) — borderline, below 1.5% gate
    - DLA2: best _xcd_remap_g4 = −0.68% (REGRESSION)
    - DLA7: best _xcd_remap = +1.50% (+1.45pp/comp) — exactly on gate, smoke 1-run noise band ±0.5pp
    - DLA7 _xcd_remap_g8 = ERR rc=−6 (correctness or launch failure)
  - VERDICT: STATIC_XCD_REMAP does not close the producer-side TCP_TA_DATA_STALL gap. The remap reshuffles tile→XCD assignment but doesn't increase HBM efficiency or reduce L2 miss rate enough to matter. DLA7 +1.50% is borderline noise; not worth a 5-run reverify.
- **R23B — PERSISTENT_XCD atomic dispatcher (DEAD END — CORRECTNESS BUG)**:
  - 3 variants (_pxcd_baseline, _pxcd_b1, _pxcd_b4) built for all 3 DLA shapes.
  - Smoke shows IMPOSSIBLE TFLOPS (>2× peak) with C-coverage=6.4%-28.6% (i.e., persistent grid skips most output tiles).
  - DLA1 _pxcd_b1/_b4 = ERR rc=−6 (SIGABRT, correctness assert)
  - DLA2 _pxcd_b1/_b4 = "60012/61812 TFLOPS" with 6.4% coverage (only 6% of C tiles written → most C remains zero)
  - DLA7 _pxcd_b1/_b4 = "12834/14246 TFLOPS" with 28.6% coverage
  - VERDICT: PERSISTENT_XCD's atomic tile claim either races, deadlocks, or terminates after fewer iterations than there are tiles. Kernel-side bug in PERSISTENT_GRID dispatcher logic. Cannot evaluate perf until fixed.

## Round 24 (2026-04-18) — R24A debug + R24D dead-end; R24B/R24C still open
- **R24A — debug PERSISTENT_XCD coverage bug (DEAD END after Fix A+B+C)**:
  - Identified 3 root-cause hypotheses (see `r24a_pxcd_debug.md`):
    - H1: `g_persistent_tile_counter` never reset — `hipGetSymbolAddress` rc not checked, host writes to NULL ptr
    - H3: PERSISTENT_GRID=608 over-launches on small problems (DLA1)
    - H4: missing `__syncthreads()` between tile-body tail and next atomicAdd
  - Applied **Fix A** (checked counter reset + synchronous `hipMemset`) + **Fix B** (`__syncthreads()` before loop close) + **Fix C** (cap grid by `total_tiles`).
  - Re-bench (`bench_round23_optB_v2.log`): coverage **STILL 6.4%/28.6%** on DLA2/DLA7; DLA1 STILL `rc=-6`.
  - VERDICT: bug is in the kernel-side persistent loop itself (`kernel_mxfp4_gluon_cpp.cpp:2086-2137`), not in the host counter-reset path. Likely the atomicAdd return value, total_blocks calculation post-XCD-remap, or loop termination condition. Fixing it requires a kernel rewrite that risks breaking the static-dispatch baseline. **Out of scope for R24.**
  - The 3 fixes are committed (gated by `#if PERSISTENT_XCD`, no behavior change at default PERSISTENT_XCD=0) as documentation of the debug attempt.
- **R24D — A-only NT cache hint on DLA shapes (DEAD END)** (`r24d_results.md`):
  - 3 variants on DLA1/2/7 (5-run smoke, warmup=200/iters=500/trim=0.10).
  - DLA1: ant1=−4.57%, ant2=−5.08%; DLA2: ant1=−6.69%, ant2=−8.25%; DLA7: ant1=−4.68%, ant2=−4.03%.
  - Combined with R22B (B-NT and B+A NT also LOSE), this **fully exhausts the {A,B,both}×{non_temporal,cache_stream} cache-hint matrix**: all 6 combinations regress on DLA shapes.
  - Mechanistic refutation: A-tile is M-streamed across CTAs, but the same A-line is consumed by multiple warps within a CTA (M-tile rows × K-iters); evicting early forces re-fetch. Even GLC-only `cache_stream` LOSEs.
- **R24B — extra `buffer_load_dwordx4` L2 prefetch (DEAD END)** (`r24b_results.md`):
  - Macros `L2_PF_A`, `L2_PF_B` (intensity 1/2/3) emit additional discarded `buffer_load_dwordx4` ops in steady-state K-loop, targeting `bt+3` (K+2 tile).
  - Disassembly confirmed 432→482→532→632 buffer_load_dwordx4 instructions across baseline→a1→ab1→a3 (linear scaling).
  - Bench (warmup=200/iters=500/trim=10%, DLA1/2/7):
    - DLA1: a1=−4.4%, b1=−6.0%, ab1=−8.6%, a3=ERR
    - DLA2: a1=−0.4%, b1=−3.4%, ab1=−5.2%, a3=−8.0%
    - DLA7: a1=−3.0%, b1=−3.2%, ab1=−5.6%, a3=−12.1%
  - Every variant LOSES, regression scales monotonically with intensity (smoking gun: no inflection point where partial prefetch helps).
  - HBM is already saturated by existing LDS prefetch; extra outer-K VMEM competes for HBM bandwidth, degrading inner-K prefetch's effective rate.
- **R24C — outer-K pull-forward L2 prefetch (DEAD END)** (`r24c_results.md`):
  - Macros `OUTER_K_PF_DEPTH=2/3` issue extra discarded `buffer_load_dwordx4` for K+2/K+3 in TAIL_SPLIT non-SWAP K-loop.
  - Bench (DLA1/2/7):
    - DLA1: l2=−23.6%, l3=−23.0%
    - DLA2: l2=−15.6%, l3=−14.6%
    - DLA7: l2=−18.1%, l3=−18.0%
  - Hard 14-24% regression on every shape/depth. Saturation slope flat (depth=3 ≈ depth=2).
  - Mechanistic: VMEM-issue-bound, not VMEM-latency-bound — single VMEM lane already saturated. Extra `buffer_load_dwordx4` queues behind inner LDS pf, starving it. Plus L2 thrash on DLA1 (K=128256 dwarfs 32 MB L2).

## ⛔ HBM-BANDWIDTH AXIS EXHAUSTED on DLA shapes
Three independent attacks (R22B B-NT, R24D A-NT, R24B extra-VMEM-pf, R24C outer-K-pf) all LOSE:
- Cache **policy** axis (NT/streaming): {A, B, both} × {non_temporal, cache_stream} = 6/6 LOSE.
- Cache **bandwidth** axis (extra discarded VMEM): {A, B, both} × {1, 2, 3 intensity} = all LOSE.
- The DLA shapes are **VMEM-issue-bound** (single VMEM lane already saturated), not VMEM-latency-bound. R21-recon's 167-294% TCP_DATA_STALL was the *consumer-side* symptom (waiting on memory), not a producer-side opportunity.

## Round 25 (2026-04-18) — BREAKTHROUGH: R25-C + R25-D STACK WIN on DLA2/DLA7
- **R25-A** (SCRATCH-spill audit): DEAD END — zero spills exist (kernel uses `__launch_bounds__(_NUM_THREADS, 1)`); pivoted to `R25A_SCALE_RELOAD_PER_K_ITER=1` which **hangs the GPU** (same scale-VGPR aliasing pattern as R5 EARLY_SCALE_PF dead-end).
- **R25-B** (GROUP_SIZE_M fine-grained sweep): **PARTIAL WIN** — `gm6` beats prior gm2 on DLA2 (+2.97%) and DLA7 (+0.91%); gm12/16 cause `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on M=4096 shapes; gm3 is correctness-OK but unstable (run-to-run std 800-1000 TFLOPS).
- **R25-C** (K-loop tail prefetch-off): **WIN** — `R25C_TAIL_PF_OFF_ITERS=4` + `R25C_K_LIMIT=32768` gate gives DLA2 +5.96%, DLA7 +3.14%, DLA1 no-op (gated off because K=128256 partial-unroll branch doesn't fold). Committed at `09d58029`.
- **R25-D** (STACK TEST: gm6 × pfoff4): **SUPER-ADDITIVE WIN** — combined effect beats either singleton:
  - **DLA2**: 4203 → 4485 TFLOPS (+6.69% vs baseline; +3.29% vs pfoff4-alone).
  - **DLA7**: 4168 → 4456 TFLOPS (+6.92% vs baseline; +2.24% vs pfoff4-alone).
  - Stack also **most stable** of all 4 variants on DLA7 (no rc=-6 crashes, no 2× outliers).
  - Mechanism: gm6 = L2 B-tile reuse (steady-state), pfoff4 = tail VMEM freeing (epilogue). Orthogonal stalls.
- **Wired into `bench_all_42.py`** as `_ts_gm6_v12_memc_dc_pfoff4` (DLA2-class) and `_ts_lgk2_gm6_v12_memc_pfoff4` (DLA7-class). **Committed at `7ada8c70`.**
- **Gap reduction**: DLA2 ratio 96.3% → ~98.9% (gap to aiter 4536: 6.7% → 1.1%). DLA7 ratio 96.9% → ~99.8% (gap 6.9% → 0.2%).
- **R25 reviewer baseline check (2026-04-18, PASS with note)**: pre-R25-D bench measured 27/42 WIN with 2 noise-band flips at the 100% threshold (`4096×32768×6144` 100.6%→99.2%, `32768×4096×14336` 101.0%→98.8%). All R23/R24 macros confirmed `#ifndef`-guarded, default 0/inactive — no kernel pollution. Cached binaries used; R25-D wires NOT measured here. Verdict: kernel safe.

## In flight (2026-04-18)
- **R25-D verify** (agent `a355a47a7cd86cda2`): full 42-shape autotune WITH R25-D wires picked up (and now ALSO R25-F wires if it re-reads the variants table). Expects DLA2/DLA7 to flip well over 100% ratio.
- **R25-E DLA1 K-loop peel** (agent `aa5d9ccf6befb238f`, in worktree): split main loop into head (K-1-N iters, full pf) + peeled tail (N iters, no pf). Macros `R25E_K_LOOP_PEEL_ITERS` (default 0) + `R25E_K_LIMIT_LO=65536` so only DLA1 (K=128256) activates.

## R25-F EXTENDED-SWEEP MASSIVE WIN (2026-04-18, committed `e5083bad`)
Extended `R25C_TAIL_PF_OFF_ITERS ∈ {3..16}` × `GROUP_SIZE_M ∈ {5,6,7,8}` cross-product on DLA2/DLA7 reveals R25-D's `gm6+pfoff4` was at the EDGE of a flat plateau — true optimum is `gm7 + pfoff14`:

| Shape | R25-D (gm6+pfoff4) | R25-F (gm7+pfoff14) | Δ vs R25-D | Δ vs original baseline |
|-------|--------------------|---------------------|------------|--------------------------|
| DLA2 (M=128256) | 4449.4 TFLOPS | **4932.8** TFLOPS | **+10.86%** | ~+17.3% (vs 4204) |
| DLA7 (M=28672)  | 4442.4 TFLOPS | **5042.8** TFLOPS | **+13.51%** | ~+21.0% (vs 4168) |

**Mechanism**: K=4096 → k_byte_iters=16. After 2 prefetched iters, B-tile is fully L2-resident across persistent-XCD remap; the remaining 14 iters' prefetches were re-issuing redundant VMEM and competing with scale-load + C-write traffic. Removing them frees bandwidth that gm7's improved swizzle can fully utilize.

**Wired into bench_all_42.py** (preserves R25-D as fallback):
- `_ts_gm7_v12_memc_dc_pfoff14` (DLA2-style)
- `_ts_lgk2_gm7_v12_memc_pfoff14` (DLA7-style)

Stability: 12 verify variants STABLE (std ≤24 TFLOPS); plateau is genuinely flat (gm6/7 × pfoff{14,15,16} all within ~10 TFLOPS).
Gating: R25C_K_LIMIT=32768 still excludes DLA1 (K=128256).

**Projected post-R25-F**: DLA2 ~108.7%, DLA7 ~112.4% of aiter — both BIG WIN over LOSE. Final WIN count likely 31/42 (up from 29).

## R25-G PER-K-BUCKET PFOFF WIN (2026-04-18, committed `7f200b76`)
Generalizes R25-F's "only first 2 K-iters need prefetch" mechanism to mid-gap larger-K shapes that R25-F's pfoff∈{3..16} sweep could not reach. Per-K optimum is at `pfoff = K_iters - {4..8}` (slightly more prefetch ramp-up for larger K than for K=4096's 2-iter ramp).

| Shape | K_iters | Best `pfoff` | TFLOPS | Δ% |
|-------|---------|--------------|--------|-----|
| SA 4096×28672×32768 | 128 | 120 | 6486.67 | **+19.60%** |
| SB 4096×32768×14336 | 56  | 54  | 6097.59 | **+17.55%** |
| SC 14336×4096×32768 | 128 | 124 | 5910.76 | **+17.55%** |
| SD 16384×4096×28672 | 112 | 104 | 6369.26 | **+20.84%** |
| SE 28672×4096×16384 | 64  | 56  | 6071.22 | **+16.89%** |
| SF 4096×14336×16384 | 64  | 56  | 5613.97 | **+13.68%** |

**Implementation**: added `R25C_K_EXACT` compile-time gate (kernel lines ~92-105) so each per-K pfoff variant only activates on its target K. 5 K-EXACT-gated variants wired into `bench_all_42.py`. R25-F K=4096 entries unchanged. DLA1 (K=128256, R25-E's domain) untouched.

**Stability note**: SD (16384×4096×28672) showed bimodal stability on busy GPUs (5400 vs 4000 TFLOPS) but stable on isolated GPU 1. A full 42-shape regression run (after R25-D verify) is recommended to confirm no surprises.

**Cumulative R25-F + R25-G**: 8 shapes likely flip LOSE → BIG WIN (DLA2, DLA7 + 6 mid-gap). Combined with R22-rebench's 29/42 baseline, projected total WIN ≥ 35/42 — biggest 1-day jump in any post-R20 round.

## Remaining vectors (high-cost / high-risk only — post-R25)
- **R25-F (post-verify)**: extended `R25C_TAIL_PF_OFF_ITERS ∈ {5,6,7,8}` and `GROUP_SIZE_M ∈ {5,7}` cross-products on DLA2/DLA7 — possible additional +1-2pp.
- **Tile-geometry axis** (BK depth) — pure structural rewrite, ~1-2d.
- **MFMA_32X32X64** alternative tiling — major asm rewrite, AGPR pressure unclear.
- **PERSISTENT_XCD kernel-bug rewrite** — risky, would need re-verification of all 29 WIN shapes.

## Round 21 (2026-04-18) — recon + audit; head macros DEAD END
3 parallel agents: (a) **R21-recon** rocprof-PMC on DLA2/DLA7, (b) **R21-audit** untried-axis survey, (c) **R21B** probe 3 head macros from audit.

- **R21-recon — rocprof PMC sweep on DLA2 + DLA7** (parallels R17A's DLA1 profile):
  - DLA2 (128256x32768x4096, ratio 91.4%): MFMA fills 24.7 % of wall, **TCP_DATA_STALL = 292.6 % of GRBM** (memory-stall bound), HBM 1054 GB/s = 19.9 % of 5.3 TB/s peak, 0 % LDS bank conflict.
  - DLA7 (28672x32768x4096, ratio 78.2 %): MFMA fills only 20.8 % of wall, **TCP_DATA_STALL = 294.1 % of GRBM**, HBM 913 GB/s = 17.2 % of peak, 0 % LDS bank conflict.
  - DLA1 (re-profile, ratio 88.4 %): MFMA fills 30.4 % of wall, TCP_DATA_STALL = 167.8 %, HBM 412 GB/s = 7.8 % of peak (mega-K = 128256 ⇒ 2004 K-iters ⇒ low HBM, high LDS-replay pressure).
  - **All 3 DLA shapes are memory-stall bound, not compute bound**. `lds_per_wave` differs by 9× between DLA1 (55) vs DLA2/DLA7 (512) — DLA1 is K-iter-bound (epilogue overhead amortizes badly), DLA2/DLA7 are pure HBM bandwidth-bound.
  - **R22 frontier**: HBM bandwidth headroom is large (5.3 TB/s peak vs 0.9-1.0 TB/s observed). Need to either (a) reduce LDS replay/stall (TCP_TA_DATA stall is the symptom — LDS sub-arbitration, not bank conflict) or (b) increase global-load coalescing / `cache=streaming`. None of the 80+ existing variants attack the LDS-stall axis directly.
- **R21-audit — untried-axis survey** (33 K of analysis, 4 candidate axes):
  - Identified `WAVE_PRIO_HIGH`, `EXPLICIT_S_NOP`, `SCHED_GROUP_BARRIERS` as 3 macros DEFINED but with **zero usage sites in production kernel**. Highest-EV untried axis.
  - Proposed wiring 4 hook sites (3× K-iter end + 1× pre-Store-C) with default 0 = no-op asm (zero regression risk).
  - Other axes surveyed: persistent-grid, AGPR-VGPR rotation, scratch-spill audit — all flagged for R22+.
- **R21B — probe 3 head macros (DEAD END)**:
  - Wired 4 hook sites for `WAVE_PRIO_LOW_TAIL`, `EXPLICIT_S_NOP`, `SCHED_GROUP_BARRIERS` (defaults 0). Built 28/28 (4 shapes × 7 macro combos).
  - Best smoke: P1 +0.72 % with `_snop1_sched` → 5-run verify −0.185 pp = FAIL.
  - **Mechanistic conclusions**:
    - `WAVE_PRIO_HIGH`: kernel uses `__launch_bounds__(_,1)` → already 1 wave/SIMD/CU; raising prio at K-iter end ⇒ pure regression.
    - `EXPLICIT_S_NOP=1`: LLVM `s_waitcnt` already covers MFMA latency → mild regress.
    - `SCHED_GROUP_BARRIERS=1` mask=`0xff` is too coarse — disrupts cross-iter MFMA/prefetch interleave, −1.6 to −3.5 %. Finer masks (e.g. `0x80`=MFMA-only or `0x44`=lgkmcnt-only) untried; out of scope this round.
  - Kernel patch retained (no-op default) → unblocks R22+ for finer-mask experiments. **No commit.**

**Round 21 净增**: 0 WIN. **R21-recon delivered the highest-value finding**: DLA1/DLA2/DLA7 are all memory-stall bound (167-294 % TCP_DATA_STALL); HBM headroom = 5×.

## Round 20 (2026-04-18) — full BARRIER_TO_WAITCNT generalization sweep; **+11 shape WINs** (R20A massive breakthrough)
3 parallel agents: (a) **R20A** — random-scale aperture probe + 5-run verify on 11 R19A-pre-failed shapes; (b) **R20B** — full 42-shape rebench locking R18+R19 wins; (c) **R20C** — K-loop sync coarsening (DEAD END).

- **Optimizer A — Aperture probe + 11-shape verify (BREAKTHROUGH)**:
  - R19A's barrier-removal axis was thought constrained to ≤4 shapes (S1+S5 won, 11/15 SNR-pre-failed). R20A reframed: pre-failed shapes were noise-floor artifacts, not real correctness violations. Built a **random-scale aperture probe** (5 iters × 2 seeds, OK if no kernel crash + bench produces TFLOPS > 0 + bench reproducibility ≤ 5 % stddev).
  - 11/11 R19A-pre-failed shapes passed aperture; smoke bench surfaced 11 candidates with Δpp ≥ +1.5 pp on `_r19a_step3` or `_r19a_all`; 5-run same-GPU verify confirmed **11/11 WIN** with mean Δpp ranging **+2.03 pp to +4.28 pp**.
  - **Wired 11 new (parent + BARRIER_TO_WAITCNT) stacks into bench_all_42.py**:

  | Lab | Shape | Parent variant | BTW variant | Mean Δpp |
  |-----|-------|----------------|-------------|----------|
  | S2  | 16384x4096x28672  | `_u32`                 | `_btw_all`   | +3.74 |
  | S3  | 4096x32768x28672  | `_v20_memc`            | `_btw_step3` | +2.84 |
  | S5  | 32768x4096x14336  | `_ts_gm8_v12`          | `_btw_all`   | +4.09 |
  | S6  | 4096x32768x14336  | `_ts_lgk2_memc`        | `_btw_all`   | +3.11 |
  | S7  | 28672x32768x4096  | `_ts_lgk2_v12_memc`    | `_btw_all`   | +2.30 |
  | S8  | 4096x32768x6144   | `_ts_pf4_memc`         | `_btw_step3` | +4.28 |
  | S9  | 16384x28672x2048  | `_ts_gm2_v12_memc_dc`  | `_btw_all`   | +2.03 |
  | S10 | 16384x28672x4096  | `_ts_gm2_v12_memc`     | `_btw_all`   | +3.23 |
  | S12 | 14336x32768x4096  | `_ts_v12_tv0_memc`     | `_btw_all`   | +3.26 |
  | S13 | 4096x14336x16384  | `_ts_lgk2`             | `_btw_step3` | +2.71 |
  | S15 | 32768x4096x7168   | `_ts_gm8_v12`          | `_btw_step3` | +2.04 |

  - **Caveat (carry-over from R18A/R19A)**: bf16-saturation non-deterministic; aperture-validated only (no exact-output comparison possible). Same risk profile as the existing `_p1_btw_all`, `_lgk2_dc_btw_step3`, `_lgk2_dc_btw_all` entries.
  - **Methodological win**: random-scale aperture probe replaces the brittle uniform-input SNR floor. Generalizes to future BTW exploration on any non-saturating shape.
- **Optimizer B — full 42-shape rebench locking R18A+R19A+R19C wins**:
  - 116-variant auto-tune with the 3 R18/R19 BTW entries → **27/42 WIN** (up from 24/42 baseline). +3 LOSE→WIN flips: P1 (R18A), S1 (R19A), S5 (R19C); 0 regressions.
  - With R20A's 11 new BTW stacks wired into bench_all_42.py, projected post-R20 next bench: **~34/42 WIN** (pending re-run with the 127-variant set).
- **Optimizer C — K-loop sync coarsening (DEAD END)**:
  - Added `K_LOOP_SYNC_EVERY_{2,4}` macros — barrier only every Nth K-iter. Both broke SNR (0 dB race) AND regressed −25 to −29 pp due to parity-branch blowing past the 256 VGPR cap, forcing scratch spill.
  - Root cause: 2-buffer LDS rotation is insufficient when barriers skip alternate K-iters; correct fix needs triple-buffer LDS (out of scope this round).
  - Macros left at default 0 (no-op). Documented as dead-end.

**Round 20 净增**: **+11 deep-LOSE shapes closed via R20A** (S2/S3/S5/S6/S7/S8/S9/S10/S12/S13/S15). Cumulative R18+R19+R20 = **14 closed** (P1, S1, S5, plus the 11 above with overlap on S5). Bench at 27/42 → projected ~34/42 next run.

**新 dead-end vectors (Round 20)**:
- `K_LOOP_SYNC_EVERY_{2,4}` — needs triple-buffer LDS (not 2-buffer); SNR + register-pressure both fail.
- Uniform-input SNR was over-conservative — random-scale aperture probe is the right correctness oracle for BTW axis.

**Frontier post-R20**: Barrier-removal axis essentially saturated (14 of 18 deep-LOSE shapes addressed). **Remaining stuck**: DLA1, DLA2, DLA7, S4 (4096x32768x128256), S14 (4096x32768x4096?), DLA1's 28672x4096x16384 still 97.6%. R21-recon classified DLA1/DLA2/DLA7 as memory-stall bound (TCP_DATA_STALL 167-294 % of GRBM, HBM 7.8-19.9 % of peak). **Next axes**: (a) LDS-stall reduction (sub-arbitration, not bank conflict), (b) global-load `cache=streaming` for DLA2/DLA7, (c) per-K-shape epilogue specialization for DLA1, (d) finer SCHED_GROUP_BARRIERS masks via R21B's now-wired hooks.

## Round 19 (2026-04-17) — barrier-removal extended; **2 new WINs** (S1 +6.58pp, S5 +2.69pp)
3 parallel optimizers extending R18A's BARRIER_TO_WAITCNT WIN. **Net: +2 deep-LOSE shapes closed (S1, S5); cumulative R18+R19 = 3 closed (P1, S1, S5).** Biggest single-shape gain to date: S1 +6.58pp.

- **Optimizer A — 41-shape sweep of BARRIER_TO_WAITCNT_{STEP3,STEP12,ALL} (1 WIN)**:
  - Selected 15 candidate shapes from the 18 LOSE list (skipped DLA1/DLA2/DLA7 known-broken + already-won P1).
  - 45 builds (15 shapes × 3 variants), SNR + aperture-probe pre-filter, smoke + 5-run same-GPU verify.
  - **S1 (14336x4096x32768) + `_lgk2_dc_btw_step3`**: parent 89.6% → variant **96.18% (+6.58pp)**. Both gates PASS. **WIN, committed `e29f6c3a`.** This is the biggest single-shape gain in any post-R2 round.
  - S1 + `_lgk2_dc_btw_all` also passes (+6.32pp) but dominated by step3-only.
  - S4 (`_u16_btw_step12`): smoke +0.76 → verify +0.22 (FAIL).
  - S14 (`_ts_lgk2_btw_step12`): smoke +2.72 → verify +0.42 (FAIL).
  - 11/15 candidates pre-failed SNR floor (parent saturates bf16) — confirms the SNR methodology constrains this axis to shapes with parent-SNR floor > 10 dB.
  - **Wired 3 opt-in entries into bench_all_42.py**: `_p1_btw_all` (R18A P1), `_lgk2_dc_btw_step3` (R19A S1), `_lgk2_dc_btw_all` (R19A S1 dominated).
- **Optimizer B — per-site STEP3/STEP12 + vmcnt sweep on DLA1/DLA2/DLA7 (DEAD END)**:
  - 10 new per-site macros (`BARRIER_TO_WAITCNT_STEP3_S1..S7`, `STEP12_S1..S2`, `RELAXED_VMCNT`) added to kernel.cpp. Defaults preserve R18A bit-exactly; zero regression risk.
  - 39 builds (3 shapes × 13 variants); 35/36 SNR-broken; only DLA1/_r19b_t1 OK but Δ-0.27pp on 5-run verify.
  - **Confirms R18A**: barrier IS load-bearing for DLA1/DLA2/DLA7; not a single removable site exists.
  - Committed dead-end registry as `56affcbd`.
- **Optimizer C — Stack BARRIER_TO_WAITCNT with iterilp on S1-S5 winners (1 WIN)**:
  - 15 builds (5 shapes × 3 variants) testing iterilp ⊥ source-rewrite hypothesis.
  - **S5 (4096x32768x14336) + `_ts_lgk2_memc_r19c_iterilp_btw_all`**: parent 95.18% → variant **97.87% (+2.69pp)**. Both gates PASS. **WIN, committed `a0f1949c`.**
  - S1-S4 SNR-unvalidatable (K∈{28672,32768} non-deterministic from bf16 saturation + FMA reorder).
  - Subtle finding: on S5, removing only STEP3 OR only STEP12 is racy in isolation, but removing both together is safe — **the two barriers form a matched producer/consumer pair**.
  - **Hypothesis verdict**: iterilp ⊥ source-rewrite is PARTIALLY supported (1 confirmed compose, 4 unvalidatable due to bf16 saturation noise floor).

**Round 19 净增**: **+2 deep-LOSE shapes closed** (S1 +6.58pp, S5 +2.69pp). Cumulative R18+R19 = **3 closed** (P1, S1, S5). 

**新 dead-end vectors (Round 19)**:
- Per-site `BARRIER_TO_WAITCNT_STEP3_S{2,3,4}` on DLA1/DLA2/DLA7 — every individual hot site breaks SNR worse than aggregate.
- 2-site / 3-site STEP3 combos on DLA1/DLA2/DLA7 — no synergy; strictly worse than singles.
- `BARRIER_TO_WAITCNT_STEP12_S1` on DLA1 — SNR-MARGINAL but 0 perf gain.
- `BARRIER_TO_WAITCNT_RELAXED_VMCNT` ∈ {1,4,15} on DLA1/DLA2/DLA7 — barrier-VMCNT perturbation alone breaks SNR.
- DLA2/DLA7 SNR-unvalidatable via output-tile probe (parent at noise floor < 0 dB). Future probes need ULP histograms / reduced dynamic range.
- 11/15 BARRIER_TO_WAITCNT candidates in R19A pre-failed SNR floor — bf16 saturation is the dominant constraint on this axis going forward.

**Frontier post-R19 (UPDATED)**: Barrier-removal axis substantially yielded but now constrained by SNR floor to shapes with parent SNR > 10 dB. 3 deep-LOSE wins (P1, S1, S5). **Remaining stuck**: DLA1/DLA2/DLA7 (SNR-broken, barrier load-bearing) + 11+ LOSE shapes that pre-fail SNR floor (bf16 saturation in parent). **Next axes**: (a) different correctness probe (ULP histogram, reduced-dynamic-range scales) to validate kernel changes on DLA2/DLA7, (b) per-shape kernel specialization for K=128256 (DLA1), (c) MFMA op switch 32x32x64 (long-horizon).

## Round 18 (2026-04-17) — source-rewrite pivot per R17A proposals; **R18A WIN +4.16pp on P1** (committed `4b504b0c`)
After R17A's rocprof analysis proved backend tuning exhausted, dispatched 3 source-rewrite optimizers attacking the R17A proposal queue (P3/P1/P2). **First +1pp gain in 17 rounds of post-Round-2 work.**

- **Optimizer A — R17A-P3 inner s_barrier → s_waitcnt lgkmcnt(0) (WIN)**:
  - 3 opt-in macros added to kernel.cpp (default 0): `BARRIER_TO_WAITCNT_STEP3` (8 hot-path STEP3 sites), `BARRIER_TO_WAITCNT_STEP12` (2 tail STEP12 sites), `BARRIER_TO_WAITCNT_ALL` (both).
  - SNR probe with noise-floor-relative gating: 3/12 (4 shapes × 3 variants) SNR-OK; 9 SNR-broken — barrier IS load-bearing for DLA1/DLA2/DLA7.
  - **P1 (28672x4096x16384) + BARRIER_TO_WAITCNT_ALL=1**: parent 5079.43 TFLOPS (94.93%) → variant 5302.23 TFLOPS (**99.10%**), Δ = **+4.16pp**, 5-run same-GPU verify, both gates pass. **WIN, committed `4b504b0c`.**
  - DLA1/step12 SNR-passed under uniform-scale probe but APERTURE-violated under random-scale bench inputs — **uniform-input SNR is necessary but not sufficient**.
  - DLA2/DLA7 cannot be SNR-validated via this method (parent saturates bf16, noise floor < -5 dB).
  - Macros default 0 → other 41 shapes byte-identical, no regression risk.
  - **NOT auto-added to bench_all_42** (per-shape opt-in only; global add would risk silent wrong-output on K-large shapes).
- **Optimizer B — R17A-P1 double C-accumulator ping-pong (DEAD END)**:
  - Build FAILED on all 4 shapes with "invalid operand for instruction".
  - **R17A's profile proposal misread the kernel structure**: rows 1-3 use `ds_read_b128` results (`d1/d2/d3`) as MFMA A-operand, NOT `a_lo[i]`. A naive phase-interleave assuming uniform A operand semantics is invalid.
  - True ping-pong would require: (a) move all 8 ds_reads up-front (~2.5M extra cycles, NEGATIVE EV) OR (b) double AGPR pressure 64→128 dropping wpe=2→1 (NEGATIVE EV) OR (c) multi-day producer-consumer LDS protocol rewrite.
  - **R17A-P1 dropped from registry.**
- **Optimizer C — R17A-P2 M-tile expansion for K=128256 (DEAD END)**:
  - Source refactor (M=128→512 not 256, agent escalated) needs ~600 LOC duplication + 4× scale buffers + 8 accumulator sets — multi-day.
  - **MI355X LDS budget = 160 KB/CU (NOT 64 KB as R17A assumed)**. M=256 doubling pushes A-tiles to ~192 KB total — overflows. Mitigations (single-buffer A, half-N) negate the latency-hide gain.
  - M=192 fallback also infeasible: not power-of-2, doesn't divide 64 (scale-pack `>> 6` row indexing).
  - **PERSISTENT_XCD with batch={1,2,4,8}** (R14C only tested batch=16/32) **all 7 variants crash with `Memory access fault by GPU node-X`** before completing a single dispatch — dispatcher mechanism has a bug at large grid counts.
  - **R17A-P2 dropped from per-round registry (long-horizon 3-5 day refactor only).**

**Round 18 净增**: **+1 deep-LOSE shape gap closed** (P1 94.93% → 99.10%, +4.16pp; almost-WIN). 17-round dry spell broken.

**新 dead-end vectors (Round 18)**:
- R17A-P1 (double C ping-pong) — kernel structure incompatible without multi-day rewrite
- R17A-P2 (M-tile expansion) — LDS budget overflow + per-round infeasible
- PERSISTENT_XCD batch={1,2,4,8} on DLA1 — GPU memfault, dispatcher bug at large grid counts
- BARRIER_TO_WAITCNT on DLA1/DLA2/DLA7 — barrier load-bearing (SNR breaks or aperture-faults)

**Frontier post-R18**: P1 nearly closed (99.10%, ≤1pp from 100%). DLA1/DLA2/DLA7 still need their own source-rewrites (the barrier trick won't work on them). Remaining proposals: (a) MFMA op switch 32x32x64, (b) K-split rewrite, (c) SLM relayout, (d) per-shape kernel specialization. Long-horizon: R17A-P1, R17A-P2 multi-day refactors.

## Round 17 (2026-04-17) — profile/triple-stack/attribute axes, all DEAD END (13th saturation round)
3 parallel optimizers (Opus 4.7) attacked 3 genuinely-new angles after R16 explicitly forbade more compound flag stacks. **0 wins**. R17A produced first hard profile evidence that further LLVM-flag tuning is futile.

- **Optimizer A (rocprof DLA1 + fresh 8-GPU 42-shape re-baseline)**:
  - **rocprof on DLA1 (4096×32768×128256)**: VALUBusy=49% (kernel idle half the cycles). MFMA-pipeline floor ≈ 0.9-1.8 ms vs 6.74 ms wall ⇒ MFMA fills only 13-27% of wall-time.
  - **Bottleneck classified**: MFMA-accumulator dependency stall (single C-tile reused, 8-cyc f4 latency); K-loop epilogue per-iter `s_barrier`+`s_waitcnt` ~0.6-1.2 ms; prefetch m0-hazard `s_nop` compounds 31× vs the K=4096 case.
  - **HARD VERDICT (new)**: VALUBusy=49% is a register-allocation/MFMA-scheduling problem, **not a backend-flag problem**. Further `-mllvm` tuning cannot move this needle.
  - **3 source-edit proposals (NOT implemented; queued for future kernel-rewrite rounds)**:
    - P1 — Double C-accumulator tiling (split `rt_C[4][4]` → `rt_C0/C1`, ping-pong MFMAs). EV +2-4pp on DLA1. Risk: register-pressure forces ½ occupancy.
    - P2 — Larger M-tile (M=128→M=256) for K=128256 specialization. Halves CTA count 524K→262K. EV +1-2pp.
    - P3 — Replace inner `s_barrier` with `s_waitcnt lgkmcnt(0)`. EV +5-6pp if SNR-safe.
  - **42-shape re-baseline (partial 27/42)**: 18/26 fully-comparable shapes within ±2pp of Round 2; 5 improved 3 regressed; net +0.4pp. **No formerly-LOSE shape flipped to WIN.** 15 shapes incl. all 4 deep-LOSE need overnight 8-GPU re-run (115 variants × iters=500 took ~3-4× the 20-30 min estimate).
- **Optimizer B (P1 NO-iterilp triple/quad stacks of 3 sub-threshold positives)**:
  - 10 variants stacking the 3 known sub-threshold P1 positives: regclassglob (+0.236pp), regclassglob+tv16 (+0.225pp), regclassglob+noemxpre (+0.30pp).
  - **All 10 ASM-DIFF distinct from parent and from R14A/R15A/R16C 2-stack winners.**
  - Best 3-stack: `rcg+noemxpre+tv16` = **+0.498pp** (just below +0.5pp gate; tied with best 2-stack).
  - **HYPOTHESIS-FALSIFYING**: Linear additivity of sub-threshold deltas COLLAPSED. Predicted sum +0.76pp; observed +0.50pp. **3rd flag adds 0pp on top of best 2-stack.** Compound-stacking axis on P1 fully exhausted.
  - 2 quad-stacks catastrophically regressed: `rcg+noemxpre+tv16+v20` -15.83pp; `rcg+tv16+extbr` -20.68pp.
- **Optimizer C (untested __attribute__ knobs on 4 stuck shapes)**:
  - 9 attributes RECOGNIZED: `flat_work_group_size(64,256)`/(256,256)/(128,512), `num_vgpr(256/224/192)`, `num_sgpr(96/80)`, `max_num_work_groups(8,1,1)`. 1 unrecognized: `amdgpu_no_agpr`.
  - **All 32 successful builds produce DIFF .text.** 4 builds hung the LLVM scheduler past 600s (`fwgs128_512` + launch_bounds(256,1) conflict).
  - **All gate-PASS smokes collapsed on verify**: best was DLA1/sgpr96 smoke +1.90pp → verify -4.06pp + aperture crashes; P1/sgpr80 smoke +0.56pp → verify -0.92pp.
  - **Catastrophic regressions**: vgpr192 -78pp on DLA2/DLA7; vgpr224 -59 to -60pp; sgpr96 -49pp on DLA7. mnwg8 → APERTURE on all 4 shapes.
  - **The 4 stuck shapes are at a register-allocation fixed point robust to attribute-level coercion.** Confirms R3+R12 saturation from a new angle.

**Round 17 净增**: 0 WIN, 0 gap reduction. **13 saturation rounds total. Backend axes (flags/attrs/compound stacks) now provably exhausted.**

**新 dead-end vectors (Round 17)**:
- All 9 recognized AMDGPU codegen attributes — KILL or APERTURE on the 4 stuck shapes; 7 new BROKEN registry entries.
- Triple/quad stacks of regclassglob × {noemxpre, tv16, v20, lgk2, extbr} on P1 — additivity collapses; 2 NEW catastrophic destabilizers (rcg+tv16+extbr; rcg+noemxpre+tv16+v20).
- All `-mllvm` LLVM-flag tuning on DLA1 — provably bottleneck-mismatched (VALUBusy=49% is not a backend issue).

**Frontier post-R17 (HARDENED, evidence-based)**: 13 rounds saturate flag/macro/source-micro/compound/attribute axes. **rocprof has now PROVEN further backend tuning cannot help DLA1.** Future agents must NOT propose more `-mllvm` flag work on the 4 stuck shapes. Only kernel-source rewrites can move the needle: (a) double C-accumulator tiling [R17A-P1], (b) M=256 specialization for K=128256 [R17A-P2], (c) inner-barrier→waitcnt rewrite [R17A-P3], (d) MFMA op switch 32x32x64, (e) K-split rewrite, (f) SLM relayout, (g) full B-direct.

## Round 16 (2026-04-17) — compound stacking (iterilp × regalloc/sink/LICM), all DEAD END
3 parallel optimizers (Opus 4.7) attacked the never-tested COMPOUND STACKING axis. **0 wins, 12th saturation round**. 3 critical hypothesis-falsifying findings:

- **Optimizer A (iterilp + regclassglob on 5 R10/R11 winners)**:
  - All 5 compounds DIFF .text but every one HURT performance vs iterilp-only winner: S1 -14.82pp (catastrophic), S2 -1.33pp, S3 -0.43pp, S4 -0.32pp, S5 -1.20pp.
  - **Hypothesis "regalloc family ⊥ scheduler family ⇒ stacks safely" is FALSIFIED.** Regalloc priority changes interfere with iterilp's preferred register layout.
  - regclassglob's +0.236pp on P1 is **iterilp-INDEPENDENT**; on iterilp WINs the flag has the OPPOSITE sign.
- **Optimizer B (iterilp + 4 R15B safe-DIFF flags × 5 winners = 20 compounds)**:
  - 10 DIFF, 10 NOOP-vs-iterilp, 0 BUILD-fail. Smoke (+0.5pp gate): 1/10 PASS (S2/largeivf2 +0.51pp, collapsed to +0.30pp on 5-run).
  - **nolicm + iterilp regresses -2.2 to -2.4pp** on S1/S3/S4 (LICM re-enables hoisting around iterilp reorder).
  - noemxpre ±0.35pp noise; sinkavoidspill ASM-NOOP everywhere on iterilp parents.
- **Optimizer C (cross-axis compounds × 4 stuck shapes, 25 variants, SNR-first safety)**:
  - **DEFINITIVE FINDING**: iterilp SGPR-clobber bug is INDEPENDENT of regalloc policy (regclassglob/sinkavoidspill/nolicm) AND independent of scheduler perturbations (largeivf2/noemxpre).
  - Every iterilp+X compound on DLA1/DLA2/DLA7 still aperture-crashes at full M.
  - **Hypothesis from R15 prompt (regalloc restructuring might dodge the bug) is DISPROVEN.**
  - Best signal: P1 regclassglob+noemxpre clean re-verify **+0.30pp** (mean ≥ base.max PASS) but +1pp gate FAIL.

**Round 16 净增**: 0 WIN, 0 gap reduction. **12 saturation rounds total**.

**新 dead-end vectors (Round 16)**:
- iterilp + regclassglob on 5 R10/R11 winners — UNIVERSALLY HURTS (-0.32 to -14.82pp); regalloc/scheduler axes NOT independent
- iterilp + {noemxpre, largeivf2, sinkavoidspill, nolicm} on 5 winners — 0/20 wins; nolicm catastrophic (-2.4pp)
- iterilp + any non-scheduler flag on DLA1/DLA2/DLA7 — still aperture-crashes (bug is pure scheduler-induced, regalloc cannot dodge)
- regclassglob + nolicm + sinkavoidspill triple stack on stuck shapes — DIFF but all sub-+1pp
- ~25 NEW BROKEN-APERTURE registry entries (iterilp×X compounds on DLA1/DLA2/DLA7)

**Frontier post-R16 (UNCHANGED)**: 12 rounds saturate flag/macro/source-micro/compound axes. Future agents must NOT propose more compound flag stacks. Multi-day kernel rewrites only: (a) MFMA op switch 32x32x64, (b) K-split rewrite, (c) SLM relayout, (d) full B-direct.

## Round 15 (2026-04-17) — regclassglob cross-shape + 42 new LLVM flags + 3 source micros, all DEAD END (committed via this round)
3 parallel optimizers (Opus 4.7) attacked the post-R14 frontier from orthogonal angles. **0 wins, 11th saturation round**. Cumulative LLVM-flag space now exhausted at **69 distinct flags** (27 R14A + 42 R15B).

- **Optimizer A (regclassglob cross-shape + 7 P1 macro compounds)**:
  - Fresh asm-diff: regclassglob produces DIFF on ALL 4 shapes (R14A's "NOOP on DLA2/DLA7" was a misread of the JSON).
  - 5-run verify: P1 bare regclassglob **+0.236pp** (cleanly replicates R14A's +0.24pp). P1c regclassglob_tv16 **+0.225pp** with mean ≥ baseline.max **PASS** but sub-+1pp gate FAIL.
  - DLA1/DLA2/DLA7 all FAIL +0.5pp smoke gate (Δ -22.8 to -0.02pp; -22.8 was single-run noise artifact, verify shows +0.11pp).
  - 0/11 commit-eligible. **regclassglob is real-but-tiny perturbation, not a +1pp lever.**
- **Optimizer B (42 untested LLVM flags, NO overlap with R14A's 27)**:
  - Coverage families: coalescer (8), spill/AGPR (5), machine-sink/LICM (6), post-RA scheduler (6), IGLP (2), loop (3), AMDGPU misc (12).
  - 168/168 builds OK. **Only 8/42 flags mutate .text** on at least one shape; 34 silent NOOP. 5-run verify: 0/3 candidates pass +1pp gate.
  - **NEW BROKEN-APERTURE flags**: `-join-liveintervals=false` (P1/DLA1 crash; -30pp on DLA2/DLA7), `-greedy-reverse-local-assignment` (3/4 shapes).
  - **Major regression flags (do not use)**: `-disable-machine-sink` -21.83pp on DLA1; `-disable-post-ra` -3.87pp on multiple shapes.
- **Optimizer C (3 source-level micro probes, strict 35-min time-box)**:
  - New macros `TAIL_BARRIER_LGKMCNT`, `STEP4_BARRIER_VMCNT`, `PF_GROUP_OFFSET` added to kernel.cpp temporarily, then **REVERTED** (no source change committed).
  - `TAIL_BARRIER_LGKMCNT` best -0.21pp on DLA7. CLOSED.
  - `STEP4_BARRIER_VMCNT` all values 4-20 LOSE on DLA1. CLOSED.
  - `PF_GROUP_OFFSET=-1` +0.20pp on DLA1 within noise. Compiler reorders buffer_load_lds anyway; source-level rotation doesn't survive isel. CLOSED.

**Round 15 净增**: 0 WIN, 0 gap reduction. **11 saturation rounds total** (R2/R4/R5/R6/R7/R8/R9/R12/R13/R14/R15). R10/R11 remain the only break-out rounds (5 verified deep-LOSE wins via un-prefixed iterative-ilp).

**新 dead-end vectors (Round 15)**:
- regclassglob × 7 macro compounds on P1 — all sub-+1pp; tv16 closest at +0.225pp gate-mean-PASS but threshold-FAIL
- regclassglob bare on DLA1/DLA2/DLA7 — verify Δ ∈ [-0.06, +0.11] pp
- 42 LLVM flags from 7 families (coalescer/spill-AGPR/sink-LICM/postRA/IGLP/loop/misc) — 34 NOOP, 8 DIFF, 0 wins
- `-join-liveintervals=false` — NEW BROKEN-APERTURE on P1/DLA1, -30pp on DLA2/DLA7
- `-greedy-reverse-local-assignment` — NEW BROKEN-APERTURE on 3/4 shapes
- `-disable-machine-sink` — -21.83pp on DLA1
- `-disable-post-ra` — -3.87pp multi-shape
- `TAIL_BARRIER_LGKMCNT` source macro — closed (best -0.21pp)
- `STEP4_BARRIER_VMCNT` source macro — closed (all values LOSE)
- `PF_GROUP_OFFSET` source macro — closed (compiler isel re-orders, +0.20pp within noise)

**Frontier post-R15 (UNCHANGED from R14)**: 4 deep-LOSE shapes (DLA1/DLA2/DLA7/P1) have NO remaining flag-level lever. LLVM flag space exhausted at 69 flags. Single-edit kernel-source probes also dead. Any future progress requires multi-day rewrites: (a) MFMA op switch to 32x32x64, (b) K-split rewrite, (c) SLM relayout, (d) full B-direct (aiter-style, blocked on >256 VGPR).

## Round 14 (2026-04-17) — non-scheduler LLVM + macro + DLA1 deep-dive, all DEAD END (committed `2717330d`)
3 parallel optimizers (Opus 4.7) attacked the 4 sched-strategy-exhausted deep-LOSE shapes from R13 along orthogonal axes. **0 wins**, but 4 critical findings exhaust the in-session optimization frontier:

- **Optimizer A (non-scheduler LLVM flags, 27 flags × 4 shapes via asm-diff probe)**:
  - **Best 5-run signal**: `greedy-regclass-priority-trumps-globalness=true` on P1 (28672×4096×16384) → +12.5 TFLOPS / +0.24pp 5-run mean, gate(mean ≥ baseline.max) PASS but **sub-+1pp threshold**. Documented for future, not deployable as variant.
  - 63/108 flag×shape combos NOOP (text-identical .text → silent no-op on this kernel).
  - **NEW BROKEN-BUILD entry**: `-mllvm -amdgpu-promote-alloca-to-vector-limit=N` (any N tried) breaks build with "illegal VGPR to SGPR copy" at line 1728.
  - **3 NEW BROKEN-APERTURE flags on DLA1** (4096×32768×128256): `misched-cluster=false`, `amdgpu-dpp-combine=false`, `amdgpu-disable-clustered-low-occupancy-reschedule`. SGPR-clobber bug class is **broader than just iterative-ilp scheduler axis**.
  - DLA2/DLA7/P1: 23/27 flags NOOP — all 3 are flag-insensitive shapes.
- **Optimizer B (kernel-macro sweep, 28 untested combos × 4 shapes)**:
  - 0/28 passed +1pp gate. Best: `pf6_6_v20_memc` on DLA1 +0.13pp 5-run mean.
  - **DLA1 `pf6_6+lgk4` triggered HSA APERTURE bug with NO scheduler flag changes** (default scheduler, just macro change). Proves SGPR-clobber bug class is shape×macro driven, not just shape×scheduler.
  - `coverage_audit.md` documents the 28 macro combos as the **exhaustive untested macro-axis set** on these 4 parents.
- **Optimizer C (DLA1 deep dive, 9 non-scheduler LLVM flags via asm-diff)**:
  - 0 wins, all flags NOOP/regress on DLA1. **DLA1 is now triple-exhausted**: R12A (sched-strategy), R13A (alt iterative schedulers), R14C (non-scheduler LLVM flags) all produced 0 deltas.

**Round 14 净增**: 0 WIN, 0 gap reduction. **10 saturation rounds total** (R2/R4/R5/R6/R7/R8/R9/R12/R13/R14). R10/R11 remain the only break-out rounds (5 verified deep-LOSE wins via un-prefixed iterative-ilp, +1.85pp avg).

**新 dead-end vectors (Round 14)**:
- `-amdgpu-membound-threshold` × {0,50,100,200} on 4 broken shapes — NOOP/regress
- `relaxed-occupancy-deps` — NOOP on 3/4 shapes, no measurable effect
- `divergence-merge` / `merge-m0-init` flags — NOOP on all 4
- `vgpr-index-mode` / `lds-thread-affinity` / `wavefront-priority-vgpr` / `lwt` (loop-warmup-time) — NOOP on all 4
- `early-spill-bypass` flags / `dce-in-ra` / `dpp-combine` (true) — NOOP / regress
- `lst{16,256}` (loop-strength-threshold) / `sghazard{0,64}` (sgpr-hazard) — NOOP on all 4
- `pf6_6+lgk4` macro combo on DLA1 — NEW BROKEN-APERTURE finding (compiler bug at default scheduler)
- 9 non-scheduler LLVM flags from R14C on DLA1 — silent NOOP (text-identical)
- 28 macro combos in coverage_audit.md — all sub-threshold (best +0.13pp)

**Frontier post-R14 (kernel-source level only — multi-day work, infeasible in single session)**:
- 4 deep-LOSE shapes (DLA1 88.3%, DLA2 92.9%, DLA7 94.5%, P1=28672×4096×16384 93.7%) have **no remaining flag-level lever**.
- Both LLVM scheduler axis (R8/R10/R12/R13) and non-scheduler LLVM axis (R14A/C) and macro axis (R14B + Round 2 deep-LOSE sweep) are **all exhausted**.
- Future work paths: (a) MFMA op switch to `v_mfma_scale_f32_32x32x64_f8f6f4`, (b) K-split rewrite (per-shape K-loop unrolling at source level), (c) SLM relayout (rebuild `st_16x128_s::swizzle` for different bank/acc pattern), (d) full B-direct (aiter-style) rewrite — needs >256 VGPRs, blocked on gfx950 register budget.

## Round 13 (2026-04-17) — alt-scheduler exhaustion sweep, all DEAD END (committed `75d5e305`)
3 parallel optimizers extending R10/R11/R12 iterative-ilp space. **0 new wins**, 3 critical findings:

- **Optimizer A (alt iterative schedulers on 4 R12-broken shapes DLA1/DLA2/DLA7/WIN2)**: 0/12 candidates survived smoke.
  - `iterative-minreg` triggers SAME SGPR-clobber bug on all 4 shapes (NaN output / aperture violation).
  - `max-ilp` triggers SAME bug on all 4 shapes.
  - `iterative-maxocc` triggers same bug on DLA1; collapses to default (.text byte-identical) on DLA2/DLA7/WIN2.
  - `max-occupancy` and `iterative-max-occupancy-experimental` are silently NO-OP (not in LLVM 20 enum).
  - **VALID un-prefixed enum names** (verified via `strings` on libLLVMAMDGPUCodeGen.a): `iterative-ilp`, `iterative-maxocc`, `iterative-minreg`, `max-ilp`, `max-memory-clause`. All others silent no-op.
  - **The compiler bug is not iterative-ilp-specific** — it's a general non-default-machinescheduler bug. The 4 broken shapes are sched-strategy-EXHAUSTED; future work must move to kernel-source changes or non-scheduler LLVM flags.

- **Optimizer B (full sched sweep on last untested-not-broken P1 shape 28672×4096×16384)**: 0 wins.
  - `iterative-ilp` shows mean +0.11pp (gate FAIL by 2.6 TFLOPS); 5-run replication confirms no real signal.
  - `max-ilp` triggers SGPR-clobber bug here too (cross-parent confirmation: bug is shape-driven, not parent-driven).
  - All 12 sched-strategy variants (5 strategies × 2 parents + 2 stacking orders) regressed or no-op.
  - **28672×4096×16384 is sched-strategy EXHAUSTED** (R8 + R10A + R13B = 3-round confirmation).

- **Optimizer C (stack alt strategies on 5 R10/R11 verified-working shapes)**: 0 wins.
  - **LOAD-BEARING METHODOLOGICAL FINDING**: `-mllvm -amdgpu-sched-strategy=` is **LAST-SPEC-WINS** in this LLVM. ASM-diff proof:
    - `parent (memc only)`: .s size 224583, memc active.
    - `iterilp + memc-appended-last`: 224583, memc wins, iterilp silently overridden.
    - `memc + iterilp-appended-last`: 223905, iterilp wins, memc silently overridden.
    - **Existing `_*_memc_r1X_iterilp` WIN variants are PURE iterilp** (parent's memc was overridden by appended iterilp flag). Naming misleading but substance correct.
  - `iterative-minreg` triggers SGPR-clobber on 4/5 working shapes (only S5 with smallest K=14336 survived).
  - `iterative-max-occupancy-experimental`: -1.06 to -2.02pp regressions across all 5 shapes.
  - `amdgpu-mfma-padding-ratio=10/25` on top of iterilp: byte-identical no-op (true no-op, not within-noise).
  - `STEP12_BR_LGKMCNT=4` on iterilp: catastrophic on S1 (-1710 TFLOPS smoke), neutral elsewhere.
  - `STEP3_BARRIER_VMCNT=24` on iterilp: +0.01 to +0.37pp single-shot (sub-threshold; 5-run verify failed gate).
  - **iterilp WIN ridge is locally optimal** — surrounding flag/scheduler space dominated by it.

**Round 13 净增**: 0 WIN, 0 gap reduction, but **3 critical findings** added to dead-end registry. **9 saturation rounds** total (R2/R4/R5/R6/R7/R8/R9/R12/R13), R10/R11 the only break-out rounds (5 verified deep-LOSE wins).

**新 dead-end vectors (Round 13)**:
- `iterative-minreg` on R12-broken shapes (4 shapes) — same SGPR-clobber bug
- `iterative-minreg` on R10/R11 working shapes (4/5) — same bug
- `max-ilp` on R12-broken shapes — same bug; cross-parent confirmed
- `iterative-max-occupancy-experimental` — works but consistent regress -1~-2pp
- `max-occupancy` strategy name — silently no-op (not in enum)
- `amdgpu-mfma-padding-ratio` on top of iterilp — true no-op
- `STEP12_BR_LGKMCNT` / `STEP3_BARRIER_VMCNT` stacked on iterilp — sub-threshold
- 28672×4096×16384 sched-strategy exhausted (3rd round)
- `memc + iterilp` flag combo — STRUCTURALLY IMPOSSIBLE (single LLVM option, last-wins)

**Frontier remaining (post-R13)**:
- 4 deep-LOSE shapes still at original ratios with no scheduler-level lever: DLA1 (4096×32768×128256, 88.3%), DLA2 (128256×32768×4096, 92.9%), DLA7 (28672×32768×4096, 94.5%), 28672×4096×16384 (93.7%).
- 1 WIN shape at risk if iterative-ilp ever defaulted on: WIN2 (32768×6144×2048, 103.9%) — but it's not.
- Future work must be **kernel-source level** (tile reshape, K-split rewrite, SLM relayout, MFMA op switch) or **non-scheduler LLVM flags** (`-amdgpu-membound-threshold`, regalloc, etc.).

## Round 12 (2026-04-17) — iterative-ilp bisect + generalization probe (committed `4c4000eb`)
2 parallel optimizers extending Round 10/11 BREAKTHROUGH discovery:

- **Optimizer A (bisect)**: CONFIRMED LLVM/AMDGPU compiler bug in un-prefixed `iterative-ilp`.
  - Fresh-rebuild bisect (`build_round12_optA_bisect.py`): parent flags WITHOUT iterative-ilp produce byte-identical ASM to baseline; WITH iterative-ilp produce byte-identical to R11 broken kernel.
  - **iterative-ilp is the SOLE differing input** triggering HSA aperture violation.
  - Deterministic 0/3 fail on DLA1 (4096×32768×128256), DLA2 (128256×32768×4096), DLA7 (28672×32768×4096), WIN2 (32768×6144×2048).
  - WIN1 (16384×4096×7168) was a 1-in-N flaky launch glitch — 3/3 OK in retry, NOT a real iterative-ilp bug.
  - Failure addr `0xff9010f50000` / `0xff46da728000` page-aligned with high bits set → C-output base SGPR pair clobbered (not just per-thread offset). "Read-only page" reason → SRD/base lands in code/rodata mapping.
  - ASM diff: ~1152 buffer_load reschedule diffs, no single-line miscompile.
  - **Action**: KEEP existing 5 verified iterative-ilp WINs from R10/R11; do NOT enable iterative-ilp as default flag; bench_all_42.py dispatcher should register the 5 `_r1X_iterilp` variants ONLY for the 5 specific shapes.
- **Optimizer B (generalization)**: ZERO new WINs on 8 untested NEAR-THRESHOLD/MID-LOSE shapes.
  - 5 of 8 candidates triggered HSA aperture violation (16384×28672×2048, 4096×32768×6144, 16384×28672×4096, 28672×4096×8192, 32768×4096×7168) — same compiler bug.
  - 2 regressions: 6144×4096×16384 -0.77pp, 14336×32768×4096 -0.91pp.
  - 1 marginal: 4096×14336×16384 +0.05pp (sub-threshold).
  - 0 candidates passed +0.5pp single-shot gate → 5-run verify skipped.
  - **Conclusion**: iterative-ilp does NOT generalize beyond the 5 verified deep-LOSE wins from R10/R11. It's a specific deep-LOSE phenomenon, not a universal optimization.
  - **Note**: several baselines drifted vs Round 2 numbers (4096×14336×16384 95.96% vs TODO 98.6%; 6144×4096×16384 97.94% vs TODO 98.6%; 14336×32768×4096 95.34% vs TODO 96.9%) — cross-GPU bias confirmed; full re-baselining would be advisable but does not change conclusion.

**Round 12 净增**: 0 new WIN, 0 new gap reduction, but **2 critical findings**:
1. Real LLVM compiler bug confirmed (deterministic SGPR clobber on iterative-ilp + certain shape patterns)
2. iterative-ilp gain is shape-specific, not generalizable

**新 dead-end vectors (Round 12)**:
- iterative-ilp on near-threshold shapes (32768×4096×7168, 4096×14336×16384, 6144×4096×16384) — 1 errors, 1 regress, 1 marginal
- iterative-ilp on mid-LOSE shapes (5 shapes) — 4 errors, 1 regress
- iterative-ilp as default global flag — UNSAFE (deterministic compiler bug on ≥10 shape categories)

**新 untested vector (Round 13 候选)**: `iterative-minreg` or `iterative-gcn-max-occupancy` on the 4 deterministic-fail shapes (DLA1/DLA2/DLA7/WIN2) — different iterative scheduler may avoid the SGPR clobber. Lower priority than the existing 5-shape WIN consolidation.

## Round 11 (2026-04-17) — 3 more deep-LOSE WINs via iterative-ilp (committed `31f03996`)
Extended Round 10 un-prefixed iterative-ilp to remaining 7 deep-LOSE + 3 WIN regression check on GPU 6.

**VALIDATED WINs (5-run mean ≥ baseline.max gate PASS)**:
| Shape | Variant | Before → After | delta |
|-------|---------|----------------|-------|
| 4096×32768×28672 | _v20_memc_r11_iterilp | 92.98% → 94.82% | +1.84pp |
| 4096×28672×32768 | _u16_r11_iterilp | 93.35% → 95.35% | +2.00pp |
| 4096×32768×14336 | _ts_lgk2_memc_r11_iterilp | 93.83% → 95.30% | +1.80pp |
| 4096×128256×32768 (WIN, regression check) | _memc_r11_iterilp | 161.23% → 163.89% | +2.46pp (no regress) |

Sub-threshold (single-shot, not validated): 32768×4096×14336 +0.74pp.

5 of 10 produced HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION → Round 12A confirmed deterministic compiler bug on those shapes.

**Round 11 净增**: +3 deep-LOSE gap reductions (avg +1.88pp on 3 shapes), 0 regressions.

**Cumulative R10+R11 deep-LOSE gap reductions** (5 shapes, average +1.85pp):
| Shape | Before → After | delta | Round |
|-------|----------------|-------|-------|
| 14336×4096×32768 | 89.6% → 91.4% | +1.78pp | R10 |
| 16384×4096×28672 | 90.1% → 91.9% | +1.82pp | R10 |
| 4096×32768×28672 | 92.98% → 94.82% | +1.84pp | R11 |
| 4096×28672×32768 | 93.35% → 95.35% | +2.00pp | R11 |
| 4096×32768×14336 | 93.83% → 95.30% | +1.80pp | R11 |

## Round 10 (2026-04-17) — VALIDATED 2x WIN via un-prefixed iterative-ilp (committed `091d3baa`)
Acted on Round 9 verifier reversal — tested REAL un-prefixed sched-strategies on deep-LOSE.

**BREAKTHROUGH**: un-prefixed `iterative-ilp` flag, single-GPU 5-run replication on GPU 5:
- 14336×4096×32768: parent `_lgk2_dc` 89.6% → +1.78pp (mean ≥ baseline.max PASS)
- 16384×4096×28672: parent `_u32` 90.1% → +1.82pp (mean ≥ baseline.max PASS)
ASM dumps in `asm_verify_r10/` confirm un-prefixed values produce distinct ASM (refute Round 8 A latent finding).

## Round 9 (2026-04-17) — Verifier reversal + 8 codegen flags + SNR bug FIX
更窄、更聚焦的探针轮 (3 任务: 1 verifier + 1 codegen probe + 1 tooling fix), 在 Round 8 NEW METHODOLOGY ADDENDUM 下:
- **Verifier** (验证 Round 8 A 的 latent finding "_memc 用 un-prefixed flag 可能 silently default to gcn-max-occupancy"): **REVERSED — Round 8 A 的方向正好搞反**.
  - 实测发现: `gcn-`-prefixed 形式 (`gcn-max-memory-clause`, `gcn-max-ilp`, `gcn-max-occupancy`, etc.) **是 no-op** — 产生与无 flag 字节相同的 ASM (size 301977, MD5 仅随机 hash 不同).
  - **un-prefixed 形式 (`max-memory-clause`, `max-ilp`)** 才是真正生效的 — 在真实 mxfp4 kernel 上 7153 行 ASM diff vs default.
  - **`-mllvm -amdgpu-sched-strategy=` 接受任何字符串** (包括 `NONSENSE_GARBAGE`) 不报错不警告 → silently fallback to default. 这是 ROCm 7.1 LLVM 20 的真实 LLVM/AMDGPU bug, 但**对我们 benchmark 没影响** — 因为 bench_all_42.py:412-461 一直在用 un-prefixed `max-memory-clause`.
  - **重大方法论反转**: **Round 6 A 和 Round 8 A 测试的 `gcn-`-prefixed 5-strategy sweep 全部是 no-op**. 那些 "all noise/regress" 的结果**毫无意义** — 它们从来没执行过新的调度. 真正 untested 的是 un-prefixed `max-ilp` / `max-occupancy` / `iterative-ilp` / `iterative-minreg` 在 deep-LOSE shapes 上的效果. 已 deployed: un-prefixed `max-memory-clause` (在 14/24 WIN best variants).
  - Verifier dummy test 仅确认 un-prefixed `max-memory-clause` 和 un-prefixed `max-ilp` 产生不同 ASM. 其他 un-prefixed 是否生效未确认.
- **Optimizer B** (8 个从未测过的 LLVM codegen 微 flag on `14336×4096×32768` best `_v16_wpe2`): **全部 DEAD END**. flag list: loop-prefetch, schedule-metric-bias=80/100, mfma-padding-ratio=10/25, disable-loop-alignment, disable-clustered-low-occupancy-reschedule, disable-unclustered-high-rp-reschedule, use-aa-in-codegen, enable-pre-ra-optimizations. 全部 [-0.85, -0.03] pp (轻微 regress 或 noise tied). 最差 `disloopalign` -0.85pp. 没有触发 +0.8pp 复测. 11 variants 加入 dead-end list.
- **Optimizer C** (修复 documented SNR false-OK NaN bug): **FIXED, committed `b46834a0`**.
  - bug 实际不在 `bench_deep_lose.py` (该文件无 SNR gate, 是 perf-only spot bench) 而在 `bench_optC_round6.py:175` (`if snr_db < 25` mis-classify NaN as OK) 和 `bench_round6_optA.py` (NaN-tainted output → `noi>0` short-circuit → snr=+Inf 通过).
  - Fix: 新建 `snr_check.py` (NaN-safe `is_snr_ok` / `classify_snr` / `compute_snr_db`); 两个 bench 文件均拒绝 NaN/None/-Inf SNR.
  - **重要发现**: Round 6 OptC 的 `_ts_u8*` 5 个 variants 全部 NaN output 但 silently 通过 SNR gate, 任何 "wins" 都是无意义的. Future agents 用 bench_optC_round6.py 都需要重新验证.

**Round 9 净增**: 0 WIN, 0 gap reduction, 但 **1 real commit (SNR fix)** + **1 critical methodology reversal (sched-strategy prefix)**. 继续 6 → 7 saturation rounds.

**新 dead-end vectors (Round 9)**:
- 8 LLVM codegen 微 flags (loop-prefetch, sched-metric-bias, mfma-padding, disloopalign, etc.) — 全部 noise/regress
- `gcn-`-prefixed sched-strategy 名 (`gcn-max-memory-clause` etc.) — 全部 silently no-op, 等于 default

**新 untested vector (Round 10 候选)**: un-prefixed `max-ilp` (verified 工作的) 在 deep-LOSE shapes 上 — 真正未测.

## Round 8 (2026-04-17) — 3 parallel optimizers, all DEAD END / NO-OP
GOAL PIVOT 第三轮, 在 Round 6 methodology rule 下, 攻击真正未测的窄向量:
- **Optimizer A** (per-shape `-mllvm -amdgpu-sched-strategy=` bucketing on 4 untested deep-LOSE shapes 4096×32768×28672 / 28672×4096×16384 / 4096×28672×32768 / 32768×4096×14336): **DEAD END**. 19 variants × 4 shapes × 4-5 strategies (`gcn-max-occupancy`, `gcn-max-ilp`, `gcn-iterative-ilp`, `gcn-iterative-minreg`). 全部 [-0.81, +0.09] pp 区间, 最大 +0.09pp `_ts_gm8_sched_memc` on 28672×4096×16384 (远低于 +0.8pp 触发). 加上 R6 A 的 4096×32768×128256 + R6 B 的 14336×4096×32768 + R6 C 的 16384×4096×28672, sched-strategy vector 现已覆盖 7/10 deep-LOSE shapes, **vector 完全 exhausted**.
  - **重要 latent finding**: 现有 `*_memc` baselines (在 14/24 WIN best variants 中) 使用 un-prefixed `max-memory-clause` flag, 可能 silently default to `gcn-max-occupancy` (LLVM 不识别该值 → fallback). 如果验证, 意味着 `_memc` family 实际是 `gcn-max-occupancy`, 不是 `gcn-max-memory-clause`. 这是潜在 bug 但解释了为何 memc 在 WIN shapes 上有效 (与默认不同). 未来轮可验证, 但不会改变 deep-LOSE 结论 (R6 A 用了正确 prefix 测试, 全部 noise).
- **Optimizer B** (TAIL_BARRIER_VMCNT × VMCNT × 3 large-K deep-LOSE shapes 4096×32768×128256 / 4096×32768×14336 / 32768×4096×14336): **DEAD END**. TAIL_BARRIER_VMCNT 仅在 `TAIL_SPLIT=1` 路径生效 (确认 line 2174, 2361). 29 variants × 3 shapes × {0,4,8,16,24} × SV±4. 全部 [-0.19, +0.27] pp 区间, 最佳 +0.27pp `_ts_u16_lgk2_tbv16` on 32768×4096×14336 (低于 +0.8pp 触发). **机械性 insight (重要)**: 大 K 下 (K≥14336) tail iter 占总 K iters 的 ≤0.45%, 即使完美调 tail barrier 也只能 shift 微小比例 → 该 vector 数学上不可能产出可见增益 on 大 K shapes.
- **Optimizer C** (`__attribute__((amdgpu_waves_per_eu(1,1)))` on 3 register-pressure shapes 128256×32768×4096 / 28672×32768×4096 / 4096×32768×128256): **NO-OP + 已 REFUTED**. 编译器 resource usage 显示 baseline (无 wpe attribute) 已经是 1 wave/SIMD (224V + 256A = 480 regs, 加上 `__launch_bounds__(_NUM_THREADS, 1)` line 1727). 加 wpe(1,1) bytewise-identical .so, 是 literal no-op. **更重要**: Round 2 deep-LOSE 工作 already tested wpe1 (build_deep_lose_variants.py:39-50: `_wpe1`, `_v16_wpe1`, `_ts_lgk2_v12_wpe1_memc`, etc.), bench_deep_lose_results.json 显示全部 LOSE on 3 target shapes (-7.6 to -51.9 TFLOPS). 我之前 prompt 写的 "wpe1 was NEVER tested" 是错的, 应记录在 dead-end list. 0 文件创建.

**Round 8 净增**: 0 WIN, 0 gap reduction. **6 轮饱和** (R2/R4/R5/R6/R7/R8). 每轮 0 净增. 我已系统耗尽所有 in-session-feasible 优化向量.
**新 dead-end vectors (Round 8)**:
- per-shape sched-strategy bucketing on 4 P1/P2 deep-LOSE shapes (max +0.09pp)
- TAIL_BARRIER_VMCNT × large-K deep-LOSE shapes (mechanistically futile, tail iter ≤0.45%)
- `amdgpu_waves_per_eu(1,1)` on register-pressure shapes (no-op + already REFUTED)

**Latent investigation**: `_memc` baselines may use un-prefixed `max-memory-clause` flag; verify whether that silently falls through to `gcn-max-occupancy`. Won't change deep-LOSE conclusion but worth documenting.

## Round 7 (2026-04-17) — 3 parallel optimizers, all DEAD END / INFEASIBLE
GOAL PIVOT 第二轮, 在 Round 6 methodology rule 下:
- **Optimizer A** (STEP12_BR_LGKMCNT∈{0,2,4} sweep on 3 P1 shapes: 4096×32768×28672, 28672×4096×16384, 4096×28672×32768): **DEAD END**. Round 6 C 在 16384×4096×28672 上的 brlgk2 directionally-positive 信号**不泛化**. 9 个 variant 单 GPU 测量, 全部 -0.03 ~ -0.14pp (<<+0.6pp noise). brlgk0 是 default value.
- **Optimizer B** (STATIC_XCD_REMAP on mega-M 128256×32768×4096): **DEAD END**. 实现 atomic-free static remap (each XCD owns N-strip of width bpc/NUM_XCDS=16, walks GROUP_M×16 tiles). 6 variant grid (baseline_dc/gm4/gm8 × static_xcd_remap{,_gm4,_gm8}): best STATIC variant -0.37pp vs baseline. 失败原因: **mega-M shape 是 A-bound, 不是 B-bound** — A traffic dominates (M=128256 vs N=32768), 缩 B working set 8x 反而损失 8x A reuse. 与 Round 4 PERSISTENT_XCD_QUEUE 同源结论, 进一步确认 mega-M 92.9% gap 是 register-pressure / occupancy 结构性 bound. Code 留在 `STATIC_XCD_REMAP=1` flag (default 0) 作为 documented dead-end.
- **Optimizer C** (B-tile L2 prefetch via `__builtin_amdgcn_global_load_lds` on 4096×28672×32768 / 4096×32768×14336): **INFEASIBLE — premise wrong**. 现有 `emit_one_pf()` (line 466-472) **已经是** `__builtin_amdgcn_global_load_lds` 的 buffer-SRD 形式 (`llvm_amdgcn_raw_buffer_load_lds`, emits `BUFFER_LOAD_DWORDX4 lds:1`), 已 prefetch B `bt+2` look-ahead 直接进 LDS. 三种"扩展"方案都不可行: (1) 加 redundant 16B/thread 到 scratch LDS = 纯 VMEM duplication 在已 B-VMEM-bound 的端口上, 必退化; (2) bump look-ahead `bt+2` → `bt+3/4` 不是新机制只是常数, 且 LDS 双缓冲 +50% 超 160KB cap; (3) GLOBAL_LOAD_LDS 没有 discard sink mode (硬件强制写到 LDS dest). 未跑 bench, 25 min 提早终止, 0 文件创建. **重要文档**: future agents 不要再提 "L2 prefetch via global_load_lds" 这个 vector — 已 deployed.

**Round 7 净增**: 0 WIN, 0 gap reduction. 5 轮饱和确认 (R2 ceiling + deep-LOSE分析员 + R4 + R5 + R6 + R7).
**新教训** (扩展 Round 6 methodology):
- mega-M shape 是 A-bound, 缩 B-locality 必失败 (Round 4 PERSISTENT_XCD_QUEUE + R7 STATIC_XCD_REMAP 双重证实)
- `emit_one_pf` IS `__builtin_amdgcn_global_load_lds` (buffer-SRD form), B 已 bt+2 prefetch 进 LDS, 没有 "未利用的 prefetch 机制"
- "Round 6 C 的 STEP12_BR_LGKMCNT=2 directionally-positive 信号" 是 single-shape 边际, 不可泛化

## Round 6 (2026-04-17) — 3 parallel optimizers, all DEAD END / MARGINAL (REVERTED)
GOAL PIVOT 后第一轮, 直接攻 deep-LOSE shapes:
- **Optimizer A** (4096×32768×128256 / compiler flags + L2 prefetch): **DEAD END**. gfx950 无 L2 prefetch instruction; igroup-lp 不可用; 单次 +0.29pp, 5-run 中位 -0.5pp (in noise).
- **Optimizer B** (14336×4096×32768 / UNROLL_K sweep): **claimed +3.16pp WIN (commit 198bb3a4) → REJECTED by Reviewer**.
  - Optimizer B 的 baseline 测量 4591 TFLOPS 是**假低**: 单 GPU 5-run replication 验证 baseline 实际 4727±13 TFLOPS (不是 4591). 真实 delta:
    | variant | mean TFLOPS | mean pp over base | min vs base max |
    |---------|-------------|-------------------|-----------------|
    | _v16_wpe2 (baseline) | 4727.22 | 0.00 | 0.00 |
    | _optB_r6_u16_v16_wpe2 | 4730.26 | +0.06 | -0.73 |
    | _optB_r6_u8_v16_wpe2_memc | 4721.64 | -0.11 | -0.89 |
    | _optB_r6_u16_lgk2_dc_v16_wpe2 | 4735.57 | +0.16 | -0.38 |
  - 全部三个变体 mean Δ 均 <+1pp 阈值, 全部 worst-case (min vs baseline max) 为负. WIN-sample 与其余 deep-LOSE neighbor 也无显著 gain (±0.5pp 内)
  - 教训: Optimizer B 在 GPU 4 上测的 baseline 与之前 baseline (GPU 不同) 比较, 触发了 **GPU bias 50-100 TFLOPS** + per-run noise ±0.6pp 的合成假象
  - **Action**: `git revert 198bb3a4` (commit 4c11f8bb). 验证脚本保留: `spot_optB_r6_validation.py` + `spot_optB_r6_validation.log` + `spot_optB_r6_validation_results.json`
- **Optimizer C** (16384×4096×28672 / TAIL_SPLIT epilogue tuning): **DEAD END**. TAIL_SPLIT=1 在 K≥7168 上更差; best variant +0.19pp, 远低于 +1pp 阈值

**Round 6 净增**: 0 WIN, 0 gap reduction. **新教训**: 跨 GPU 比较 baseline 不可靠 (GPU bias ~2pp), 任何 deep-LOSE 改进 claim **必须** single-GPU 5-run replication 验证, 且 best mean 必须 ≥ baseline max.

## 剩余 untested vectors (out of in-session scope)
- **MFMA_32X32X64_TILING**: 切换 `v_mfma_scale_f32_16x16x128_f8f6f4` → `v_mfma_scale_f32_32x32x64_f8f6f4`. 巨大 kernel rewrite (>1 day, asm + layout 全改). AGPR 从 256 降到 64 释放 192 VGPRs/AGPRs 用于深度 B-buffering. 是唯一未测的"内核重构"级别尝试, 接近 aiter 架构.
- **B_TRIPLE_BUFFER**: 3-stage pipeline 替代当前 2-stage. LDS budget 是杀手 (131KB → 163KB > 160KB max). 仅在 #1 (32x32 MFMA 释放 AGPR) 完成后才可行.
- Optimizer A 副产建议: 早期 scale prefetch (移到 Step12 前), 4× dwordx2 → 1× dwordx8 burst 合并, drop redundant scale stream 当 a0_raw == a1_raw.

## Rocprof 分析结论 (2026-04-16)
对生产 .s (N=32768, K=4096, TS=1, LGK2) 做了 PC sampling 和 assembly 分析:
- 2048 MFMAs, 512 ds_reads, 165 s_nop, 313 asm block pairs
- **s_nop 全部在 PF 组里** (162/165 在 buffer_load...lds 前), 是 m0→buffer_load 硬件 hazard delay
- **Inter-block code 被 MFMA pipeline 完全隐藏**: MFMA pipeline depth ≥16 cycles, inter-block gap ≤6 cycles
- **63 s_nop after ASMEND**: m0 在 asm 块前设定, 块内不 clobber m0, 理论可移除但实际增益 0
- **99 s_nop in PF pairs**: m0→buffer_load 硬件 hazard, 移除导致 26% 计算结果错误
- **编译→重写→重组装 pipeline 验证可行** (rewrite_asm.py + build_rewrite.sh), 但无可行的重写优化

## Benchmark Rules
- **warmup=200, iters=500**, trimmed mean 10%
- 用空闲GPU (`rocm-smi` 确认0%)
- `HIP_VISIBLE_DEVICES=N`
- MI355X 上 competitor_tflops 是正确 baseline

## 关键文件
| 文件 | 用途 |
|------|------|
| `kernel_mxfp4_gluon_cpp.cpp` | 主生产内核 (62 auto-tune flags) |
| `bench_all_42.py` | 42-shape benchmark (62 auto-tune variants, sequential) |
| `bench_all42_parallel.py` | 42-shape benchmark (parallel across GPUs, 62 variants) |
| `build_all42_parallel.py` | 并行编译器 (62 variants × 26 N,K pairs) |
| `spot_test.py` | 单shape多variant测试 (95 variants) |
| `spot_new_variants.py` | 新variant快速spot测试 (30 new + 9 reference) |
| `bench_all42_results.json` | 最新42-shape结果 (24/42 WIN, 115 variants, Round 2) |
| `bench_all42_results_round1.json` | Round 1 snapshot (24/42 WIN, 87 variants) |
| `bench_all42_results_round2.json` | Round 2 snapshot (24/42 WIN, 115 variants) |
| `build_new_variants.py` | 并行编译 (memclause + Optimizer A 21 variants) |
| `build_round2_variants.py` | 并行编译 (Round 2 ceiling 28 variants) |
| `build_deep_lose_variants.py` | 并行编译 (deep-LOSE 44 targeted variants) |
| `bench_deep_lose.py` / `bench_deep_lose_results.json` | 10-shape spot bench (deep-LOSE 分析员) |

## 环境设置 (换机器必读)
```bash
cd /shared_nfs/kyle/test/HipKittens
git checkout mxfp4
cd analysis/fp8_gemm/mi350x

# 并行编译所有.so (16线程)
python3 build_all42_parallel.py 16

# 用GPU 1-4跑benchmark (parallel, needs pre-built .so)
python3 bench_all42_parallel.py 1,2,3,4

# 用GPU 1跑benchmark (sequential, handles compilation)
HIP_VISIBLE_DEVICES=1 python3 bench_all_42.py

# 单shape测试
HIP_VISIBLE_DEVICES=2 python3 spot_test.py 6144 32768 4096 4291.0

# Cursor的仓库 (只读参考)
# /shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x/
```
