# R59 Opt R — Mixed-Protocol ITERS Policy Decision (POLICY_ONLY)

**Round**: R59
**Worker**: A (cohort J-1)
**Scope**: Analysis-only, no GPU usage. Decides whether R59+ rounds adopt
ITERS=500 (R58 default), ITERS=1000 (R57 protocol bump made permanent),
or a mixed L1-only ITERS=1000 protocol.

**Inputs**:
- `R57_INTEGRATION_10RUN.json` (ITERS=1000, R57 baseline)
- `R58_INTEGRATION_10RUN.json` (ITERS=500, R58 baseline)
- `R58_INTEGRATION_VERDICT.md` (§"ITERS=500 revert observation")

**Population**: 41 unchanged-binary cells (P-2 cell `16384x4096x3072` excluded
from drift analysis; that cell was R57 R40B HK → R58 R58P2_AITER swap, so
the R57→R58 delta is dominated by the swap and not by the ITERS change).

---

## R-1: Per-cell drift table (R57 ITERS=1000 − R58 ITERS=500)

| cell                         | source            | R58 @ 500 | R57 @ 1000 | Δ (R57−R58) | wcf_max@R58 | fin_min@R58 |
|-------------------------------|-------------------|-----------|------------|-------------|-------------|-------------|
| 16384x4096x2048              | R40B              | 107.48    | 109.57     | **+2.09**   | 0.0006      | 0.9971      |
| 14336x4096x32768             | R51D1_AITER       | 104.16    | 103.01     | **−1.15**   | 0.0000      | 1.0000      |
| 4096x4096x8192               | R55D5A_3_AITER    | 120.49    | 121.58     | +1.09       | 0.0000      | 1.0000      |
| 6144x4096x8192               | R54D4A_3_AITER    | 117.97    | 118.86     | +0.89       | 0.0000      | 1.0000      |
| 16384x14336x2048             | R55E3_1_AITER     | 106.95    | 106.34     | −0.61       | 0.0000      | 1.0000      |
| 4096x32768x128256            | R52D2B_AITER      |  98.28    |  97.75     | −0.53       | 0.0000      | 1.0000      |
| 16384x4096x6144              | R54E2_3_AITER     | 115.66    | 116.19     | +0.53       | 0.0000      | 1.0000      |
| 6144x4096x16384              | R53D3A_2_AITER    | 105.41    | 105.92     | +0.51       | 0.0000      | 1.0000      |
| 32768x6144x2048              | R55D5B_3_AITER    | 106.67    | 106.17     | −0.50       | 0.0000      | 1.0000      |
| 6144x32768x4096              | R55E4_2_AITER     | 106.10    | 105.67     | −0.43       | 0.0000      | 1.0000      |
| 4096x14336x16384             | R54D4A_1_AITER    | 106.93    | 107.30     | +0.37       | 0.0000      | 1.0000      |
| 32768x4096x2048              | R54E1_1_AITER     | 106.96    | 106.61     | −0.35       | 0.0000      | 1.0000      |
| 16384x6144x4096              | R55E4_1_AITER     | 114.51    | 114.84     | +0.33       | 0.0000      | 1.0000      |
| 4096x4096x16384              | R54D4B_1_AITER    | 113.26    | 113.58     | +0.32       | 0.0000      | 1.0000      |
| 16384x4096x7168              | R54D4B_3_AITER    | 113.35    | 113.67     | +0.32       | 0.0000      | 1.0000      |
| 4096x128256x32768            | R56G4_C1_AITER    | 178.51    | 178.24     | −0.28       | 0.0000      | 1.0000      |
| 28672x32768x4096             | R56G2_L4_AITER    | 102.72    | 102.45     | −0.27       | 0.0000      | 1.0000      |
| 16384x14336x4096             | R55E3_2_AITER     | 108.53    | 108.79     | +0.26       | 0.0000      | 1.0000      |
| 14336x32768x4096             | R56G2_L5_AITER    | 103.67    | 103.42     | −0.25       | 0.0000      | 1.0000      |
| 32768x4096x3072              | R54E1_2_AITER     | 118.59    | 118.82     | +0.23       | 0.0000      | 1.0000      |
| 32768x28672x2048             | R55D5B_2_AITER    | 103.20    | 103.42     | +0.22       | 0.0000      | 1.0000      |
| 16384x4096x14336             | R56G1_L6_AITER    | 107.08    | 107.29     | +0.21       | 0.0000      | 1.0000      |
| 32768x4096x7168              | R54D4A_2_AITER    | 106.07    | 106.28     | +0.21       | 0.0000      | 1.0000      |
| 128256x32768x4096            | R56G2_L3_AITER    | 102.61    | 102.81     | +0.21       | 0.0000      | 1.0000      |
| 4096x4096x32768              | R52D2C_AITER      | 108.20    | 108.40     | +0.20       | 0.0000      | 1.0000      |
| 28672x4096x8192              | R54E1_3_AITER     | 105.79    | 105.96     | +0.17       | 0.0000      | 1.0000      |
| 16384x28672x4096             | R55E3_4_AITER     | 104.32    | 104.43     | +0.11       | 0.0000      | 1.0000      |
| **4096x32768x14336**         | **R57J1_L1_AITER**|  **99.94**| **100.04** | **+0.11**   | **0.0000**  | **1.0000**  |
| 4096x32768x4096              | R54E2_1_AITER     | 105.32    | 105.42     | +0.10       | 0.0000      | 1.0000      |
| 16384x4096x28672             | R51D2_AITER       | 102.98    | 102.88     | −0.10       | 0.0000      | 1.0000      |
| 16384x28672x2048             | R55E3_3_AITER     | 102.18    | 102.27     | +0.09       | 0.0000      | 1.0000      |
| 4096x6144x32768              | R53D3A_3_AITER    | 130.58    | 130.53     | −0.05       | 0.0000      | 1.0000      |
| 32768x4096x14336             | R56G1_L2_AITER    | 103.97    | 104.02     | +0.05       | 0.0000      | 1.0000      |
| **32768x14336x2048**         | **R40B**          |**100.49** | **100.53** | **+0.04**   | **0.0111**  | **0.9108**  |
| 4096x32768x6144              | R54E2_2_AITER     | 105.83    | 105.86     | +0.03       | 0.0000      | 1.0000      |
| 16384x4096x4096              | R55D5A_1_AITER    | 113.20    | 113.22     | +0.02       | 0.0000      | 1.0000      |
| 16384x6144x2048              | R55D5A_2_AITER    | 111.87    | 111.89     | +0.02       | 0.0000      | 1.0000      |
| 4096x14336x8192              | R54D4B_2_AITER    | 115.85    | 115.84     | −0.01       | 0.0000      | 1.0000      |
| 28672x4096x16384             | R51D3_AITER       | 103.91    | 103.90     | −0.01       | 0.0000      | 1.0000      |
| 4096x28672x32768             | R52D2A_AITER      | 102.50    | 102.49     | −0.01       | 0.0000      | 1.0000      |
| 4096x32768x28672             | R50D_AITER        | 101.17    | 101.17     | +0.00       | 0.0000      | 1.0000      |

**Aggregate drift (41 cells)**:
- mean Δ = **+0.102 pp** (R57 ITERS=1000 slightly higher on average)
- abs-mean = **0.323 pp**
- max +Δ = +2.09 pp (R40B HK on `16384x4096x2048`)
- max −Δ = −1.15 pp (R51D1_AITER on `14336x4096x32768`)
- HK cells (n=2): mean Δ = **+1.06 pp** (consistent with R57 prediction
  "HK cells benefit most from longer ITERS via wider trimmed-distribution
  shape")
- AITER cells (n=39): mean Δ = **+0.05 pp** (essentially noise; AITER kernel
  binaries are bit-deterministic so trim shape is dominated by GPU clock noise
  not race outcomes)

**Sign analysis**: 25 of 41 cells are +Δ, 15 are −Δ, 1 is exactly 0.
The drift is bidirectional with magnitude clearly stratified — large drifts
(|Δ| > 0.5pp) cluster in cells with HK kernels or with high comp ratios
(>117%, where the absolute pct_comp is more sensitive to small tflops
deltas).

---

## R-2: WIN-flip-sensitive band identification

Cells with R58 pct_comp ∈ [99.5%, 100.5%]:

| cell                       | source            | R58 @ 500 | R57 @ 1000 | Δ      | wcf_max | fin_min | flip-sensitive? |
|----------------------------|-------------------|-----------|------------|--------|---------|---------|-----------------|
| **4096x32768x14336**       | R57J1_L1_AITER    | **99.94** | **100.04** | +0.11  | 0.0000  | 1.0000  | **YES (LOSE↔WIN)** |
| 32768x14336x2048           | R40B              | 100.49    | 100.53     | +0.04  | 0.0111  | 0.9108  | NO (both sides WIN; VC-flipped, not pct-flipped) |

**Confirmation**: L1 (`4096x32768x14336`, R57J1_L1_AITER, R56G1 binary) is
the **only** cell whose WIN/LOSE classification flips under the ITERS choice
on the R58 manifest. Per R58 verdict §"ITERS=500 revert observation":
> R58 should NOT re-bump ITERS for L1 only [...] the cost of mixed protocols
> across a 42-shape leaderboard is not worth a single-cell flip when the
> kernel binary is identical.

The second band-resident cell (`32768x14336x2048` R40B HK) sits at 100.49%
under R58 and 100.53% under R57 — both **numerically WIN**. Its issue is
**VC** not pct_comp: under R58 ITERS=500 it tail-draws fin_min=0.911 (below
0.97 gate), whereas R57 ITERS=1000 had fin_min=0.9847 (above gate). This is
addressed by R59 cohort J-2 (Opt S) alt-tile probe, NOT by the ITERS policy.

**Total cells in WIN-flip-sensitive band: 2** (1 LOSE↔WIN flip-eligible: L1;
1 in-band but not flip: `32768x14336x2048`).

---

## R-3: Cost / benefit table

| Option | Description | Cost | Benefit |
|--------|-------------|------|---------|
| **(A)** | **ITERS=500 default + L1 footnote (current R58 disposition)** | L1 (`4096x32768x14336`) stays at 99.94% LOSE-edge; 1 LOSE-edge cell visible on leaderboard. Footnote required in each round verdict noting "L1 sits within 0.06pp of WIN line; bit-deterministic; classified LOSE under default protocol." | Bench wall-clock unchanged (~17 min reviewer integration). Single uniform protocol across all 42 cells → leaderboard rows directly comparable. No manifest schema changes. No protocol-mixing precedent. |
| **(B)** | **ITERS=1000 default (R57 bump made permanent)** | 2× wall per round (~17 → ~34 min reviewer integration). At 6+ planned rounds (R59-R64+), incremental cost ≈ +90 min budget. AITER cells drift ~−0.05 pp on average (negligible). HK cells gain ~+1.06 pp on average — but only 2 HK survivors remain, dropping to 1 once L3 is rescued. | L1 +1 WIN cell stable across rounds. HK survivors `16384x4096x2048` and `32768x14336x2048` see tighter trim distributions (R57 fin_min=0.9847 vs R58 fin_min=0.911 on the latter) → may protect VC on the worst-margin survivor. |
| **(C)** | **Mixed-protocol L1-only ITERS=1000 (per-cell protocol annotation in manifest)** | Leaderboard inconsistency: different ITERS per cell complicates row-to-row comparison and any aggregate "mean pct_comp" metric. Manifest schema gains a `bench_iters` per-cell field. **First protocol-mixed leaderboard in project history** — sets precedent for further per-cell tuning and erodes the "single protocol gate" property that has held since R45. | Minimal incremental wall (~+1 min for one cell @ ITERS=1000). L1 +1 WIN cell stable. |

---

## R-4: Recommendation + future-bump validity envelope

### Recommendation: **(A) ITERS=500 default + L1 footnote**

**Justification**:

1. **Leaderboard consistency value > single-cell flip.** A uniform protocol
   makes all 42 rows directly comparable and is the property that lets
   round-to-round aggregate metrics (mean pct_comp, AITER bit-det share,
   HK pool size) carry meaning. Mixed protocols in (C) destroy this.

2. **R57 ITERS=1000 already validated as one-shot mechanism for ≤0.5pp
   boundary cells.** R57 Opt J demonstrated the bump works exactly when needed
   (L1 99.98% → 100.04%) and the round verdict captured both numbers. There
   is no scientific reason to repeat the bump every round — the mechanism
   has been characterized.

3. **L1 is bit-deterministic at 99.94% (wcf_max=0).** The LOSE classification
   is purely measurement-noise, not a real perf gap. The WIN line is an
   arbitrary 100% threshold; L1 is functionally identical under both
   protocols. Adopting (B) just to flip the binary classification on a
   bit-deterministic cell whose true perf is fixed is paying ~90 min of
   bench wall-clock for a cosmetic leaderboard change.

4. **The HK fin_min benefit of (B) is concentrated on 1 cell**
   (`32768x14336x2048` R40B HK): R57 fin_min=0.9847 vs R58 fin_min=0.911. The
   correct fix for that cell is a structural rescue (R59 Opt S alt-tile
   probe), not a global ITERS bump that protects the symptom while the
   underlying cohort race persists.

### Future ITERS-bump validity envelope

A one-off `R<N>_OPT_<X>_LONG_BENCH` ITERS=1000 boundary-bump probe is
justified ONLY when **all four** of the following hold:

1. **Bit-determinism**: the cell must have wcf_max = 0 across 10 INDEPENDENT
   seeds at ITERS=500 (otherwise the noise mechanism is cohort-race, not
   trim-shape, and ITERS bump does not address it).

2. **Boundary proximity**: the cell's pct_comp must be within ≤0.5 pp of a
   classification boundary — either the WIN/LOSE line at 100% or the
   strict-VC fin_min gate at 0.97 (otherwise the bump cannot change
   classification).

3. **Documented in round verdict**: both ITERS=500 and ITERS=1000 numbers
   must appear in the round verdict, with the bump explicitly flagged as a
   one-off probe (e.g. "Opt J: ITERS=1000 protocol bump on L1 binary AS-IS").

4. **Not merged into manifest**: the bump must NOT be promoted to a
   permanent per-cell `bench_iters` annotation in the manifest. The default
   protocol remains ITERS=500 for the leaderboard run.

If a cell repeatedly fails (1) or (2) across multiple rounds, the correct
disposition is a structural mechanism fix (alt-tile, kernel rebuild, .co
swap) — not a protocol band-aid.

---

## Verdict

**POLICY_ONLY.** R59 cohort J-1 / Opt R closes the "L1 ITERS=1000 one-off
bump" axis (per R59 decider plan §2 row 41) by formalizing Option (A) as
project policy and defining the validity envelope for any future probes.
No manifest changes; no PROMOTE; no DEAD; no kernel work.

**Artifacts**:
- `R59_OPT_R_POLICY.md` (this file)
- `R59_OPT_R_POLICY.json` (machine-readable summary)
- `R59J1_R_INTEGRATION_FRAGMENT.json` (verdict fragment)
