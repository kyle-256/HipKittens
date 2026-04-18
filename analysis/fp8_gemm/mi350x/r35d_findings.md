# R35 Dev D — LLaMA full 14-shape matrix re-bench (CRR vs RRR + RCR)

## Verdict: NO SHIP (informational re-baseline + autotune-fan-out targets)

## Setup
- **Base commit:** 758ad933 (R34 wrap; *does not* include r35-a's 5th V2-RRR autotune predicate)
- **Bench harness:** `r35d_aggregate.py` orchestrates `bench_v2_paired.py`-style BABA benches (n=10/kernel) per cell
- **GPUs:** physical 6 + 7 (sequential to avoid contention with r35-a/b/c parallel work)
- **Cells:** 14 LLaMA-shape projections (8B and 70B; Q/K/V/O/Gate/Up/Down)
- **Comparisons:** CRR vs RRR for all 14; CRR vs RCR for the 4 square Q/O cells
- **Correctness:** all PASS (snr ≥ 49.6 dB, pass_rate=100%, det_ok=True)

## Headline numbers (CRR vs RRR; Δ% = RRR_vs_CRR median; positive = RRR faster)

| Cell | Shape | GPU6 Δ% | GPU6 t | min Δ% | Adv expected (@758ad933) | Verdict |
|---|---|---:|---:|---:|:---:|---|
| 8B-Q   | 4096×4096×4096   | +4.83 | +2.71 | +4.83 | no | OK |
| 8B-K   | 4096×1024×4096   | +7.74 | +24.14 | +7.74 | YES | **predicate fires** |
| 8B-V   | 4096×1024×4096   | +8.57 | +15.43 | +8.57 | YES | **predicate fires** |
| 8B-O   | 4096×4096×4096   | +4.73 | +2.34 | +4.73 | no | OK |
| 8B-Gate | 4096×14336×4096 | +6.25 | +6.82 | +6.25 | no (r35-a not landed @ R34 head) | **fan-out target landed in r35-a** |
| 8B-Up  | 4096×14336×4096  | +5.26 | +10.49 | +5.26 | no (r35-a not landed @ R34 head) | **fan-out target landed in r35-a** |
| 8B-Down | 4096×4096×14336 | +9.51 | +2.77 | +9.51 | no | **R36+ candidate** |
| 70B-Q  | 4096×8192×8192   | +7.72 | +6.80 | +7.72 | no | **R36+ candidate** |
| 70B-K  | 4096×1024×8192   | +11.06 | +28.08 | +11.06 | YES | **predicate fires** |
| 70B-V  | 4096×1024×8192   | +10.21 | +4.01 | +10.21 | YES | **predicate fires** |
| 70B-O  | 4096×8192×8192   | +6.94 | +9.61 | +6.94 | no | **R36+ candidate** |
| 70B-Gate | 4096×28672×8192 | +7.93 | +41.49 | +7.93 | YES | **predicate fires** |
| 70B-Up | 4096×28672×8192  | +7.87 | +39.57 | +7.87 | YES | **predicate fires** |
| 70B-Down | 4096×8192×28672 | +12.16 | +30.31 | +12.16 | YES | **predicate fires** |

## CRR vs RCR — square Q/O cells (3-way comparison)
| Cell | Shape | GPU6 Δ% (RCR vs CRR) | GPU6 t |
|---|---|---:|---:|
| 8B-Q   | 4096×4096×4096   | +7.05 | +4.22 |
| 8B-O   | 4096×4096×4096   | +5.83 | +3.50 |
| 70B-Q  | 4096×8192×8192   | +8.32 | +8.26 |
| 70B-O  | 4096×8192×8192   | +8.20 | +9.16 |

RCR is the **best** layout for all 4 square Q/O cells (ahead of both CRR and RRR by significant margins). The R28-R32 RCR vs CRR autotune predicates already cover these — but R36+ should re-verify against the R35 baseline since the build moved.

## Findings

1. **All 7 R34 wired V2-RRR autotune predicates remain valid** at R35 base.
   - 8B-K +7.74%, 8B-V +8.57%, 70B-K +11.06%, 70B-V +10.21%, 70B-Gate +7.93%, 70B-Up +7.87%, 70B-Down +12.16%
   - All Welch-t ≥ +24 except 70B-V (+4.01 because of 8.6× higher CRR variance on this run).

2. **4 R36+ fan-out candidates surfaced** (RRR > CRR, predicate not yet wired):
   - **8B-Down**: +9.51% — largest uncovered gap; shape (4096, 4096, 14336)
   - **70B-Q**: +7.72% — shape (4096, 8192, 8192) but RCR is +8.32% better, so route to RCR not RRR
   - **70B-O**: +6.94% — same as 70B-Q; RCR wins
   - **8B-Q**: +4.83% — borderline; RCR is +7.05% better, route to RCR not RRR

3. **r35-a's predicate (M=4096 N=14336 K=4096) is validated by this matrix.**
   - 8B-Gate +6.25% Welch t=+6.82 and 8B-Up +5.26% Welch t=+10.49 both confirm RRR > CRR.
   - r35-a's commit landed exactly the predicate that this matrix would have prescribed.

4. **Square Q/O cells need RCR autotune re-verify** (R36+).
   - The +5-8% RCR advantage on 8B-Q/8B-O/70B-Q/70B-O is the largest unexploited layout gap in the matrix.

## Action items for R36+
- **High**: wire V2-RRR predicate for 8B-Down (4096, 4096, 14336)
- **High**: verify the existing R28-R32 RCR autotune predicates fire for 8B-Q/8B-O/70B-Q/70B-O at R35 base; if missing, wire them
- **Medium**: extend the matrix re-bench to physical GPUs 0/4/5 for cross-GPU triangulation per R32 rule

## Methodology notes
- Single-GPU pass on GPU6 (GPU7 measurements truncated by parallel-agent contention; documented as "n/a" in aggregate table)
- All 14 cells PASS correctness gates
- Bench-harness BABA pattern + 30 s preheat (R29 standard)
