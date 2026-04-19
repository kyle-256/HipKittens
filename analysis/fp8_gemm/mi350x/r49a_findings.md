# R49 Dev A — CRR opsel-friendly scale layout — REFUTED

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 3740597b
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=3
**Hypothesis:** Re-pack CRR scale-pack so the 4 quadrant scales sit at byte
offsets that the MMA opsel selector can consume directly (no
`v_lshrrev_b32`), eliminating the 6 shifts/K-pair currently bottlenecking
CRR (R47/R48 cycle: CRR ceiling ~92% vs RCR ~95%).

## TL;DR — VERDICT: REFUTED

`MXFP8_CRR_OPSEL=1` regresses every prod cell. Geomean delta -14.16%
across 7 shapes; worst -34.78% (8192³), best -7.78% (70B Down). SNR + det
clean across all cells (kernel correct), so the regression is purely a
performance pessimization, not a correctness bug.

| Shape         | OFF med | ON med | Δ% | spread |
|---------------|--------:|-------:|---:|-------:|
| 8192³         | 2801.1  | 1826.8 | **-34.78%** | 78.61% |
| 8B Q/O 4096³  | 2285.6  | 2019.4 | -11.65% | 4.22% |
| 8B Gate/Up    | 2367.8  | 2134.2 | -9.87%  | 1.32% |
| 8B Down       | 2800.8  | 2475.4 | -11.62% | 0.40% |
| 70B Q/O       | 2742.7  | 2415.6 | -11.93% | 3.38% |
| 70B Gate/Up   | 2503.2  | 2305.5 | -7.90%  | 0.95% |
| 70B Down      | 2655.1  | 2448.7 | -7.78%  | 0.80% |

Geomean -14.16%. SHIP gate (geomean ≥ +1.5%, worst ≥ -2%) FAILS on both axes.

## Diagnosis

The opsel-friendly re-packing eliminates the 6 `v_lshrrev_b32`/K-pair (as
intended) but the resulting scale layout forces the compiler to *insert
extra `v_perm_b32` lane permutations* to feed the MMA. Net instruction
count rose, not fell. The 8192³ extreme regression (-34.78% with 78.61%
spread) suggests the new layout interacts pathologically with the SPI
dispatch pattern at this shape — likely an unintended VGPR-pressure
inflection causing occupancy-2 → occupancy-1 collapse on some waves.

## Outcome

No source landed. CRR opsel layout closed as a lever. CRR's structural
~92% ceiling remains; alternative levers for CRR (e.g. R49b RRR phase
split — also REFUTED, see r49b_findings.md) and CRR scale prefetch
lead-distance (proposed in r49c_findings.md §7.2) remain on the
candidate list.

## Files

- `r49a_bench.sh` — bench driver (5 runs/cell, 30s cooldown, 60s rebuild)
- `r49a_results/` — per-cell run logs + `SUMMARY.txt`
- `crr_mxfp8_exact_8wave_opsel_fastpath.inc` — opsel-friendly CRR variant
  (kept in tree for documentation; default-OFF via `MXFP8_CRR_OPSEL`)
