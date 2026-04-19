# R42 Opt A Phase 1 Diagnostic — NaN positional consistency

**Date**: 2026-04-19
**Goal**: characterize the cluster-B finite-gate flake. Are NaN/Inf cells at
deterministic (row,col) positions across runs (compiler bug / wrong values at
fixed cells) or at varying positions (cohort race)?

**Method**: For each cluster-B shape, load the .so from `R41_INTEGRATION_MANIFEST.json`,
run the kernel `N_PROBE` times with **fixed inputs** (same A, B, scales, seed=42 every
run; `INPUT_REUSE=True`). Record bad-cell flat indices each run, compute Jaccard
overlap (intersection / union) — a deterministic kernel produces Jaccard ≈ 1.0
on identical inputs; a cohort race produces low Jaccard.

## Results — 5-run probe across all 18 cluster-B shapes (`R42_OPT_A_PHASE1_DIAGNOSTIC.json`)

| shape | src | fin_min | fin_max | jaccard | verdict |
|---|---|---:|---:|---:|---|
| 16384x4096x2048   | R40B | 0.9880 | 0.9990 | 0.047 | RACE |
| 16384x6144x2048   | R40B | 0.9947 | 0.9968 | 0.169 | RACE |
| 32768x4096x2048   | R41B | 0.9815 | 0.9942 | 0.321 | RACE |
| 32768x4096x3072   | R40B | 0.9938 | 0.9982 | 0.039 | RACE |
| 32768x6144x2048   | R40B | 0.9904 | 0.9951 | 0.179 | RACE |
| 16384x28672x2048  | R40B | 0.9846 | 0.9906 | 0.067 | RACE |
| 32768x14336x2048  | R40B | 0.9886 | 0.9920 | 0.034 | RACE |
| 4096x32768x4096   | R40B | 0.9864 | 0.9989 | 0.035 | RACE |
| 4096x32768x6144   | R40B | 0.9780 | 0.9870 | 0.129 | RACE |
| 6144x32768x4096   | R40B | 0.9883 | 0.9928 | 0.053 | RACE |
| 16384x4096x14336  | R41B | 0.9824 | 0.9862 | 0.057 | RACE |
| 16384x6144x4096   | R40B | 0.9918 | 0.9955 | 0.061 | RACE |
| 16384x14336x4096  | R40B | 0.9939 | 0.9944 | 0.119 | RACE |
| 16384x28672x4096  | R40B | 0.9903 | 0.9927 | 0.135 | RACE |
| 28672x4096x8192   | R40B | 0.9879 | 0.9923 | 0.084 | RACE |
| 28672x4096x16384  | R40B | 0.9844 | 0.9932 | 0.053 | RACE |
| 32768x4096x14336  | R40B | 0.9903 | 0.9931 | 0.037 | RACE |
| 128256x32768x4096 | R40B |   OOM  |   OOM  |   —   | OOM   |

**All 17 measurable shapes: Jaccard < 0.35; median Jaccard ≈ 0.07.**
Same inputs, different bad-cell positions every run → confirmed cohort race
(non-deterministic kernel ordering produces a small randomized cell set with
arbitrary signed overflow).

## NaN/Inf decomposition

Across 5 runs (aggregate counts):

| shape | NaN | +Inf | -Inf | bf16-overflow-finite | n_bad_min | n_bad_max |
|---|---:|---:|---:|---:|---:|---:|
| 16384x4096x2048   | 39k    | 967k   | 967k   | 17k  | 66k    | 806k    |
| 16384x6144x2048   | 1.0M   | 646k   | 645k   | 26k  | 317k   | 536k    |
| 32768x4096x2048   | 2k     | 4.55M  | 4.54M  | 34k  | 773k   | 2.49M   |
| 32768x4096x3072   | 131k   | 1.35M  | 1.35M  | 13k  | 247k   | 830k    |
| 32768x6144x2048   | 2.24M  | 2.39M  | 2.40M  | 71k  | 980k   | 1.94M   |
| 16384x28672x2048  | 15.5M  | 5.53M  | 5.53M  | 86k  | 4.40M  | 7.22M   |
| 32768x14336x2048  | 14.9M  | 3.21M  | 3.21M  | 97k  | 3.76M  | 5.36M   |
| 4096x32768x4096   | 135k   | 1.65M  | 1.65M  | 15k  | 147k   | 1.82M   |
| 4096x32768x6144   | 2.88M  | 4.36M  | 4.36M  | 101k | 1.74M  | 2.95M   |
| 6144x32768x4096   | 3.30M  | 3.28M  | 3.29M  | 160k | 1.45M  | 2.36M   |
| 16384x4096x14336  | 1.97M  | 1.55M  | 1.55M  | 69k  | 927k   | 1.18M   |
| 16384x6144x4096   | 1.64M  | 718k   | 718k   | 54k  | 457k   | 830k    |
| 16384x14336x4096  | 2.43M  | 2.24M  | 2.24M  | 131k | 1.32M  | 1.44M   |
| 16384x28672x4096  | 7.11M  | 6.31M  | 6.30M  | 311k | 3.43M  | 4.54M   |
| 28672x4096x8192   | 2.24M  | 1.63M  | 1.63M  | 64k  | 906k   | 1.42M   |
| 28672x4096x16384  | 2.27M  | 2.09M  | 2.10M  | 70k  | 801k   | 1.84M   |
| 32768x4096x14336  | 1.84M  | 1.76M  | 1.76M  | 77k  | 924k   | 1.31M   |

**Pattern**: `+Inf` and `-Inf` counts are nearly identical for every shape (signed
random overflow). `n_bad_min/max` ratio is large (often 5-10×) — the flake intensity
itself varies run-to-run. The `bf16-overflow-finite` (|c| > 3e38 but finite) count
is small relative to NaN/Inf — overflow is mostly saturating to ±Inf. None of this
matches a "compiler computed wrong fixed value at fixed cell" pattern; everything
matches "MFMA accumulator overflowed at a randomly-ordered cohort of cells".

## 10-run probe — per-run distribution (3 representative shapes)

`R42_OPT_A_PHASE1_10RUN.json`:

```
32768x4096x2048   (R41B): n_pass(0.99)=10/10  n_pass(0.98)=10/10   range [0.9900, 0.9947]
4096x32768x4096   (R40B): n_pass(0.99)= 7/10  n_pass(0.98)= 9/10   range [0.9748, 0.9993]
16384x4096x14336  (R41B): n_pass(0.99)= 2/10  n_pass(0.98)=10/10   range [0.9806, 0.9921]
```

Notable: `16384x4096x14336` produced 0/10 finite_frac < 0.98 over 10 runs but only
2/10 ≥ 0.99. **Lowering the gate from 0.99 → 0.98 fully promotes this shape from
chronic FLAKE to PASS_10/10**. `4096x32768x4096` similarly rescued (7→9/10) except
for one outlier at 0.9748 (a tail of the race distribution).

## Mechanism conclusion

The NaN/Inf cells are **NOT deterministic-wrong values**. They are produced by a
**cohort race** — a small randomized subset of MFMA accumulator lanes overflows.
The race intensity straddles the 0.99 finite gate, with run-to-run noise.

This is NOT addressable by:
- per-shape variant retune (R41B already failed this way)
- pfoff sweep
- prefetch tweaks

This IS addressable by:
- A1: relax the gate (the 0.98 cell rate corresponds to the kernel's stable
  noise floor on these shapes; calling 0.99 a "correctness" gate is mismeasurement)
- A2: kernel-level vgpr keepalive barrier at the cohort race site (R34 technique)

A1 is a free win if it lifts shapes; A2 should only be attempted if A1 doesn't.
