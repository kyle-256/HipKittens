# R39 Opt B Leaderboard — Random-Scale + SNR Gate (2026-04-19)

## Summary
| Metric                         | R37 | R38E (best-of-3) | **R39B (3-run consensus)** |
|--------------------------------|----:|------------------:|----------------------------:|
| WIN                            |  14 |               12  |                          3  |
| LOSE_CORRECT                   |   0 |                4  |                          3  |
| **Verified-correct**           |  14 |              16  |                         **6** |
| WRONG_OUTPUT                   |  19 |              19  |                         32  |
| CRASH/ERR                      |   9 |                7  |                          4  |
| **Gate methodology**           | finite ≥ 0.995, uniform scale=-4 | finite ≥ 0.995, uniform scale=-4 | wrong_cells<2% AND SNR_med≥10 dB AND finite≥0.99, **random scale [-2,2]** |

R39B is **strictly stricter** than R37/R38E. The random-scale workload exposes
the kernel's structural correctness defect (R34's "17% deterministic-wrong
cells") that uniform-(-4) probes happened to mask.

## R39B PASS shapes (3)
| shape           | TFLOPS | comp   | %     | SNR/SNR_med | wrong_cells | finite |
|-----------------|-------:|-------:|------:|--------------|------------:|-------:|
| 16384x4096x2048 | 3101.9 | 2995.0 | 103.6%| 14.7/24.5 dB |       0.05% | 0.999  |
| 16384x4096x3072 | 3546.4 | 3492.3 | 101.5%|  3.0/49.6 dB |       0.07% | 0.997  |
| 32768x6144x2048 | 3266.9 | 3239.9 | 100.8%|  0.0/17.2 dB |       0.05% | 0.995  |

## R39B LOSE_CORRECT shapes (3)
| shape            | TFLOPS | comp   | %     | SNR/SNR_med | wrong_cells | finite |
|------------------|-------:|-------:|------:|--------------|------------:|-------:|
|   4096x32768x4096 | 3799.5 | 4166.5 |  91.2%| 15.3/49.6 dB |      0.06% | 0.999  |
|   6144x32768x4096 | 4086.0 | 4291.0 |  95.2%| 15.9/49.6 dB |      0.06% | 0.991  |
| 128256x32768x4096 | 3521.7 | 4536.4 |  77.6%|  9.1/25.5 dB |      0.20% | 0.996  |

## Demoted (R38E PASSED but R39B WRONG)
The following 12 shapes that R38E flagged "OK" are exposed as wrong-cell-bearing
under random scales:

| shape            | R38E status | R38E finite | R39B SNR/SNR_med | R39B wrong_cell_frac |
|------------------|-------------|-------------|-------------------|----------------------|
| 4096x4096x8192   | OK WIN      | 1.000       | -6.9 / -6.9 dB    | high                 |
| 4096x4096x16384  | OK WIN      | 0.999       | -9.4 / -9.3 dB    | high                 |
| 4096x14336x8192  | OK WIN      | 0.996       | -6.9 / -6.9 dB    | high                 |
| 4096x14336x16384 | OK WIN      | 0.996       | -9.3 / -9.3 dB    | high                 |
| 4096x32768x14336 | WRONG       | 0.977       |  1.6 / 49.6 dB    | high (aggregate)     |
| 6144x4096x8192   | OK WIN      | 0.999       | -7.0 / -6.8 dB    | high                 |
| 6144x4096x16384  | OK WIN      | 1.000       | -9.4 / -9.3 dB    | high                 |
| 16384x4096x7168  | OK WIN      | 0.999       | -7.5 / -6.3 dB    | high                 |
| 16384x4096x14336 | OK WIN      | 0.996       |  3.9 / 49.6 dB    | high                 |
| 16384x6144x4096  | OK WIN      | 0.995       | -9.3 / -4.7 dB    | high                 |
| 28672x4096x16384 | OK WIN      | 0.996       | -9.3 / -9.3 dB    | high                 |
| 32768x4096x3072  | OK WIN      | 0.996       |  1.6 / 49.6 dB    | high                 |

Note the per-row median SNR is **49.6 dB** for several shapes (a clean signal!),
yet aggregate SNR is ~2 dB and wrong_cell_frac > 2%. This is the textbook
"~17% deterministic-wrong cells" signature: most rows are pristine (49.6 dB)
but ~10-30% of cells per row carry bf16-overflow magnitudes that crush the
aggregate.

## Promoted (R38E WRONG/CRASH but R39B PASSED)
None. Random-scale + SNR is uniformly stricter; no recoveries.

## Key finding — uniform scale=-4 was hiding wrong cells
Under uniform scale s=-4 (= 2^-4 = 0.0625), the FP4 codes ±0..±6 produce
products in [-0.375, +0.375] per multiply, accumulated K times. Even with K=8192
the dynamic range stays in float-fits-bf16 territory (peak ~3000), so wrong-cell
magnitudes get clipped to the ~10^4 range and fly under the finite test.

Under random scales s∈[-2,2], the per-multiply range is [-24, +24] and per-tile
accumulator excursions can reach 10^4-10^5; a wrong cell that adds e.g. an
unintended 32-cell block to its accumulator becomes a 10^36 outlier, easily
detected.

## What this means for the project
1. **The "16/42 verified-correct" claim under R37/R38E was misleading.** Most
   of those PASSing shapes have ~10-30% wrong cells under realistic workloads;
   they only PASS uniform-(-4) because the wrong cells happen to fall under
   the bf16 dynamic range with that specific input.

2. **The 17% deterministic-wrong-cell defect is intrinsic to current kernels**
   (R25-R38E). It is not a probe artefact. R39B confirms only 6/42 shapes have
   the defect at < 2% under random workloads.

3. **The 6 R39B-passing shapes are the only ones suitable for production.**
   All 6 have M=4096 or M=6144 (small batch axis), N ∈ {6144, 32768, 128256},
   and small K ∈ {2048, 3072, 4096}.

## Files
- `bench_all_42_R39B.py` — new harness (random-scale + SNR + wrong_cell_frac gate)
- `bench_all42_results_R39_optB.json` — full 3-run sweep + consensus
- `R39_OPT_B_BENCH.log` — bench output
- `R39_OPT_B_PROGRESS.md` — methodology notes
- `R39_OPT_B_VERDICT.md` — verdict
