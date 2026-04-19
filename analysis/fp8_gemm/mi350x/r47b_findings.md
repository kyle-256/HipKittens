# R47 Dev B: CRR XCD-aware block swizzle

## Summary

Ported R46 Dev D's RRR XCD-aware chiplet + grouped-M block swizzle to the MXFP8
**CRR** exact 8-wave fastpath kernel (`crr_exact_8wave_scaled_kernel`).
Macro-gated `MXFP8_CRR_BLOCK_SWIZZLE` — **default ON** based on net-positive
3-run-averaged sweep across all 7 compute-bound shapes.

## Motivation

R46 closed CRR cells at 85-95% MX/FP8 with worst at:
- 70B Down  CRR = 85.3% (4096 × 8192 × 28672)
- 70B Gate/Up CRR = 86.3% (4096 × 28672 × 8192)

Both shapes have large N (28672) or large K (28672). Per R46 Dev A's profiling,
these regimes are penalized by L2 scale cache pressure (large N → large B-scale
working set) and VMEM contention (large K → many scale-load pairs).

R46 Dev D demonstrated that XCD-aware block swizzle on RRR closed +6.51% on
70B Gate/Up and +2.99% on 70B Down by interleaving WG dispatch so adjacent WGs
land on the same XCD and share L2-resident B-tiles. CRR uses the same row-major
`bid → (br, bc)` mapping, so the same lever should apply.

## Implementation

`analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc`:

1. **Three new macros** (default `MXFP8_CRR_BLOCK_SWIZZLE=1`,
   `MXFP8_CRR_BLOCK_SWIZZLE_NUM_XCDS=8`, `MXFP8_CRR_BLOCK_SWIZZLE_GROUP_M=4`).
2. **Constexpr `swizzle_blocks_per_row/col`** from compile-time `M_DIM/N_DIM`
   macros (only used in swizzle path; runtime `blocks_per_col` retained for
   the rest of the kernel since other call-sites use `g.n / BLK`).
3. **Chiplet swizzle** (8 XCDs on MI355X):
   ```
   bid' = (bid % 8) * (num_wgs / 8) + (bid / 8)
   ```
4. **Grouped-M swizzle** (group of 4 M-tiles share B-tiles):
   ```
   group_id = bid' / (4 * blocks_per_col)
   br = group_id*4 + (bid' % (4*blocks_per_col)) % group_size_m
   bc = (bid' % (4*blocks_per_col)) / group_size_m
   ```

The swizzle is identical to RRR (same gate guard `num_wgs % 8 == 0`).

## Results — 3-run averaged sweep

GPU 1, MXFP8_WARMUP=50 ITERS=100 LAYOUTS=crr PRESHUFFLE_QUANT=1.

| Shape           | Baseline (TF) | +Swizzle (TF) |    Δ%      |
|-----------------|--------------:|--------------:|-----------:|
| 8192³           |      2777.6   |      2776.7   |    -0.03%  |
| 8B Q/O          |      2111.1   |      2168.1   |   **+2.70% ★** |
| 8B Gate/Up      |      2326.0   |      2326.3   |    +0.01%  |
| 8B Down         |      2714.5   |      2745.3   |   **+1.13% ★** |
| 70B Q/O         |      2699.8   |      2672.3   |    -1.02%  |
| **70B Gate/Up** |      2401.5   |      2454.6   |   **+2.21% ★** |
| **70B Down**    |      2559.3   |      2645.0   |   **+3.35% ★** |

Per-run data: see `r47b_crr_swizzle_results/3run_variance.txt`.

**Net positive**: 4 shapes ≥ +1.0% (★), 3 shapes within ±1.1%.
Worst -1.02% (70B Q/O) is within the R44-established within-GPU variance
envelope (±2%). The largest gain (+3.35%) is on exactly the worst CRR cell
(70B Down was 85.3% MX/FP8).

## Correctness + Determinism (SHIP gate)

70B Down CRR with swizzle ON (4096×8192×28672):
- SNR: 49.60 dB (≥ 45 dB SHIP gate) **PASS**
- Determinism: 3/3 runs byte-identical **PASS**

8192³ CRR with swizzle ON:
- SNR: 49.60 dB **PASS**
- Determinism: 3/3 runs byte-identical **PASS**

## Comparison to RRR (R46 Dev D)

| Shape           | RRR Δ% (R46d) | CRR Δ% (R47b) |
|-----------------|--------------:|--------------:|
| 8192³           |    -0.25%     |    -0.03%     |
| 8B Q/O          |    +1.43%     |    +2.70%     |
| 8B Gate/Up      |    -0.74%     |    +0.01%     |
| 8B Down         |    +1.14%     |    +1.13%     |
| 70B Q/O         |    +0.19%     |    -1.02%     |
| 70B Gate/Up     |    +6.51%     |    +2.21%     |
| 70B Down        |    +2.99%     |    +3.35%     |

CRR sees larger gains on 8B Q/O and 70B Down, smaller on 70B Gate/Up. Plausible
explanation: CRR's A-transpose load path means adjacent br share cache lines
across the K-dimension differently than RRR. The grouped-M swizzle still helps
B-tile reuse across the M-group, but A-tile reuse pattern is altered. Net win
remains positive on the worst cells.

## Hot/cold variance note

Initial single-shot sweep showed -2.41% on 8192³ and -1.77% on 70B Q/O. Both
were re-bench'd at 3 runs and resolved to within ±1% (variance / GPU contention
from concurrent sibling agents on other GPUs caused the worst-case cold-run
outliers). This matches R44's established within-GPU envelope of ±2%.

## R47 Updated CRR baseline (estimated MX/FP8 ratios)

Assuming FP8 baseline TFLOPS unchanged from R46:

| Shape       | Old MX/FP8 % | New MX/FP8 % (est) |
|-------------|-------------:|-------------------:|
| 8192³       |    94.3%     |    94.3% (~flat)   |
| 8B Q/O      |    88.4%     |    90.8%           |
| 8B Gate/Up  |    91.4%     |    91.4% (flat)    |
| 8B Down     |    94.6%     |    95.7% (PASS)    |
| 70B Q/O     |    94.9%     |    93.9%           |
| 70B Gate/Up |    86.3%     |    88.2%           |
| **70B Down**|    **85.3%** |    **88.2%**       |

Expected to graduate **8B Down CRR** to PASS (95.7% > 95%). The two worst CRR
cells (70B Gate/Up + 70B Down) still below 95% but materially closer.

## Files modified

- `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc`:
  - Lines 54-67: 3 new macros (default ON for `MXFP8_CRR_BLOCK_SWIZZLE`)
  - Lines 286-290: constexpr `swizzle_blocks_per_row/col`
  - Lines 300-329: bid swizzle + grouped-M tile mapping (replaces 3-line baseline)

## Files added

- `analysis/fp8_gemm/mi350x/r47b_crr_swizzle_sweep.sh` — 7-shape baseline-vs-swizzle
- `analysis/fp8_gemm/mi350x/r47b_crr_swizzle_results/` — 14 sweep logs + build logs
- `analysis/fp8_gemm/mi350x/r47b_crr_swizzle_results/3run_variance.txt` — 42-run validation

## R47 cumulative levers update

R46 cumulative: 46 closed levers, 4 refutations.
R47 Dev B adds: **+1 closed**: CRR XCD-aware block swizzle (+3.35% on 70B Down).
