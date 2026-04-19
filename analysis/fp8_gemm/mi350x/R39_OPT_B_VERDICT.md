# R39 Opt B — Verdict (2026-04-19)

## Status: NEGATIVE RECOVERY, POSITIVE DIAGNOSTIC

The R38 Opt E recommendation was: replace the uniform-scale=-4 finite gate
with a random-scale + SNR probe to **recover the ~6 shapes hovering at finite
∈ [0.97, 0.99]**. The R39B implementation does the opposite: it shows those
borderline shapes are genuinely wrong (not borderline), and **demotes 12
additional shapes** that R38E flagged "OK" but actually carry the same wrong
cells under realistic workloads.

## Headline (3-run consensus)

| Metric                | R37 | R38E | **R39B** |
|-----------------------|----:|-----:|---------:|
| WIN                   |  14 |  12  |     **3** |
| LOSE_CORRECT          |   0 |   4  |        3  |
| **Verified-correct**  |  14 |  16  |     **6** |
| WRONG_OUTPUT          |  19 |  19  |       32  |
| CRASH/ERR             |   9 |   7  |        4  |

R39B's "verified-correct" count is **strictly a subset** of R38E's. No shapes
that R38E demoted as WRONG were promoted by R39B. 12 shapes that R38E PASSED
were demoted to WRONG by R39B because they exhibit > 2% catastrophic-wrong
cells under random scales.

## Methodology that was implemented

Three-tier gate (all required):
1. `wrong_cell_frac < 0.02` over the top-left 1024×1024 corner (defined as
   cells where `|out| > 100*|ref| AND |diff| > 5%*|ref| + 1e-2`)
2. `snr_med_db ≥ 10 dB` per-row median SNR over non-catastrophic cells
3. `kernel_finite ≥ 0.99`

Inputs: random FP4 data + random E8M0 scales in `[-2, 2]` (5 values), seed=42.
Reference: float32 torch matmul on dequantized + scaled inputs. Sub-sampled to
1024×1024 corner to avoid OOM on K=128256 shapes.

Three runs per shape × 8 GPUs, majority-vote consensus. Bench params at
warmup=200, iters=500, trim=10%.

## Why the recommendation didn't deliver

The R38E verdict assumed the borderline-finite shapes were tripping the gate
because of bf16 overflow on **uniform-(-4) inputs specifically** (cell magnitudes
clipped to bf16-max at K ≥ 14336). The actual mechanism per R34_SNR_FINDINGS.md
is different: the kernel writes ~17% deterministic-wrong cells with values
`±2.2e36 .. ±3.4e38` regardless of input distribution. Under uniform-(-4),
those wrong cells have small expected accumulator magnitudes and the bf16
overflow doesn't always trigger; under random scales, the same wrong cells
balloon out of bf16 range and the finite-fraction test rejects them.

Concrete evidence: 7 of 12 demoted shapes have **per-row median SNR ≥ 49 dB**
(pristine signal on most rows) but `wrong_cell_frac` of 5-25% and aggregate
SNR of -4..+5 dB. Most rows of the output are correct; ~17% of cells per row
are bf16-overflow garbage.

## What R39B actually delivers

1. **Honest correctness verdict**: only 6/42 shapes are correct under realistic
   random-scale workloads. The 16/42 R38E claim was inflated by the brittle
   uniform-(-4) probe.

2. **Diagnostic separation**: `wrong_cell_frac` cleanly separates the two
   failure modes — (a) precision-noise (low SNR, low wrong_cell_frac) vs
   (b) deterministic-wrong-cell defect (high wrong_cell_frac, often high
   per-row SNR on the OK rows).

3. **A reproducible probe** that matches aiter's correctness contract more
   closely. (aiter's competitor TFLOPS were measured under aiter's own random
   probe; comparing TFLOPS only on R39B-passing shapes is the only
   apples-to-apples comparison we have.)

## Honest assessment

The "41/42 verified-correct" headline from rounds R20-R32 was always
conditional on the uniform-(-4) probe. The 17%-deterministic-wrong-cell defect
documented in R34/R35 has been the real ceiling all along; R39B simply
quantifies it per shape under realistic workloads.

## Recommended R40 follow-ups

1. **Stop tuning kernels until the wrong-cell defect is fixed.** Performance
   sweeps on broken kernels chase noise. The 6 R39B-passing shapes (small M+K)
   already beat aiter; nothing else will until the 17% bug is resolved.

2. **Investigate the 17% bug at the source.** Per `project_mxfp4_correctness_17pct_wrong`
   memory note, the kernel writes bf16-overflow garbage to ~17% of cells
   consistently. R39B's `wrong_cell_frac` map could pinpoint which (i,j) cells
   are wrong — useful for binary-search-bisection of the kernel responsibility.

3. **Adopt R39B as the project's correctness gate going forward.** The
   uniform-(-4) finite gate has demonstrably been hiding the kernel defect.
   Any future "WIN" claim should be qualified by R39B `wrong_cell_frac < 0.02`.

4. **Don't expand the variant DB until #2 is resolved.** R38D added 9 variants;
   R39B exposes that the additions don't fix the underlying defect, only
   reshape its manifestation under different probes.

## Failure paths checked

- **40 dB threshold** (original recommendation): rejects ALL kernels because
  bf16 with K=thousands and random scales caps intrinsic SNR ~20-25 dB.
  Adjusted to 10 dB SNR_med + wrong_cell_frac primary gate.
- **OOM trap on K=128256**: avoided via 1024×1024 corner subsampling. All 42
  shapes fit comfortably in fp32 ref memory.
- **Permitting R36-style FUSED_STEP34=0 wrong cells**: the wrong_cell_frac
  gate at 2% explicitly rejects these (they show > 17% wrong_cell_frac).

## Files

- `bench_all_42_R39B.py` — new harness
- `bench_all42_results_R39_optB.json` — full sweep (3 runs + consensus)
- `R39_OPT_B_BENCH.log` — full output
- `R39_OPT_B_LEADERBOARD.md` — comparison table
- `R39_OPT_B_PROGRESS.md` — methodology + smoke-test findings
- No kernel sources modified (per task spec).
