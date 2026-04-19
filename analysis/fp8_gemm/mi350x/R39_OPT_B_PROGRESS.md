# R39 Opt B — Progress (2026-04-19)

## Mission
Replace R37/R38E's brittle uniform-scale=-4 finite-fraction gate with a
random-scale + SNR-aware probe that matches the kernel's actual workload, then
re-bench R38E binaries through the new harness.

## Methodology adopted
- **Random FP4 data** (uniform over 16 codes per nibble), `seed=42`.
- **Random E8M0 scales** in int8 range `[-2, 2]` (5 values) — same distribution
  used in `snr_diag_random.py` rand_-2_3 mode that historically hit 47 dB at
  M=N=4096, K=2048 with the `ext_br` reference kernel.
- **Float32 torch reference** built per shape: `dequant_fp4(A) * 2^sa[:,None]`
  multiplied by `(dequant_fp4(B) * 2^sb[:,None])^T`.
- **Sub-sampled SNR**: only the top-left `1024 x 1024` corner of the output is
  compared, which means the float32 reference is bounded to ~1024×1024×4 = 4 MB
  and the K-dimension chunking is implicit (single matmul). This bypasses the
  R34 OOM trap on K=128256 shapes.
- **Three-tier gate** (all must pass):
  1. `wrong_cell_frac < 0.02` — fraction of "catastrophically wrong" cells
     (defined as cells where `|out| > 100 * |ref| AND |diff| > 5% * |ref| + 1e-2`).
     This isolates the R34-documented "17% deterministic-wrong cells" tier
     (cells the kernel writes with bf16-overflow magnitudes).
  2. `snr_med_db >= 10 dB` — per-row median SNR over non-catastrophic cells.
     bf16 with K=thousands and random scales caps intrinsic SNR around 20-25 dB.
  3. `kernel_finite >= 0.99` — same finite gate as R37 (relaxed from 0.995
     since random scales naturally produce some non-finite cells).
- **3 runs per shape × 8 GPUs**, majority vote consensus.

## Critical finding before full sweep
Smoke-tested 5 representative shapes:

| Shape           | R38E verdict   | R39B SNR/SNR_med | wrong_cell_frac | R39B verdict |
|-----------------|----------------|------------------|------------------|--------------|
| 16384x4096x2048 | WIN            | 14.6 / 24.4 dB   | 0.05%            | WIN          |
| 16384x4096x4096 | WRONG (f=0.98) | -7.1 / -4.7 dB   | high             | WRONG        |
| 16384x4096x6144 | WRONG (f=0.99) | -6.0 / -5.2 dB   | high             | WRONG        |
| 4096x4096x8192  | WIN  (f=1.00)  | -6.9 / -6.9 dB   | high             | **WRONG**    |
| 4096x4096x32768 | WRONG (f=0.73) | -12.9/ -12.6 dB  | very high        | WRONG        |

**Key finding**: The (4096x4096x8192) case is `finite=1.0` under uniform-(-4)
(R38E PASSED it as WIN) but **fails the random-scale SNR gate**. This means
R37/R38E's `finite>=0.99` test was hiding ~17%-deterministic-wrong cells under
uniform-(-4) inputs precisely as predicted by R34_SNR_FINDINGS.md. Under random
scales, the wrong-cell magnitudes balloon out of bf16 range and become visible.

## Implication
The R39B random-scale + SNR gate is **stricter, not laxer**, than R38E's
uniform-(-4) finite gate. It does NOT recover the borderline shapes the verdict
predicted; instead it exposes ADDITIONAL wrong-cell shapes that uniform-(-4)
masked. The R34 finding "the kernel has structural correctness issues" is now
quantified per-shape:
- bf16-overflow on random scales is genuine, not a probe artefact.
- Uniform-(-4) probes are too kind; the kernel's wrong cells happen to fall in
  bf16 range when all scales are -4.

## Sweep status
3-run sweep launched on GPUs 0-7. ETA ~10 minutes (5x normal due to per-shape
SNR computation overhead). Results in `bench_all42_results_R39_optB.json`.
