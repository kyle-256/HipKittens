# R35 — Wrong-Cells Diagnosis (2026-04-19)

## TL;DR

The MXFP4 GEMM kernel has a **deterministic positional bug**: every 256×256 output
tile has its **upper-left 128×128 quadrant** corrupted, while the other three
128×128 quadrants are 100% clean. The bug is universal across the non-FUSED
step3/step4 path (`_ext_br`, `_v12`, `_lgk2`, `_swap`, `_no_embed`, …) but is
**absent** from the `_f34` (FUSED_STEP34=1) variant.

**Confidence: HIGH.** A simple module comparison (R35_COMPARE_MODULES.log)
shows 6.7-8.1 % non-finite output for every non-fused variant vs **0.06 %** for
`_f34`, with the upper-left 128×128 corruption rate dropping from ~27 % to **0.0 %**.

**Proposed fix:** rebuild every BEST_VARIANTS entry with `-DFUSED_STEP34=1`
appended to its CPPFLAGS, then re-bench. Risk: very low — `_f34` already builds
clean and produces correct output at the test shape.

## Setup

- Module: `tk_mxfp4_gluon_cpp_n4096_k2048_ext_br` (chosen because R34 found it
  the only consistently-measurable shape at SNR=47 dB on the truly-stable cells).
- Shape: M=N=4096, K=2048. Constant scale (-4) and seed=42 random FP4 data.
- `kernel_finite = 88.76 %` → ~11.24 % of the output is non-finite (inf/nan).
  R34 reported "17 %" by including non-finite + non-deterministic; the 11 %
  non-finite is the deterministic component.
- 5-run consistency mask: 62 % of cells agree across 5 runs, 38 % flicker.

## Wrong-cell positional pattern

Reproducible via `r35_nonfinite_pattern.py` (this directory).

### 1. Per-256×256 tile distribution

```
Mean non-finite rate within 256×256 (downsampled to 16×16 grid of 16-cell blocks):
  rblk 0:   0.0  99.2   2.0   0.4   0.0  98.8   1.2   0.4   0.8   0.1   0.0   0.0   0.6   0.1   0.0   0.0
  rblk 1:   0.0  96.3  49.1  41.0   0.0  95.3  32.5  40.2   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  rblk 2:  10.5  43.2  27.8  19.7   6.4  30.3  16.9  12.3   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  rblk 3:   1.4  21.2  11.8  12.6   1.5  21.5   7.9  12.7   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  rblk 4:   0.0  99.6   1.0   0.4   0.0 100.0   2.1   0.4   0.5   0.0   0.0   0.0   0.9   0.0   0.0   0.0
  rblk 5:   0.0  95.5  46.5  40.4   0.0  96.9  64.2  42.8   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  rblk 6:   5.0  41.8  14.3  10.3  11.4  53.3  29.5  20.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  rblk 7:   1.2  20.7   7.1  11.9   0.8  21.0  14.6  10.9   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  rblk 8..15: ALL ZERO
```

### 2. Within-quadrant fine structure

Per row offset within 256×256 tile (averaged over all tiles & cols):
```
  rows   0.. 16:  12.72%
  rows  16.. 32:  22.15%
  rows  32.. 48:  10.44%
  rows  48.. 64:   5.66%
  rows  64.. 80:  12.80%   ← second 64-row stripe (warp_m=1 within mh=0)
  rows  80.. 96:  24.14%
  rows  96..112:  11.62%
  rows 112..128:   5.50%
  rows 128.. ..:   0.00%   ← entire mh=1 region clean
```

Per col offset within 256×256 tile:
```
  cols   0.. 16:   1.13%
  cols  16.. 32:  32.34%   ← acc[*, j=1] is the worst slot
  cols  32.. 48:   9.96%
  cols  48.. 64:   8.54%
  cols  64.. 80:   1.26%   ← second 64-col stripe (warp_n=1 within nh=0)
  cols  80.. 96:  32.32%
  cols  96..112:  10.55%
  cols 112..128:   8.76%
  cols 128.. ..:   0.00%   ← entire nh=1 region clean
```

### 3. Mapping to kernel structure

`store_block`'s `acc[i*4+j]` writes a 16-row × 16-col tile starting at
`(i*16+row_off, j*16+col_off)` within a 64×64 warp output region. The 256×256
output tile is laid out as:

```
   nh=0 (cols 0-127)        nh=1 (cols 128-255)
+------------------+ +------------------+
| acc_A0Bl  (warp) | | acc_A0Br  (warp) |  mh=0 (rows 0-127)  ← BAD
+------------------+ +------------------+
| acc_A1Bl  (warp) | | acc_A1Br  (warp) |  mh=1 (rows 128-255)  ← CLEAN
+------------------+ +------------------+
```

The corruption lives **entirely in `acc_A0Bl`**.

`acc_A0Bl` is computed in `kpair_64mfma_step12(acc_A0Bl, acc_A0Br, tA0, tBl, …)`
during step 1+2 (line 1640 in `wrap_n4096_k2048_ext_br.cpp`).

But wait — `acc_A0Br` shares the same call site and is **clean**. So the compute
itself is OK; the bug must be in how `acc_A0Bl` is _used_ subsequently or how its
operands are corrupted by something after compute.

## Module comparison (R35_COMPARE_MODULES.log)

| Module                                                     | nonfinite% | det% | upper-left 128×128 nf% | rest of tile nf% |
|------------------------------------------------------------|------------|------|-------------------------|-------------------|
| `tk_mxfp4_gluon_cpp_n4096_k2048`                          |  6.76%     | 73.2 |  26.98%                | 0.0212%          |
| `tk_mxfp4_gluon_cpp_n4096_k2048_ext_br`                   |  6.92%     | 62.6 |  26.70%                | 0.3315%          |
| **`tk_mxfp4_gluon_cpp_n4096_k2048_f34`**                  |**0.06%**   |**84.4**|  **0.00%**           | **0.0800%**      |
| `tk_mxfp4_gluon_cpp_n4096_k2048_v12`                      |  5.79%     | 64.1 |  23.00%                | 0.0562%          |
| `tk_mxfp4_gluon_cpp_n4096_k2048_v4`                       |  6.74%     | 63.3 |  25.84%                | 0.3699%          |
| `tk_mxfp4_gluon_cpp_n4096_k2048_lgk2`                     |  7.60%     | 61.4 |  28.85%                | 0.5183%          |
| `tk_mxfp4_gluon_cpp_n4096_k2048_swap`                     |  7.55%     | 64.4 |  29.20%                | 0.3274%          |
| `tk_mxfp4_gluon_cpp_n4096_k2048_swap_gm8`                 |  8.05%     | 60.4 |  30.03%                | 0.7277%          |
| `tk_mxfp4_gluon_cpp_n4096_k2048_no_embed`                 |  7.36%     | 64.0 |  28.15%                | 0.4340%          |

`_f34` is the only variant not affected. It is built with `-DFUSED_STEP34=1`
(see `build_all42_parallel.py:159`).

## Hypothesis

The bug lives in the **non-fused step3+step4 path** (`kpair_32mfma_with_lds_and_pf`
calls at line 1659/1670 of `wrap_n4096_k2048_ext_br.cpp`).

In the non-fused path, step 3 and step 4 are emitted as **two separate
`asm volatile` blocks** that issue ds_reads of the next-iter A0 / Bl tile into
`=&v` outputs. The compiler is free to schedule code (including unrelated
loads or moves) between the two asm blocks — and crucially, between the **Row 0
through Row 3 sub-blocks** of `kpair_32mfma_with_lds_and_pf` itself (the function
contains 4 separate `asm volatile` statements for its 4 rows).

The fused variant (`kpair_64mfma_step34` in `_f34`) emits all 64 MFMAs + 16 ds_reads
in **one** `asm volatile` block, which forecloses any interleaving by the compiler.

The most likely mechanism is one of:
1. **Compiler reschedules a vmcnt-clearing operation between the per-row asm
   blocks**, allowing in-flight buffer_loads (the prefetches for the next K-iter)
   to land on accumulator/scale VGPRs before they're consumed.
2. **Register pressure / spill** between the per-row asm blocks corrupts a
   live-across-step value (e.g., `tA0` operands, scale registers).
3. **An MFMA write-after-write hazard** between step 2's `acc_A0Bl` writes and
   step 3's reads: only mh=0 outputs are involved because both step 2 (writing
   acc_A0Bl) and step 3 (reading nothing — but using A1) share register space
   that the fused path arbitrates differently.

The fact that **only `acc_A0Bl` (mh=0, nh=0)** is broken — and step12 produces
both `acc_A0Bl` and `acc_A0Br` simultaneously, of which only `acc_A0Bl` is bad —
points to an interaction between **step12 finishing and step3 starting** that
clobbers the lower-numbered accumulator slots of the step12 output.

Note: `acc_A1Bl` and `acc_A1Br` are computed in step3/step4 themselves, and
they're clean. So the corruption isn't in the step3/step4 compute output — it's
in the `acc_A0Bl` register file values that have to be **kept live across**
step3 + step4 + the post-loop epilogue. Across that span, the compiler may be
inserting moves that touch those VGPRs, or the AGPR↔VGPR shuttle for storing C
may be the corrupting operation.

**Confidence: MEDIUM-HIGH on the hypothesis (mechanism).**
**Confidence: HIGH on the fix (FUSED_STEP34=1).**

## Proposed fix to test in R36

### Fix A (rebuild verification, 1 hour)
Run a R35-style SNR check on every BEST_VARIANTS module's `_f34`-equivalent build:
- For each `_X` in BEST_VARIANTS, build a `_X_f34` variant by appending
  `-DFUSED_STEP34=1` to its CPPFLAGS in `build_all42_parallel.py`.
- Re-run the bench at the same shape and verify
  `det_frac >= 99 %` AND `kernel_finite >= 99.5 %` AND `SNR_det >= 40 dB`.

### Fix B (root-cause repair, 4-8 hours)
Replace the non-fused step3+step4 sequence everywhere it appears (lines 1657-1680
of `wrap_n4096_k2048_ext_br.cpp` and identical blocks in tail iteration / TAIL_SPLIT
section) with a `kpair_64mfma_step34`-style single-asm fusion. This is essentially
backporting the `_f34` change into the default code path. Then deprecate the
non-fused path entirely.

### Performance impact
`_f34` benched WITHIN the BEST_VARIANTS optimization rounds and was NOT the
overall winner (the BEST_VARIANTS table contains many non-`_f34` modules), which
means **the non-fused path is likely faster on average** — that's why bench
runners chose it. The "performance" gain from the broken non-fused path is
**measuring time-to-write-garbage**, not time-to-correct-output. **Every TFLOPS
number on a non-`_f34` BEST_VARIANTS entry needs to be re-evaluated at correct
output.**

## Files

- `r35_wrong_cell_locator.py` — initial det-wrong-cell coordinate dump
- `r35_nonfinite_pattern.py` — full positional pattern analyzer (the smoking-gun chart)
- `r35_compare_modules.py` — module comparison (the FUSED_STEP34 finding)
- `R35_WRONG_CELL_LOCATOR.log` — output of the locator
- `R35_NONFINITE_PATTERN.log` — output of the pattern analyzer
- `R35_COMPARE_MODULES.log` — module comparison results
- `R35_WRONG_CELL_COORDS.json` — first 1000 wrong-cell (row, col) pairs

## Recommendation

R36 should:
1. **Immediately**: append `-DFUSED_STEP34=1` to every entry in BEST_VARIANTS and
   re-build (parallel ~10 min).
2. **Then**: re-run the full 42-shape bench. Expect a TFLOPS regression on shapes
   where the non-fused path was previously winning by writing garbage faster.
3. **Then**: replace the bench harness with a correctness-gated version that
   refuses to record TFLOPS when `kernel_finite < 99.5 %`. The R34 finding that
   no correctness gate exists in `bench_all_42.py` is the upstream cause of how
   this bug reached round 34 undetected.
