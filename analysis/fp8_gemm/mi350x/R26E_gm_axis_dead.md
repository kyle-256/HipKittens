# R26-E GROUP_SIZE_M sweep — DEAD

**Date:** 2026-04-18
**GPU:** MI355X #4 (isolated, idle)
**Bench params:** warmup=200, iters=500, trim=10%
**Builds:** 12 (4 shapes × 3 gm values: 5, 7, 9)
**Result file:** `bench_r26e_gm_sweep_gpu4.json`, log `bench_r26e_gpu4.log`

## Hypothesis tested

R25-F/G/H all wire `GROUP_SIZE_M ∈ {6,7}`. The M-block-size axis was never
explored at odd values (5, 9). Hypothesis: a small swing (gm5 or gm9) might
yield +1-3pp on M-bound shapes whose M dimension has a more favorable
divisibility / persistent-XCD-remap fit.

## Results

| Shape | M×N×K              | comp T | gm5    | gm7 (cur) | gm9    | best Δ vs gm7 |
|-------|--------------------|--------|--------|-----------|--------|---------------|
| DLA2  | 128256×32768×4096  | 4536.4 | 4885.2 | **4892.6**| 4877.8 | -0.30%        |
| SD    | 16384×4096×28672   | 5525.3 | **6308.4** | 6292.8 | 6269.9 | +0.25%   |
| SC    | 4096×32768×6144    | 4548.6 | **5388.4** | 5376.7 | 5384.3 | +0.22%   |
| SE    | 28672×4096×16384   | 5350.6 | **6084.9** | 6079.8 | 6078.5 | +0.08%   |

All three SD/SC/SE swings to gm5 are **below the +1.5pp WIN threshold** —
they are within run-to-run noise (std ≈ 0.001-0.04 ms across configs).
DLA2 strictly prefers gm7. **No commits.**

## Verdict: DEAD

The R25-F/G/H per-shape gm choice (gm6 or gm7) is at or within noise of the
local optimum on the gm axis for these 4 representative shapes.

## Hypothesis why

1. **gm7 is already a local optimum**: the auto-tuner that originally selected
   gm7 (R25-F/G/H) plus the persistent-XCD scheduler interaction puts gm7 inside
   a flat basin. gm5 and gm9 land in the same basin (within 0.3%).
2. **Persistent-XCD wave packing**: the active wave count per XCD on MI355X
   appears insensitive to gm in the {5..9} range for these M dimensions
   (M ∈ {4096, 16384, 28672, 128256}). Each gm value still produces a near-
   identical fill of the 32 XCDs.
3. **L2 reuse on B-tile is the binding constraint**, not M-block locality —
   consistent with the R25-F/G mechanism note that "B-tile is L2-resident
   after 2 iters via persistent-XCD remap." Once L2-resident, gm choice
   barely shifts the work geometry.

## Files / artifacts

- `bench_r26e_gm_sweep.py` — sweep driver
- `bench_r26e_gpu4.log` — full output
- `bench_r26e_gm_sweep_gpu4.json` — TFLOPS table
- `build_r26e/` — 12 .so files (kept for re-verify if needed)

## Implications for future rounds

- **Skip** systematic gm-axis sweeps on R25-G/H wires. The gm dimension is exhausted.
- **Don't** test gm10 — by extrapolation it would underperform gm9 on these shapes.
- **Productive next axes** (untested by R26-E):
  - Per-shape `STEP3_BARRIER_VMCNT` finetune around the current 12 (try 8, 10, 14, 16)
  - Cross-product of `STEP12_BR_LGKMCNT` ∈ {1, 3} with current pfoff wires
  - SCALE_PACK / KPAIR_LOOP variants gated by R25C_K_EXACT
