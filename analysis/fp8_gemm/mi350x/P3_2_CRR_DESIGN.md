# P3.2 CRR var_k Operand Layout Rewrite (R65)

## Current CRR var_k Spill Status

| Variant | V | A | spill | scratch |
|---------|---|---|-------|---------|
| CRR var_k fp8 | 256 | 4 | 32 | 132B (some variants 41/168) |

bf16 CRR var_k same topology: spill=1 only. **Spill gap is operand-layout-driven**, not algorithmic.

## bf16 Reference Path

bf16 uses:
- `ST_v2a` swizzle (alternative pattern, see kernel_fp8_layouts.cpp:1471 / 4029)
- Fragment types `A_col_reg` / `B_col_reg` with col-layout 128×32 / 128×16
- Same `crr_mma_agpr_inplace` wrapper (from `[[fp8-crr-agpr-inplace]]` win 2026-05-16)
- Spill = 0-1 on all 33 bf16 instantiations

fp8 var_k uses identical body topology but spills 32-41. The delta = fp8 operand layout (16x128 vs bf16 alignment).

## P3.2 Plan (~300-500 LOC, 3-4 sessions)

### Session 1: Operand layout audit (~50 LOC analysis)
- Diff bf16 vs fp8 CRR var_k body line-by-line
- Identify exact spill sites via amdhsa.kernels per-symbol metadata
- Compare A_col_reg / B_col_reg storage layout

### Session 2: ST_v2a alignment for fp8 (~150 LOC)
- Already uses ST_v2a for A (line 4029). Confirm B uses ST_v2 — maybe should also use ST_v2a
- Test swizzle change isolated

### Session 3: Fragment type alignment (~200 LOC)
- Match bf16 fragment alignment exactly
- May need new `rt_fp8e4m3<128, 16, col_l, rt_128x16_8_s>` (different alignment factor)
- Validate via amdhsa.kernels probe

### Session 4: Bench + cleanup (~50 LOC)
- 12 var_k shapes vs Triton, target wgrad 1.79× geomean (current 1.556)

## Expected Win

- spill 32-41 → 0-1 (parity with bf16)
- ~10% perf gain from releasing occupancy + scratch elimination
- Triton geomean 1.556× → ~1.7-1.8× (close to +15% target)

## Risk

- Same risk as P1.2/P2.2: fragment layout change cascades through all use sites
- bf16 path is reference but bf16 mfma is different shape (16×16×32 not 16×16×128) — alignment math differs
