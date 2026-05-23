# P2.2 RRR 32×32 Rewrite Design (R64 — multi-session handoff)

## Current RRR Production Spill Status

| Variant | V | A | spill | scratch |
|---------|---|---|-------|---------|
| RRR FUSED=false (BN=256) | 256 | 256 | 37 | 152B |
| RRR FUSED=true (BN=256) | 256 | 256 | **67** | **272B** |
| RRR BN=128 | <240 | 0-128 | 0 | 0 |

RRR FUSED=true is the **worst spill in the entire codebase** (67 VGPR / 272B scratch). gpt_oss-style shapes with K_rem=64 hit this path.

## RRR vs RCR Key Difference

RRR uses:
- `B_col_reg` = `rt_fp8e4m3<BK=128, RBN=32, col_l, rt_128x16_s>` (col-layout B, vs RCR's row-layout)
- `ds_read_b64_tr_b8` for B-load (transposed 8B), vs RCR's `ds_read_b128` (16B)
- 2x more `ds_read` ops per K-iter (B carries 8B/lane vs RCR's 16B/lane on B)
- Same 4-acc layout per warp

The B-load LDS bandwidth doubling is the structural reason RRR is harder to optimize. Per `[[fp8-rrr-attempt-h14]]`, this is bandwidth-bound on worst shapes.

## P2.2 Multi-Session Plan

### Session 1: B-load width audit (~80 LOC analysis)
- Disassemble RRR main loop, count exact `ds_read_b*` per iter
- Compare with RCR (R3 commit moved RCR to 40 ds_read, dense parity)
- Identify: is RRR's ds_read_b64_tr_b8 forced by gfx950 ISA (per `[[fp8-rrr-h17-clean-baseline]]` finding gfx950 lacks ds_read_b128_tr_b8) or can it pre-transpose B in LDS swizzle?

### Session 2: B LDS pre-transpose (~250 LOC)
- New `ST_v2_b_transposed` swizzle that stores B already-transposed in LDS
- Per G::load global→LDS pattern: shuffle bytes during global read
- Switch RRR B-load to `ds_read_b128` (16B/lane, half the load count)
- Risk: gfx950 buffer_load_lds may not support strided byte shuffle. `[[fp8-rrr-attempt-h17-clean-baseline]]` says no ds_read_b128_tr_b8 — confirm if pre-transpose path works.

### Session 3: FUSED_KTAIL=true register pressure (~150 LOC)
- Spill 67 vs FUSED=false 37 → FUSED block adds 30 VGPR spill. Same root cause as RCR P1.2 (lambda + scratch + 4 acc overlap)
- Apply R22-style serialization (memory `[[grouped-rcr-agpr-inplace]]`) but in RRR variant
- Reuse cA/cB/cC/cD instead of cA_kt0/cA_kt1 patterns
- Target spill 67 → 35 or lower

### Session 4: 32×32 mfma + body rewrite (~300 LOC)
- Mirror P1.2 Session 2-3 but for RRR layout
- B_col_reg → B_col_reg_32 (32-col K=64 fragments)
- mma_32_int4 wrapper accepts col-layout via mfma signature variant
- May need mfma_f32_32x32x64 with cbsz/blgp/abid permutation for col-major B

### Session 5: Full integration + bench
- 24-shape RRR bench vs Triton + dense
- Target: spill=0 on FUSED=true, ≥ baseline perf, ratio v2/Triton from 0.955 → 1.15+

## Expected Difficulty

Higher than P1.2 RCR because:
1. RRR transpose-load is in mfma fragment layout level, not just K-loop topology
2. B LDS layout change has cascading impact on all RRR variants
3. `[[fp8-rrr-attempt-h6]]` showed RRR is more compiler-scheduling sensitive

Estimated 5-7 sessions vs P1.2's 4-5.

## Recommended Sequence

P1.2 first (validated foundation R52-R58), then P2.2 (reuses 32×32 wrapper from P1.2). Doing in parallel risks divergent fragment type evolution.
