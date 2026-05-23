# P1.2 32×32 MFMA Integration Design (R63 — multi-session handoff)

## Foundation Validation (R52-R58, all spill=0)

| Probe | Topology | V | A | spill | LDS |
|-------|----------|---|---|-------|-----|
| R53 isolated 1-mfma | 1 warp | 36 | 0 | 0 | 0 |
| R54 1-acc K=22 chain | 1 warp | 44 | 0 | 0 | 0 |
| R55 2-acc K=22 | 1 warp | 64 | 0 | 0 | 0 |
| R56 4-acc K=22 | 1 warp | 108 | 32 | 0 | 0 |
| R57 4-acc K=22 (8-warp WG) | 8 warps × 64 | 104 | 0 | 0 | 0 |
| R58 4-acc K=22 + LDS prefetch | 8 warps × 64 | 104 | 16 | 0 | 32 KB |

Conclusion: 32×32×64 mfma wrapper with 4-acc + 8-warp WG + LDS prefetch achieves spill=0. Production kernel target (V/A/spill = ?/?/0) is feasible.

## Integration Blocking Issues (R60/R61 — failed)

1. **Fragment layout mismatch**: HK `A_row_reg = rt_fp8e4m3<RBM=64, BK=128, row_l, rt_16x128_s>` uses 16-row tiles with full K=128 per tile. 32×32×64 mfma requires 32-row × K=64 per operand. Direct wrapper integration causes type mismatch.

2. **Per-warp acc area**: Per-warp 64×32 area maps to:
   - 16×16×128 path: 4 mfma_M × 2 mfma_N = 8 mfma per acc (current production)
   - 32×32×64 path: 2 mfma_M × 1 mfma_N × 2 K-halves = 4 mfma per acc (target)
   - Half the mfma count = potentially less live fragment overlap = less spill pressure.

3. **Namespace include order**: v2 file `#include "kernel_fp8_layouts.cpp"` brings v1 helpers into v2 namespace. Re-defining types after include causes "redefinition with different types" errors (R60). New typedefs must be uniquely named.

## Recommended Multi-Session Plan

### Session 1: Fragment types + reload primitive (~150 LOC)
- Define `A_row_reg_32` / `B_row_reg_32` = `rt_fp8e4m3<32, 64, row_l, rt_32x64_s>` with **distinct namespace** (v2_pinned::) to avoid R60 conflict
- Write `load_a_32` / `load_b_32` using HK `subtile_inplace<32, 64>` + `load()` primitives
- Add isolation probe: load → mfma → store. Expect V<60 A=0 spill=0.

### Session 2: K-loop body skeleton (~200 LOC)
- Clone `grouped_rcr_kernel_body_pinned` to `grouped_rcr_kernel_body_32`
- Replace 16×16 frag types with 32×64 frag types
- Replace `rcr_mma_v2_wrapper` calls with `rcr_mma_v2_wrapper_32` (2 K-halves per acc)
- Adjust ST_v2 LDS read pattern: `ds_read_b64` (16-row) → `ds_read_b128` (32-row) if alignment permits, or keep b64 + double the K-iter
- Initial smoke run; metadata check spill=0

### Session 3: FUSED_KTAIL port (~100 LOC)
- Mirror FUSED block but with 32×64 frag + mfma_32
- K_rem=64 still needs 1 mfma_32 call (covers K=64 exactly)
- Smoke gpt_oss K=2880

### Session 4: Dispatcher integration + bench (~50 LOC)
- Add `TK_RCR_V2_USE_32` env knob to select v2 body variant
- 24-shape bench compare 32×32 vs 16×16 body
- Spill metadata: target 0/0 on all 4 template variants
- Perf delta target: ≥ v2 baseline 1.029× Triton

### Session 5: Production routing + cleanup
- Make 32×32 default if perf wins; remove env knob
- Remove old 16×16 body; ~600 LOC reduction
- Final smoke + bench + commit

## Risk Notes

- **Per-warp area constraint** (`[[fp8-rrr-32x32-flawed-premise]]`): warning said wrapper swap alone won't reduce spill. R56 4-acc probe spill=0 may be due to lack of full K-loop body state. Real production body may not reach spill=0 even with 32×32 — R57/R58 probes don't include masked store, persistent state, group dispatch, FUSED_KTAIL — extra ~30-50 V could push past comfort zone, but should still fit 1 wave/SIMD cap.

- **Build cache risk**: per `[[fp8-rrr-grad-a-fix-2026-05-19]]`, build/temp/csrc/kernels/grouped_gemm/HipKittens/.o + .so must be force-cleaned when fragment types change. PT _hip.cpp mirrors are .gitignored and don't auto-clean.

- **Correctness debugging cycle**: mfma 32×32 vs 16×16 numerics are bit-equivalent (both fp8 e4m3 with f32 accumulate), but lane mapping differs. SNR-30dB gate may need adjustment to bit-equal during transition.

## Estimated Effort

- 4-5 sessions × 4-8 hours of focused work each
- ~600 LOC additions + ~400 LOC deletions (net ~+200 LOC during transition)
- Cycle: each session ends with smoke + spill metadata check; revert if regression

## Lever Hierarchy Post-P1.2

Once P1.2 lands spill=0 + ≥ baseline perf:

1. P1.3 (a) split-K cross-group B share — only achievable algorithmic lever for the 12pp gap to 1.15× Triton
2. P2.2 RRR similar 32×32 rewrite (mirrors P1.2 work, ~3 sessions)
3. P3.2 CRR var_k operand layout rewrite (~2 sessions)

Total remaining multi-session work to fully meet user goals: ~10-15 sessions of focused engineering.
