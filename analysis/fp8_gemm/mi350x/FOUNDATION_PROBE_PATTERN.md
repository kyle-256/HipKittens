# Foundation Probe Pattern (R83 — methodology doc)

## Pattern Recipe

When evaluating multi-session architectural rewrite viability, build a 6-stage probe ladder before touching production code:

1. **isolated mfma** — 1 mma call in single-warp 64-thread WG. Validates wrapper compiles + correctness primitives in place. Expect V<50 A<32 spill=0.
2. **single-acc K-chain** — accumulator + K-iter loop. Validates compiler can manage frag through K loop. Expect V<60.
3. **multi-acc K-chain (1 warp)** — 2/4 acc as separate variables. Validates compiler doesn't spill at scale.
4. **multi-acc K-chain (multi-warp WG)** — true production WG topology (8 warps × 64 = 512 threads). Validates `launch_bounds(_NUM_THREADS, 1)` constraint propagates.
5. **+ LDS double-buffer** — adds shared mem state. Tests if LDS handling spikes register pressure.
6. **+ persistent + masked store** — closest to production minus group dispatch. Final pre-integration check.

## Stop Rules

Per stage, if `vgpr_spill_count > 0`:
- **Stage 1-3 spill > 0**: foundation broken, rewrite path invalid, abandon
- **Stage 4 spill > 0**: 8-warp WG topology incompatible, may need 4-warp WG fallback or recount V+A cap
- **Stage 5-6 spill > 0**: LDS/store overhead exceeds budget, need to reduce per-acc area (smaller BLK) or reduce acc count

## Real-World Calibration (R52-R58 for 32×32 mfma on RCR)

| Stage | Probe | V | A | spill |
|-------|-------|---|---|-------|
| 1 | __probe_v2_mma_32_isolated | 36 | 0 | 0 ✓ |
| 2 | __probe_v2_mma_32_kchain_22 | 44 | 0 | 0 ✓ |
| 3 | __probe_v2_mma_32_2acc_k22 | 64 | 0 | 0 ✓ |
| 3 | __probe_v2_mma_32_4acc_k22 | 108 | 32 | 0 ✓ |
| 4 | __probe_v2_mma_32_8w_4acc_k22 | 104 | 0 | 0 ✓ |
| 5 | __probe_v2_mma_32_8w_4acc_lds_k22 | 104 | 16 | 0 ✓ |
| 6 | (not yet built — multi-session integration starts here) | ? | ? | ? |

Stages 1-5 all pass → foundation validated.

## Build Cost

Each probe: ~40-60 LOC + 1 build + metadata extract = ~5 min wall time
Total foundation 5 probes: ~25-30 min, then can decide multi-session investment.

## Use in Decision Making

If all 5 probes spill=0, ratio of probe-V at stage 5 to current production-V is the **spill-headroom ratio**:
- R58 stage 5 V=104 vs current production V=256 → 152 V headroom for production state (persistent + group dispatch + masked store)
- 152 V is comfortable budget; 4-acc 16x16 path used 256+ spill 24/35 because compiler scheduled all 8 mma per acc concurrently (frag overlap)

Conclusion: P1.2 32×32 likely lands spill=0 in production with reasonable confidence.
