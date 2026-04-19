# R38 Opt F — PROGRESS

## Mission
Per R38C verdict, the right structural fix for the 6 WIN→WRONG_OUTPUT and 6
CRASH→WRONG_OUTPUT demotions is to insert an explicit drain
(`s_waitcnt vmcnt(N)[+s_barrier]`) on the tail iters in the R37_FIX_B fused
step3+step4 path, so the previous iter's `buffer_load_to_lds` lands before the
next iter's `s_barrier` releases the LDS slot for reuse.

## Implementation
- Added macros `R38F_TAIL_DRAIN` (default 0) + `R38F_VARIANT` (default 2) to
  `kernel_mxfp4_gluon_cpp.cpp`.
- Variants:
  - F1: `s_waitcnt vmcnt(0)`
  - F2: `s_waitcnt vmcnt(0)\ns_barrier`
  - F3: `s_waitcnt 0`
  - F4: `s_waitcnt vmcnt(0) lgkmcnt(0)`
- Drain placement: TOP of each iter, gated on `_r38f_in_tail` (same R25-C tail
  range — last `R25C_TAIL_PF_OFF_ITERS` iters). Folded compile-time on shapes
  with K_DIM ≤ R25C_K_LIMIT.
- `R38F_TAIL_DRAIN` is layered ON TOP of `R38B_TAIL_FIX=1` (always-emit) — R38B
  closes the underlying CRASH so the drain has something to drain. Without R38B
  the canonical CRASH variant just CRASHes (R37 baseline behavior, irrespective
  of any drain insertion).
- Mutex with R38C only.

## Smoke test results (m32768_n4096_k2048 ts_lgk2_gm6_v12_memc_pfoff4, 30-50 reps)

| Variant | Crash? | finite range | failures |
|---|---|---|---|
| R37 (baseline) | YES | n/a | CRASH on rep 0 |
| R38B alone | NO  | 0.987–0.992 | 50/50 below 0.995 |
| R38B + R38F1 (vmcnt(0))            | NO | 0.989–0.994 | 48/50 below gate |
| R38B + R38F2 (vmcnt(0)+s_barrier)  | NO | 0.989–0.991 | 50/50 below gate |
| R38B + R38F3 (s_waitcnt 0)         | NO | 0.992–0.995 | 26/30 below gate |
| R38B + R38F4 (vmcnt(0)+lgkmcnt(0)) | NO | 0.983–0.987 | 30/30 below gate |

Also tested R38F WITHOUT R38B (gated drain at TOP of tail iter): all four
variants CRASHed (HSA aperture violation), confirming the underlying CRASH must
be closed first by R38B; R38F alone is structurally insufficient.

## Conclusion
Hypothesis F is **REFUTED**. The drain DOES drain in-flight loads (verified by
no fault on R38B+F runs) but does not restore correctness above the 0.995
finite gate. The R38C verdict's mechanism description ("LDS write from previous
iter hasn't drained") is not the operative bug — or at minimum, a drain alone
at the iter boundary cannot fix it.

The likely real bug is one of:
1. **Scale-pack mismatch** — `load_pq_scale_x2_async(... bt+1 ...)` continues to
   advance the scale index even on tail iters where the prefetched data is
   clamped to `pf_bt = k_byte_iters - 1`. Mismatch between scale index and data
   tile produces arithmetic-wrong outputs (which look like `bf16-overflow garbage`
   per the project memo on `17pct_wrong cells`). A drain cannot fix this.
2. **LDS double-buffer state divergence between WG threads** — drain helps the
   producer side, but if some lanes already advanced past the slot, no drain
   on the consumer side restores the correct data.
3. **Compiler reordering of `make_pf_params` + `emit_pf_tail` across the inner
   `if` boundary** — even with R38B's "always-emit", the construction inside
   the `{ ... }` scope leaves room for the compiler to re-issue under different
   pressure assumptions on tail iters.

## Recommended next steps for the orchestrator
- **Pivot away from drain-based fixes**. The "correctness 17%-wrong cells" memo
  suggests the bug is arithmetic/scale-side, not LDS-race.
- For shapes where R37_FIX_B + R25-C combo produces `finite < 0.995`, drop
  `_pfoff*` variants from BEST_VARIANTS (accept ~3% perf loss; restore
  correctness across the leaderboard). This was R38C verdict's recommendation
  #3 and remains the most promising path.
- Investigate the scale-pack offset on tail iters as a separate root-cause
  investigation (R39 candidate: `R39_TAIL_SCALE_CLAMP`).

## Files
- `kernel_mxfp4_gluon_cpp.cpp` — adds `R38F_TAIL_DRAIN` + `R38F_VARIANT` macros
  (default OFF). The drain is emitted at the TOP of each tail iter inside the
  R37_FIX_B path.
- `build_R38F.py` — builds `_R38F{1..4}` variants with R38B+R38F co-enabled.
- `R38F_smoke_one.py` — single-shape smoke harness.
- `R38F1_BUILD_MANIFEST.json` ... `R38F4_BUILD_MANIFEST.json` — per-variant
  build manifests.
