R30 Dev A — rectangular BLK_M=256/BLK_N=128 V2-CRR fastpath: NO SHIP, scaffolding preserved + Path A breakdown refined
=====================================================================================================================

Status: NO SHIP. Path A (true rect-V2 fastpath kernel) confirmed infeasible
within the 90-min budget — R28D's 2.5-day estimate stands and is corroborated
by a fresh audit of the constraints below. Path B (V1 fallback) was already
shipped as R29 guard and is catastrophically slow (294x slower than V2 baseline
on the rect target shape — no perf win possible). This cycle re-establishes
the baseline numbers on r30-a, confirms the cherry-picked R28D+R29A scaffolding
adds zero overhead (8192³ V2-CRR within -0.08% of R28 baseline), and produces
a refined Path A work breakdown that the next cycle author can pick up directly.

TL;DR
-----
- 70B KV V2-CRR baseline (4096×1024×8192): **791.25 TFLOPS median** (5 runs,
  preheat-then-bench, GPU0). Stable: stdev 9.34, lowest 774.99, highest 793.79.
- 8192³ V2-CRR regression check: **2841.64 TFLOPS median** vs R28 baseline
  2844 — Δ -0.08%, well within ±2%. Scaffolding adds zero overhead at default
  build.
- Determinism 3/3 PASS, SNR 49.59-49.60 dB on both shapes.
- Path A NOT ATTEMPTED — confirmed infeasible in budget after audit (see below).
- Path B (V1 fallback via R29 guard) confirmed still works but irrelevant for
  perf (~2.66 TFLOPS = 0.34% of square baseline per R29 Cell 2).
- Net deliverable: refined Path A work breakdown + cherry-picked scaffolding
  preserved on r30-a + clean baseline numbers for next-cycle SHIP comparison.

Path decision audit (why not Path A)
------------------------------------
R28 Dev D explicitly estimated ~2.5 days for Path A:
  "Estimated next-cycle effort: 1.5 days kernel, 0.5 day Python, 0.5 day
   validation. Total ~2.5 days — matches R27 Dev D's audit estimate."

The 90-min R30 budget is ~1/40th of that estimate. Re-validating Dev D's
breakdown against the kernel today:

1. **`crr_exact_8wave_scaled_kernel` is 600+ lines of hand-tuned MXFP8 code**
   (`crr_mxfp8_exact_8wave_fastpath.inc:88-645`) with the following hardcoded
   assumptions visible at compile-time static_assert:
   ```
   static_assert(BLK == 256, "MXFP8 exact 8-wave fast path requires BLK=256");
   static_assert(BK == 128, "MXFP8 exact 8-wave fast path requires BK=128");
   static_assert(WARPS_M == 2, "...WARPS_M=2");
   static_assert(WARPS_N == 4, "...WARPS_N=4");
   static_assert(RBM == 64 && RBN == 32);  // in cA_with_b1_interleave_*
   ```
   For rect (BLK_N=128) all four of these need to be loosened or duplicated
   into a parallel template.

2. **B-side helper `load_col_from_v2_st_half_rect` only covers K_HALF=0**
   (R28D scaffolded). The math reveals K_HALF=1 reads N-rows 64..87 from a
   tile that only has 64 rows total in rect mode → must be replaced by two
   K_HALF=0 calls with `col_start` offset. That requires per-call book-keeping
   in the new kernel. Dev D's scaffolding documents this explicitly but does
   not implement the wiring.

3. **Scale slab math at `crr_exact_8wave_scaled_kernel:262-273` and
   `:307-333`** indexes by `bc * BLK + ...` and uses `crr_b_pack_count =
   (RBN + 31) / 32 = 2` for default (RBN=32). For rect RBN_RECT=16,
   `crr_b_pack_count` drops to `(16+31)/32 = 1`. Every loop annotated by
   `#pragma unroll` over `crr_b_pack_count` needs to be re-validated and the
   B_col_reg type retyped to `rt_fp8e4m3<BK, RBN_RECT, col_l, ...>`.

4. **Host-side V2 preshuffle `preshuffle_v2_b` in r29a_bench.py** assumes
   `blk=256, hb=128, rbn=32, warps_n=4`. The rect variant needs
   `blk_n=128, hb_n=64, rbn=16, warps_n=4` and `pack_count_b = 1`. Same change
   needs propagating to `test_mxfp8_python.py` for production callers. Both
   the host preshuffle math and the GPU-side scale base address `(crr_scale_b_base(half) + p * 32) >> 5`
   need to stay in lock-step or correctness silently breaks (no SNR gate
   would catch a single-pack scale shift).

5. **New dispatcher entry `dispatch_crr_exact_8wave_scaled_v2_rect<true>(g)`**
   needs to be added next to the existing `dispatch_crr_exact_8wave_scaled_v2`
   (`crr_mxfp8_exact_8wave_fastpath.inc:663-668`) AND wired into
   `dispatch_pq_v2<CRR>` (`kernel_mxfp8_layouts.cpp:5276-5281` region) AND
   gated by a new `crr_can_use_exact_8wave_scaled_rect` shape predicate that
   accepts both square (default) and rect (RECT_BLK_N=64) callers without
   accidentally hijacking the existing square fastpath.

6. **Validation surface**: must run on (a) 70B KV rect target 4096×1024×8192,
   (b) 8192³ for no-regression, (c) at least 2 LLaMA shapes per the SHIP gate
   matrix in `agent_prompt.md`. SNR + det + 5x preheat-bench × 3 shapes ≈ 1
   GPU-hour just for the post-implementation gate.

Even an aggressive implementer producing ~50 working lines of HIP/asm per hour
needs ~12 hours just for the kernel; the budget is 1.5 hours. Path A is not
achievable here.

What was actually done this cycle
---------------------------------
1. Cherry-picked r28-d (`ddfd2f80`) scaffolding + r29-a (`b2cc032f`) dispatcher
   guard onto r30-a in clean order. Verified `MXFP8_RECT_BLK_N` macro present
   at `kernel_mxfp8_layouts.cpp:303-318` and rect helper `load_col_from_v2_st_half_rect`
   present at `:482-578` and dispatcher guard at `:5290-5298`.

2. Built default (no MXFP8_RECT_BLK_N) for 70B KV target shape and 8192³.
   Both compile clean with the R28 cachepolicy auto-select gate active.
   `.so` md5 hashes captured per-build per R29 Dev C methodology rule.

3. Bench cells (5x preheat-then-bench, 50 warmup / 100 iters,
   HIP_VISIBLE_DEVICES=0):

   | Cell | Build flags                                       | Stage | Shape          | TFLOPS (median) | TFLOPS (mean ± stdev) | SNR    | Det |
   |------|---------------------------------------------------|-------|----------------|----------------:|----------------------:|-------:|-----|
   | 1    | -DM_DIM=4096 -DN_DIM=1024 -DK_DIM=8192            | v2    | 4096x1024x8192 |          791.25 |       785.42 ± 9.34   | 49.60  | OK  |
   | 2    | -DM_DIM=8192 -DN_DIM=8192 -DK_DIM=8192            | v2    | 8192x8192x8192 |         2841.64 |      2836.03 ± 17.07  | 49.59  | OK  |
   | 3    | -DM_DIM=4096 -DN_DIM=1024 -DK_DIM=8192 -DMXFP8_RECT_BLK_N=64 | compile | n/a       |             n/a |                   n/a |   n/a  | OK  |

   Cell 3 confirms the rect build still compiles clean; functional behavior
   unchanged from R29 (V1 fallback proven SNR 49.59 / 2.66 TFLOPS / det 3/3
   per R29 Cell 2; no change made this cycle).

4. Comparison vs prior cycles:

   | Reference        | Shape         | Cycle | Median TFLOPS | Δ vs R30 |
   |------------------|---------------|-------|--------------:|---------:|
   | R29A Cell 1      | 4096x1024x8192| R29   |        781.91 |   +9.34  |
   | R30A Cell 1      | 4096x1024x8192| R30   |        791.25 |        — |
   | R28A baseline    | 8192³         | R28   |        2844   |   -2.36  |
   | R29A Cell 3      | 8192³         | R29   |        2832.11|  +9.53  |
   | R30A Cell 2      | 8192³         | R30   |        2841.64|        — |

   Both numbers are within 2-cycle bench-to-bench noise (R29A / R28A spreads
   were ±15 TFLOPS). No regression introduced.

SHIP gate evaluation
--------------------
SHIP gate per task spec:
  - Correctness: SNR ≥ 48 dB on 70B KV + 8192³ → **PASS** (49.60 / 49.59).
  - Determinism: 3/3 → **PASS**.
  - Performance: 70B KV V2-CRR rect ≥ square × 1.5 → **N/A** (no rect kernel
    exists; SHIP gate cannot be evaluated).
  - No regression: 8192³ within ±2% of R28 baseline 2844 → **PASS** (-0.08%).
  - Welch t > 3.0 between rect-V2 and square-V2 baseline → **N/A** (no rect
    kernel exists).

Verdict: **NO SHIP**. Two SHIP gates (perf + Welch t-test) cannot be evaluated
because no rect-V2 fastpath kernel exists. The two gates that can be evaluated
(correctness + no-regression) PASS.

Refined Path A breakdown for R31
--------------------------------
The R28D TODO list remains correct but can be sharpened with the audit above:

**Stage A1 (kernel — 1.0 day)**:
- Copy `crr_exact_8wave_scaled_kernel` template body (lines 88-645 of
  `crr_mxfp8_exact_8wave_fastpath.inc`) to a new file
  `crr_mxfp8_exact_8wave_rect_fastpath.inc`.
- Replace `BLK` → `BLK_N` everywhere indexing the N dimension (the M-dim
  references stay `BLK` since BLK_M=256 is unchanged): `bc * BLK` →
  `bc * BLK_N` at lines 235, 295, 303-304, 497-498, 559, 569, 641-644.
- Replace `RBN` → `RBN_RECT` everywhere it parameterizes B-side register tiles:
  `B_col_reg` typedef around line 200 area, all `rt_fl<RBM, RBN, ...>` of
  cA/cB/cC/cD register types (lines 213, 89-100 region).
- Loosen `static_assert(RBM == 64 && RBN == 32)` in
  `crr_exact_cA_with_b1_interleave_fixed_phase` (line 100) and
  `crr_exact_cA_with_b1_interleave_var_phase` to allow `RBN == 32 || RBN == 16`.
- Replace `load_col_from_v2_st` → 2x `load_col_from_v2_st_half_rect<RT, 0>` with
  col_start offset (helper already exists in `kernel_mxfp8_layouts.cpp:528-578`).
- Recompute `crr_b_pack_count = (RBN_RECT + 31) / 32 = 1` — all loops over
  `crr_b_pack_count` will collapse to 1 iteration; verify no off-by-one in
  the b1 prefetch / scale-pipelining at `:265-273` and `:329-333`.

**Stage A2 (host preshuffle — 0.5 day)**:
- In `r29a_bench.py` (and the production `test_mxfp8_python.py`),
  parameterize `preshuffle_v2_b` to accept `blk_n=128, hb_n=64, rbn=16,
  warps_n=4` for the rect variant. The slab geometry math at lines
  114-138 of `r29a_bench.py` becomes:
    `num_slabs_b_rect = (N // BLK_N) * WARPS_N` (= 4 * 4 = 16 for N=1024)
    `pack_count_b_rect = (RBN_RECT + 31) // 32 = 1`
  Then add a `preshuffle_v2_b_rect` callable that the bench harness invokes
  when `MXFP8_RECT_BLK_N=64` is in effect.
- Cross-check against the GPU-side scale base addresses in the new rect
  kernel — the host slab indexing must match `crr_scale_b_base(half) + p * 32`
  exactly (a single-pack offset bug is silent and won't trip SNR).

**Stage A3 (dispatcher — 0.25 day)**:
- Add `dispatch_crr_exact_8wave_scaled_v2_rect<bool>(g)` next to
  `dispatch_crr_exact_8wave_scaled_v2` (currently
  `crr_mxfp8_exact_8wave_fastpath.inc:663-668`):
    `const dim3 grid((g.m / BLK) * (g.n / BLK_N));  // BLK_N=128 in rect mode`
    `crr_exact_8wave_scaled_rect_kernel<...><<<grid, ...>>>(g);`
- Add a `crr_can_use_exact_8wave_scaled_rect(g)` predicate that returns true
  when MXFP8_RECT_BLK_N=64 AND `g.m % BLK == 0` AND `g.n % BLK_N == 0`.
- In `dispatch_pq_v2<CRR>` (`kernel_mxfp8_layouts.cpp:5272-5281`), replace
  the R29 `fprintf+return` guard at `:5290-5298` with:
    ```
    #if defined(MXFP8_RECT_BLK_N) && (MXFP8_RECT_BLK_N == 64)
        if constexpr (L == Layout::CRR) {
            if (crr_can_use_exact_8wave_scaled_rect(g)) {
                dispatch_crr_exact_8wave_scaled_v2_rect<true>(g);
                return;
            }
        }
    #endif
    ```

**Stage A4 (validation — 0.5 day)**:
- 5x preheat-then-bench at 4096×1024×8192 (rect target). SHIP target ≥ 1187
  TFLOPS (= 791.25 × 1.5).
- 5x preheat-then-bench at 8192³ (no-regression check, must stay within ±2%
  of 2844 = [2787, 2901]).
- 5x preheat-then-bench at LLaMA 70B Q (4096×8192×8192) and LLaMA 70B Down
  (4096×28672×8192) per `agent_prompt.md` SHIP matrix (these shapes don't
  build with MXFP8_RECT_BLK_N=64 because they're already large-N — should
  fall through the rect predicate and use the existing square fastpath).
- Welch t-test rect vs square at 4096×1024×8192 — must be > 3.0.

Total: 2.25 days, broadly matching R28D's 2.5-day estimate.

**Risk note**: The grid for rect mode at 70B KV target shape becomes
`16 × 8 = 128 blocks`. With 304 CUs, occupancy = 42% (vs 21% for square).
That's a max 2x improvement and is the upper bound on the SHIP-gate perf win.
If the rect kernel introduces any per-block overhead (1 fewer scale pack does
NOT compensate for 1/2 fewer N-tiles per kernel grid), the SHIP target of
1.5x may not be reachable. R31 should plan a 1.3x stretch goal as a soft pass.

A more aggressive grid increase requires either:
  - BLK_N=64 (yet another scaffolding round, RBN_RECT=8, B_col_reg width drops
    to 16), giving 4x grid = 256 blocks at 84% occupancy on 304 CUs, OR
  - split-K with atomic accumulation OR a separate reduction kernel.

Both are out of scope for R31 and should be deferred until R30A's Stage A
is shipped and measured.

Files touched this cycle
------------------------
  + analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp (cherry-pick R28D+R29A,
    no new edits)
  + analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc (cherry-pick
    R28D MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE rect gate, no new edits)
  + analysis/fp8_gemm/mi350x/r28d_validate.py (cherry-pick R28D)
  + analysis/fp8_gemm/mi350x/r28d_findings.md (cherry-pick R28D)
  + analysis/fp8_gemm/mi350x/r29a_bench.py (cherry-pick R29A)
  + analysis/fp8_gemm/mi350x/r29a_orchestrate.sh (cherry-pick R29A)
  + analysis/fp8_gemm/mi350x/r29a_findings.md (cherry-pick R29A)
  + analysis/fp8_gemm/mi350x/r29a_cell{1,2,3}_*.txt (cherry-pick R29A)
  + analysis/fp8_gemm/mi350x/r30a_orchestrate.sh (NEW — R30 runner)
  + analysis/fp8_gemm/mi350x/r30a_cell{1,2,3}_*.txt (NEW — R30 bench output)
  + analysis/fp8_gemm/mi350x/r30a_findings.md (NEW — this file)

Print stamp: `=== R30 DEV A NO SHIP — PATH A INFEASIBLE IN BUDGET, BASELINE PRESERVED ===`
