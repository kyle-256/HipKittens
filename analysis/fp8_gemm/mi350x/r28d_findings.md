R28 Dev D — rectangular BLK_M=256/BLK_N=128 SCAFFOLDING
========================================================

Status: SCAFFOLDING ONLY (Stage 1 + Stage 2 PASS, Stage 3 FAIL as expected)

TL;DR
-----
Compile-ready scaffolding for the realistic 2-3 day rectangular tile path
that R27 Dev D's audit identified as the next-cycle priority. Adds:

  1. `MXFP8_RECT_BLK_N` compile-time macro (default 128 = baseline,
     set to 64 for rect mode). Validated at compile time
     (constexpr ladder + static_assert).
  2. New rectangular B-side helper `load_col_from_v2_st_half_rect`
     templated on `RECT_HB_N`. Compiles for both square (HB_N=128)
     and rect (HB_N=64) modes. NOT YET CALLED FROM ANY HOT KERNEL —
     reserved for the next-cycle rect-V2 fastpath.
  3. Compile-time gate that force-disables the existing V2-CRR exact
     8-wave fastpath when `MXFP8_RECT_BLK_N=64`. Mirrors R27 Dev D's
     `MXFP8_BLK128` pattern. Default build is byte-identical to
     pre-patch.

Stage outcomes
--------------
Stage 1 (compile validation): PASS
  - Default build (no `-DMXFP8_RECT_BLK_N`): clean compile, .so 494,920 B,
    8192^3 V2-CRR bench 2700 TFLOPS, SNR 49.59 dB, det OK. Within ±15 of
    R28 Dev A baseline 2844 (ambient sclk noise).
  - Rect build (`-DMXFP8_RECT_BLK_N=64`): clean compile, .so 454,600 B
    (−40,320 B = -8.1%, confirms V2-CRR exact-8wave fastpath is dropped
    from binary). Verified at both 4096×1024×8192 (KV-attn target) and
    8192³ (sanity).

Stage 2 (V1 fallback PASS): PASS
  - Built with `-DMXFP8_RECT_BLK_N=64`, ran via `r28d_validate.py v1
    crr 4096 1024 8192` which calls `gemm_crr_pq` (V1 dispatch) with
    V1 host preshuffle. SNR 49.59 dB, pass_rate 100.00%, determinism
    3/3, ~2.67 TFLOPS (no fastpath, expected slow).

Stage 3 (V2 correctness): FAIL (expected — same failure mode as R27d)
  - `gemm_crr_pq_v2` with V2 host preshuffle and rect gate ON crashes
    with `Memory access fault by GPU node-3`. Mirrors R27 Dev D's
    Test 3 finding: when the V2 fastpath is gated off, the V2 dispatch
    (`dispatch_pq_v2`) falls through to `dispatch<L, true>` which
    expects V1 layout but is being fed V2-preshuffled scales. The
    out-of-bounds access reflects the layout mismatch.

Stage 4 (bench): NOT REACHED (Stage 3 prerequisite failed).

What was built (concrete diff anchors)
--------------------------------------
File: `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp`
  - L289-310 region: added `MXFP8_RECT_BLK_N` macro (default 128) +
    static check that value is in {64, 128}.
  - L320-326 region: new constexpr ladder `HB_N = MXFP8_RECT_BLK_N`,
    `BLK_N = HB_N * 2`, `RBN_RECT = BLK_N / WARPS_N / 2`. Names chosen
    so they cannot collide with the existing default constants
    (HB / BLK / RBN); next-cycle author can swap usage incrementally.
  - L498-578 region (right after `load_col_from_v2_st`): new
    `load_col_from_v2_st_half_rect<RT, K_HALF, RECT_HB_N>` helper
    templated on RECT_HB_N. Asserts K_HALF=1 invalid when HB_N=64
    (out-of-bounds). Documents the BK-derived `offset:1024 = 8*BK`
    inline-asm stride (the audit slightly mischaracterized this as
    HB-derived; in fact only the per-tile row count is HB-dependent —
    the per-subtile stride 2176 = 16*128 + 128 padding is fixed by
    the `st_16x128_v2` swizzle, and the K-stride 1024 is fixed by BK).

File: `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc`
  - L3-15 region: added rect gate that #undefs +force-defines
    MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE to 0 when MXFP8_RECT_BLK_N=64.
    Same pattern R27 Dev D used for MXFP8_BLK128.

File: `analysis/fp8_gemm/mi350x/r28d_validate.py` (new)
  - Stage 2/3 validation harness. Supports `v1` (gemm_crr_pq + V1
    host preshuffle) and `v2` (gemm_crr_pq_v2 + V2 host preshuffle).
    Used to verify the rect build above.

Key insight from helper analysis (correction to audit)
------------------------------------------------------
R27 Dev D's audit characterized the `ds_read_b64_tr_b8 ... offset:1024`
inline-asm stride as "BK=128 bytes × HB=128 rows / 16" implying
HB-dependence. Re-deriving it shows the 1024 byte offset is in fact
**8 rows × BK bytes** (the second ds_read_b64_tr_b8 in the pair reads
8 rows further down the K direction within the same M/N column). It is
therefore independent of HB and remains 1024 for any HB_N as long as
BK stays 128. The HB dependency is concentrated in **two** places only:

  1. `k_row = row_off + K_HALF * 64`: when K_HALF=1, k_row reaches 87.
     For HB_N=64 this overflows the 64-row tile. → New helper either
     restricts K_HALF=0 (loading 24 of 64 rows; called repeatedly with
     col_start advanced) or recomputes the addressing.
  2. The number of subtiles per ST tile (HB / 16) — affects how many
     full sweeps the helper needs to cover.

The audit's complexity-4 estimate for the helper rewrite is correct in
calendar time (the kernel-level wiring is the bulk of the work), but
the per-helper surgery is more surgical than the audit suggested.

Next-cycle starting point (concrete TODO list)
----------------------------------------------
1. Implement a rect-V2 fastpath kernel: copy
   `crr_exact_8wave_scaled_kernel` to a new file
   `crr_mxfp8_exact_8wave_rect_fastpath.inc` with:
   - `BLK_N` (=128) substituted for `BLK` everywhere it indexes the
     N dimension: scale slab math at L262-273 (slab_idx_b counting),
     L307-333 (b_voff stride uses RBN_RECT=16, b pack count drops
     from 2 to 1 since RBN_RECT/32 == 0 → need (RBN_RECT+31)/32 = 1
     pack), L641-652 (grid `(g.m / BLK) * (g.n / BLK_N)`),
   - `B_col_reg` retyped to `rt_fp8e4m3<BK, RBN_RECT, col_l, ...>` so
     RBN_RECT=16 width is honored,
   - `load_col_from_v2_st_half_rect<RT, 0>` × 2 (with col_start
     advanced) substituted for `load_col_from_v2_st`,
   - `static_assert(RBM == 64 && RBN == 32)` in
     `crr_exact_cA_with_b1_interleave_*` weakened to allow RBN=16,
2. Add `dispatch_crr_exact_8wave_scaled_v2_rect<true>(g)` and route
   it from `dispatch_pq_v2<Layout::CRR>` when both
   `MXFP8_RECT_BLK_N=64` and a `crr_can_use_exact_8wave_scaled_rect`
   shape predicate match.
3. Update `preshuffle_v2_b` in `r28a_bench5x.py` (and equivalent in
   `test_mxfp8_python.py`) to accept `blk_n=128` so the host-side
   slab geometry matches: `num_slabs_b = (N / BLK_N) * WARPS_N` and
   `pack_count_b = (RBN_RECT + 31) / 32 = 1` (down from 2).
4. Validate on 70B KV V2-CRR (4096×1024×8192). With 304 CUs and the
   current grid `(M/BLK) * (N/BLK) = 16 * 4 = 64 < 304` the wave
   under-occupancy is the bottleneck; rect doubles the grid to
   `64 * 2 = 128`, still under 304 but +2x improvement. Optional
   follow-up: swizzle group-M to bring this up further.

Estimated next-cycle effort: 1.5 days kernel, 0.5 day Python, 0.5 day
validation. Total ~2.5 days — matches R27 Dev D's audit estimate.

Files touched this cycle
------------------------
  + analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp
  + analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc
  + analysis/fp8_gemm/mi350x/r28d_validate.py (new)
  + analysis/fp8_gemm/mi350x/r28d_findings.md (this file, new)

Print stamp: `=== R28 DEV D SCAFFOLDING ONLY ===`
(per task spec — Stage 1+2 PASS, Stage 3 FAIL as expected,
documented failure mode for next cycle.)
