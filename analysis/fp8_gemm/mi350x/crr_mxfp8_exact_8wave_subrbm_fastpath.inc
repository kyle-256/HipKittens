#pragma once

// R33 Dev D — Stage 1b: sub-RBM (RBM=32) V2-CRR exact 8-wave fastpath
// scaffolding. This is a parallel template to crr_mxfp8_exact_8wave_fastpath.inc
// with the operand-tile M-dim (RBM) halved from 64 to 32. The default
// fastpath remains the source of truth for RBM=64 mode (MXFP8_CRR_RBM=64);
// this file only provides the sub-RBM kernel when MXFP8_CRR_RBM=32.
//
// **Stage 1b deliverable** (this cycle): clean compile + parallel template
// exists with the new register/accumulator types + dispatch entry.
// **Stage 2 deliverable** (R34+): correctness on at least one shape (V2-CRR
// 8192^3) — requires looping each warp twice over the M-dim now that each
// warp covers RBM_SUB=32 rows (half the previous coverage), AND host-side
// scale layout audit (the slab geometry `slab_bytes_a = 4 * 32 * padded_k_blocks
// = 128 * padded_k_blocks` in the square kernel is computed for PC_A=4 from
// crr_a_pack_count=RBM/32; with RBM_SUB=32 this becomes PC_A=1 → 32 *
// padded_k_blocks, which would invalidate any host preshuffle that targets
// the 128-byte slab).
// **Stage 3 deliverable** (R34+ perf): re-run R31 Dev B + R32 Dev B SB
// pipelining matrix on the sub-RBM kernel and confirm A_col_reg dropping
// from 32 → 16 VGPR/wave eliminates the 26-lane spill barrier.
//
// SCOPE / CAVEATS
// ---------------
// - Only the V2 SCALE_VERSION=2 path is provided (PRESHUFFLED_QUANT=true).
//   V1 path is not scaffolded here (no consumers in the SHIP perf cell).
// - The kernel body is a STUB: it issues a single grid-cooperative early
//   return so the compiler emits the symbol but the kernel does no work.
//   This is intentional for Stage 1b — the goal is "compiles + symbol
//   present in .so", not correctness. Stage 2 fills in the loop body.
// - Register tile types use _SUBRBM suffix to avoid collision with the
//   square types (A_col_reg, B_col_reg) defined in kernel_mxfp8_layouts.cpp.
// - WARPS_M=2 is preserved: each warp now covers RBM_SUB=32 of the
//   block's BLK/2=128 M-rows that the warp owns, so each warp must loop
//   over (HB / RBM_SUB) = 4 strides instead of (HB / RBM) = 2 strides.
//   Stage 2 will implement this loop; Stage 1b leaves it as a // TODO(R34).
// - The accumulator tile shrinks from 64x32=2KB (32 fp32/lane) to
//   32x32=1KB (16 fp32/lane) per accumulator, so cA/cB/cC/cD total
//   collapses from 128 → 64 VGPR/wave (saving 64 VGPR — exactly what
//   R32 Dev B identified as the headroom needed for SB pipelining).
//
// Reference (R32 Dev B closure r32b_findings.md §8):
// > Future LDS-reduction work for V2-CRR must come from a different lever:
// > either smaller tile geometries (rewrite, not a macro flip) or per-wave
// > LDS partition (rewrite, breaking the load/store helpers).
// This file IS that "smaller tile geometries" lever as a parallel template.

#if defined(MXFP8_CRR_RBM) && (MXFP8_CRR_RBM == 32)

#ifndef MXFP8_CRR_SUBRBM_8WAVE_FAST_ENABLE
#define MXFP8_CRR_SUBRBM_8WAVE_FAST_ENABLE 1
#endif

#if MXFP8_CRR_SUBRBM_8WAVE_FAST_ENABLE

// Sanity: this sub-RBM kernel only makes sense when the global
// parametrization matches BLK=256, BK=128, WARPS_M=2, WARPS_N=4 (same as
// square fastpath). Note that RBM is still the *default* 64 here because
// the global RBM constexpr is BLK/WARPS_M/2 — the sub-RBM kernel uses
// RBM_SUB=32 INTERNALLY without rebinding the global RBM (so the rest of
// the translation unit's helpers that depend on RBM=64 keep working).
static_assert(BLK == 256, "MXFP8 sub-RBM CRR fast path requires BLK=256");
static_assert(BK == 128, "MXFP8 sub-RBM CRR fast path requires BK=128");
static_assert(WARPS_M == 2, "MXFP8 sub-RBM CRR fast path requires WARPS_M=2");
static_assert(WARPS_N == 4, "MXFP8 sub-RBM CRR fast path requires WARPS_N=4");
static_assert(!CRR_ROW_SHARED_TRANSPOSE,
              "Sub-RBM CRR fast path only supports strict shared layout");
static_assert(!CRR_USE_V3_SWIZZLE,
              "Sub-RBM CRR fast path is wired for v2/v2a shared layouts");

// Sub-RBM constants. RBM_SUB halves the M-dim of each warp's register
// tile. RBN_SUB matches RBN (no N-dim change yet — could be added in a
// later cycle as an orthogonal lever).
constexpr int RBM_SUB = 32;
constexpr int RBN_SUB = RBN;  // 32 today

// HB_M / HB unchanged (256/2 = 128). Strides per warp:
//   M_strides_per_warp = HB / RBM_SUB = 4 (was 2 at RBM=64).
// Stage 2 will use this as the inner loop count.
constexpr int CRR_SUBRBM_M_STRIDES = HB / RBM_SUB;
static_assert(CRR_SUBRBM_M_STRIDES == 4,
              "Sub-RBM expects 4 M-strides per warp (HB=128 / RBM_SUB=32)");

// Sub-RBM register tile types. These are NEW types (suffix _subrbm) — they
// do NOT replace the square A_col_reg/B_col_reg in kernel_mxfp8_layouts.cpp.
using A_col_reg_subrbm = rt_fp8e4m3<BK, RBM_SUB, col_l, rt_128x16_s>;  // 128x32
using B_col_reg_subrbm = rt_fp8e4m3<BK, RBN_SUB, col_l, rt_128x16_s>;  // 128x32

// Pack counts for sub-RBM. crr_a_pack_count_subrbm = RBM_SUB / 32 = 1
// (was 2 at RBM=64). This is the load-loop count delta that breaks the
// existing scale slab math without a host-side preshuffle update.
constexpr int crr_a_pack_count_subrbm = RBM_SUB / 32;             // 1
constexpr int crr_b_pack_count_subrbm = (RBN_SUB + 31) / 32;      // 1
static_assert(crr_a_pack_count_subrbm == 1,
              "Sub-RBM expects A pack count 1 (was 2 at RBM=64)");

// Sub-RBM B-side load helper: re-use the existing load_col_from_v2_st
// (RBN-sized so unaffected by RBM change). A-side helper reuses
// load_col_from_v2a_st but with the smaller RBM_SUB-sized destination.
// No new helper needed for Stage 1b (the existing template-on-RT helpers
// dispatch on the destination tile size).

// Sub-RBM mma_scaled_base: forwards to the existing column-major MMA
// dispatcher used by crr_mma_scaled_base. The 16x16x128 MFMA intrinsic
// shape is INDEPENDENT of the surrounding accumulator tile size, so the
// per-quadrant call is identical — only the tile-index range differs
// (acc_h = RBM_SUB/16 = 2 instead of 4, acc_w = RBN_SUB/16 = 2 unchanged).
//
// Stage 1b: marked TODO(R34) — the existing crr_mma_scaled_base relies
// on `reinterpret_cast<const A_row_reg&>(a)` to bridge the col-major
// register-tile layout to the MMA's row-major operand expectation, and
// A_row_reg is hardcoded to RBM=64. A sub-RBM version requires a parallel
// A_row_reg_subrbm = rt_fp8e4m3<RBM_SUB, BK, row_l, rt_16x128_s>` and
// associated mma_ABt_base_scaled overload that accepts the smaller tile.
// For Stage 1b we only declare the helper signature so future authors
// see the call site; the body is left as a deliberate compile failure
// when MXFP8_CRR_SUBRBM_PROBE=1, demonstrating the static_assert chain
// that fires (see r33d_findings.md Stage 1a row "crr_mma_scaled_base").
template<int opsel_a, int opsel_b>
__device__ __forceinline__ void crr_mma_scaled_base_subrbm(
    rt_fl<RBM_SUB, RBN_SUB, col_l, rt_16x16_s>& acc,
    const A_col_reg_subrbm& a,
    const B_col_reg_subrbm& b,
    int n,
    int m,
    const fp8e8m0_4 a_scale_pack,
    const fp8e8m0_4 b_scale_pack)
{
#if MXFP8_CRR_SUBRBM_PROBE
    // PROBE=1 mode: try to wire to the existing intrinsic. Will fail to
    // resolve mma_AB_base_scaled because A_col_reg_subrbm doesn't match
    // any registered operand-tile shape. Documenting the failure mode is
    // the intent of Stage 1b probe.
    mma_AB_base_scaled<opsel_a, opsel_b>(
        acc.tiles[n][m],
        a.tiles[0][n],
        b.tiles[0][m],
        acc.tiles[n][m],
        a_scale_pack,
        b_scale_pack
    );
#else
    // Stage 1b non-probe build: helper is dead (kernel stub does an
    // early return, never calls this). The body uses the parameters to
    // suppress unused-warnings while keeping the helper signature stable
    // for future Stage 2 implementation.
    (void)acc; (void)a; (void)b; (void)n; (void)m;
    (void)a_scale_pack; (void)b_scale_pack;
#endif
}

// Sub-RBM mma_scaled fixed-phase wrapper. RBM_SUB=32 → acc_h = RBM_SUB/16 = 2
// (was 4 at RBM=64). Pack count for A is 1 (was 2). The 8-MMA chain
// collapses to a 4-MMA chain; the "second pack" half of the original
// fixed_phase helper becomes dead.
template<int A_PACK_COUNT, int B_PACK_COUNT>
__device__ __forceinline__ void crr_mma_scaled_from_packs_fixed_phase_subrbm(
    rt_fl<RBM_SUB, RBN_SUB, col_l, rt_16x16_s>& acc,
    const A_col_reg_subrbm& a,
    const B_col_reg_subrbm& b,
    const fp8e8m0_4 (&a_scale_packs)[A_PACK_COUNT],
    const fp8e8m0_4 (&b_scale_packs)[B_PACK_COUNT])
{
    static_assert(RBM_SUB == 32 && RBN_SUB == 32,
                  "fixed_phase_subrbm helper expects 32x32 tile");
    static_assert(A_PACK_COUNT >= 1 && B_PACK_COUNT >= 1);
    // 4-MMA chain (was 8 at RBM=64). The two halves of the original
    // chain (a0_phase[0] vs a0_phase[1]) collapse to a single half.
    crr_mma_scaled_base_subrbm<0, 0>(acc, a, b, 0, 0, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_subrbm<0, 1>(acc, a, b, 0, 1, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_subrbm<1, 0>(acc, a, b, 1, 0, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_subrbm<1, 1>(acc, a, b, 1, 1, a_scale_packs[0], b_scale_packs[0]);
}

// Sub-RBM K-phase variant.
template<int K_PHASE, int A_PACK_COUNT, int B_PACK_COUNT>
__device__ __forceinline__ void crr_mma_scaled_phase_subrbm(
    rt_fl<RBM_SUB, RBN_SUB, col_l, rt_16x16_s>& acc,
    const A_col_reg_subrbm& a,
    const B_col_reg_subrbm& b,
    const fp8e8m0_4 (&a_scale_packs)[A_PACK_COUNT],
    const fp8e8m0_4 (&b_scale_packs)[B_PACK_COUNT])
{
    static_assert(K_PHASE == 0 || K_PHASE == 1);
    static_assert(RBM_SUB == 32 && RBN_SUB == 32);
    static_assert(A_PACK_COUNT >= 1 && B_PACK_COUNT >= 1);
    constexpr int lo = K_PHASE * 2;
    constexpr int hi = K_PHASE * 2 + 1;
    crr_mma_scaled_base_subrbm<lo, lo>(acc, a, b, 0, 0, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_subrbm<lo, hi>(acc, a, b, 0, 1, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_subrbm<hi, lo>(acc, a, b, 1, 0, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_subrbm<hi, hi>(acc, a, b, 1, 1, a_scale_packs[0], b_scale_packs[0]);
}

// Sub-RBM CRR exact 8-wave scaled kernel STUB.
// Stage 1b: emits the symbol with all sub-RBM types instantiated so the
// compiler validates them, but the K-loop body is left as a TODO(R34).
// The stub returns immediately after declaring the new register tile
// types and accumulators; this is sufficient to confirm:
//   (a) sub-RBM types compile cleanly
//   (b) sub-RBM mma helper chain instantiates
//   (c) symbol appears in the .so
//   (d) NO call site is wired in the dispatcher (so it's dead code in
//       the default and sub-RBM builds)
//
// Stage 2 (R34+) will:
//   1. Implement the 4-stride M-loop per warp (HB / RBM_SUB = 4).
//   2. Update host-side scale preshuffle to match the new slab geometry
//      (slab_bytes_a = 32 * padded_k_blocks instead of 128).
//   3. Re-run R31/R32 SB pipelining matrix on this kernel.
template<bool PRESHUFFLED_QUANT, int SCALE_VERSION = 2>
__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void crr_exact_8wave_scaled_subrbm_kernel(const layout_globals g)
{
    using ST_crr_a = ST_v2a;
    using ST_crr_b = ST_v2;

    constexpr int crr_a_pack_count_local = crr_a_pack_count_subrbm;  // 1
    constexpr int crr_b_pack_count_local = crr_b_pack_count_subrbm;  // 1

    // Suppress unused-warning for stub body (Stage 1b only).
    (void)g;

    // Instantiate the new register tile types so the compiler validates
    // them and emits the relevant MFMA intrinsic shapes. These are the
    // load-bearing types whose VGPR cost is the entire reason for this
    // kernel:
    //   A_col_reg_subrbm: 128x32 fp8e4m3 = 4KB tile / 64 lanes = 64 B/lane
    //                                    = 16 VGPR/wave (was 32 at RBM=64)
    //   cA/cB/cC/cD:      32x32 fp32     = 4KB tile / 64 lanes = 64 B/lane
    //                                    = 16 VGPR/wave (was 32 at RBM=64)
    //   Total cA+cB+cC+cD = 64 VGPR/wave (was 128 — saving 64 VGPR exactly
    //   matches R32 Dev B's spill threshold).
    A_col_reg_subrbm a_subrbm;
    B_col_reg_subrbm b0_subrbm, b1_subrbm;
    rt_fl<RBM_SUB, RBN_SUB, col_l, rt_16x16_s> cA_sub, cB_sub, cC_sub, cD_sub;
    zero(cA_sub);
    zero(cB_sub);
    zero(cC_sub);
    zero(cD_sub);
    (void)a_subrbm; (void)b0_subrbm; (void)b1_subrbm;

    // PROBE=1 only: touch one mma helper to force instantiation. Default
    // PROBE=0 leaves the kernel as a pure type-validation stub (kernel
    // body is empty after the zero(cA/cB/cC/cD) sequence above, so the
    // resource report will not show any MFMA usage but the type
    // instantiations are validated by the compiler).
#if MXFP8_CRR_SUBRBM_PROBE
    {
        fp8e8m0_4 a_scale_packs_dummy[crr_a_pack_count_local] = {};
        fp8e8m0_4 b_scale_packs_dummy[crr_b_pack_count_local] = {};
        crr_mma_scaled_from_packs_fixed_phase_subrbm(
            cA_sub, a_subrbm, b0_subrbm, a_scale_packs_dummy, b_scale_packs_dummy);
        crr_mma_scaled_phase_subrbm<0>(
            cB_sub, a_subrbm, b0_subrbm, a_scale_packs_dummy, b_scale_packs_dummy);
        crr_mma_scaled_phase_subrbm<1>(
            cC_sub, a_subrbm, b0_subrbm, a_scale_packs_dummy, b_scale_packs_dummy);
    }
#endif

    // Stage 1b stub: early return. Stage 2 (R34+) replaces this with the
    // K-loop body translated from crr_exact_8wave_scaled_kernel with:
    //   - the 4-stride M-loop per warp
    //   - sub-RBM types throughout
    //   - V2 scale slab geometry updated for slab_bytes_a = 32 *
    //     padded_k_blocks (from PC_A=1 instead of PC_A=4)
    return;
}

// Host-side dispatch entry: NOT wired into gemm_*_pq_v2 path in
// kernel_mxfp8_layouts.cpp. Kept here so future authors can switch the
// dispatcher over with a single edit.
__host__ inline bool crr_can_use_exact_8wave_scaled_subrbm(const layout_globals& g)
{
    // Same shape predicate as the square fastpath: only fires when the
    // compile-time M_DIM/N_DIM/K_DIM matches the runtime shape. Stage 1b
    // keeps the predicate identical so the dispatcher swap is a one-line
    // change in the future.
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM;
}

template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch_crr_exact_8wave_scaled_v2_subrbm(const layout_globals& g)
{
    const dim3 grid((g.m / BLK) * (g.n / BLK));
    crr_exact_8wave_scaled_subrbm_kernel<PRESHUFFLED_QUANT, 2>
        <<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

// Force-instantiation probe: when MXFP8_CRR_SUBRBM_PROBE=1 we emit a
// tiny host stub that takes the address of the dispatcher, which forces
// the kernel template to be instantiated and the symbol to appear in
// the resource-usage report. Default 0 (no probe — Stage 1b builds
// just need the file to parse). Set to 1 to confirm the kernel
// generates correct resource numbers for VGPR/LDS budget validation.
#ifndef MXFP8_CRR_SUBRBM_PROBE
#define MXFP8_CRR_SUBRBM_PROBE 0
#endif
#if MXFP8_CRR_SUBRBM_PROBE
__host__ inline void* crr_subrbm_probe_addr_handle()
{
    return reinterpret_cast<void*>(&dispatch_crr_exact_8wave_scaled_v2_subrbm<true>);
}
#endif

#endif  // MXFP8_CRR_SUBRBM_8WAVE_FAST_ENABLE

#endif  // MXFP8_CRR_RBM == 32
