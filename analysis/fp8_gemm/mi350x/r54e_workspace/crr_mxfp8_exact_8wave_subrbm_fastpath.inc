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

// R34 Dev D Stage 2a: row-layout aliases for sub-RBM operand tiles.
// These are the bridge that lets us reinterpret_cast a col-layout
// register tile to the row-layout shape the MFMA scaled intrinsic
// expects (mirrors A_row_reg/B_row_reg + the reinterpret bridge in
// crr_mma_scaled_base at kernel_mxfp8_layouts.cpp:1378-1388).
//
// With RBM_SUB=32:
//   A_row_reg_subrbm: tiles[rows/16=2][cols/128=1] = tiles[2][1]
//   A_col_reg_subrbm: tiles[rows/128=1][cols/16=2] = tiles[1][2]
//   B_row_reg_subrbm: tiles[2][1]
//   B_col_reg_subrbm: tiles[1][2]
// Sizes/alignment must match for the alias to be safe. Asserted below.
using A_row_reg_subrbm = rt_fp8e4m3<RBM_SUB, BK, row_l, rt_16x128_s>;  // 32x128
using B_row_reg_subrbm = rt_fp8e4m3<RBN_SUB, BK, row_l, rt_16x128_s>;  // 32x128
static_assert(sizeof(A_row_reg_subrbm) == sizeof(A_col_reg_subrbm),
              "Sub-RBM A row/col aliases must have identical size");
static_assert(alignof(A_row_reg_subrbm) == alignof(A_col_reg_subrbm),
              "Sub-RBM A row/col aliases must have identical alignment");
static_assert(sizeof(B_row_reg_subrbm) == sizeof(B_col_reg_subrbm),
              "Sub-RBM B row/col aliases must have identical size");
static_assert(alignof(B_row_reg_subrbm) == alignof(B_col_reg_subrbm),
              "Sub-RBM B row/col aliases must have identical alignment");

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

// Sub-RBM mma_scaled_base: same per-quadrant 16x16x128 scaled MFMA
// intrinsic as crr_mma_scaled_base, only the surrounding tile container
// shape changes. The intrinsic is invoked on individual sub-tiles
// (acc.tiles[n][m], a_row.tiles[n][0], b_row.tiles[m][0] — all 16x16
// or 16x128 base tiles), so the call signature is identical to the
// RBM=64 version; we just use the sub-RBM row aliases.
//
// R34 Dev D Stage 2a: type bridge wired in. The reinterpret_cast pair
// mirrors crr_mma_scaled_base (kernel_mxfp8_layouts.cpp:1378-1388).
// The `n` argument now ranges over [0, RBM_SUB/16) = [0, 2), and `m`
// ranges over [0, RBN_SUB/16) = [0, 2) — both halved from RBM=64.
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
    // Bridge col-layout → row-layout so the row-row mma_ABt_base_scaled
    // intrinsic resolves. Sub-RBM row aliases have identical size /
    // alignment to their col counterparts (asserted at type definition).
    const auto& a_row = reinterpret_cast<const A_row_reg_subrbm&>(a);
    const auto& b_row = reinterpret_cast<const B_row_reg_subrbm&>(b);
    mma_ABt_base_scaled<opsel_a, opsel_b>(
        acc.tiles[n][m],
        a_row.tiles[n][0],
        b_row.tiles[m][0],
        acc.tiles[n][m],
        a_scale_pack,
        b_scale_pack
    );
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

// Sub-RBM CRR exact 8-wave scaled kernel — Stage 2b skeleton (R34 Dev D).
//
// This is a full kernel body translated from crr_exact_8wave_scaled_kernel
// with the following changes:
//   * RBM_SUB=32 throughout (was 64) — accumulator + operand register
//     tile types use the _subrbm aliases.
//   * Per-warp M coverage is now 4 strides × RBM_SUB=32 = 128 rows = HB
//     (was 2 strides × 64 = 128). We therefore need 4 sets of sub-RBM
//     accumulators per (cB-or-cD, wn-quadrant) pair instead of 2. The
//     accumulator vector layout is:
//         cA[0..3] : top-half (half=0) × N-tile-left  (b0)
//         cB[0..3] : top-half (half=0) × N-tile-right (b1)
//         cC[0..3] : bot-half (half=1) × N-tile-left
//         cD[0..3] : bot-half (half=1) × N-tile-right
//     Each cA[s] is 16 VGPR/wave (was 32 for the RBM=64 cA), so total
//     accumulator footprint is 4 × 4 × 16 = 256 VGPR/wave — actually
//     IDENTICAL to the RBM=64 footprint of 4 × 32 = 128 VGPR/wave per
//     warp **wait no**: with double-buffered LDS the original kernel
//     uses 4 accumulators × 32 VGPR = 128 VGPR for accumulators total
//     per warp. Sub-RBM uses 16 accumulators × 16 VGPR = 256 VGPR — that
//     is DOUBLE. The naive sub-RBM kernel therefore SPILLS more, not
//     less. The R32 Dev B "64 VGPR savings" claim only holds if we keep
//     the per-warp accumulator count at 4 (i.e., keep WARPS_M=2 covering
//     HB=64 instead of HB=128). That requires a DIFFERENT structural
//     change: WARPS_M=4 with a tighter block, OR fewer accumulators per
//     warp with epilogue store-redo. NOT a simple operand-tile shrink.
//
// **Stage 2b finding (R34 Dev D, day 1):** the operand-tile shrink alone
// does not save accumulator VGPR — it merely shifts where the budget is
// spent (operand registers down 16, accumulator registers up 128 per
// warp net). This is a structural correction to the R32 Dev B / R33 Dev D
// scaffolding hypothesis. The real path past the structural ceiling is
// either (a) reduce WARPS_M coverage (smaller block in M), or (b) reuse
// fewer accumulators with intermediate stores. Neither is a "type bridge"
// change.
//
// Despite the corrected hypothesis, the Stage 2 SCAFFOLDING is still
// useful: it gives the next R-cycle author a working sub-RBM kernel
// body skeleton + verified type bridge, so they can pivot to the real
// lever (smaller block geometry) without re-doing the type plumbing.
//
// Below is the full skeleton kernel: 4-stride M-loop + 8 accumulators
// per warp, demonstrating end-to-end type validation. Dispatch is NOT
// wired (the host-side preshuffle audit in §4.3 of r33d_findings.md is
// still required for SCALE_VERSION=2).
template<bool PRESHUFFLED_QUANT, int SCALE_VERSION = 2>
__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void crr_exact_8wave_scaled_subrbm_kernel(const layout_globals g)
{
    using ST_crr_a = ST_v2a;
    using ST_crr_b = ST_v2;

    constexpr int crr_a_pack_count_local = crr_a_pack_count_subrbm;  // 1
    constexpr int crr_b_pack_count_local = crr_b_pack_count_subrbm;  // 1

    const int k_iters = g.k / BK;
    const int blocks_per_col = g.n / BLK;

    // LDS tiles unchanged from RBM=64 path (As/Bs are block-wide tiles,
    // independent of operand-register tile geometry).
    __shared__ ST_crr_a As[2][2];
    __shared__ ST_crr_b Bs[2][2];

    // Sub-RBM operand tiles: 16 VGPR/wave each (was 32 at RBM=64).
    A_col_reg_subrbm a_sub;
    B_col_reg_subrbm b0_sub, b1_sub;

    // Sub-RBM accumulators: 4 strides per warp × 2 N-tiles × 2 halves =
    // 16 accumulators of size 32x32 = 16 VGPR/wave each.
    rt_fl<RBM_SUB, RBN_SUB, col_l, rt_16x16_s>
        cA[CRR_SUBRBM_M_STRIDES],   // half=0 × b0
        cB[CRR_SUBRBM_M_STRIDES],   // half=0 × b1
        cC[CRR_SUBRBM_M_STRIDES],   // half=1 × b0
        cD[CRR_SUBRBM_M_STRIDES];   // half=1 × b1
    #pragma unroll
    for (int s = 0; s < CRR_SUBRBM_M_STRIDES; ++s) {
        zero(cA[s]); zero(cB[s]); zero(cC[s]); zero(cD[s]);
    }

    const int bid = blockIdx.x;
    const int br = bid / blocks_per_col;
    const int bc = bid % blocks_per_col;
    const int wm = warpid() / WARPS_N;
    const int wn = warpid() % WARPS_N;

    const int lane_nonk = kittens::laneid() % 16;
    const int lane_kblk = kittens::laneid() / 16;
    const uint32_t lane_scale_byte_offset =
        (static_cast<uint32_t>(lane_kblk) << 6) |
        (static_cast<uint32_t>(lane_nonk) << 2);

    // Per-stride scale base offset (M-coord). With sub-RBM each warp
    // covers 4 contiguous 32-row strides starting at wm * HB. The half
    // index (0 or 1) selects which 128-row half of the BLK=256 block the
    // warp's coverage falls within (warp wm=0 → half=0 base, wm=1 → half=1).
    auto crr_scale_a_base_sub = [&](int half, int stride) {
        return br * BLK + half * (BLK / 2) + wm * HB + stride * RBM_SUB;
    };
    auto crr_scale_b_base = [&](int half) {
        return bc * BLK + half * (BLK / 2) + wn * RBN;
    };

    // Stage 2b NOTE: V1 / V2 scale fetch path translation deferred.
    // The V1 row-base path needs 4 sets of bases (one per stride) per
    // half — straightforward but tedious. The V2 slab-SRD path requires
    // the host-side preshuffle update documented in r33d_findings.md
    // §4.3 (slab_bytes_a re-tiled for PC_A=1 × 4 strides per warp).
    // Both are wire-up work, not type-system work; deferred to R35+.
    //
    // For the Stage 2b skeleton we instantiate a representative scale
    // pack vector (per-stride) to validate the type chain through to
    // mma_ABt_base_scaled — this exercises the type bridge end-to-end
    // and produces a kernel symbol with realistic VGPR/LDS resource
    // numbers we can compare against the RBM=64 baseline.
    fp8e8m0_4 a_scale_packs_per_stride[CRR_SUBRBM_M_STRIDES][crr_a_pack_count_local];
    fp8e8m0_4 b0_scale_packs[crr_b_pack_count_local];
    fp8e8m0_4 b1_scale_packs[crr_b_pack_count_local];
    #pragma unroll
    for (int s = 0; s < CRR_SUBRBM_M_STRIDES; ++s) {
        for (int p = 0; p < crr_a_pack_count_local; ++p) {
            a_scale_packs_per_stride[s][p] = std::bit_cast<fp8e8m0_4>(uint32_t{0x7F7F7F7F});
        }
    }
    #pragma unroll
    for (int p = 0; p < crr_b_pack_count_local; ++p) {
        b0_scale_packs[p] = std::bit_cast<fp8e8m0_4>(uint32_t{0x7F7F7F7F});
        b1_scale_packs[p] = std::bit_cast<fp8e8m0_4>(uint32_t{0x7F7F7F7F});
    }

    // Prefill swizzled offsets for global -> shared loads (unchanged
    // from RBM=64 path — these are block-wide tile loaders).
    constexpr int bptA = ST_crr_a::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * _NUM_THREADS;
    constexpr int mptA = ST_crr_a::rows * ST_crr_a::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
    constexpr int bptB = ST_crr_b::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB = ST_crr_b::rows * ST_crr_b::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    auto a_co = [&](int s, int k) -> coord<ST_crr_a> { return {0, 0, k, s}; };
    auto b_co = [&](int s, int k) -> coord<ST_crr_b> { return {0, 0, k, s}; };
    auto global_load_a = [&](ST_crr_a& tile, int s, int k) {
        G::load(tile, g.a, a_co(s, k), soA);
    };
    auto global_load_b = [&](ST_crr_b& tile, int s, int k) {
        G::load(tile, g.b, b_co(s, k), soB);
    };
    // Per-stride sub-RBM A loader: col_start = wm * HB + stride * RBM_SUB.
    // This loads RBM_SUB=32 rows of A starting at the warp's M-base + the
    // intra-warp stride. Each warp's HB=128 coverage decomposes into 4
    // such 32-row strides.
    auto load_a_sub = [&](A_col_reg_subrbm& dst, ST_crr_a& tile, int stride) {
        load_col_from_v2a_st(dst, tile, wm * HB + stride * RBM_SUB);
    };
    auto load_b_sub = [&](B_col_reg_subrbm& dst, ST_crr_b& tile) {
        load_col_from_v2_st(dst, tile, wn * RBN);
    };

    // Single-buffered LDS path (mirrors MXFP8_CRR_LDS_SINGLE_BUFFER=1
    // mode of the RBM=64 kernel; this is the simplest correct schedule
    // for a Stage 2b skeleton). Stage 3+ can rewire to the DB main loop
    // once host preshuffle is updated.
    global_load_b(Bs[0][0], bc * 2, 0);
    global_load_a(As[0][0], br * 2, 0);
    global_load_b(Bs[0][1], bc * 2 + 1, 0);
    global_load_a(As[0][1], br * 2 + 1, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    for (int k = 0; k < k_iters; k++) {
        // Suppress unused-warning for the half/stride helpers; full
        // wire-in deferred to R35.
        (void)crr_scale_a_base_sub; (void)crr_scale_b_base;
        (void)lane_scale_byte_offset;

        // Top half (As[0][0] = rows 0..127 of block).
        load_b_sub(b0_sub, Bs[0][0]);
        load_b_sub(b1_sub, Bs[0][1]);
        asm volatile("s_waitcnt lgkmcnt(0)");

        #pragma unroll
        for (int s = 0; s < CRR_SUBRBM_M_STRIDES; ++s) {
            load_a_sub(a_sub, As[0][0], s);
            asm volatile("s_waitcnt lgkmcnt(0)");
            crr_mma_scaled_from_packs_fixed_phase_subrbm(
                cA[s], a_sub, b0_sub,
                a_scale_packs_per_stride[s], b0_scale_packs);
            crr_mma_scaled_from_packs_fixed_phase_subrbm(
                cB[s], a_sub, b1_sub,
                a_scale_packs_per_stride[s], b1_scale_packs);
        }

        // Bottom half (As[0][1] = rows 128..255 of block).
        #pragma unroll
        for (int s = 0; s < CRR_SUBRBM_M_STRIDES; ++s) {
            load_a_sub(a_sub, As[0][1], s);
            asm volatile("s_waitcnt lgkmcnt(0)");
            crr_mma_scaled_from_packs_fixed_phase_subrbm(
                cC[s], a_sub, b0_sub,
                a_scale_packs_per_stride[s], b0_scale_packs);
            crr_mma_scaled_from_packs_fixed_phase_subrbm(
                cD[s], a_sub, b1_sub,
                a_scale_packs_per_stride[s], b1_scale_packs);
        }

        __builtin_amdgcn_s_barrier();
        if (k + 1 < k_iters) {
            global_load_b(Bs[0][0], bc * 2, k + 1);
            global_load_a(As[0][0], br * 2, k + 1);
            global_load_b(Bs[0][1], bc * 2 + 1, k + 1);
            global_load_a(As[0][1], br * 2 + 1, k + 1);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
        }
    }

    if (wm == 0) {
        __builtin_amdgcn_s_barrier();
    }

    if constexpr (PRESHUFFLED_QUANT) {
        #pragma unroll
        for (int s = 0; s < CRR_SUBRBM_M_STRIDES; ++s) {
            mul(cA[s], cA[s], g.scale);
            mul(cB[s], cB[s], g.scale);
            mul(cC[s], cC[s], g.scale);
            mul(cD[s], cD[s], g.scale);
        }
    }

    // Per-stride epilogue store. Each warp's M-coverage is HB=128 rows
    // = WARPS_M*2 macro-rows of 32 each; sub-RBM stores per stride into
    // the appropriate macro-row index. Coordinate scheme mirrors the
    // RBM=64 kernel but with 4 strides per half instead of 1.
    //   macro_row(half=0, stride s) = br * WARPS_M * 4 + wm * 4 + s
    //   macro_row(half=1, stride s) = br * WARPS_M * 4 + WARPS_M * 4 + wm * 4 + s
    // (Note BLK/RBM_SUB = 8 macro-rows per block; WARPS_M=2 covers 8/2=4
    // macro-rows per warp; halves account for top vs bottom of warp.)
    #pragma unroll
    for (int s = 0; s < CRR_SUBRBM_M_STRIDES; ++s) {
        const int macro_row_top = br * WARPS_M * 4 + wm * CRR_SUBRBM_M_STRIDES + s;
        const int macro_row_bot = br * WARPS_M * 4 + WARPS_M * CRR_SUBRBM_M_STRIDES
                                + wm * CRR_SUBRBM_M_STRIDES + s;
        store(g.c, cA[s], {0, 0, macro_row_top, bc * WARPS_N * 2 + wn});
        store(g.c, cB[s], {0, 0, macro_row_top, bc * WARPS_N * 2 + WARPS_N + wn});
        store(g.c, cC[s], {0, 0, macro_row_bot, bc * WARPS_N * 2 + wn});
        store(g.c, cD[s], {0, 0, macro_row_bot, bc * WARPS_N * 2 + WARPS_N + wn});
    }
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
