#pragma once

// R31 Dev A — Stage A1: rectangular BLK_M=256/BLK_N=128 V2-CRR exact 8-wave
// fastpath kernel. This is a parallel template to crr_mxfp8_exact_8wave_fastpath.inc
// with all N-dim parameters replaced by the rect equivalents (BLK_N, RBN_RECT,
// HB_N). The default fastpath remains the source of truth for square mode
// (MXFP8_RECT_BLK_N=128); this file only provides the rect kernel when
// MXFP8_RECT_BLK_N=64.
//
// **Stage A1a deliverable**: clean compile + parallel template exists.
// **Stage A1b deliverable**: rect build runs without GPU fault (degenerate K=128).
// **Stage A1c (stretch)**: correct numerics on at least one config.
//
// SCOPE / CAVEATS
// ---------------
// - Only the V2 SCALE_VERSION=2 path is implemented (PRESHUFFLED_QUANT=true,
//   matches dispatch_crr_exact_8wave_scaled_v2_rect). The V1 path is left as
//   a fall-through to the existing (square) kernel via the dispatcher gate.
// - B-side helper uses load_col_from_v2_st_half_rect<RT, 0> twice with col_start
//   advanced by HB_N (=64) for the second half-tile. K_HALF=1 path is invalid
//   for HB_N=64 (would address out-of-bounds N-rows) per the static_assert
//   in the helper at kernel_mxfp8_layouts.cpp:522-524.
// - crr_b_pack_count_rect = (RBN_RECT + 31) / 32 = 1 (was 1 anyway for RBN=32,
//   but the bookkeeping is now explicit). Pack-1-only loops reduce VGPR
//   pressure and may unlock 2-blocks/CU occupancy — TBD measurement.
// - V2 scale slab geometry: num_slabs_b_rect = (N/BLK_N)*WARPS_N (was
//   (N/BLK)*WARPS_N), slab_bytes_b_rect = 1 * 32 * padded_k_blocks =
//   32 * padded_k_blocks (was 64 * padded_k_blocks).
// - **Host preshuffle MUST match this geometry for correctness** (Stage A2).
//   For Stage A1a the kernel is wired but numerics will be wrong unless the
//   host preshuffle matches. SHIP gate is NOT evaluable until Stage A2.

#if defined(MXFP8_RECT_BLK_N) && (MXFP8_RECT_BLK_N == 64)

#ifndef MXFP8_CRR_RECT_8WAVE_FAST_ENABLE
#define MXFP8_CRR_RECT_8WAVE_FAST_ENABLE 1
#endif

#if MXFP8_CRR_RECT_8WAVE_FAST_ENABLE

// Sanity: this rect kernel only makes sense when the global parametrization
// matches BLK_M=256, BK=128, WARPS_M=2, WARPS_N=4 (same as square fastpath).
static_assert(BLK == 256, "MXFP8 exact 8-wave rect fast path requires BLK_M=256");
static_assert(BK == 128, "MXFP8 exact 8-wave rect fast path requires BK=128");
static_assert(WARPS_M == 2, "MXFP8 exact 8-wave rect fast path requires WARPS_M=2");
static_assert(WARPS_N == 4, "MXFP8 exact 8-wave rect fast path requires WARPS_N=4");
static_assert(BLK_N == 128, "MXFP8 exact 8-wave rect fast path requires BLK_N=128");
static_assert(RBN_RECT == 16, "MXFP8 exact 8-wave rect fast path requires RBN_RECT=16");
static_assert(HB_N == 64, "MXFP8 exact 8-wave rect fast path requires HB_N=64");
static_assert(!CRR_ROW_SHARED_TRANSPOSE, "Rect CRR fast path only supports strict shared layout");
static_assert(!CRR_USE_V3_SWIZZLE, "Rect CRR fast path is wired for v2/v2a shared layouts");

// Rect-N specific shared / register tile types. Names suffixed with _rect to
// avoid collision with the square types in kernel_mxfp8_layouts.cpp.
using ST_v2_rect = st_fp8e4m3<HB_N, BK, st_16x128_v2_s>;       // 64x128 rect B tile
using B_col_reg_rect = rt_fp8e4m3<BK, RBN_RECT, col_l, rt_128x16_s>;  // 128x16 rect

// Rect-N B-side load helper: split into two K_HALF=0 calls because HB_N=64
// only supports K_HALF=0 (helper static_assert at kernel_mxfp8_layouts.cpp:522).
// For rect mode, a "full" B-load covers BLK_N=128 N-rows in 2 K_HALF=0 issues
// of the helper, advancing col_start by HB_N (=64) between them.
//
// However in the rect kernel, the per-warp B register tile is RBN_RECT=16
// wide along N — so a single warp only needs ONE K_HALF=0 issue to fill its
// 16 N-rows starting at wn*RBN_RECT (0..16, 16..32, 32..48, 48..64). The
// _rect helper inherently handles K_HALF=1 via the same RT::width loop
// because RT::width = (RBN_RECT+15)/16 = 1 → only one iteration. The K-axis
// is handled by 2 calls to the helper (one for each K_HALF). For HB_N=64
// the second call must use a different mechanism — the helper requires
// K_HALF=0 — and there's NO valid K_HALF=1 call for the rect tile.
//
// **Resolution**: in rect mode the B shared tile is HB_N=64 N-rows ×
// BK=128 K-cols, and a single warp needs RBN_RECT=16 N-rows. The K-dim
// is BK=128, indexed as 4 inner k-positions per packed register. The
// existing K_HALF templating refers to which 64-row half of the K dim
// the helper loads from. For rect:
//   - The shared tile only contains 64 N-rows total, which is the entire
//     N-half-block (HB_N).
//   - BK=128 still means the K-dim is 128, so we still need K_HALF=0 AND
//     K_HALF=1 to cover both halves of K.
//   - The static_assert in load_col_from_v2_st_half_rect rejects K_HALF=1
//     for HB_N=64 because the helper math addresses N-rows (k_row =
//     row_off + K_HALF*64) which would go out of bounds for a 64-row tile.
//
// **The fix**: the helper's k_row math is for a 128-row tile. For HB_N=64
// the K-dim is still 128 but laid out as a *separate* tile per K-half.
// In rect mode the kernel must issue TWO ds_read_b64_tr_b8 sequences from
// TWO separate B shared tiles (one per K-half) — OR (preferred) keep the
// shared tile sized 128×128 (HB=128) but only fill the first 64 N-rows.
// For Stage A1a we take the SIMPLER approach: keep ST_v2_rect=HB_N=64 but
// reissue the existing K_HALF=0 helper for both K-halves, indexing into
// the SAME tile but from a different K-base.
//
// For Stage A1a (clean compile + no GPU fault), we provide a placeholder
// load_col_rect that calls the K_HALF=0 helper twice with the SAME tile
// and adjusts the destination register index manually. Numeric correctness
// is OUT OF SCOPE for A1a — that's Stage A2.
template<typename RT>
__device__ __forceinline__ void load_col_from_v2_st_rect(
    RT& dst, const ST_v2_rect& tile, int col_start)
{
    // Stage A1a placeholder: only K_HALF=0 path is wired correctly.
    // K_HALF=1 path requires either (a) a separate tile that holds the
    // K=64..127 slice or (b) a redesigned helper that addresses K-rows
    // directly. For Stage A1a we issue the K_HALF=0 path for both halves
    // — the second issue will load the same data but write to the
    // K_HALF=1 register slot (idx=4). Numerics are wrong but the issue
    // is well-formed at compile + runtime and will not GPU-fault.
    load_col_from_v2_st_half_rect<RT, 0, HB_N>(dst, tile, col_start);
    // K_HALF=1 stub: cannot use load_col_from_v2_st_half_rect<RT, 1, HB_N>
    // because of the helper's static_assert. For Stage A1a we manually
    // populate the second K-half slot with a duplicate of the K_HALF=0
    // load — guarantees no GPU fault, accepts wrong numerics. Stage A2
    // will replace this with a proper second-tile read.
    {
        const int laneid = kittens::laneid();
        const int row_off = ((laneid % 16) / 2) + ((laneid / 16) * 16);
        const int col_off = (laneid % 2) * 8;
        const uint32_t tile_base = reinterpret_cast<uintptr_t>(&tile.data[0]);

        // Re-issue same K_HALF=0 math but write to idx=4 (K_HALF=1 slot).
        // This is the Stage A1a stub — Stage A2 must replace with a real
        // K=64..127 load (likely from a second 64-row tile).
        constexpr int idx = 4;  // K_HALF=1 register slot
        const int k_row = row_off;  // K_HALF=0 row math (stub)
        const uint32_t stidx = k_row >> 4;
        const uint32_t base_k = tile_base + (stidx << 11) + (stidx << 7) + ((k_row & 15) << 7);
        const uint32_t sw_k   = (k_row & 7) << 4;

        #pragma unroll
        for (int j = 0; j < RT::width; j++) {
            const uint32_t nc = col_start + j * 16 + col_off;
            const uint32_t addr = base_k + (nc ^ sw_k);
            asm volatile(
                "ds_read_b64_tr_b8 %0, %2 offset:0\n"
                "ds_read_b64_tr_b8 %1, %2 offset:1024\n"
                : "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx])),
                  "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx + 2]))
                : "v"(addr)
                : "memory"
            );
        }
    }
}

// Rect-N MMA primitives. The existing crr_mma_scaled_base/dispatch/_phase/
// _from_packs_fixed_phase helpers in kernel_mxfp8_layouts.cpp are hardcoded
// to the square accumulator type rt_fl<RBM, RBN, ...> and the square B reg
// type B_col_reg. We provide _rect variants here that use the rect types.
// The body is structurally identical and just calls the underlying
// mma_AB_base_scaled / mma_ABt_base_scaled / mma_AtB_base_scaled primitives.

template<int opsel_a, int opsel_b>
__device__ __forceinline__ void crr_mma_scaled_base_rect(
    rt_fl<RBM, RBN_RECT, col_l, rt_16x16_s>& acc,
    const A_col_reg& a,
    const B_col_reg_rect& b,
    int n,
    int m,
    const fp8e8m0_4 a_scale_pack,
    const fp8e8m0_4 b_scale_pack)
{
#if CRR_A_REG_ROW_LOAD_TRANSPOSE && CRR_A_REG_ROW_LOAD_ALIAS
    const auto& a_row = reinterpret_cast<const A_row_reg&>(a);
    #if CRR_B_REG_ROW_LOAD_TRANSPOSE && CRR_B_REG_ROW_LOAD_ALIAS
    // (Not relevant for rect path — rect kernel disables row-shared transpose.)
    using B_row_reg_rect = rt_fp8e4m3<RBN_RECT, BK, row_l, rt_16x128_s>;
    const auto& b_row = reinterpret_cast<const B_row_reg_rect&>(b);
    mma_ABt_base_scaled<opsel_a, opsel_b>(
        acc.tiles[n][m],
        a_row.tiles[n][0],
        b_row.tiles[m][0],
        acc.tiles[n][m],
        a_scale_pack,
        b_scale_pack
    );
    #else
    mma_AB_base_scaled<opsel_a, opsel_b>(
        acc.tiles[n][m],
        a_row.tiles[n][0],
        b.tiles[0][m],
        acc.tiles[n][m],
        a_scale_pack,
        b_scale_pack
    );
    #endif
#else
    mma_AtB_base_scaled<opsel_a, opsel_b>(
        acc.tiles[n][m],
        a.tiles[0][n],
        b.tiles[0][m],
        acc.tiles[n][m],
        a_scale_pack,
        b_scale_pack
    );
#endif
}

template<int K_PHASE, int A_PACK_COUNT, int B_PACK_COUNT>
__device__ __forceinline__ void crr_mma_scaled_phase_rect(
    rt_fl<RBM, RBN_RECT, col_l, rt_16x16_s>& acc,
    const A_col_reg& a,
    const B_col_reg_rect& b,
    const fp8e8m0_4 (&a_scale_packs)[A_PACK_COUNT],
    const fp8e8m0_4 (&b_scale_packs)[B_PACK_COUNT])
{
    static_assert(K_PHASE == 0 || K_PHASE == 1);
    static_assert(RBM == 64 && RBN_RECT == 16);
    static_assert(A_PACK_COUNT >= 2 && B_PACK_COUNT >= 1);
    constexpr int lo = K_PHASE * 2;
    constexpr int hi = K_PHASE * 2 + 1;
    crr_mma_scaled_base_rect<lo, lo>(acc, a, b, 0, 0, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_rect<lo, hi>(acc, a, b, 0, 1, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_rect<hi, lo>(acc, a, b, 1, 0, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_rect<hi, hi>(acc, a, b, 1, 1, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_rect<lo, lo>(acc, a, b, 2, 0, a_scale_packs[1], b_scale_packs[0]);
    crr_mma_scaled_base_rect<lo, hi>(acc, a, b, 2, 1, a_scale_packs[1], b_scale_packs[0]);
    crr_mma_scaled_base_rect<hi, lo>(acc, a, b, 3, 0, a_scale_packs[1], b_scale_packs[0]);
    crr_mma_scaled_base_rect<hi, hi>(acc, a, b, 3, 1, a_scale_packs[1], b_scale_packs[0]);
}

template<int A_PACK_COUNT, int B_PACK_COUNT>
__device__ __forceinline__ void crr_mma_scaled_from_packs_fixed_phase_rect(
    rt_fl<RBM, RBN_RECT, col_l, rt_16x16_s>& acc,
    const A_col_reg& a,
    const B_col_reg_rect& b,
    const fp8e8m0_4 (&a_scale_packs)[A_PACK_COUNT],
    const fp8e8m0_4 (&b_scale_packs)[B_PACK_COUNT])
{
    static_assert(RBM == 64 && RBN_RECT == 16, "fixed_phase_rect helper expects 64x16 tile");
    static_assert(A_PACK_COUNT >= 2 && B_PACK_COUNT >= 1);
    crr_mma_scaled_base_rect<0, 0>(acc, a, b, 0, 0, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_rect<0, 1>(acc, a, b, 0, 1, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_rect<1, 0>(acc, a, b, 1, 0, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_rect<1, 1>(acc, a, b, 1, 1, a_scale_packs[0], b_scale_packs[0]);
    crr_mma_scaled_base_rect<0, 0>(acc, a, b, 2, 0, a_scale_packs[1], b_scale_packs[0]);
    crr_mma_scaled_base_rect<0, 1>(acc, a, b, 2, 1, a_scale_packs[1], b_scale_packs[0]);
    crr_mma_scaled_base_rect<1, 0>(acc, a, b, 3, 0, a_scale_packs[1], b_scale_packs[0]);
    crr_mma_scaled_base_rect<1, 1>(acc, a, b, 3, 1, a_scale_packs[1], b_scale_packs[0]);
}

// Rect-N interleave helper: identical structure to
// crr_exact_cA_with_b1_interleave_fixed_phase but uses _rect tile types
// and load_col_from_v2_st_rect. The pack count for B-rect = 1, so the
// scale pack indexing collapses to b0_phase[0] for all m-iterations.
template<int INSERT_AFTER, int A_PACK_COUNT, int B_PACK_COUNT>
__device__ __forceinline__ void crr_exact_cA_with_b1_interleave_fixed_phase_rect(
    rt_fl<RBM, RBN_RECT, col_l, rt_16x16_s>& cA,
    const A_col_reg& a,
    const B_col_reg_rect& b0,
    B_col_reg_rect& b1,
    const ST_v2_rect& b1_tile,
    int wn,
    const fp8e8m0_4 (&a0_phase)[A_PACK_COUNT],
    const fp8e8m0_4 (&b0_phase)[B_PACK_COUNT])
{
    static_assert(INSERT_AFTER >= 0 && INSERT_AFTER <= 8);
    static_assert(RBM == 64 && RBN_RECT == 16);

    __builtin_amdgcn_s_setprio(1);
    if constexpr (INSERT_AFTER == 0) { CRR_SCHED_BARRIER(); load_col_from_v2_st_rect(b1, b1_tile, wn * RBN_RECT); }
    crr_mma_scaled_base_rect<0, 0>(cA, a, b0, 0, 0, a0_phase[0], b0_phase[0]);
    if constexpr (INSERT_AFTER == 1) { CRR_SCHED_BARRIER(); load_col_from_v2_st_rect(b1, b1_tile, wn * RBN_RECT); }
    crr_mma_scaled_base_rect<0, 1>(cA, a, b0, 0, 1, a0_phase[0], b0_phase[0]);
    if constexpr (INSERT_AFTER == 2) { CRR_SCHED_BARRIER(); load_col_from_v2_st_rect(b1, b1_tile, wn * RBN_RECT); }
    crr_mma_scaled_base_rect<1, 0>(cA, a, b0, 1, 0, a0_phase[0], b0_phase[0]);
    if constexpr (INSERT_AFTER == 3) { CRR_SCHED_BARRIER(); load_col_from_v2_st_rect(b1, b1_tile, wn * RBN_RECT); }
    crr_mma_scaled_base_rect<1, 1>(cA, a, b0, 1, 1, a0_phase[0], b0_phase[0]);
    if constexpr (INSERT_AFTER == 4) { CRR_SCHED_BARRIER(); load_col_from_v2_st_rect(b1, b1_tile, wn * RBN_RECT); }
    crr_mma_scaled_base_rect<0, 0>(cA, a, b0, 2, 0, a0_phase[1], b0_phase[0]);
    if constexpr (INSERT_AFTER == 5) { CRR_SCHED_BARRIER(); load_col_from_v2_st_rect(b1, b1_tile, wn * RBN_RECT); }
    crr_mma_scaled_base_rect<0, 1>(cA, a, b0, 2, 1, a0_phase[1], b0_phase[0]);
    if constexpr (INSERT_AFTER == 6) { CRR_SCHED_BARRIER(); load_col_from_v2_st_rect(b1, b1_tile, wn * RBN_RECT); }
    crr_mma_scaled_base_rect<1, 0>(cA, a, b0, 3, 0, a0_phase[1], b0_phase[0]);
    if constexpr (INSERT_AFTER == 7) { CRR_SCHED_BARRIER(); load_col_from_v2_st_rect(b1, b1_tile, wn * RBN_RECT); }
    crr_mma_scaled_base_rect<1, 1>(cA, a, b0, 3, 1, a0_phase[1], b0_phase[0]);
    if constexpr (INSERT_AFTER == 8) { CRR_SCHED_BARRIER(); load_col_from_v2_st_rect(b1, b1_tile, wn * RBN_RECT); }
    __builtin_amdgcn_s_setprio(0);
    CRR_SCHED_BARRIER();
}

// Rect-N CRR exact 8-wave scaled kernel. Mirrors crr_exact_8wave_scaled_kernel
// but with rect-N register/shared types and rect-N V2 scale slab geometry.
template<bool PRESHUFFLED_QUANT>
__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void crr_exact_8wave_scaled_rect_kernel(const layout_globals g)
{
    using ST_crr_a = ST_v2a;
    using ST_crr_b = ST_v2_rect;

    constexpr int crr_a_pack_count = RBM / 32;                  // 2 (unchanged)
    constexpr int crr_b_pack_count_rect = (RBN_RECT + 31) / 32; // 1 (was 1 already at RBN=32 boundary, but explicit here)

    const int k_iters = g.k / BK;
    const int blocks_per_col = g.n / BLK_N;  // rect: BLK_N=128

    __shared__ ST_crr_a As[2][2];
    __shared__ ST_crr_b Bs[2][2];

    A_col_reg a;
    B_col_reg_rect b0, b1;
    rt_fl<RBM, RBN_RECT, col_l, rt_16x16_s> cA, cB, cC, cD;
    zero(cA);
    zero(cB);
    zero(cC);
    zero(cD);

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

    // A-side scale base unchanged from square (BLK_M=BLK=256).
    auto crr_scale_a_base = [&](int half) {
        return br * BLK + half * (BLK / 2) + wm * RBM;
    };
    // B-side scale base uses BLK_N (rect) instead of BLK.
    auto crr_scale_b_base_rect = [&](int half) {
        return bc * BLK_N + half * (BLK_N / 2) + wn * RBN_RECT;
    };

    fp8e8m0_4 a0_scale_packs[crr_a_pack_count];
    fp8e8m0_4 a1_scale_packs[crr_a_pack_count];
    fp8e8m0_4 b0_scale_packs[crr_b_pack_count_rect];
    fp8e8m0_4 b1_scale_packs[crr_b_pack_count_rect];
    const uint8_t* a0_scale_row_bases[crr_a_pack_count];
    const uint8_t* a1_scale_row_bases[crr_a_pack_count];
    const uint8_t* b0_scale_row_bases[crr_b_pack_count_rect];
    const uint8_t* b1_scale_row_bases[crr_b_pack_count_rect];

    // V2 wave-tile SRDs: 1 SRD per wave for A, 1 per wave for B.
    i32x4 a_v2_srsrc = {}, b_v2_srsrc = {};
    if constexpr (PRESHUFFLED_QUANT) {
        #pragma unroll
        for (int p = 0; p < crr_a_pack_count; ++p) {
            a0_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.a_scale, (crr_scale_a_base(0) + p * 32) >> 5);
            a1_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.a_scale, (crr_scale_a_base(1) + p * 32) >> 5);
        }
        #pragma unroll
        for (int p = 0; p < crr_b_pack_count_rect; ++p) {
            b0_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.b_scale, (crr_scale_b_base_rect(0) + p * 32) >> 5);
            b1_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.b_scale, (crr_scale_b_base_rect(1) + p * 32) >> 5);
        }
        // V2-CRR rect: per-wave-tile slab SRDs.
        // A side: pack_count=2 (unchanged), slab_bytes_a = 128 * padded_k_blocks.
        //   slab_idx_a = br * WARPS_M + wm.
        // B side: pack_count_rect=1, slab_bytes_b_rect = 32 * padded_k_blocks.
        //   num_slabs_b_rect = (N/BLK_N) * WARPS_N (= 16*4 for N=1024, BLK_N=128).
        //   slab_idx_b = bc * WARPS_N + wn.
        const int padded_k_blocks =
            (g.k / 32 + 7) & ~7; // round up to multiple of 8
        const size_t slab_bytes_a = 128u * padded_k_blocks;
        const size_t slab_bytes_b_rect = 32u * padded_k_blocks;
        const size_t slab_idx_a = static_cast<size_t>(br) * WARPS_M + wm;
        const size_t slab_idx_b = static_cast<size_t>(bc) * WARPS_N + wn;
        const uint8_t* a_v2_base =
            reinterpret_cast<const uint8_t*>(g.a_scale.raw_ptr) +
            slab_idx_a * slab_bytes_a;
        const uint8_t* b_v2_base =
            reinterpret_cast<const uint8_t*>(g.b_scale.raw_ptr) +
            slab_idx_b * slab_bytes_b_rect;
        auto make_v2_srd = [](const void* ptr, uint32_t bytes) -> i32x4 {
            i32x4 srd = std::bit_cast<i32x4>(
                make_buffer_resource(
                    static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(ptr)),
                    bytes,
                    0x00110000u
                )
            );
            srd[0] = __builtin_amdgcn_readfirstlane(srd[0]);
            srd[1] = __builtin_amdgcn_readfirstlane(srd[1]);
            srd[2] = __builtin_amdgcn_readfirstlane(srd[2]);
            srd[3] = __builtin_amdgcn_readfirstlane(srd[3]);
            return srd;
        };
        a_v2_srsrc = make_v2_srd(a_v2_base,
                                 static_cast<uint32_t>(slab_bytes_a));
        b_v2_srsrc = make_v2_srd(b_v2_base,
                                 static_cast<uint32_t>(slab_bytes_b_rect));
    }

    auto load_raw_scales = [&](int k_pair) __attribute__((always_inline)) {
        if constexpr (PRESHUFFLED_QUANT) {
            // V2-CRR rect: 1 b128 fetches all 4 A packs (unchanged).
            // For B (PC=1 rect), 1 b32 fetches both B-half packs as a single
            // 4-byte word: {b0p0_lo16, b1p0_lo16}? Actually with PC=1 the V2
            // layout interleaves b0p0 and b1p0 the same way as before but with
            // half the slab size. We use b32 instead of b64.
            //   A (PC=2): voff = lane_kblk*256 + lane_nonk*16, soff = k_pair*1024
            //   B (PC=1): voff = lane_kblk*64  + lane_nonk*4,  soff = k_pair*256
            const uint32_t a_voff =
                (static_cast<uint32_t>(lane_kblk) << 8) |
                (static_cast<uint32_t>(lane_nonk) << 4);
            const uint32_t a_soff = static_cast<uint32_t>(k_pair) << 10;
            const __uint128_t a_raw =
                llvm_amdgcn_raw_buffer_load_b128(a_v2_srsrc, a_voff, a_soff, MXFP8_CRR_V2_SCALE_CACHEPOLICY);
            a0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw      ));
            a1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 32));
            if constexpr (crr_a_pack_count > 1) {
                a0_scale_packs[1] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 64));
                a1_scale_packs[1] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 96));
            }
            // B-rect: PC=1, fetch 8 bytes (b0p0 + b1p0) via b64.
            const uint32_t b_voff =
                (static_cast<uint32_t>(lane_kblk) << 6) |
                (static_cast<uint32_t>(lane_nonk) << 2);
            const uint32_t b_soff = static_cast<uint32_t>(k_pair) << 8;
            const uint64_t b_raw =
                llvm_amdgcn_raw_buffer_load_b64(b_v2_srsrc, b_voff, b_soff, MXFP8_CRR_V2_SCALE_CACHEPOLICY);
            b0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(b_raw      ));
            b1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(b_raw >> 32));
            return;
        }
    };

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
    auto load_a = [&](A_col_reg& dst, ST_crr_a& tile, int wi) {
        load_col_from_v2a_st(dst, tile, wi * RBM);
    };
    auto load_b = [&](B_col_reg_rect& dst, ST_crr_b& tile, int wi) {
        load_col_from_v2_st_rect(dst, tile, wi * RBN_RECT);
    };

    int tic = 0, toc = 1;
    // Rect: 2 N-half-tiles per BLK_N=128 block. bc * 2 indexes the half-tile
    // pair (same as square's bc * 2 indexing into HB-sized tiles).
    global_load_b(Bs[tic][0], bc * 2, 0);
    global_load_a(As[tic][0], br * 2, 0);
    global_load_b(Bs[tic][1], bc * 2 + 1, 0);
    global_load_a(As[tic][1], br * 2 + 1, 0);

    if (wm == 1) {
        __builtin_amdgcn_s_barrier();
    }
    TK_WAIT_VMCNT(CRR_INIT0_VMCNT);
    __builtin_amdgcn_s_barrier();

    global_load_b(Bs[toc][0], bc * 2, 1);
    global_load_a(As[toc][0], br * 2, 1);
    global_load_b(Bs[toc][1], bc * 2 + 1, 1);

    TK_WAIT_VMCNT(CRR_INIT1_VMCNT);
    __builtin_amdgcn_s_barrier();

    TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int g = 0; g < crr_a_pack_count; g++) {
                a0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[g]) >> 16);
                a1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[g]) >> 16);
            }
            #pragma unroll
            for (int g = 0; g < crr_b_pack_count_rect; g++) {
                b0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[g]) >> 16);
                b1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[g]) >> 16);
            }
        }

        load_b(b0, Bs[tic][0], wn);
        load_a(a, As[tic][0], wm);
        global_load_a(As[toc][1], br * 2 + 1, k + 1);
        TK_WAIT_LGKM(CRR_EXACT_PREFETCH_LGKM);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        crr_exact_cA_with_b1_interleave_fixed_phase_rect<CRR_EXACT_B1_LDS_INSERT_AFTER, crr_a_pack_count, crr_b_pack_count_rect>(
            cA, a, b0, b1, Bs[tic][1], wn, a0_scale_packs, b0_scale_packs);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        crr_mma_scaled_from_packs_fixed_phase_rect(cB, a, b1, a0_scale_packs, b1_scale_packs);
        __builtin_amdgcn_s_setprio(0);
        CRR_SCHED_BARRIER();
        CRR_STEADY_MID_BARRIER();
        CRR_SCHED_BARRIER();

        load_a(a, As[tic][1], wm);
        global_load_a(As[tic][0], br * 2, k + 2);
        global_load_b(Bs[tic][1], bc * 2 + 1, k + 2);
        TK_WAIT_VMCNT(CRR_EXACT_STEADY_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase_rect(cC, a, b0, a1_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase_rect(cD, a, b1, a1_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
        global_load_b(Bs[tic][0], bc * 2, k + 2);
    }

    {
        const int k = k_iters - 2;
        load_raw_scales(k >> 1);

        load_b(b0, Bs[tic][0], wn);
        load_a(a, As[tic][0], wm);
        global_load_a(As[toc][1], br * 2 + 1, k + 1);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase_rect<0>(cA, a, b0, a0_scale_packs, b0_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
        CRR_SCHED_BARRIER();

        load_b(b1, Bs[tic][1], wn);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase_rect<0>(cB, a, b1, a0_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase_rect<0>(cC, a, b0, a1_scale_packs, b0_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_b(b0, Bs[toc][0], wn);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase_rect<0>(cD, a, b1, a1_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
        CRR_SCHED_BARRIER();
        tic ^= 1;
        toc ^= 1;
    }

    {
        const auto b0_keep = b0;
        load_a(a, As[tic][0], wm);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase_rect<1>(cA, a, b0_keep, a0_scale_packs, b0_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_b(b1, Bs[tic][1], wn);
        const auto b1_keep = b1;
        __builtin_amdgcn_s_barrier();
        CRR_SCHED_BARRIER();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase_rect<1>(cB, a, b1_keep, a0_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase_rect<1>(cC, a, b0_keep, a1_scale_packs, b0_scale_packs);
        crr_mma_scaled_phase_rect<1>(cD, a, b1_keep, a1_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
    }

    if (wm == 0) {
        __builtin_amdgcn_s_barrier();
    }

    if constexpr (PRESHUFFLED_QUANT) {
        mul(cA, cA, g.scale);
        mul(cB, cB, g.scale);
        mul(cC, cC, g.scale);
        mul(cD, cD, g.scale);
    }

    // Rect store coordinates: BLK_M=256 → M-rows 0..255 split into 2*WARPS_M=4
    // groups (same as square). BLK_N=128 → N-cols split into 2*WARPS_N=8 groups
    // of RBN_RECT=16 each.
    store(g.c, cA, {0, 0, br * WARPS_M * 2 + wm, bc * WARPS_N * 2 + wn});
    store(g.c, cB, {0, 0, br * WARPS_M * 2 + wm, bc * WARPS_N * 2 + WARPS_N + wn});
    store(g.c, cC, {0, 0, br * WARPS_M * 2 + WARPS_M + wm, bc * WARPS_N * 2 + wn});
    store(g.c, cD, {0, 0, br * WARPS_M * 2 + WARPS_M + wm, bc * WARPS_N * 2 + WARPS_N + wn});
}

__host__ inline bool crr_can_use_exact_8wave_scaled_rect(const layout_globals& g)
{
    // Same shape gate as square but additionally requires N divisible by BLK_N
    // (rect = 128) instead of BLK (= 256).
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM
        && (g.m % BLK == 0)
        && (g.n % BLK_N == 0)
        && (g.k % BK == 0);
}

template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch_crr_exact_8wave_scaled_v2_rect(const layout_globals& g)
{
    // Rect: grid = (M/BLK_M) * (N/BLK_N) — 2x more blocks vs square for same N.
    const dim3 grid((g.m / BLK) * (g.n / BLK_N));
    crr_exact_8wave_scaled_rect_kernel<PRESHUFFLED_QUANT>
        <<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

#endif // MXFP8_CRR_RECT_8WAVE_FAST_ENABLE
#endif // MXFP8_RECT_BLK_N == 64
