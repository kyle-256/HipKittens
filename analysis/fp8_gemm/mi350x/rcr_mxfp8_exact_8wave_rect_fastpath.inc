#pragma once

// R32 Dev D — Stage A1: rectangular BLK_M=256/BLK_N=128 V2-RCR exact 8-wave
// fastpath kernel. Mirrors the R31 Dev A rect-V2-CRR scaffolding pattern in
// crr_mxfp8_exact_8wave_rect_fastpath.inc, adapted for the V2-RCR codepath:
//   - A is row-layout (A_row_reg) and is loaded from an `st_16x128_s` tile via
//     `rcr_exact_load_st_to_rt` (NOT the v2_s ds_read helper used by CRR).
//   - B is row-layout (B_row_reg) and likewise uses `rcr_exact_load_st_to_rt`.
//   - Accumulator is `rcr_exact_acc` (8 floatx4 regs in square; we provide
//     `rcr_exact_acc_rect` with 4 floatx4 regs because RBN_RECT/16 = 1 vs 2).
//   - The RCR steady loop uses `rcr_mma_scaled_from_packs_fixed_phase_impl`
//     which assumes RBN==32 (RBN/16==2). We provide _rect variants here that
//     call the underlying `rcr_exact_mfma_scale_builtin_*_inplace` primitives
//     for RBN_RECT==16 (RBN_RECT/16==1).
//
// **Stage A1a deliverable**: clean compile + parallel template exists.
// **Stage A1b deliverable**: rect build runs without GPU fault on V2-RCR
// 4096×4096×4096 (the structural-CU-underutilization target — at BLK=256 that
// shape has 256 tiles for 304 CUs; rect BLK_N=128 → 16×32=512 tiles → much
// better wave-fill — see r31d_findings.md for the structural argument).
// **Stage A1c (correct numerics)**: explicitly DEFERRED to R33+ (matches Dev
// A's R31 CRR Stage A1 timeline). Stage A1a stub for the K_HALF=1 B-side load
// produces NaN; Stage A2.x will replace with a correct second-tile read.
//
// SCOPE / CAVEATS
// ---------------
// - Only the V2 SCALE_VERSION=2 path is implemented (PRESHUFFLED_QUANT=true,
//   matches dispatch_rcr_exact_8wave_scaled_v2_rect). The V1 path is left as
//   a fall-through to the existing (square) kernel via the dispatcher gate.
// - B-side shared tile is `st_fp8e4m3<HB_N=64, BK=128, st_16x128_s>` (NOT v2_s
//   — this is the RCR data tile, not the V2 scale slab). The K-axis is full
//   BK=128 within a single 64-row tile.
// - The `rcr_exact_load_st_to_rt` helper is generic over ST/RT types so it
//   handles the rect tile dimensions transparently.
// - The rect helper for the k_iters reduction reuses the existing RCR MFMA
//   builtins (16x16x128 mfma_scale) so no new intrinsic plumbing is needed.
// - V2 scale slab geometry: num_slabs_b_rect = (N/BLK_N)*WARPS_N (was
//   (N/BLK)*WARPS_N for square), pack_count_b_rect = 1 (vs 2 for square).
//   slab_bytes_b_rect = 32 * padded_k_blocks (vs 64 * padded_k_blocks).
// - **Host preshuffle MUST match this geometry for correctness** (Stage A2).
//   For Stage A1a the kernel is wired but numerics will be wrong unless the
//   host preshuffle matches AND the K_HALF=1 stub is fixed. SHIP gate is
//   NOT evaluable until Stage A2.
//
// MIRROR DEVIATIONS FROM CRR RECT FILE (intentional)
// --------------------------------------------------
// 1. RCR-side B reads use `rcr_exact_load_st_to_rt` (a generic ST→RT copy)
//    not the `load_col_from_v2_st_*` ds_read_b64_tr_b8 path. The CRR file's
//    K_HALF stub does not apply: a single `rcr_exact_load_st_to_rt` call on
//    a `st_fp8e4m3<HB_N=64, BK=128, st_16x128_s>` tile loads ALL of K=0..127
//    for the warp's 16 N-rows. NO K_HALF stub needed.
// 2. The 4-quadrant accumulator structure (cA/cB/cC/cD) is preserved (matches
//    square RCR). cA = (lo_M_half, lo_N_half), cB = (lo_M_half, hi_N_half),
//    etc. With BLK_N=128 and HB_N=64, the two N halves are still separate
//    shared tiles `Bs[tic][0]` and `Bs[tic][1]` — same indexing as square.
// 3. Store coordinates (epilogue) use `bc * WARPS_N * 2 + wn` and
//    `bc * WARPS_N * 2 + WARPS_N + wn` exactly as square — but the underlying
//    coord system in HipKittens uses *register-tile* coordinates not byte
//    coordinates, so the store math is invariant under RBN_RECT vs RBN at
//    fixed WARPS_N=4. (Verify in Stage A2 numerics.)

#if defined(MXFP8_RECT_BLK_N) && (MXFP8_RECT_BLK_N == 64)

#ifndef MXFP8_RCR_RECT_8WAVE_FAST_ENABLE
#define MXFP8_RCR_RECT_8WAVE_FAST_ENABLE 1
#endif

#if MXFP8_RCR_RECT_8WAVE_FAST_ENABLE

// Sanity: rect kernel only makes sense at the same global tile geometry as
// the square fastpath (BLK_M=256 / BK=128 / WARPS_M=2 / WARPS_N=4) but with
// BLK_N=128 (rect) instead of BLK_N=256 (square).
static_assert(BLK == 256, "MXFP8 RCR exact 8-wave rect fast path requires BLK_M=256");
static_assert(BK == 128, "MXFP8 RCR exact 8-wave rect fast path requires BK=128");
static_assert(WARPS_M == 2, "MXFP8 RCR exact 8-wave rect fast path requires WARPS_M=2");
static_assert(WARPS_N == 4, "MXFP8 RCR exact 8-wave rect fast path requires WARPS_N=4");
static_assert(BLK_N == 128, "MXFP8 RCR exact 8-wave rect fast path requires BLK_N=128");
static_assert(RBN_RECT == 16, "MXFP8 RCR exact 8-wave rect fast path requires RBN_RECT=16");
static_assert(HB_N == 64, "MXFP8 RCR exact 8-wave rect fast path requires HB_N=64");
static_assert(std::is_same_v<RCR_B_reg, B_row_reg>,
              "Rect RCR fast path assumes row-layout B registers (matches square RCR)");

// Rect-N specific shared / register tile types.
// Note: RCR uses `st_16x128_s` (NOT `st_16x128_v2_s`) for both A and B shared
// tiles. The "V2" in V2-RCR refers to the scale-slab layout, not the data tile.
using ST_A_rcr_rect = st_fp8e4m3<HB,   BK, st_16x128_s>;  // 128x128 (M-side, unchanged)
using ST_B_rcr_rect = st_fp8e4m3<HB_N, BK, st_16x128_s>;  // 64x128 rect B tile
using B_row_reg_rect = rt_fp8e4m3<RBN_RECT, BK, row_l, rt_16x128_s>;  // 16x128 rect

// Rect-N exact accumulator: same struct as `rcr_exact_acc` but with RBN_RECT.
// regs has (RBM/16)*(RBN_RECT/16) = 4*1 = 4 floatx4 elements (vs 4*2=8 square).
struct alignas(16) rcr_exact_acc_rect {
    rcr_exact_floatx4_t regs[(RBM / 16) * (RBN_RECT / 16)];
};

// Rect-N store helper: mirrors rcr_exact_acc_to_rt but for the smaller acc.
template<typename RT_C>
__device__ __forceinline__ void rcr_exact_acc_rect_to_rt(
    RT_C& dst,
    const rcr_exact_acc_rect& src,
    float scale)
{
    static_assert(RT_C::rows == RBM && RT_C::cols == RBN_RECT,
                  "rect exact acc store helper expects RBM x RBN_RECT");

    #pragma unroll
    for (int row = 0; row < (RBM / 16); ++row) {
        #pragma unroll
        for (int col = 0; col < (RBN_RECT / 16); ++col) {
            *reinterpret_cast<rcr_exact_floatx4_t*>(&dst.tiles[row][col].data[0]) =
                src.regs[row * (RBN_RECT / 16) + col] * scale;
        }
    }
}

// Rect-N MFMA dispatch helper: square `rcr_mma_scaled_from_packs_fixed_phase_row`
// processes 2 N-tiles (b.tiles[0][0] and b.tiles[1][0]) per ROW_BASE because
// RBN/16 == 2. For rect (RBN_RECT/16 == 1) we process only b.tiles[0][0].
template<int ROW_BASE, int A_PACK_COUNT, int B_PACK_COUNT>
__device__ __forceinline__ void rcr_mma_scaled_from_packs_fixed_phase_row_rect(
    rcr_exact_acc_rect& acc,
    const A_row_reg& a,
    const B_row_reg_rect& b,
    const fp8e8m0_4 (&a_scale_packs)[A_PACK_COUNT],
    const fp8e8m0_4 (&b_scale_packs)[B_PACK_COUNT])
{
    static_assert(RBM == 64 && RBN_RECT == 16,
                  "rect RCR exact builtin helper expects a 64x16 accumulator tile");
    static_assert(ROW_BASE >= 0 && ROW_BASE < 4, "invalid row base");
    static_assert(A_PACK_COUNT >= 2, "rect RCR exact helper expects two A scale packs");
    static_assert(B_PACK_COUNT >= 1, "rect RCR exact helper expects one B scale pack");

    auto& d0 = acc.regs[ROW_BASE * (RBN_RECT / 16) + 0];

    const auto& a0 = *reinterpret_cast<const rcr_exact_intx8_t*>(&a.tiles[ROW_BASE][0].data[0]);
    const auto& b0 = *reinterpret_cast<const rcr_exact_intx8_t*>(&b.tiles[0][0].data[0]);
    const fp8e8m0_4 a_scale0 = a_scale_packs[ROW_BASE / 2];
    const fp8e8m0_4 b_scale0 = b_scale_packs[0];

    if constexpr ((ROW_BASE & 1) == 0) {
        rcr_exact_mfma_scale_builtin_inplace<0, 0>(d0, a0, b0, a_scale0, b_scale0);
    } else {
        rcr_exact_mfma_scale_builtin_inplace<1, 0>(d0, a0, b0, a_scale0, b_scale0);
    }
}

template<int A_PACK_COUNT, int B_PACK_COUNT>
__device__ __forceinline__ void rcr_mma_scaled_from_packs_fixed_phase_impl_rect(
    rcr_exact_acc_rect& acc,
    const A_row_reg& a,
    const B_row_reg_rect& b,
    const fp8e8m0_4 (&a_scale_packs)[A_PACK_COUNT],
    const fp8e8m0_4 (&b_scale_packs)[B_PACK_COUNT])
{
    rcr_mma_scaled_from_packs_fixed_phase_row_rect<0>(acc, a, b, a_scale_packs, b_scale_packs);
    rcr_mma_scaled_from_packs_fixed_phase_row_rect<1>(acc, a, b, a_scale_packs, b_scale_packs);
    rcr_mma_scaled_from_packs_fixed_phase_row_rect<2>(acc, a, b, a_scale_packs, b_scale_packs);
    rcr_mma_scaled_from_packs_fixed_phase_row_rect<3>(acc, a, b, a_scale_packs, b_scale_packs);
}

// Rect-N RCR exact 8-wave scaled kernel. Mirrors rcr_exact_8wave_scaled_kernel
// (square, V2 path). The structural deviations are:
//   - Bs uses ST_B_rcr_rect (64x128) instead of ST_B (128x128).
//   - B register is B_row_reg_rect (16x128) instead of B_row_reg (32x128).
//   - Accumulators use rcr_exact_acc_rect (4 floatx4 regs).
//   - blocks_per_col = N_DIM / BLK_N (= 32 for N=4096, BLK_N=128) vs square's
//     N_DIM / BLK (= 16 for N=4096, BLK=256).
//   - V2 scale slab geometry: pack_count_b_rect=1, slab_bytes_b_rect=32*kp.
//   - All optional configurability (PIPELINE_SCALE, KPAIR_LOOP, REMAP_ONCE,
//     PHASE_U16_CACHE, etc.) is OMITTED for Stage A1a — the rect kernel uses
//     the simplest possible inline scale-load + remap-each-iteration path.
//     If/when Stage A2.x ships and we tune perf, we can layer those back in.
template<bool PRESHUFFLED_QUANT>
__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void rcr_exact_8wave_scaled_rect_kernel(const layout_globals g) {
    static_assert(N_DIM % BLK_N == 0,
                  "MXFP8 RCR rect fast path requires N_DIM divisible by BLK_N");
    static_assert(K_DIM % BK == 0,
                  "MXFP8 RCR rect fast path requires K_DIM divisible by BK");
    constexpr int blocks_per_col_rect = N_DIM / BLK_N;
    const int k_iters = g.k / BK;
    constexpr int half_block_size_row = BLK / 2;     // 128 (M-side, unchanged)
    constexpr int half_block_size_col_rect = BLK_N / 2; // 64 (N-side, rect)

    using ST_A = ST_A_rcr_rect;
    using ST_B = ST_B_rcr_rect;
    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    using RT_A = A_row_reg;
    using RT_B = B_row_reg_rect;
    using ExactAcc = std::conditional_t<PRESHUFFLED_QUANT, rcr_exact_acc_rect,
                                        rt_fl<RBM, RBN_RECT, col_l, rt_16x16_s>>;

    RT_A a;
    RT_B b0, b1;
    ExactAcc cA{}, cB{}, cC{}, cD{};
    if constexpr (!PRESHUFFLED_QUANT) {
        zero(cA); zero(cB); zero(cC); zero(cD);
    }

    const int bid = blockIdx.x;
    const int br = bid / blocks_per_col_rect;
    const int bc = bid % blocks_per_col_rect;
    const int wm = warpid() / WARPS_N;
    const int wn = warpid() % WARPS_N;
    const int lane_nonk = kittens::laneid() % 16;
    const int lane_kblk = kittens::laneid() / 16;

    constexpr int bytes_per_thread = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_memcpy = bytes_per_thread * _NUM_THREADS;
    constexpr int memcpy_per_tile_a =
        half_block_size_row * BK * sizeof(fp8e4m3) / bytes_per_memcpy;
    constexpr int memcpy_per_tile_b =
        half_block_size_col_rect * BK * sizeof(fp8e4m3) / bytes_per_memcpy;

    uint32_t swizzled_offsets_a[memcpy_per_tile_a];
    uint32_t swizzled_offsets_b[memcpy_per_tile_b];
    G::prefill_swizzled_offsets(As[0][0], g.a, swizzled_offsets_a);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_b);

    auto rcr_scale_a_base = [&](int half) {
        return br * BLK + half * HB + wm * RBM;
    };
    // B-side scale base uses BLK_N (rect) and HB_N (rect).
    auto rcr_scale_b_base_rect = [&](int half) {
        return bc * BLK_N + half * HB_N + wn * RBN_RECT;
    };
    constexpr int scale_a_pack_count = RBM / 32;            // 2
    constexpr int scale_b_pack_count_rect = (RBN_RECT + 31) / 32;  // 1

    fp8e8m0_4 a0_scale_packs[scale_a_pack_count];
    fp8e8m0_4 a1_scale_packs[scale_a_pack_count];
    fp8e8m0_4 b0_scale_packs[scale_b_pack_count_rect];
    fp8e8m0_4 b1_scale_packs[scale_b_pack_count_rect];
    const uint8_t* a0_scale_row_bases[scale_a_pack_count];
    const uint8_t* a1_scale_row_bases[scale_a_pack_count];
    const uint8_t* b0_scale_row_bases[scale_b_pack_count_rect];
    const uint8_t* b1_scale_row_bases[scale_b_pack_count_rect];

    const uint32_t lane_scale_byte_offset =
        (static_cast<uint32_t>(lane_kblk) << 6) |
        (static_cast<uint32_t>(lane_nonk) << 2);

    // V2 wave-tile SRDs: 1 SRD per wave for A, 1 per wave for B.
    i32x4 a_v2_srsrc = {}, b_v2_srsrc = {};
    if constexpr (PRESHUFFLED_QUANT) {
        #pragma unroll
        for (int p = 0; p < scale_a_pack_count; ++p) {
            a0_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.a_scale, (rcr_scale_a_base(0) + p * 32) >> 5);
            a1_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.a_scale, (rcr_scale_a_base(1) + p * 32) >> 5);
        }
        #pragma unroll
        for (int p = 0; p < scale_b_pack_count_rect; ++p) {
            b0_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.b_scale, (rcr_scale_b_base_rect(0) + p * 32) >> 5);
            b1_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.b_scale, (rcr_scale_b_base_rect(1) + p * 32) >> 5);
        }
        // V2-RCR rect: per-wave-tile slab SRDs.
        // A side: pack_count=2 (unchanged), slab_bytes_a = 128 * padded_k_blocks.
        //   slab_idx_a = br * WARPS_M + wm.
        // B side: pack_count_rect=1, slab_bytes_b_rect = 32 * padded_k_blocks.
        //   num_slabs_b_rect = (N/BLK_N) * WARPS_N (= 32*4 = 128 for N=4096).
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
        a_v2_srsrc = make_v2_srd(a_v2_base, static_cast<uint32_t>(slab_bytes_a));
        b_v2_srsrc = make_v2_srd(b_v2_base, static_cast<uint32_t>(slab_bytes_b_rect));
    }

    // V2-RCR rect: load A (b128, 4 packs interleaved {a0p0,a1p0,a0p1,a1p1}) and
    // B (b32, 2 packs interleaved {b0p0,b1p0} as a single 4-byte word). Note
    // the B b32 fetch (vs square b64): with pack_count_b_rect=1 the slab is
    // half the size, so soff stride is k_pair*256 instead of k_pair*512.
    auto load_scale_packs_for_pair = [&](int k_pair) __attribute__((always_inline)) {
        if constexpr (PRESHUFFLED_QUANT) {
            const uint32_t a_voff =
                (static_cast<uint32_t>(lane_kblk) << 8) |
                (static_cast<uint32_t>(lane_nonk) << 4);
            const uint32_t a_soff = static_cast<uint32_t>(k_pair) << 10;
            const __uint128_t a_raw =
                llvm_amdgcn_raw_buffer_load_b128(a_v2_srsrc, a_voff, a_soff,
                                                 MXFP8_RCR_V2_SCALE_CACHEPOLICY);
            a0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw      ));
            a1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 32));
            if constexpr (scale_a_pack_count > 1) {
                a0_scale_packs[1] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 64));
                a1_scale_packs[1] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(a_raw >> 96));
            }
            const uint32_t b_voff =
                (static_cast<uint32_t>(lane_kblk) << 6) |
                (static_cast<uint32_t>(lane_nonk) << 2);
            const uint32_t b_soff = static_cast<uint32_t>(k_pair) << 8;
            const uint64_t b_raw =
                llvm_amdgcn_raw_buffer_load_b64(b_v2_srsrc, b_voff, b_soff,
                                                MXFP8_RCR_V2_SCALE_CACHEPOLICY);
            b0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(b_raw      ));
            b1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(static_cast<uint32_t>(b_raw >> 32));
        }
    };

    int tic = 0, toc = 1;
    G::load(Bs[tic][0], g.b, {0, 0, bc * 2,     0}, swizzled_offsets_b);
    G::load(As[tic][0], g.a, {0, 0, br * 2,     0}, swizzled_offsets_a);
    G::load(Bs[tic][1], g.b, {0, 0, bc * 2 + 1, 0}, swizzled_offsets_b);
    G::load(As[tic][1], g.a, {0, 0, br * 2 + 1, 0}, swizzled_offsets_a);

    if (wm == 1) {
        __builtin_amdgcn_s_barrier();
    }
    TK_WAIT_VMCNT(4);
    __builtin_amdgcn_s_barrier();

    G::load(Bs[toc][0], g.b, {0, 0, bc * 2,     1}, swizzled_offsets_b);
    G::load(As[toc][0], g.a, {0, 0, br * 2,     1}, swizzled_offsets_a);
    G::load(Bs[toc][1], g.b, {0, 0, bc * 2 + 1, 1}, swizzled_offsets_b);

    TK_WAIT_VMCNT(6);
    __builtin_amdgcn_s_barrier();

    // Steady-state loop. Mirrors the square RCR steady loop but with rect
    // helpers and rect-sized B subtile selections.
    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
        const int k_pair = k >> 1;
        const int k_phase = k & 1;
        load_scale_packs_for_pair(k_pair);

        // A-tile read: same as square (RBM=64 from a 128x128 ST_A tile).
        auto as_subtile0 = kittens::subtile_inplace<RBM, BK>(As[tic][0], {wm, 0});
        rcr_exact_load_st_to_rt<RT_A, decltype(as_subtile0)>(a, as_subtile0);
        // B-tile read: rect — RBN_RECT=16 from a 64x128 ST_B_rcr_rect tile.
        auto bs_subtile0 = kittens::subtile_inplace<RBN_RECT, BK>(Bs[tic][0], {wn, 0});
        rcr_exact_load_st_to_rt<RT_B, decltype(bs_subtile0)>(b0, bs_subtile0);

        G::load(As[toc][1], g.a, {0, 0, br * 2 + 1, k + 1}, swizzled_offsets_a);
        TK_WAIT_LGKM(8);
        __builtin_amdgcn_s_barrier();

        // Phase remap for rect (uses the existing remap helper, parametrized
        // on PACK_COUNT). For a_packs (count=2) and b_packs (count=1) this
        // does the standard 16-bit shift.
        fp8e8m0_4 a0_phase_packs[scale_a_pack_count];
        fp8e8m0_4 a1_phase_packs[scale_a_pack_count];
        fp8e8m0_4 b0_phase_packs[scale_b_pack_count_rect];
        fp8e8m0_4 b1_phase_packs[scale_b_pack_count_rect];
        remap_phase_scale_packs_16x128(a0_phase_packs, a0_scale_packs, k_phase);
        remap_phase_scale_packs_16x128(a1_phase_packs, a1_scale_packs, k_phase);
        remap_phase_scale_packs_16x128(b0_phase_packs, b0_scale_packs, k_phase);
        remap_phase_scale_packs_16x128(b1_phase_packs, b1_scale_packs, k_phase);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        if constexpr (PRESHUFFLED_QUANT) {
            rcr_mma_scaled_from_packs_fixed_phase_impl_rect(cA, a, b0,
                a0_phase_packs, b0_phase_packs);
        }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs_subtile1 = kittens::subtile_inplace<RBN_RECT, BK>(Bs[tic][1], {wn, 0});
        rcr_exact_load_st_to_rt<RT_B, decltype(bs_subtile1)>(b1, bs_subtile1);
        G::load(Bs[tic][0], g.b, {0, 0, bc * 2, k + 2}, swizzled_offsets_b);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        if constexpr (PRESHUFFLED_QUANT) {
            rcr_mma_scaled_from_packs_fixed_phase_impl_rect(cB, a, b1,
                a0_phase_packs, b1_phase_packs);
        }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as_subtile1 = kittens::subtile_inplace<RBM, BK>(As[tic][1], {wm, 0});
        rcr_exact_load_st_to_rt<RT_A, decltype(as_subtile1)>(a, as_subtile1);
        G::load(As[tic][0], g.a, {0, 0, br * 2, k + 2}, swizzled_offsets_a);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        if constexpr (PRESHUFFLED_QUANT) {
            rcr_mma_scaled_from_packs_fixed_phase_impl_rect(cC, a, b0,
                a1_phase_packs, b0_phase_packs);
        }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[tic][1], g.b, {0, 0, bc * 2 + 1, k + 2}, swizzled_offsets_b);
        TK_WAIT_VMCNT(6);
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        if constexpr (PRESHUFFLED_QUANT) {
            rcr_mma_scaled_from_packs_fixed_phase_impl_rect(cD, a, b1,
                a1_phase_packs, b1_phase_packs);
        }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    // Pre-tail: k = k_iters - 2.
    {
        const int k = k_iters - 2;
        const int k_pair = k >> 1;
        const int k_phase = k & 1;
        load_scale_packs_for_pair(k_pair);

        auto bs_subtile0 = kittens::subtile_inplace<RBN_RECT, BK>(Bs[tic][0], {wn, 0});
        rcr_exact_load_st_to_rt<RT_B, decltype(bs_subtile0)>(b0, bs_subtile0);
        auto as_subtile0 = kittens::subtile_inplace<RBM, BK>(As[tic][0], {wm, 0});
        rcr_exact_load_st_to_rt<RT_A, decltype(as_subtile0)>(a, as_subtile0);
        G::load(As[toc][1], g.a, {0, 0, br * 2 + 1, k + 1}, swizzled_offsets_a);

        fp8e8m0_4 a0_phase_packs[scale_a_pack_count];
        fp8e8m0_4 a1_phase_packs[scale_a_pack_count];
        fp8e8m0_4 b0_phase_packs[scale_b_pack_count_rect];
        fp8e8m0_4 b1_phase_packs[scale_b_pack_count_rect];
        remap_phase_scale_packs_16x128(a0_phase_packs, a0_scale_packs, k_phase);
        remap_phase_scale_packs_16x128(a1_phase_packs, a1_scale_packs, k_phase);
        remap_phase_scale_packs_16x128(b0_phase_packs, b0_scale_packs, k_phase);
        remap_phase_scale_packs_16x128(b1_phase_packs, b1_scale_packs, k_phase);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        if constexpr (PRESHUFFLED_QUANT) {
            rcr_mma_scaled_from_packs_fixed_phase_impl_rect(cA, a, b0,
                a0_phase_packs, b0_phase_packs);
        }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto bs_subtile1 = kittens::subtile_inplace<RBN_RECT, BK>(Bs[tic][1], {wn, 0});
        rcr_exact_load_st_to_rt<RT_B, decltype(bs_subtile1)>(b1, bs_subtile1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        if constexpr (PRESHUFFLED_QUANT) {
            rcr_mma_scaled_from_packs_fixed_phase_impl_rect(cB, a, b1,
                a0_phase_packs, b1_phase_packs);
        }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as_subtile1 = kittens::subtile_inplace<RBM, BK>(As[tic][1], {wm, 0});
        rcr_exact_load_st_to_rt<RT_A, decltype(as_subtile1)>(a, as_subtile1);
        TK_WAIT_VMCNT(4);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        if constexpr (PRESHUFFLED_QUANT) {
            rcr_mma_scaled_from_packs_fixed_phase_impl_rect(cC, a, b0,
                a1_phase_packs, b0_phase_packs);
        }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        bs_subtile0 = kittens::subtile_inplace<RBN_RECT, BK>(Bs[toc][0], {wn, 0});
        rcr_exact_load_st_to_rt<RT_B, decltype(bs_subtile0)>(b0, bs_subtile0);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        if constexpr (PRESHUFFLED_QUANT) {
            rcr_mma_scaled_from_packs_fixed_phase_impl_rect(cD, a, b1,
                a1_phase_packs, b1_phase_packs);
        }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        tic ^= 1;
        toc ^= 1;
    }

    // Tail: k = k_iters - 1.
    {
        auto as_subtile0 = kittens::subtile_inplace<RBM, BK>(As[tic][0], {wm, 0});
        const int k_pair = (k_iters - 1) >> 1;
        const int k_phase = (k_iters - 1) & 1;
        load_scale_packs_for_pair(k_pair);
        rcr_exact_load_st_to_rt<RT_A, decltype(as_subtile0)>(a, as_subtile0);

        fp8e8m0_4 a0_phase_packs[scale_a_pack_count];
        fp8e8m0_4 a1_phase_packs[scale_a_pack_count];
        fp8e8m0_4 b0_phase_packs[scale_b_pack_count_rect];
        fp8e8m0_4 b1_phase_packs[scale_b_pack_count_rect];
        remap_phase_scale_packs_16x128(a0_phase_packs, a0_scale_packs, k_phase);
        remap_phase_scale_packs_16x128(a1_phase_packs, a1_scale_packs, k_phase);
        remap_phase_scale_packs_16x128(b0_phase_packs, b0_scale_packs, k_phase);
        remap_phase_scale_packs_16x128(b1_phase_packs, b1_scale_packs, k_phase);

        TK_WAIT_VMCNT(0);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        if constexpr (PRESHUFFLED_QUANT) {
            rcr_mma_scaled_from_packs_fixed_phase_impl_rect(cA, a, b0,
                a0_phase_packs, b0_phase_packs);
        }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto bs_subtile1 = kittens::subtile_inplace<RBN_RECT, BK>(Bs[tic][1], {wn, 0});
        rcr_exact_load_st_to_rt<RT_B, decltype(bs_subtile1)>(b1, bs_subtile1);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        if constexpr (PRESHUFFLED_QUANT) {
            rcr_mma_scaled_from_packs_fixed_phase_impl_rect(cB, a, b1,
                a0_phase_packs, b1_phase_packs);
        }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as_subtile1 = kittens::subtile_inplace<RBM, BK>(As[tic][1], {wm, 0});
        rcr_exact_load_st_to_rt<RT_A, decltype(as_subtile1)>(a, as_subtile1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        if constexpr (PRESHUFFLED_QUANT) {
            rcr_mma_scaled_from_packs_fixed_phase_impl_rect(cC, a, b0,
                a1_phase_packs, b0_phase_packs);
            rcr_mma_scaled_from_packs_fixed_phase_impl_rect(cD, a, b1,
                a1_phase_packs, b1_phase_packs);
        }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (wm == 0) {
        __builtin_amdgcn_s_barrier();
    }

    if constexpr (PRESHUFFLED_QUANT) {
        rt_fl<RBM, RBN_RECT, col_l, rt_16x16_s> c_store;
        rcr_exact_acc_rect_to_rt(c_store, cA, g.scale);
        store(g.c, c_store, {0, 0, br * WARPS_M * 2 + wm,            bc * WARPS_N * 2 + wn});
        rcr_exact_acc_rect_to_rt(c_store, cB, g.scale);
        store(g.c, c_store, {0, 0, br * WARPS_M * 2 + wm,            bc * WARPS_N * 2 + WARPS_N + wn});
        rcr_exact_acc_rect_to_rt(c_store, cC, g.scale);
        store(g.c, c_store, {0, 0, br * WARPS_M * 2 + WARPS_M + wm,  bc * WARPS_N * 2 + wn});
        rcr_exact_acc_rect_to_rt(c_store, cD, g.scale);
        store(g.c, c_store, {0, 0, br * WARPS_M * 2 + WARPS_M + wm,  bc * WARPS_N * 2 + WARPS_N + wn});
    } else {
        mul(cA, cA, g.scale);
        mul(cB, cB, g.scale);
        mul(cC, cC, g.scale);
        mul(cD, cD, g.scale);
        store(g.c, cA, {0, 0, br * WARPS_M * 2 + wm,            bc * WARPS_N * 2 + wn});
        store(g.c, cB, {0, 0, br * WARPS_M * 2 + wm,            bc * WARPS_N * 2 + WARPS_N + wn});
        store(g.c, cC, {0, 0, br * WARPS_M * 2 + WARPS_M + wm,  bc * WARPS_N * 2 + wn});
        store(g.c, cD, {0, 0, br * WARPS_M * 2 + WARPS_M + wm,  bc * WARPS_N * 2 + WARPS_N + wn});
    }
}

__host__ inline bool rcr_can_use_exact_8wave_scaled_rect(const layout_globals& g) {
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM
        && (g.m % BLK == 0)
        && (g.n % BLK_N == 0)
        && (g.k % BK == 0);
}

template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch_rcr_exact_8wave_scaled_v2_rect(const layout_globals& g) {
    // Rect: grid = (M/BLK_M) * (N/BLK_N) — 2x more blocks vs square for same N.
    // For 4096^3 this gives 16 * 32 = 512 tiles vs square's 256.
    const dim3 grid((g.m / BLK) * (g.n / BLK_N));
    rcr_exact_8wave_scaled_rect_kernel<PRESHUFFLED_QUANT>
        <<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

#endif // MXFP8_RCR_RECT_8WAVE_FAST_ENABLE
#endif // MXFP8_RECT_BLK_N == 64
