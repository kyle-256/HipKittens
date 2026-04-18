#pragma once

// R35 Dev B — HB shrink (BLK_M=128, HB_M=64) V2-CRR exact 8-wave fastpath
// scaffolding. Parallel template to crr_mxfp8_exact_8wave_fastpath.inc with
// the M-direction block geometry halved. Mirrors the structural pattern of
// the rect-V2 CRR Path 1 scaffolding (crr_mxfp8_exact_8wave_rect_fastpath.inc)
// which halves the N-direction block geometry; here we mirror the same
// approach to halve the M-direction block geometry.
//
// PARADIGM PIVOT (R34 Dev D §3.2 Option 1):
//   R32 Dev B identified that V2-CRR LDS SB pipelining hits a -15% structural
//   ceiling at PIPE=3 because A_col_reg=32 VGPR/wave + accumulator=128 VGPR
//   leaves no room for the `a_next` register PIPE=2..3 needs. R34 Dev D Stage
//   2 REFUTED the sub-RBM (RBM=32) hypothesis: shrinking the operand-tile
//   doubles the accumulator vector count, leaving total VGPR worse not
//   better. The pivot identified by R34 Dev D is to instead shrink the
//   per-warp M-coverage (HB) — this actually reduces accumulator vector
//   count.
//
// HB SHRINK APPROACH:
//   * BLK_M = 128 (was 256). Block now covers HB_M = BLK_M / 2 = 64 rows
//     per M-half. Each warp's M-coverage = HB_M = 64 rows.
//   * WARPS_M = 2 (unchanged). Each warp covers 1 M-half-tile via RBM=64
//     (single stride per warp; was 2 strides at production HB=128).
//   * Grid doubles in M: blocks_per_row = M / BLK_M (= 2x default).
//   * Drop cC/cD: only cA/cB remain (top-half only). Accumulator footprint
//     drops from 4 × 32 = 128 VGPR to 2 × 32 = 64 VGPR per warp.
//
// R36 Dev A — Stage B re-pipelining variants (this revision):
//   The R35 Stage A skeleton had VGPR 168 (-66 vs default 234), 0 spill,
//   occ=2 — but raw perf was -12.16% because it used a single-buffered LDS
//   schedule with no cycle-2 prefetch and no B1 LDS interleave. R36 Dev A
//   adds three pipelining variants selected by MXFP8_CRR_HBSHRINK_PIPELINE:
//
//     0 = Stage A skeleton (single-buffered, R35 baseline; -12.16% perf).
//     1 = Stage B1: DB main loop with cycle-2 B-tile prefetch + cycle-2
//                   A-tile prefetch + B1 LDS interleave inside the cA MMA
//                   chain. Mirrors the default kernel's DB structure with
//                   the second-half halved (single MMA pair instead of two).
//     2 = Stage B2: PIPE=3 LDS pipelining (single-buffer LDS + early-issue
//                   VMEM→LDS, all 3 LDS reads up front + late wait). Uses
//                   the freed VGPR headroom to keep 2 a_col vectors in
//                   flight (not needed here — only 1 a-load per iter — so
//                   the variant degenerates to the single-A early-issue
//                   form, matching R32 PIPE=1 with halved compute).
//     3 = Stage B3: B1 LDS interleave inside Stage B2 (sub-tile interleave
//                   of A-read + B1-read across MMA chain).

// HB shrink macro: when defined to 128, the parallel HB shrink kernel
// activates. Default 256 = no change (default fastpath remains source of
// truth). Independent of MXFP8_CRR_RBM (R33 Dev D sub-RBM scaffolding) and
// MXFP8_RECT_BLK_N (R28 Dev D rect-V2 scaffolding) — those are orthogonal
// progressive macros.
#ifndef MXFP8_CRR_BLK_M
#define MXFP8_CRR_BLK_M 256
#endif
#if (MXFP8_CRR_BLK_M != 128) && (MXFP8_CRR_BLK_M != 256)
#error "MXFP8_CRR_BLK_M must be 128 or 256 (R35 Dev B scaffolding)"
#endif

#if (MXFP8_CRR_BLK_M == 128)

#ifndef MXFP8_CRR_HBSHRINK_8WAVE_FAST_ENABLE
#define MXFP8_CRR_HBSHRINK_8WAVE_FAST_ENABLE 1
#endif

// R36 Dev A pipelining selector. 0 = R35 Stage A skeleton (single-buffered,
// no prefetch). 1 = Stage B1 DB main loop + B1 LDS interleave + cycle-2
// prefetch. 2 = Stage B2 SB-PIPE=3 (early-issue + late-wait). 3 = Stage B3
// SB-PIPE=3 + B1 LDS interleave.
#ifndef MXFP8_CRR_HBSHRINK_PIPELINE
#define MXFP8_CRR_HBSHRINK_PIPELINE 0
#endif
#if (MXFP8_CRR_HBSHRINK_PIPELINE < 0) || (MXFP8_CRR_HBSHRINK_PIPELINE > 4)
#error "MXFP8_CRR_HBSHRINK_PIPELINE must be 0..4 (R36 Dev A; R37 Dev C added 4=B3v2)"
#endif

// Stage B1/B3 selector: insert-after position for the B1 LDS-interleave inside
// the cA MMA chain. Default 4 mirrors the production default kernel.
#ifndef MXFP8_CRR_HBSHRINK_B1_INSERT_AFTER
#define MXFP8_CRR_HBSHRINK_B1_INSERT_AFTER 4
#endif

#if MXFP8_CRR_HBSHRINK_8WAVE_FAST_ENABLE

// HB shrink invariants. We KEEP square A LDS tile (HB=128, BK=128) so the
// host preshuffle is unchanged — we just consume HALF of each square A
// LDS tile per block (mhalf selects top vs bottom). This mirrors rect-V2
// Path 1's halved N-work approach (Dev A R33 Stage A2).
static_assert(BLK == 256, "HB shrink CRR fast path requires BLK_M (default) = 256 — we halve to BLK_M=128 internally");
static_assert(BK == 128, "HB shrink CRR fast path requires BK=128");
static_assert(WARPS_M == 2, "HB shrink CRR fast path requires WARPS_M=2");
static_assert(WARPS_N == 4, "HB shrink CRR fast path requires WARPS_N=4");
static_assert(HB == 128, "HB shrink CRR fast path requires HB=128 (square LDS tile)");
static_assert(RBM == 64, "HB shrink CRR fast path requires RBM=64 (square reg tile)");
static_assert(RBN == 32, "HB shrink CRR fast path requires RBN=32 (square reg tile)");
static_assert(!CRR_ROW_SHARED_TRANSPOSE, "HB shrink CRR fast path only supports strict shared layout");
static_assert(!CRR_USE_V3_SWIZZLE, "HB shrink CRR fast path is wired for v2/v2a shared layouts");

// HB shrink internal block-tile constants. These name the *effective*
// block geometry the HB shrink kernel produces; the LDS tile geometry
// is still the square HB=128.
constexpr int HBSHRINK_BLK_M = 128;          // halved M-direction block
constexpr int HBSHRINK_HB_M  = HBSHRINK_BLK_M / 2;  // 64 = per-warp M coverage
// A LDS tile geometry kept square (HB=128); we consume only HALF per block
// via the mhalf selector below.

// ---------------------------------------------------------------------------
// R36 Dev A Stage B1/B3: B1 LDS-interleave helper. Parallel to the default
// kernel's crr_exact_cA_with_b1_interleave_fixed_phase, but operates on the
// runtime mhalf-selected A scale-pack array (a_sel_packs) instead of the
// default a0_scale_packs. Same shape/contract: 8 MMA emit slots, the b1 LDS
// load is inserted at slot INSERT_AFTER.
// ---------------------------------------------------------------------------
template<int INSERT_AFTER, int A_PACK_COUNT, int B_PACK_COUNT>
__device__ __forceinline__ void crr_hbshrink_cA_with_b1_interleave_fixed_phase(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& cA,
    const A_col_reg& a,
    const B_col_reg& b0,
    B_col_reg& b1,
    const ST_v2& b1_tile,
    int wn,
    const fp8e8m0_4 (&a_phase)[A_PACK_COUNT],
    const fp8e8m0_4 (&b0_phase)[B_PACK_COUNT])
{
    static_assert(INSERT_AFTER >= 0 && INSERT_AFTER <= 8);
    static_assert(RBM == 64 && RBN == 32);

    __builtin_amdgcn_s_setprio(1);
    if constexpr (INSERT_AFTER == 0) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<0, 0>(cA, a, b0, 0, 0, a_phase[0], b0_phase[0]);
    if constexpr (INSERT_AFTER == 1) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<0, 1>(cA, a, b0, 0, 1, a_phase[0], b0_phase[0]);
    if constexpr (INSERT_AFTER == 2) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<1, 0>(cA, a, b0, 1, 0, a_phase[0], b0_phase[0]);
    if constexpr (INSERT_AFTER == 3) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<1, 1>(cA, a, b0, 1, 1, a_phase[0], b0_phase[0]);
    if constexpr (INSERT_AFTER == 4) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<0, 0>(cA, a, b0, 2, 0, a_phase[1], b0_phase[0]);
    if constexpr (INSERT_AFTER == 5) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<0, 1>(cA, a, b0, 2, 1, a_phase[1], b0_phase[0]);
    if constexpr (INSERT_AFTER == 6) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<1, 0>(cA, a, b0, 3, 0, a_phase[1], b0_phase[0]);
    if constexpr (INSERT_AFTER == 7) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<1, 1>(cA, a, b0, 3, 1, a_phase[1], b0_phase[0]);
    if constexpr (INSERT_AFTER == 8) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    __builtin_amdgcn_s_setprio(0);
    CRR_SCHED_BARRIER();
}

// ---------------------------------------------------------------------------
// HB shrink V2-CRR exact 8-wave scaled kernel.
//
// Each block processes HB_M=64 (M) × BLK_N=256 (N) of C. Compared to the
// default kernel (BLK_M=256 × BLK_N=256):
//   * Per-warp M-coverage: HB_M=64 (was HB=128).
//   * M-strides per warp:  1 (was 2).
//   * Accumulator vectors: 2 (cA, cB) instead of 4 (cA, cB, cC, cD) —
//     drops 64 VGPR/wave from the accumulator footprint.
//   * Grid:  blocks_per_row = M / HBSHRINK_BLK_M = 2 × default.
//
// MXFP8_CRR_HBSHRINK_PIPELINE selects the pipelining strategy:
//   0 = Stage A skeleton (single-buffered, R35 baseline).
//   1 = Stage B1: DB main loop + cycle-2 prefetch + B1 LDS interleave.
//   2 = Stage B2: SB PIPE=3 (early-issue + late-wait, single-A).
//   3 = Stage B3: SB PIPE=3 + B1 LDS interleave.
// ---------------------------------------------------------------------------
template<bool PRESHUFFLED_QUANT, int SCALE_VERSION = 2>
__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void crr_exact_8wave_scaled_hbshrink_kernel(const layout_globals g)
{
    using ST_crr_a = ST_v2a;     // square A LDS tile (HB=128, BK=128)
    using ST_crr_b = ST_v2;      // square B LDS tile (HB=128, BK=128)

    constexpr int crr_a_pack_count = RBM / 32;        // 2
    constexpr int crr_b_pack_count = (RBN + 31) / 32; // 1 for RBN=32

    const int k_iters = g.k / BK;
    const int blocks_per_col = g.n / BLK;

    // LDS: single-tile A per slot (we only need ONE M-half-tile per block);
    // B unchanged (two N-halves per block). Two slots for double-buffering
    // in Stage B1; only slot [0] used in Stages 0/2/3.
    __shared__ ST_crr_a As[2];
    __shared__ ST_crr_b Bs[2][2];

    A_col_reg a;
    B_col_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB;
    zero(cA);
    zero(cB);

    const int bid = blockIdx.x;
    // M-direction grid is doubled: br indexes HBSHRINK_BLK_M=128 rows.
    // br_orig = br/2 indexes the original square BLK=256 row-block; mhalf
    // = br & 1 picks which M-half (0 = top, 1 = bot).
    const int br      = bid / blocks_per_col;
    const int bc      = bid % blocks_per_col;
    const int br_orig = br >> 1;
    const int mhalf   = br & 1;
    const int wm      = warpid() / WARPS_N;
    const int wn      = warpid() % WARPS_N;

    const int lane_nonk = kittens::laneid() % 16;
    const int lane_kblk = kittens::laneid() / 16;
    (void)lane_nonk; (void)lane_kblk;

    // Scale base: B side unchanged. A side uses br_orig * BLK + mhalf*(BLK/2)
    // to address the original square BLK-block scale slab — we still load
    // the same scale layout since host preshuffle is unchanged.
    auto crr_scale_a_base = [&](int local_half) {
        (void)local_half;
        return br_orig * BLK + mhalf * (BLK / 2) + wm * RBM;
    };
    auto crr_scale_b_base = [&](int half) {
        return bc * BLK + half * (BLK / 2) + wn * RBN;
    };

    // Scale-pack arrays. A side: only ONE half active (we only do top-half
    // MMA in this kernel); we still allocate a0/a1 to keep the control
    // flow parallel to the default kernel and let the slab loader emit
    // the same b128 fetch (the unused half is then DCEd).
    fp8e8m0_4 a0_scale_packs[crr_a_pack_count];
    fp8e8m0_4 a1_scale_packs[crr_a_pack_count];
    fp8e8m0_4 b0_scale_packs[crr_b_pack_count];
    fp8e8m0_4 b1_scale_packs[crr_b_pack_count];
    const uint8_t* a0_scale_row_bases[crr_a_pack_count];
    const uint8_t* a1_scale_row_bases[crr_a_pack_count];
    const uint8_t* b0_scale_row_bases[crr_b_pack_count];
    const uint8_t* b1_scale_row_bases[crr_b_pack_count];

    // V2 wave-tile SRDs — same slab geometry as square (host preshuffle
    // unchanged). slab_idx_a uses br_orig (NOT br) to address the original
    // square BLK-block.
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
        for (int p = 0; p < crr_b_pack_count; ++p) {
            b0_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.b_scale, (crr_scale_b_base(0) + p * 32) >> 5);
            b1_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.b_scale, (crr_scale_b_base(1) + p * 32) >> 5);
        }
        const int padded_k_blocks =
            (g.k / 32 + 7) & ~7;
        const size_t slab_bytes_a = 128u * padded_k_blocks;
        const size_t slab_bytes_b = 64u * padded_k_blocks;
        const size_t slab_idx_a =
            static_cast<size_t>(br_orig) * WARPS_M + wm;
        const size_t slab_idx_b =
            static_cast<size_t>(bc) * WARPS_N + wn;
        const uint8_t* a_v2_base =
            reinterpret_cast<const uint8_t*>(g.a_scale.raw_ptr) +
            slab_idx_a * slab_bytes_a;
        const uint8_t* b_v2_base =
            reinterpret_cast<const uint8_t*>(g.b_scale.raw_ptr) +
            slab_idx_b * slab_bytes_b;
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
                                 static_cast<uint32_t>(slab_bytes_b));
    }

    auto load_raw_scales = [&](int k_pair) __attribute__((always_inline)) {
        if constexpr (PRESHUFFLED_QUANT) {
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
            const uint32_t b_voff =
                (static_cast<uint32_t>(lane_kblk) << 7) |
                (static_cast<uint32_t>(lane_nonk) << 3);
            const uint32_t b_soff = static_cast<uint32_t>(k_pair) << 9;
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
    G::prefill_swizzled_offsets(As[0], g.a, soA);

    constexpr int bptB = ST_crr_b::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB = ST_crr_b::rows * ST_crr_b::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    // A coord: each block loads ONE square A LDS tile (HB=128, BK=128).
    // The "row" dimension uses (br_orig * 2 + mhalf) — for mhalf=0 we load
    // the top half of the square BLK-block (rows 0..127), for mhalf=1 the
    // bottom half (rows 128..255). The square A LDS tile is 128 rows, so
    // this single load covers exactly one M-half.
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
    auto load_b = [&](B_col_reg& dst, ST_crr_b& tile, int wi) {
        load_col_from_v2_st(dst, tile, wi * RBN);
    };

    // The A-side global K-tile coord. With BLK_M=128 we want the top half
    // OR bot half of the original BLK=256 row-block. (br_orig * 2 + mhalf)
    // gives the row-tile id at HB=128 granularity.
    auto a_tile_row = [&]() { return br_orig * 2 + mhalf; };

#if (MXFP8_CRR_HBSHRINK_PIPELINE == 0)
    // -----------------------------------------------------------------------
    // Stage A skeleton (R35 Dev B baseline; -12.16% vs default).
    // Single-buffered LDS schedule: load → barrier → compute → barrier → load.
    // -----------------------------------------------------------------------
    global_load_b(Bs[0][0], bc * 2,     0);
    global_load_a(As[0],    a_tile_row(), 0);
    global_load_b(Bs[0][1], bc * 2 + 1, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    for (int k = 0; k < k_iters; k++) {
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int gp = 0; gp < crr_a_pack_count; gp++) {
                a0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[gp]) >> 16);
                a1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[gp]) >> 16);
            }
            #pragma unroll
            for (int gp = 0; gp < crr_b_pack_count; gp++) {
                b0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[gp]) >> 16);
                b1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[gp]) >> 16);
            }
        }
        fp8e8m0_4 a_sel_packs[crr_a_pack_count];
        #pragma unroll
        for (int p = 0; p < crr_a_pack_count; p++) {
            a_sel_packs[p] = (mhalf == 0) ? a0_scale_packs[p] : a1_scale_packs[p];
        }

        load_b(b0, Bs[0][0], wn);
        load_b(b1, Bs[0][1], wn);
        load_a(a, As[0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a_sel_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a_sel_packs, b1_scale_packs);

        __builtin_amdgcn_s_barrier();
        if (k + 1 < k_iters) {
            global_load_b(Bs[0][0], bc * 2,     k + 1);
            global_load_a(As[0],    a_tile_row(), k + 1);
            global_load_b(Bs[0][1], bc * 2 + 1, k + 1);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
        }
    }

#elif (MXFP8_CRR_HBSHRINK_PIPELINE == 1)
    // -----------------------------------------------------------------------
    // Stage B1: double-buffered LDS main loop with classic cross-buffer
    // prefetch (read tic, prefetch into toc for k+1) + optional B1 LDS
    // interleave inside the cA MMA chain. Two LDS outer slots eliminate
    // the in-place-prefetch race that cripples a single-slot DB.
    //
    // Schedule per iter k (0 .. k_iters-2):
    //   read tic for k     →  cA(b1 interleave) → cB
    //   prefetch toc for k+1 (issued before the wait so VMEM overlaps MMA)
    //   wait vmcnt + lgkm; barrier; tic ^= 1, toc ^= 1
    // Final iter k = k_iters-1 reads tic only (no prefetch).
    // -----------------------------------------------------------------------
    int tic = 0, toc = 1;
    // Initial loads: prime k=0 in As[tic]/Bs[tic][0..1].
    global_load_b(Bs[tic][0], bc * 2,     0);
    global_load_a(As[tic],    a_tile_row(), 0);
    global_load_b(Bs[tic][1], bc * 2 + 1, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)
    for (int k = 0; k < k_iters - 1; k++, tic ^= 1, toc ^= 1) {
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int gp = 0; gp < crr_a_pack_count; gp++) {
                a0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[gp]) >> 16);
                a1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[gp]) >> 16);
            }
            #pragma unroll
            for (int gp = 0; gp < crr_b_pack_count; gp++) {
                b0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[gp]) >> 16);
                b1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[gp]) >> 16);
            }
        }
        fp8e8m0_4 a_sel_packs[crr_a_pack_count];
        #pragma unroll
        for (int p = 0; p < crr_a_pack_count; p++) {
            a_sel_packs[p] = (mhalf == 0) ? a0_scale_packs[p] : a1_scale_packs[p];
        }

        // Issue prefetch for k+1 into toc (different slot than tic). VMEM
        // overlaps with subsequent LDS reads + MMAs.
        global_load_b(Bs[toc][0], bc * 2,     k + 1);
        global_load_a(As[toc],    a_tile_row(), k + 1);
        global_load_b(Bs[toc][1], bc * 2 + 1, k + 1);

        // LDS reads: b0, a; defer b1 into MMA chain interleave.
        load_b(b0, Bs[tic][0], wn);
        load_a(a, As[tic], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");

        // cA MMA chain with b1 LDS-interleave at INSERT_AFTER position.
        crr_hbshrink_cA_with_b1_interleave_fixed_phase<MXFP8_CRR_HBSHRINK_B1_INSERT_AFTER, crr_a_pack_count, crr_b_pack_count>(
            cA, a, b0, b1, Bs[tic][1], wn, a_sel_packs, b0_scale_packs);
        asm volatile("s_waitcnt lgkmcnt(0)");

        // cB MMA (uses b1 just loaded by interleave).
        __builtin_amdgcn_s_setprio(1);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a_sel_packs, b1_scale_packs);
        __builtin_amdgcn_s_setprio(0);
        CRR_SCHED_BARRIER();

        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }

    // Final iter k = k_iters - 1: read tic (toc was just primed by last
    // prefetch). No more VMEM.
    {
        const int k = k_iters - 1;
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int gp = 0; gp < crr_a_pack_count; gp++) {
                a0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[gp]) >> 16);
                a1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[gp]) >> 16);
            }
            #pragma unroll
            for (int gp = 0; gp < crr_b_pack_count; gp++) {
                b0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[gp]) >> 16);
                b1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[gp]) >> 16);
            }
        }
        fp8e8m0_4 a_sel_packs[crr_a_pack_count];
        #pragma unroll
        for (int p = 0; p < crr_a_pack_count; p++) {
            a_sel_packs[p] = (mhalf == 0) ? a0_scale_packs[p] : a1_scale_packs[p];
        }
        load_b(b0, Bs[tic][0], wn);
        load_b(b1, Bs[tic][1], wn);
        load_a(a, As[tic], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a_sel_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a_sel_packs, b1_scale_packs);
        __builtin_amdgcn_s_barrier();
    }

#elif (MXFP8_CRR_HBSHRINK_PIPELINE == 2)
    // -----------------------------------------------------------------------
    // Stage B2: SB PIPE=3 form (single-buffered LDS, early-issue VMEM,
    // late-wait). Mirrors R32 Dev B Approach 3 with single MMA pair.
    // -----------------------------------------------------------------------
    global_load_b(Bs[0][0], bc * 2,     0);
    global_load_a(As[0],    a_tile_row(), 0);
    global_load_b(Bs[0][1], bc * 2 + 1, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    for (int k = 0; k < k_iters - 1; k++) {
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int gp = 0; gp < crr_a_pack_count; gp++) {
                a0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[gp]) >> 16);
                a1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[gp]) >> 16);
            }
            #pragma unroll
            for (int gp = 0; gp < crr_b_pack_count; gp++) {
                b0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[gp]) >> 16);
                b1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[gp]) >> 16);
            }
        }
        fp8e8m0_4 a_sel_packs[crr_a_pack_count];
        #pragma unroll
        for (int p = 0; p < crr_a_pack_count; p++) {
            a_sel_packs[p] = (mhalf == 0) ? a0_scale_packs[p] : a1_scale_packs[p];
        }
        // Issue all 3 LDS reads up front.
        load_b(b0, Bs[0][0], wn);
        load_b(b1, Bs[0][1], wn);
        load_a(a, As[0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        // Issue all next-iter VMEM→LDS writes (asynchronous).
        global_load_b(Bs[0][0], bc * 2,     k + 1);
        global_load_a(As[0],    a_tile_row(), k + 1);
        global_load_b(Bs[0][1], bc * 2 + 1, k + 1);
        // MMAs overlap with VMEM in flight.
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a_sel_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a_sel_packs, b1_scale_packs);
        CRR_MMA_END();
        // Drain VMEM at end-of-iter.
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
    // Final iter (k = k_iters - 1): no more VMEM to issue.
    {
        const int k = k_iters - 1;
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int gp = 0; gp < crr_a_pack_count; gp++) {
                a0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[gp]) >> 16);
                a1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[gp]) >> 16);
            }
            #pragma unroll
            for (int gp = 0; gp < crr_b_pack_count; gp++) {
                b0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[gp]) >> 16);
                b1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[gp]) >> 16);
            }
        }
        fp8e8m0_4 a_sel_packs[crr_a_pack_count];
        #pragma unroll
        for (int p = 0; p < crr_a_pack_count; p++) {
            a_sel_packs[p] = (mhalf == 0) ? a0_scale_packs[p] : a1_scale_packs[p];
        }
        load_b(b0, Bs[0][0], wn);
        load_b(b1, Bs[0][1], wn);
        load_a(a, As[0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a_sel_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a_sel_packs, b1_scale_packs);
        __builtin_amdgcn_s_barrier();
    }

#elif (MXFP8_CRR_HBSHRINK_PIPELINE == 3)
    // -----------------------------------------------------------------------
    // Stage B3: SB PIPE=3 + B1 LDS interleave inside the cA MMA chain.
    // Combines benefits of B1 interleave (b0/a issue + drain asynchronously
    // of b1) with PIPE=3 VMEM-overlap.
    // -----------------------------------------------------------------------
    global_load_b(Bs[0][0], bc * 2,     0);
    global_load_a(As[0],    a_tile_row(), 0);
    global_load_b(Bs[0][1], bc * 2 + 1, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    for (int k = 0; k < k_iters - 1; k++) {
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int gp = 0; gp < crr_a_pack_count; gp++) {
                a0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[gp]) >> 16);
                a1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[gp]) >> 16);
            }
            #pragma unroll
            for (int gp = 0; gp < crr_b_pack_count; gp++) {
                b0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[gp]) >> 16);
                b1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[gp]) >> 16);
            }
        }
        fp8e8m0_4 a_sel_packs[crr_a_pack_count];
        #pragma unroll
        for (int p = 0; p < crr_a_pack_count; p++) {
            a_sel_packs[p] = (mhalf == 0) ? a0_scale_packs[p] : a1_scale_packs[p];
        }
        // Issue b0, a LDS reads (defer b1 into cA chain).
        load_b(b0, Bs[0][0], wn);
        load_a(a, As[0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        // cA MMA chain with b1 LDS interleave.
        crr_hbshrink_cA_with_b1_interleave_fixed_phase<MXFP8_CRR_HBSHRINK_B1_INSERT_AFTER, crr_a_pack_count, crr_b_pack_count>(
            cA, a, b0, b1, Bs[0][1], wn, a_sel_packs, b0_scale_packs);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        // Issue all next-iter VMEM→LDS writes BEFORE cB MMA so they overlap.
        global_load_b(Bs[0][0], bc * 2,     k + 1);
        global_load_a(As[0],    a_tile_row(), k + 1);
        global_load_b(Bs[0][1], bc * 2 + 1, k + 1);
        // cB MMA — overlaps with VMEM in flight.
        __builtin_amdgcn_s_setprio(1);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a_sel_packs, b1_scale_packs);
        __builtin_amdgcn_s_setprio(0);
        CRR_SCHED_BARRIER();
        // Drain VMEM at end-of-iter.
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
    // Final iter (k = k_iters - 1): no more VMEM to issue.
    {
        const int k = k_iters - 1;
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int gp = 0; gp < crr_a_pack_count; gp++) {
                a0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[gp]) >> 16);
                a1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[gp]) >> 16);
            }
            #pragma unroll
            for (int gp = 0; gp < crr_b_pack_count; gp++) {
                b0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[gp]) >> 16);
                b1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[gp]) >> 16);
            }
        }
        fp8e8m0_4 a_sel_packs[crr_a_pack_count];
        #pragma unroll
        for (int p = 0; p < crr_a_pack_count; p++) {
            a_sel_packs[p] = (mhalf == 0) ? a0_scale_packs[p] : a1_scale_packs[p];
        }
        load_b(b0, Bs[0][0], wn);
        load_b(b1, Bs[0][1], wn);
        load_a(a, As[0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a_sel_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a_sel_packs, b1_scale_packs);
        __builtin_amdgcn_s_barrier();
    }

#elif (MXFP8_CRR_HBSHRINK_PIPELINE == 4)
    // -----------------------------------------------------------------------
    // R37 Dev C — Stage B3v2: reordered B3 hybrid testing the "sequencing
    // bug, not structural" hypothesis for the R36 Dev A B3 -6.27% regression.
    //
    // R36 Dev A B3 (PIPE=3) had:
    //   load b0,a → wait → BARRIER → cA(b1 interleave) → wait → BARRIER →
    //   issue VMEM → cB → wait vmcnt → BARRIER
    //
    // The hypothesis: B3 issues the next-iter VMEM write into Bs[0][1]
    // BETWEEN cA (which contains the b1 LDS read) and cB (which consumes b1).
    // The b1 LDS read inside cA may still be in flight (no lgkmcnt drain on
    // it specifically before VMEM issue) when the VMEM write into the same
    // slot starts, causing a same-slot LDS write/read hazard. Also there are
    // TWO redundant barriers (between cA and VMEM, after lgkmcnt waits)
    // serializing MMA execution.
    //
    // R37 Dev C B3v2 fix: ISSUE VMEM AFTER cB (not between cA and cB), AND
    // remove the redundant intermediate barrier. Schedule:
    //   load b0,a from slot 0 → wait lgkm → BARRIER →
    //   cA(with b1 LDS interleave from slot 0) → wait lgkm →
    //   cB(consumes b1) → BARRIER (sync MMA-side LDS reads) →
    //   issue VMEM next-iter into slot 0 → wait vmcnt → BARRIER
    // -----------------------------------------------------------------------
    global_load_b(Bs[0][0], bc * 2,     0);
    global_load_a(As[0],    a_tile_row(), 0);
    global_load_b(Bs[0][1], bc * 2 + 1, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    for (int k = 0; k < k_iters - 1; k++) {
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int gp = 0; gp < crr_a_pack_count; gp++) {
                a0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[gp]) >> 16);
                a1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[gp]) >> 16);
            }
            #pragma unroll
            for (int gp = 0; gp < crr_b_pack_count; gp++) {
                b0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[gp]) >> 16);
                b1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[gp]) >> 16);
            }
        }
        fp8e8m0_4 a_sel_packs[crr_a_pack_count];
        #pragma unroll
        for (int p = 0; p < crr_a_pack_count; p++) {
            a_sel_packs[p] = (mhalf == 0) ? a0_scale_packs[p] : a1_scale_packs[p];
        }
        // Issue b0, a LDS reads (defer b1 into cA chain).
        load_b(b0, Bs[0][0], wn);
        load_a(a, As[0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        // cA MMA chain with b1 LDS interleave (b1 LDS read happens inside).
        crr_hbshrink_cA_with_b1_interleave_fixed_phase<MXFP8_CRR_HBSHRINK_B1_INSERT_AFTER, crr_a_pack_count, crr_b_pack_count>(
            cA, a, b0, b1, Bs[0][1], wn, a_sel_packs, b0_scale_packs);
        asm volatile("s_waitcnt lgkmcnt(0)");
        // cB MMA — uses b1 just loaded by interleave. NO intermediate
        // barrier between cA and cB (vs B3): both read from same slot/regs.
        __builtin_amdgcn_s_setprio(1);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a_sel_packs, b1_scale_packs);
        __builtin_amdgcn_s_setprio(0);
        CRR_SCHED_BARRIER();
        // Sync ALL MMA-side LDS reads complete before issuing VMEM writes
        // into the same slot — this is the key fix vs B3 which had VMEM
        // issued WHILE cA's b1 read was still in flight.
        __builtin_amdgcn_s_barrier();
        // Now safe to issue next-iter VMEM→LDS writes into slot 0.
        global_load_b(Bs[0][0], bc * 2,     k + 1);
        global_load_a(As[0],    a_tile_row(), k + 1);
        global_load_b(Bs[0][1], bc * 2 + 1, k + 1);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
    // Final iter (k = k_iters - 1): no more VMEM to issue.
    {
        const int k = k_iters - 1;
        if ((k & 1) == 0) {
            load_raw_scales(k >> 1);
        } else {
            #pragma unroll
            for (int gp = 0; gp < crr_a_pack_count; gp++) {
                a0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a0_scale_packs[gp]) >> 16);
                a1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(a1_scale_packs[gp]) >> 16);
            }
            #pragma unroll
            for (int gp = 0; gp < crr_b_pack_count; gp++) {
                b0_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[gp]) >> 16);
                b1_scale_packs[gp] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[gp]) >> 16);
            }
        }
        fp8e8m0_4 a_sel_packs[crr_a_pack_count];
        #pragma unroll
        for (int p = 0; p < crr_a_pack_count; p++) {
            a_sel_packs[p] = (mhalf == 0) ? a0_scale_packs[p] : a1_scale_packs[p];
        }
        load_b(b0, Bs[0][0], wn);
        load_b(b1, Bs[0][1], wn);
        load_a(a, As[0], wm);
        asm volatile("s_waitcnt lgkmcnt(0)");
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a_sel_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a_sel_packs, b1_scale_packs);
        __builtin_amdgcn_s_barrier();
    }

#endif  // MXFP8_CRR_HBSHRINK_PIPELINE

    if (wm == 0) {
        __builtin_amdgcn_s_barrier();
    }

    if constexpr (PRESHUFFLED_QUANT) {
        mul(cA, cA, g.scale);
        mul(cB, cB, g.scale);
    }

    // Store coords: each block stores HBSHRINK_BLK_M=128 rows × BLK_N=256 cols.
    const int macro_row =
        br_orig * WARPS_M * 2 + mhalf * WARPS_M + wm;
    store(g.c, cA, {0, 0, macro_row, bc * WARPS_N * 2 + wn});
    store(g.c, cB, {0, 0, macro_row, bc * WARPS_N * 2 + WARPS_N + wn});
}

// Host-side dispatch entry — wired into gemm_*_pq_v2 path in
// kernel_mxfp8_layouts.cpp behind MXFP8_CRR_BLK_M=128.
__host__ inline bool crr_can_use_exact_8wave_scaled_hbshrink(const layout_globals& g)
{
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM
        && (g.m % HBSHRINK_BLK_M == 0)
        && (g.n % BLK == 0)
        && (g.k % BK == 0);
}

template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch_crr_exact_8wave_scaled_v2_hbshrink(const layout_globals& g)
{
    const dim3 grid((g.m / HBSHRINK_BLK_M) * (g.n / BLK));
    crr_exact_8wave_scaled_hbshrink_kernel<PRESHUFFLED_QUANT, 2>
        <<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

#ifndef MXFP8_CRR_HBSHRINK_PROBE
#define MXFP8_CRR_HBSHRINK_PROBE 0
#endif
#if MXFP8_CRR_HBSHRINK_PROBE
__host__ inline void* crr_hbshrink_probe_addr_handle()
{
    return reinterpret_cast<void*>(&dispatch_crr_exact_8wave_scaled_v2_hbshrink<true>);
}
#endif

#endif  // MXFP8_CRR_HBSHRINK_8WAVE_FAST_ENABLE

#endif  // MXFP8_CRR_BLK_M == 128
