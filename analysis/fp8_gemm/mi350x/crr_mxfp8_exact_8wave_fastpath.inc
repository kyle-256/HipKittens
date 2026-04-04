#pragma once

#ifndef MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE
#define MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE 0
#endif
#ifndef MXFP8_CRR_EXACT_PQ_PIPELINE_SCALE_ENABLE
#define MXFP8_CRR_EXACT_PQ_PIPELINE_SCALE_ENABLE 0
#endif

#ifndef CRR_USE_EXACT_8WAVE_FASTPATH
#define CRR_USE_EXACT_8WAVE_FASTPATH 1
#endif

#ifndef CRR_EXACT_INTERLEAVE_B1_LDS
#define CRR_EXACT_INTERLEAVE_B1_LDS 1
#endif

#ifndef CRR_EXACT_B1_LDS_INSERT_AFTER
#define CRR_EXACT_B1_LDS_INSERT_AFTER 4
#endif

#ifndef CRR_EXACT_PREFETCH_LGKM
#define CRR_EXACT_PREFETCH_LGKM CRR_PREFETCH_LGKM
#endif

#ifndef CRR_EXACT_STEADY_VMCNT
#define CRR_EXACT_STEADY_VMCNT CRR_STEADY_VMCNT
#endif

#if MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE

#if CRR_A_LDS_REENCODE
#error "CRR_A_LDS_REENCODE not supported for MXFP8 CRR exact 8-wave fast path"
#endif

static_assert(BLK == 256, "MXFP8 exact 8-wave fast path requires BLK=256");
static_assert(BK == 128, "MXFP8 exact 8-wave fast path requires BK=128");
static_assert(WARPS_M == 2, "MXFP8 exact 8-wave fast path requires WARPS_M=2");
static_assert(WARPS_N == 4, "MXFP8 exact 8-wave fast path requires WARPS_N=4");
static_assert(!CRR_ROW_SHARED_TRANSPOSE, "Exact CRR fast path only supports strict shared layout");
static_assert(!CRR_USE_V3_SWIZZLE, "Exact CRR fast path is wired for v2/v2a shared layouts");
static_assert(CRR_A_REG_ROW_LOAD_TRANSPOSE == 1, "Exact CRR fast path expects default A row-load transpose");
static_assert(CRR_A_REG_ROW_LOAD_ALIAS == 1, "Exact CRR fast path expects default A row-load alias");
static_assert(CRR_B_REG_ROW_LOAD_TRANSPOSE == 0, "Exact CRR fast path expects default B col-load path");
static_assert(CRR_B_REG_ROW_LOAD_ALIAS == 0, "Exact CRR fast path expects default B col-load path");
static_assert(CRR_BATCHED_PAIR_MMA == 1, "Exact CRR fast path expects batched pair MMA");
static_assert(CRR_BATCHED_EPILOGUE_MMA == 0, "Exact CRR fast path expects scalar epilogue MMA");

template<int A_PACK_COUNT, int B_PACK_COUNT>
__device__ __forceinline__ void crr_exact_mma_one_scaled(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_col_reg& a,
    const B_col_reg& b,
    int n,
    int m,
    int k_phase,
    const fp8e8m0_4 (&a_scale_packs)[A_PACK_COUNT],
    const fp8e8m0_4 (&b_scale_packs)[B_PACK_COUNT])
{
    const int a_opsel = (k_phase << 1) | (n & 1);
    const int b_opsel = (k_phase << 1) | (m & 1);
    crr_mma_scaled_dispatch(
        acc,
        a,
        b,
        n,
        m,
        a_opsel,
        b_opsel,
        a_scale_packs[n / 2],
        b_scale_packs[m / 2]);
}

template<int INSERT_AFTER, int A_PACK_COUNT, int B_PACK_COUNT>
__device__ __forceinline__ void crr_exact_cA_with_b1_interleave_fixed_phase(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& cA,
    const A_col_reg& a,
    const B_col_reg& b0,
    B_col_reg& b1,
    const ST_v2& b1_tile,
    int wn,
    const fp8e8m0_4 (&a0_phase)[A_PACK_COUNT],
    const fp8e8m0_4 (&b0_phase)[B_PACK_COUNT])
{
    static_assert(INSERT_AFTER >= 0 && INSERT_AFTER <= 8);
    static_assert(RBM == 64 && RBN == 32);

    __builtin_amdgcn_s_setprio(1);
    if constexpr (INSERT_AFTER == 0) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<0, 0>(cA, a, b0, 0, 0, a0_phase[0], b0_phase[0]);
    if constexpr (INSERT_AFTER == 1) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<0, 1>(cA, a, b0, 0, 1, a0_phase[0], b0_phase[0]);
    if constexpr (INSERT_AFTER == 2) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<1, 0>(cA, a, b0, 1, 0, a0_phase[0], b0_phase[0]);
    if constexpr (INSERT_AFTER == 3) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<1, 1>(cA, a, b0, 1, 1, a0_phase[0], b0_phase[0]);
    if constexpr (INSERT_AFTER == 4) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<0, 0>(cA, a, b0, 2, 0, a0_phase[1], b0_phase[0]);
    if constexpr (INSERT_AFTER == 5) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<0, 1>(cA, a, b0, 2, 1, a0_phase[1], b0_phase[0]);
    if constexpr (INSERT_AFTER == 6) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<1, 0>(cA, a, b0, 3, 0, a0_phase[1], b0_phase[0]);
    if constexpr (INSERT_AFTER == 7) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<1, 1>(cA, a, b0, 3, 1, a0_phase[1], b0_phase[0]);
    if constexpr (INSERT_AFTER == 8) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    __builtin_amdgcn_s_setprio(0);
    CRR_SCHED_BARRIER();
}

template<int K_PHASE, int INSERT_AFTER, int A_PACK_COUNT, int B_PACK_COUNT>
__device__ __forceinline__ void crr_exact_cA_with_b1_interleave_raw_phase(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& cA,
    const A_col_reg& a,
    const B_col_reg& b0,
    B_col_reg& b1,
    const ST_v2& b1_tile,
    int wn,
    const fp8e8m0_4 (&a0_packs)[A_PACK_COUNT],
    const fp8e8m0_4 (&b0_packs)[B_PACK_COUNT])
{
    static_assert(K_PHASE == 0 || K_PHASE == 1);
    static_assert(INSERT_AFTER >= 0 && INSERT_AFTER <= 8);
    static_assert(RBM == 64 && RBN == 32);
    constexpr int lo = K_PHASE * 2;
    constexpr int hi = K_PHASE * 2 + 1;

    __builtin_amdgcn_s_setprio(1);
    if constexpr (INSERT_AFTER == 0) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<lo, lo>(cA, a, b0, 0, 0, a0_packs[0], b0_packs[0]);
    if constexpr (INSERT_AFTER == 1) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<lo, hi>(cA, a, b0, 0, 1, a0_packs[0], b0_packs[0]);
    if constexpr (INSERT_AFTER == 2) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<hi, lo>(cA, a, b0, 1, 0, a0_packs[0], b0_packs[0]);
    if constexpr (INSERT_AFTER == 3) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<hi, hi>(cA, a, b0, 1, 1, a0_packs[0], b0_packs[0]);
    if constexpr (INSERT_AFTER == 4) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<lo, lo>(cA, a, b0, 2, 0, a0_packs[1], b0_packs[0]);
    if constexpr (INSERT_AFTER == 5) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<lo, hi>(cA, a, b0, 2, 1, a0_packs[1], b0_packs[0]);
    if constexpr (INSERT_AFTER == 6) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<hi, lo>(cA, a, b0, 3, 0, a0_packs[1], b0_packs[0]);
    if constexpr (INSERT_AFTER == 7) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    crr_mma_scaled_base<hi, hi>(cA, a, b0, 3, 1, a0_packs[1], b0_packs[0]);
    if constexpr (INSERT_AFTER == 8) { CRR_SCHED_BARRIER(); load_col_from_v2_st(b1, b1_tile, wn * RBN); }
    __builtin_amdgcn_s_setprio(0);
    CRR_SCHED_BARRIER();
}

template<bool PRESHUFFLED_QUANT>
__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void crr_exact_8wave_scaled_kernel(const layout_globals g)
{
#if CRR_ROW_SHARED_TRANSPOSE
    using ST_crr_a = ST_row;
    using ST_crr_b = ST_row;
#elif CRR_USE_V3_SWIZZLE
    using ST_crr_a = ST_v3;
    using ST_crr_b = ST_v3;
#else
    using ST_crr_a = ST_v2a;
    using ST_crr_b = ST_v2;
#endif

    constexpr int crr_a_pack_count = RBM / 32;
    constexpr int crr_b_pack_count = (RBN + 31) / 32;

    const int k_iters = g.k / BK;
    const int blocks_per_col = g.n / BLK;

    __shared__ ST_crr_a As[2][2];
    __shared__ ST_crr_b Bs[2][2];

    A_col_reg a;
    B_col_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;
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

    auto crr_scale_a_base = [&](int half) {
        return br * BLK + half * (BLK / 2) + wm * RBM;
    };
    auto crr_scale_b_base = [&](int half) {
        return bc * BLK + half * (BLK / 2) + wn * RBN;
    };

    fp8e8m0_4 a0_scale_packs[crr_a_pack_count];
    fp8e8m0_4 a1_scale_packs[crr_a_pack_count];
    fp8e8m0_4 b0_scale_packs[crr_b_pack_count];
    fp8e8m0_4 b1_scale_packs[crr_b_pack_count];
    const uint8_t* a0_scale_row_bases[crr_a_pack_count];
    const uint8_t* a1_scale_row_bases[crr_a_pack_count];
    const uint8_t* b0_scale_row_bases[crr_b_pack_count];
    const uint8_t* b1_scale_row_bases[crr_b_pack_count];
#if MXFP8_CRR_EXACT_PQ_PIPELINE_SCALE_ENABLE
    i32x4 a0p0_srsrc = {}, a0p1_srsrc = {}, a1p0_srsrc = {}, a1p1_srsrc = {};
    i32x4 b0p0_srsrc = {}, b1p0_srsrc = {};
#endif
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
#if MXFP8_CRR_EXACT_PQ_PIPELINE_SCALE_ENABLE
        {
            auto make_scale_srd = [](const void* ptr) -> i32x4 {
                i32x4 srd = std::bit_cast<i32x4>(
                    make_buffer_resource(
                        static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(ptr)),
                        0xFFFFFFFFu,
                        0x00110000u
                    )
                );
                srd[0] = __builtin_amdgcn_readfirstlane(srd[0]);
                srd[1] = __builtin_amdgcn_readfirstlane(srd[1]);
                srd[2] = __builtin_amdgcn_readfirstlane(srd[2]);
                srd[3] = __builtin_amdgcn_readfirstlane(srd[3]);
                return srd;
            };
            a0p0_srsrc = make_scale_srd(a0_scale_row_bases[0]);
            a1p0_srsrc = make_scale_srd(a1_scale_row_bases[0]);
            b0p0_srsrc = make_scale_srd(b0_scale_row_bases[0]);
            b1p0_srsrc = make_scale_srd(b1_scale_row_bases[0]);
            if constexpr (crr_a_pack_count > 1) {
                a0p1_srsrc = make_scale_srd(a0_scale_row_bases[1]);
                a1p1_srsrc = make_scale_srd(a1_scale_row_bases[1]);
            }
        }
#endif
    }
    auto load_raw_scales = [&](int k_pair) __attribute__((always_inline)) {
#if MXFP8_CRR_EXACT_PQ_PIPELINE_SCALE_ENABLE
        if constexpr (PRESHUFFLED_QUANT) {
            const uint32_t soff = static_cast<uint32_t>(k_pair) << 8;
            a0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(a0p0_srsrc, lane_scale_byte_offset, soff, 0));
            a1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(a1p0_srsrc, lane_scale_byte_offset, soff, 0));
            if constexpr (crr_a_pack_count > 1) {
                a0_scale_packs[1] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(a0p1_srsrc, lane_scale_byte_offset, soff, 0));
                a1_scale_packs[1] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(a1p1_srsrc, lane_scale_byte_offset, soff, 0));
            }
            b0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(b0p0_srsrc, lane_scale_byte_offset, soff, 0));
            b1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(b1p0_srsrc, lane_scale_byte_offset, soff, 0));
            return;
        }
#endif
#pragma unroll
        for (int p = 0; p < crr_a_pack_count; ++p) {
            if constexpr (PRESHUFFLED_QUANT) {
                a0_scale_packs[p] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    a0_scale_row_bases[p], k_pair, lane_scale_byte_offset);
                a1_scale_packs[p] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    a1_scale_row_bases[p], k_pair, lane_scale_byte_offset);
            } else {
                a0_scale_packs[p] = load_scale_pair_pack_16x128(
                    g.a_scale, crr_scale_a_base(0) + p * 32, k_pair, lane_nonk, lane_kblk);
                a1_scale_packs[p] = load_scale_pair_pack_16x128(
                    g.a_scale, crr_scale_a_base(1) + p * 32, k_pair, lane_nonk, lane_kblk);
            }
        }
#pragma unroll
        for (int p = 0; p < crr_b_pack_count; ++p) {
            if constexpr (PRESHUFFLED_QUANT) {
                b0_scale_packs[p] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    b0_scale_row_bases[p], k_pair, lane_scale_byte_offset);
                b1_scale_packs[p] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    b1_scale_row_bases[p], k_pair, lane_scale_byte_offset);
            } else {
                b0_scale_packs[p] = load_scale_pair_pack_16x128(
                    g.b_scale, crr_scale_b_base(0) + p * 32, k_pair, lane_nonk, lane_kblk);
                b1_scale_packs[p] = load_scale_pair_pack_16x128(
                    g.b_scale, crr_scale_b_base(1) + p * 32, k_pair, lane_nonk, lane_kblk);
            }
        }
    };

    constexpr int bptA = ST_crr_a::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * _NUM_THREADS;
    constexpr int mptA = ST_crr_a::rows * ST_crr_a::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
#if CRR_ROW_SHARED_TRANSPOSE
    prefill_transpose_swizzled_offsets<_NUM_THREADS>(As[0][0], g.a, soA);
#else
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
#endif

    constexpr int bptB = ST_crr_b::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB = ST_crr_b::rows * ST_crr_b::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
#if CRR_ROW_SHARED_TRANSPOSE
    prefill_transpose_swizzled_offsets<_NUM_THREADS>(Bs[0][0], g.b, soB);
#else
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);
#endif

#if CRR_ROW_SHARED_TRANSPOSE
    auto a_co = [&](int s, int k) -> coord<ST_crr_a> { return {0, 0, s, k}; };
    auto b_co = [&](int s, int k) -> coord<ST_crr_b> { return {0, 0, s, k}; };
    auto global_load_a = [&](ST_crr_a& tile, int s, int k) {
        load_transpose<_NUM_THREADS>(tile, g.a, a_co(s, k), soA);
    };
    auto global_load_b = [&](ST_crr_b& tile, int s, int k) {
        load_transpose<_NUM_THREADS>(tile, g.b, b_co(s, k), soB);
    };
    auto load_a = [&](A_col_reg& dst, ST_crr_a& tile, int wi) {
        A_row_reg tmp;
        auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
        load(tmp, sub);
        transpose(dst, tmp);
    };
    auto load_b = [&](B_col_reg& dst, ST_crr_b& tile, int wi) {
        B_row_reg tmp;
        auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
        load(tmp, sub);
        transpose(dst, tmp);
    };
#else
    auto a_co = [&](int s, int k) -> coord<ST_crr_a> { return {0, 0, k, s}; };
    auto b_co = [&](int s, int k) -> coord<ST_crr_b> { return {0, 0, k, s}; };
    auto global_load_a = [&](ST_crr_a& tile, int s, int k) {
        G::load(tile, g.a, a_co(s, k), soA);
    };
    auto global_load_b = [&](ST_crr_b& tile, int s, int k) {
        G::load(tile, g.b, b_co(s, k), soB);
    };
    auto load_a = [&](A_col_reg& dst, ST_crr_a& tile, int wi) {
#if CRR_USE_V3_SWIZZLE
        load_col_from_v3_st(dst, tile, wi * RBM);
#else
#if CRR_A_REG_ROW_LOAD_TRANSPOSE
#if CRR_A_REG_ROW_LOAD_ALIAS
        load_col_from_v2a_st(dst, tile, wi * RBM);
#else
        A_row_reg tmp;
        auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
        load(tmp, sub);
        transpose(dst, tmp);
#endif
#else
        load_col_from_v2a_st(dst, tile, wi * RBM);
#endif
#endif
    };
    auto load_b = [&](B_col_reg& dst, ST_crr_b& tile, int wi) {
#if CRR_USE_V3_SWIZZLE
        load_col_from_v3_st(dst, tile, wi * RBN);
#else
#if CRR_B_REG_ROW_LOAD_TRANSPOSE
#if CRR_B_REG_ROW_LOAD_ALIAS
        load_col_from_v2_st(dst, tile, wi * RBN);
#else
        B_row_reg tmp;
        auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
        load(tmp, sub);
        transpose(dst, tmp);
#endif
#else
        load_col_from_v2_st(dst, tile, wi * RBN);
#endif
#endif
    };
#endif

    int tic = 0, toc = 1;
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
            for (int g = 0; g < crr_b_pack_count; g++) {
                b0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b0_scale_packs[g]) >> 16);
                b1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(std::bit_cast<uint32_t>(b1_scale_packs[g]) >> 16);
            }
        }

        load_b(b0, Bs[tic][0], wn);
#if !CRR_EXACT_INTERLEAVE_B1_LDS
        load_b(b1, Bs[tic][1], wn);
#endif
        load_a(a, As[tic][0], wm);
        global_load_a(As[toc][1], br * 2 + 1, k + 1);
        TK_WAIT_LGKM(CRR_EXACT_PREFETCH_LGKM);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
#if CRR_EXACT_INTERLEAVE_B1_LDS
        crr_exact_cA_with_b1_interleave_fixed_phase<CRR_EXACT_B1_LDS_INSERT_AFTER, crr_a_pack_count, crr_b_pack_count>(
            cA, a, b0, b1, Bs[tic][1], wn, a0_scale_packs, b0_scale_packs);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a0_scale_packs, b1_scale_packs);
        __builtin_amdgcn_s_setprio(0);
        CRR_SCHED_BARRIER();
#else
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cA, a, b0, a0_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cB, a, b1, a0_scale_packs, b1_scale_packs);
        CRR_MMA_END();
#endif
        CRR_STEADY_MID_BARRIER();
        CRR_SCHED_BARRIER();

        load_a(a, As[tic][1], wm);
        global_load_a(As[tic][0], br * 2, k + 2);
        global_load_b(Bs[tic][1], bc * 2 + 1, k + 2);
        TK_WAIT_VMCNT(CRR_EXACT_STEADY_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_from_packs_fixed_phase(cC, a, b0, a1_scale_packs, b0_scale_packs);
        crr_mma_scaled_from_packs_fixed_phase(cD, a, b1, a1_scale_packs, b1_scale_packs);
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
        crr_mma_scaled_phase<0>(cA, a, b0, a0_scale_packs, b0_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
        CRR_SCHED_BARRIER();

        load_b(b1, Bs[tic][1], wn);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<0>(cB, a, b1, a0_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<0>(cC, a, b0, a1_scale_packs, b0_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_b(b0, Bs[toc][0], wn);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<0>(cD, a, b1, a1_scale_packs, b1_scale_packs);
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
        crr_mma_scaled_phase<1>(cA, a, b0_keep, a0_scale_packs, b0_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_b(b1, Bs[tic][1], wn);
        const auto b1_keep = b1;
        __builtin_amdgcn_s_barrier();
        CRR_SCHED_BARRIER();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<1>(cB, a, b1_keep, a0_scale_packs, b1_scale_packs);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma_scaled_phase<1>(cC, a, b0_keep, a1_scale_packs, b0_scale_packs);
        crr_mma_scaled_phase<1>(cD, a, b1_keep, a1_scale_packs, b1_scale_packs);
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

    store(g.c, cA, {0, 0, br * WARPS_M * 2 + wm, bc * WARPS_N * 2 + wn});
    store(g.c, cB, {0, 0, br * WARPS_M * 2 + wm, bc * WARPS_N * 2 + WARPS_N + wn});
    store(g.c, cC, {0, 0, br * WARPS_M * 2 + WARPS_M + wm, bc * WARPS_N * 2 + wn});
    store(g.c, cD, {0, 0, br * WARPS_M * 2 + WARPS_M + wm, bc * WARPS_N * 2 + WARPS_N + wn});
}

__host__ inline bool crr_can_use_exact_8wave_scaled(const layout_globals& g)
{
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM;
}

template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch_crr_exact_8wave_scaled(const layout_globals& g)
{
    const dim3 grid((g.m / BLK) * (g.n / BLK));
    crr_exact_8wave_scaled_kernel<PRESHUFFLED_QUANT><<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

#endif
