#pragma once

#ifndef MXFP8_RRR_EXACT_8WAVE_FAST_ENABLE
#define MXFP8_RRR_EXACT_8WAVE_FAST_ENABLE 0
#endif
#ifndef MXFP8_RRR_EXACT_PQ_PIPELINE_SCALE_ENABLE
#define MXFP8_RRR_EXACT_PQ_PIPELINE_SCALE_ENABLE 0
#endif

#if MXFP8_RRR_EXACT_8WAVE_FAST_ENABLE

static_assert(BLK == 256, "MXFP8 exact 8-wave fast path requires BLK=256");
static_assert(BK == 128, "MXFP8 exact 8-wave fast path requires BK=128");
static_assert(WARPS_M == 2, "MXFP8 exact 8-wave fast path requires WARPS_M=2");
static_assert(WARPS_N == 4, "MXFP8 exact 8-wave fast path requires WARPS_N=4");

template<bool PRESHUFFLED_QUANT>
__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void rrr_exact_8wave_scaled_kernel(const layout_globals g) {
    const int blocks_per_col = g.n / BLK;
    const int k_iters = g.k / BK;

    __shared__ ST_row As[2][2];
#if RRR_ROW_SHARED_TRANSPOSE
    __shared__ ST_row Bs[2][2];
#else
    __shared__ ST_rrr_b Bs[2][2];
#endif

    A_row_reg a;
    B_col_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    const int bid = blockIdx.x;
    const int br = bid / blocks_per_col;
    const int bc = bid % blocks_per_col;
    const int wm = warpid() / WARPS_N;
    const int wn = warpid() % WARPS_N;

    auto rrr_scale_a_base = [&](int half) {
        return br * BLK + half * HB + wm * RBM;
    };
    auto rrr_scale_b_base = [&](int half) {
        return bc * BLK + half * HB + wn * RBN;
    };
    const int lane_nonk = kittens::laneid() % 16;
    const int lane_kblk = kittens::laneid() / 16;
    const uint32_t lane_scale_byte_offset =
        (static_cast<uint32_t>(lane_kblk) << 6) |
        (static_cast<uint32_t>(lane_nonk) << 2);

    constexpr int rrr_a_pack_count = RBM / 32;
    constexpr int rrr_b_pack_count = (RBN + 31) / 32;
    fp8e8m0_4 a0_scale_packs[rrr_a_pack_count], a1_scale_packs[rrr_a_pack_count];
    fp8e8m0_4 b0_scale_packs[rrr_b_pack_count], b1_scale_packs[rrr_b_pack_count];
    const uint8_t* a0_scale_row_bases[rrr_a_pack_count];
    const uint8_t* a1_scale_row_bases[rrr_a_pack_count];
    const uint8_t* b0_scale_row_bases[rrr_b_pack_count];
    const uint8_t* b1_scale_row_bases[rrr_b_pack_count];
#if MXFP8_RRR_EXACT_PQ_PIPELINE_SCALE_ENABLE
    i32x4 a0p0_srsrc = {}, a0p1_srsrc = {}, a1p0_srsrc = {}, a1p1_srsrc = {};
    i32x4 b0p0_srsrc = {}, b1p0_srsrc = {};
#endif
    if constexpr (PRESHUFFLED_QUANT) {
        #pragma unroll
        for (int p = 0; p < rrr_a_pack_count; ++p) {
            a0_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.a_scale, (rrr_scale_a_base(0) + p * 32) >> 5);
            a1_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.a_scale, (rrr_scale_a_base(1) + p * 32) >> 5);
        }
        #pragma unroll
        for (int p = 0; p < rrr_b_pack_count; ++p) {
            b0_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.b_scale, (rrr_scale_b_base(0) + p * 32) >> 5);
            b1_scale_row_bases[p] = preshuffled_scale_row_base_ptr(
                g.b_scale, (rrr_scale_b_base(1) + p * 32) >> 5);
        }
#if MXFP8_RRR_EXACT_PQ_PIPELINE_SCALE_ENABLE
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
            if constexpr (rrr_a_pack_count > 1) {
                a0p1_srsrc = make_scale_srd(a0_scale_row_bases[1]);
                a1p1_srsrc = make_scale_srd(a1_scale_row_bases[1]);
            }
        }
#endif
    }
    auto load_scale_packs = [&](int k_pair) __attribute__((always_inline)) {
#if MXFP8_RRR_EXACT_PQ_PIPELINE_SCALE_ENABLE
        if constexpr (PRESHUFFLED_QUANT) {
            const uint32_t soff = static_cast<uint32_t>(k_pair) << 8;
            a0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(a0p0_srsrc, lane_scale_byte_offset, soff, 0));
            a1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(a1p0_srsrc, lane_scale_byte_offset, soff, 0));
            if constexpr (rrr_a_pack_count > 1) {
                a0_scale_packs[1] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(a0p1_srsrc, lane_scale_byte_offset, soff, 0));
                a1_scale_packs[1] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(a1p1_srsrc, lane_scale_byte_offset, soff, 0));
            }
            b0_scale_packs[0] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(b0p0_srsrc, lane_scale_byte_offset, soff, 0));
            b1_scale_packs[0] = std::bit_cast<fp8e8m0_4>(llvm_amdgcn_raw_buffer_load_b32(b1p0_srsrc, lane_scale_byte_offset, soff, 0));
            return;
        }
#endif
        #pragma unroll
        for (int p = 0; p < rrr_a_pack_count; ++p) {
            if constexpr (PRESHUFFLED_QUANT) {
                a0_scale_packs[p] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    a0_scale_row_bases[p], k_pair, lane_scale_byte_offset);
                a1_scale_packs[p] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    a1_scale_row_bases[p], k_pair, lane_scale_byte_offset);
            } else {
                a0_scale_packs[p] = load_scale_pair_pack_16x128(
                    g.a_scale, rrr_scale_a_base(0) + p * 32, k_pair, lane_nonk, lane_kblk);
                a1_scale_packs[p] = load_scale_pair_pack_16x128(
                    g.a_scale, rrr_scale_a_base(1) + p * 32, k_pair, lane_nonk, lane_kblk);
            }
        }
        #pragma unroll
        for (int p = 0; p < rrr_b_pack_count; ++p) {
            if constexpr (PRESHUFFLED_QUANT) {
                b0_scale_packs[p] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    b0_scale_row_bases[p], k_pair, lane_scale_byte_offset);
                b1_scale_packs[p] = load_scale_pair_pack_16x128_preshuffled_from_row_base(
                    b1_scale_row_bases[p], k_pair, lane_scale_byte_offset);
            } else {
                b0_scale_packs[p] = load_scale_pair_pack_16x128(
                    g.b_scale, rrr_scale_b_base(0) + p * 32, k_pair, lane_nonk, lane_kblk);
                b1_scale_packs[p] = load_scale_pair_pack_16x128(
                    g.b_scale, rrr_scale_b_base(1) + p * 32, k_pair, lane_nonk, lane_kblk);
            }
        }
    };

    constexpr int bptA = ST_row::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * _NUM_THREADS;
    constexpr int mptA = ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);

    constexpr int bptB =
#if RRR_ROW_SHARED_TRANSPOSE
        ST_row::underlying_subtile_bytes_per_thread;
#else
        ST_rrr_b::underlying_subtile_bytes_per_thread;
#endif
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB =
#if RRR_ROW_SHARED_TRANSPOSE
        ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmB;
#else
        ST_rrr_b::rows * ST_rrr_b::cols * sizeof(fp8e4m3) / bpmB;
#endif
    uint32_t soB[mptB];
#if RRR_ROW_SHARED_TRANSPOSE
    prefill_transpose_swizzled_offsets<_NUM_THREADS>(Bs[0][0], g.b, soB);
#else
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);
#endif

    auto a_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };
#if RRR_ROW_SHARED_TRANSPOSE
    auto b_co = [&](int s, int k) -> coord<ST_row> { return {0, 0, s, k}; };
#else
    auto b_co = [&](int s, int k) -> coord<ST_rrr_b> { return {0, 0, k, s}; };
#endif

    auto load_a = [&](A_row_reg& dst, ST_row& tile, int wi) {
        auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
        load(dst, sub);
    };
#if RRR_ROW_SHARED_TRANSPOSE
    auto load_b = [&](B_col_reg& dst, ST_row& tile, int wi) {
        B_row_reg tmp;
        auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
        load(tmp, sub);
        transpose(dst, tmp);
    };
#else
    auto load_b = [&](B_col_reg& dst, ST_rrr_b& tile, int wi) {
    #if RRR_B_REG_ROW_LOAD_TRANSPOSE
    #if RRR_B_REG_ROW_LOAD_ALIAS
    #if RRR_USE_V2A_SWIZZLE
        load_col_from_v2a_st(dst, tile, wi * RBN);
    #else
        load_col_from_v2_st(dst, tile, wi * RBN);
    #endif
    #else
        auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
        B_row_reg tmp;
        load(tmp, sub);
        transpose(dst, tmp);
    #endif
    #else
    #if RRR_USE_V2A_SWIZZLE
        load_col_from_v2a_st(dst, tile, wi * RBN);
    #else
        load_col_from_v2_st(dst, tile, wi * RBN);
    #endif
    #endif
    };
#endif

    int tic = 0, toc = 1;
#if RRR_ROW_SHARED_TRANSPOSE
    load_transpose<_NUM_THREADS>(Bs[tic][0], g.b, b_co(bc * 2, 0), soB);
#else
    G::load(Bs[tic][0], g.b, b_co(bc * 2, 0), soB);
#endif
    G::load(As[tic][0], g.a, a_co(br * 2, 0), soA);
#if RRR_ROW_SHARED_TRANSPOSE
    load_transpose<_NUM_THREADS>(Bs[tic][1], g.b, b_co(bc * 2 + 1, 0), soB);
#else
    G::load(Bs[tic][1], g.b, b_co(bc * 2 + 1, 0), soB);
#endif
    G::load(As[tic][1], g.a, a_co(br * 2 + 1, 0), soA);

    if (wm == 1) {
        __builtin_amdgcn_s_barrier();
    }
    TK_WAIT_VMCNT(RRR_INIT0_VMCNT);
    __builtin_amdgcn_s_barrier();

#if RRR_ROW_SHARED_TRANSPOSE
    load_transpose<_NUM_THREADS>(Bs[toc][0], g.b, b_co(bc * 2, 1), soB);
#else
    G::load(Bs[toc][0], g.b, b_co(bc * 2, 1), soB);
#endif
    G::load(As[toc][0], g.a, a_co(br * 2, 1), soA);
#if RRR_ROW_SHARED_TRANSPOSE
    load_transpose<_NUM_THREADS>(Bs[toc][1], g.b, b_co(bc * 2 + 1, 1), soB);
#else
    G::load(Bs[toc][1], g.b, b_co(bc * 2 + 1, 1), soB);
#endif

    TK_WAIT_VMCNT(RRR_INIT1_VMCNT);
    __builtin_amdgcn_s_barrier();

    auto do_k_iter = [&](int k, auto phase_tag) __attribute__((always_inline)) {
        constexpr int phase = decltype(phase_tag)::value;

        load_b(b0, Bs[tic][0], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        const auto b0_keep = b0;
#endif
        load_a(a, As[tic][0], wm);
        G::load(As[toc][1], g.a, a_co(br * 2 + 1, k + 1), soA);
        TK_WAIT_LGKM(RRR_PREFETCH_LGKM);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<phase>(cA, a, b0_keep, a0_scale_packs, b0_scale_packs);
#else
        rrr_mma_scaled_phase<phase>(cA, a, b0, a0_scale_packs, b0_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        RRR_SCHED_BARRIER();

        load_b(b1, Bs[tic][1], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        const auto b1_keep = b1;
#endif
#if RRR_ROW_SHARED_TRANSPOSE
        load_transpose<_NUM_THREADS>(Bs[tic][0], g.b, b_co(bc * 2, k + 2), soB);
#else
        G::load(Bs[tic][0], g.b, b_co(bc * 2, k + 2), soB);
#endif
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<phase>(cB, a, b1_keep, a0_scale_packs, b1_scale_packs);
#else
        rrr_mma_scaled_phase<phase>(cB, a, b1, a0_scale_packs, b1_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
#if RRR_ROW_SHARED_TRANSPOSE
        load_transpose<_NUM_THREADS>(Bs[tic][1], g.b, b_co(bc * 2 + 1, k + 2), soB);
#else
        G::load(Bs[tic][1], g.b, b_co(bc * 2 + 1, k + 2), soB);
#endif
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<phase>(cC, a, b0_keep, a1_scale_packs, b0_scale_packs);
#else
        rrr_mma_scaled_phase<phase>(cC, a, b0, a1_scale_packs, b0_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        RRR_SCHED_BARRIER();

        G::load(As[tic][0], g.a, a_co(br * 2, k + 2), soA);
        TK_WAIT_VMCNT(RRR_STEADY_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<phase>(cD, a, b1_keep, a1_scale_packs, b1_scale_packs);
#else
        rrr_mma_scaled_phase<phase>(cD, a, b1, a1_scale_packs, b1_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    };

    {
        const int k_iters_main = k_iters - 2;
        const int k_pairs = k_iters_main / 2;
        for (int kp = 0; kp < k_pairs; kp++) {
            load_scale_packs(kp);
            do_k_iter(kp * 2, phase_t<0>{});
            tic ^= 1; toc ^= 1;
            do_k_iter(kp * 2 + 1, phase_t<1>{});
            tic ^= 1; toc ^= 1;
        }
    }

    {
        const int k = k_iters - 2;
        load_scale_packs(k >> 1);

        load_b(b0, Bs[tic][0], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        const auto b0_keep = b0;
#endif
        load_a(a, As[tic][0], wm);
        G::load(As[toc][1], g.a, a_co(br * 2 + 1, k + 1), soA);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<0>(cA, a, b0_keep, a0_scale_packs, b0_scale_packs);
#else
        rrr_mma_scaled_phase<0>(cA, a, b0, a0_scale_packs, b0_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        RRR_SCHED_BARRIER();

        load_b(b1, Bs[tic][1], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        const auto b1_keep = b1;
#endif
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<0>(cB, a, b1_keep, a0_scale_packs, b1_scale_packs);
#else
        rrr_mma_scaled_phase<0>(cB, a, b1, a0_scale_packs, b1_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<0>(cC, a, b0_keep, a1_scale_packs, b0_scale_packs);
#else
        rrr_mma_scaled_phase<0>(cC, a, b0, a1_scale_packs, b0_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_b(b0, Bs[toc][0], wn);
        TK_WAIT_VMCNT(RRR_EPILOGUE_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<0>(cD, a, b1_keep, a1_scale_packs, b1_scale_packs);
#else
        rrr_mma_scaled_phase<0>(cD, a, b1, a1_scale_packs, b1_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        RRR_SCHED_BARRIER();
        tic ^= 1;
        toc ^= 1;
    }

    {
        load_a(a, As[tic][0], wm);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        const auto b0_keep = b0;
#endif
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<1>(cA, a, b0_keep, a0_scale_packs, b0_scale_packs);
#else
        rrr_mma_scaled_phase<1>(cA, a, b0, a0_scale_packs, b0_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_b(b1, Bs[tic][1], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        const auto b1_keep = b1;
#endif
        __builtin_amdgcn_s_barrier();
        RRR_SCHED_BARRIER();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<1>(cB, a, b1_keep, a0_scale_packs, b1_scale_packs);
#else
        rrr_mma_scaled_phase<1>(cB, a, b1, a0_scale_packs, b1_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma_scaled_phase<1>(cC, a, b0_keep, a1_scale_packs, b0_scale_packs);
        rrr_mma_scaled_phase<1>(cD, a, b1_keep, a1_scale_packs, b1_scale_packs);
#else
        rrr_mma_scaled_phase<1>(cC, a, b0, a1_scale_packs, b0_scale_packs);
        rrr_mma_scaled_phase<1>(cD, a, b1, a1_scale_packs, b1_scale_packs);
#endif
        __builtin_amdgcn_s_setprio(0);
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

template<bool PRESHUFFLED_QUANT>
__host__ inline void dispatch_rrr_exact_8wave_scaled(const layout_globals& g) {
    const dim3 grid((g.m / BLK) * (g.n / BLK));
    rrr_exact_8wave_scaled_kernel<PRESHUFFLED_QUANT>
        <<<grid, dim3(_NUM_THREADS), 0, g.stream>>>(g);
}

__host__ inline bool rrr_can_use_exact_8wave_scaled(const layout_globals& g) {
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM;
}

#endif
