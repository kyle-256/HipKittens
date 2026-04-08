#pragma once

#ifndef RRR_USE_EXACT_8WAVE_FASTPATH
#define RRR_USE_EXACT_8WAVE_FASTPATH 1
#endif

#if RRR_USE_EXACT_8WAVE_FASTPATH

using RRR_exact_a_gl = gl<fp8e4m3, 1, 1, M_DIM, K_DIM>;
using RRR_exact_b_gl = gl<fp8e4m3, 1, 1, K_DIM, N_DIM>;
using RRR_exact_c_gl = gl<bf16, 1, 1, M_DIM, N_DIM>;

struct rrr_exact_8wave_globals {
    RRR_exact_a_gl a;
    RRR_exact_b_gl b;
    RRR_exact_c_gl c;
    hipStream_t stream;

    dim3 grid() { return dim3((M_DIM / BLK) * (N_DIM / BLK)); }
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

static_assert(BLK == 256, "Exact 8-wave fast path requires BLK=256");
static_assert(BK == 128, "Exact 8-wave fast path requires BK=128");
static_assert(WARPS_M == 2, "Exact 8-wave fast path requires WARPS_M=2");
static_assert(WARPS_N == 4, "Exact 8-wave fast path requires WARPS_N=4");

__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void rrr_exact_8wave_kernel(const rrr_exact_8wave_globals g) {
    constexpr int blocks_per_col = N_DIM / BLK;
    constexpr int k_iters = K_DIM / BK;

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

    TK_PRAGMA_UNROLL(RRR_MAIN_UNROLL)
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
        load_b(b0, Bs[tic][0], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        const auto b0_keep = b0;
#endif
        load_a(a, As[tic][0], wm);
        G::load(As[toc][1], g.a, a_co(br * 2 + 1, k + 1), soA);
        TK_WAIT_LGKM(RRR_PREFETCH_LGKM);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0_keep); __builtin_amdgcn_s_setprio(0);
#else
        __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
#endif
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
        rrr_mma(cB, a, b1_keep);
#else
        rrr_mma(cB, a, b1);
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
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0_keep); __builtin_amdgcn_s_setprio(0);
#else
        __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
#endif
        __builtin_amdgcn_s_barrier();
        RRR_SCHED_BARRIER();

        G::load(As[tic][0], g.a, a_co(br * 2, k + 2), soA);
        TK_WAIT_VMCNT(RRR_STEADY_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma(cD, a, b1_keep);
#else
        rrr_mma(cD, a, b1);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    {
        constexpr int k = k_iters - 2;

        load_b(b0, Bs[tic][0], wn);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        const auto b0_keep = b0;
#endif
        load_a(a, As[tic][0], wm);
        G::load(As[toc][1], g.a, a_co(br * 2 + 1, k + 1), soA);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0_keep); __builtin_amdgcn_s_setprio(0);
#else
        __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
#endif
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
        rrr_mma(cB, a, b1_keep);
#else
        rrr_mma(cB, a, b1);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0_keep); __builtin_amdgcn_s_setprio(0);
#else
        __builtin_amdgcn_s_setprio(1); rrr_mma(cC, a, b0); __builtin_amdgcn_s_setprio(0);
#endif
        __builtin_amdgcn_s_barrier();

        load_b(b0, Bs[toc][0], wn);
        TK_WAIT_VMCNT(RRR_EPILOGUE_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        __builtin_amdgcn_s_setprio(1); rrr_mma(cD, a, b1_keep); __builtin_amdgcn_s_setprio(0);
#else
        __builtin_amdgcn_s_setprio(1); rrr_mma(cD, a, b1); __builtin_amdgcn_s_setprio(0);
#endif
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
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0_keep); __builtin_amdgcn_s_setprio(0);
#else
        __builtin_amdgcn_s_setprio(1); rrr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
#endif
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
        rrr_mma(cB, a, b1_keep);
#else
        rrr_mma(cB, a, b1);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
#if RRR_B_REG_ROW_LOAD_TRANSPOSE && RRR_B_REG_ROW_LOAD_ALIAS
        rrr_mma(cC, a, b0_keep);
        rrr_mma(cD, a, b1_keep);
#else
        rrr_mma(cC, a, b0);
        rrr_mma(cD, a, b1);
#endif
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (wm == 0) {
        __builtin_amdgcn_s_barrier();
    }

    store(g.c, cA, {0, 0, br * WARPS_M * 2 + wm, bc * WARPS_N * 2 + wn});
    store(g.c, cB, {0, 0, br * WARPS_M * 2 + wm, bc * WARPS_N * 2 + WARPS_N + wn});
    store(g.c, cC, {0, 0, br * WARPS_M * 2 + WARPS_M + wm, bc * WARPS_N * 2 + wn});
    store(g.c, cD, {0, 0, br * WARPS_M * 2 + WARPS_M + wm, bc * WARPS_N * 2 + WARPS_N + wn});
}

__host__ inline bool rrr_can_use_exact_8wave(const layout_globals& g) {
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM && g.scale_a * g.scale_b == 1.0f;
}

__host__ inline void dispatch_rrr_exact_8wave(const layout_globals& g) {
    rrr_exact_8wave_globals eg{
        make_gl<RRR_exact_a_gl>(reinterpret_cast<uint64_t>(g.a.raw_ptr), 1, 1, g.m, g.k),
        make_gl<RRR_exact_b_gl>(reinterpret_cast<uint64_t>(g.b.raw_ptr), 1, 1, g.k, g.n),
        make_gl<RRR_exact_c_gl>(reinterpret_cast<uint64_t>(g.c.raw_ptr), 1, 1, g.m, g.n),
        g.stream,
    };
    rrr_exact_8wave_kernel<<<eg.grid(), eg.block(), 0, eg.stream>>>(eg);
}

#endif
