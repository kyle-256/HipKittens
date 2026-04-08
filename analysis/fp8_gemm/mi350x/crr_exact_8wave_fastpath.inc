#pragma once

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

#if CRR_USE_EXACT_8WAVE_FASTPATH

using CRR_exact_a_gl = gl<fp8e4m3, 1, 1, K_DIM, M_DIM>;
using CRR_exact_b_gl = gl<fp8e4m3, 1, 1, K_DIM, N_DIM>;
using CRR_exact_c_gl = gl<bf16, 1, 1, M_DIM, N_DIM>;

struct crr_exact_8wave_globals {
    CRR_exact_a_gl a;
    CRR_exact_b_gl b;
    CRR_exact_c_gl c;
    float scale_a, scale_b;
    int group_m;
    hipStream_t stream;

    dim3 grid() { return dim3((M_DIM / BLK) * (N_DIM / BLK)); }
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

static_assert(BLK == 256, "Exact 8-wave fast path requires BLK=256");
static_assert(BK == 128, "Exact 8-wave fast path requires BK=128");
static_assert(WARPS_M == 2, "Exact 8-wave fast path requires WARPS_M=2");
static_assert(WARPS_N == 4, "Exact 8-wave fast path requires WARPS_N=4");
static_assert(!CRR_ROW_SHARED_TRANSPOSE, "Exact CRR fast path only supports strict shared layout");
static_assert(!CRR_USE_V3_SWIZZLE, "Exact CRR fast path is wired for v2/v2a shared layouts");
static_assert(!CRR_A_LDS_REENCODE, "Exact CRR fast path does not support A LDS reencode");
static_assert(CRR_A_REG_ROW_LOAD_TRANSPOSE == 1, "Exact CRR fast path expects default A row-load transpose");
static_assert(CRR_A_REG_ROW_LOAD_ALIAS == 1, "Exact CRR fast path expects default A row-load alias");
static_assert(CRR_B_REG_ROW_LOAD_TRANSPOSE == 0, "Exact CRR fast path expects default B col-load path");
static_assert(CRR_B_REG_ROW_LOAD_ALIAS == 0, "Exact CRR fast path expects default B col-load path");
static_assert(CRR_BATCHED_PAIR_MMA == 1, "Exact CRR fast path expects batched pair MMA");
static_assert(CRR_BATCHED_EPILOGUE_MMA == 0, "Exact CRR fast path expects scalar epilogue MMA");

__device__ __forceinline__ void crr_exact_mma_one(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_col_reg& a,
    const B_col_reg& b,
    int n,
    int m)
{
    const auto& a_row = reinterpret_cast<const A_row_reg&>(a);
    mma_AB_base(
        acc.tiles[n][m],
        a_row.tiles[n][0],
        b.tiles[0][m],
        acc.tiles[n][m]
    );
}

template<int INSERT_AFTER>
__device__ __forceinline__ void crr_exact_cA_with_b1_interleave(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& cA,
    const A_col_reg& a,
    const B_col_reg& b0,
    B_col_reg& b1,
    const ST_v2& b1_tile,
    int wn)
{
    static_assert(INSERT_AFTER >= 0 && INSERT_AFTER <= 8, "CRR exact b1 insert point must be in [0, 8]");

    __builtin_amdgcn_s_setprio(1);
    if constexpr (INSERT_AFTER == 0) {
        CRR_SCHED_BARRIER();
        load_col_from_v2_st(b1, b1_tile, wn * RBN);
    }
    crr_exact_mma_one(cA, a, b0, 0, 0);
    if constexpr (INSERT_AFTER == 1) {
        CRR_SCHED_BARRIER();
        load_col_from_v2_st(b1, b1_tile, wn * RBN);
    }
    crr_exact_mma_one(cA, a, b0, 0, 1);
    if constexpr (INSERT_AFTER == 2) {
        CRR_SCHED_BARRIER();
        load_col_from_v2_st(b1, b1_tile, wn * RBN);
    }
    crr_exact_mma_one(cA, a, b0, 1, 0);
    if constexpr (INSERT_AFTER == 3) {
        CRR_SCHED_BARRIER();
        load_col_from_v2_st(b1, b1_tile, wn * RBN);
    }
    crr_exact_mma_one(cA, a, b0, 1, 1);
    if constexpr (INSERT_AFTER == 4) {
        CRR_SCHED_BARRIER();
        load_col_from_v2_st(b1, b1_tile, wn * RBN);
    }
    crr_exact_mma_one(cA, a, b0, 2, 0);
    if constexpr (INSERT_AFTER == 5) {
        CRR_SCHED_BARRIER();
        load_col_from_v2_st(b1, b1_tile, wn * RBN);
    }
    crr_exact_mma_one(cA, a, b0, 2, 1);
    if constexpr (INSERT_AFTER == 6) {
        CRR_SCHED_BARRIER();
        load_col_from_v2_st(b1, b1_tile, wn * RBN);
    }
    crr_exact_mma_one(cA, a, b0, 3, 0);
    if constexpr (INSERT_AFTER == 7) {
        CRR_SCHED_BARRIER();
        load_col_from_v2_st(b1, b1_tile, wn * RBN);
    }
    crr_exact_mma_one(cA, a, b0, 3, 1);
    if constexpr (INSERT_AFTER == 8) {
        CRR_SCHED_BARRIER();
        load_col_from_v2_st(b1, b1_tile, wn * RBN);
    }
    __builtin_amdgcn_s_setprio(0);
    CRR_SCHED_BARRIER();
}

__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void crr_exact_8wave_kernel(const crr_exact_8wave_globals g) {
    constexpr int blocks_per_col = N_DIM / BLK;
    constexpr int k_iters = K_DIM / BK;
    using ST_crr_a = ST_v2a;
    using ST_crr_b = ST_v2;

    __shared__ ST_crr_a As[2][2];
    __shared__ ST_crr_b Bs[2][2];

    A_col_reg a;
    B_col_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    constexpr int blocks_per_row = M_DIM / BLK;
    int bid = blockIdx.x;
    int br, bc;
    gemm_compute_block_coords(bid, blocks_per_row, blocks_per_col, g.group_m, br, bc);
    const int wm = warpid() / WARPS_N;
    const int wn = warpid() % WARPS_N;

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
    auto load_b = [&](B_col_reg& dst, ST_crr_b& tile, int wi) {
        load_col_from_v2_st(dst, tile, wi * RBN);
    };

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
        crr_exact_cA_with_b1_interleave<CRR_EXACT_B1_LDS_INSERT_AFTER>(
            cA, a, b0, b1, Bs[tic][1], wn);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        crr_mma(cB, a, b1);
        __builtin_amdgcn_s_setprio(0);
        CRR_SCHED_BARRIER();
#else
        CRR_MMA_BEGIN();
        crr_mma(cA, a, b0);
        crr_mma(cB, a, b1);
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
        crr_mma(cC, a, b0);
        crr_mma(cD, a, b1);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
        global_load_b(Bs[tic][0], bc * 2, k + 2);
    }

    {
        constexpr int k = k_iters - 2;

        load_b(b0, Bs[tic][0], wn);
        const auto b0_keep = b0;
        load_a(a, As[tic][0], wm);
        global_load_a(As[toc][1], br * 2 + 1, k + 1);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma(cA, a, b0_keep);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
        CRR_SCHED_BARRIER();

        load_b(b1, Bs[tic][1], wn);
        const auto b1_keep = b1;
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma(cB, a, b1_keep);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        TK_WAIT_VMCNT(CRR_EPILOGUE_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma(cC, a, b0_keep);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_b(b0, Bs[toc][0], wn);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma(cD, a, b1_keep);
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
        crr_mma(cA, a, b0_keep);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_b(b1, Bs[tic][1], wn);
        const auto b1_keep = b1;
        __builtin_amdgcn_s_barrier();
        CRR_SCHED_BARRIER();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma(cB, a, b1_keep);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        CRR_MMA_BEGIN();
        crr_mma(cC, a, b0_keep);
        crr_mma(cD, a, b1_keep);
        CRR_MMA_END();
        __builtin_amdgcn_s_barrier();
    }

    if (wm == 0) {
        __builtin_amdgcn_s_barrier();
    }

    const float sc = g.scale_a * g.scale_b;
    mul(cA, cA, sc); mul(cB, cB, sc); mul(cC, cC, sc); mul(cD, cD, sc);

    store(g.c, cA, {0, 0, br * WARPS_M * 2 + wm, bc * WARPS_N * 2 + wn});
    store(g.c, cB, {0, 0, br * WARPS_M * 2 + wm, bc * WARPS_N * 2 + WARPS_N + wn});
    store(g.c, cC, {0, 0, br * WARPS_M * 2 + WARPS_M + wm, bc * WARPS_N * 2 + wn});
    store(g.c, cD, {0, 0, br * WARPS_M * 2 + WARPS_M + wm, bc * WARPS_N * 2 + WARPS_N + wn});
}

__host__ inline bool crr_can_use_exact_8wave(const layout_globals& g) {
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM;
}

__host__ inline void dispatch_crr_exact_8wave(const layout_globals& g) {
    crr_exact_8wave_globals eg{
        make_gl<CRR_exact_a_gl>(reinterpret_cast<uint64_t>(g.a.raw_ptr), 1, 1, g.k, g.m),
        make_gl<CRR_exact_b_gl>(reinterpret_cast<uint64_t>(g.b.raw_ptr), 1, 1, g.k, g.n),
        make_gl<CRR_exact_c_gl>(reinterpret_cast<uint64_t>(g.c.raw_ptr), 1, 1, g.m, g.n),
        g.scale_a, g.scale_b, g.group_m,
        g.stream,
    };
    crr_exact_8wave_kernel<<<eg.grid(), eg.block(), 0, eg.stream>>>(eg);
}

#endif
