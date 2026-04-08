#pragma once

#ifndef CRR_USE_EXACT_8WAVE_DOUBLE_PUMP_FASTPATH
#define CRR_USE_EXACT_8WAVE_DOUBLE_PUMP_FASTPATH 0
#endif

#if CRR_USE_EXACT_8WAVE_DOUBLE_PUMP_FASTPATH

namespace crr_exact_8wave_double_pump {

using exact_a_gl = gl<fp8e4m3, 1, 1, K_DIM, M_DIM>;
using exact_b_gl = gl<fp8e4m3, 1, 1, K_DIM, N_DIM>;
using exact_c_gl = gl<bf16, 1, 1, M_DIM, N_DIM>;

struct globals {
    exact_a_gl a;
    exact_b_gl b;
    exact_c_gl c;
    hipStream_t stream;

    dim3 grid() { return dim3((M_DIM / BLK) * (N_DIM / BLK)); }
    dim3 block() { return dim3(_NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

static_assert(BLK == 256, "Exact 8-wave double-pump fast path requires BLK=256");
static_assert(BK == 128, "Exact 8-wave double-pump fast path requires BK=128");
static_assert(WARPS_M == 2, "Exact 8-wave double-pump fast path requires WARPS_M=2");
static_assert(WARPS_N == 4, "Exact 8-wave double-pump fast path requires WARPS_N=4");
static_assert(!CRR_ROW_SHARED_TRANSPOSE, "Exact CRR fast path only supports strict shared layout");
static_assert(!CRR_USE_V3_SWIZZLE, "Exact CRR fast path is wired for v2/v2a shared layouts");
static_assert(!CRR_A_LDS_REENCODE, "Exact CRR fast path does not support A LDS reencode");
static_assert(CRR_A_REG_ROW_LOAD_TRANSPOSE == 1, "Exact CRR fast path expects default A row-load transpose");
static_assert(CRR_A_REG_ROW_LOAD_ALIAS == 1, "Exact CRR fast path expects default A row-load alias");
static_assert(CRR_B_REG_ROW_LOAD_TRANSPOSE == 0, "Exact CRR fast path expects default B col-load path");
static_assert(CRR_B_REG_ROW_LOAD_ALIAS == 0, "Exact CRR fast path expects default B col-load path");
static_assert(CRR_BATCHED_PAIR_MMA == 1, "Exact CRR fast path expects batched pair MMA");
static_assert(CRR_BATCHED_EPILOGUE_MMA == 0, "Exact CRR fast path expects scalar epilogue MMA");

__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void kernel(const globals g) {
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

    const int bid = blockIdx.x;
    const int br = bid / blocks_per_col;
    const int bc = bid % blocks_per_col;
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

    for (int k = 0; k < k_iters - 2; k += 2) {
        load_b(b0, Bs[0][0], wn);
        load_a(a, As[0][0], wm);
        global_load_a(As[1][1], br * 2 + 1, k + 1);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        crr_mma(cA, a, b0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        CRR_SCHED_BARRIER();

        load_b(b1, Bs[0][1], wn);
        global_load_b(Bs[0][0], bc * 2, k + 2);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        crr_mma(cB, a, b1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[0][1], wm);
        global_load_a(As[0][0], br * 2, k + 2);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        crr_mma(cC, a, b0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        CRR_SCHED_BARRIER();

        load_b(b0, Bs[1][0], wn);
        global_load_b(Bs[0][1], bc * 2 + 1, k + 2);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        crr_mma(cD, a, b1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[1][0], wm);
        global_load_a(As[0][1], br * 2 + 1, k + 2);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        crr_mma(cA, a, b0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        CRR_SCHED_BARRIER();

        load_b(b1, Bs[1][1], wn);
        global_load_b(Bs[1][0], bc * 2, k + 3);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        crr_mma(cB, a, b1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[1][1], wm);
        global_load_a(As[1][0], br * 2, k + 3);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        crr_mma(cC, a, b0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        CRR_SCHED_BARRIER();

        global_load_b(Bs[1][1], bc * 2 + 1, k + 3);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        crr_mma(cD, a, b1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    {
        constexpr int k = k_iters - 2;

        load_b(b0, Bs[tic][0], wn);
        const auto b0_keep = b0;
        load_a(a, As[tic][0], wm);
        global_load_a(As[toc][1], br * 2 + 1, k + 1);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_s_setprio(1);
        crr_mma(cA, a, b0_keep);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_b(b1, Bs[tic][1], wn);
        const auto b1_keep = b1;
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        crr_mma(cB, a, b1_keep);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        crr_mma(cC, a, b0_keep);
        crr_mma(cD, a, b1_keep);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        tic ^= 1;
        toc ^= 1;
    }

    {
        load_b(b0, Bs[tic][0], wn);
        const auto b0_keep = b0;
        load_a(a, As[tic][0], wm);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        crr_mma(cA, a, b0_keep);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_b(b1, Bs[tic][1], wn);
        const auto b1_keep = b1;
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        crr_mma(cB, a, b1_keep);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        crr_mma(cC, a, b0_keep);
        crr_mma(cD, a, b1_keep);
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

} // namespace crr_exact_8wave_double_pump

__host__ inline bool crr_can_use_exact_8wave_double_pump(const layout_globals& g) {
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM && g.scale_a * g.scale_b == 1.0f;
}

__host__ inline void dispatch_crr_exact_8wave_double_pump(const layout_globals& g) {
    crr_exact_8wave_double_pump::globals eg{
        make_gl<crr_exact_8wave_double_pump::exact_a_gl>(
            reinterpret_cast<uint64_t>(g.a.raw_ptr), 1, 1, g.k, g.m),
        make_gl<crr_exact_8wave_double_pump::exact_b_gl>(
            reinterpret_cast<uint64_t>(g.b.raw_ptr), 1, 1, g.k, g.n),
        make_gl<crr_exact_8wave_double_pump::exact_c_gl>(
            reinterpret_cast<uint64_t>(g.c.raw_ptr), 1, 1, g.m, g.n),
        g.stream,
    };
    crr_exact_8wave_double_pump::kernel<<<eg.grid(), eg.block(), 0, eg.stream>>>(eg);
}

#endif
