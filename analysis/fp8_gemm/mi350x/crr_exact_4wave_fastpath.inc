#pragma once

#ifndef CRR_USE_EXACT_4WAVE_FASTPATH
#define CRR_USE_EXACT_4WAVE_FASTPATH 0
#endif

#ifndef CRR_4WAVE_PREFETCH_LGKM
#define CRR_4WAVE_PREFETCH_LGKM CRR_PREFETCH_LGKM
#endif

#ifndef CRR_4WAVE_INIT0_VMCNT
#define CRR_4WAVE_INIT0_VMCNT CRR_INIT0_VMCNT
#endif

#ifndef CRR_4WAVE_INIT1_VMCNT
#define CRR_4WAVE_INIT1_VMCNT CRR_INIT1_VMCNT
#endif

#ifndef CRR_4WAVE_STEADY_VMCNT
#define CRR_4WAVE_STEADY_VMCNT CRR_STEADY_VMCNT
#endif

#ifndef CRR_4WAVE_EPILOGUE_VMCNT
#define CRR_4WAVE_EPILOGUE_VMCNT CRR_EPILOGUE_VMCNT
#endif

#if CRR_USE_EXACT_4WAVE_FASTPATH

namespace crr_exact_4wave {

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
constexpr int RBM4 = BLK / WARPS_M / 2;
constexpr int RBN4 = BLK / WARPS_N / 2;

using G4 = kittens::group<NUM_WARPS>;
using exact_a_gl = gl<fp8e4m3, 1, 1, K_DIM, M_DIM>;
using exact_b_gl = gl<fp8e4m3, 1, 1, K_DIM, N_DIM>;
using exact_c_gl = gl<bf16, 1, 1, M_DIM, N_DIM>;

using ST_A = ST_v2a;
using ST_B = ST_v2;
using A_reg = A_col_reg;
using B_reg = rt_fp8e4m3<BK, RBN4, col_l, rt_128x16_s>;
using Acc = rt_fl<RBM4, RBN4, col_l, rt_16x16_s>;

struct globals {
    exact_a_gl a;
    exact_b_gl b;
    exact_c_gl c;
    hipStream_t stream;

    dim3 grid() { return dim3((M_DIM / BLK) * (N_DIM / BLK)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

static_assert(BLK == 256, "Exact 4-wave fast path requires BLK=256");
static_assert(BK == 128, "Exact 4-wave fast path requires BK=128");
static_assert(!CRR_ROW_SHARED_TRANSPOSE, "Exact CRR fast path only supports strict shared layout");
static_assert(!CRR_USE_V3_SWIZZLE, "Exact CRR 4-wave fast path is wired for v2/v2a shared layouts");
static_assert(!CRR_A_LDS_REENCODE, "Exact CRR 4-wave fast path does not support A LDS reencode");
static_assert(CRR_A_REG_ROW_LOAD_TRANSPOSE == 1, "Exact CRR 4-wave fast path expects default A row-load transpose");
static_assert(CRR_A_REG_ROW_LOAD_ALIAS == 1, "Exact CRR 4-wave fast path expects default A row-load alias");
static_assert(CRR_B_REG_ROW_LOAD_TRANSPOSE == 0, "Exact CRR 4-wave fast path expects default B col-load path");
static_assert(CRR_B_REG_ROW_LOAD_ALIAS == 0, "Exact CRR 4-wave fast path expects default B col-load path");
static_assert(RBM4 == RBM, "Exact CRR 4-wave fast path assumes RBM matches current default");
static_assert(RBN4 == 64, "Exact CRR 4-wave fast path expects 64-column per-warp B tile");

__device__ __forceinline__ void mma_one(
    Acc& acc,
    const A_reg& a,
    const B_reg& b,
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

__device__ __forceinline__ void mma_all(
    Acc& acc,
    const A_reg& a,
    const B_reg& b)
{
    mma_one(acc, a, b, 0, 0);
    mma_one(acc, a, b, 0, 1);
    mma_one(acc, a, b, 0, 2);
    mma_one(acc, a, b, 0, 3);
    mma_one(acc, a, b, 1, 0);
    mma_one(acc, a, b, 1, 1);
    mma_one(acc, a, b, 1, 2);
    mma_one(acc, a, b, 1, 3);
    mma_one(acc, a, b, 2, 0);
    mma_one(acc, a, b, 2, 1);
    mma_one(acc, a, b, 2, 2);
    mma_one(acc, a, b, 2, 3);
    mma_one(acc, a, b, 3, 0);
    mma_one(acc, a, b, 3, 1);
    mma_one(acc, a, b, 3, 2);
    mma_one(acc, a, b, 3, 3);
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void kernel(const globals g) {
    constexpr int blocks_per_col = N_DIM / BLK;
    constexpr int k_iters = K_DIM / BK;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    A_reg a;
    B_reg b0, b1;
    Acc cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    const int bid = blockIdx.x;
    const int br = bid / blocks_per_col;
    const int bc = bid % blocks_per_col;
    const int wm = warpid() / WARPS_N;
    const int wn = warpid() % WARPS_N;

    constexpr int bptA = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * NUM_THREADS;
    constexpr int mptA = ST_A::rows * ST_A::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
    G4::prefill_swizzled_offsets(As[0][0], g.a, soA);

    constexpr int bptB = ST_B::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * NUM_THREADS;
    constexpr int mptB = ST_B::rows * ST_B::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
    G4::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    auto a_co = [&](int s, int k) -> coord<ST_A> { return {0, 0, k, s}; };
    auto b_co = [&](int s, int k) -> coord<ST_B> { return {0, 0, k, s}; };
    auto global_load_a = [&](ST_A& tile, int s, int k) {
        G4::load(tile, g.a, a_co(s, k), soA);
    };
    auto global_load_b = [&](ST_B& tile, int s, int k) {
        G4::load(tile, g.b, b_co(s, k), soB);
    };
    auto load_a = [&](A_reg& dst, ST_A& tile, int wi) {
        load_col_from_v2a_st(dst, tile, wi * RBM4);
    };
    auto load_b = [&](B_reg& dst, ST_B& tile, int wi) {
        load_col_from_v2_st(dst, tile, wi * RBN4);
    };

    int tic = 0, toc = 1;
    global_load_b(Bs[tic][0], bc * 2, 0);
    global_load_a(As[tic][0], br * 2, 0);
    global_load_b(Bs[tic][1], bc * 2 + 1, 0);
    global_load_a(As[tic][1], br * 2 + 1, 0);

    if (wm == 1) {
        __builtin_amdgcn_s_barrier();
    }
    TK_WAIT_VMCNT(CRR_4WAVE_INIT0_VMCNT);
    __builtin_amdgcn_s_barrier();

    global_load_b(Bs[toc][0], bc * 2, 1);
    global_load_a(As[toc][0], br * 2, 1);
    global_load_b(Bs[toc][1], bc * 2 + 1, 1);

    TK_WAIT_VMCNT(CRR_4WAVE_INIT1_VMCNT);
    __builtin_amdgcn_s_barrier();

    #pragma unroll 1
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
        load_b(b0, Bs[tic][0], wn);
        load_b(b1, Bs[tic][1], wn);
        load_a(a, As[tic][0], wm);
        global_load_a(As[toc][1], br * 2 + 1, k + 1);
        TK_WAIT_LGKM(CRR_4WAVE_PREFETCH_LGKM);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_all(cA, a, b0);
        mma_all(cB, a, b1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        global_load_a(As[tic][0], br * 2, k + 2);
        global_load_b(Bs[tic][1], bc * 2 + 1, k + 2);
        TK_WAIT_VMCNT(CRR_4WAVE_STEADY_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_all(cC, a, b0);
        mma_all(cD, a, b1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        global_load_b(Bs[tic][0], bc * 2, k + 2);
    }

    {
        load_b(b0, Bs[tic][0], wn);
        const auto b0_keep = b0;
        load_a(a, As[tic][0], wm);
        global_load_a(As[toc][1], br * 2 + 1, k_iters - 1);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_all(cA, a, b0_keep);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_b(b1, Bs[tic][1], wn);
        const auto b1_keep = b1;
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_all(cB, a, b1_keep);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        TK_WAIT_VMCNT(CRR_4WAVE_EPILOGUE_VMCNT);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_all(cC, a, b0_keep);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_b(b0, Bs[toc][0], wn);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_all(cD, a, b1_keep);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        tic ^= 1;
        toc ^= 1;
    }

    {
        const auto b0_keep = b0;
        load_a(a, As[tic][0], wm);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_all(cA, a, b0_keep);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_b(b1, Bs[tic][1], wn);
        const auto b1_keep = b1;
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_all(cB, a, b1_keep);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(a, As[tic][1], wm);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_all(cC, a, b0_keep);
        mma_all(cD, a, b1_keep);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (wm == 0) {
        __builtin_amdgcn_s_barrier();
    }

    store(g.c, cA, {0, 0, br * WARPS_M * 2 + wm,           bc * WARPS_N * 2 + wn});
    store(g.c, cB, {0, 0, br * WARPS_M * 2 + wm,           bc * WARPS_N * 2 + WARPS_N + wn});
    store(g.c, cC, {0, 0, br * WARPS_M * 2 + WARPS_M + wm, bc * WARPS_N * 2 + wn});
    store(g.c, cD, {0, 0, br * WARPS_M * 2 + WARPS_M + wm, bc * WARPS_N * 2 + WARPS_N + wn});
}

} // namespace crr_exact_4wave

__host__ inline bool crr_can_use_exact_4wave(const layout_globals& g) {
    return g.m == M_DIM && g.n == N_DIM && g.k == K_DIM && g.scale_a * g.scale_b == 1.0f;
}

__host__ inline void dispatch_crr_exact_4wave(const layout_globals& g) {
    crr_exact_4wave::globals eg{
        make_gl<crr_exact_4wave::exact_a_gl>(
            reinterpret_cast<uint64_t>(g.a.raw_ptr), 1, 1, g.k, g.m),
        make_gl<crr_exact_4wave::exact_b_gl>(
            reinterpret_cast<uint64_t>(g.b.raw_ptr), 1, 1, g.k, g.n),
        make_gl<crr_exact_4wave::exact_c_gl>(
            reinterpret_cast<uint64_t>(g.c.raw_ptr), 1, 1, g.m, g.n),
        g.stream,
    };
    crr_exact_4wave::kernel<<<eg.grid(), eg.block(), 0, eg.stream>>>(eg);
}

#endif
