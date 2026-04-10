#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE       = 128;
constexpr int HALF_BLOCK_SIZE  = BLOCK_SIZE / 2;
constexpr int K_STEP           = 64;
constexpr int WARPS_M          = 2;
constexpr int WARPS_N          = 2;
constexpr int REG_BLOCK_M      = BLOCK_SIZE / WARPS_M;
constexpr int REG_BLOCK_N      = BLOCK_SIZE / WARPS_N;
constexpr int HALF_REG_BLOCK_M = REG_BLOCK_M / 2;
constexpr int HALF_REG_BLOCK_N = REG_BLOCK_N / 2;

#define NUM_WARPS (WARPS_M * WARPS_N)
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl = gl<bf16, -1, -1, -1, -1>;
using G = kittens::group<NUM_WARPS>;

struct layout_globals {
    _gl a, b, c;
    hipStream_t stream;
    int m, n, k, ki, bpr, bpc, group_m;
    dim3 block() { return dim3(NUM_THREADS); }
};

using ST_A = st_bf<HALF_BLOCK_SIZE, K_STEP, st_16x32_s>;
using ST_B = st_bf<HALF_BLOCK_SIZE, K_STEP, st_16x32_s>;
using A_reg_t = rt_bf<HALF_REG_BLOCK_M, K_STEP, row_l, rt_16x32_s>;
using B_reg_t = rt_bf<HALF_REG_BLOCK_N, K_STEP, row_l, rt_16x32_s>;
using C_reg_t = rt_fl<HALF_REG_BLOCK_M, HALF_REG_BLOCK_N, col_l, rt_16x16_s>;

__global__ __launch_bounds__(NUM_THREADS, 2)
void gemm_rcr_128(const layout_globals g) {
    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    A_reg_t A_tile;
    B_reg_t B_tile_0, B_tile_1;
    C_reg_t C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);

    int wgid = blockIdx.x;
    const int NUM_WGS = gridDim.x;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, 64);
    const int WGM = g.group_m;
    const int num_wgid_in_group = WGM * g.bpc;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(g.bpr - first_pid_m, WGM);
    if (group_size_m <= 0) return;
    int row = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int col = (wgid % num_wgid_in_group) / group_size_m;
    if (row >= g.bpr || col >= g.bpc) return;

    const int warp_row = warpid() / WARPS_N;
    const int warp_col = warpid() % WARPS_N;
    const int num_tiles = g.ki;

    const bf16* a_base = (bf16*)&g.a[{0, 0, 0, 0}];
    const bf16* b_base = (bf16*)&g.b[{0, 0, 0, 0}];
    const int a_row_stride = g.a.template stride<2>() * sizeof(bf16);
    const int b_row_stride = g.b.template stride<2>() * sizeof(bf16);
    i32x4 a_srsrc = make_srsrc(a_base, g.m * a_row_stride, a_row_stride);
    i32x4 b_srsrc = make_srsrc(b_base, g.n * b_row_stride, b_row_stride);

    constexpr int elem_per_warp = (16 / sizeof(bf16)) * kittens::WARP_THREADS;
    constexpr uint32_t A_TILE_LDS = sizeof(ST_A);
    constexpr uint32_t B_TILE_LDS = sizeof(ST_B);
    const int wid = warpid() % NUM_WARPS;
    uint32_t a_lds = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    uint32_t b_lds = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&Bs[0][0].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    const uint32_t a_lds_00 = a_lds;
    const uint32_t a_lds_01 = a_lds + A_TILE_LDS;
    const uint32_t a_lds_10 = a_lds + 2 * A_TILE_LDS;
    const uint32_t a_lds_11 = a_lds + 3 * A_TILE_LDS;
    const uint32_t b_lds_00 = b_lds;
    const uint32_t b_lds_01 = b_lds + B_TILE_LDS;
    const uint32_t b_lds_10 = b_lds + 2 * B_TILE_LDS;
    const uint32_t b_lds_11 = b_lds + 3 * B_TILE_LDS;

    auto a_co = [&](int s, int k) -> coord<ST_A> { return {0, 0, s, k}; };
    auto b_co = [&](int s, int k) -> coord<ST_B> { return {0, 0, s, k}; };

    using T = typename st_bf<BLOCK_SIZE, K_STEP, st_32x16_s>::dtype;
    constexpr int bytes_per_thread = st_32x16_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = HALF_BLOCK_SIZE * K_STEP * sizeof(T) / bytes_per_memcpy;
    uint32_t soA[memcpy_per_tile];
    uint32_t soB[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    auto load_a = [&](A_reg_t& dst, ST_A& tile, int wi) {
        auto sub = subtile_inplace<HALF_REG_BLOCK_M, K_STEP>(tile, {wi, 0});
        load(dst, sub);
    };
    auto load_b = [&](B_reg_t& dst, ST_B& tile, int wi) {
        auto sub = subtile_inplace<HALF_REG_BLOCK_N, K_STEP>(tile, {wi, 0});
        load(dst, sub);
    };

    int tic = 0, toc = 1;
    G::load(Bs[tic][0], g.b, b_co(col*2, 0), soB, b_srsrc, b_base, b_lds_00);
    G::load(As[tic][0], g.a, a_co(row*2, 0), soA, a_srsrc, a_base, a_lds_00);
    G::load(Bs[tic][1], g.b, b_co(col*2+1, 0), soB, b_srsrc, b_base, b_lds_01);
    G::load(As[tic][1], g.a, a_co(row*2+1, 0), soA, a_srsrc, a_base, a_lds_01);

    if (warp_row == 1) __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(Bs[toc][0], g.b, b_co(col*2, 1), soB, b_srsrc, b_base, b_lds_10);
    G::load(As[toc][0], g.a, a_co(row*2, 1), soA, a_srsrc, a_base, a_lds_10);
    G::load(Bs[toc][1], g.b, b_co(col*2+1, 1), soB, b_srsrc, b_base, b_lds_11);

    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    auto main_loop_iter = [&](int tile) {
        load_b(B_tile_0, Bs[0][0], warp_col);
        load_a(A_tile, As[0][0], warp_row);
        G::load(As[1][1], g.a, a_co(row*2+1, tile+1), soA, a_srsrc, a_base, a_lds_11);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1); mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]); __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

        load_b(B_tile_1, Bs[0][1], warp_col);
        G::load(Bs[0][0], g.b, b_co(col*2, tile+2), soB, b_srsrc, b_base, b_lds_00);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1); mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]); __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(A_tile, As[0][1], warp_row);
        G::load(As[0][0], g.a, a_co(row*2, tile+2), soA, a_srsrc, a_base, a_lds_00);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1); mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]); __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

        load_b(B_tile_0, Bs[1][0], warp_col);
        G::load(Bs[0][1], g.b, b_co(col*2+1, tile+2), soB, b_srsrc, b_base, b_lds_01);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1); mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]); __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(A_tile, As[1][0], warp_row);
        G::load(As[0][1], g.a, a_co(row*2+1, tile+2), soA, a_srsrc, a_base, a_lds_01);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1); mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]); __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

        load_b(B_tile_1, Bs[1][1], warp_col);
        G::load(Bs[1][0], g.b, b_co(col*2, tile+3), soB, b_srsrc, b_base, b_lds_10);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1); mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]); __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(A_tile, As[1][1], warp_row);
        G::load(As[1][0], g.a, a_co(row*2, tile+3), soA, a_srsrc, a_base, a_lds_10);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1); mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]); __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[1][1], g.b, b_co(col*2+1, tile+3), soB, b_srsrc, b_base, b_lds_11);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1); mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]); __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    };

    #pragma unroll 2
    for (int tile = 0; tile < num_tiles - 2; tile += 2) {
        main_loop_iter(tile);
    }

    // Epilogue 1: penultimate
    {
        const int tile = num_tiles - 2;
        load_b(B_tile_0, Bs[tic][0], warp_col);
        load_a(A_tile, As[tic][0], warp_row);
        G::load(As[toc][1], g.a, a_co(row*2+1, tile+1), soA, a_srsrc, a_base, a_lds_11);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1); mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]); __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_b(B_tile_1, Bs[tic][1], warp_col);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1); mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]); __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(A_tile, As[tic][1], warp_row);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        tic ^= 1; toc ^= 1;
    }

    // Epilogue 2: last
    {
        load_b(B_tile_0, Bs[tic][0], warp_col);
        load_a(A_tile, As[tic][0], warp_row);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1); mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]); __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_b(B_tile_1, Bs[tic][1], warp_col);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1); mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]); __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        load_a(A_tile, As[tic][1], warp_row);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (warp_row == 0) __builtin_amdgcn_s_barrier();
    store(g.c, C_accum[0][0], {0, 0, row*2*WARPS_M+warp_row, col*2*WARPS_N+warp_col});
    store(g.c, C_accum[0][1], {0, 0, row*2*WARPS_M+warp_row, col*2*WARPS_N+WARPS_N+warp_col});
    store(g.c, C_accum[1][0], {0, 0, row*2*WARPS_M+WARPS_M+warp_row, col*2*WARPS_N+warp_col});
    store(g.c, C_accum[1][1], {0, 0, row*2*WARPS_M+WARPS_M+warp_row, col*2*WARPS_N+WARPS_N+warp_col});
}

static void dispatch_rcr(layout_globals g) {
    g.m = g.c.rows(); g.n = g.c.cols(); g.k = g.a.cols();
    g.ki = g.k / K_STEP; g.bpr = g.m / BLOCK_SIZE; g.bpc = g.n / BLOCK_SIZE;
    gemm_rcr_128<<<dim3(g.bpr * g.bpc), g.block(), 0, g.stream>>>(g);
}

static float to_float(pybind11::object obj) {
    if (pybind11::hasattr(obj, "item")) return obj.attr("item")().cast<float>();
    return obj.cast<float>();
}

static void gemm_rcr_py(pybind11::object a, pybind11::object b, pybind11::object c, int group_m) {
    layout_globals g{
        py::from_object<_gl>::make(a), py::from_object<_gl>::make(b),
        py::from_object<_gl>::make(c), {}, 0, 0, 0, 0, 0, 0, group_m,
    };
    dispatch_rcr(g);
}

PYBIND11_MODULE(tk_bf16_128, m) {
    using namespace pybind11::literals;
    m.def("gemm_rcr", &gemm_rcr_py, "a"_a, "b"_a, "c"_a, "group_m"_a = 4);
    m.attr("BLOCK_SIZE") = BLOCK_SIZE;
    m.attr("K_STEP") = K_STEP;
}
