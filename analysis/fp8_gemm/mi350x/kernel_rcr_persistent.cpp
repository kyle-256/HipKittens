#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define TK_STRINGIFY_IMPL(x) #x
#define TK_STRINGIFY(x) TK_STRINGIFY_IMPL(x)
#define TK_WAIT_LGKM(x) asm volatile("s_waitcnt lgkmcnt(" TK_STRINGIFY(x) ")")
#define TK_WAIT_VMCNT(x) asm volatile("s_waitcnt vmcnt(" TK_STRINGIFY(x) ")")

#ifndef GEMM_BLOCK_SIZE
#define GEMM_BLOCK_SIZE 256
#endif
#ifndef GEMM_K_BLOCK
#define GEMM_K_BLOCK 128
#endif
#ifndef GEMM_MIN_BLOCKS_PER_CU
#define GEMM_MIN_BLOCKS_PER_CU 2
#endif
#ifndef GEMM_BLOCK_SWIZZLE
#define GEMM_BLOCK_SWIZZLE 0
#endif
#ifndef GEMM_BLOCK_SWIZZLE_NUM_XCDS
#define GEMM_BLOCK_SWIZZLE_NUM_XCDS 8
#endif

constexpr int BLK = GEMM_BLOCK_SIZE, BK = GEMM_K_BLOCK;
constexpr int WARPS_M = 2, WARPS_N = 4;
constexpr int _NUM_THREADS = WARPS_M * WARPS_N * kittens::WARP_THREADS;
constexpr int RBM = BLK / (2 * WARPS_M);
constexpr int RBN = BLK / (2 * WARPS_N);

using _gl_fp8  = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;
using G = group<WARPS_M * WARPS_N>;
using ST = st_fp8e4m3<BLK/2, BK, st_16x128_v2_s>;
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l>;

__device__ __forceinline__ int gemm_chiplet_swizzle_bid(int bid, int num_wgs) {
#if GEMM_BLOCK_SWIZZLE
    constexpr int NUM_XCDS = GEMM_BLOCK_SWIZZLE_NUM_XCDS;
    if (num_wgs >= NUM_XCDS && (num_wgs % NUM_XCDS) == 0)
        return (bid % NUM_XCDS) * (num_wgs / NUM_XCDS) + (bid / NUM_XCDS);
#endif
    return bid;
}

__device__ __forceinline__ void gemm_compute_block_coords(
    int bid, int bpr, int bpc, int group_m, int &br, int &bc) {
#if GEMM_BLOCK_SWIZZLE
    bid = gemm_chiplet_swizzle_bid(bid, bpr * bpc);
    const int num_wgid_in_group = group_m * bpc;
    const int group_id = bid / num_wgid_in_group;
    const int first_pid_m = group_id * group_m;
    const int gsm = (first_pid_m + group_m <= bpr) ? group_m : (bpr - first_pid_m);
    if (gsm <= 0) { br = bpr; bc = bpc; return; }
    br = first_pid_m + ((bid % num_wgid_in_group) % gsm);
    bc = (bid % num_wgid_in_group) / gsm;
#else
    br = bid / bpc;
    bc = bid % bpc;
#endif
}

struct persistent_globals {
    _gl_fp8 a, b;
    _gl_bf16 c;
    float scale_a, scale_b;
    int bpr, bpc, ki, group_m;
    int total_tiles;
    int* tile_counter;
    hipStream_t stream;
};

__global__ __launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU)
void persistent_rcr_kernel(const persistent_globals g) {
    __shared__ ST As[2][2];
    __shared__ ST Bs[2][2];

    constexpr int bpt = ST::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt = ST::rows * ST::cols * sizeof(fp8e4m3) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    const int wm = warpid() / WARPS_N;
    const int wn = warpid() % WARPS_N;

    auto a_co = [&](int s, int k) -> coord<ST> { return {0, 0, s, k}; };
    auto b_co = [&](int s, int k) -> coord<ST> { return {0, 0, s, k}; };

    auto load_a = [&](A_row_reg& dst, ST& tile, int wi) {
        auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
        load(dst, sub);
    };
    auto load_b = [&](B_row_reg& dst, ST& tile, int wi) {
        auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
        load(dst, sub);
    };

    while (true) {
        __shared__ int s_tile_id;
        if (threadIdx.x == 0)
            s_tile_id = atomicAdd(g.tile_counter, 1);
        __builtin_amdgcn_s_barrier();
        const int tile_id = s_tile_id;
        if (tile_id >= g.total_tiles) return;

        int br, bc;
        gemm_compute_block_coords(tile_id, g.bpr, g.bpc, g.group_m, br, bc);

        A_row_reg a;
        B_row_reg b0, b1;
        rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;
        zero(cA); zero(cB); zero(cC); zero(cD);

        int tic = 0, toc = 1;
        G::load(Bs[tic][0], g.b, b_co(bc*2,   0), soB);
        G::load(As[tic][0], g.a, a_co(br*2,   0), soA);
        G::load(Bs[tic][1], g.b, b_co(bc*2+1, 0), soB);
        G::load(As[tic][1], g.a, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(4);
        __builtin_amdgcn_s_barrier();

        G::load(Bs[toc][0], g.b, b_co(bc*2,   1), soB);
        G::load(As[toc][0], g.a, a_co(br*2,   1), soA);
        G::load(Bs[toc][1], g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(6);
        __builtin_amdgcn_s_barrier();

        #pragma unroll 2
        for (int k = 0; k < g.ki - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(4);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_b(b1, Bs[tic][1], wn);
            G::load(Bs[tic][0], g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            G::load(As[tic][0], g.a, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            G::load(Bs[tic][1], g.b, b_co(bc*2+1, k+2), soB);
            TK_WAIT_VMCNT(8);
            __builtin_amdgcn_s_barrier();

            __builtin_amdgcn_s_setprio(1); mma_ABt(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // Penultimate K-iteration
        {
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            G::load(As[toc][1], g.a, a_co(br*2+1, g.ki-1), soA);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(4);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cC, a, b0, cC); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, Bs[toc][0], wn);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cD, a, b1, cD); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            tic ^= 1; toc ^= 1;
        }

        // Last K-iteration
        {
            load_a(a, As[tic][0], wm);
            TK_WAIT_VMCNT(0);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cA, a, b0, cA); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, Bs[tic][1], wn);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1); mma_ABt(cB, a, b1, cB); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(cC, a, b0, cC);
            mma_ABt(cD, a, b1, cD);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        const float sc = g.scale_a * g.scale_b;
        mul(cA, cA, sc); mul(cB, cB, sc); mul(cC, cC, sc); mul(cD, cD, sc);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        store(g.c, cA, {0, 0, br*WARPS_M*2+wm, bc*WARPS_N*2+wn});
        store(g.c, cB, {0, 0, br*WARPS_M*2+wm, bc*WARPS_N*2+WARPS_N+wn});
        store(g.c, cC, {0, 0, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+wn});
        store(g.c, cD, {0, 0, br*WARPS_M*2+WARPS_M+wm, bc*WARPS_N*2+WARPS_N+wn});

        __builtin_amdgcn_s_barrier();
    }
}

static int* d_tile_counter = nullptr;

static float to_float(pybind11::object obj) {
    if (pybind11::hasattr(obj, "item")) return obj.attr("item")().cast<float>();
    return obj.cast<float>();
}

static void gemm_rcr(pybind11::object a, pybind11::object b, pybind11::object c,
                      pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                      int group_m) {
    auto ga = py::from_object<_gl_fp8>::make(a);
    auto gb = py::from_object<_gl_fp8>::make(b);
    auto gc = py::from_object<_gl_bf16>::make(c);

    int M = gc.rows(), N = gc.cols(), K = ga.cols();
    int bpr = M / BLK, bpc = N / BLK, ki = K / BK;

    if (!d_tile_counter) hipMalloc(&d_tile_counter, sizeof(int));
    hipMemsetAsync(d_tile_counter, 0, sizeof(int), 0);

    persistent_globals g{
        ga, gb, gc,
        to_float(scale_a_obj), to_float(scale_b_obj),
        bpr, bpc, ki, group_m,
        bpr * bpc, d_tile_counter, {},
    };

    constexpr int NUM_CUS = 304;
    int grid_size = min(NUM_CUS, g.total_tiles);
    persistent_rcr_kernel<<<grid_size, _NUM_THREADS, 0, g.stream>>>(g);
}

PYBIND11_MODULE(tk_fp8_layouts, m) {
    m.def("gemm_rcr", &gemm_rcr,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = 4);
}
