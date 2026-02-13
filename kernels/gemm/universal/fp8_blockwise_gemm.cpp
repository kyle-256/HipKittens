/**
 * FP8 Block-wise Scaled GEMM kernel using HipKittens.
 *
 * RCR layout: C[M,N] = sum_k (scale_a[m_blk, k_blk] * scale_b[n_blk, k_blk] * A_blk @ B_blk^T)
 *
 * A: fp8e4m3 [M,K] row-major
 * B: fp8e4m3 [N,K] row-major
 * C: bf16    [M,N] row-major
 * scale_a: float [ceil(M/128), ceil(K/128)]
 * scale_b: float [ceil(N/128), ceil(K/128)]
 *
 * Uses mma_ABt with fp8e4m3 operands.
 * 256x256 output tile, K_STEP=128, 8 warps (2x4).
 */

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLK       = 256;
constexpr int HALF_BLK  = BLK / 2;
constexpr int K_STEP    = 128;
constexpr int SCALE_BLK = 128;
constexpr int WM        = 2;
constexpr int WN        = 4;
constexpr int NUM_W     = WM * WN;
constexpr int NUM_T     = kittens::WARP_THREADS * NUM_W;

constexpr int RBM       = BLK / WM / 2;       // 64
constexpr int RBN       = BLK / WN / 2;       // 32

using G8 = kittens::group<NUM_W>;

using _gl_fp8 = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_out = gl<bf16, -1, -1, -1, -1>;
using _gl_fl  = gl<float, -1, -1, -1, -1>;

struct fp8_bw_globals {
    _gl_fp8 a, b;
    _gl_out c;
    _gl_fl scale_a, scale_b;
    hipStream_t stream;
    int M = a.rows();
    int N = c.cols();
    int K = a.cols();
    dim3 grid()  { return dim3(ceil_div(N, BLK) * ceil_div(M, BLK)); }
    dim3 block() { return dim3(NUM_T); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_T, 2)
void fp8_blockwise_kernel(const fp8_bw_globals g, int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_fp8e4m3<HALF_BLK, K_STEP, st_16x128_s>;
    using ST_B = st_fp8e4m3<HALF_BLK, K_STEP, st_16x128_s>;
    ST_A (&As)[2] = al.allocate<ST_A, 2>();  // Only need 2 A tiles (upper/lower half)
    ST_B (&Bs)[2] = al.allocate<ST_B, 2>();  // Only need 2 B tiles (upper/lower half)

    rt_fp8e4m3<RBM, K_STEP> A_tile;
    rt_fp8e4m3<RBN, K_STEP> B_tile_0;
    rt_fp8e4m3<RBN, K_STEP> B_tile_1;

    // Main accumulators
    rt_fl<RBM, RBN, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);

    // Block ID with XCD swizzle
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    const int WGM = 8;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, 64);
    const int num_pid_m = ceil_div(M, BLK);
    const int num_pid_n = ceil_div(N, BLK);
    const int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    int row = pid_m;
    int col = pid_n;

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    const int num_tiles = K / K_STEP;

    // Swizzled offsets
    using T = typename ST_A::dtype;
    constexpr int bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_T;
    constexpr int mpt = HALF_BLK * K_STEP * sizeof(T) / bpm;
    uint32_t so_a[mpt], so_b[mpt];
    G8::prefill_swizzled_offsets(As[0], g.a, so_a);
    G8::prefill_swizzled_offsets(Bs[0], g.b, so_b);

    int scale_cols = g.scale_a.cols();

    // Non-pipelined loop: load, compute, scale, accumulate for each K tile
    for (int k_tile = 0; k_tile < num_tiles; k_tile++) {
        // Load A and B tiles from global to shared
        G8::load(As[0], g.a, {0, 0, row*2, k_tile}, so_a);
        G8::load(As[1], g.a, {0, 0, row*2 + 1, k_tile}, so_a);
        G8::load(Bs[0], g.b, {0, 0, col*2, k_tile}, so_b);
        G8::load(Bs[1], g.b, {0, 0, col*2 + 1, k_tile}, so_b);

        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        // Read block-wise scales using gl operator[]
        float sa_0 = g.scale_a[{0, 0, row*2, k_tile}];
        float sa_1 = g.scale_a[{0, 0, row*2 + 1, k_tile}];
        float sb_0 = g.scale_b[{0, 0, col*2, k_tile}];
        float sb_1 = g.scale_b[{0, 0, col*2 + 1, k_tile}];

        // Load subtiles to registers
        auto sta0 = subtile_inplace<RBM, K_STEP>(As[0], {warp_row, 0});
        load(A_tile, sta0);
        auto stb0 = subtile_inplace<RBN, K_STEP>(Bs[0], {warp_col, 0});
        load(B_tile_0, stb0);
        auto stb1 = subtile_inplace<RBN, K_STEP>(Bs[1], {warp_col, 0});
        load(B_tile_1, stb1);

        asm volatile("s_waitcnt lgkmcnt(0)");

        // Compute and accumulate for first M-half (m_blk = row*2)
        // Use mma_ABt accumulating directly into a temporary
        rt_fl<RBM, RBN, col_l, rt_16x16_s> tmp00, tmp01;
        zero(tmp00); zero(tmp01);
        mma_ABt(tmp00, A_tile, B_tile_0, tmp00);
        mma_ABt(tmp01, A_tile, B_tile_1, tmp01);

        // Scale and accumulate: C_accum += scale * tmp
        float s00 = sa_0 * sb_0;
        float s01 = sa_0 * sb_1;
        mul(tmp00, tmp00, s00);
        mul(tmp01, tmp01, s01);
        add(C_accum[0][0], C_accum[0][0], tmp00);
        add(C_accum[0][1], C_accum[0][1], tmp01);

        // Second M-half (m_blk = row*2 + 1)
        auto sta1 = subtile_inplace<RBM, K_STEP>(As[1], {warp_row, 0});
        load(A_tile, sta1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        rt_fl<RBM, RBN, col_l, rt_16x16_s> tmp10, tmp11;
        zero(tmp10); zero(tmp11);
        mma_ABt(tmp10, A_tile, B_tile_0, tmp10);
        mma_ABt(tmp11, A_tile, B_tile_1, tmp11);

        float s10 = sa_1 * sb_0;
        float s11 = sa_1 * sb_1;
        mul(tmp10, tmp10, s10);
        mul(tmp11, tmp11, s11);
        add(C_accum[1][0], C_accum[1][0], tmp10);
        add(C_accum[1][1], C_accum[1][1], tmp11);

        __builtin_amdgcn_s_barrier();
    }

    // Store C[M,N]
    store(g.c, C_accum[0][0], {0, 0, (row * 2) * WM + warp_row, col * 2 * WN + warp_col});
    store(g.c, C_accum[0][1], {0, 0, (row * 2) * WM + warp_row, col * 2 * WN + WN + warp_col});
    store(g.c, C_accum[1][0], {0, 0, (row * 2) * WM + WM + warp_row, col * 2 * WN + warp_col});
    store(g.c, C_accum[1][1], {0, 0, (row * 2) * WM + WM + warp_row, col * 2 * WN + WN + warp_col});
}

void dispatch_fp8_blockwise(fp8_bw_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)fp8_blockwise_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    fp8_blockwise_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M, g.N, g.K);
}

PYBIND11_MODULE(fp8_blockwise_gemm, m) {
    m.doc() = "FP8 block-wise scaled GEMM kernel (RCR layout)";

    py::bind_function<dispatch_fp8_blockwise>(m, "rcr",
        &fp8_bw_globals::a, &fp8_bw_globals::b, &fp8_bw_globals::c,
        &fp8_bw_globals::scale_a, &fp8_bw_globals::scale_b);
}
