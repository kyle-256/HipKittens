/**
 * Grouped GEMM kernels using HipKittens (BF16 + FP8 per-tensor).
 *
 * BF16 Forward (RCR): C_g[M_g,N] = A_g[M_g,K] * B_g[N,K]^T
 * FP8  Forward (RCR): C_g[M_g,N] = scale * A_g[M_g,K] * B_g[N,K]^T
 *
 * Memory layout:
 *   A: [M_total, K] bf16/fp8 contiguous (all groups concatenated along M)
 *   B: [G*N, K] bf16/fp8 contiguous (all groups stacked: B[g*N:(g+1)*N, :])
 *   C: [M_total, N] bf16 contiguous
 *
 * Python side computes tile_map[total_blocks, 3]:
 *   tile_map[bid] = {m_tile_abs, n_tile_abs, b_row_tile_abs}
 *   where m_tile_abs = (group_m_start + tile_m * 256) / HALF_BLK
 *         n_tile_abs = tile_n * 256 / HALF_BLK ... etc.
 *
 * Requirements:
 *   - Each group's M must be a multiple of 256
 *   - N and K must be multiples of 256 and 128 respectively
 */

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// Tile configuration
constexpr int BLK       = 256;
constexpr int HALF_BLK  = BLK / 2;   // 128
constexpr int K_BF16    = 64;
constexpr int K_FP8     = 128;
constexpr int WM        = 2;
constexpr int WN        = 4;
constexpr int NUM_W     = WM * WN;
constexpr int NUM_T     = kittens::WARP_THREADS * NUM_W;

// BF16 register sub-tiles
constexpr int HRM_BF16  = BLK / WM / 2;       // 64
constexpr int HRN_BF16  = BLK / WN / 2;       // 32
// FP8 register sub-tiles
constexpr int RBM_FP8   = BLK / WM / 2;       // 64
constexpr int RBN_FP8   = BLK / WN / 2;       // 32

using G8 = kittens::group<NUM_W>;

// ============================================================
// BF16 Grouped GEMM Forward Kernel
// ============================================================

using gl_bf16 = gl<bf16, 1, 1, -1, -1>;

__global__ __launch_bounds__(NUM_T, 2)
void grouped_bf16_fwd_kernel(
    const gl_bf16 gl_a,  // [1, 1, M_total, K]
    const gl_bf16 gl_b,  // [1, 1, G*N, K]
    const gl_bf16 gl_c,  // [1, 1, M_total, N]
    const int* __restrict__ tile_m_abs,    // [total_blocks] absolute m-tile in HALF_BLK units
    const int* __restrict__ tile_n_abs,    // [total_blocks] absolute n-tile in HALF_BLK units for C
    const int* __restrict__ tile_bn_abs,   // [total_blocks] absolute n-tile in HALF_BLK units for B
    int total_blocks
) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_bf<HALF_BLK, K_BF16, st_16x32_s>;
    using ST_B = st_bf<HALF_BLK, K_BF16, st_16x32_s>;
    ST_A (&As)[2][2] = al.allocate<ST_A, 2, 2>();
    ST_B (&Bs)[2][2] = al.allocate<ST_B, 2, 2>();

    rt_bf<HRM_BF16, K_BF16, row_l, rt_16x32_s> A_tile;
    rt_bf<HRN_BF16, K_BF16, row_l, rt_16x32_s> B_tile_0;
    rt_bf<HRN_BF16, K_BF16, row_l, rt_16x32_s> B_tile_1;
    rt_fl<HRM_BF16, HRN_BF16, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);

    int bid = blockIdx.x;
    if (bid >= total_blocks) return;

    // Read tile mapping (coordinates in HALF_BLK units for gl)
    int a_row = tile_m_abs[bid];    // absolute row tile for A (and C)
    int c_col = tile_n_abs[bid];    // absolute col tile for C
    int b_row = tile_bn_abs[bid];   // absolute row tile for B

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    const int K = gl_a.cols();
    const int num_tiles = K / K_BF16;

    using T = typename ST_A::dtype;
    constexpr int bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_T;
    constexpr int mpt = HALF_BLK * K_BF16 * sizeof(T) / bpm;
    uint32_t so_a[mpt], so_b[mpt];
    G8::prefill_swizzled_offsets(As[0][0], gl_a, so_a);
    G8::prefill_swizzled_offsets(Bs[0][0], gl_b, so_b);

    int tic = 0, toc = 1;

    // Prologue
    G8::load(Bs[tic][0], gl_b, {0, 0, b_row,     0}, so_b);
    G8::load(As[tic][0], gl_a, {0, 0, a_row,     0}, so_a);
    G8::load(Bs[tic][1], gl_b, {0, 0, b_row + 1, 0}, so_b);
    G8::load(As[tic][1], gl_a, {0, 0, a_row + 1, 0}, so_a);

    if (warp_row == 1) __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G8::load(As[toc][0], gl_a, {0, 0, a_row,     1}, so_a);
    G8::load(Bs[toc][0], gl_b, {0, 0, b_row,     1}, so_b);
    G8::load(Bs[toc][1], gl_b, {0, 0, b_row + 1, 1}, so_b);
    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    // Main loop
    for (int tile = 0; tile < num_tiles - 2; tile++, tic^=1, toc^=1) {
        auto stb0 = subtile_inplace<HRN_BF16, K_BF16>(Bs[tic][0], {warp_col, 0});
        load(B_tile_0, stb0);
        auto sta0 = subtile_inplace<HRM_BF16, K_BF16>(As[tic][0], {warp_row, 0});
        load(A_tile, sta0);
        G8::load(As[toc][1], gl_a, {0, 0, a_row + 1, tile + 1}, so_a);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto stb1 = subtile_inplace<HRN_BF16, K_BF16>(Bs[tic][1], {warp_col, 0});
        load(B_tile_1, stb1);
        G8::load(As[tic][0], gl_a, {0, 0, a_row, tile + 2}, so_a);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        sta0 = subtile_inplace<HRM_BF16, K_BF16>(As[tic][1], {warp_row, 0});
        load(A_tile, sta0);
        G8::load(Bs[tic][0], gl_b, {0, 0, b_row, tile + 2}, so_b);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G8::load(Bs[tic][1], gl_b, {0, 0, b_row + 1, tile + 2}, so_b);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    // Epilogue k-2
    {
        auto stb0 = subtile_inplace<HRN_BF16, K_BF16>(Bs[tic][0], {warp_col, 0});
        load(B_tile_0, stb0);
        auto sta0 = subtile_inplace<HRM_BF16, K_BF16>(As[tic][0], {warp_row, 0});
        load(A_tile, sta0);
        G8::load(As[toc][1], gl_a, {0, 0, a_row + 1, num_tiles - 1}, so_a);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto stb1 = subtile_inplace<HRN_BF16, K_BF16>(Bs[tic][1], {warp_col, 0});
        load(B_tile_1, stb1);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        sta0 = subtile_inplace<HRM_BF16, K_BF16>(As[tic][1], {warp_row, 0});
        load(A_tile, sta0);
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
    // Epilogue k-1
    {
        auto stb0 = subtile_inplace<HRN_BF16, K_BF16>(Bs[tic][0], {warp_col, 0});
        load(B_tile_0, stb0);
        auto sta0 = subtile_inplace<HRM_BF16, K_BF16>(As[tic][0], {warp_row, 0});
        load(A_tile, sta0);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto stb1 = subtile_inplace<HRN_BF16, K_BF16>(Bs[tic][1], {warp_col, 0});
        load(B_tile_1, stb1);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        sta0 = subtile_inplace<HRM_BF16, K_BF16>(As[tic][1], {warp_row, 0});
        load(A_tile, sta0);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (warp_row == 0) __builtin_amdgcn_s_barrier();

    // Store C[M_total, N]
    // C accumulator coord in RT units: each rt is HRM_BF16=64 rows x HRN_BF16=32 cols
    // a_row is in HALF_BLK=128 units → in RT units: a_row * (128/64) = a_row * 2
    // c_col is in HALF_BLK=128 units → in RT units: c_col * (128/32) = c_col * 4
    int cr = a_row * (HALF_BLK / HRM_BF16);  // = a_row * 2
    int cc = c_col * (HALF_BLK / HRN_BF16);  // = c_col * 4
    store(gl_c, C_accum[0][0], {0, 0, cr + WM*0 + warp_row, cc + WN*0 + warp_col});
    store(gl_c, C_accum[0][1], {0, 0, cr + WM*0 + warp_row, cc + WN*1 + warp_col});
    store(gl_c, C_accum[1][0], {0, 0, cr + WM*1 + warp_row, cc + WN*0 + warp_col});
    store(gl_c, C_accum[1][1], {0, 0, cr + WM*1 + warp_row, cc + WN*1 + warp_col});
}


// ============================================================
// FP8 Grouped GEMM Forward Kernel
// ============================================================

using gl_fp8 = gl<fp8e4m3, 1, 1, -1, -1>;
using gl_bf16_out = gl<bf16, 1, 1, -1, -1>;

__global__ __launch_bounds__(NUM_T, 2)
void grouped_fp8_fwd_kernel(
    const gl_fp8 gl_a, const gl_fp8 gl_b, const gl_bf16_out gl_c,
    const int* __restrict__ tile_m_abs,
    const int* __restrict__ tile_n_abs,
    const int* __restrict__ tile_bn_abs,
    int total_blocks, float scale
) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_fp8e4m3<HALF_BLK, K_FP8, st_16x128_s>;
    using ST_B = st_fp8e4m3<HALF_BLK, K_FP8, st_16x128_s>;
    ST_A (&As)[2][2] = al.allocate<ST_A, 2, 2>();
    ST_B (&Bs)[2][2] = al.allocate<ST_B, 2, 2>();

    rt_fp8e4m3<RBM_FP8, K_FP8> A_tile;
    rt_fp8e4m3<RBN_FP8, K_FP8> B_tile_0;
    rt_fp8e4m3<RBN_FP8, K_FP8> B_tile_1;
    rt_fl<RBM_FP8, RBN_FP8, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);

    int bid = blockIdx.x;
    if (bid >= total_blocks) return;

    int a_row = tile_m_abs[bid];
    int c_col = tile_n_abs[bid];
    int b_row = tile_bn_abs[bid];

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    const int K = gl_a.cols();
    const int num_tiles = K / K_FP8;

    using T = typename ST_A::dtype;
    constexpr int bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_T;
    constexpr int mpt = HALF_BLK * K_FP8 * sizeof(T) / bpm;
    uint32_t so_a[mpt], so_b[mpt];
    G8::prefill_swizzled_offsets(As[0][0], gl_a, so_a);
    G8::prefill_swizzled_offsets(Bs[0][0], gl_b, so_b);

    int tic = 0, toc = 1;

    // Prologue
    G8::load(Bs[tic][0], gl_b, {0, 0, b_row,     0}, so_b);
    G8::load(As[tic][0], gl_a, {0, 0, a_row,     0}, so_a);
    G8::load(Bs[tic][1], gl_b, {0, 0, b_row + 1, 0}, so_b);
    G8::load(As[tic][1], gl_a, {0, 0, a_row + 1, 0}, so_a);

    if (warp_row == 1) __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G8::load(As[toc][0], gl_a, {0, 0, a_row,     1}, so_a);
    G8::load(Bs[toc][0], gl_b, {0, 0, b_row,     1}, so_b);
    G8::load(Bs[toc][1], gl_b, {0, 0, b_row + 1, 1}, so_b);
    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    // Main loop (simplified — no interleaved loads for brevity)
    for (int tile = 0; tile < num_tiles - 2; tile++, tic^=1, toc^=1) {
        auto stb0 = subtile_inplace<RBN_FP8, K_FP8>(Bs[tic][0], {warp_col, 0});
        load(B_tile_0, stb0);
        auto sta0 = subtile_inplace<RBM_FP8, K_FP8>(As[tic][0], {warp_row, 0});
        load(A_tile, sta0);
        G8::load(As[toc][1], gl_a, {0, 0, a_row + 1, tile + 1}, so_a);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto stb1 = subtile_inplace<RBN_FP8, K_FP8>(Bs[tic][1], {warp_col, 0});
        load(B_tile_1, stb1);
        G8::load(As[tic][0], gl_a, {0, 0, a_row, tile + 2}, so_a);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        sta0 = subtile_inplace<RBM_FP8, K_FP8>(As[tic][1], {warp_row, 0});
        load(A_tile, sta0);
        G8::load(Bs[tic][0], gl_b, {0, 0, b_row, tile + 2}, so_b);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G8::load(Bs[tic][1], gl_b, {0, 0, b_row + 1, tile + 2}, so_b);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    // Epilogue (same pattern as BF16)
    {
        auto stb0 = subtile_inplace<RBN_FP8, K_FP8>(Bs[tic][0], {warp_col, 0});
        load(B_tile_0, stb0);
        auto sta0 = subtile_inplace<RBM_FP8, K_FP8>(As[tic][0], {warp_row, 0});
        load(A_tile, sta0);
        G8::load(As[toc][1], gl_a, {0, 0, a_row + 1, num_tiles - 1}, so_a);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto stb1 = subtile_inplace<RBN_FP8, K_FP8>(Bs[tic][1], {warp_col, 0});
        load(B_tile_1, stb1);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        sta0 = subtile_inplace<RBM_FP8, K_FP8>(As[tic][1], {warp_row, 0});
        load(A_tile, sta0);
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
    {
        auto stb0 = subtile_inplace<RBN_FP8, K_FP8>(Bs[tic][0], {warp_col, 0});
        load(B_tile_0, stb0);
        auto sta0 = subtile_inplace<RBM_FP8, K_FP8>(As[tic][0], {warp_row, 0});
        load(A_tile, sta0);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto stb1 = subtile_inplace<RBN_FP8, K_FP8>(Bs[tic][1], {warp_col, 0});
        load(B_tile_1, stb1);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        sta0 = subtile_inplace<RBM_FP8, K_FP8>(As[tic][1], {warp_row, 0});
        load(A_tile, sta0);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (warp_row == 0) __builtin_amdgcn_s_barrier();

    // Apply scale
    mul(C_accum[0][0], C_accum[0][0], scale);
    mul(C_accum[0][1], C_accum[0][1], scale);
    mul(C_accum[1][0], C_accum[1][0], scale);
    mul(C_accum[1][1], C_accum[1][1], scale);

    // Store
    int cr = a_row * (HALF_BLK / RBM_FP8);  // = a_row * 2
    int cc = c_col * (HALF_BLK / RBN_FP8);  // = c_col * 4
    store(gl_c, C_accum[0][0], {0, 0, cr + WM*0 + warp_row, cc + WN*0 + warp_col});
    store(gl_c, C_accum[0][1], {0, 0, cr + WM*0 + warp_row, cc + WN*1 + warp_col});
    store(gl_c, C_accum[1][0], {0, 0, cr + WM*1 + warp_row, cc + WN*0 + warp_col});
    store(gl_c, C_accum[1][1], {0, 0, cr + WM*1 + warp_row, cc + WN*1 + warp_col});
}

// ============================================================
// Template: Per-thread transposed load from global → shared
// Loads [SRC_ROWS, SRC_COLS] block from global and stores transposed
// as [DST_ROWS, DST_COLS] in swizzled shared memory.
// SRC_ROWS = DST_COLS, SRC_COLS = DST_ROWS.
// ============================================================

template<typename ST_TYPE>
__device__ void load_transposed(
    ST_TYPE& dst,
    const typename ST_TYPE::dtype* __restrict__ src_ptr,
    int src_stride,
    int src_row_start,
    int src_col_start
) {
    using T = typename ST_TYPE::dtype;
    constexpr int DST_ROWS = ST_TYPE::rows;
    constexpr int DST_COLS = ST_TYPE::cols;
    constexpr int TOTAL = DST_ROWS * DST_COLS;
    constexpr int SUB_R = ST_TYPE::underlying_subtile_rows;
    constexpr int SUB_C = ST_TYPE::underlying_subtile_cols;
    constexpr int SUB_BYTES = ST_TYPE::underlying_subtile_bytes;
    constexpr int SUBS_PER_ROW = ST_TYPE::underlying_subtiles_per_row;

    #pragma unroll
    for (int idx = threadIdx.x; idx < TOTAL; idx += blockDim.x) {
        int src_r = idx / DST_ROWS;    // [0, DST_COLS)
        int src_c = idx % DST_ROWS;    // [0, DST_ROWS) — coalesced
        T val = src_ptr[(src_row_start + src_r) * src_stride + (src_col_start + src_c)];

        int dst_r = src_c;
        int dst_c = src_r;
        int subtile_id = (dst_r / SUB_R) * SUBS_PER_ROW + (dst_c / SUB_C);
        int local_r = dst_r % SUB_R;
        int local_c = dst_c % SUB_C;
        uint32_t byte_off = subtile_id * SUB_BYTES + ST_TYPE::swizzle({local_r, local_c});
        dst.data[byte_off / sizeof(T)] = val;
    }
}

// ============================================================
// BF16 Grouped GEMM Backward dA
// dA[M_total, K] = dC[M_total, N] * B[G*N, K]
// RRR: reduction over N
// ============================================================

__global__ __launch_bounds__(NUM_T, 2)
void grouped_bf16_bwd_dA_kernel(
    const gl_bf16 gl_dc,  // [1, 1, M_total, N]
    const gl_bf16 gl_b,   // [1, 1, G*N, K]
    const gl_bf16 gl_da,  // [1, 1, M_total, K]
    const int* __restrict__ tile_m,      // [total_blocks] m-tile for dC/dA rows (HALF_BLK units)
    const int* __restrict__ tile_k,      // [total_blocks] k-tile for dA cols (HALF_BLK units)
    const int* __restrict__ b_n_start,   // [total_blocks] starting row in B for group (element rows)
    int total_blocks
) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_DC = st_bf<HALF_BLK, K_BF16, st_16x32_s>;
    using ST_BT = st_bf<HALF_BLK, K_BF16, st_16x32_s>;
    ST_DC (&DCs)[2] = al.allocate<ST_DC, 2>();
    ST_BT (&BTs)[2] = al.allocate<ST_BT, 2>();

    rt_bf<HRM_BF16, K_BF16, row_l, rt_16x32_s> DC_tile;
    rt_bf<HRN_BF16, K_BF16, row_l, rt_16x32_s> BT_tile_0, BT_tile_1;
    rt_fl<HRM_BF16, HRN_BF16, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);

    int bid = blockIdx.x;
    if (bid >= total_blocks) return;

    int m_tile = tile_m[bid];
    int k_tile = tile_k[bid];
    int b_n_off = b_n_start[bid];   // element row offset of B for this group

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    const int N = gl_dc.cols();
    const int K = gl_b.cols();
    const int n_iters = N / K_BF16;

    using T = bf16;
    constexpr int bpt = ST_DC::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_T;
    constexpr int mpt = HALF_BLK * K_BF16 * sizeof(T) / bpm;
    uint32_t so_dc[mpt];
    G8::prefill_swizzled_offsets(DCs[0], gl_dc, so_dc);

    const bf16* b_ptr = gl_b.raw_ptr;

    // Non-pipelined loop over N reduction
    for (int ni = 0; ni < n_iters; ni++) {
        // Load dC[m_half, K_BF16] via G8::load
        G8::load(DCs[0], gl_dc, {0, 0, m_tile, ni}, so_dc);
        G8::load(DCs[1], gl_dc, {0, 0, m_tile + 1, ni}, so_dc);

        // Load B^T: transpose B[K_BF16 rows, HALF_BLK cols] → [HALF_BLK, K_BF16] in shared
        load_transposed<ST_BT>(BTs[0], b_ptr, K, b_n_off + ni * K_BF16, k_tile * HALF_BLK);
        load_transposed<ST_BT>(BTs[1], b_ptr, K, b_n_off + ni * K_BF16, (k_tile + 1) * HALF_BLK);

        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        // Compute
        auto sta0 = subtile_inplace<HRM_BF16, K_BF16>(DCs[0], {warp_row, 0});
        load(DC_tile, sta0);
        auto stb0 = subtile_inplace<HRN_BF16, K_BF16>(BTs[0], {warp_col, 0});
        load(BT_tile_0, stb0);
        auto stb1 = subtile_inplace<HRN_BF16, K_BF16>(BTs[1], {warp_col, 0});
        load(BT_tile_1, stb1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[0][0], DC_tile, BT_tile_0, C_accum[0][0]);
        mma_ABt(C_accum[0][1], DC_tile, BT_tile_1, C_accum[0][1]);

        auto sta1 = subtile_inplace<HRM_BF16, K_BF16>(DCs[1], {warp_row, 0});
        load(DC_tile, sta1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[1][0], DC_tile, BT_tile_0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], DC_tile, BT_tile_1, C_accum[1][1]);

        __builtin_amdgcn_s_barrier();
    }

    // Store dA
    int cr = m_tile * (HALF_BLK / HRM_BF16);
    int cc = k_tile * (HALF_BLK / HRN_BF16);
    store(gl_da, C_accum[0][0], {0, 0, cr + WM*0 + warp_row, cc + WN*0 + warp_col});
    store(gl_da, C_accum[0][1], {0, 0, cr + WM*0 + warp_row, cc + WN*1 + warp_col});
    store(gl_da, C_accum[1][0], {0, 0, cr + WM*1 + warp_row, cc + WN*0 + warp_col});
    store(gl_da, C_accum[1][1], {0, 0, cr + WM*1 + warp_row, cc + WN*1 + warp_col});
}

// ============================================================
// BF16 Grouped GEMM Backward dB
// dB_g[N, K] = dC_g[M_g, N]^T * A_g[M_g, K]
// CRR: reduction over M_g
// ============================================================

__global__ __launch_bounds__(NUM_T, 2)
void grouped_bf16_bwd_dB_kernel(
    const gl_bf16 gl_dc,   // [1, 1, M_total, N]
    const gl_bf16 gl_a,    // [1, 1, M_total, K]
    const gl_bf16 gl_db,   // [1, 1, G*N, K]
    const int* __restrict__ tile_n,       // [total_blocks] n-tile for dB rows (HALF_BLK units)
    const int* __restrict__ tile_k,       // [total_blocks] k-tile for dB cols (HALF_BLK units)
    const int* __restrict__ group_m_start,// [total_blocks] starting M-row for this group (elements)
    const int* __restrict__ num_m_iters,  // [total_blocks] M_g / K_BF16
    int total_blocks
) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_DCT = st_bf<HALF_BLK, K_BF16, st_16x32_s>;
    using ST_AT  = st_bf<HALF_BLK, K_BF16, st_16x32_s>;
    ST_DCT (&DCTs)[2] = al.allocate<ST_DCT, 2>();
    ST_AT  (&ATs)[2]  = al.allocate<ST_AT, 2>();

    rt_bf<HRM_BF16, K_BF16, row_l, rt_16x32_s> DCT_tile;
    rt_bf<HRN_BF16, K_BF16, row_l, rt_16x32_s> AT_tile_0, AT_tile_1;
    rt_fl<HRM_BF16, HRN_BF16, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);

    int bid = blockIdx.x;
    if (bid >= total_blocks) return;

    int n_tile = tile_n[bid];
    int k_tile = tile_k[bid];
    int m_start = group_m_start[bid];  // element row
    int m_iters = num_m_iters[bid];    // M_g / K_BF16

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    const int N = gl_dc.cols();
    const int K = gl_a.cols();

    const bf16* dc_ptr = gl_dc.raw_ptr;
    const bf16* a_ptr  = gl_a.raw_ptr;

    // Loop over M_g reduction (M_g / K_BF16 iterations)
    for (int mi = 0; mi < m_iters; mi++) {
        int m_off = m_start + mi * K_BF16;

        // Load dC^T: transpose dC[K_BF16 rows, HALF_BLK cols] → [HALF_BLK, K_BF16] in shared
        load_transposed<ST_DCT>(DCTs[0], dc_ptr, N, m_off, n_tile * HALF_BLK);
        load_transposed<ST_DCT>(DCTs[1], dc_ptr, N, m_off, (n_tile + 1) * HALF_BLK);

        // Load A^T: transpose A[K_BF16 rows, HALF_BLK cols] → [HALF_BLK, K_BF16] in shared
        load_transposed<ST_AT>(ATs[0], a_ptr, K, m_off, k_tile * HALF_BLK);
        load_transposed<ST_AT>(ATs[1], a_ptr, K, m_off, (k_tile + 1) * HALF_BLK);

        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        auto sta0 = subtile_inplace<HRM_BF16, K_BF16>(DCTs[0], {warp_row, 0});
        load(DCT_tile, sta0);
        auto stb0 = subtile_inplace<HRN_BF16, K_BF16>(ATs[0], {warp_col, 0});
        load(AT_tile_0, stb0);
        auto stb1 = subtile_inplace<HRN_BF16, K_BF16>(ATs[1], {warp_col, 0});
        load(AT_tile_1, stb1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[0][0], DCT_tile, AT_tile_0, C_accum[0][0]);
        mma_ABt(C_accum[0][1], DCT_tile, AT_tile_1, C_accum[0][1]);

        auto sta1 = subtile_inplace<HRM_BF16, K_BF16>(DCTs[1], {warp_row, 0});
        load(DCT_tile, sta1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[1][0], DCT_tile, AT_tile_0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], DCT_tile, AT_tile_1, C_accum[1][1]);

        __builtin_amdgcn_s_barrier();
    }

    // Store dB
    int cr = n_tile * (HALF_BLK / HRM_BF16);
    int cc = k_tile * (HALF_BLK / HRN_BF16);
    store(gl_db, C_accum[0][0], {0, 0, cr + WM*0 + warp_row, cc + WN*0 + warp_col});
    store(gl_db, C_accum[0][1], {0, 0, cr + WM*0 + warp_row, cc + WN*1 + warp_col});
    store(gl_db, C_accum[1][0], {0, 0, cr + WM*1 + warp_row, cc + WN*0 + warp_col});
    store(gl_db, C_accum[1][1], {0, 0, cr + WM*1 + warp_row, cc + WN*1 + warp_col});
}

// ============================================================
// FP8 Grouped GEMM Backward dA
// dA[M_total, K] = scale * dC_fp8[M_total, N] * B_fp8[G*N, K]
// RRR: reduction over N
// ============================================================

__global__ __launch_bounds__(NUM_T, 2)
void grouped_fp8_bwd_dA_kernel(
    const gl_fp8 gl_dc,
    const gl_fp8 gl_b,
    const gl_bf16_out gl_da,
    const int* __restrict__ tile_m,
    const int* __restrict__ tile_k,
    const int* __restrict__ b_n_start,
    int total_blocks, float scale
) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_DC = st_fp8e4m3<HALF_BLK, K_FP8, st_16x128_s>;
    using ST_BT = st_fp8e4m3<HALF_BLK, K_FP8, st_16x128_s>;
    ST_DC (&DCs)[2] = al.allocate<ST_DC, 2>();
    ST_BT (&BTs)[2] = al.allocate<ST_BT, 2>();

    rt_fp8e4m3<RBM_FP8, K_FP8> DC_tile;
    rt_fp8e4m3<RBN_FP8, K_FP8> BT_tile_0, BT_tile_1;
    rt_fl<RBM_FP8, RBN_FP8, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);

    int bid = blockIdx.x;
    if (bid >= total_blocks) return;

    int m_tile = tile_m[bid];
    int k_tile = tile_k[bid];
    int b_n_off = b_n_start[bid];

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    const int N = gl_dc.cols();
    const int K = gl_b.cols();
    const int n_iters = N / K_FP8;

    constexpr int bpt = ST_DC::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_T;
    constexpr int mpt = HALF_BLK * K_FP8 * sizeof(fp8e4m3) / bpm;
    uint32_t so_dc[mpt];
    G8::prefill_swizzled_offsets(DCs[0], gl_dc, so_dc);

    const fp8e4m3* b_ptr = gl_b.raw_ptr;

    for (int ni = 0; ni < n_iters; ni++) {
        G8::load(DCs[0], gl_dc, {0, 0, m_tile, ni}, so_dc);
        G8::load(DCs[1], gl_dc, {0, 0, m_tile + 1, ni}, so_dc);

        load_transposed<ST_BT>(BTs[0], b_ptr, K, b_n_off + ni * K_FP8, k_tile * HALF_BLK);
        load_transposed<ST_BT>(BTs[1], b_ptr, K, b_n_off + ni * K_FP8, (k_tile + 1) * HALF_BLK);

        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        auto sta0 = subtile_inplace<RBM_FP8, K_FP8>(DCs[0], {warp_row, 0});
        load(DC_tile, sta0);
        auto stb0 = subtile_inplace<RBN_FP8, K_FP8>(BTs[0], {warp_col, 0});
        load(BT_tile_0, stb0);
        auto stb1 = subtile_inplace<RBN_FP8, K_FP8>(BTs[1], {warp_col, 0});
        load(BT_tile_1, stb1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[0][0], DC_tile, BT_tile_0, C_accum[0][0]);
        mma_ABt(C_accum[0][1], DC_tile, BT_tile_1, C_accum[0][1]);

        auto sta1 = subtile_inplace<RBM_FP8, K_FP8>(DCs[1], {warp_row, 0});
        load(DC_tile, sta1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[1][0], DC_tile, BT_tile_0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], DC_tile, BT_tile_1, C_accum[1][1]);

        __builtin_amdgcn_s_barrier();
    }

    mul(C_accum[0][0], C_accum[0][0], scale);
    mul(C_accum[0][1], C_accum[0][1], scale);
    mul(C_accum[1][0], C_accum[1][0], scale);
    mul(C_accum[1][1], C_accum[1][1], scale);

    int cr = m_tile * (HALF_BLK / RBM_FP8);
    int cc = k_tile * (HALF_BLK / RBN_FP8);
    store(gl_da, C_accum[0][0], {0, 0, cr + WM*0 + warp_row, cc + WN*0 + warp_col});
    store(gl_da, C_accum[0][1], {0, 0, cr + WM*0 + warp_row, cc + WN*1 + warp_col});
    store(gl_da, C_accum[1][0], {0, 0, cr + WM*1 + warp_row, cc + WN*0 + warp_col});
    store(gl_da, C_accum[1][1], {0, 0, cr + WM*1 + warp_row, cc + WN*1 + warp_col});
}

// ============================================================
// FP8 Grouped GEMM Backward dB
// dB_g[N, K] = scale * dC_fp8_g^T * A_fp8_g
// CRR: reduction over M_g
// ============================================================

__global__ __launch_bounds__(NUM_T, 2)
void grouped_fp8_bwd_dB_kernel(
    const gl_fp8 gl_dc,
    const gl_fp8 gl_a,
    const gl_bf16_out gl_db,
    const int* __restrict__ tile_n,
    const int* __restrict__ tile_k,
    const int* __restrict__ group_m_start,
    const int* __restrict__ num_m_iters,
    int total_blocks, float scale
) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_DCT = st_fp8e4m3<HALF_BLK, K_FP8, st_16x128_s>;
    using ST_AT  = st_fp8e4m3<HALF_BLK, K_FP8, st_16x128_s>;
    ST_DCT (&DCTs)[2] = al.allocate<ST_DCT, 2>();
    ST_AT  (&ATs)[2]  = al.allocate<ST_AT, 2>();

    rt_fp8e4m3<RBM_FP8, K_FP8> DCT_tile;
    rt_fp8e4m3<RBN_FP8, K_FP8> AT_tile_0, AT_tile_1;
    rt_fl<RBM_FP8, RBN_FP8, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);

    int bid = blockIdx.x;
    if (bid >= total_blocks) return;

    int n_tile = tile_n[bid];
    int k_tile = tile_k[bid];
    int m_start = group_m_start[bid];
    int m_iters = num_m_iters[bid];

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    const int N = gl_dc.cols();
    const int K = gl_a.cols();

    const fp8e4m3* dc_ptr = gl_dc.raw_ptr;
    const fp8e4m3* a_ptr  = gl_a.raw_ptr;

    for (int mi = 0; mi < m_iters; mi++) {
        int m_off = m_start + mi * K_FP8;

        load_transposed<ST_DCT>(DCTs[0], dc_ptr, N, m_off, n_tile * HALF_BLK);
        load_transposed<ST_DCT>(DCTs[1], dc_ptr, N, m_off, (n_tile + 1) * HALF_BLK);

        load_transposed<ST_AT>(ATs[0], a_ptr, K, m_off, k_tile * HALF_BLK);
        load_transposed<ST_AT>(ATs[1], a_ptr, K, m_off, (k_tile + 1) * HALF_BLK);

        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        auto sta0 = subtile_inplace<RBM_FP8, K_FP8>(DCTs[0], {warp_row, 0});
        load(DCT_tile, sta0);
        auto stb0 = subtile_inplace<RBN_FP8, K_FP8>(ATs[0], {warp_col, 0});
        load(AT_tile_0, stb0);
        auto stb1 = subtile_inplace<RBN_FP8, K_FP8>(ATs[1], {warp_col, 0});
        load(AT_tile_1, stb1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[0][0], DCT_tile, AT_tile_0, C_accum[0][0]);
        mma_ABt(C_accum[0][1], DCT_tile, AT_tile_1, C_accum[0][1]);

        auto sta1 = subtile_inplace<RBM_FP8, K_FP8>(DCTs[1], {warp_row, 0});
        load(DCT_tile, sta1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[1][0], DCT_tile, AT_tile_0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], DCT_tile, AT_tile_1, C_accum[1][1]);

        __builtin_amdgcn_s_barrier();
    }

    mul(C_accum[0][0], C_accum[0][0], scale);
    mul(C_accum[0][1], C_accum[0][1], scale);
    mul(C_accum[1][0], C_accum[1][0], scale);
    mul(C_accum[1][1], C_accum[1][1], scale);

    int cr = n_tile * (HALF_BLK / RBM_FP8);
    int cc = k_tile * (HALF_BLK / RBN_FP8);
    store(gl_db, C_accum[0][0], {0, 0, cr + WM*0 + warp_row, cc + WN*0 + warp_col});
    store(gl_db, C_accum[0][1], {0, 0, cr + WM*0 + warp_row, cc + WN*1 + warp_col});
    store(gl_db, C_accum[1][0], {0, 0, cr + WM*1 + warp_row, cc + WN*0 + warp_col});
    store(gl_db, C_accum[1][1], {0, 0, cr + WM*1 + warp_row, cc + WN*1 + warp_col});
}

// ============================================================
// Host dispatch functions
// ============================================================

void dispatch_grouped_bf16_fwd(
    pybind11::object A_obj, pybind11::object B_obj, pybind11::object C_obj,
    pybind11::object tile_m_obj, pybind11::object tile_n_obj, pybind11::object tile_bn_obj,
    int total_blocks
) {
    // Create gl objects on host
    auto a_gl = kittens::py::from_object<gl_bf16>::make(A_obj);
    auto b_gl = kittens::py::from_object<gl_bf16>::make(B_obj);
    auto c_gl = kittens::py::from_object<gl_bf16>::make(C_obj);

    uint64_t tm_ptr = tile_m_obj.attr("data_ptr")().cast<uint64_t>();
    uint64_t tn_ptr = tile_n_obj.attr("data_ptr")().cast<uint64_t>();
    uint64_t tbn_ptr = tile_bn_obj.attr("data_ptr")().cast<uint64_t>();

    dim3 grid(total_blocks);
    dim3 block(NUM_T);
    unsigned long mem = MAX_SHARED_MEMORY;
    hipFuncSetAttribute((void*)grouped_bf16_fwd_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    grouped_bf16_fwd_kernel<<<grid, block, mem, 0>>>(
        a_gl, b_gl, c_gl,
        (const int*)tm_ptr, (const int*)tn_ptr, (const int*)tbn_ptr,
        total_blocks
    );
}

void dispatch_grouped_fp8_fwd(
    pybind11::object A_obj, pybind11::object B_obj, pybind11::object C_obj,
    pybind11::object tile_m_obj, pybind11::object tile_n_obj, pybind11::object tile_bn_obj,
    int total_blocks, float scale
) {
    auto a_gl = kittens::py::from_object<gl_fp8>::make(A_obj);
    auto b_gl = kittens::py::from_object<gl_fp8>::make(B_obj);
    auto c_gl = kittens::py::from_object<gl_bf16_out>::make(C_obj);

    uint64_t tm_ptr = tile_m_obj.attr("data_ptr")().cast<uint64_t>();
    uint64_t tn_ptr = tile_n_obj.attr("data_ptr")().cast<uint64_t>();
    uint64_t tbn_ptr = tile_bn_obj.attr("data_ptr")().cast<uint64_t>();

    dim3 grid(total_blocks);
    dim3 block(NUM_T);
    unsigned long mem = MAX_SHARED_MEMORY;
    hipFuncSetAttribute((void*)grouped_fp8_fwd_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    grouped_fp8_fwd_kernel<<<grid, block, mem, 0>>>(
        a_gl, b_gl, c_gl,
        (const int*)tm_ptr, (const int*)tn_ptr, (const int*)tbn_ptr,
        total_blocks, scale
    );
}

// ---- BF16 backward dA dispatch ----
void dispatch_grouped_bf16_bwd_dA(
    pybind11::object dC_obj, pybind11::object B_obj, pybind11::object dA_obj,
    pybind11::object tile_m_obj, pybind11::object tile_k_obj, pybind11::object b_n_start_obj,
    int total_blocks
) {
    auto dc_gl = kittens::py::from_object<gl_bf16>::make(dC_obj);
    auto b_gl  = kittens::py::from_object<gl_bf16>::make(B_obj);
    auto da_gl = kittens::py::from_object<gl_bf16>::make(dA_obj);

    uint64_t tm_ptr = tile_m_obj.attr("data_ptr")().cast<uint64_t>();
    uint64_t tk_ptr = tile_k_obj.attr("data_ptr")().cast<uint64_t>();
    uint64_t bn_ptr = b_n_start_obj.attr("data_ptr")().cast<uint64_t>();

    dim3 grid(total_blocks);
    dim3 block(NUM_T);
    unsigned long mem = MAX_SHARED_MEMORY;
    hipFuncSetAttribute((void*)grouped_bf16_bwd_dA_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    grouped_bf16_bwd_dA_kernel<<<grid, block, mem, 0>>>(
        dc_gl, b_gl, da_gl,
        (const int*)tm_ptr, (const int*)tk_ptr, (const int*)bn_ptr,
        total_blocks
    );
}

// ---- BF16 backward dB dispatch ----
void dispatch_grouped_bf16_bwd_dB(
    pybind11::object dC_obj, pybind11::object A_obj, pybind11::object dB_obj,
    pybind11::object tile_n_obj, pybind11::object tile_k_obj,
    pybind11::object group_m_start_obj, pybind11::object num_m_iters_obj,
    int total_blocks
) {
    auto dc_gl = kittens::py::from_object<gl_bf16>::make(dC_obj);
    auto a_gl  = kittens::py::from_object<gl_bf16>::make(A_obj);
    auto db_gl = kittens::py::from_object<gl_bf16>::make(dB_obj);

    uint64_t tn_ptr   = tile_n_obj.attr("data_ptr")().cast<uint64_t>();
    uint64_t tk_ptr   = tile_k_obj.attr("data_ptr")().cast<uint64_t>();
    uint64_t gms_ptr  = group_m_start_obj.attr("data_ptr")().cast<uint64_t>();
    uint64_t nmi_ptr  = num_m_iters_obj.attr("data_ptr")().cast<uint64_t>();

    dim3 grid(total_blocks);
    dim3 block(NUM_T);
    unsigned long mem = MAX_SHARED_MEMORY;
    hipFuncSetAttribute((void*)grouped_bf16_bwd_dB_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    grouped_bf16_bwd_dB_kernel<<<grid, block, mem, 0>>>(
        dc_gl, a_gl, db_gl,
        (const int*)tn_ptr, (const int*)tk_ptr,
        (const int*)gms_ptr, (const int*)nmi_ptr,
        total_blocks
    );
}

// ---- FP8 backward dA dispatch ----
void dispatch_grouped_fp8_bwd_dA(
    pybind11::object dC_obj, pybind11::object B_obj, pybind11::object dA_obj,
    pybind11::object tile_m_obj, pybind11::object tile_k_obj, pybind11::object b_n_start_obj,
    int total_blocks, float scale
) {
    auto dc_gl = kittens::py::from_object<gl_fp8>::make(dC_obj);
    auto b_gl  = kittens::py::from_object<gl_fp8>::make(B_obj);
    auto da_gl = kittens::py::from_object<gl_bf16_out>::make(dA_obj);

    uint64_t tm_ptr = tile_m_obj.attr("data_ptr")().cast<uint64_t>();
    uint64_t tk_ptr = tile_k_obj.attr("data_ptr")().cast<uint64_t>();
    uint64_t bn_ptr = b_n_start_obj.attr("data_ptr")().cast<uint64_t>();

    dim3 grid(total_blocks);
    dim3 block(NUM_T);
    unsigned long mem = MAX_SHARED_MEMORY;
    hipFuncSetAttribute((void*)grouped_fp8_bwd_dA_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    grouped_fp8_bwd_dA_kernel<<<grid, block, mem, 0>>>(
        dc_gl, b_gl, da_gl,
        (const int*)tm_ptr, (const int*)tk_ptr, (const int*)bn_ptr,
        total_blocks, scale
    );
}

// ---- FP8 backward dB dispatch ----
void dispatch_grouped_fp8_bwd_dB(
    pybind11::object dC_obj, pybind11::object A_obj, pybind11::object dB_obj,
    pybind11::object tile_n_obj, pybind11::object tile_k_obj,
    pybind11::object group_m_start_obj, pybind11::object num_m_iters_obj,
    int total_blocks, float scale
) {
    auto dc_gl = kittens::py::from_object<gl_fp8>::make(dC_obj);
    auto a_gl  = kittens::py::from_object<gl_fp8>::make(A_obj);
    auto db_gl = kittens::py::from_object<gl_bf16_out>::make(dB_obj);

    uint64_t tn_ptr   = tile_n_obj.attr("data_ptr")().cast<uint64_t>();
    uint64_t tk_ptr   = tile_k_obj.attr("data_ptr")().cast<uint64_t>();
    uint64_t gms_ptr  = group_m_start_obj.attr("data_ptr")().cast<uint64_t>();
    uint64_t nmi_ptr  = num_m_iters_obj.attr("data_ptr")().cast<uint64_t>();

    dim3 grid(total_blocks);
    dim3 block(NUM_T);
    unsigned long mem = MAX_SHARED_MEMORY;
    hipFuncSetAttribute((void*)grouped_fp8_bwd_dB_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    grouped_fp8_bwd_dB_kernel<<<grid, block, mem, 0>>>(
        dc_gl, a_gl, db_gl,
        (const int*)tn_ptr, (const int*)tk_ptr,
        (const int*)gms_ptr, (const int*)nmi_ptr,
        total_blocks, scale
    );
}

// ============================================================
// Pybind11
// ============================================================

PYBIND11_MODULE(grouped_gemm, m) {
    m.doc() = "Grouped GEMM kernels (BF16 + FP8 per-tensor, forward + backward)";

    m.def("bf16_fwd", &dispatch_grouped_bf16_fwd,
          "Grouped BF16 GEMM forward (RCR)");
    m.def("fp8_fwd", &dispatch_grouped_fp8_fwd,
          "Grouped FP8 per-tensor GEMM forward (RCR)");

    // Backward
    m.def("bf16_bwd_dA", &dispatch_grouped_bf16_bwd_dA,
          "Grouped BF16 GEMM backward dA = dC * B");
    m.def("bf16_bwd_dB", &dispatch_grouped_bf16_bwd_dB,
          "Grouped BF16 GEMM backward dB = dC^T * A");
    m.def("fp8_bwd_dA", &dispatch_grouped_fp8_bwd_dA,
          "Grouped FP8 GEMM backward dA = scale * dC_fp8 * B_fp8");
    m.def("fp8_bwd_dB", &dispatch_grouped_fp8_bwd_dB,
          "Grouped FP8 GEMM backward dB = scale * dC_fp8^T * A_fp8");
}
