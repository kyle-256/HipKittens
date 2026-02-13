/**
 * FP8 Per-Tensor GEMM kernels using HipKittens.
 *
 * RCR layout: C[M,N] = A[M,K] * B[N,K]^T * scale     — mma_ABt, G::load for both A and B
 * RRR layout: C[M,N] = A[M,K] * B[K,N]   * scale      — mma_ABt, G::load for A, transposed load for B
 * CRR layout: C[M,N] = At[K,M]^T * B[K,N] * scale     — mma_ABt, transposed load for both A and B
 *
 * A: fp8e4m3, B: fp8e4m3, C: bf16
 * scale: float (= scale_a * scale_b)
 *
 * FP8 MFMA only supports mma_ABt (row_l * row_l^T). For RRR/CRR layouts where
 * inputs are not in [N,K] form, we use per-thread global loads with shared-memory
 * transposition to convert B[K,N] → B^T[N,K] (and At[K,M] → A[M,K] for CRR).
 */

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// Tile configuration
constexpr int BLOCK_SIZE      = 256;
constexpr int HALF_BLOCK_SIZE = BLOCK_SIZE / 2;   // 128
constexpr int K_STEP          = 128;               // FP8 K step
constexpr int WARPS_M         = 2;
constexpr int WARPS_N         = 4;
constexpr int REG_BLOCK_M     = BLOCK_SIZE / WARPS_M / 2;  // 64
constexpr int REG_BLOCK_N     = BLOCK_SIZE / WARPS_N / 2;   // 32

#define NUM_WARPS (WARPS_M * WARPS_N)
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_fp8 = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_out = gl<bf16, -1, -1, -1, -1>;
using G = kittens::group<NUM_WARPS>;
using ST_FP8 = st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_16x128_s>;


struct block_mapping {
    int row, col;
    int warp_row, warp_col;
    int num_tiles;
};

__device__ block_mapping compute_block_mapping(int M, int N, int K) {
    block_mapping bm;

    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    const int WGM = 8;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, 64);
    const int num_pid_m = ceil_div(M, BLOCK_SIZE);
    const int num_pid_n = ceil_div(N, BLOCK_SIZE);
    const int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;

    bm.row = pid_m;
    bm.col = pid_n;

    const int warp_id = kittens::warpid();
    bm.warp_row = warp_id / 4;   // M sub-tile (0 or 1)
    bm.warp_col = warp_id % 4;   // N sub-tile (0..3)
    bm.num_tiles = K / K_STEP;

    return bm;
}

__device__ void store_output(const _gl_out& c, 
    rt_fl<REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s> C_accum[2][2],
    float scale, int row, int col, int warp_row, int warp_col) {
    mul(C_accum[0][0], C_accum[0][0], scale);
    mul(C_accum[0][1], C_accum[0][1], scale);
    mul(C_accum[1][0], C_accum[1][0], scale);
    mul(C_accum[1][1], C_accum[1][1], scale);

    store(c, C_accum[0][0], {0, 0, (row * 2) * WARPS_M + warp_row, col * 2 * WARPS_N + warp_col});
    store(c, C_accum[0][1], {0, 0, (row * 2) * WARPS_M + warp_row, col * 2 * WARPS_N + WARPS_N + warp_col});
    store(c, C_accum[1][0], {0, 0, (row * 2) * WARPS_M + WARPS_M + warp_row, col * 2 * WARPS_N + warp_col});
    store(c, C_accum[1][1], {0, 0, (row * 2) * WARPS_M + WARPS_M + warp_row, col * 2 * WARPS_N + WARPS_N + warp_col});
}

// ============================================================
// FP8 Per-Tensor RCR: C[M,N] = A[M,K] * B[N,K]^T * scale
// ============================================================

struct fp8_rcr_globals {
    _gl_fp8 a, b;
    _gl_out c;
    float scale;
    hipStream_t stream;
    int M = a.rows();
    int N = c.cols();
    int K = a.cols();
    dim3 grid()  { return dim3(ceil_div(N, BLOCK_SIZE) * ceil_div(M, BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void fp8_gemm_rcr_kernel(const fp8_rcr_globals g, int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_16x128_s>;
    using ST_B = st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_16x128_s>;
    ST_A (&As)[2][2] = al.allocate<ST_A, 2, 2>();
    ST_B (&Bs)[2][2] = al.allocate<ST_B, 2, 2>();

    rt_fp8e4m3<REG_BLOCK_M, K_STEP> A_tile;
    rt_fp8e4m3<REG_BLOCK_N, K_STEP> B_tile_0;
    rt_fp8e4m3<REG_BLOCK_N, K_STEP> B_tile_1;

    rt_fl<REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);

    auto bm = compute_block_mapping(M, N, K);
    int row = bm.row, col = bm.col;
    int warp_row = bm.warp_row, warp_col = bm.warp_col;
    int num_tiles = bm.num_tiles;

    using T = typename ST_A::dtype;
    constexpr int bytes_per_thread = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = HALF_BLOCK_SIZE * K_STEP * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);

    int tic = 0, toc = 1;

    // === Prologue: load first two K tiles ===
    G::load(Bs[tic][0], g.b, {0, 0, col*2, 0}, swizzled_offsets_B);
    G::load(As[tic][0], g.a, {0, 0, row*2, 0}, swizzled_offsets_A);
    G::load(Bs[tic][1], g.b, {0, 0, col*2 + 1, 0}, swizzled_offsets_B);
    G::load(As[tic][1], g.a, {0, 0, row*2 + 1, 0}, swizzled_offsets_A);

    if (warp_row == 1) __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(As[toc][0], g.a, {0, 0, row*2, 1}, swizzled_offsets_A);
    G::load(Bs[toc][0], g.b, {0, 0, col*2, 1}, swizzled_offsets_B);
    G::load(Bs[toc][1], g.b, {0, 0, col*2 + 1, 1}, swizzled_offsets_B);
    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    // === Main loop ===
    #pragma unroll
    for (int tile = 0; tile < num_tiles - 2; tile++, tic^=1, toc^=1) {
        auto stb0 = subtile_inplace<REG_BLOCK_N, K_STEP>(Bs[tic][0], {warp_col, 0});
        load(B_tile_0, stb0);
        auto sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][0], {warp_row, 0});
        load(A_tile, sta0);
        G::load(As[toc][1], g.a, {0, 0, row*2 + 1, tile + 1}, swizzled_offsets_A);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto stb1 = subtile_inplace<REG_BLOCK_N, K_STEP>(Bs[tic][1], {warp_col, 0});
        load(B_tile_1, stb1);
        G::load(As[tic][0], g.a, {0, 0, row*2, tile + 2}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][1], {warp_row, 0});
        load(A_tile, sta0);
        G::load(Bs[tic][0], g.b, {0, 0, col*2, tile + 2}, swizzled_offsets_B);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[tic][1], g.b, {0, 0, col*2 + 1, tile + 2}, swizzled_offsets_B);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    // === Epilogue: drain last two K tiles ===
    {
        auto stb0 = subtile_inplace<REG_BLOCK_N, K_STEP>(Bs[tic][0], {warp_col, 0});
        load(B_tile_0, stb0);
        auto sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][0], {warp_row, 0});
        load(A_tile, sta0);
        G::load(As[toc][1], g.a, {0, 0, row*2 + 1, num_tiles - 1}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto stb1 = subtile_inplace<REG_BLOCK_N, K_STEP>(Bs[tic][1], {warp_col, 0});
        load(B_tile_1, stb1);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][1], {warp_row, 0});
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
        auto stb0 = subtile_inplace<REG_BLOCK_N, K_STEP>(Bs[tic][0], {warp_col, 0});
        load(B_tile_0, stb0);
        auto sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][0], {warp_row, 0});
        load(A_tile, sta0);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto stb1 = subtile_inplace<REG_BLOCK_N, K_STEP>(Bs[tic][1], {warp_col, 0});
        load(B_tile_1, stb1);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][1], {warp_row, 0});
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

    store_output(g.c, C_accum, g.scale, row, col, warp_row, warp_col);
}

void dispatch_fp8_rcr(fp8_rcr_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)fp8_gemm_rcr_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    fp8_gemm_rcr_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M, g.N, g.K);
}

// ============================================================
// FP8 Per-Tensor RRR: C[M,N] = A[M,K] * B[K,N] * scale
//
// A row-major [M,K]: loaded as row_l from st_16x128 tiles (same as RCR)
// B row-major [K,N]: loaded as col_l from st_128x16 tiles (ds_read_b64_tr_b8)
// mma_AB: D += A_row * B_col   → native FP8, no transposes
// ============================================================

struct fp8_rrr_globals {
    _gl_fp8 a, b;
    _gl_out c;
    float scale;
    hipStream_t stream;
    int M = a.rows();   // A is [M,K]
    int N = b.cols();   // B is [K,N]
    int K = a.cols();
    dim3 grid()  { return dim3(ceil_div(N, BLOCK_SIZE) * ceil_div(M, BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void fp8_gemm_rrr_kernel(const fp8_rrr_globals g, int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // A: [M_half=128, K=128] row_l (same as RCR)
    using ST_A = st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_16x128_s>;
    // B: [K=128, N_half=128] col_l via ds_read_b64_tr_b8
    using ST_B = st_fp8e4m3<K_STEP, HALF_BLOCK_SIZE, st_128x16_s>;
    // Double-buffered: [tic/toc][m/n half]
    ST_A (&As)[2][2] = al.allocate<ST_A, 2, 2>();
    ST_B (&Bs)[2][2] = al.allocate<ST_B, 2, 2>();

    // Register tiles: A row_l, B col_l for mma_AB
    rt_fp8e4m3<REG_BLOCK_M, K_STEP, row_l> A_tile;
    rt<fp8e4m3, K_STEP, REG_BLOCK_N, col_l, rt_128x16_s> B_tile_0;
    rt<fp8e4m3, K_STEP, REG_BLOCK_N, col_l, rt_128x16_s> B_tile_1;
    rt_fl<REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);

    auto bm = compute_block_mapping(M, N, K);
    int row = bm.row, col = bm.col;
    int warp_row = bm.warp_row, warp_col = bm.warp_col;
    int num_tiles = bm.num_tiles;

    using T = typename ST_A::dtype;
    constexpr int a_bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int a_bpm = a_bpt * NUM_THREADS;
    constexpr int a_mpt = HALF_BLOCK_SIZE * K_STEP * sizeof(T) / a_bpm;
    uint32_t swizzled_offsets_A[a_mpt];
    G::prefill_swizzled_offsets(As[0][0], g.a, swizzled_offsets_A);

    constexpr int b_bpt = ST_B::underlying_subtile_bytes_per_thread;
    constexpr int b_bpm = b_bpt * NUM_THREADS;
    constexpr int b_mpt = K_STEP * HALF_BLOCK_SIZE * sizeof(T) / b_bpm;
    uint32_t swizzled_offsets_B[b_mpt];
    G::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);

    int tic = 0, toc = 1;

    // === Prologue: load first two K tiles ===
    // B tile indices: {0, 0, k_tile, n_half}
    G::load(Bs[tic][0], g.b, {0, 0, 0, col*2},     swizzled_offsets_B);
    G::load(As[tic][0], g.a, {0, 0, row*2, 0},       swizzled_offsets_A);
    G::load(Bs[tic][1], g.b, {0, 0, 0, col*2 + 1}, swizzled_offsets_B);
    G::load(As[tic][1], g.a, {0, 0, row*2 + 1, 0},   swizzled_offsets_A);

    if (warp_row == 1) __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(As[toc][0], g.a, {0, 0, row*2, 1},       swizzled_offsets_A);
    G::load(Bs[toc][0], g.b, {0, 0, 1, col*2},       swizzled_offsets_B);
    G::load(Bs[toc][1], g.b, {0, 0, 1, col*2 + 1},   swizzled_offsets_B);
    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    // === Main loop ===
    #pragma unroll
    for (int tile = 0; tile < num_tiles - 2; tile++, tic^=1, toc^=1) {
        auto stb0 = subtile_inplace<K_STEP, REG_BLOCK_N>(Bs[tic][0], {0, warp_col});
        load(B_tile_0, stb0);
        auto sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][0], {warp_row, 0});
        load(A_tile, sta0);
        G::load(As[toc][1], g.a, {0, 0, row*2 + 1, tile + 1}, swizzled_offsets_A);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto stb1 = subtile_inplace<K_STEP, REG_BLOCK_N>(Bs[tic][1], {0, warp_col});
        load(B_tile_1, stb1);
        G::load(As[tic][0], g.a, {0, 0, row*2, tile + 2}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][1], {warp_row, 0});
        load(A_tile, sta0);
        G::load(Bs[tic][0], g.b, {0, 0, tile + 2, col*2}, swizzled_offsets_B);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[tic][1], g.b, {0, 0, tile + 2, col*2 + 1}, swizzled_offsets_B);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    // === Epilogue: drain last two K tiles ===
    {
        auto stb0 = subtile_inplace<K_STEP, REG_BLOCK_N>(Bs[tic][0], {0, warp_col});
        load(B_tile_0, stb0);
        auto sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][0], {warp_row, 0});
        load(A_tile, sta0);
        G::load(As[toc][1], g.a, {0, 0, row*2 + 1, num_tiles - 1}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto stb1 = subtile_inplace<K_STEP, REG_BLOCK_N>(Bs[tic][1], {0, warp_col});
        load(B_tile_1, stb1);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][1], {warp_row, 0});
        load(A_tile, sta0);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_AB(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        tic ^= 1; toc ^= 1;
    }
    {
        auto stb0 = subtile_inplace<K_STEP, REG_BLOCK_N>(Bs[tic][0], {0, warp_col});
        load(B_tile_0, stb0);
        auto sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][0], {warp_row, 0});
        load(A_tile, sta0);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto stb1 = subtile_inplace<K_STEP, REG_BLOCK_N>(Bs[tic][1], {0, warp_col});
        load(B_tile_1, stb1);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][1], {warp_row, 0});
        load(A_tile, sta0);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_AB(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (warp_row == 0) __builtin_amdgcn_s_barrier();

    store_output(g.c, C_accum, g.scale, row, col, warp_row, warp_col);
}

void dispatch_fp8_rrr(fp8_rrr_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)fp8_gemm_rrr_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    fp8_gemm_rrr_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M, g.N, g.K);
}

// ============================================================
// FP8 Per-Tensor RRR (4-wave): C[M,N] = A[M,K] * B[K,N] * scale
//
// Uses mfma_32x32x64_fp8 (4-wave) instead of mfma_16x16x128 (8-wave).
// B col_l load uses rt_64x32 base (laneid/32 → only 2 row groups → 2-way bank conflict)
// vs rt_128x16 base (laneid/16 → 4 row groups → 4-way bank conflict) for 8-wave.
// Expected: ~98% of RCR speed (matching BF16 RRR/RCR ratio).
// ============================================================

__global__ __launch_bounds__(NUM_THREADS, 2)
void fp8_gemm_rrr_4wave_kernel(const fp8_rrr_globals g, int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // 4-wave: A uses st_32x64 subtiles (for rt_32x64 row_l), B uses st_128x16 (for rt_64x32 col_l)
    using ST_A = st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_32x64_s>;
    using ST_B = st_fp8e4m3<K_STEP, HALF_BLOCK_SIZE, st_128x16_s>;
    ST_A (&As)[2][2] = al.allocate<ST_A, 2, 2>();
    ST_B (&Bs)[2][2] = al.allocate<ST_B, 2, 2>();

    // Register tiles: 4-wave base shapes (rt_32x64 for A, rt_64x32 for B)
    rt<fp8e4m3, REG_BLOCK_M, K_STEP, row_l, rt_32x64_s> A_tile;
    rt<fp8e4m3, K_STEP, REG_BLOCK_N, col_l, rt_64x32_s> B_tile_0;
    rt<fp8e4m3, K_STEP, REG_BLOCK_N, col_l, rt_64x32_s> B_tile_1;
    rt_fl<REG_BLOCK_M, REG_BLOCK_N, col_l, rt_32x32_s> C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);

    auto bm = compute_block_mapping(M, N, K);
    int row = bm.row, col = bm.col;
    int warp_row = bm.warp_row, warp_col = bm.warp_col;
    int num_tiles = bm.num_tiles;

    using T = typename ST_A::dtype;
    constexpr int a_bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int a_bpm = a_bpt * NUM_THREADS;
    constexpr int a_mpt = HALF_BLOCK_SIZE * K_STEP * sizeof(T) / a_bpm;
    uint32_t swizzled_offsets_A[a_mpt];
    G::prefill_swizzled_offsets(As[0][0], g.a, swizzled_offsets_A);

    constexpr int b_bpt = ST_B::underlying_subtile_bytes_per_thread;
    constexpr int b_bpm = b_bpt * NUM_THREADS;
    constexpr int b_mpt = K_STEP * HALF_BLOCK_SIZE * sizeof(T) / b_bpm;
    uint32_t swizzled_offsets_B[b_mpt];
    G::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);

    int tic = 0, toc = 1;

    // === Prologue ===
    G::load(Bs[tic][0], g.b, {0, 0, 0, col*2},     swizzled_offsets_B);
    G::load(As[tic][0], g.a, {0, 0, row*2, 0},       swizzled_offsets_A);
    G::load(Bs[tic][1], g.b, {0, 0, 0, col*2 + 1}, swizzled_offsets_B);
    G::load(As[tic][1], g.a, {0, 0, row*2 + 1, 0},   swizzled_offsets_A);

    if (warp_row == 1) __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(As[toc][0], g.a, {0, 0, row*2, 1},       swizzled_offsets_A);
    G::load(Bs[toc][0], g.b, {0, 0, 1, col*2},       swizzled_offsets_B);
    G::load(Bs[toc][1], g.b, {0, 0, 1, col*2 + 1},   swizzled_offsets_B);
    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    // === Main loop (identical structure to 8-wave, different tile types) ===
    #pragma unroll
    for (int tile = 0; tile < num_tiles - 2; tile++, tic^=1, toc^=1) {
        auto stb0 = subtile_inplace<K_STEP, REG_BLOCK_N>(Bs[tic][0], {0, warp_col});
        load(B_tile_0, stb0);
        auto sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][0], {warp_row, 0});
        load(A_tile, sta0);
        G::load(As[toc][1], g.a, {0, 0, row*2 + 1, tile + 1}, swizzled_offsets_A);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto stb1 = subtile_inplace<K_STEP, REG_BLOCK_N>(Bs[tic][1], {0, warp_col});
        load(B_tile_1, stb1);
        G::load(As[tic][0], g.a, {0, 0, row*2, tile + 2}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][1], {warp_row, 0});
        load(A_tile, sta0);
        G::load(Bs[tic][0], g.b, {0, 0, tile + 2, col*2}, swizzled_offsets_B);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[tic][1], g.b, {0, 0, tile + 2, col*2 + 1}, swizzled_offsets_B);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    // === Epilogue ===
    {
        auto stb0 = subtile_inplace<K_STEP, REG_BLOCK_N>(Bs[tic][0], {0, warp_col});
        load(B_tile_0, stb0);
        auto sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][0], {warp_row, 0});
        load(A_tile, sta0);
        G::load(As[toc][1], g.a, {0, 0, row*2 + 1, num_tiles - 1}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto stb1 = subtile_inplace<K_STEP, REG_BLOCK_N>(Bs[tic][1], {0, warp_col});
        load(B_tile_1, stb1);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][1], {warp_row, 0});
        load(A_tile, sta0);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_AB(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        tic ^= 1; toc ^= 1;
    }
    {
        auto stb0 = subtile_inplace<K_STEP, REG_BLOCK_N>(Bs[tic][0], {0, warp_col});
        load(B_tile_0, stb0);
        auto sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][0], {warp_row, 0});
        load(A_tile, sta0);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto stb1 = subtile_inplace<K_STEP, REG_BLOCK_N>(Bs[tic][1], {0, warp_col});
        load(B_tile_1, stb1);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][1], {warp_row, 0});
        load(A_tile, sta0);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_AB(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (warp_row == 0) __builtin_amdgcn_s_barrier();

    // Store output — use rt_32x32 accumulator
    mul(C_accum[0][0], C_accum[0][0], g.scale);
    mul(C_accum[0][1], C_accum[0][1], g.scale);
    mul(C_accum[1][0], C_accum[1][0], g.scale);
    mul(C_accum[1][1], C_accum[1][1], g.scale);
    store(g.c, C_accum[0][0], {0, 0, (row * 2) * WARPS_M + warp_row, col * 2 * WARPS_N + warp_col});
    store(g.c, C_accum[0][1], {0, 0, (row * 2) * WARPS_M + warp_row, col * 2 * WARPS_N + WARPS_N + warp_col});
    store(g.c, C_accum[1][0], {0, 0, (row * 2) * WARPS_M + WARPS_M + warp_row, col * 2 * WARPS_N + warp_col});
    store(g.c, C_accum[1][1], {0, 0, (row * 2) * WARPS_M + WARPS_M + warp_row, col * 2 * WARPS_N + WARPS_N + warp_col});
}

void dispatch_fp8_rrr_4wave(fp8_rrr_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)fp8_gemm_rrr_4wave_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    fp8_gemm_rrr_4wave_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M, g.N, g.K);
}

// ============================================================
// FP8 Per-Tensor CRR: C[M,N] = At[K,M]^T * B[K,N] * scale
//
// At row-major [K,M] (col-major A), B row-major [K,N], C row-major [M,N]
//
// Strategy: transpose At in shared memory to get A[M,K] (st_16x128 layout),
// then use mma_AB (same as RRR) with A row_l and B col_l.
// This avoids the incompatibility of ds_read_b64_tr_b8 col_l data with
// MFMA srcA when K=128 (confirmed by CK: ds_read_tr requires K_warp ≤ 64 for fp8).
// ============================================================

struct fp8_crr_globals {
    _gl_fp8 a, b;
    _gl_out c;
    float scale;
    hipStream_t stream;
    int M = a.cols();   // a is At[K,M]
    int N = b.cols();   // b is B[K,N]
    int K = a.rows();
    dim3 grid()  { return dim3(ceil_div(N, BLOCK_SIZE) * ceil_div(M, BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// Transpose helper: copy [128, 128] data from st_128x16 (no swizzle) layout
// to st_16x128 (swizzled) layout in shared memory.
// At_col stores At[k, m] → A_row stores A[m, k] = At[k, m]
__device__ void transpose_128x128_fp8(
    const st_fp8e4m3<K_STEP, HALF_BLOCK_SIZE, st_128x16_s>& src,
    st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_16x128_s>& dst)
{
    // src layout (st_128x16, no swizzle): 8 subtiles of [128, 16]
    //   src.data[subtile*2048 + k*16 + m_local] = At[k, subtile*16 + m_local]
    // dst layout (st_16x128, swizzled): 8 subtiles of [16, 128]
    //   dst.data[subtile*2048 + swizzle_16x128(m_local, k)] = A[subtile*16 + m_local, k]
    //   where swizzle_16x128(row, col) = (row*128 + col) ^ (((row*128+col) >> 8) << 4)

    const int tid = threadIdx.x;  // 0..511
    constexpr int ELEMS_PER_THREAD = (K_STEP * HALF_BLOCK_SIZE) / NUM_THREADS;  // 128*128/512 = 32

    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        const int flat_idx = tid * ELEMS_PER_THREAD + e;
        const int k = flat_idx / HALF_BLOCK_SIZE;  // 0..127 (K dimension)
        const int m = flat_idx % HALF_BLOCK_SIZE;  // 0..127 (M dimension)

        // Source: st_128x16, no swizzle
        const int src_subtile = m / 16;
        const int src_offset = src_subtile * (128 * 16) + k * 16 + (m % 16);

        // Read from source
        const fp8e4m3 val = src.data[src_offset];

        // Destination: st_16x128, with swizzle
        const int dst_subtile = m / 16;
        const int dst_row = m % 16;
        const int dst_col = k;
        const int raw_offset = dst_row * 128 + dst_col;
        const int swiz = ((raw_offset >> 8) & 7) << 4;
        const int dst_offset = dst_subtile * (16 * 128) + (raw_offset ^ swiz);

        dst.data[dst_offset] = val;
    }
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void fp8_gemm_crr_kernel(const fp8_crr_globals g, int M_param, int N_param, int K_param) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // Shared memory: At input (st_128x16) + A transposed (st_16x128) + B (st_128x16)
    using ST_At  = st_fp8e4m3<K_STEP, HALF_BLOCK_SIZE, st_128x16_s>;       // At[K, M_half]
    using ST_A   = st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_16x128_s>;       // A[M_half, K] transposed
    using ST_B   = st_fp8e4m3<K_STEP, HALF_BLOCK_SIZE, st_128x16_s>;       // B[K, N_half]

    ST_At (&At_buf)[2] = al.allocate<ST_At, 2>();   // At input: 2 × 16KB = 32KB
    ST_A  (&A_buf)[2]  = al.allocate<ST_A, 2>();    // A transposed: 2 × 16KB = 32KB
    ST_B  (&Bs)[2]     = al.allocate<ST_B, 2>();    // B input: 2 × 16KB = 32KB
    // Total: 96KB < 160KB ✓

    // Register tiles — same as RRR kernel
    rt_fp8e4m3<REG_BLOCK_M, K_STEP> A_tile;                               // [64,128] row_l
    rt<fp8e4m3, K_STEP, REG_BLOCK_N, col_l, rt_128x16_s> B_tile_0;       // [128,32] col_l
    rt<fp8e4m3, K_STEP, REG_BLOCK_N, col_l, rt_128x16_s> B_tile_1;       // [128,32] col_l
    rt_fl<REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);

    auto bm = compute_block_mapping(M_param, N_param, K_param);
    int row = bm.row, col = bm.col;
    int warp_row = bm.warp_row, warp_col = bm.warp_col;
    int num_tiles = bm.num_tiles;

    // Swizzled offsets for G::load
    constexpr int at_bpt = ST_At::underlying_subtile_bytes_per_thread;
    constexpr int at_bpm = at_bpt * NUM_THREADS;
    constexpr int at_mpt = K_STEP * HALF_BLOCK_SIZE * sizeof(fp8e4m3) / at_bpm;
    uint32_t swizzled_offsets_At[at_mpt];
    G::prefill_swizzled_offsets(At_buf[0], g.a, swizzled_offsets_At);

    constexpr int b_bpt = ST_B::underlying_subtile_bytes_per_thread;
    constexpr int b_bpm = b_bpt * NUM_THREADS;
    constexpr int b_mpt = K_STEP * HALF_BLOCK_SIZE * sizeof(fp8e4m3) / b_bpm;
    uint32_t swizzled_offsets_B[b_mpt];
    G::prefill_swizzled_offsets(Bs[0], g.b, swizzled_offsets_B);

    // Swizzled offsets for A row_l loads from st_16x128
    using ST_A_load = ST_A;
    constexpr int a_bpt = ST_A_load::underlying_subtile_bytes_per_thread;
    constexpr int a_bpm = a_bpt * NUM_THREADS;
    constexpr int a_mpt = HALF_BLOCK_SIZE * K_STEP * sizeof(fp8e4m3) / a_bpm;

    // ===== Simple single-buffered loop =====
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load At[K, M] and B[K, N] from global to shared
        G::load(At_buf[0], g.a, {0, 0, tile, row*2},       swizzled_offsets_At);
        G::load(At_buf[1], g.a, {0, 0, tile, row*2 + 1},   swizzled_offsets_At);
        G::load(Bs[0],     g.b, {0, 0, tile, col*2},       swizzled_offsets_B);
        G::load(Bs[1],     g.b, {0, 0, tile, col*2 + 1},   swizzled_offsets_B);

        // Wait for all global loads to complete
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        // Transpose At[K, M_half] → A[M_half, K] in shared memory
        transpose_128x128_fp8(At_buf[0], A_buf[0]);
        transpose_128x128_fp8(At_buf[1], A_buf[1]);
        __builtin_amdgcn_s_barrier();

        // Load from shared to registers — same as RRR
        auto sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(A_buf[0], {warp_row, 0});
        load(A_tile, sta0);

        auto stb0 = subtile_inplace<K_STEP, REG_BLOCK_N>(Bs[0], {0, warp_col});
        load(B_tile_0, stb0);

        auto stb1 = subtile_inplace<K_STEP, REG_BLOCK_N>(Bs[1], {0, warp_col});
        load(B_tile_1, stb1);

        asm volatile("s_waitcnt lgkmcnt(0)");

        // Compute: C[mh=0] += A[mh=0] * B
        mma_AB(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        mma_AB(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);

        // Load A for second M-half
        auto sta1 = subtile_inplace<REG_BLOCK_M, K_STEP>(A_buf[1], {warp_row, 0});
        load(A_tile, sta1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        // Compute: C[mh=1] += A[mh=1] * B
        mma_AB(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_AB(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);

        __builtin_amdgcn_s_barrier();
    }

    // Store output — same as RCR/RRR
    store_output(g.c, C_accum, g.scale, row, col, warp_row, warp_col);
}

void dispatch_fp8_crr(fp8_crr_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)fp8_gemm_crr_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    fp8_gemm_crr_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M, g.N, g.K);
}

// ============================================================
// Pybind11 bindings
// ============================================================

PYBIND11_MODULE(fp8_gemm, m) {
    m.doc() = "FP8 Per-Tensor GEMM kernels (RCR/RRR/CRR) — all native, no transposes";

    py::bind_function<dispatch_fp8_rcr>(m, "rcr",
        &fp8_rcr_globals::a, &fp8_rcr_globals::b, &fp8_rcr_globals::c, &fp8_rcr_globals::scale);

    py::bind_function<dispatch_fp8_rrr>(m, "rrr",
        &fp8_rrr_globals::a, &fp8_rrr_globals::b, &fp8_rrr_globals::c, &fp8_rrr_globals::scale);

    py::bind_function<dispatch_fp8_rrr_4wave>(m, "rrr_4wave",
        &fp8_rrr_globals::a, &fp8_rrr_globals::b, &fp8_rrr_globals::c, &fp8_rrr_globals::scale);

    py::bind_function<dispatch_fp8_crr>(m, "crr",
        &fp8_crr_globals::a, &fp8_crr_globals::b, &fp8_crr_globals::c, &fp8_crr_globals::scale);
}
