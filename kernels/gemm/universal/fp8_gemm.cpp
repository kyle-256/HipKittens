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

// ============================================================
// Helper: Per-thread transposed load from global to shared memory.
//
// Loads a [SRC_ROWS, SRC_COLS] block from global memory and stores it
// transposed as [SRC_COLS, SRC_ROWS] into a shared tile with swizzled layout.
//
// SRC_ROWS = K_STEP = 128 (source rows), SRC_COLS = HALF_BLOCK_SIZE = 128 (source cols)
// Destination tile shape: [HALF_BLOCK_SIZE, K_STEP] = [128, 128]
//
// Global reads are coalesced: consecutive threads read consecutive source columns.
// ============================================================

__device__ void load_transposed_fp8(
    ST_FP8& dst,
    const fp8e4m3* __restrict__ src_ptr,
    int src_stride,     // stride between rows in source (= N for B[K,N], = M for At[K,M])
    int src_row_start,  // starting row in source
    int src_col_start   // starting col in source
) {
    constexpr int DST_ROWS = HALF_BLOCK_SIZE;  // 128
    constexpr int DST_COLS = K_STEP;           // 128
    constexpr int TOTAL = DST_ROWS * DST_COLS; // 16384
    constexpr int SUB_R = ST_FP8::underlying_subtile_rows;     // 16
    constexpr int SUB_C = ST_FP8::underlying_subtile_cols;     // 128
    constexpr int SUB_BYTES = ST_FP8::underlying_subtile_bytes; // 2048
    constexpr int SUBS_PER_ROW = ST_FP8::underlying_subtiles_per_row; // 1

    #pragma unroll
    for (int idx = threadIdx.x; idx < TOTAL; idx += blockDim.x) {
        // Source coordinates — consecutive threads get consecutive src_c (coalesced)
        int src_r = idx / DST_ROWS;   // [0, K_STEP)
        int src_c = idx % DST_ROWS;   // [0, HALF_BLOCK_SIZE) — coalesced

        // Read from global memory: src[src_row_start + src_r, src_col_start + src_c]
        fp8e4m3 val = src_ptr[(src_row_start + src_r) * src_stride + (src_col_start + src_c)];

        // Destination coordinates (transposed): dst[src_c, src_r]
        int dst_r = src_c;  // [0, 128)
        int dst_c = src_r;  // [0, 128)

        // Compute byte offset in shared tile (subtile layout + swizzle)
        int subtile_id = dst_r / SUB_R;   // which 16x128 subtile
        int local_r = dst_r % SUB_R;      // row within subtile
        int local_c = dst_c;              // col within subtile (always < 128 = SUB_C)

        uint32_t byte_off = subtile_id * SUB_BYTES + ST_FP8::swizzle({local_r, local_c});
        dst.data[byte_off] = val;
    }
}

// ============================================================
// Block ID mapping with XCD swizzle (shared by all kernels)
// ============================================================

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
// A[M,K] row-major → G::load as in RCR
// B[K,N] row-major → per-thread transposed load → B^T[N,K] in shared
// Then mma_ABt(A, B^T) = A * (B^T)^T = A * B  ✓
//
// Uses single-buffered shared memory with explicit barriers.
// ============================================================

struct fp8_rrr_globals {
    _gl_fp8 a, b;   // a: [M,K], b: [K,N]
    _gl_out c;
    float scale;
    hipStream_t stream;
    int M = a.rows();
    int N = b.cols();
    int K = a.cols();
    dim3 grid()  { return dim3(ceil_div(N, BLOCK_SIZE) * ceil_div(M, BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void fp8_gemm_rrr_kernel(const fp8_rrr_globals g, int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_16x128_s>;
    using ST_B = st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_16x128_s>;
    // Single-buffered: 2 A halves + 2 B halves = 4 × 16 KB = 64 KB
    ST_A (&As)[2] = al.allocate<ST_A, 2>();
    ST_B (&Bs)[2] = al.allocate<ST_B, 2>();

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

    // Swizzled offsets for A (loaded via G::load)
    constexpr int bytes_per_thread = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = HALF_BLOCK_SIZE * K_STEP * sizeof(fp8e4m3) / bytes_per_memcpy;
    uint32_t so_a[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0], g.a, so_a);

    // B raw pointer for per-thread transposed loads
    const fp8e4m3* b_ptr = g.b.raw_ptr;

    // === Main loop (non-pipelined) ===
    for (int k_tile = 0; k_tile < num_tiles; k_tile++) {
        // Load A[M_half, K_STEP] via G::load (same as RCR)
        G::load(As[0], g.a, {0, 0, row*2, k_tile}, so_a);
        G::load(As[1], g.a, {0, 0, row*2 + 1, k_tile}, so_a);

        // Load B^T[N_half, K_STEP] via per-thread transposed load from B[K, N]
        load_transposed_fp8(Bs[0], b_ptr, N, k_tile * K_STEP, col * 2 * HALF_BLOCK_SIZE);
        load_transposed_fp8(Bs[1], b_ptr, N, k_tile * K_STEP, (col * 2 + 1) * HALF_BLOCK_SIZE);

        // Synchronize: wait for G::load (vmcnt) and per-thread loads (vmcnt + lgkmcnt)
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        // Load from shared to registers and compute
        auto sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[0], {warp_row, 0});
        load(A_tile, sta0);
        auto stb0 = subtile_inplace<REG_BLOCK_N, K_STEP>(Bs[0], {warp_col, 0});
        load(B_tile_0, stb0);
        auto stb1 = subtile_inplace<REG_BLOCK_N, K_STEP>(Bs[1], {warp_col, 0});
        load(B_tile_1, stb1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);

        auto sta1 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[1], {warp_row, 0});
        load(A_tile, sta1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);

        // Barrier before next iteration writes to shared memory
        __builtin_amdgcn_s_barrier();
    }

    store_output(g.c, C_accum, g.scale, row, col, warp_row, warp_col);
}

void dispatch_fp8_rrr(fp8_rrr_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)fp8_gemm_rrr_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    fp8_gemm_rrr_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M, g.N, g.K);
}

// ============================================================
// FP8 Per-Tensor CRR: C[M,N] = At[K,M]^T * B[K,N] * scale
//
// At[K,M] row-major → per-thread transposed load → A[M,K] in shared
// B[K,N]  row-major → per-thread transposed load → B^T[N,K] in shared
// Then mma_ABt(A, B^T) = A * (B^T)^T = A * B = At^T * B  ✓
//
// Uses single-buffered shared memory with explicit barriers.
// ============================================================

struct fp8_crr_globals {
    _gl_fp8 a, b;   // a: At[K,M], b: B[K,N]
    _gl_out c;
    float scale;
    hipStream_t stream;
    int M = a.cols();   // At is [K,M]
    int N = b.cols();   // B is [K,N]
    int K = a.rows();   // At rows = K
    dim3 grid()  { return dim3(ceil_div(N, BLOCK_SIZE) * ceil_div(M, BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void fp8_gemm_crr_kernel(const fp8_crr_globals g, int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_16x128_s>;
    using ST_B = st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_16x128_s>;
    // Single-buffered: 2 A halves + 2 B halves = 4 × 16 KB = 64 KB
    ST_A (&As)[2] = al.allocate<ST_A, 2>();
    ST_B (&Bs)[2] = al.allocate<ST_B, 2>();

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

    // Raw pointers for per-thread transposed loads
    const fp8e4m3* a_ptr = g.a.raw_ptr;  // At[K, M]
    const fp8e4m3* b_ptr = g.b.raw_ptr;  // B[K, N]

    // === Main loop (non-pipelined) ===
    for (int k_tile = 0; k_tile < num_tiles; k_tile++) {
        // Load A[M_half, K_STEP] from At[K, M] via transposed load
        // At[K, M] with stride M → transpose to A[M_half, K_STEP]
        load_transposed_fp8(As[0], a_ptr, M, k_tile * K_STEP, row * 2 * HALF_BLOCK_SIZE);
        load_transposed_fp8(As[1], a_ptr, M, k_tile * K_STEP, (row * 2 + 1) * HALF_BLOCK_SIZE);

        // Load B^T[N_half, K_STEP] from B[K, N] via transposed load
        load_transposed_fp8(Bs[0], b_ptr, N, k_tile * K_STEP, col * 2 * HALF_BLOCK_SIZE);
        load_transposed_fp8(Bs[1], b_ptr, N, k_tile * K_STEP, (col * 2 + 1) * HALF_BLOCK_SIZE);

        // Synchronize: wait for all per-thread loads (vmcnt + lgkmcnt)
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        // Load from shared to registers and compute
        auto sta0 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[0], {warp_row, 0});
        load(A_tile, sta0);
        auto stb0 = subtile_inplace<REG_BLOCK_N, K_STEP>(Bs[0], {warp_col, 0});
        load(B_tile_0, stb0);
        auto stb1 = subtile_inplace<REG_BLOCK_N, K_STEP>(Bs[1], {warp_col, 0});
        load(B_tile_1, stb1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);

        auto sta1 = subtile_inplace<REG_BLOCK_M, K_STEP>(As[1], {warp_row, 0});
        load(A_tile, sta1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);

        // Barrier before next iteration writes to shared memory
        __builtin_amdgcn_s_barrier();
    }

    store_output(g.c, C_accum, g.scale, row, col, warp_row, warp_col);
}

void dispatch_fp8_crr(fp8_crr_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)fp8_gemm_crr_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    fp8_gemm_crr_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M, g.N, g.K);
}

// ############################################################
// 4-WAVE VARIANTS
// ############################################################
//
// 4 warps (WARPS_M=2, WARPS_N=2), 256 threads, occupancy 1
// Each warp handles 64×64 of output (vs 64×32 for 8-wave)
// Fewer warps → less parallelism but more registers per warp.

constexpr int WARPS_M_4W     = 2;
constexpr int WARPS_N_4W     = 2;
constexpr int NUM_WARPS_4W   = WARPS_M_4W * WARPS_N_4W;  // 4
constexpr int NUM_THREADS_4W = kittens::WARP_THREADS * NUM_WARPS_4W;  // 256
constexpr int REG_BLOCK_M_4W = BLOCK_SIZE / WARPS_M_4W / 2;  // 64
constexpr int REG_BLOCK_N_4W = BLOCK_SIZE / WARPS_N_4W / 2;  // 64

using G4 = kittens::group<NUM_WARPS_4W>;

__device__ block_mapping compute_block_mapping_4w(int M, int N, int K) {
    block_mapping bm;
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    const int WGM = 4;
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
    bm.warp_row = warp_id / WARPS_N_4W;  // 0 or 1
    bm.warp_col = warp_id % WARPS_N_4W;  // 0 or 1
    bm.num_tiles = K / K_STEP;
    return bm;
}

__device__ void store_output_4w(const _gl_out& c,
    rt_fl<REG_BLOCK_M_4W, REG_BLOCK_N_4W, col_l, rt_16x16_s> C_accum[2][2],
    float scale, int row, int col, int warp_row, int warp_col) {
    mul(C_accum[0][0], C_accum[0][0], scale);
    mul(C_accum[0][1], C_accum[0][1], scale);
    mul(C_accum[1][0], C_accum[1][0], scale);
    mul(C_accum[1][1], C_accum[1][1], scale);

    store(c, C_accum[0][0], {0, 0, (row * 2) * WARPS_M_4W + warp_row, col * 2 * WARPS_N_4W + warp_col});
    store(c, C_accum[0][1], {0, 0, (row * 2) * WARPS_M_4W + warp_row, col * 2 * WARPS_N_4W + WARPS_N_4W + warp_col});
    store(c, C_accum[1][0], {0, 0, (row * 2) * WARPS_M_4W + WARPS_M_4W + warp_row, col * 2 * WARPS_N_4W + warp_col});
    store(c, C_accum[1][1], {0, 0, (row * 2) * WARPS_M_4W + WARPS_M_4W + warp_row, col * 2 * WARPS_N_4W + WARPS_N_4W + warp_col});
}

// ============================================================
// FP8 4-wave RCR: C[M,N] = A[M,K] * B[N,K]^T * scale
// ============================================================

struct fp8_4w_rcr_globals {
    _gl_fp8 a, b;
    _gl_out c;
    float scale;
    hipStream_t stream;
    int M = a.rows();
    int N = c.cols();
    int K = a.cols();
    dim3 grid()  { return dim3(ceil_div(N, BLOCK_SIZE) * ceil_div(M, BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS_4W); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS_4W, 1)
void fp8_gemm_4w_rcr_kernel(const fp8_4w_rcr_globals g, int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_16x128_s>;
    using ST_B = st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_16x128_s>;
    ST_A (&As)[2][2] = al.allocate<ST_A, 2, 2>();
    ST_B (&Bs)[2][2] = al.allocate<ST_B, 2, 2>();

    rt_fp8e4m3<REG_BLOCK_M_4W, K_STEP> A_tile;
    rt_fp8e4m3<REG_BLOCK_N_4W, K_STEP> B_tile_0, B_tile_1;

    rt_fl<REG_BLOCK_M_4W, REG_BLOCK_N_4W, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);

    auto bm = compute_block_mapping_4w(M, N, K);
    int row = bm.row, col = bm.col;
    int warp_row = bm.warp_row, warp_col = bm.warp_col;
    int num_tiles = bm.num_tiles;

    using T = typename ST_A::dtype;
    constexpr int bytes_per_thread = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS_4W;
    constexpr int memcpy_per_tile = HALF_BLOCK_SIZE * K_STEP * sizeof(T) / bytes_per_memcpy;
    uint32_t so_a[memcpy_per_tile], so_b[memcpy_per_tile];
    G4::prefill_swizzled_offsets(As[0][0], g.a, so_a);
    G4::prefill_swizzled_offsets(Bs[0][0], g.b, so_b);

    int curr = 0, next = 1;

    // === Prologue ===
    G4::load(As[curr][0], g.a, {0, 0, row*2, 0}, so_a);
    G4::load(Bs[curr][0], g.b, {0, 0, col*2, 0}, so_b);
    G4::load(Bs[curr][1], g.b, {0, 0, col*2 + 1, 0}, so_b);
    G4::load(As[curr][1], g.a, {0, 0, row*2 + 1, 0}, so_a);

    G4::load(As[next][0], g.a, {0, 0, row*2, 1}, so_a);
    G4::load(Bs[next][0], g.b, {0, 0, col*2, 1}, so_b);
    G4::load(Bs[next][1], g.b, {0, 0, col*2 + 1, 1}, so_b);
    G4::load(As[next][1], g.a, {0, 0, row*2 + 1, 1}, so_a);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(28)");
    __builtin_amdgcn_s_barrier();

    rt_fp8e4m3<REG_BLOCK_M_4W, K_STEP> a_buf[2];
    rt_fp8e4m3<REG_BLOCK_N_4W, K_STEP> b_buf[2];

    auto a_st0 = subtile_inplace<REG_BLOCK_M_4W, K_STEP>(As[curr][0], {warp_row, 0});
    load(a_buf[0], a_st0);

    asm volatile("s_waitcnt vmcnt(24)");
    __builtin_amdgcn_s_barrier();

    auto b_st0 = subtile_inplace<REG_BLOCK_N_4W, K_STEP>(Bs[curr][0], {warp_col, 0});
    load(b_buf[0], b_st0);

    // === Main loop ===
    #pragma unroll
    for (int k = 0; k < num_tiles - 2; k++, curr ^= 1, next ^= 1) {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        auto bs1 = subtile_inplace<REG_BLOCK_N_4W, K_STEP>(Bs[curr][1], {warp_col, 0});
        G4::load(As[curr][0], g.a, {0, 0, row*2, k + 2}, so_a);
        load(b_buf[1], bs1);
        mma_ABt(C_accum[0][0], a_buf[0], b_buf[0], C_accum[0][0]);

        asm volatile("s_waitcnt lgkmcnt(0)");

        auto as1 = subtile_inplace<REG_BLOCK_M_4W, K_STEP>(As[curr][1], {warp_row, 0});
        G4::load(Bs[curr][0], g.b, {0, 0, col*2, k + 2}, so_b);
        load(a_buf[1], as1);
        mma_ABt(C_accum[0][1], a_buf[0], b_buf[1], C_accum[0][1]);

        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        auto as0 = subtile_inplace<REG_BLOCK_M_4W, K_STEP>(As[next][0], {warp_row, 0});
        G4::load(Bs[curr][1], g.b, {0, 0, col*2 + 1, k + 2}, so_b);
        load(a_buf[0], as0);
        mma_ABt(C_accum[1][0], a_buf[1], b_buf[0], C_accum[1][0]);

        auto bs0 = subtile_inplace<REG_BLOCK_N_4W, K_STEP>(Bs[next][0], {warp_col, 0});
        G4::load(As[curr][1], g.a, {0, 0, row*2 + 1, k + 2}, so_a);
        load(b_buf[0], bs0);
        mma_ABt(C_accum[1][1], a_buf[1], b_buf[1], C_accum[1][1]);
    }

    // === Epilogue k = num_tiles - 2 ===
    {
        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        auto bs1 = subtile_inplace<REG_BLOCK_N_4W, K_STEP>(Bs[curr][1], {warp_col, 0});
        load(b_buf[1], bs1);
        mma_ABt(C_accum[0][0], a_buf[0], b_buf[0], C_accum[0][0]);
        asm volatile("s_waitcnt lgkmcnt(0)");

        auto as1 = subtile_inplace<REG_BLOCK_M_4W, K_STEP>(As[curr][1], {warp_row, 0});
        load(a_buf[1], as1);
        mma_ABt(C_accum[0][1], a_buf[0], b_buf[1], C_accum[0][1]);

        asm volatile("s_waitcnt vmcnt(8)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        auto as0 = subtile_inplace<REG_BLOCK_M_4W, K_STEP>(As[next][0], {warp_row, 0});
        load(a_buf[0], as0);
        mma_ABt(C_accum[1][0], a_buf[1], b_buf[0], C_accum[1][0]);

        auto bs0 = subtile_inplace<REG_BLOCK_N_4W, K_STEP>(Bs[next][0], {warp_col, 0});
        load(b_buf[0], bs0);
        mma_ABt(C_accum[1][1], a_buf[1], b_buf[1], C_accum[1][1]);

        curr ^= 1; next ^= 1;
    }

    // === Epilogue k = num_tiles - 1 ===
    {
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        auto bs1 = subtile_inplace<REG_BLOCK_N_4W, K_STEP>(Bs[curr][1], {warp_col, 0});
        load(b_buf[1], bs1);
        mma_ABt(C_accum[0][0], a_buf[0], b_buf[0], C_accum[0][0]);
        asm volatile("s_waitcnt lgkmcnt(0)");

        auto as1 = subtile_inplace<REG_BLOCK_M_4W, K_STEP>(As[curr][1], {warp_row, 0});
        load(a_buf[1], as1);
        mma_ABt(C_accum[0][1], a_buf[0], b_buf[1], C_accum[0][1]);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[1][0], a_buf[1], b_buf[0], C_accum[1][0]);
        mma_ABt(C_accum[1][1], a_buf[1], b_buf[1], C_accum[1][1]);
    }

    store_output_4w(g.c, C_accum, g.scale, row, col, warp_row, warp_col);
}

void dispatch_fp8_4w_rcr(fp8_4w_rcr_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)fp8_gemm_4w_rcr_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    fp8_gemm_4w_rcr_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M, g.N, g.K);
}

// ============================================================
// FP8 4-wave RRR: C[M,N] = A[M,K] * B[K,N] * scale
// ============================================================

struct fp8_4w_rrr_globals {
    _gl_fp8 a, b;
    _gl_out c;
    float scale;
    hipStream_t stream;
    int M = a.rows();
    int N = b.cols();
    int K = a.cols();
    dim3 grid()  { return dim3(ceil_div(N, BLOCK_SIZE) * ceil_div(M, BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS_4W); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS_4W, 1)
void fp8_gemm_4w_rrr_kernel(const fp8_4w_rrr_globals g, int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_16x128_s>;
    using ST_B = st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_16x128_s>;
    ST_A (&As)[2] = al.allocate<ST_A, 2>();
    ST_B (&Bs)[2] = al.allocate<ST_B, 2>();

    rt_fp8e4m3<REG_BLOCK_M_4W, K_STEP> A_tile;
    rt_fp8e4m3<REG_BLOCK_N_4W, K_STEP> B_tile_0, B_tile_1;

    rt_fl<REG_BLOCK_M_4W, REG_BLOCK_N_4W, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);

    auto bm = compute_block_mapping_4w(M, N, K);
    int row = bm.row, col = bm.col;
    int warp_row = bm.warp_row, warp_col = bm.warp_col;
    int num_tiles = bm.num_tiles;

    constexpr int bytes_per_thread = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS_4W;
    constexpr int memcpy_per_tile = HALF_BLOCK_SIZE * K_STEP * sizeof(fp8e4m3) / bytes_per_memcpy;
    uint32_t so_a[memcpy_per_tile];
    G4::prefill_swizzled_offsets(As[0], g.a, so_a);

    const fp8e4m3* b_ptr = g.b.raw_ptr;

    for (int k_tile = 0; k_tile < num_tiles; k_tile++) {
        G4::load(As[0], g.a, {0, 0, row*2, k_tile}, so_a);
        G4::load(As[1], g.a, {0, 0, row*2 + 1, k_tile}, so_a);
        load_transposed_fp8(Bs[0], b_ptr, N, k_tile * K_STEP, col * 2 * HALF_BLOCK_SIZE);
        load_transposed_fp8(Bs[1], b_ptr, N, k_tile * K_STEP, (col * 2 + 1) * HALF_BLOCK_SIZE);

        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        auto sta0 = subtile_inplace<REG_BLOCK_M_4W, K_STEP>(As[0], {warp_row, 0});
        load(A_tile, sta0);
        auto stb0 = subtile_inplace<REG_BLOCK_N_4W, K_STEP>(Bs[0], {warp_col, 0});
        load(B_tile_0, stb0);
        auto stb1 = subtile_inplace<REG_BLOCK_N_4W, K_STEP>(Bs[1], {warp_col, 0});
        load(B_tile_1, stb1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);

        auto sta1 = subtile_inplace<REG_BLOCK_M_4W, K_STEP>(As[1], {warp_row, 0});
        load(A_tile, sta1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);

        __builtin_amdgcn_s_barrier();
    }

    store_output_4w(g.c, C_accum, g.scale, row, col, warp_row, warp_col);
}

void dispatch_fp8_4w_rrr(fp8_4w_rrr_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)fp8_gemm_4w_rrr_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    fp8_gemm_4w_rrr_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M, g.N, g.K);
}

// ============================================================
// FP8 4-wave CRR: C[M,N] = At[K,M]^T * B[K,N] * scale
// ============================================================

struct fp8_4w_crr_globals {
    _gl_fp8 a, b;
    _gl_out c;
    float scale;
    hipStream_t stream;
    int M = a.cols();
    int N = b.cols();
    int K = a.rows();
    dim3 grid()  { return dim3(ceil_div(N, BLOCK_SIZE) * ceil_div(M, BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS_4W); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS_4W, 1)
void fp8_gemm_4w_crr_kernel(const fp8_4w_crr_globals g, int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_16x128_s>;
    using ST_B = st_fp8e4m3<HALF_BLOCK_SIZE, K_STEP, st_16x128_s>;
    ST_A (&As)[2] = al.allocate<ST_A, 2>();
    ST_B (&Bs)[2] = al.allocate<ST_B, 2>();

    rt_fp8e4m3<REG_BLOCK_M_4W, K_STEP> A_tile;
    rt_fp8e4m3<REG_BLOCK_N_4W, K_STEP> B_tile_0, B_tile_1;

    rt_fl<REG_BLOCK_M_4W, REG_BLOCK_N_4W, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]); zero(C_accum[0][1]);
    zero(C_accum[1][0]); zero(C_accum[1][1]);

    auto bm = compute_block_mapping_4w(M, N, K);
    int row = bm.row, col = bm.col;
    int warp_row = bm.warp_row, warp_col = bm.warp_col;
    int num_tiles = bm.num_tiles;

    const fp8e4m3* a_ptr = g.a.raw_ptr;
    const fp8e4m3* b_ptr = g.b.raw_ptr;

    for (int k_tile = 0; k_tile < num_tiles; k_tile++) {
        load_transposed_fp8(As[0], a_ptr, M, k_tile * K_STEP, row * 2 * HALF_BLOCK_SIZE);
        load_transposed_fp8(As[1], a_ptr, M, k_tile * K_STEP, (row * 2 + 1) * HALF_BLOCK_SIZE);
        load_transposed_fp8(Bs[0], b_ptr, N, k_tile * K_STEP, col * 2 * HALF_BLOCK_SIZE);
        load_transposed_fp8(Bs[1], b_ptr, N, k_tile * K_STEP, (col * 2 + 1) * HALF_BLOCK_SIZE);

        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        auto sta0 = subtile_inplace<REG_BLOCK_M_4W, K_STEP>(As[0], {warp_row, 0});
        load(A_tile, sta0);
        auto stb0 = subtile_inplace<REG_BLOCK_N_4W, K_STEP>(Bs[0], {warp_col, 0});
        load(B_tile_0, stb0);
        auto stb1 = subtile_inplace<REG_BLOCK_N_4W, K_STEP>(Bs[1], {warp_col, 0});
        load(B_tile_1, stb1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        mma_ABt(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);

        auto sta1 = subtile_inplace<REG_BLOCK_M_4W, K_STEP>(As[1], {warp_row, 0});
        load(A_tile, sta1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);

        __builtin_amdgcn_s_barrier();
    }

    store_output_4w(g.c, C_accum, g.scale, row, col, warp_row, warp_col);
}

void dispatch_fp8_4w_crr(fp8_4w_crr_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)fp8_gemm_4w_crr_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    fp8_gemm_4w_crr_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M, g.N, g.K);
}

// ============================================================
// Pybind11 bindings
// ============================================================

PYBIND11_MODULE(fp8_gemm, m) {
    m.doc() = "FP8 per-tensor GEMM kernels (RCR/RRR/CRR layouts, 8-wave and 4-wave)";

    // 8-wave (512 threads, 8 warps, occupancy 2)
    py::bind_function<dispatch_fp8_rcr>(m, "rcr",
        &fp8_rcr_globals::a, &fp8_rcr_globals::b, &fp8_rcr_globals::c, &fp8_rcr_globals::scale);
    py::bind_function<dispatch_fp8_rrr>(m, "rrr",
        &fp8_rrr_globals::a, &fp8_rrr_globals::b, &fp8_rrr_globals::c, &fp8_rrr_globals::scale);
    py::bind_function<dispatch_fp8_crr>(m, "crr",
        &fp8_crr_globals::a, &fp8_crr_globals::b, &fp8_crr_globals::c, &fp8_crr_globals::scale);

    // 4-wave (256 threads, 4 warps, occupancy 1)
    py::bind_function<dispatch_fp8_4w_rcr>(m, "rcr_4w",
        &fp8_4w_rcr_globals::a, &fp8_4w_rcr_globals::b, &fp8_4w_rcr_globals::c, &fp8_4w_rcr_globals::scale);
    py::bind_function<dispatch_fp8_4w_rrr>(m, "rrr_4w",
        &fp8_4w_rrr_globals::a, &fp8_4w_rrr_globals::b, &fp8_4w_rrr_globals::c, &fp8_4w_rrr_globals::scale);
    py::bind_function<dispatch_fp8_4w_crr>(m, "crr_4w",
        &fp8_4w_crr_globals::a, &fp8_4w_crr_globals::b, &fp8_4w_crr_globals::c, &fp8_4w_crr_globals::scale);
}
