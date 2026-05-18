// FP8 blockwise GEMM (RCR/NT fwd) — MI300X — v2: 256×128×128 + full pipeline.
//
// Layout: C[M,N] = sum_ki (A[M,k] * B[N,k]^T) * a_scale[m, ki] * b_scale[n_block, ki]
//   A[M,K], B[N,K] FP8 e4m3fnuz row-major
//   A_scale[Kb, M] fp32, B_scale[Kb, Nb] fp32 (Nb = N/128)
//   C[M,N] bf16
//
// Tile: BM=256, BN=128 (= 1 scale block!), BK=128 (= 1 scale block).
// 8 warps (2 row × 4 col). Each warp covers 128M × 32N (top 64 + bot 64 in M).
// Per warp: 2 partial acc (top/bot) + 2 main acc, all rt_fl<64,32,col> (32 fp32/thread).
// Each warp uses ONE b_scale per K_TILE (since BN=128 = 1 scale block).
// Drain partials → main with scale at end of every K_TILE.
// Pipeline: PR #52's 8-cluster software pipelining with reg-buffered prefetch.

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#ifndef NUM_WARPS
#define NUM_WARPS 8
#endif
#ifndef BLOCK_M
#define BLOCK_M 256
#endif
#ifndef BLOCK_N
#define BLOCK_N 128
#endif
#ifndef BLOCK_K
#define BLOCK_K 128
#endif

// Module name override — Makefile `tuned` target sets TK_MODULE_NAME so each
// per-shape build produces a uniquely-named .so that test_python.py can pick
// without rebuilding the default tk_kernel module.
#ifndef TK_MODULE_NAME
#define TK_MODULE_NAME tk_kernel
#endif

#ifndef CHIPLET_CHUNK
#define CHIPLET_CHUNK 4
#endif

#ifndef REG_M_BUILD
#define REG_M_BUILD 64
#endif
constexpr int REG_M  =  REG_M_BUILD;
constexpr int REG_N  =  32;
constexpr int REG_K  =  32;

constexpr int PARTIAL_N_HEIGHT  = REG_M / 32;
constexpr int SCALE_STRIDE_HALF = PARTIAL_N_HEIGHT * 4;

constexpr int WARP_COLS         = BLOCK_N / REG_N;        // 4 default
constexpr int WARP_ROWS         = NUM_WARPS / WARP_COLS;  // 2 default
constexpr int M_TILES_PER_BLOCK = BLOCK_M / REG_M;        // 4 default
constexpr int N_TILES_PER_BLOCK = BLOCK_N / REG_N;        // 4 default

static_assert(NUM_WARPS == WARP_ROWS * WARP_COLS,
              "NUM_WARPS must equal WARP_ROWS * WARP_COLS");
static_assert(BLOCK_M == 2 * WARP_ROWS * REG_M,
              "BLOCK_M must equal 2 * WARP_ROWS * REG_M (top+bot per warp)");
static_assert(BLOCK_N == WARP_COLS * REG_N,
              "BLOCK_N must equal WARP_COLS * REG_N");

using G = kittens::group<NUM_WARPS>;
using _gl_A = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_B = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_C = gl<bf16,    -1, -1, -1, -1>;
using _gl_S = gl<float,   -1, -1, -1, -1>;

// Grouped fwd: A is [M_total, K], B is [G, N, K], scales mirror layout.
// group_offs[G+1] is int32 prefix-sum of per-group M lengths. Persistent
// kernel: launches NUM_CUs WGs and each WG iterates over its assigned
// global tile id (walking group_offs to map tile_id → (group, pid_m, pid_n)).
using _gl_O = gl<int32_t, -1, -1, -1, -1>;

struct micro_globals {
    _gl_A A;          // [1, 1, M_total, K]
    _gl_B B;          // [1, G, N, K]
    _gl_C C;          // [1, 1, M_total, N]
    _gl_S A_scale;    // [1, 1, Kb, M_total]
    _gl_S B_scale;    // [1, G, Kb, Nb]
    _gl_O group_offs; // [1, 1, 1, G+1]   prefix sum of M lengths
    _gl_O cum_tiles;  // [1, 1, 1, G+1]   prefix sum of (M_g/BM)*num_pid_n
    int   num_groups;
    int   total_tiles;
    hipStream_t stream;
    // Persistent kernel: 1 WG per CU (304 on MI300X).
    dim3 grid()  { return dim3(304); }
    dim3 block() { return dim3(NUM_WARPS * WARP_THREADS); }
    size_t dynamic_shared_memory() {
        // 32K (As) + 16K (Bs) + 0.5K (offs+cum_tiles_lds, 64 ints each) = 48.5K.
        return (size_t)BLOCK_M * BLOCK_K * sizeof(fp8e4m3)
             + (size_t)BLOCK_N * BLOCK_K * sizeof(fp8e4m3)
             + (size_t)64 * sizeof(int) * 2
             ;
    }
};

#define MMA_HALF(d, a, b, c, off) do {                                              \
    _Pragma("unroll")                                                                \
    for (int _hn = 0; _hn < (d).height; ++_hn) {                                     \
        _Pragma("unroll")                                                            \
        for (int _hm = 0; _hm < (d).width; ++_hm) {                                  \
            ::kittens::mma_ABt_base((d).tiles[_hn][_hm],                             \
                                    (a).tiles[_hn + (off)][0],                       \
                                    (b).tiles[_hm][0],                               \
                                    (c).tiles[_hn][_hm]);                            \
        }                                                                            \
    }                                                                                \
} while (0)

#define MMA_QUAD_PRIO(a_top, a_bot, bt) do {                                        \
    __builtin_amdgcn_s_setprio(1);                                                  \
    MMA_HALF(partial[0], a_tiles[a_top], b_tiles[bt], partial[0], 0);               \
    MMA_HALF(partial[1], a_tiles[a_top], b_tiles[bt], partial[1], (REG_M/32));      \
    MMA_HALF(partial[2], a_tiles[a_bot], b_tiles[bt], partial[2], 0);               \
    MMA_HALF(partial[3], a_tiles[a_bot], b_tiles[bt], partial[3], (REG_M/32));      \
    __builtin_amdgcn_s_setprio(0);                                                  \
} while (0)

// SCALE_DRAIN_HALF: drains a 2-M-row partial<32,32,col> into a 2-M-row slice
// of a 4-M-row C_accum<64,32,col>. dst_off picks the slice (0 = top two
// sub-tiles tiles[0..1][:], 2 = bot two sub-tiles tiles[2..3][:]). sv_ptr
// points at 8 floats (2 sub-tiles × 4 fp32-per-sub-tile-per-thread).
#define SCALE_DRAIN_HALF(dst, src, sv_ptr, bs, dst_off) do {                        \
    _Pragma("unroll")                                                                \
    for (int _n = 0; _n < (src).height; ++_n) {                                      \
        const float _s0 = (sv_ptr)[_n*4]   * (bs);                                   \
        const float _s1 = (sv_ptr)[_n*4+1] * (bs);                                   \
        const float _s2 = (sv_ptr)[_n*4+2] * (bs);                                   \
        const float _s3 = (sv_ptr)[_n*4+3] * (bs);                                   \
        _Pragma("unroll")                                                            \
        for (int _m = 0; _m < 2; ++_m) {                                             \
            float *_df = reinterpret_cast<float*>((dst).tiles[_n+(dst_off)][_m].data);\
            float *_sf = reinterpret_cast<float*>((src).tiles[_n][_m].data);         \
            _df[0] += _sf[0] * _s0;  _df[1] += _sf[1] * _s1;                         \
            _df[2] += _sf[2] * _s2;  _df[3] += _sf[3] * _s3;                         \
            _sf[0] = _sf[1] = _sf[2] = _sf[3] = 0.f;                                 \
        }                                                                            \
    }                                                                                \
} while(0)

__global__ __launch_bounds__(NUM_WARPS * WARP_THREADS, 1)
void micro_tk(const micro_globals g) {
    constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    auto (&As) = al.allocate<st<fp8e4m3, BLOCK_M, BLOCK_K>>();   // 256×128 = 32K
    auto (&Bs) = al.allocate<st<fp8e4m3, BLOCK_N, BLOCK_K>>();   // 128×128 = 16K

    rt<fp8e4m3, REG_M, REG_K> a_tiles[6];
    rt<fp8e4m3, REG_N, REG_K> b_tiles[3];

    rt_fl<REG_M, REG_N, ducks::rt_layout::col> C_accum[2];
    rt_fl<REG_M/2, REG_N, ducks::rt_layout::col> partial[4];
    for (int i = 0; i < 2; i++) zero(C_accum[i]);
    for (int i = 0; i < 4; i++) zero(partial[i]);

    const int NUM_WGS  = gridDim.x;
    const int N_runtime = g.B.rows();   // B is [G, N, K], rows()=N
    const int num_pid_n = ceil_div(N_runtime, BLOCK_N);

    const int warp_id = warpid();
    const int warp_row = warp_id / WARP_COLS, warp_col = warp_id % WARP_COLS;

    for (int tile_raw = blockIdx.x; tile_raw < g.total_tiles; tile_raw += NUM_WGS) {
        const int tile_id = chiplet_transform_chunked(tile_raw, g.total_tiles, NUM_XCDS, CHIPLET_CHUNK);
        int group_idx = 0;
        #pragma unroll 1
        for (int gi = 0; gi < g.num_groups; ++gi) {
            if (tile_id >= g.cum_tiles[coord<>(gi+1)]) group_idx = gi + 1;
        }
        const int tile_start = g.cum_tiles[coord<>(group_idx)];
        const int m_start_g  = g.group_offs[coord<>(group_idx)];
        const int local_tile = tile_id - tile_start;
        // n-fast traversal within a group: pid_n iterates fastest (matches
        // single-GEMM convention; preserves A-tile L2 reuse across n-tiles).
        const int pid_m_local = local_tile / num_pid_n;
        const int pid_n       = local_tile % num_pid_n;
        const int output_m = (m_start_g / BLOCK_M) + pid_m_local;
        const int output_n = pid_n;

        // Reset C_accum per tile. partial[i] is left zeroed by SCALE_DRAIN_HALF
        // at end of each K-loop, so the per-tile reset is only needed for the
        // first tile (handled by the zero() at kernel entry above).
        for (int i = 0; i < 2; i++) zero(C_accum[i]);
    const int k_iters = g.A.cols() / BLOCK_K;

    const int mt_base = output_m * BLOCK_M + warp_row * REG_M;
    const int mb_base = output_m * BLOCK_M + (BLOCK_M/2) + warp_row * REG_M;
    const int b_scale_idx = (output_n * BLOCK_N + warp_col * REG_N) / 128;
    const int r16 = 4 * (laneid() / 16);

    constexpr int BUFFER_SIZE_A = (BLOCK_M * BLOCK_K) / NUM_THREADS / sizeof(float4) / sizeof(fp8e4m3);
    constexpr int BUFFER_SIZE_B = (BLOCK_N * BLOCK_K) / NUM_THREADS / sizeof(float4) / sizeof(fp8e4m3);

    G::load(As, g.A, {0, 0, output_m, 0});
    G::load(Bs, g.B, {0, group_idx, output_n, 0});
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    for (int K_TILE = 0; K_TILE < k_iters - 1; ++K_TILE) {
      {
        float4 a_buffer_next[BUFFER_SIZE_A];
        float4 b_buffer_next[BUFFER_SIZE_B];

        // Cluster 0: A prefetch + load sub-K=0 tiles
        load_global_to_register_buffer<2, false, NUM_THREADS>(
            a_buffer_next, BUFFER_SIZE_A, g.A, {0, 0, output_m, K_TILE + 1}, As);
        load(a_tiles[1], subtile_inplace<REG_M, REG_K>(As, {warp_row, 0}));
        load(a_tiles[2], subtile_inplace<REG_M, REG_K>(As, {warp_row + WARP_ROWS, 0}));
        load(b_tiles[0], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 0}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1: sub-K=0 MFMAs
        asm volatile("s_waitcnt lgkmcnt(0)");
        MMA_QUAD_PRIO(1, 2, 0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2: load sub-K=1 + start of sub-K=2
        load(b_tiles[1], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 1}));
        load(a_tiles[4], subtile_inplace<REG_M, REG_K>(As, {warp_row, 1}));
        load(a_tiles[5], subtile_inplace<REG_M, REG_K>(As, {warp_row + WARP_ROWS, 1}));
        load(b_tiles[0], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 2}));
        load(a_tiles[1], subtile_inplace<REG_M, REG_K>(As, {warp_row, 2}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3: sub-K=1 MFMAs
        asm volatile("s_waitcnt lgkmcnt(0)");
        MMA_QUAD_PRIO(4, 5, 1);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4: B prefetch + load sub-K=3
        load_global_to_register_buffer<2, false, NUM_THREADS>(
            b_buffer_next, BUFFER_SIZE_B, g.B, {0, group_idx, output_n, K_TILE + 1}, Bs);
        load(a_tiles[2], subtile_inplace<REG_M, REG_K>(As, {warp_row + WARP_ROWS, 2}));
        load(b_tiles[2], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 3}));
        load(a_tiles[0], subtile_inplace<REG_M, REG_K>(As, {warp_row, 3}));
        load(a_tiles[5], subtile_inplace<REG_M, REG_K>(As, {warp_row + WARP_ROWS, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        MMA_QUAD_PRIO(1, 2, 0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6: commit prefetches → LDS
        asm volatile("s_waitcnt vmcnt(0)");
        store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
        store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        const float b_s = g.B_scale[coord<>(0, group_idx, K_TILE, b_scale_idx)];
        float svt[16], svb[16];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                svt[i*4+d] = g.A_scale[coord<>(K_TILE, mt_base + i*16 + r16 + d)];
                svb[i*4+d] = g.A_scale[coord<>(K_TILE, mb_base + i*16 + r16 + d)];
            }
        }

        MMA_QUAD_PRIO(0, 5, 2);

        SCALE_DRAIN_HALF(C_accum[0], partial[0], svt + 0,                 b_s, 0);
        SCALE_DRAIN_HALF(C_accum[0], partial[1], svt + SCALE_STRIDE_HALF, b_s, (REG_M/32));
        SCALE_DRAIN_HALF(C_accum[1], partial[2], svb + 0,                 b_s, 0);
        SCALE_DRAIN_HALF(C_accum[1], partial[3], svb + SCALE_STRIDE_HALF, b_s, (REG_M/32));
      }
    }

    // Epilogue: final K_TILE
    {
        const int K_TILE = k_iters - 1;
        const float b_s = g.B_scale[coord<>(0, group_idx, K_TILE, b_scale_idx)];
        float svt[16], svb[16];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                svt[i*4+d] = g.A_scale[coord<>(K_TILE, mt_base + i*16 + r16 + d)];
                svb[i*4+d] = g.A_scale[coord<>(K_TILE, mb_base + i*16 + r16 + d)];
            }
        }

        __builtin_amdgcn_sched_barrier(0);
        load(b_tiles[0], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 0}));
        load(a_tiles[1], subtile_inplace<REG_M, REG_K>(As, {warp_row, 0}));
        load(a_tiles[2], subtile_inplace<REG_M, REG_K>(As, {warp_row + WARP_ROWS, 0}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        MMA_QUAD_PRIO(1, 2, 0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load(b_tiles[1], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 1}));
        load(a_tiles[4], subtile_inplace<REG_M, REG_K>(As, {warp_row, 1}));
        load(a_tiles[5], subtile_inplace<REG_M, REG_K>(As, {warp_row + WARP_ROWS, 1}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        MMA_QUAD_PRIO(4, 5, 1);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load(b_tiles[0], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 2}));
        load(a_tiles[1], subtile_inplace<REG_M, REG_K>(As, {warp_row, 2}));
        load(a_tiles[2], subtile_inplace<REG_M, REG_K>(As, {warp_row + WARP_ROWS, 2}));
        load(b_tiles[2], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 3}));
        load(a_tiles[4], subtile_inplace<REG_M, REG_K>(As, {warp_row, 3}));
        load(a_tiles[5], subtile_inplace<REG_M, REG_K>(As, {warp_row + WARP_ROWS, 3}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        MMA_QUAD_PRIO(1, 2, 0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        MMA_QUAD_PRIO(4, 5, 2);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        SCALE_DRAIN_HALF(C_accum[0], partial[0], svt + 0,                 b_s, 0);
        SCALE_DRAIN_HALF(C_accum[0], partial[1], svt + SCALE_STRIDE_HALF, b_s, (REG_M/32));
        SCALE_DRAIN_HALF(C_accum[1], partial[2], svb + 0,                 b_s, 0);
        SCALE_DRAIN_HALF(C_accum[1], partial[3], svb + SCALE_STRIDE_HALF, b_s, (REG_M/32));
    }

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }
    store(g.C, C_accum[0], {0, 0, output_m * M_TILES_PER_BLOCK + warp_row,             output_n * N_TILES_PER_BLOCK + warp_col});
    store(g.C, C_accum[1], {0, 0, output_m * M_TILES_PER_BLOCK + warp_row + WARP_ROWS, output_n * N_TILES_PER_BLOCK + warp_col});
    }   // end persistent tile loop
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

// See blockwise_8192/blockwise.cpp header for the PRIMUS_TURBO_HK_INTEGRATION contract.
#ifndef PRIMUS_TURBO_HK_INTEGRATION
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "FP8 blockwise grouped GEMM (fwd) — MI300X gfx942 (persistent)";
    py::bind_function<dispatch_micro>(m, "dispatch_grouped",
        &micro_globals::A, &micro_globals::B, &micro_globals::C,
        &micro_globals::A_scale, &micro_globals::B_scale,
        &micro_globals::group_offs, &micro_globals::cum_tiles,
        &micro_globals::num_groups, &micro_globals::total_tiles);
}
#endif  // PRIMUS_TURBO_HK_INTEGRATION
