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
// Grouped WGRAD (CRR/TN via K-contig col-T inputs, persistent).
// Per group g: dW[g, :, :] = sum_{m in [m_start_g, m_end_g)} dY[m,n] * X[m,k]
//   = g_T[:, m_start_g:m_end_g] @ a_T[:, m_start_g:m_end_g]^T
//
// Per-tile k_iters varies (= M_g / BLOCK_K). Persistent loop maps
//   tile_id → (group_idx, n_block, k_block) via fixed tile count per group
//   (= (N/BM_fwd) * (K/BN_fwd), uniform across groups since N,K fixed).
//
// Reuses the polymorphic micro_tk<true> body (per-element B_scale + WGRAD_DRAIN).
using _gl_O = gl<int32_t, -1, -1, -1, -1>;

struct micro_globals {
    _gl_A A;          // [1, 1, N, M_total]   g_T = dY^T (kernel A)
    _gl_B B;          // [1, 1, K, M_total]   a_T = X^T  (kernel B)
    _gl_C C;          // [1, G, N, K]         dW
    _gl_S A_scale;    // [1, 1, Mb, N]        per-N-element (g_scale_kc)
    _gl_S B_scale;    // [1, 1, Mb, K]        per-K-element (a_scale_kc)
    _gl_O group_offs; // [1, 1, 1, G+1]       prefix sum of M lengths
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

// WGRAD_DRAIN_HALF: drains a 2-M-row partial<32,32,col> into a 2-M-row slice
// of a 4-M-row dst<64,32,col> with per-K-col b_scale (1×1×128 wgrad pattern).
//   sv_ptr: per-N-row a_scale (4 floats per sub-tile per thread)
//   bs_arr[2]: per-K-sub-tile b_scale (one per K-col-group, distinct per lane)
#define WGRAD_DRAIN_HALF(dst, src, sv_ptr, bs_arr, dst_off) do {                    \
    _Pragma("unroll")                                                                \
    for (int _n = 0; _n < (src).height; ++_n) {                                      \
        _Pragma("unroll")                                                            \
        for (int _m = 0; _m < 2; ++_m) {                                             \
            const float _bs = (bs_arr)[_m];                                          \
            const float _s0 = (sv_ptr)[_n*4]   * _bs;                                \
            const float _s1 = (sv_ptr)[_n*4+1] * _bs;                                \
            const float _s2 = (sv_ptr)[_n*4+2] * _bs;                                \
            const float _s3 = (sv_ptr)[_n*4+3] * _bs;                                \
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
    // Wgrad-specific: A=[N, M_total], so g.A.rows()=N and g.B.rows()=K.
    const int N_runtime = g.A.rows();
    const int K_runtime = g.B.rows();
    const int n_tiles_per_g = N_runtime / BLOCK_M;   // wgrad N → fwd's M-axis
    const int k_tiles_per_g = K_runtime / BLOCK_N;   // wgrad K → fwd's N-axis
    const int tiles_per_group = n_tiles_per_g * k_tiles_per_g;
    // Reuse num_pid_n alias (used by macros below) for k_tiles_per_g.
    const int num_pid_n = k_tiles_per_g;

    const int warp_id = warpid();
    const int warp_row = warp_id / WARP_COLS, warp_col = warp_id % WARP_COLS;

    // Persistent loop: tile_id → (group_idx, n_block, k_block).
    // Wgrad has FIXED tiles per group (since N, K constant); only k_iters varies.
    for (int tile_raw = blockIdx.x; tile_raw < g.total_tiles; tile_raw += NUM_WGS) {
        const int tile_id = chiplet_transform_chunked(tile_raw, g.total_tiles, NUM_XCDS, 8);
        const int group_idx   = tile_id / tiles_per_group;
        const int local_tile  = tile_id % tiles_per_group;
        const int pid_m_local = local_tile / num_pid_n;   // n_block within group
        const int pid_n       = local_tile % num_pid_n;   // k_block within group
        const int m_start_g   = g.group_offs[coord<>(group_idx)];
        const int M_g_local   = g.group_offs[coord<>(group_idx+1)] - m_start_g;
        // wgrad-specific K_TILE offset base (in BLOCK_K-units along the
        // concatenated reduction axis A[:, m_start_g:..]).
        const int k_offset_base = m_start_g / BLOCK_K;
        // For wgrad, output_m corresponds to wgrad's n_block (in BM_fwd units),
        // output_n corresponds to wgrad's k_block (in BN_fwd units).
        const int output_m = pid_m_local;
        const int output_n = pid_n;

        for (int i = 0; i < 2; i++) zero(C_accum[i]);
    // Per-tile k_iters varies for wgrad: M_g / BLOCK_K (NOT g.A.cols()/BK
    // which would be M_total/BK summed across all groups).
    const int k_iters = M_g_local / BLOCK_K;

    const int mt_base = output_m * BLOCK_M + warp_row * REG_M;
    const int mb_base = output_m * BLOCK_M + (BLOCK_M/2) + warp_row * REG_M;
    // Per-element b_scale anchors: col_l layout, each lane owns 1 N-col within
    // a 16×16 sub-tile at offset c16 = laneid()%16. 32-col warp × 2 K-sub-tiles
    // = 2 b_scale per thread.
    const int kt_anchor = output_n * BLOCK_N + warp_col * REG_N;
    const int c16 = laneid() % 16;
    const int r16 = 4 * (laneid() / 16);

    constexpr int BUFFER_SIZE_A = (BLOCK_M * BLOCK_K) / NUM_THREADS / sizeof(float4) / sizeof(fp8e4m3);
    constexpr int BUFFER_SIZE_B = (BLOCK_N * BLOCK_K) / NUM_THREADS / sizeof(float4) / sizeof(fp8e4m3);

    G::load(As, g.A, {0, 0, output_m, k_offset_base});
    G::load(Bs, g.B, {0, 0, output_n, k_offset_base});
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
            a_buffer_next, BUFFER_SIZE_A, g.A, {0, 0, output_m, K_TILE + 1 + k_offset_base}, As);
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
            b_buffer_next, BUFFER_SIZE_B, g.B, {0, 0, output_n, K_TILE + 1 + k_offset_base}, Bs);
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

        float b_s_per[2];
        b_s_per[0] = g.B_scale[coord<>(K_TILE + k_offset_base, kt_anchor + 0*16 + c16)];
        b_s_per[1] = g.B_scale[coord<>(K_TILE + k_offset_base, kt_anchor + 1*16 + c16)];
        float svt[16], svb[16];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                svt[i*4+d] = g.A_scale[coord<>(K_TILE + k_offset_base, mt_base + i*16 + r16 + d)];
                svb[i*4+d] = g.A_scale[coord<>(K_TILE + k_offset_base, mb_base + i*16 + r16 + d)];
            }
        }

        MMA_QUAD_PRIO(0, 5, 2);

        WGRAD_DRAIN_HALF(C_accum[0], partial[0], svt + 0,                 b_s_per, 0);
        WGRAD_DRAIN_HALF(C_accum[0], partial[1], svt + SCALE_STRIDE_HALF, b_s_per, (REG_M/32));
        WGRAD_DRAIN_HALF(C_accum[1], partial[2], svb + 0,                 b_s_per, 0);
        WGRAD_DRAIN_HALF(C_accum[1], partial[3], svb + SCALE_STRIDE_HALF, b_s_per, (REG_M/32));
      }
    }

    // Epilogue: final K_TILE
    {
        const int K_TILE = k_iters - 1;
        float b_s_per[2];
        b_s_per[0] = g.B_scale[coord<>(K_TILE + k_offset_base, kt_anchor + 0*16 + c16)];
        b_s_per[1] = g.B_scale[coord<>(K_TILE + k_offset_base, kt_anchor + 1*16 + c16)];
        float svt[16], svb[16];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                svt[i*4+d] = g.A_scale[coord<>(K_TILE + k_offset_base, mt_base + i*16 + r16 + d)];
                svb[i*4+d] = g.A_scale[coord<>(K_TILE + k_offset_base, mb_base + i*16 + r16 + d)];
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

        WGRAD_DRAIN_HALF(C_accum[0], partial[0], svt + 0,                 b_s_per, 0);
        WGRAD_DRAIN_HALF(C_accum[0], partial[1], svt + SCALE_STRIDE_HALF, b_s_per, (REG_M/32));
        WGRAD_DRAIN_HALF(C_accum[1], partial[2], svb + 0,                 b_s_per, 0);
        WGRAD_DRAIN_HALF(C_accum[1], partial[3], svb + SCALE_STRIDE_HALF, b_s_per, (REG_M/32));
    }

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }
    store(g.C, C_accum[0], {0, group_idx, output_m * M_TILES_PER_BLOCK + warp_row,             output_n * N_TILES_PER_BLOCK + warp_col});
    store(g.C, C_accum[1], {0, group_idx, output_m * M_TILES_PER_BLOCK + warp_row + WARP_ROWS, output_n * N_TILES_PER_BLOCK + warp_col});
    }   // end persistent tile loop
}

void dispatch_grouped_wgrad(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "FP8 blockwise grouped WGRAD (CRR via K-contig col-T) — MI300X gfx942";
    py::bind_function<dispatch_grouped_wgrad>(m, "dispatch_grouped_wgrad",
        &micro_globals::A, &micro_globals::B, &micro_globals::C,
        &micro_globals::A_scale, &micro_globals::B_scale,
        &micro_globals::group_offs,
        &micro_globals::num_groups, &micro_globals::total_tiles);
}
