// FP8 blockwise GEMM (fwd/dgrad/wgrad) — MI300X (gfx942).
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
//
// Section routing:
//   fwd   (RCR/NT) : C[M,N]  = A[M,K] @ B[N,K]^T              — micro_tk<false>
//   dgrad (RRR/NN) : C[M,K]  = A[M,N] @ B[N,K]                — micro_tk<false> with
//                    caller-side B.T.contiguous() pre-transpose
//   wgrad (CRR/TN) : C[N,K]  = g_T[N,M] @ a_T[K,M]^T          — micro_tk<true>
//                    Caller must supply K-contig (col-T) FP8 inputs g_T and a_T
//                    (g_T plays kernel A, a_T plays kernel B). Per-element
//                    b_scale (one scale per K-col, distinct per lane) instead
//                    of fwd's per-128-block scalar. The reduction-axis-first
//                    scale layout (B_scale [Mb, K], A_scale [Mb, N]) must
//                    match — see test_python.py wgrad branch.
//
//                    **wgrad numerics caveat**: if the caller's bf16 source
//                    tensors (dY, X) were originally quantized to row-major
//                    FP8 with a per-K-block scale layout (e.g. for fwd), they
//                    must be RE-quantized along M (per-M-block scales) to
//                    feed wgrad. That second quantization is the caller's
//                    responsibility; the kernel sees fp8 inputs and trusts
//                    them. Empirically this round-trip costs ~0.5-1 dB SNR
//                    vs a dedicated wgrad kernel — visible mostly at small M
//                    where the per-element error doesn't average out across
//                    the reduction axis. Default SNR gate (49 dB) catches
//                    egregious cases; tighten to 50 dB if your application
//                    is more sensitive.
//
// FALSIFIED directions (don't re-try without new evidence):
//   - KBPT=2 unpipelined (`BW_KBPT2_A`): -2.26 dB SNR, structural correctness
//     break (round-3 / round-99 close-out).
//   - KBPT=2 INTERLEAVED: correctness OK (49.59 dB) but perf 83-85% Tri on
//     K=7168 fwd vs KBPT=1 baseline 95%. NB: when bringing this back, the
//     kittens `st<fp8, *, 256>` swizzle path has a structural bug — see
//     workaround in round-120 (TWO 128-wide tile pairs). Upstream issue TBD.
//   - BLOCK_K=64: K=7168 fwd 54-56% Tri vs current 95%.
//   - WGM > 1, MMA_QUAD_PRIO_RESET, BW_HOIST_BS, persistent kernel (VGPR
//     spill cliff): all FALSIFIED across 130+ daemon rounds.
//   See scripts/_goal_blockwise_fp8.md for the full exhausted-knobs table
//   and the remaining 4 multi-week structural attacks.

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

#ifndef BW_RAW_DRAIN
#define BW_RAW_DRAIN 0
#endif

#ifndef BW_PRESCALE_BS
#define BW_PRESCALE_BS 0
#endif

#ifndef BW_CHIPLET_CHUNK
#define BW_CHIPLET_CHUNK 1
#endif

#define MFMA_RAW_HALF(buf, a, b, off) do {                                          \
    _Pragma("unroll")                                                                \
    for (int _hn = 0; _hn < 2; ++_hn) {                                              \
        _Pragma("unroll")                                                            \
        for (int _hm = 0; _hm < 2; ++_hm) {                                          \
            ::kittens::mfma161632((buf)[_hn][_hm],                                   \
                                  (a).tiles[_hn + (off)][0].data,                    \
                                  (b).tiles[_hm][0].data,                            \
                                  (buf)[_hn][_hm]);                                  \
        }                                                                            \
    }                                                                                \
} while (0)

#define MFMA_RAW_QUAD(a_top, a_bot, bt) do {                                        \
    __builtin_amdgcn_s_setprio(1);                                                  \
    MFMA_RAW_HALF(raw_partial[0], a_tiles[a_top], b_tiles[bt], 0);                  \
    MFMA_RAW_HALF(raw_partial[1], a_tiles[a_top], b_tiles[bt], (REG_M/32));          \
    MFMA_RAW_HALF(raw_partial[2], a_tiles[a_bot], b_tiles[bt], 0);                  \
    MFMA_RAW_HALF(raw_partial[3], a_tiles[a_bot], b_tiles[bt], (REG_M/32));          \
    __builtin_amdgcn_s_setprio(0);                                                  \
} while (0)

#define RAW_DRAIN_HALF(dst, buf, sv_ptr, bs, dst_off) do {                              \
    _Pragma("unroll")                                                                    \
    for (int _n = 0; _n < PARTIAL_N_HEIGHT; ++_n) {                                      \
        const float _s0 = (sv_ptr)[_n*4]   * (bs);                                       \
        const float _s1 = (sv_ptr)[_n*4+1] * (bs);                                       \
        const float _s2 = (sv_ptr)[_n*4+2] * (bs);                                       \
        const float _s3 = (sv_ptr)[_n*4+3] * (bs);                                       \
        _Pragma("unroll")                                                                \
        for (int _m = 0; _m < 2; ++_m) {                                                 \
            float2 *_df = (dst).tiles[_n+(dst_off)][_m].data;                            \
            _df[0].x += (buf)[_n][_m][0].x * _s0;                                        \
            _df[0].y += (buf)[_n][_m][0].y * _s1;                                        \
            _df[1].x += (buf)[_n][_m][1].x * _s2;                                        \
            _df[1].y += (buf)[_n][_m][1].y * _s3;                                        \
            (buf)[_n][_m][0] = float2{0.f, 0.f};                                         \
            (buf)[_n][_m][1] = float2{0.f, 0.f};                                         \
        }                                                                                \
    }                                                                                    \
} while(0)

#define RAW_DRAIN_HALF_PREBS(dst, buf, sv_ptr, dst_off) do {                             \
    _Pragma("unroll")                                                                    \
    for (int _n = 0; _n < PARTIAL_N_HEIGHT; ++_n) {                                      \
        const float _s0 = (sv_ptr)[_n*4];                                                \
        const float _s1 = (sv_ptr)[_n*4+1];                                              \
        const float _s2 = (sv_ptr)[_n*4+2];                                              \
        const float _s3 = (sv_ptr)[_n*4+3];                                              \
        _Pragma("unroll")                                                                \
        for (int _m = 0; _m < 2; ++_m) {                                                 \
            float2 *_df = (dst).tiles[_n+(dst_off)][_m].data;                            \
            _df[0].x += (buf)[_n][_m][0].x * _s0;                                        \
            _df[0].y += (buf)[_n][_m][0].y * _s1;                                        \
            _df[1].x += (buf)[_n][_m][1].x * _s2;                                        \
            _df[1].y += (buf)[_n][_m][1].y * _s3;                                        \
            (buf)[_n][_m][0] = float2{0.f, 0.f};                                         \
            (buf)[_n][_m][1] = float2{0.f, 0.f};                                         \
        }                                                                                \
    }                                                                                    \
} while(0)

#define RAW_WGRAD_DRAIN_HALF(dst, buf, sv_ptr, bs_arr, dst_off) do {                    \
    _Pragma("unroll")                                                                    \
    for (int _n = 0; _n < PARTIAL_N_HEIGHT; ++_n) {                                      \
        _Pragma("unroll")                                                                \
        for (int _m = 0; _m < 2; ++_m) {                                                 \
            const float _bs = (bs_arr)[_m];                                              \
            const float _s0 = (sv_ptr)[_n*4]   * _bs;                                    \
            const float _s1 = (sv_ptr)[_n*4+1] * _bs;                                    \
            const float _s2 = (sv_ptr)[_n*4+2] * _bs;                                    \
            const float _s3 = (sv_ptr)[_n*4+3] * _bs;                                    \            float *_df = reinterpret_cast<float*>((dst).tiles[_n+(dst_off)][_m].data);   \
            float *_bf = reinterpret_cast<float*>(&(buf)[_n][_m][0]);                    \
            _df[0] += _bf[0] * _s0;  _df[1] += _bf[1] * _s1;                             \
            _df[2] += _bf[2] * _s2;  _df[3] += _bf[3] * _s3;                             \
            (buf)[_n][_m][0] = float2{0.f, 0.f};                                         \
            (buf)[_n][_m][1] = float2{0.f, 0.f};                                         \
        }                                                                                \
    }                                                                                    \
} while(0)

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

struct micro_globals {
    _gl_A A;
    _gl_B B;
    _gl_C C;
    _gl_S A_scale;   // [Kb, M]
    _gl_S B_scale;   // [Kb, Nb]
    hipStream_t stream;
    // Runtime grid: M = A.rows(), N = B.rows() (kernel handles any M%256==0, N%128==0).
    dim3 grid()  { return dim3((B.rows() / BLOCK_N) * (A.rows() / BLOCK_M)); }
    dim3 block() { return dim3(NUM_WARPS * WARP_THREADS); }
    size_t dynamic_shared_memory() {
        return (size_t)BLOCK_M * BLOCK_K * sizeof(fp8e4m3)
             + (size_t)BLOCK_N * BLOCK_K * sizeof(fp8e4m3)
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

#define SCALE_DRAIN_HALF_PREBS(dst, src, sv_ptr, dst_off) do {                       \
    _Pragma("unroll")                                                                \
    for (int _n = 0; _n < (src).height; ++_n) {                                      \
        const float _s0 = (sv_ptr)[_n*4];                                            \
        const float _s1 = (sv_ptr)[_n*4+1];                                          \
        const float _s2 = (sv_ptr)[_n*4+2];                                          \
        const float _s3 = (sv_ptr)[_n*4+3];                                          \
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

// WGRAD_DRAIN_HALF: per-(N-row, K-col) variant of SCALE_DRAIN_HALF for the
// real CRR wgrad path. Same shape contract (drains 2-M-row partial<32,32,col>
// into a 2-M-row slice of a 4-M-row dst<64,32,col>) but bs is a per-K-sub-tile
// array bs_arr[2] (one per K-col-group, distinct per lane). CK 1×1×128
// ABQuantGrouped pattern. Used by micro_tk<B_SCALE_PER_ELEMENT=true>.
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

template <bool B_SCALE_PER_ELEMENT = false>
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
#if !BW_RAW_DRAIN
    rt_fl<REG_M/2, REG_N, ducks::rt_layout::col> partial[4];
#endif
    for (int i = 0; i < 2; i++) zero(C_accum[i]);
#if !BW_RAW_DRAIN
    for (int i = 0; i < 4; i++) zero(partial[i]);
#endif

    #if BW_RAW_DRAIN
    float2 raw_partial[4][PARTIAL_N_HEIGHT][2][2];
    #pragma unroll
    for (int _i = 0; _i < 4; ++_i) {
        #pragma unroll
        for (int _n = 0; _n < PARTIAL_N_HEIGHT; ++_n) {
            #pragma unroll
            for (int _m = 0; _m < 2; ++_m) {
                raw_partial[_i][_n][_m][0] = float2{0.f, 0.f};
                raw_partial[_i][_n][_m][1] = float2{0.f, 0.f};
            }
        }
    }
    #endif

    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, BW_CHIPLET_CHUNK);
    const int num_pid_n = ceil_div(g.B.rows(), BLOCK_N);
    const int output_m = wgid / num_pid_n;
    const int output_n = wgid % num_pid_n;

    const int warp_id = warpid();
    const int warp_row = warp_id / WARP_COLS, warp_col = warp_id % WARP_COLS;
    const int k_iters = g.A.cols() / BLOCK_K;

    const int mt_base = output_m * BLOCK_M + warp_row * REG_M;
    const int mb_base = output_m * BLOCK_M + (BLOCK_M/2) + warp_row * REG_M;
    const int b_scale_idx = (output_n * BLOCK_N + warp_col * REG_N) / 128;
    // Per-element b_scale anchors (used only when B_SCALE_PER_ELEMENT=true).
    // col_l layout: each lane owns 1 N-col within a 16×16 sub-tile, at offset
    // c16 = laneid()%16. For 32-col warp range × 2 K-sub-tiles → 2 b_scale per thread.
    const int kt_anchor = output_n * BLOCK_N + warp_col * REG_N;
    const int c16 = laneid() % 16;
    const int r16 = 4 * (laneid() / 16);

    constexpr int BUFFER_SIZE_A = (BLOCK_M * BLOCK_K) / NUM_THREADS / sizeof(float4) / sizeof(fp8e4m3);
    constexpr int BUFFER_SIZE_B = (BLOCK_N * BLOCK_K) / NUM_THREADS / sizeof(float4) / sizeof(fp8e4m3);

    G::load(As, g.A, {0, 0, output_m, 0});
    G::load(Bs, g.B, {0, 0, output_n, 0});
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    for (int K_TILE = 0; K_TILE < k_iters - 1; ++K_TILE) {

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
        #if BW_RAW_DRAIN
        MFMA_RAW_QUAD(1, 2, 0);
        #else
        MMA_QUAD_PRIO(1, 2, 0);
        #endif
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
        #if BW_RAW_DRAIN
        MFMA_RAW_QUAD(4, 5, 1);
        #else
        MMA_QUAD_PRIO(4, 5, 1);
        #endif
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4: B prefetch + load sub-K=3
        load_global_to_register_buffer<2, false, NUM_THREADS>(
            b_buffer_next, BUFFER_SIZE_B, g.B, {0, 0, output_n, K_TILE + 1}, Bs);
        load(a_tiles[2], subtile_inplace<REG_M, REG_K>(As, {warp_row + WARP_ROWS, 2}));
        load(b_tiles[2], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 3}));
        load(a_tiles[0], subtile_inplace<REG_M, REG_K>(As, {warp_row, 3}));
        load(a_tiles[5], subtile_inplace<REG_M, REG_K>(As, {warp_row + WARP_ROWS, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        #if BW_RAW_DRAIN
        MFMA_RAW_QUAD(1, 2, 0);
        #else
        MMA_QUAD_PRIO(1, 2, 0);
        #endif
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6: commit prefetches → LDS
        asm volatile("s_waitcnt vmcnt(0)");
        store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
        store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        float b_s = 0.f;
        float b_s_per[2]{};
        if constexpr (B_SCALE_PER_ELEMENT) {
            b_s_per[0] = g.B_scale[coord<>(K_TILE, kt_anchor + 0*16 + c16)];
            b_s_per[1] = g.B_scale[coord<>(K_TILE, kt_anchor + 1*16 + c16)];
        } else {
            b_s = g.B_scale[coord<>(K_TILE, b_scale_idx)];
        }
        constexpr int SCALE_HALF_COUNT = REG_M / 4;
        constexpr int SCALE_LOAD_ITERS = REG_M / 16;
        float svt[SCALE_HALF_COUNT], svb[SCALE_HALF_COUNT];
        #pragma unroll
        for (int i = 0; i < SCALE_LOAD_ITERS; ++i) {
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                svt[i*4+d] = g.A_scale[coord<>(K_TILE, mt_base + i*16 + r16 + d)];
                svb[i*4+d] = g.A_scale[coord<>(K_TILE, mb_base + i*16 + r16 + d)];
            }
        }
#if BW_PRESCALE_BS
        if constexpr (!B_SCALE_PER_ELEMENT) {
            #pragma unroll
            for (int _i = 0; _i < SCALE_HALF_COUNT; ++_i) {
                svt[_i] *= b_s;
                svb[_i] *= b_s;
            }
        }
#endif

        #if BW_RAW_DRAIN
        MFMA_RAW_QUAD(0, 5, 2);
        #else
        MMA_QUAD_PRIO(0, 5, 2);
        #endif

            #if BW_RAW_DRAIN
            if constexpr (B_SCALE_PER_ELEMENT) {
                RAW_WGRAD_DRAIN_HALF(C_accum[0], raw_partial[0], svt + 0, b_s_per, 0);
                RAW_WGRAD_DRAIN_HALF(C_accum[0], raw_partial[1], svt + SCALE_STRIDE_HALF, b_s_per, (REG_M/32));
                RAW_WGRAD_DRAIN_HALF(C_accum[1], raw_partial[2], svb + 0, b_s_per, 0);
                RAW_WGRAD_DRAIN_HALF(C_accum[1], raw_partial[3], svb + SCALE_STRIDE_HALF, b_s_per, (REG_M/32));
            } else {
                #if BW_PRESCALE_BS
                // Round-72: svt/svb already × b_s at cluster-6 load.
                RAW_DRAIN_HALF_PREBS(C_accum[0], raw_partial[0], svt + 0, 0);
                RAW_DRAIN_HALF_PREBS(C_accum[0], raw_partial[1], svt + SCALE_STRIDE_HALF, (REG_M/32));
                RAW_DRAIN_HALF_PREBS(C_accum[1], raw_partial[2], svb + 0, 0);
                RAW_DRAIN_HALF_PREBS(C_accum[1], raw_partial[3], svb + SCALE_STRIDE_HALF, (REG_M/32));
                #else
                RAW_DRAIN_HALF(C_accum[0], raw_partial[0], svt + 0, b_s, 0);
                RAW_DRAIN_HALF(C_accum[0], raw_partial[1], svt + SCALE_STRIDE_HALF, b_s, (REG_M/32));
                RAW_DRAIN_HALF(C_accum[1], raw_partial[2], svb + 0, b_s, 0);
                RAW_DRAIN_HALF(C_accum[1], raw_partial[3], svb + SCALE_STRIDE_HALF, b_s, (REG_M/32));
                #endif
            }
            #else
            if constexpr (B_SCALE_PER_ELEMENT) {
                WGRAD_DRAIN_HALF(C_accum[0], partial[0], svt + 0, b_s_per, 0);
                WGRAD_DRAIN_HALF(C_accum[0], partial[1], svt + SCALE_STRIDE_HALF, b_s_per, (REG_M/32));
                WGRAD_DRAIN_HALF(C_accum[1], partial[2], svb + 0, b_s_per, 0);
                WGRAD_DRAIN_HALF(C_accum[1], partial[3], svb + SCALE_STRIDE_HALF, b_s_per, (REG_M/32));
            } else {
                #if BW_PRESCALE_BS
                SCALE_DRAIN_HALF_PREBS(C_accum[0], partial[0], svt + 0, 0);
                SCALE_DRAIN_HALF_PREBS(C_accum[0], partial[1], svt + SCALE_STRIDE_HALF, (REG_M/32));
                SCALE_DRAIN_HALF_PREBS(C_accum[1], partial[2], svb + 0, 0);
                SCALE_DRAIN_HALF_PREBS(C_accum[1], partial[3], svb + SCALE_STRIDE_HALF, (REG_M/32));
                #else
                SCALE_DRAIN_HALF(C_accum[0], partial[0], svt + 0, b_s, 0);
                SCALE_DRAIN_HALF(C_accum[0], partial[1], svt + SCALE_STRIDE_HALF, b_s, (REG_M/32));
                SCALE_DRAIN_HALF(C_accum[1], partial[2], svb + 0, b_s, 0);
                SCALE_DRAIN_HALF(C_accum[1], partial[3], svb + SCALE_STRIDE_HALF, b_s, (REG_M/32));
                #endif
            }
            #endif
    }

    // Epilogue: final K_TILE.
    {
        const int K_TILE = k_iters - 1;
        float b_s = 0.f;
        float b_s_per[2]{};
        if constexpr (B_SCALE_PER_ELEMENT) {
            b_s_per[0] = g.B_scale[coord<>(K_TILE, kt_anchor + 0*16 + c16)];
            b_s_per[1] = g.B_scale[coord<>(K_TILE, kt_anchor + 1*16 + c16)];
        } else {
            b_s = g.B_scale[coord<>(K_TILE, b_scale_idx)];
        }
        constexpr int SCALE_HALF_COUNT = REG_M / 4;
        constexpr int SCALE_LOAD_ITERS = REG_M / 16;
        float svt[SCALE_HALF_COUNT], svb[SCALE_HALF_COUNT];
        #pragma unroll
        for (int i = 0; i < SCALE_LOAD_ITERS; ++i) {
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                svt[i*4+d] = g.A_scale[coord<>(K_TILE, mt_base + i*16 + r16 + d)];
                svb[i*4+d] = g.A_scale[coord<>(K_TILE, mb_base + i*16 + r16 + d)];
            }
        }
#if BW_PRESCALE_BS
        if constexpr (!B_SCALE_PER_ELEMENT) {
            #pragma unroll
            for (int _i = 0; _i < SCALE_HALF_COUNT; ++_i) {
                svt[_i] *= b_s;
                svb[_i] *= b_s;
            }
        }
#endif

        __builtin_amdgcn_sched_barrier(0);
        load(b_tiles[0], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 0}));
        load(a_tiles[1], subtile_inplace<REG_M, REG_K>(As, {warp_row, 0}));
        load(a_tiles[2], subtile_inplace<REG_M, REG_K>(As, {warp_row + WARP_ROWS, 0}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        #if BW_RAW_DRAIN
        MFMA_RAW_QUAD(1, 2, 0);
        #else
        MMA_QUAD_PRIO(1, 2, 0);
        #endif
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load(b_tiles[1], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 1}));
        load(a_tiles[4], subtile_inplace<REG_M, REG_K>(As, {warp_row, 1}));
        load(a_tiles[5], subtile_inplace<REG_M, REG_K>(As, {warp_row + WARP_ROWS, 1}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        #if BW_RAW_DRAIN
        MFMA_RAW_QUAD(4, 5, 1);
        #else
        MMA_QUAD_PRIO(4, 5, 1);
        #endif
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

        #if BW_RAW_DRAIN
        MFMA_RAW_QUAD(1, 2, 0);
        #else
        MMA_QUAD_PRIO(1, 2, 0);
        #endif
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        #if BW_RAW_DRAIN
        MFMA_RAW_QUAD(4, 5, 2);
        #else
        MMA_QUAD_PRIO(4, 5, 2);
        #endif
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

            #if BW_RAW_DRAIN
            if constexpr (B_SCALE_PER_ELEMENT) {
                RAW_WGRAD_DRAIN_HALF(C_accum[0], raw_partial[0], svt + 0, b_s_per, 0);
                RAW_WGRAD_DRAIN_HALF(C_accum[0], raw_partial[1], svt + SCALE_STRIDE_HALF, b_s_per, (REG_M/32));
                RAW_WGRAD_DRAIN_HALF(C_accum[1], raw_partial[2], svb + 0, b_s_per, 0);
                RAW_WGRAD_DRAIN_HALF(C_accum[1], raw_partial[3], svb + SCALE_STRIDE_HALF, b_s_per, (REG_M/32));
            } else {
                #if BW_PRESCALE_BS
                RAW_DRAIN_HALF_PREBS(C_accum[0], raw_partial[0], svt + 0, 0);
                RAW_DRAIN_HALF_PREBS(C_accum[0], raw_partial[1], svt + SCALE_STRIDE_HALF, (REG_M/32));
                RAW_DRAIN_HALF_PREBS(C_accum[1], raw_partial[2], svb + 0, 0);
                RAW_DRAIN_HALF_PREBS(C_accum[1], raw_partial[3], svb + SCALE_STRIDE_HALF, (REG_M/32));
                #else
                RAW_DRAIN_HALF(C_accum[0], raw_partial[0], svt + 0, b_s, 0);
                RAW_DRAIN_HALF(C_accum[0], raw_partial[1], svt + SCALE_STRIDE_HALF, b_s, (REG_M/32));
                RAW_DRAIN_HALF(C_accum[1], raw_partial[2], svb + 0, b_s, 0);
                RAW_DRAIN_HALF(C_accum[1], raw_partial[3], svb + SCALE_STRIDE_HALF, b_s, (REG_M/32));
                #endif
            }
            #else
            if constexpr (B_SCALE_PER_ELEMENT) {
                WGRAD_DRAIN_HALF(C_accum[0], partial[0], svt + 0, b_s_per, 0);
                WGRAD_DRAIN_HALF(C_accum[0], partial[1], svt + SCALE_STRIDE_HALF, b_s_per, (REG_M/32));
                WGRAD_DRAIN_HALF(C_accum[1], partial[2], svb + 0, b_s_per, 0);
                WGRAD_DRAIN_HALF(C_accum[1], partial[3], svb + SCALE_STRIDE_HALF, b_s_per, (REG_M/32));
            } else {
                #if BW_PRESCALE_BS
                SCALE_DRAIN_HALF_PREBS(C_accum[0], partial[0], svt + 0, 0);
                SCALE_DRAIN_HALF_PREBS(C_accum[0], partial[1], svt + SCALE_STRIDE_HALF, (REG_M/32));
                SCALE_DRAIN_HALF_PREBS(C_accum[1], partial[2], svb + 0, 0);
                SCALE_DRAIN_HALF_PREBS(C_accum[1], partial[3], svb + SCALE_STRIDE_HALF, (REG_M/32));
                #else
                SCALE_DRAIN_HALF(C_accum[0], partial[0], svt + 0, b_s, 0);
                SCALE_DRAIN_HALF(C_accum[0], partial[1], svt + SCALE_STRIDE_HALF, b_s, (REG_M/32));
                SCALE_DRAIN_HALF(C_accum[1], partial[2], svb + 0, b_s, 0);
                SCALE_DRAIN_HALF(C_accum[1], partial[3], svb + SCALE_STRIDE_HALF, b_s, (REG_M/32));
                #endif
            }
            #endif
    }

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }
    store(g.C, C_accum[0], {0, 0, output_m * M_TILES_PER_BLOCK + warp_row,             output_n * N_TILES_PER_BLOCK + warp_col});
    store(g.C, C_accum[1], {0, 0, output_m * M_TILES_PER_BLOCK + warp_row + WARP_ROWS, output_n * N_TILES_PER_BLOCK + warp_col});
}

template <bool B_SCALE_PER_ELEMENT>
void launch_micro_tk(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk<B_SCALE_PER_ELEMENT>,
                        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<B_SCALE_PER_ELEMENT><<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

void dispatch_micro(micro_globals g)       { launch_micro_tk<false>(g); }
void dispatch_micro_dgrad(micro_globals g) { launch_micro_tk<false>(g); }  // caller pre-transposes B (Python side)
void dispatch_micro_wgrad(micro_globals g) { launch_micro_tk<true>(g);  }  // per-element B_scale (CRR wgrad)

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "FP8 blockwise GEMM (fwd/dgrad/wgrad) — MI300X gfx942";
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &micro_globals::A, &micro_globals::B, &micro_globals::C,
        &micro_globals::A_scale, &micro_globals::B_scale);
    py::bind_function<dispatch_micro_dgrad>(m, "dispatch_micro_dgrad",
        &micro_globals::A, &micro_globals::B, &micro_globals::C,
        &micro_globals::A_scale, &micro_globals::B_scale);
    py::bind_function<dispatch_micro_wgrad>(m, "dispatch_micro_wgrad",
        &micro_globals::A, &micro_globals::B, &micro_globals::C,
        &micro_globals::A_scale, &micro_globals::B_scale);
}
