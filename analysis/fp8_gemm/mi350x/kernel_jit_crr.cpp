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
#ifndef GEMM_WARPS_M
#define GEMM_WARPS_M 2
#endif
#ifndef GEMM_WARPS_N
#define GEMM_WARPS_N 4
#endif
#ifndef GEMM_BLOCK_SWIZZLE
#define GEMM_BLOCK_SWIZZLE 1
#endif
#ifndef GEMM_BLOCK_SWIZZLE_NUM_XCDS
#define GEMM_BLOCK_SWIZZLE_NUM_XCDS 8
#endif
#ifndef GEMM_MIN_BLOCKS_PER_CU
#define GEMM_MIN_BLOCKS_PER_CU 2
#endif

// CRR vmcnt/scheduling knobs (same defaults as dynamic kernel)
#ifndef CRR_PREFETCH_LGKM
#define CRR_PREFETCH_LGKM 3
#endif
#ifndef CRR_INIT0_VMCNT
#define CRR_INIT0_VMCNT 2
#endif
#ifndef CRR_INIT1_VMCNT
#define CRR_INIT1_VMCNT 6
#endif
#ifndef CRR_STEADY_VMCNT
#define CRR_STEADY_VMCNT 4
#endif
#ifndef CRR_EPILOGUE_VMCNT
#define CRR_EPILOGUE_VMCNT 2
#endif
#ifndef CRR_ENABLE_SCHED_BARRIER
#define CRR_ENABLE_SCHED_BARRIER 0
#endif
#ifndef CRR_ENABLE_STEADY_MID_BARRIER
#define CRR_ENABLE_STEADY_MID_BARRIER 1
#endif
#ifndef CRR_BATCHED_PAIR_MMA
#define CRR_BATCHED_PAIR_MMA 1
#endif
#ifndef CRR_BATCHED_EPILOGUE_MMA
#define CRR_BATCHED_EPILOGUE_MMA 0
#endif
#ifndef CRR_A_LDS_REENCODE
#define CRR_A_LDS_REENCODE 0
#endif
#ifndef CRR_ROW_SHARED_TRANSPOSE
#define CRR_ROW_SHARED_TRANSPOSE 0
#endif
#ifndef CRR_USE_V3_SWIZZLE
#define CRR_USE_V3_SWIZZLE 0
#endif
#ifndef CRR_A_REG_ROW_LOAD_TRANSPOSE
#define CRR_A_REG_ROW_LOAD_TRANSPOSE 1
#endif
#ifndef CRR_B_REG_ROW_LOAD_TRANSPOSE
#define CRR_B_REG_ROW_LOAD_TRANSPOSE 0
#endif
#ifndef CRR_A_REG_ROW_LOAD_ALIAS
#define CRR_A_REG_ROW_LOAD_ALIAS 1
#endif
#ifndef CRR_B_REG_ROW_LOAD_ALIAS
#define CRR_B_REG_ROW_LOAD_ALIAS 0
#endif

constexpr int BLK = GEMM_BLOCK_SIZE, BK = GEMM_K_BLOCK;
constexpr int HB  = BLK / 2;
constexpr int WARPS_M = GEMM_WARPS_M, WARPS_N = GEMM_WARPS_N;
constexpr int _NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int _NUM_THREADS = _NUM_WARPS * WARP_THREADS;
constexpr int RBM = BLK / WARPS_M / 2;
constexpr int RBN = BLK / WARPS_N / 2;

using G = kittens::group<_NUM_WARPS>;
using _gl_fp8  = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;

// Shared memory tile types used by crr_exact_4wave_fastpath.inc
using ST_v2  = st_fp8e4m3<HB, BK, st_16x128_v2_s>;
using ST_v2a = st_fp8e4m3<HB, BK, st_16x128_v2a_s>;

// Register types (needed by the fastpath inc)
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using A_col_reg = rt_fp8e4m3<BK, RBM, col_l, rt_128x16_s>;

// ---- load_col_from_v2_st ----
template<typename RT, int K_HALF>
__device__ __forceinline__ void load_col_from_v2_st_half(
    RT& dst, const ST_v2& tile, int col_start)
{
    const int laneid = kittens::laneid();
    const int row_off = ((laneid % 16) / 2) + ((laneid / 16) * 16);
    const int col_off = (laneid % 2) * 8;
    const uint32_t tile_base = reinterpret_cast<uintptr_t>(&tile.data[0]);
    constexpr int idx = K_HALF * 4;
    const int k_row = row_off + K_HALF * 64;
    const uint32_t stidx = k_row >> 4;
    const uint32_t base_k = tile_base + (stidx << 11) + (stidx << 7) + ((k_row & 15) << 7);
    const uint32_t sw_k   = (k_row & 7) << 4;
    #pragma unroll
    for (int j = 0; j < RT::width; j++) {
        const uint32_t nc = col_start + j * 16 + col_off;
        const uint32_t addr = base_k + (nc ^ sw_k);
        asm volatile(
            "ds_read_b64_tr_b8 %0, %2 offset:0\n"
            "ds_read_b64_tr_b8 %1, %2 offset:1024\n"
            : "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx])),
              "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx + 2]))
            : "v"(addr)
            : "memory"
        );
    }
}

template<typename RT>
__device__ __forceinline__ void load_col_from_v2_st(
    RT& dst, const ST_v2& tile, int col_start)
{
    load_col_from_v2_st_half<RT, 0>(dst, tile, col_start);
    load_col_from_v2_st_half<RT, 1>(dst, tile, col_start);
}

// ---- load_col_from_v2a_st ----
template<typename RT, int K_HALF, typename ST>
__device__ __forceinline__ void load_col_from_v2a_st_half(
    RT& dst, const ST& tile, int col_start)
{
    const int laneid = kittens::laneid();
    const int row_off = ((laneid % 16) / 2) + ((laneid / 16) * 16);
    const int col_off = (laneid % 2) * 8;
    const uint32_t tile_base = reinterpret_cast<uintptr_t>(&tile.data[0]);
    constexpr int idx = K_HALF * 4;
    const int k_row = row_off + K_HALF * 64;
    const uint32_t stidx = k_row >> 4;
    const uint32_t base_k = tile_base + (stidx << 11) + (stidx << 7) + ((k_row & 15) << 7);
    const uint32_t sw_k   = (k_row & 7) << 4;
    #pragma unroll
    for (int j = 0; j < RT::width; j++) {
        const uint32_t nc = col_start + j * 16 + col_off;
        const uint32_t addr = base_k + (nc ^ sw_k);
        asm volatile(
            "ds_read_b64_tr_b8 %0, %2 offset:0\n"
            "ds_read_b64_tr_b8 %1, %2 offset:1024\n"
            : "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx])),
              "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx + 2]))
            : "v"(addr)
            : "memory"
        );
    }
}

template<typename RT, typename ST>
__device__ __forceinline__ void load_col_from_v2a_st(
    RT& dst, const ST& tile, int col_start)
{
    load_col_from_v2a_st_half<RT, 0>(dst, tile, col_start);
    load_col_from_v2a_st_half<RT, 1>(dst, tile, col_start);
}

struct layout_globals {
    _gl_fp8 a, b;
    _gl_bf16 c;
    float scale_a, scale_b;
    hipStream_t stream;
    int m, n, k;
    int bpr, bpc, ki;
    int fast_m, fast_n, fast_k;
    int group_m;
    dim3 grid()  { return dim3(bpr * bpc); }
    dim3 block() { return dim3(_NUM_THREADS); }
};

#ifndef CRR_USE_EXACT_4WAVE_FASTPATH
#define CRR_USE_EXACT_4WAVE_FASTPATH 1
#endif

#include "crr_exact_4wave_fastpath.inc"

static float to_float(pybind11::object obj) {
    if (pybind11::hasattr(obj, "item"))
        return obj.attr("item")().cast<float>();
    return obj.cast<float>();
}

constexpr int DEFAULT_GROUP_M = 4;

// CRR: A是(K, M), B是(K, N), C是(M, N)
// g.a.rows()=K, g.a.cols()=M; g.b.rows()=K, g.b.cols()=N
static void gemm_crr(pybind11::object a, pybind11::object b, pybind11::object c,
                      pybind11::object sa, pybind11::object sb, int group_m) {
    layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        to_float(sa), to_float(sb),
        {}, 0, 0, 0, 0, 0, 0, 0, 0, 0, group_m,
    };
    g.m = static_cast<int>(g.c.rows());
    g.n = static_cast<int>(g.c.cols());
    g.k = static_cast<int>(g.a.rows());  // CRR: K是A的行维

#if CRR_USE_EXACT_4WAVE_FASTPATH
    if (crr_can_use_exact_4wave(g)) {
        dispatch_crr_exact_4wave(g);
        return;
    }
#endif
    // JIT内核只处理可用4-wave的shape，不提供fallback
}

PYBIND11_MODULE(tk_fp8_layouts, m) {
    m.def("gemm_crr", &gemm_crr,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
}
