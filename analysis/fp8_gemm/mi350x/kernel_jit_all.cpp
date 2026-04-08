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
#ifndef RCR_STEADY_VMCNT
#define RCR_STEADY_VMCNT 8
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

using ST_row = st_fp8e4m3<HB, BK, st_16x128_s>;
using ST_v2  = st_fp8e4m3<HB, BK, st_16x128_v2_s>;
using ST_v2a = st_fp8e4m3<HB, BK, st_16x128_v2a_s>;
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;
using A_col_reg = rt_fp8e4m3<BK, RBM, col_l, rt_128x16_s>;
using B_col_reg = rt_fp8e4m3<BK, RBN, col_l, rt_128x16_s>;

#ifndef RRR_MAIN_UNROLL
#define RRR_MAIN_UNROLL 4
#endif
#ifndef RRR_PREFETCH_LGKM
#define RRR_PREFETCH_LGKM 8
#endif
#ifndef RRR_INIT0_VMCNT
#define RRR_INIT0_VMCNT 4
#endif
#ifndef RRR_INIT1_VMCNT
#define RRR_INIT1_VMCNT 6
#endif
#ifndef RRR_STEADY_VMCNT
#define RRR_STEADY_VMCNT 4
#endif
#ifndef RRR_EPILOGUE_VMCNT
#define RRR_EPILOGUE_VMCNT 2
#endif
#define TK_PRAGMA_UNROLL(x) _Pragma(TK_STRINGIFY(unroll x))
#ifndef CRR_ROW_SHARED_TRANSPOSE
#define CRR_ROW_SHARED_TRANSPOSE 0
#endif
#ifndef CRR_A_LDS_REENCODE
#define CRR_A_LDS_REENCODE 0
#endif
#ifndef CRR_BATCHED_EPILOGUE_MMA
#define CRR_BATCHED_EPILOGUE_MMA 0
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
#ifndef CRR_PREFETCH_LGKM
#define CRR_PREFETCH_LGKM 3
#endif
#ifndef CRR_MAIN_UNROLL
#define CRR_MAIN_UNROLL 1
#endif
#ifndef CRR_ENABLE_SCHED_BARRIER
#define CRR_ENABLE_SCHED_BARRIER 0
#endif
#if CRR_ENABLE_SCHED_BARRIER
#define CRR_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
#else
#define CRR_SCHED_BARRIER() do {} while (0)
#endif
#define CRR_MMA_BEGIN() do { CRR_SCHED_BARRIER(); __builtin_amdgcn_s_setprio(1); } while (0)
#define CRR_MMA_END() do { __builtin_amdgcn_s_setprio(0); CRR_SCHED_BARRIER(); } while (0)
#ifndef RRR_USE_V2A_SWIZZLE
#define RRR_USE_V2A_SWIZZLE 0
#endif
#if RRR_USE_V2A_SWIZZLE
using ST_rrr_b = ST_v2a;
#else
using ST_rrr_b = ST_v2;
#endif

#ifndef RRR_B_REG_ROW_LOAD_TRANSPOSE
#define RRR_B_REG_ROW_LOAD_TRANSPOSE 0
#endif
#ifndef RRR_B_REG_ROW_LOAD_ALIAS
#define RRR_B_REG_ROW_LOAD_ALIAS 0
#endif
#ifndef RRR_ROW_SHARED_TRANSPOSE
#define RRR_ROW_SHARED_TRANSPOSE 0
#endif
#ifndef RRR_ENABLE_SCHED_BARRIER
#define RRR_ENABLE_SCHED_BARRIER 1
#endif
#if RRR_ENABLE_SCHED_BARRIER
#define RRR_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
#else
#define RRR_SCHED_BARRIER() do {} while (0)
#endif

#ifndef CRR_USE_V3_SWIZZLE
#define CRR_USE_V3_SWIZZLE 0
#endif
#ifndef CRR_B_REG_ROW_LOAD_TRANSPOSE
#define CRR_B_REG_ROW_LOAD_TRANSPOSE 0
#endif
#ifndef CRR_B_REG_ROW_LOAD_ALIAS
#define CRR_B_REG_ROW_LOAD_ALIAS 0
#endif

__device__ __forceinline__ void rrr_mma(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_row_reg& a, const B_col_reg& b) {
    mma_AB(acc, a, b, acc);
}

enum class Layout { RCR, RRR, CRR };

__device__ __forceinline__ int gemm_chiplet_swizzle_bid(int bid, int num_wgs) {
#if GEMM_BLOCK_SWIZZLE
    if (num_wgs >= GEMM_BLOCK_SWIZZLE_NUM_XCDS &&
        (num_wgs % GEMM_BLOCK_SWIZZLE_NUM_XCDS) == 0) {
        return (bid % GEMM_BLOCK_SWIZZLE_NUM_XCDS) *
               (num_wgs / GEMM_BLOCK_SWIZZLE_NUM_XCDS) +
               (bid / GEMM_BLOCK_SWIZZLE_NUM_XCDS);
    }
#endif
    return bid;
}

__device__ __forceinline__ void gemm_compute_block_coords(
    int bid, int bpr, int bpc, int group_m, int &br, int &bc) {
#if GEMM_BLOCK_SWIZZLE
    bid = gemm_chiplet_swizzle_bid(bid, gridDim.x);
    const int num_wgid_in_group = group_m * bpc;
    const int group_id = bid / num_wgid_in_group;
    const int first_pid_m = group_id * group_m;
    const int group_size_m =
        (first_pid_m + group_m <= bpr) ? group_m : (bpr - first_pid_m);
    if (group_size_m <= 0) { br = bpr; bc = bpc; return; }
    br = first_pid_m + ((bid % num_wgid_in_group) % group_size_m);
    bc = (bid % num_wgid_in_group) / group_size_m;
#else
    br = bid / bpc; bc = bid % bpc;
#endif
}

template<typename RT, int K_HALF>
__device__ __forceinline__ void load_col_from_v2_st_half(
    RT& dst, const ST_v2& tile, int col_start) {
    const int laneid = kittens::laneid();
    const int row_off = ((laneid % 16) / 2) + ((laneid / 16) * 16);
    const int col_off = (laneid % 2) * 8;
    const uint32_t tile_base = reinterpret_cast<uintptr_t>(&tile.data[0]);
    constexpr int idx = K_HALF * 4;
    const int k_row = row_off + K_HALF * 64;
    const uint32_t stidx = k_row >> 4;
    const uint32_t base_k = tile_base + (stidx << 11) + (stidx << 7) + ((k_row & 15) << 7);
    const uint32_t sw_k = (k_row & 7) << 4;
    #pragma unroll
    for (int j = 0; j < RT::width; j++) {
        const uint32_t nc = col_start + j * 16 + col_off;
        const uint32_t addr = base_k + (nc ^ sw_k);
        asm volatile("ds_read_b64_tr_b8 %0, %2 offset:0\n"
                     "ds_read_b64_tr_b8 %1, %2 offset:1024\n"
            : "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx])),
              "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx + 2]))
            : "v"(addr) : "memory");
    }
}
template<typename RT>
__device__ __forceinline__ void load_col_from_v2_st(RT& dst, const ST_v2& tile, int col_start) {
    load_col_from_v2_st_half<RT, 0>(dst, tile, col_start);
    load_col_from_v2_st_half<RT, 1>(dst, tile, col_start);
}
template<typename RT, int K_HALF, typename ST>
__device__ __forceinline__ void load_col_from_v2a_st_half(RT& dst, const ST& tile, int col_start) {
    const int laneid = kittens::laneid();
    const int row_off = ((laneid % 16) / 2) + ((laneid / 16) * 16);
    const int col_off = (laneid % 2) * 8;
    const uint32_t tile_base = reinterpret_cast<uintptr_t>(&tile.data[0]);
    constexpr int idx = K_HALF * 4;
    const int k_row = row_off + K_HALF * 64;
    const uint32_t stidx = k_row >> 4;
    const uint32_t base_k = tile_base + (stidx << 11) + (stidx << 7) + ((k_row & 15) << 7);
    const uint32_t sw_k = (k_row & 7) << 4;
    #pragma unroll
    for (int j = 0; j < RT::width; j++) {
        const uint32_t nc = col_start + j * 16 + col_off;
        const uint32_t addr = base_k + (nc ^ sw_k);
        asm volatile("ds_read_b64_tr_b8 %0, %2 offset:0\n"
                     "ds_read_b64_tr_b8 %1, %2 offset:1024\n"
            : "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx])),
              "=&v"(*reinterpret_cast<float2*>(&dst.tiles[0][j].data[idx + 2]))
            : "v"(addr) : "memory");
    }
}
template<typename RT, typename ST>
__device__ __forceinline__ void load_col_from_v2a_st(RT& dst, const ST& tile, int col_start) {
    load_col_from_v2a_st_half<RT, 0>(dst, tile, col_start);
    load_col_from_v2a_st_half<RT, 1>(dst, tile, col_start);
}

__device__ __forceinline__ void crr_mma(
    rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
    const A_col_reg& a, const B_col_reg& b) {
    mma_AtB(acc, a, b, acc);
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

// --- Include all fastpaths ---
#ifndef RCR_USE_EXACT_4WAVE_FASTPATH
#define RCR_USE_EXACT_4WAVE_FASTPATH 1
#endif
#ifndef RCR_4WAVE_GROUP_M
#define RCR_4WAVE_GROUP_M 4
#endif
#ifndef RCR_4WAVE_NUM_XCDS
#define RCR_4WAVE_NUM_XCDS 8
#endif
#ifndef RCR_4WAVE_ENABLE_XCD_SWIZZLE
#define RCR_4WAVE_ENABLE_XCD_SWIZZLE 1
#endif
#ifndef RCR_USE_EXACT_8WAVE_FASTPATH
#define RCR_USE_EXACT_8WAVE_FASTPATH 1
#endif
#ifndef RRR_USE_EXACT_8WAVE_FASTPATH
#define RRR_USE_EXACT_8WAVE_FASTPATH 1
#endif
#ifndef CRR_USE_EXACT_4WAVE_FASTPATH
#define CRR_USE_EXACT_4WAVE_FASTPATH 0
#endif
#ifndef CRR_USE_EXACT_8WAVE_FASTPATH
#define CRR_USE_EXACT_8WAVE_FASTPATH 1
#endif
#ifndef CRR_USE_EXACT_8WAVE_DOUBLE_PUMP_FASTPATH
#define CRR_USE_EXACT_8WAVE_DOUBLE_PUMP_FASTPATH 0
#endif
#ifndef CRR_BATCHED_PAIR_MMA
#define CRR_BATCHED_PAIR_MMA 1
#endif
#ifndef CRR_ENABLE_STEADY_MID_BARRIER
#define CRR_ENABLE_STEADY_MID_BARRIER 1
#endif
#if CRR_ENABLE_STEADY_MID_BARRIER
#define CRR_STEADY_MID_BARRIER() __builtin_amdgcn_s_barrier()
#else
#define CRR_STEADY_MID_BARRIER() do {} while (0)
#endif
#ifndef CRR_A_REG_ROW_LOAD_TRANSPOSE
#define CRR_A_REG_ROW_LOAD_TRANSPOSE 1
#endif
#ifndef CRR_A_REG_ROW_LOAD_ALIAS
#define CRR_A_REG_ROW_LOAD_ALIAS 1
#endif

#ifndef JIT_LAYOUT
#define JIT_LAYOUT 0
#endif

#if JIT_LAYOUT == 0 || JIT_LAYOUT == 1
#include "rcr_exact_4wave_fastpath.inc"
#include "rcr_exact_8wave_fastpath.inc"
#endif
#if JIT_LAYOUT == 0 || JIT_LAYOUT == 2
#include "rrr_exact_8wave_fastpath.inc"
#endif
#if JIT_LAYOUT == 0 || JIT_LAYOUT == 3
#include "crr_exact_8wave_fastpath.inc"
#endif

// --- Pybind ---
static float to_float(pybind11::object obj) {
    if (pybind11::hasattr(obj, "item"))
        return obj.attr("item")().cast<float>();
    return obj.cast<float>();
}

constexpr int DEFAULT_GROUP_M = 4;

template<Layout L>
static void gemm_wrapper(pybind11::object a, pybind11::object b, pybind11::object c,
                          pybind11::object sa, pybind11::object sb, int group_m) {
    layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        to_float(sa), to_float(sb),
        {}, 0, 0, 0, 0, 0, 0, 0, 0, 0, group_m,
    };
    if constexpr (L == Layout::RCR) {
        g.m = static_cast<int>(g.c.rows());
        g.n = static_cast<int>(g.c.cols());
        g.k = static_cast<int>(g.a.cols());
    } else if constexpr (L == Layout::RRR) {
        g.m = static_cast<int>(g.c.rows());
        g.n = static_cast<int>(g.c.cols());
        g.k = static_cast<int>(g.a.cols());
    } else {
        g.m = static_cast<int>(g.c.rows());
        g.n = static_cast<int>(g.c.cols());
        g.k = static_cast<int>(g.b.rows());
    }

#if (JIT_LAYOUT == 0 || JIT_LAYOUT == 1) && RCR_USE_EXACT_4WAVE_FASTPATH
    if constexpr (L == Layout::RCR) {
        if (rcr_can_use_exact_4wave(g)) { dispatch_rcr_exact_4wave(g); return; }
    }
#endif
#if (JIT_LAYOUT == 0 || JIT_LAYOUT == 1) && RCR_USE_EXACT_8WAVE_FASTPATH
    if constexpr (L == Layout::RCR) {
        if (rcr_can_use_exact_8wave(g)) { dispatch_rcr_exact_8wave(g); return; }
    }
#endif
#if (JIT_LAYOUT == 0 || JIT_LAYOUT == 2) && RRR_USE_EXACT_8WAVE_FASTPATH
    if constexpr (L == Layout::RRR) {
        if (rrr_can_use_exact_8wave(g)) { dispatch_rrr_exact_8wave(g); return; }
    }
#endif
#if (JIT_LAYOUT == 0 || JIT_LAYOUT == 3) && CRR_USE_EXACT_8WAVE_FASTPATH
    if constexpr (L == Layout::CRR) {
        if (crr_can_use_exact_8wave(g)) { dispatch_crr_exact_8wave(g); return; }
    }
#endif
}

PYBIND11_MODULE(tk_fp8_layouts, m) {
#if JIT_LAYOUT == 0 || JIT_LAYOUT == 1
    m.def("gemm_rcr", &gemm_wrapper<Layout::RCR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
#endif
#if JIT_LAYOUT == 0 || JIT_LAYOUT == 2
    m.def("gemm_rrr", &gemm_wrapper<Layout::RRR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
#endif
#if JIT_LAYOUT == 0 || JIT_LAYOUT == 3
    m.def("gemm_crr", &gemm_wrapper<Layout::CRR>,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
#endif
}
