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
using A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>;
using B_row_reg = rt_fp8e4m3<RBN, BK, row_l, rt_16x128_s>;

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
    br = bid / bpc;
    bc = bid % bpc;
#endif
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

#include "rcr_exact_4wave_fastpath.inc"
#include "rcr_exact_8wave_fastpath.inc"

static float to_float(pybind11::object obj) {
    if (pybind11::hasattr(obj, "item"))
        return obj.attr("item")().cast<float>();
    return obj.cast<float>();
}

constexpr int DEFAULT_GROUP_M = 4;

static void gemm_rcr(pybind11::object a, pybind11::object b, pybind11::object c,
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
    g.k = static_cast<int>(g.a.cols());

#if RCR_USE_EXACT_4WAVE_FASTPATH
    if (rcr_can_use_exact_4wave(g)) {
        dispatch_rcr_exact_4wave(g);
        return;
    }
#endif
#if RCR_USE_EXACT_8WAVE_FASTPATH
    if (rcr_can_use_exact_8wave(g)) {
        dispatch_rcr_exact_8wave(g);
        return;
    }
#endif
}

PYBIND11_MODULE(tk_fp8_layouts, m) {
    m.def("gemm_rcr", &gemm_rcr,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = DEFAULT_GROUP_M);
}
