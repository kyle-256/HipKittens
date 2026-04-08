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

constexpr int BLK = GEMM_BLOCK_SIZE, BK = GEMM_K_BLOCK;
constexpr int HB  = BLK / 2;

using _gl_fp8  = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;

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

struct layout_globals {
    _gl_fp8 a, b;
    _gl_bf16 c;
    float scale_a, scale_b;
    hipStream_t stream;
    int m, n, k;
    int group_m;
};

#include "rcr_exact_4wave_fastpath.inc"

static void dispatch_rcr(layout_globals g) {
    g.m = static_cast<int>(g.c.rows());
    g.n = static_cast<int>(g.c.cols());
    g.k = static_cast<int>(g.a.cols());

#if RCR_USE_EXACT_4WAVE_FASTPATH
    if (g.m == M_DIM && g.n == N_DIM && g.k == K_DIM) {
        rcr_exact_4wave::globals eg{
            make_gl<rcr_exact_4wave::exact_a_gl>(
                reinterpret_cast<uint64_t>(g.a.raw_ptr), 1, 1, g.m, g.k),
            make_gl<rcr_exact_4wave::exact_b_gl>(
                reinterpret_cast<uint64_t>(g.b.raw_ptr), 1, 1, g.n, g.k),
            make_gl<rcr_exact_4wave::exact_c_gl>(
                reinterpret_cast<uint64_t>(g.c.raw_ptr), 1, 1, g.m, g.n),
            g.scale_a, g.scale_b,
            g.stream,
        };
        rcr_exact_4wave::kernel<<<eg.grid(), eg.block(), 0, eg.stream>>>(eg);
        return;
    }
#endif
}

static float to_float(pybind11::object obj) {
    if (pybind11::hasattr(obj, "item"))
        return obj.attr("item")().cast<float>();
    return obj.cast<float>();
}

static void gemm_rcr(pybind11::object a, pybind11::object b, pybind11::object c,
                      pybind11::object scale_a_obj, pybind11::object scale_b_obj,
                      int group_m) {
    layout_globals g{
        py::from_object<_gl_fp8>::make(a),
        py::from_object<_gl_fp8>::make(b),
        py::from_object<_gl_bf16>::make(c),
        to_float(scale_a_obj),
        to_float(scale_b_obj),
        {}, 0, 0, 0, group_m,
    };
    dispatch_rcr(g);
}

PYBIND11_MODULE(tk_fp8_layouts, m) {
    m.def("gemm_rcr", &gemm_rcr,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
          pybind11::arg("scale_a"), pybind11::arg("scale_b"),
          pybind11::arg("group_m") = 4);
}
