// MXFP4 GEMM: Pure HipKittens with hand-written GCN assembly kernel.
// Device code: kernel_mxfp4_asm.s (assembled → kernel_mxfp4_asm_data.h)
// Host code: this file (hipModule launcher + pybind11)

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cstdint>
#include <stdexcept>
#include <mutex>

#include "kernel_mxfp4_asm_data.h"

namespace py = pybind11;

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); \
} while(0)

struct __attribute__((packed)) AsmKernelArgs {
    uint64_t a_ptr;
    uint64_t b_ptr;
    uint64_t c_ptr;
    uint64_t a_scales_ptr;
    uint64_t b_scales_ptr;
    int32_t  M;
    int32_t  N;
    int32_t  stride_am;
    int32_t  stride_bn;
    int32_t  stride_cm;
    int32_t  stride_ask;
    int32_t  stride_bsk;
    uint64_t scratch0;
    uint64_t scratch1;
};

static hipModule_t g_mod = nullptr;
static hipFunction_t g_func = nullptr;
static std::once_flag g_once;

static void init() {
    HIP_CHECK(hipModuleLoadData(&g_mod, mxfp4_asm_hsaco));
    HIP_CHECK(hipModuleGetFunction(&g_func, g_mod, "mxfp4_gluon_asm_kernel"));
}

void mxfp4_gemm_asm(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor a_scales, torch::Tensor b_scales,
    torch::Tensor c)
{
    std::call_once(g_once, init);
    int M = a.size(0), N = b.size(0);
    int grid = ((M + 255) / 256) * ((N + 255) / 256);

    AsmKernelArgs args = {
        reinterpret_cast<uint64_t>(a.data_ptr()),
        reinterpret_cast<uint64_t>(b.data_ptr()),
        reinterpret_cast<uint64_t>(c.data_ptr()),
        reinterpret_cast<uint64_t>(a_scales.data_ptr()),
        reinterpret_cast<uint64_t>(b_scales.data_ptr()),
        M, N,
        static_cast<int32_t>(a.stride(0)),
        static_cast<int32_t>(b.stride(0)),
        static_cast<int32_t>(c.stride(0)),
        static_cast<int32_t>(a_scales.stride(1)),
        static_cast<int32_t>(b_scales.stride(1)),
        0, 0
    };
    size_t sz = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
        HIP_LAUNCH_PARAM_END
    };
    HIP_CHECK(hipModuleLaunchKernel(g_func,
        grid, 1, 1, 256, 1, 1, 138144, nullptr, nullptr, config));
}

PYBIND11_MODULE(tk_mxfp4_asm, m) {
    m.doc() = "MXFP4 GEMM — hand-written GCN assembly (gluon-optimized ISA in HipKittens)";
    m.def("gemm_rcr", &mxfp4_gemm_asm);
}
