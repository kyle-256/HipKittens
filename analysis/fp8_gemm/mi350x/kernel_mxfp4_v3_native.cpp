// MXFP4 GEMM V3: Loads a pre-compiled kernel binary (from Gluon/Triton with LLIR scheduler)
// and launches it via hipModule API. Achieves ~5400 TFLOPS on MI355X for 8192^3.
//
// The kernel binary was compiled from the gluon a4w4 kernel with:
//   TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1
//
// Data format (no preshuffle):
//   A:        (M, K//2) uint8, row-major K-contiguous
//   B:        (N, K//2) uint8, row-major K-contiguous
//   a_scales: (M, K//32) uint8 e8m0, with stride(0)=1 stride(1)=M_pad
//   b_scales: (N, K//32) uint8 e8m0, with stride(0)=1 stride(1)=N
//   C:        (M, N) bfloat16

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <cstdint>
#include <stdexcept>
#include <mutex>

#include "gluon_a4w4_hsaco.h" // Embedded kernel binary

namespace py = pybind11;

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(e)); \
} while(0)

// Kernel arg layout matching the gluon LLVM IR signature:
//   5 ptrs (a, b, c, a_scales, b_scales) + 7 i32 (M, N, strides) + 2 ptrs (unused scratch)
// Triton JIT specializes stride=1 args away. Remaining non-constexpr strides:
// stride_am (=K//2), stride_bn (=K//2), stride_cm (=N), stride_ask (=M_pad), stride_bsk (=N)
// Removed (constexpr=1): stride_ak, stride_bk, stride_cn, stride_asm, stride_bsn
struct __attribute__((packed)) GluonKernelArgs {
    uint64_t a_ptr;
    uint64_t b_ptr;
    uint64_t c_ptr;
    uint64_t a_scales_ptr;
    uint64_t b_scales_ptr;
    int32_t  M;
    int32_t  N;
    int32_t  stride_am;    // a.stride(0) = K//2
    int32_t  stride_bn;    // b.stride(0) = K//2
    int32_t  stride_cm;    // c.stride(0) = N
    int32_t  stride_ask;   // a_scales.stride(1) = M_pad (after .T)
    int32_t  stride_bsk;   // b_scales.stride(1) = N
    uint64_t scratch_ptr_0;
    uint64_t scratch_ptr_1;
};

static hipModule_t g_module = nullptr;
static hipFunction_t g_kernel = nullptr;
static std::once_flag g_init_flag;

static void init_module() {
    HIP_CHECK(hipModuleLoadData(&g_module, gluon_a4w4_hsaco));
    HIP_CHECK(hipModuleGetFunction(&g_kernel, g_module, "a4w4_kernel"));
}

void mxfp4_gemm_v3(
    torch::Tensor a,         // (M, K//2) uint8
    torch::Tensor b,         // (N, K//2) uint8
    torch::Tensor a_scales,  // stride(0)=1, stride(1)=M_pad
    torch::Tensor b_scales,  // stride(0)=1, stride(1)=N
    torch::Tensor c          // (M, N) bf16
) {
    std::call_once(g_init_flag, init_module);

    int M = a.size(0);
    int K_half = a.size(1);
    int N = b.size(0);

    constexpr int BM = 256, BN = 256;
    int grid_mn = ((M + BM - 1) / BM) * ((N + BN - 1) / BN);

    GluonKernelArgs args;
    args.a_ptr        = reinterpret_cast<uint64_t>(a.data_ptr());
    args.b_ptr        = reinterpret_cast<uint64_t>(b.data_ptr());
    args.c_ptr        = reinterpret_cast<uint64_t>(c.data_ptr());
    args.a_scales_ptr = reinterpret_cast<uint64_t>(a_scales.data_ptr());
    args.b_scales_ptr = reinterpret_cast<uint64_t>(b_scales.data_ptr());
    args.M            = M;
    args.N            = N;
    args.stride_am    = static_cast<int32_t>(a.stride(0));
    args.stride_bn    = static_cast<int32_t>(b.stride(0));
    args.stride_cm    = static_cast<int32_t>(c.stride(0));
    args.stride_ask   = static_cast<int32_t>(a_scales.stride(1));
    args.stride_bsk   = static_cast<int32_t>(b_scales.stride(1));
    args.scratch_ptr_0 = 0;
    args.scratch_ptr_1 = 0;

    size_t arg_size = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        HIP_LAUNCH_PARAM_END
    };

    // 256 threads (4 warps), shared mem = 138144 bytes (from kernel metadata)
    HIP_CHECK(hipModuleLaunchKernel(
        g_kernel,
        grid_mn, 1, 1,     // grid
        256, 1, 1,          // block (4 warps × 64 threads)
        138144,             // shared mem bytes
        nullptr,            // stream
        nullptr,            // kernel params (unused with config)
        config
    ));
}

PYBIND11_MODULE(tk_mxfp4_v3, m) {
    m.doc() = "MXFP4 GEMM V3: native gluon kernel binary (~5400 TFLOPS)";
    m.def("gemm_rcr", &mxfp4_gemm_v3);
}
