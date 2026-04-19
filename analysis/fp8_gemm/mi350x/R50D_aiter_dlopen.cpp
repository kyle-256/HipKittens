// R50 Opt D — aiter `.co` dlopen shim for `(4096,32768,28672)` MXFP4 escape hatch.
//
// Self-contained: depends ONLY on hip_runtime, pybind11, torch ATen tensors.
// Does NOT link against aiter library. Loads aiter's `.co` directly via
// hipModuleLoad and launches the kernel with aiter's KernelArgs ABI.
//
// Reference: /shared_nfs/kyle/test/aiter/csrc/py_itfs_cu/asm_gemm_a4w4.cu
// Aiter `.co`: /shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/
//              f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co
// Symbol:      _ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E
//
// Layout assumptions (mirrors aiter):
//   A : [M, K/2]  fp4_x2 (uint8 packed; 2 fp4 per byte)
//   B : [N, K/2]  fp4_x2 PRESHUFFLED via aiter shuffle_weight(layout=(16,16))
//   A_scale : [M_pad, K/32] e8m0 padded (PRESHUFFLED via aiter per_1x32_f4_quant(shuffle=True))
//   B_scale : [N_pad, K/32] e8m0 padded (PRESHUFFLED via aiter per_1x32_f4_quant(shuffle=True))
//   C : [M_pad32, N] bf16  (M_pad32 = ((M + 31) / 32) * 32)
//
// Launch: gdx = ceil(N/SUBN), gdy = ceil(M/SUBM), gdz = 1 (no splitK)
//         bdx = 256, bdy = bdz = 1; sharedMemBytes = 0.

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <iostream>
#include <mutex>

namespace py = pybind11;

// -----------------------------------------------------------------------------
// KernelArgs ABI — mirror of /shared_nfs/kyle/test/aiter/csrc/py_itfs_cu/asm_gemm_a4w4.cu
// -----------------------------------------------------------------------------
struct __attribute__((packed)) p3 {
    unsigned int _p0;
    unsigned int _p1;
    unsigned int _p2;
};
struct __attribute__((packed)) p2 {
    unsigned int _p0;
    unsigned int _p1;
};

struct __attribute__((packed)) KernelArgs {
    void* ptr_D;
    p2 _p0;
    void* ptr_C;
    p2 _p1;
    void* ptr_A;
    p2 _p2;
    void* ptr_B;
    p2 _p3;
    float alpha;
    p3 _p4;
    float beta;
    p3 _p5;
    unsigned int stride_D0;
    p3 _p6;
    unsigned int stride_D1;
    p3 _p7;
    unsigned int stride_C0;
    p3 _p8;
    unsigned int stride_C1;
    p3 _p9;
    unsigned int stride_A0;
    p3 _p10;
    unsigned int stride_A1;
    p3 _p11;
    unsigned int stride_B0;
    p3 _p12;
    unsigned int stride_B1;
    p3 _p13;
    unsigned int M;
    p3 _p14;
    unsigned int N;
    p3 _p15;
    unsigned int K;
    p3 _p16;
    void* ptr_ScaleA;
    p2 _p17;
    void* ptr_ScaleB;
    p2 _p18;
    unsigned int stride_ScaleA0;
    p3 _p19;
    unsigned int stride_ScaleA1;
    p3 _p20;
    unsigned int stride_ScaleB0;
    p3 _p21;
    unsigned int stride_ScaleB1;
    p3 _p22;
    int log2_k_split;
};

static_assert(sizeof(KernelArgs) == 372, "KernelArgs size mismatch with aiter ABI");

// -----------------------------------------------------------------------------
// Module / kernel cache (per-process, per-co_path).
// -----------------------------------------------------------------------------
static std::mutex g_mod_mtx;
static hipModule_t g_module = nullptr;
static hipFunction_t g_kernel = nullptr;
static std::string g_loaded_path;

#define HIP_CHECK(expr) do {                                                \
    hipError_t _err = (expr);                                               \
    if (_err != hipSuccess) {                                               \
        std::string msg = std::string("HIP error: ") + #expr + " -> "       \
                          + hipGetErrorString(_err);                        \
        throw std::runtime_error(msg);                                      \
    }                                                                       \
} while(0)

static void load_kernel_if_needed(const std::string& co_path,
                                  const std::string& kernel_name) {
    std::lock_guard<std::mutex> lk(g_mod_mtx);
    if (g_module != nullptr && g_loaded_path == co_path) return;
    if (g_module != nullptr) {
        // Different co_path requested — unload prior.
        hipModuleUnload(g_module);
        g_module = nullptr;
        g_kernel = nullptr;
        g_loaded_path.clear();
    }
    HIP_CHECK(hipModuleLoad(&g_module, co_path.c_str()));
    HIP_CHECK(hipModuleGetFunction(&g_kernel, g_module, kernel_name.c_str()));
    g_loaded_path = co_path;
}

// -----------------------------------------------------------------------------
// Public Python entrypoint.
//
// Inputs (all torch tensors on the same CUDA device):
//   A          : [M, K/2] uint8       (fp4x2, row-major)
//   B          : [N, K/2] uint8       (fp4x2, PRESHUFFLED via aiter shuffle_weight)
//   A_scale    : [Mp, Kg] uint8       (e8m0, preshuffled per aiter per_1x32_f4_quant)
//   B_scale    : [Np, Kg] uint8       (e8m0, preshuffled per aiter per_1x32_f4_quant)
//   C          : [M_pad32, N] bf16    (output buffer, must be allocated)
//   M, N, K    : python ints (logical problem sizes)
//   tile_M, tile_N : aiter kernel tile (256, 256 for the 256x256 .co)
//   co_path, kernel_name : strings
//   alpha, beta : floats
// -----------------------------------------------------------------------------
void aiter_gemm_a4w4_launch(at::Tensor A, at::Tensor B,
                            at::Tensor A_scale, at::Tensor B_scale,
                            at::Tensor C,
                            int64_t M, int64_t N, int64_t K,
                            int64_t tile_M, int64_t tile_N,
                            const std::string& co_path,
                            const std::string& kernel_name,
                            double alpha, double beta,
                            uint64_t stream_handle) {
    load_kernel_if_needed(co_path, kernel_name);

    KernelArgs args;
    std::memset(&args, 0, sizeof(args));

    args.ptr_D     = C.data_ptr();
    args.ptr_C     = nullptr;          // no bias
    args.ptr_A     = A.data_ptr();
    args.ptr_B     = B.data_ptr();
    args.alpha     = static_cast<float>(alpha);
    args.beta      = static_cast<float>(beta);
    args.M         = static_cast<unsigned int>(M);
    args.N         = static_cast<unsigned int>(N);
    args.K         = static_cast<unsigned int>(K);

    // Mirror aiter: stride_*0 captures the leading-dim (row) stride only.
    // For fp4_x2 packed inputs A, B aiter records `stride * 2` (per .cu).
    args.stride_C0 = static_cast<unsigned int>(C.stride(0));      // C is bf16
    args.stride_A0 = static_cast<unsigned int>(A.stride(0) * 2);  // fp4x2
    args.stride_B0 = static_cast<unsigned int>(B.stride(0) * 2);  // fp4x2
    args.stride_ScaleA0 = static_cast<unsigned int>(A_scale.stride(0));
    args.stride_ScaleB0 = static_cast<unsigned int>(B_scale.stride(0));

    args.ptr_ScaleA = A_scale.data_ptr();
    args.ptr_ScaleB = B_scale.data_ptr();

    args.log2_k_split = 0;

    int gdx = static_cast<int>((N + tile_N - 1) / tile_N);
    int gdy = static_cast<int>((M + tile_M - 1) / tile_M);
    int gdz = 1;
    int bdx = 256, bdy = 1, bdz = 1;

    size_t arg_size = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
        HIP_LAUNCH_PARAM_END
    };

    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_handle);
    HIP_CHECK(hipModuleLaunchKernel(g_kernel,
                                    gdx, gdy, gdz,
                                    bdx, bdy, bdz,
                                    /*sharedMemBytes*/ 0,
                                    stream,
                                    nullptr,
                                    (void**)&config));
}

void aiter_unload() {
    std::lock_guard<std::mutex> lk(g_mod_mtx);
    if (g_module != nullptr) {
        hipModuleUnload(g_module);
        g_module = nullptr;
        g_kernel = nullptr;
        g_loaded_path.clear();
    }
}

int aiter_args_size() { return static_cast<int>(sizeof(KernelArgs)); }

PYBIND11_MODULE(R50D_aiter_shim, m) {
    m.doc() = "R50 Opt D — aiter f4gemm .co dlopen shim for HipKittens MXFP4 escape hatch";
    m.def("launch", &aiter_gemm_a4w4_launch,
          py::arg("A"), py::arg("B"), py::arg("A_scale"), py::arg("B_scale"),
          py::arg("C"), py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("tile_M") = 256, py::arg("tile_N") = 256,
          py::arg("co_path"),
          py::arg("kernel_name") = std::string(
              "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E"),
          py::arg("alpha") = 1.0,
          py::arg("beta")  = 0.0,
          py::arg("stream_handle") = uint64_t{0},
          "Launch aiter f4gemm via direct .co dlopen. "
          "stream_handle=0 -> default stream; pass torch.cuda.current_stream().cuda_stream "
          "to use the active stream.");
    m.def("unload", &aiter_unload);
    m.def("args_size", &aiter_args_size);
}
