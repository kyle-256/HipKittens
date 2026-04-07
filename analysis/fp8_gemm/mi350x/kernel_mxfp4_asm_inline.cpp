#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
namespace py = pybind11;

#include "kernel_mxfp4_asm_inline.h"

// Kernel with entire body as inline GCN assembly (2227 instructions).
// No external .s files. Everything compiles from this single .cpp.
extern "C" __global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__attribute__((amdgpu_waves_per_eu(1)))
void mxfp4_asm_inline_kernel(
    const void* a, const void* b, void* c,
    const void* a_sc, const void* b_sc,
    int M, int N,
    int s_am, int s_bn, int s_cm, int s_ask, int s_bsk)
{
    asm volatile( MXFP4_KERNEL_ASM_BODY ::: "memory" );
}

void launch(torch::Tensor a, torch::Tensor b,
            torch::Tensor a_sc, torch::Tensor b_sc,
            torch::Tensor c) {
    int M=a.size(0), N=b.size(0);
    int g=((M+255)/256)*((N+255)/256);
    mxfp4_asm_inline_kernel<<<dim3(g),dim3(256),138144>>>(
        a.data_ptr(),b.data_ptr(),c.data_ptr(),
        a_sc.data_ptr(),b_sc.data_ptr(),
        M,N,(int)a.stride(0),(int)b.stride(0),(int)c.stride(0),
        (int)a_sc.stride(1),(int)b_sc.stride(1));
}

PYBIND11_MODULE(tk_mxfp4_asm_inline, m) {
    m.def("gemm_rcr", &launch);
}
