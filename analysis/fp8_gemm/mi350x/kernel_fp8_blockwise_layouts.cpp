// FP8 blockwise GEMM (reference HIP kernel) — HipKittens / kittens gl + coord.
// Computes the same semantics as CK/Primus blockwise: per K-block (128) scale multiply
// after each chunk dot product. A/B tensors stay in native layouts; only scale tensors
// may be transposed/contiguous on host (preshuffle-scale).
//
// Performance: thread-per-output element; intended for correctness / ratio baselines.
// Full-speed path belongs in kernel_fp8_layouts.cpp MFMA loops (accumulator + per-ki scale).

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

using _gl_fp8 = gl<fp8e4m3, -1, -1, -1, -1>;
using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;
using _gl_f32 = gl<float, -1, -1, -1, -1>;

__device__ __forceinline__ float load_fp8(const _gl_fp8& src, int row, int col) {
    return base_types::convertor<float, fp8e4m3>::convert(src[coord<>(row, col)]);
}

__device__ __forceinline__ float load_f32(const _gl_f32& src, int row, int col) {
    return src[coord<>(row, col)];
}

__device__ __forceinline__ void store_bf16(_gl_bf16& dst, int row, int col, float v) {
    dst[coord<>(row, col)] = base_types::convertor<bf16, float>::convert(v);
}

// NT / RCR: C = A[M,K] @ B[N,K]^T  (B stored [N,K] like test_python RCR)
// a_scale: [M, Kb], b_scale: [Nb, Kb] with Kb = ceil(K/128), Nb = ceil(N/128)
struct layout_bw_rcr {
    _gl_fp8 a, b;
    _gl_bf16 c;
    _gl_f32 a_scale, b_scale;
    hipStream_t stream = nullptr;
};

__global__ void gemm_rcr_blockwise_kernel(layout_bw_rcr g, int M, int N, int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = idx / N;
    const int n = idx % N;
    if (m >= M || n >= N) {
        return;
    }
    const int Kb = (K + 127) / 128;
    const int nb = n / 128;
    float acc = 0.f;
    for (int ki = 0; ki < Kb; ++ki) {
        const int k0 = ki * 128;
        const int k1 = (k0 + 128 < K) ? (k0 + 128) : K;
        float dot = 0.f;
        for (int k = k0; k < k1; ++k) {
            dot += load_fp8(g.a, m, k) * load_fp8(g.b, n, k);
        }
        const float sa = load_f32(g.a_scale, m, ki);
        const float sb = load_f32(g.b_scale, nb, ki);
        acc += dot * sa * sb;
    }
    store_bf16(g.c, m, n, acc);
}

// NN / RRR: C = A[M,K] @ B[K,N]
// a_scale: [M, Kb], b_scale: [Nb, Kb] (same as Primus after b_scale.T.contiguous())
struct layout_bw_rrr {
    _gl_fp8 a, b;
    _gl_bf16 c;
    _gl_f32 a_scale, b_scale;
    hipStream_t stream = nullptr;
};

__global__ void gemm_rrr_blockwise_kernel(layout_bw_rrr g, int M, int N, int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = idx / N;
    const int n = idx % N;
    if (m >= M || n >= N) {
        return;
    }
    const int Kb = (K + 127) / 128;
    const int nb = n / 128;
    float acc = 0.f;
    for (int ki = 0; ki < Kb; ++ki) {
        const int k0 = ki * 128;
        const int k1 = (k0 + 128 < K) ? (k0 + 128) : K;
        float dot = 0.f;
        for (int k = k0; k < k1; ++k) {
            dot += load_fp8(g.a, m, k) * load_fp8(g.b, k, n);
        }
        const float sa = load_f32(g.a_scale, m, ki);
        const float sb = load_f32(g.b_scale, nb, ki);
        acc += dot * sa * sb;
    }
    store_bf16(g.c, m, n, acc);
}

// TN / CRR: C = A[K,M]^T @ B[K,N] = A^T @ B with A [K,M], B [K,N]
// a_scale: [Kb, M], b_scale: [Kb, N] (column-wise / axis=0 quantization layout)
struct layout_bw_crr {
    _gl_fp8 a, b;
    _gl_bf16 c;
    _gl_f32 a_scale, b_scale;
    hipStream_t stream = nullptr;
};

__global__ void gemm_crr_blockwise_kernel(layout_bw_crr g, int M, int N, int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = idx / N;
    const int n = idx % N;
    if (m >= M || n >= N) {
        return;
    }
    const int Kb = (K + 127) / 128;
    float acc = 0.f;
    for (int ki = 0; ki < Kb; ++ki) {
        const int k0 = ki * 128;
        const int k1 = (k0 + 128 < K) ? (k0 + 128) : K;
        float dot = 0.f;
        for (int k = k0; k < k1; ++k) {
            dot += load_fp8(g.a, k, m) * load_fp8(g.b, k, n);
        }
        const float sa = load_f32(g.a_scale, ki, m);
        const float sb = load_f32(g.b_scale, ki, n);
        acc += dot * sa * sb;
    }
    store_bf16(g.c, m, n, acc);
}

void dispatch_gemm_rcr_blockwise(layout_bw_rcr g) {
    const int M = g.c.rows();
    const int N = g.c.cols();
    const int K = g.a.cols();
    const int total = M * N;
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    gemm_rcr_blockwise_kernel<<<blocks, threads, 0, g.stream>>>(g, M, N, K);
}

void dispatch_gemm_rrr_blockwise(layout_bw_rrr g) {
    const int M = g.c.rows();
    const int N = g.c.cols();
    const int K = g.a.cols();
    const int total = M * N;
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    gemm_rrr_blockwise_kernel<<<blocks, threads, 0, g.stream>>>(g, M, N, K);
}

void dispatch_gemm_crr_blockwise(layout_bw_crr g) {
    const int M = g.c.rows();
    const int N = g.c.cols();
    const int K = g.a.rows();
    const int total = M * N;
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    gemm_crr_blockwise_kernel<<<blocks, threads, 0, g.stream>>>(g, M, N, K);
}

PYBIND11_MODULE(tk_fp8_blockwise_layouts, m) {
    m.doc() = "FP8 blockwise GEMM (reference HIP): RCR/RRR/CRR with per-K-block scales";
    py::bind_function<dispatch_gemm_rcr_blockwise>(
        m, "gemm_rcr_blockwise",
        &layout_bw_rcr::a, &layout_bw_rcr::b, &layout_bw_rcr::c,
        &layout_bw_rcr::a_scale, &layout_bw_rcr::b_scale);
    py::bind_function<dispatch_gemm_rrr_blockwise>(
        m, "gemm_rrr_blockwise",
        &layout_bw_rrr::a, &layout_bw_rrr::b, &layout_bw_rrr::c,
        &layout_bw_rrr::a_scale, &layout_bw_rrr::b_scale);
    py::bind_function<dispatch_gemm_crr_blockwise>(
        m, "gemm_crr_blockwise",
        &layout_bw_crr::a, &layout_bw_crr::b, &layout_bw_crr::c,
        &layout_bw_crr::a_scale, &layout_bw_crr::b_scale);
}
