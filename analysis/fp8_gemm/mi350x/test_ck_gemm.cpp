// Test CK's MXFP4 GEMM template with BLayout=Col (non-preshuffled B)
// This is the training scenario where B is stored as [N, K/2] (K contiguous)

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/stream_config.hpp"

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using MFMA = ck::tensor_layout::gemm::MFMA;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using F4PK = ck::f4x2_pk_t;
using B16 = ck::bhalf_t;
using F32 = float;
using E8M0PK = int32_t;

using ADataType = F4PK;
using BDataType = F4PK;
using XPackedDataType = E8M0PK;
using AccDataType = float;

using ALayout = Row;  // A is [M, K/2], K contiguous
using CLayout = Row;  // C is [M, N], N contiguous

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

constexpr ck::index_t ScaleBlockSize = 32;

static constexpr auto Intrawave = ck::BlockGemmPipelineScheduler::Intrawave;

// Config matching aiter's best kernel for large shapes:
// 256x256x128, AK1=16, BK1=16, 16x16 MFMA, 8x8 wave map, Intrawave v3
template <typename BLayout, typename CDataType>
using DeviceGemmHelper = ck::tensor_operation::device::DeviceGemmMX_Xdl_CShuffleV3<
    ALayout, BLayout, CLayout,
    ADataType, XPackedDataType, BDataType, XPackedDataType, CDataType, AccDataType, CDataType,
    AElementOp, BElementOp, CElementOp,
    ck::tensor_operation::device::GemmSpecialization::Default,
    ScaleBlockSize,
    256,                    // BlockSize
    256, 256, 128,          // MPerBlock, NPerBlock, KPerBlock (packed: 128*2=256 FP4)
    16, 16,                 // AK1, BK1
    16, 16,                 // MPerXDL, NPerXDL
    8, 8,                   // MXdlPerWave, NXdlPerWave
    S<8, 32, 1>,            // ABlockTransfer
    S<1, 0, 2>,             // ABlockTransferArrangeOrder
    S<1, 0, 2>,             // ABlockTransferSrcAccessOrder
    2,                      // ABlockTransferSrcVectorDim
    16,                     // ABlockTransferSrcScalarPerVector
    16,                     // ABlockTransferDstScalarPerVector_AK1
    true,                   // ABlockLdsExtraM
    S<8, 32, 1>,            // BBlockTransfer
    S<1, 0, 2>,             // BBlockTransferArrangeOrder
    S<1, 0, 2>,             // BBlockTransferSrcAccessOrder
    2,                      // BBlockTransferSrcVectorDim
    16,                     // BBlockTransferSrcScalarPerVector
    16,                     // BBlockTransferDstScalarPerVector_BK1
    true,                   // BBlockLdsExtraN
    2,                      // CShuffleMXdlPerWavePerShuffle
    8,                      // CShuffleNXdlPerWavePerShuffle
    S<1, 8, 1, 32>,         // CShuffleBlockTransferClusterLengths
    8,                      // CShuffleBlockTransferScalarPerVector
    Intrawave,
    ck::BlockGemmPipelineVersion::v3,
    ADataType, BDataType>;

// Also try 16x4 wave map (aiter kernel #18)
template <typename BLayout, typename CDataType>
using DeviceGemmHelper_16x4 = ck::tensor_operation::device::DeviceGemmMX_Xdl_CShuffleV3<
    ALayout, BLayout, CLayout,
    ADataType, XPackedDataType, BDataType, XPackedDataType, CDataType, AccDataType, CDataType,
    AElementOp, BElementOp, CElementOp,
    ck::tensor_operation::device::GemmSpecialization::Default,
    ScaleBlockSize,
    256,                    // BlockSize
    256, 256, 128,          // MPerBlock, NPerBlock, KPerBlock
    16, 16,                 // AK1, BK1
    16, 16,                 // MPerXDL, NPerXDL
    16, 4,                  // MXdlPerWave, NXdlPerWave
    S<8, 32, 1>,            // ABlockTransfer
    S<1, 0, 2>,
    S<1, 0, 2>,
    2, 16, 16,
    true,
    S<8, 32, 1>,            // BBlockTransfer
    S<1, 0, 2>,
    S<1, 0, 2>,
    2, 16, 16,
    true,
    2,                      // CShuffleMXdlPerWavePerShuffle
    4,                      // CShuffleNXdlPerWavePerShuffle
    S<1, 8, 1, 32>,
    8,
    Intrawave,
    ck::BlockGemmPipelineVersion::v3,
    ADataType, BDataType>;

template <typename DeviceGemmInstance>
float benchmark_ck(int M, int N, int K, int warmup, int iters) {
    // K is in FP4 elements, allocate K/2 bytes for packed FP4
    size_t a_bytes = (size_t)M * (K / 2);
    size_t b_bytes = (size_t)N * (K / 2);
    size_t c_elems = (size_t)M * N;
    size_t c_bytes = c_elems * sizeof(ck::bhalf_t);

    // Scales: A is [M, K/ScaleBlockSize] row-major, B is [N, K/ScaleBlockSize] col-major
    // But CK expects pre-shuffled scales, so we use the stride convention from the constructor
    size_t a_scale_elems = (size_t)M * (K / ScaleBlockSize);
    size_t b_scale_elems = (size_t)N * (K / ScaleBlockSize);

    // Allocate device memory
    void *d_a, *d_b, *d_c, *d_a_scale, *d_b_scale;
    HIP_CHECK(hipMalloc(&d_a, a_bytes));
    HIP_CHECK(hipMalloc(&d_b, b_bytes));
    HIP_CHECK(hipMalloc(&d_c, c_bytes));
    // Scale elements are E8M0 packed as int32 (4 scales per int32)
    size_t a_scale_bytes = a_scale_elems * sizeof(uint8_t);  // E8M0 = 1 byte each
    size_t b_scale_bytes = b_scale_elems * sizeof(uint8_t);
    // But CK uses int32 as the scale packed type, so pack 4 per int32
    HIP_CHECK(hipMalloc(&d_a_scale, a_scale_bytes));
    HIP_CHECK(hipMalloc(&d_b_scale, b_scale_bytes));

    // Initialize with random data
    {
        std::vector<uint8_t> h_a(a_bytes);
        std::vector<uint8_t> h_b(b_bytes);
        std::mt19937 rng(42);
        for (auto& x : h_a) x = rng() & 0xFF;
        for (auto& x : h_b) x = rng() & 0xFF;
        HIP_CHECK(hipMemcpy(d_a, h_a.data(), a_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, h_b.data(), b_bytes, hipMemcpyHostToDevice));

        // Init scales (all zeros = exponent 0 = scale 1.0)
        HIP_CHECK(hipMemset(d_a_scale, 127, a_scale_bytes));  // E8M0 bias=127 = 2^0 = 1.0
        HIP_CHECK(hipMemset(d_b_scale, 127, b_scale_bytes));
        HIP_CHECK(hipMemset(d_c, 0, c_bytes));
    }

    auto device_gemm = DeviceGemmInstance{};
    auto invoker = device_gemm.MakeInvoker();

    // Strides in FP4 element units
    int StrideA = K;             // A is [M, K/2] packed -> stride = K (in FP4 elements)
    int StrideB = K;             // B is [N, K/2] packed -> stride = K (in FP4 elements)
    int StrideC = N;             // C is [M, N]
    int StrideScaleA = K / ScaleBlockSize;  // A scale stride
    int StrideScaleB = K / ScaleBlockSize;  // B scale stride

    auto argument = device_gemm.MakeArgument(
        static_cast<ADataType*>(d_a),
        static_cast<XPackedDataType*>(d_a_scale),
        static_cast<BDataType*>(d_b),
        static_cast<XPackedDataType*>(d_b_scale),
        static_cast<ck::bhalf_t*>(d_c),
        M, N, K,
        StrideA, StrideScaleA,
        StrideB, StrideScaleB,
        StrideC,
        1,  // KBatch
        AElementOp{}, BElementOp{}, CElementOp{});

    if (!device_gemm.IsSupportedArgument(argument)) {
        std::cerr << "ERROR: GEMM config not supported for M=" << M << " N=" << N << " K=" << K << std::endl;
        hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_a_scale); hipFree(d_b_scale);
        return -1.0f;
    }

    std::cout << "  Config supported, running benchmark..." << std::endl;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        invoker.Run(argument, StreamConfig{nullptr, false});
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    std::vector<float> times(iters);
    for (int i = 0; i < iters; i++) {
        HIP_CHECK(hipEventRecord(start));
        invoker.Run(argument, StreamConfig{nullptr, false});
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        times[i] = ms;
    }

    // Trimmed mean (10% trim)
    std::sort(times.begin(), times.end());
    int trim = iters / 10;
    float sum = 0;
    int count = 0;
    for (int i = trim; i < iters - trim; i++) {
        sum += times[i];
        count++;
    }
    float avg_ms = sum / count;

    // Compute TFLOPS
    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_a_scale); hipFree(d_b_scale);

    return tflops;
}

int main() {
    std::cout << "=== CK MXFP4 GEMM Benchmark ===" << std::endl;
    std::cout << "Testing BLayout=Col (non-preshuffled B, training scenario)" << std::endl;

    int warmup = 200;
    int iters = 500;

    // Test shapes
    struct Shape { int M, N, K; };
    Shape shapes[] = {
        {8192, 8192, 8192},
        {4096, 32768, 128256},
    };

    for (auto& s : shapes) {
        // Ensure K is divisible by ScaleBlockSize and tile sizes
        int K_aligned = (s.K / 256) * 256;  // align to 256 FP4 elements
        if (K_aligned != s.K) {
            std::cout << "\nAdjusting K from " << s.K << " to " << K_aligned << std::endl;
            s.K = K_aligned;
        }

        std::cout << "\n--- Shape: " << s.M << " x " << s.N << " x " << s.K << " ---" << std::endl;

        // Test with BLayout=Col (our training scenario)
        std::cout << "BLayout=Col, WaveMap=8x8:" << std::endl;
        using GemmCol_8x8 = DeviceGemmHelper<Col, B16>;
        float tflops_col = benchmark_ck<GemmCol_8x8>(s.M, s.N, s.K, warmup, iters);
        if (tflops_col > 0) {
            std::cout << "  TFLOPS: " << tflops_col << std::endl;
        }

        std::cout << "BLayout=Col, WaveMap=16x4:" << std::endl;
        using GemmCol_16x4 = DeviceGemmHelper_16x4<Col, B16>;
        float tflops_col_16x4 = benchmark_ck<GemmCol_16x4>(s.M, s.N, s.K, warmup, iters);
        if (tflops_col_16x4 > 0) {
            std::cout << "  TFLOPS: " << tflops_col_16x4 << std::endl;
        }
    }

    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
