// R34 SCOUT: mfma_323264 vs mfma_1616128 single-warp throughput microbench.
//
// Validation gate per round-64-dm-fp8-grouped-lever-d-rb-step-1-st-32x64-type-LANDED.md:
// "If mfma_323264 is < 3 pp faster per-iter (single-warp, no LDS overhead),
//  abandon Lever D full port."
//
// Each kernel does ~1M MFMA issues per warp in a tight chained-accumulator
// loop. With WARPS_PER_CU = 8 (one per WGP-half) we approximate the kernel's
// per-CU compute density. Wall-clock is converted to FLOPs/s using the
// per-MFMA FLOP count.

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

// 64 lanes per wave on gfx950
constexpr int WARP_THREADS = 64;
// 8 warps per WG to mirror kernel_fp8_layouts grouped_rcr WG layout
constexpr int WARPS_PER_WG = 8;
// 1 WG per CU for persistent kernel
constexpr int NUM_CUS_USED = 256;  // 32 CUs/XCD * 8 XCDs on MI355X

constexpr int ITERS = 8000;        // mfma calls per warp per dispatch
constexpr int UNROLL = 32;          // unroll factor (helps issue rate)

// FLOPs per single mfma_1616128_f8f6f4 = 16 * 16 * 128 * 2 (FMA = 2 ops)
constexpr long long FLOPS_PER_1616128 = 16LL * 16 * 128 * 2;
// FLOPs per single mfma_3232x64_f8f6f4 = 32 * 32 * 64 * 2
constexpr long long FLOPS_PER_3232x64 = 32LL * 32 * 64 * 2;

typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;
typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

// ------------------------------------------------------------------------
// Kernel A: tight loop of mfma_1616128
// ------------------------------------------------------------------------
__global__ __launch_bounds__(WARPS_PER_WG * WARP_THREADS, 1)
void mfma_1616128_loop_kernel(float* __restrict__ sink) {
    intx8_t a; intx8_t b; floatx4_t c;
    // Initialize from thread index so compiler can't constant-fold
    int tid = threadIdx.x;
    for (int i = 0; i < 8; i++) ((int*)&a)[i] = tid * 7 + i;
    for (int i = 0; i < 8; i++) ((int*)&b)[i] = tid * 11 + i;
    for (int i = 0; i < 4; i++) ((float*)&c)[i] = (float)tid * 0.0001f;

    #pragma unroll 1
    for (int it = 0; it < ITERS; it += UNROLL) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                a, b, c, 0, 0, 0, 0, 0, 0);
        }
    }

    // Sink so compiler keeps c alive
    sink[blockIdx.x * blockDim.x + tid] = ((float*)&c)[0] + ((float*)&c)[1] +
                                            ((float*)&c)[2] + ((float*)&c)[3];
}

// ------------------------------------------------------------------------
// Kernel B: tight loop of mfma_3232x64
// ------------------------------------------------------------------------
__global__ __launch_bounds__(WARPS_PER_WG * WARP_THREADS, 1)
void mfma_3232x64_loop_kernel(float* __restrict__ sink) {
    intx8_t a; intx8_t b; floatx16_t c;
    int tid = threadIdx.x;
    for (int i = 0; i < 8; i++) ((int*)&a)[i] = tid * 7 + i;
    for (int i = 0; i < 8; i++) ((int*)&b)[i] = tid * 11 + i;
    for (int i = 0; i < 16; i++) ((float*)&c)[i] = (float)tid * 0.0001f;

    #pragma unroll 1
    for (int it = 0; it < ITERS; it += UNROLL) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a, b, c, 0, 0, 0, 0, 0, 0);
        }
    }

    float acc = 0;
    for (int i = 0; i < 16; i++) acc += ((float*)&c)[i];
    sink[blockIdx.x * blockDim.x + tid] = acc;
}

// ------------------------------------------------------------------------
// Host driver
// ------------------------------------------------------------------------
template<typename Kfn>
double bench_kernel(Kfn kfn, float* sink, const char* label, long long flops_per_mfma) {
    // Warmup
    for (int i = 0; i < 3; i++) {
        kfn<<<NUM_CUS_USED, WARPS_PER_WG * WARP_THREADS>>>(sink);
    }
    hipDeviceSynchronize();

    constexpr int RUNS = 5;
    double best_ms = 1e9;
    for (int r = 0; r < RUNS; r++) {
        hipEvent_t e0, e1;
        hipEventCreate(&e0); hipEventCreate(&e1);
        hipEventRecord(e0);
        kfn<<<NUM_CUS_USED, WARPS_PER_WG * WARP_THREADS>>>(sink);
        hipEventRecord(e1);
        hipEventSynchronize(e1);
        float ms;
        hipEventElapsedTime(&ms, e0, e1);
        hipEventDestroy(e0); hipEventDestroy(e1);
        if (ms < best_ms) best_ms = ms;
    }

    long long total_mfma = (long long)NUM_CUS_USED * WARPS_PER_WG * ITERS;
    double total_flops = (double)total_mfma * (double)flops_per_mfma;
    double tflops = total_flops / (best_ms * 1e-3) / 1e12;
    printf("%-20s best_ms=%8.4f  mfma_count=%10lld  TFLOPS=%8.1f\n",
            label, best_ms, total_mfma, tflops);
    return tflops;
}

int main() {
    int dev_count;
    hipGetDeviceCount(&dev_count);
    printf("hipGetDeviceCount=%d, using device 0\n", dev_count);

    float* sink;
    size_t sink_bytes = (size_t)NUM_CUS_USED * WARPS_PER_WG * WARP_THREADS * sizeof(float);
    hipMalloc(&sink, sink_bytes);
    hipMemset(sink, 0, sink_bytes);

    printf("\n=== R34 mfma microbench (single-warp throughput, no LDS) ===\n");
    printf("Config: %d CUs * %d warps/WG * %d MFMA/warp\n", NUM_CUS_USED, WARPS_PER_WG, ITERS);
    printf("FLOPS_PER_1616128 = %lld\n", FLOPS_PER_1616128);
    printf("FLOPS_PER_3232x64 = %lld\n\n", FLOPS_PER_3232x64);

    double tflops_16 = bench_kernel(mfma_1616128_loop_kernel,    sink, "mfma_16x16x128", FLOPS_PER_1616128);
    double tflops_32 = bench_kernel(mfma_3232x64_loop_kernel,    sink, "mfma_32x32x64",  FLOPS_PER_3232x64);

    double ratio_pp = (tflops_32 - tflops_16) / tflops_16 * 100.0;
    printf("\nRESULT: 32x32x64 vs 16x16x128 throughput delta = %+.2f%% (gate at +3pp)\n", ratio_pp);
    printf("VERDICT: %s\n", (ratio_pp >= 3.0) ? "PROCEED with Lever D full port" : "ABANDON Lever D full port");

    hipFree(sink);
    return 0;
}
