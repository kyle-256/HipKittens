// Lever D microbench gate (R64-dm prerequisite).
// Compares throughput of mfma_scale_f32_16x16x128_f8f6f4 (current kernel)
// vs mfma_scale_f32_32x32x64_f8f6f4 (hypothetical Lever D port).
// Both are normalised to cover an equivalent 64x32 fp32 output region
// with K=128 accumulation depth, matching the per-warp accum footprint
// in grouped_rcr_kernel main loop (4× cA/cB/cC/cD = 4× rt_fl<64,32>).
//
// Workload per "iter":
//   16x16x128 path: 8 mfma calls (4 height × 2 width × 1 K-chain)
//   32x32x64 path:  4 mfma calls (2 height × 1 width × 2 K-chain)
// Both accumulate 32 dw/lane of fp32 output (8×4 vs 2×16).

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;
typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

#ifndef N_ITERS
#define N_ITERS 10000
#endif

// 8 mfma_16x16x128 per iter — mirrors the per-K-iter call count for one
// 64x32 acc region (4 height × 2 width = 8 base tiles in current kernel).
__global__ __launch_bounds__(64, 8)
void bench_16x16x128(const intx8_t* __restrict__ a_buf,
                     const intx8_t* __restrict__ b_buf,
                     float* __restrict__ c_buf,
                     int N) {
    intx8_t A = a_buf[threadIdx.x];
    intx8_t B = b_buf[threadIdx.x];
    floatx4_t C0 = {0,0,0,0}, C1 = {0,0,0,0};
    floatx4_t C2 = {0,0,0,0}, C3 = {0,0,0,0};
    floatx4_t C4 = {0,0,0,0}, C5 = {0,0,0,0};
    floatx4_t C6 = {0,0,0,0}, C7 = {0,0,0,0};

    for (int i = 0; i < N; i++) {
        // 8 mfma_16x16x128 — each accumulates K=128 into one 16x16 tile.
        // No K-chain (each call is a fresh K=128 contribution).
        C0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            A, B, C0, 0, 0, 0, 0, 0, 0);
        C1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            A, B, C1, 0, 0, 0, 0, 0, 0);
        C2 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            A, B, C2, 0, 0, 0, 0, 0, 0);
        C3 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            A, B, C3, 0, 0, 0, 0, 0, 0);
        C4 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            A, B, C4, 0, 0, 0, 0, 0, 0);
        C5 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            A, B, C5, 0, 0, 0, 0, 0, 0);
        C6 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            A, B, C6, 0, 0, 0, 0, 0, 0);
        C7 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            A, B, C7, 0, 0, 0, 0, 0, 0);
    }

    // Force result to be live (prevent DCE).
    *(floatx4_t*)&c_buf[threadIdx.x * 32 + 0]  = C0;
    *(floatx4_t*)&c_buf[threadIdx.x * 32 + 4]  = C1;
    *(floatx4_t*)&c_buf[threadIdx.x * 32 + 8]  = C2;
    *(floatx4_t*)&c_buf[threadIdx.x * 32 + 12] = C3;
    *(floatx4_t*)&c_buf[threadIdx.x * 32 + 16] = C4;
    *(floatx4_t*)&c_buf[threadIdx.x * 32 + 20] = C5;
    *(floatx4_t*)&c_buf[threadIdx.x * 32 + 24] = C6;
    *(floatx4_t*)&c_buf[threadIdx.x * 32 + 28] = C7;
}

// 4 mfma_32x32x64 per iter — covers the same 64x32 output × K=128 region.
// 2 height (64/32) × 1 width (32/32) × 2 K-chain (K=64 per mfma → 2 mfma
// per K=128 accumulation per tile).
__global__ __launch_bounds__(64, 8)
void bench_32x32x64(const intx8_t* __restrict__ a_buf,
                    const intx8_t* __restrict__ b_buf,
                    float* __restrict__ c_buf,
                    int N) {
    intx8_t A = a_buf[threadIdx.x];
    intx8_t B = b_buf[threadIdx.x];
    floatx16_t C0 = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    floatx16_t C1 = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    for (int i = 0; i < N; i++) {
        // 4 mfma_32x32x64 — each covers K=64 of one 32x32 tile.
        // K-chain: C0 accumulates two K=64 calls into K=128, same for C1.
        C0 = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            A, B, C0, 0, 0, 0, 0, 0, 0);
        C1 = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            A, B, C1, 0, 0, 0, 0, 0, 0);
        C0 = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            A, B, C0, 0, 0, 0, 0, 0, 0);
        C1 = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            A, B, C1, 0, 0, 0, 0, 0, 0);
    }

    *(floatx16_t*)&c_buf[threadIdx.x * 32 + 0]  = C0;
    *(floatx16_t*)&c_buf[threadIdx.x * 32 + 16] = C1;
}

// Helper to compute median of 11 trial timings.
static float median_of(float* arr, int n) {
    // Simple bubble sort for tiny n.
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[i]) {
                float t = arr[i]; arr[i] = arr[j]; arr[j] = t;
            }
        }
    }
    return arr[n / 2];
}

int main(int argc, char** argv) {
    int N = N_ITERS;
    if (argc >= 2) N = atoi(argv[1]);

    intx8_t* d_a = nullptr;
    intx8_t* d_b = nullptr;
    float* d_c = nullptr;
    hipMalloc(&d_a, 64 * sizeof(intx8_t));
    hipMalloc(&d_b, 64 * sizeof(intx8_t));
    hipMalloc(&d_c, 64 * 32 * sizeof(float));
    hipMemset(d_a, 0, 64 * sizeof(intx8_t));
    hipMemset(d_b, 0, 64 * sizeof(intx8_t));
    hipMemset(d_c, 0, 64 * 32 * sizeof(float));

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    constexpr int N_TRIALS = 11;

    // Warmup
    bench_16x16x128<<<1, 64>>>(d_a, d_b, d_c, N);
    bench_32x32x64 <<<1, 64>>>(d_a, d_b, d_c, N);
    hipDeviceSynchronize();

    // Measure 16x16x128
    float t_16[N_TRIALS];
    for (int i = 0; i < N_TRIALS; i++) {
        hipEventRecord(start);
        bench_16x16x128<<<1, 64>>>(d_a, d_b, d_c, N);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&t_16[i], start, stop);
    }
    float t16_med = median_of(t_16, N_TRIALS);

    // Measure 32x32x64
    float t_32[N_TRIALS];
    for (int i = 0; i < N_TRIALS; i++) {
        hipEventRecord(start);
        bench_32x32x64<<<1, 64>>>(d_a, d_b, d_c, N);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&t_32[i], start, stop);
    }
    float t32_med = median_of(t_32, N_TRIALS);

    // Per-iter wall (ns) — N iters × kernel-call overhead amortised.
    float per_iter_16 = (t16_med * 1e6f) / N;  // ms→ns scale
    float per_iter_32 = (t32_med * 1e6f) / N;

    // Throughput = 64x32 output × K=128 / per_iter
    // = 524288 mac ops / per_iter ns = TFLOPS
    constexpr double mac_per_iter = 64.0 * 32.0 * 128.0;     // 262144 macs
    constexpr double flops_per_iter = mac_per_iter * 2.0;     // 524288 fp ops
    double tflops_16 = flops_per_iter / (per_iter_16 * 1e-9) / 1e12;
    double tflops_32 = flops_per_iter / (per_iter_32 * 1e-9) / 1e12;

    double pct_advantage = 100.0 * (tflops_32 - tflops_16) / tflops_16;

    printf("Lever D microbench gate (R64-dm prerequisite)\n");
    printf("  N iters per kernel = %d, trials = %d (median)\n", N, N_TRIALS);
    printf("  Workload per iter  = 64x32 output @ K=128 fp8e4m3 input -> fp32 acc\n");
    printf("  16x16x128 path     = 8 mfma calls, 8x floatx4 acc = 32 dw/lane\n");
    printf("  32x32x64  path     = 4 mfma calls, 2x floatx16 acc = 32 dw/lane\n");
    printf("\n");
    printf("  16x16x128 wall (median) = %.3f ms / %d iter = %.2f ns/iter -> %.1f TFLOPS\n",
           t16_med, N, per_iter_16, tflops_16);
    printf("  32x32x64  wall (median) = %.3f ms / %d iter = %.2f ns/iter -> %.1f TFLOPS\n",
           t32_med, N, per_iter_32, tflops_32);
    printf("\n");
    printf("  Per-iter delta: 32x32x64 vs 16x16x128 = %+.2f %% throughput\n",
           pct_advantage);
    printf("  Decision threshold (R64-dm gate)      = >= +3.00 %%\n");
    printf("  Verdict: %s\n",
           pct_advantage >= 3.0
               ? "PASS - proceed with Lever D R6+ full port (4-6 rounds)"
               : "FAIL - abandon Lever D, accept plateau ~956-962 / FP8 1.12");

    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    hipEventDestroy(start); hipEventDestroy(stop);
    return 0;
}
