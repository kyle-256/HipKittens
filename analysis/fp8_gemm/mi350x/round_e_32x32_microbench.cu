// Round-E microbench gate: 16x16x128 vs 32x32x64 fp8 MFMA prim throughput
// for the SAME 64x32x128 work tile (matches one rcr_mma in the production
// grouped_rcr_kernel main loop).
//
// Question (deciding whether to invest in the full main-loop port — see
// analysis/_notes/round-D-fp8-pmc-grounded-saturation-and-structural-roadmap.md):
//   Does halving the MFMA inst count (32 prims of 16x16x128 ->
//   16 prims of 32x32x64) translate into measurable per-iter throughput
//   gain on the gfx950 SIMD?
//
// Round-D PMC said issue density is 29 % MFMA / 110 inst per outer
// K-iter; halving MFMA -> 16 MFMA / 94 inst -> 17 % issue density.
// The PMC argument was that with same FLOP/cyc per SIMD, fewer issues
// = less issue overhead = same wall. But this microbench DIRECTLY
// measures the per-iter wall to settle the question empirically before
// committing to a multi-round port.
//
// Per-iter workload (mirrors ONE outer K-iter of grouped_rcr_kernel
// with K_BLOCK=128 and 4 accumulators cA/cB/cC/cD on a single warp):
//   16 ds_read_b128 for A (64 rows * 128 K-bytes = 8192 B / warp)
//      x 2 loads (one for cA/cB pair sharing A, one for cC/cD pair)
//      total = 32 ds_read_b128 for A
//      Wait — re-derived: 64x128 fp8 = 8192 B / warp = 128 B/lane =
//      8 ds_read_b128 per warp per A-load. x2 distinct A's = 16 reads.
//    8 ds_read_b128 for B (32 rows * 128 K-bytes = 4096 B / warp =
//      64 B/lane = 4 ds_read_b128 per warp per B-load. x2 (b0, b1) = 8.
//   Total LDS reads/iter = 24 (identical between both paths).
//
//   Path (a) 16x16x128: 4 accs * 8 prims/acc = 32 prim mfmas
//     each prim takes intx8_t A + intx8_t B + floatx4_t C
//   Path (b) 32x32x64:  4 accs * (2 height * 1 width) * 2 inner-K
//                     = 16 prim mfmas
//     each prim takes intx8_t A + intx8_t B + floatx16_t C
//
// Total FLOPs per iter (both paths): 4 * 64 * 32 * 128 * 2 = 2.097 MFLOP.
// Per-warp peak (CDNA4 fp8) = 5033 TFLOPS / 256 CU / 4 SIMD * 2 waves
//   ~= 9.83 GFLOP/sec/wave (note: oversimplified, real peak per SIMD
//    pipeline is ~4096 FLOP/cyc).
//
// Decision rule (committed to a-priori):
//   tflops(b) >= tflops(a) * 1.05  -> Round-E CONFIRMED (>=5% wall gain
//     justifies the 2-3 round port effort)
//   tflops(b) in [tflops(a) * 0.97, * 1.05] -> AMBIGUOUS, run full
//     kernel A/B with prototype to break tie
//   tflops(b) < tflops(a) * 0.97 -> Round-E FALSIFIED (32x32 prim
//     is no faster or slower; structural ceiling confirmed)
//
// LIMITATIONS (must be acknowledged in the round-E note):
//   * SINGLE-WAVE microbench. Real kernel runs 8 waves/CTA + barriers.
//     Microbench under-reports cross-wave sync overhead. The DELTA
//     between (a) and (b) is what matters.
//   * No LDS swizzle. Real kernel uses st_16x128_v2 swizzle; both paths
//     in this bench use plain consecutive offsets, so any swizzle-
//     related differential cost is not captured.
//   * Single accum tile fully resident in registers; no cross-iter
//     register pressure modeled.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;
typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

#ifndef N_INNER
#define N_INNER 50000
#endif

constexpr int LDS_BYTES = 32 * 1024;
constexpr int LDS_ELEMS = LDS_BYTES / sizeof(intx8_t);

__device__ __managed__ int g_drain[16];

// ============================================================================
// Path (a): 32 prims of mfma_f32_16x16x128_f8f6f4 per iter.
//
// Mirrors the production rcr_mma 16x16x128 path: 4 accs (cA/cB/cC/cD),
// each contributing 8 prim mfmas. Layout: cA/cB share one A operand,
// cC/cD share a different A operand. cA/cC share b0, cB/cD share b1.
// Each acc has 8 fragments (rt_fl<64,32,col_l,rt_16x16_s>: height=4,
// width=2 = 8 cells).
// ============================================================================
__global__ __launch_bounds__(64, 8)
void bench_a_16x16x128(const intx8_t* __restrict__ init_data, int N) {
    __shared__ intx8_t lds_buf[LDS_ELEMS];
    for (int i = threadIdx.x; i < LDS_ELEMS; i += 64) {
        lds_buf[i] = init_data[i & 63];
    }
    __builtin_amdgcn_s_barrier();

    // 4 accumulators x 8 cells each = 32 floatx4_t fragments per warp.
    // Mirrors rt_fl<64,32,col_l,rt_16x16_s>::data layout.
    floatx4_t cA[8] = {{0,0,0,0}}; floatx4_t cB[8] = {{0,0,0,0}};
    floatx4_t cC[8] = {{0,0,0,0}}; floatx4_t cD[8] = {{0,0,0,0}};

    int idx = threadIdx.x & 63;

    for (int iter = 0; iter < N; iter++) {
        // Mirror real kernel: 2 A loads (cA/cB pair, cC/cD pair) +
        // 2 B loads (b0, b1).
        intx8_t a0 = lds_buf[idx];           // 8 ds_read_b128 worth (intx8 = 32B/lane = 2 b128)
        intx8_t a1 = lds_buf[(idx + 1) & 63];
        intx8_t b0 = lds_buf[(idx + 2) & 63];
        intx8_t b1 = lds_buf[(idx + 3) & 63];
        // NOTE: 4 intx8_t reads = 4 * 2 = 8 ds_read_b128 in codegen
        // (intx8 = 32 B per lane = 2 ds_read_b128). The real per-iter
        // LDS read count (24) is approximated here as 8 to match
        // single-wave microbench scaling — what matters is the
        // mfma-prim DELTA, not absolute LDS bandwidth.

        asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(a0[0]), "+v"(a1[0]), "+v"(b0[0]), "+v"(b1[0]));

        // 4 accs x 8 prims = 32 prim mfmas.
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            cA[c] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a0, b0, cA[c], 0,0,0,0,0,0);
            cB[c] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a0, b1, cB[c], 0,0,0,0,0,0);
            cC[c] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a1, b0, cC[c], 0,0,0,0,0,0);
            cD[c] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a1, b1, cD[c], 0,0,0,0,0,0);
        }

        idx = (idx + 4) & (LDS_ELEMS - 1);
    }

    if (threadIdx.x == 0) {
        float s = 0;
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            s += cA[c][0] + cB[c][0] + cC[c][0] + cD[c][0];
        }
        g_drain[0] = (int)s;
    }
}

// ============================================================================
// Path (b): 16 prims of mfma_scale_f32_32x32x64_f8f6f4 per iter.
//
// Same per-iter total work (4 accs * 64x32x128 = 2.097 MFLOP), but
// expressed as 16 prims of larger MFMA. Per-acc layout:
//   rt_fl<64,32,col_l,rt_32x32_s>: height=2, width=1 -> 2 cells per acc.
//   each cell needs 2 inner-K iters (K_BLOCK=128 / prim_K=64) -> 4 prims
//   per acc.
//   4 accs * 4 prims = 16 prims.
//
// Each cell stores floatx16_t (16 fp32/lane), matching the prim output
// shape.
// ============================================================================
__global__ __launch_bounds__(64, 8)
void bench_b_32x32x64(const intx8_t* __restrict__ init_data, int N) {
    __shared__ intx8_t lds_buf[LDS_ELEMS];
    for (int i = threadIdx.x; i < LDS_ELEMS; i += 64) {
        lds_buf[i] = init_data[i & 63];
    }
    __builtin_amdgcn_s_barrier();

    // 4 accs x 2 cells = 8 floatx16_t fragments. Same per-acc fp32/lane
    // count (16 fp32/lane * 2 cells = 32 fp32/lane, matching path a's
    // 4 fp32/lane * 8 cells).
    floatx16_t cA[2] = {{0}}; floatx16_t cB[2] = {{0}};
    floatx16_t cC[2] = {{0}}; floatx16_t cD[2] = {{0}};

    int idx = threadIdx.x & 63;

    for (int iter = 0; iter < N; iter++) {
        // Same load pattern: 4 intx8_t loads (= 8 ds_read_b128).
        intx8_t a0 = lds_buf[idx];
        intx8_t a1 = lds_buf[(idx + 1) & 63];
        intx8_t b0 = lds_buf[(idx + 2) & 63];
        intx8_t b1 = lds_buf[(idx + 3) & 63];

        asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(a0[0]), "+v"(a1[0]), "+v"(b0[0]), "+v"(b1[0]));

        // 4 accs x 2 cells x 2 inner-K = 16 prim mfmas.
        // For 32x32x64 the K-prim consumes 64 fp8 (= half an intx8_t,
        // i.e. half of a single ds_read_b128). To match per-iter total
        // K=128, we reuse a0/a1/b0/b1 (each intx8 = 32 fp8 in this lane
        // mapping) twice — once per inner-K. The microbench is throughput-
        // only (no correctness check) so the data is intentionally reused.
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            cA[c] = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a0, b0, cA[c], 0,0,0,0,0,0);
            cB[c] = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a0, b1, cB[c], 0,0,0,0,0,0);
            cC[c] = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a1, b0, cC[c], 0,0,0,0,0,0);
            cD[c] = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a1, b1, cD[c], 0,0,0,0,0,0);
            // Inner-K iter 1 (re-use loads — see comment above).
            cA[c] = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a0, b0, cA[c], 0,0,0,0,0,0);
            cB[c] = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a0, b1, cB[c], 0,0,0,0,0,0);
            cC[c] = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a1, b0, cC[c], 0,0,0,0,0,0);
            cD[c] = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a1, b1, cD[c], 0,0,0,0,0,0);
        }

        idx = (idx + 4) & (LDS_ELEMS - 1);
    }

    if (threadIdx.x == 0) {
        float s = 0;
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            s += cA[c][0] + cB[c][0] + cC[c][0] + cD[c][0];
        }
        g_drain[1] = (int)s;
    }
}

static float median_of(float* arr, int n) {
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (arr[j] < arr[i]) { float t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
    return arr[n / 2];
}

int main(int argc, char** argv) {
    int N = N_INNER;
    if (argc >= 2) N = atoi(argv[1]);

    intx8_t* d_init = nullptr;
    hipMalloc(&d_init, 64 * sizeof(intx8_t));
    hipMemset(d_init, 0x12, 64 * sizeof(intx8_t));

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    constexpr int N_TRIALS = 11;

    bench_a_16x16x128<<<1, 64>>>(d_init, N);
    bench_b_32x32x64<<<1, 64>>>(d_init, N);
    hipDeviceSynchronize();

    float t_a[N_TRIALS], t_b[N_TRIALS];
    for (int i = 0; i < N_TRIALS; i++) {
        hipEventRecord(start);
        bench_a_16x16x128<<<1, 64>>>(d_init, N);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&t_a[i], start, stop);
    }
    for (int i = 0; i < N_TRIALS; i++) {
        hipEventRecord(start);
        bench_b_32x32x64<<<1, 64>>>(d_init, N);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&t_b[i], start, stop);
    }
    float t_a_med = median_of(t_a, N_TRIALS);
    float t_b_med = median_of(t_b, N_TRIALS);

    float per_iter_a = (t_a_med * 1e6f) / N;
    float per_iter_b = (t_b_med * 1e6f) / N;

    constexpr double mac_per_iter = 4.0 * 64.0 * 32.0 * 128.0;
    constexpr double flops_per_iter = mac_per_iter * 2.0;
    double tflops_a = flops_per_iter / (per_iter_a * 1e-9) / 1e12;
    double tflops_b = flops_per_iter / (per_iter_b * 1e-9) / 1e12;

    double pct = 100.0 * (tflops_b - tflops_a) / tflops_a;

    printf("Round-E microbench gate (16x16x128 vs 32x32x64 fp8 MFMA)\n");
    printf("  N inner iter         = %d, trials = %d (median)\n", N, N_TRIALS);
    printf("  Workload per iter    = 4 accs * 64x32x128 = 2.097 MFLOP (single wave)\n");
    printf("\n");
    printf("  (a) 16x16x128 (32 prims) = %.3f ms / %d -> %.2f ns/iter -> %.1f TFLOPS\n",
           t_a_med, N, per_iter_a, tflops_a);
    printf("  (b) 32x32x64  (16 prims) = %.3f ms / %d -> %.2f ns/iter -> %.1f TFLOPS\n",
           t_b_med, N, per_iter_b, tflops_b);
    printf("\n");
    printf("  Per-iter delta: 32x32x64 vs 16x16x128 = %+.2f %% throughput\n", pct);
    printf("  Decision (Round-E gate, committed pre-measurement):\n");
    printf("    pct >= +5.0 %%  -> Round-E CONFIRMED. Invest in main-loop port.\n");
    printf("    pct in (-3,+5) -> AMBIGUOUS. Run full kernel A/B prototype.\n");
    printf("    pct <= -3.0 %% -> Round-E FALSIFIED. Structural ceiling confirmed.\n");
    printf("  Verdict: %s\n",
           pct >= 5.0 ? "PASS - 32x32x64 wins, port worth investing"
                      : (pct > -3.0 ? "AMBIGUOUS - need full kernel prototype"
                                    : "FAIL - 32x32x64 no faster, port not worth it"));

    hipFree(d_init);
    hipEventDestroy(start); hipEventDestroy(stop);
    return 0;
}
