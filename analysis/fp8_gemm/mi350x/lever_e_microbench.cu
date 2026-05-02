// Lever E microbench gate (R11 prepared, R12 to run + decide).
//
// Question: can software-pipelining the ds_read -> mfma chain
// (currently serialised by `s_waitcnt lgkmcnt(0); s_barrier;` fences in
// kernel_fp8_layouts.cpp::main_loop_iter, lines 1576-1638) give the
// 8-wave grouped FP8 main-loop a measurable per-iter throughput gain?
//
// Hypothesis: the inner main-loop chunk emits ~8 ds_reads followed
// by an `s_waitcnt lgkmcnt(0)` then 8 mfma_scale_f32_16x16x128_f8f6f4
// calls before the next chunk. This serialises the ~14 cy ds_read
// latency with the ~13 cy/mfma pipeline, leaving roughly 8 cy/iter of
// LDS-stall headroom that Lever E proposes to recover by issuing
// next-iter LDS reads while the current iter's mfma's execute.
//
// Per-iter workload mirrors a single 64x32 acc region in the real kernel:
//   - 2 intx8 LDS reads (= effectively 4 ds_read_b128 in codegen)
//   - 8 mfma_16x16x128 calls
//   - loop induction
//
// Two paths:
//   path (a) "current style": LDS read x2 -> wait lgkmcnt(0)
//                             -> mfma x8 (no cross-iter prefetch)
//   path (b) "pipelined"    : double-buffered. iter k issues LDS read
//                             for iter k+1 BEFORE its own 8 mfma's,
//                             and uses lgkmcnt(>0) drain semantics so
//                             iter k's mfma's overlap with k+1's reads.
//
// Decision rule (R12 will execute):
//   - throughput(b) >= +5.0 pp throughput(a) -> Lever E CONFIRMED,
//     proceed to in-situ kernel rewrite (LDS read prefetch hoist into
//     prior iter's epilogue + reordered s_waitcnt mask).
//   - throughput(b) within +/- 3.0 pp        -> Lever E FALSIFIED.
//     Plateau accepted at score ~962-963. Switch agenda to backward-
//     only optimisations (bench_grouped_gemm_turbo --bwd) which the
//     forward metric does not exercise.
//   - throughput(b) <= -3.0 pp                -> Lever E WORSE
//     (LLVM was already pipelining better than hand schedule).
//     FALSIFIED. Same agenda switch as above.
//
// Build:
//   hipcc lever_e_microbench.cu -o lever_e_microbench --offload-arch=gfx950 -O3
// Run:
//   ./lever_e_microbench [N_INNER=10000]
//
// IMPORTANT LIMITATIONS (R12 must read before trusting numbers):
//   * SINGLE-WAVE microbench. Real grouped_rcr_kernel runs 8 waves/CTA
//     with cross-wave s_barriers. Barrier cost between waves is NOT
//     captured here; the absolute numbers under-report main-loop wall
//     on the real kernel. The DELTA between (a) and (b) is what
//     matters for the gate decision.
//   * LDS scratch is sized to 32 KB (smaller than the real 137 KB).
//     We don't model full LDS pressure here; per-wave LDS read pattern
//     is identical regardless of slab count.
//   * `ds_read` swizzle pattern is NOT modelled (real kernel uses
//     `st_16x128_v2_s` swizzled offsets via `prefill_swizzled_offsets`).
//     We use plain consecutive offsets here. This may UNDER-report the
//     gain because swizzled reads have higher LDS bank-conflict costs
//     that pipelining helps more.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;

#ifndef N_INNER
#define N_INNER 10000
#endif

constexpr int LDS_BYTES = 32 * 1024;
constexpr int LDS_ELEMS = LDS_BYTES / sizeof(intx8_t);  // = 1024 intx8 elems

// Use a global volatile to prevent LLVM from constant-folding the LDS
// content out of the ds_reads.
__device__ __managed__ int g_drain[16];

// Path (a): "current style" — 2 LDS reads followed by 8 mfma per iter.
__global__ __launch_bounds__(64, 8)
void bench_a_serial(const intx8_t* __restrict__ init_data, int N) {
    __shared__ intx8_t lds_buf[LDS_ELEMS];

    // Initialise LDS so subsequent reads do not return garbage. Pull
    // the seed from device memory so LLVM can't predict the result.
    for (int i = threadIdx.x; i < LDS_ELEMS; i += 64) {
        lds_buf[i] = init_data[i & 63];
    }
    __builtin_amdgcn_s_barrier();

    // 4 accum tiles to mirror cA/cB/cC/cD pressure (32 dw/lane).
    floatx4_t C0 = {0,0,0,0}, C1 = {0,0,0,0};
    floatx4_t C2 = {0,0,0,0}, C3 = {0,0,0,0};
    floatx4_t C4 = {0,0,0,0}, C5 = {0,0,0,0};
    floatx4_t C6 = {0,0,0,0}, C7 = {0,0,0,0};

    int idx = threadIdx.x & 63;

    for (int iter = 0; iter < N; iter++) {
        // 2 LDS reads of intx8 each. LLVM will emit ds_read_b128 x4 +
        // a single s_waitcnt lgkmcnt(0) before the mfmas (no per-read
        // wait).
        intx8_t A = lds_buf[idx];
        intx8_t B = lds_buf[(idx + 1) & 63];

        // Force the loads to be live and treated as a single barrier
        // before mfma's (mirrors the kernel's `s_waitcnt lgkmcnt(0)`).
        asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(A[0]), "+v"(B[0]));

        C0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A, B, C0, 0,0,0,0,0,0);
        C1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A, B, C1, 0,0,0,0,0,0);
        C2 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A, B, C2, 0,0,0,0,0,0);
        C3 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A, B, C3, 0,0,0,0,0,0);
        C4 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A, B, C4, 0,0,0,0,0,0);
        C5 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A, B, C5, 0,0,0,0,0,0);
        C6 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A, B, C6, 0,0,0,0,0,0);
        C7 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A, B, C7, 0,0,0,0,0,0);

        idx = (idx + 2) & (LDS_ELEMS - 1);
    }

    // Force result live (prevent DCE).
    if (threadIdx.x == 0) {
        floatx4_t s = C0 + C1 + C2 + C3 + C4 + C5 + C6 + C7;
        g_drain[0] = (int)s[0];
        g_drain[1] = (int)s[1];
    }
}

// Path (b): "pipelined" — overlap iter k+1's LDS read with iter k's mfma.
// We force the schedule by reading next-iter buffers before the mfmas
// fire and split the lgkmcnt drain.
__global__ __launch_bounds__(64, 8)
void bench_b_pipelined(const intx8_t* __restrict__ init_data, int N) {
    __shared__ intx8_t lds_buf[LDS_ELEMS];
    for (int i = threadIdx.x; i < LDS_ELEMS; i += 64) {
        lds_buf[i] = init_data[i & 63];
    }
    __builtin_amdgcn_s_barrier();

    floatx4_t C0 = {0,0,0,0}, C1 = {0,0,0,0};
    floatx4_t C2 = {0,0,0,0}, C3 = {0,0,0,0};
    floatx4_t C4 = {0,0,0,0}, C5 = {0,0,0,0};
    floatx4_t C6 = {0,0,0,0}, C7 = {0,0,0,0};

    int idx = threadIdx.x & 63;

    // Prologue: load iter 0.
    intx8_t A_curr = lds_buf[idx];
    intx8_t B_curr = lds_buf[(idx + 1) & 63];
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(A_curr[0]), "+v"(B_curr[0]));
    idx = (idx + 2) & (LDS_ELEMS - 1);

    // Main loop: iter 0..N-2 prefetch, iter N-1 drains.
    for (int iter = 0; iter < N - 1; iter++) {
        // Issue iter+1 prefetch BEFORE this iter's mfma's.
        intx8_t A_next = lds_buf[idx];
        intx8_t B_next = lds_buf[(idx + 1) & 63];
        // No lgkmcnt drain here — mfma's run while the new reads are
        // in flight. The wait happens AFTER the mfma's.

        C0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A_curr, B_curr, C0, 0,0,0,0,0,0);
        C1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A_curr, B_curr, C1, 0,0,0,0,0,0);
        C2 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A_curr, B_curr, C2, 0,0,0,0,0,0);
        C3 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A_curr, B_curr, C3, 0,0,0,0,0,0);
        C4 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A_curr, B_curr, C4, 0,0,0,0,0,0);
        C5 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A_curr, B_curr, C5, 0,0,0,0,0,0);
        C6 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A_curr, B_curr, C6, 0,0,0,0,0,0);
        C7 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A_curr, B_curr, C7, 0,0,0,0,0,0);

        // Drain reads only now — they should have completed during the
        // mfma execution (~8 mfma * 13 cy = 104 cy >> 14 cy ds_read).
        asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(A_next[0]), "+v"(B_next[0]));
        A_curr = A_next;
        B_curr = B_next;
        idx = (idx + 2) & (LDS_ELEMS - 1);
    }

    // Final iter: just consume.
    C0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A_curr, B_curr, C0, 0,0,0,0,0,0);
    C1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A_curr, B_curr, C1, 0,0,0,0,0,0);
    C2 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A_curr, B_curr, C2, 0,0,0,0,0,0);
    C3 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A_curr, B_curr, C3, 0,0,0,0,0,0);
    C4 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A_curr, B_curr, C4, 0,0,0,0,0,0);
    C5 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A_curr, B_curr, C5, 0,0,0,0,0,0);
    C6 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A_curr, B_curr, C6, 0,0,0,0,0,0);
    C7 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(A_curr, B_curr, C7, 0,0,0,0,0,0);

    if (threadIdx.x == 0) {
        floatx4_t s = C0 + C1 + C2 + C3 + C4 + C5 + C6 + C7;
        g_drain[2] = (int)s[0];
        g_drain[3] = (int)s[1];
    }
}

static float median_of(float* arr, int n) {
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
    int N = N_INNER;
    if (argc >= 2) N = atoi(argv[1]);

    intx8_t* d_init = nullptr;
    hipMalloc(&d_init, 64 * sizeof(intx8_t));
    hipMemset(d_init, 0x12, 64 * sizeof(intx8_t));

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    constexpr int N_TRIALS = 11;

    bench_a_serial<<<1, 64>>>(d_init, N);
    bench_b_pipelined<<<1, 64>>>(d_init, N);
    hipDeviceSynchronize();

    float t_a[N_TRIALS], t_b[N_TRIALS];
    for (int i = 0; i < N_TRIALS; i++) {
        hipEventRecord(start);
        bench_a_serial<<<1, 64>>>(d_init, N);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&t_a[i], start, stop);
    }
    for (int i = 0; i < N_TRIALS; i++) {
        hipEventRecord(start);
        bench_b_pipelined<<<1, 64>>>(d_init, N);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&t_b[i], start, stop);
    }
    float t_a_med = median_of(t_a, N_TRIALS);
    float t_b_med = median_of(t_b, N_TRIALS);

    float per_iter_a = (t_a_med * 1e6f) / N;
    float per_iter_b = (t_b_med * 1e6f) / N;

    constexpr double mac_per_iter = 64.0 * 32.0 * 128.0;
    constexpr double flops_per_iter = mac_per_iter * 2.0;
    double tflops_a = flops_per_iter / (per_iter_a * 1e-9) / 1e12;
    double tflops_b = flops_per_iter / (per_iter_b * 1e-9) / 1e12;

    double pct = 100.0 * (tflops_b - tflops_a) / tflops_a;

    printf("Lever E microbench gate (R11 prepared, R12 to decide)\n");
    printf("  N inner iter         = %d, trials = %d (median)\n", N, N_TRIALS);
    printf("  Workload per iter    = 2 intx8 LDS reads + 8 mfma_16x16x128 (single wave)\n");
    printf("\n");
    printf("  (a) serial           = %.3f ms / %d -> %.2f ns/iter -> %.1f TFLOPS\n",
           t_a_med, N, per_iter_a, tflops_a);
    printf("  (b) pipelined        = %.3f ms / %d -> %.2f ns/iter -> %.1f TFLOPS\n",
           t_b_med, N, per_iter_b, tflops_b);
    printf("\n");
    printf("  Per-iter delta: pipelined vs serial = %+.2f %% throughput\n", pct);
    printf("  Decision (R10 plan / R12 to execute):\n");
    printf("    pct >= +5.0 %%  -> Lever E CONFIRMED. Commit kernel rewrite.\n");
    printf("    pct in (-3,+5) -> Lever E FALSIFIED. Plateau accepted; switch to bwd opt.\n");
    printf("    pct <= -3.0 %% -> Lever E WORSE. Same as falsified, plus codegen note.\n");
    printf("  Verdict: %s\n",
           pct >= 5.0 ? "PASS - Lever E confirmed"
                      : (pct > -3.0 ? "FAIL - plateau accepted"
                                    : "FAIL - hand schedule worse than LLVM"));

    hipFree(d_init);
    hipEventDestroy(start); hipEventDestroy(stop);
    return 0;
}
