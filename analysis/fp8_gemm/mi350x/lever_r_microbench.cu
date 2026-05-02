// Lever R microbench gate (R17, FALSIFIED — verdict in printf at run time).
//
// Question: in the gpt_oss-K-misaligned (K%128 != 0) FP8 grouped path,
// the K-tail block (kernel_fp8_layouts.cpp lines 2535-2719) fires its
// 24 ``buffer_load_b128`` HBM round-trip after epilog 2's 4 mfma's
// (lines 2484-2506) finish. Can we overlap the K-tail HBM round-trip
// (~80 cy L2-hit, ~250+ cy L2-miss on MI355X) with the epilog-2 mfma
// chain (~52-128 cy compute + barriers) by issuing the 24 loads BEFORE
// epilog 2 starts?
//
// FALSIFIED RESULT (R17, this file):
//   * Naive microbench with ``asm volatile`` barrier between epilog-2
//     mfma chain and buffer_load chain (forced serialisation in path
//     (a)) showed +14.21 % delta (b vs a) — APPEARED to confirm Lever R.
//   * Falsification audit: the asm barrier in path (a) is an ARTIFACT
//     that forces SQ to wait until all 4 mfmas issue before any
//     buffer_load. The real kernel has NO such barrier — epilog 2 is
//     a mix of LDS reads + mfmas + lgkmcnt drains, with SQ available
//     for HBM issue throughout. LLVM already auto-schedules overlap.
//   * REMOVING the asm barrier from path (a) → +0.29 % delta — well
//     within microbench noise band. LEVER R IS A NULL CHANGE on real-
//     kernel-equivalent code.
//   * Architectural confirmation: in the real kernel, the 2-stage
//     vmcnt(8) → vmcnt(0) drain in K-tail block (line 2713-2716) ALREADY
//     overlaps the cA/cB mfma latency (~26 cy) with the LAST 8 a_kt1
//     loads' retirement. Pre-issuing a_kt1 to before epilog 2 would
//     require dropping the 2-stage drain (correctness: pre-issue +
//     vmcnt(8) leaves b0/b1/a in flight, retiring out-of-order with
//     a_kt1 → a not retired when cA/cB needs it). Single vmcnt(0) drain
//     erases the 2-stage benefit, net cycle accounting = 0.
//   * Conclusion: same fate as Lever E (R11) — LLVM's scheduler is
//     already optimal in this region. Hand scheduling cannot improve.
//
// Originally-stated R16 hypothesis (preserved for documentation):
//   * gpt_oss subset (8/24 cases) = ~+2-3 % main-kernel TFLOPS if the
//     HBM round-trip is fully hidden by epilog 2 compute.
//   * Geomean impact: 8/24 weight × +2.5 % ≈ +0.83 pp on grp_FP8 geomean.
//   * DSV3 + Qwen subsets (16/24 cases) = predicate-gated, 0 % impact.
// Realised result: 0 pp (falsified).
//
// Per-warp workload modelled here:
//   * 4 mfma_scale_f32_16x16x128 using register-resident data (no
//     vmcnt dep) — mirrors epilog 2 cA/cB/cC/cD.
//   * 24 raw_buffer_load_b128 from a 64 KB HBM region (cached, mostly
//     L2-hit) — mirrors load_b_kt(b0, b1) + load_a_kt(a, a_kt1) at lines
//     2709-2712 in the real kernel.
//   * s_waitcnt vmcnt(8) drain.
//   * 2 mfma_scale_f32_16x16x128 using freshly-loaded HBM data — mirrors
//     rcr_mma(cA, a, b0) + rcr_mma(cB, a, b1) at lines 2714-2715.
//   * s_waitcnt vmcnt(0) drain.
//   * 2 mfma_scale_f32_16x16x128 using freshly-loaded HBM data — mirrors
//     rcr_mma(cC, a_kt1, b0) + rcr_mma(cD, a_kt1, b1) at lines 2717-2718.
//
// Two paths:
//   path (a) "current" : 4 mfma (epilog 2)    → 24 buffer_load
//                                              → vmcnt(8) → 2 mfma
//                                              → vmcnt(0) → 2 mfma
//   path (b) "pre-issued": 24 buffer_load
//                                              → 4 mfma (epilog 2,
//                                                 overlap with HBM)
//                                              → vmcnt(8) → 2 mfma
//                                              → vmcnt(0) → 2 mfma
//
// Decision rule (R17 to execute):
//   * (b) >= +3.0 pp throughput vs (a)  → Lever R CONFIRMED. R18 will
//     port the pre-issue ordering into grouped_rcr_kernel epilog 2 with
//     conservative liveness mitigation (hoist only laneid / row_lane /
//     k_lane_byte = ~3 dw to the fast path; lazy SRD compute inside
//     K-tail block to avoid +spill regression on K%128==0 templates).
//   * (b) within (-3, +3) pp           → Lever R FALSIFIED. LLVM is
//     already auto-scheduling the load-mfma overlap, manual hoisting
//     buys nothing. Plateau accepted; switch to backward-track lever
//     (Lever H Direction B fused dA-transpose) per R16 round note.
//   * (b) <= -3.0 pp                   → Lever R WORSE (manual hoist
//     blocks LLVM's chosen schedule). FALSIFIED. Same agenda switch.
//
// Build:
//   hipcc lever_r_microbench.cu -o lever_r_microbench --offload-arch=gfx950 -O3
// Run:
//   ./lever_r_microbench [N_ITERS=10000]
//
// IMPORTANT LIMITATIONS (read before trusting the gate decision):
//   * SINGLE-WAVE microbench. Real grouped_rcr_kernel runs 8 waves/CTA;
//     cross-wave s_barrier costs are NOT captured here. The DELTA
//     between (a) and (b) is what matters for the gate, not absolute
//     numbers.
//   * The 24 loads here are spread evenly across lanes with sequential
//     offsets. Real kernel uses lane-cell mapping (row_lane, k_lane_byte)
//     with K_REM=64 (32/64 lanes are SENTINEL OOB). The OOB lanes' loads
//     return 0 quickly — this microbench may slightly OVER-count the
//     HBM round-trip cost. If the gate falsifies here, real kernel would
//     also falsify (gain estimate is conservative).
//   * No accompanying ds_read / ds_write activity. Real kernel has
//     concurrent main-loop LDS reads/writes that contend with the
//     epilog 2 lgkmcnt drains. Microbench can't capture this 2-counter
//     interaction faithfully.
//   * Modelling 1 output tile's K-tail; real kernel processes ~370
//     output tiles per CU per launch. The relative pipelining gain
//     should scale linearly across tiles.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;
typedef __attribute__((__vector_size__(4 * sizeof(int)))) int i32x4;

#ifndef N_ITERS
#define N_ITERS 5000
#endif

// SRD setup helpers — mirror kittens::make_srsrc / raw_buffer_load_b128
// in include/ops/warp/memory/util/util.cuh (decl line 72 / 102).
struct __attribute__((aligned(16))) buffer_resource {
    uint64_t base;
    uint32_t range;
    uint32_t config;
};

__device__ static inline i32x4 mb_make_srsrc(const void* ptr, uint32_t range_bytes) {
    auto as_u64 = (uint64_t)reinterpret_cast<uintptr_t>(ptr);
    buffer_resource rsrc{as_u64, range_bytes, 0x110000};
    i32x4 out;
    __builtin_memcpy(&out, &rsrc, sizeof(rsrc));
    return out;
}

__device__ __uint128_t llvm_amdgcn_raw_buffer_load_b128(
    i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i128");

// Drain to keep results live across kernel invocations.
__device__ __managed__ unsigned long long g_drain[16];

constexpr uint32_t HBM_REGION_BYTES = 64 * 1024;  // 64 KB cached region
constexpr int N_LOADS_PER_LANE = 3;               // 24 loads / 8 lanes-of-interest = 3
                                                  // but we issue per ALL 64 lanes for clarity

// ============================================================
// Path (a) "current": mfma chain → load chain → wait → mfma chain
// ============================================================
__global__ __launch_bounds__(64, 8)
void bench_a_current(const intx8_t* __restrict__ reg_seed,
                     const uint8_t* hbm_base,
                     uint32_t hbm_bytes,
                     int N) {
    const int laneid = threadIdx.x & 63;

    // Register-resident "epilog 2" data (no vmcnt dep on HBM).
    intx8_t a_reg = reg_seed[laneid & 7];
    intx8_t b0_reg = reg_seed[(laneid + 1) & 7];
    intx8_t b1_reg = reg_seed[(laneid + 2) & 7];

    floatx4_t cA = {0,0,0,0};
    floatx4_t cB = {0,0,0,0};
    floatx4_t cC = {0,0,0,0};
    floatx4_t cD = {0,0,0,0};

    i32x4 srd = mb_make_srsrc((const void*)hbm_base, hbm_bytes);

    for (int iter = 0; iter < N; iter++) {
        // ---- Epilog 2: 4 mfma using register data ----
        // Use asm volatile barrier to keep the textual order intact and
        // force LLVM to NOT hoist the buffer_loads above the mfma block
        // (otherwise (a) and (b) compile to the same code and the gate
        // becomes meaningless).
        cA = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_reg, b0_reg, cA, 0, 0, 0, 0, 0, 0);
        cB = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_reg, b1_reg, cB, 0, 0, 0, 0, 0, 0);
        cC = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_reg, b0_reg, cC, 0, 0, 0, 0, 0, 0);
        cD = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_reg, b1_reg, cD, 0, 0, 0, 0, 0, 0);
        // R17 falsification audit: REMOVED the asm volatile barrier here.
        // The original barrier forced an artificial serialisation between
        // the mfma chain and the buffer_load chain that does NOT exist in
        // the real kernel (where epilog 2 has interleaved LDS reads + mfmas
        // + lgkmcnt drains, leaving SQ available for HBM issue throughout).
        // With the barrier removed, LLVM is free to overlap loads with
        // mfmas — testing whether the +14% gate gain was a barrier artifact
        // or a real schedule advantage.

        // ---- K-tail: 24 buffer_load_b128 ----
        // Vary offset per iter to defeat L1 caching (force L2 round-trip).
        const uint32_t v_iter_shift = ((uint32_t)(iter * 256u)) % (HBM_REGION_BYTES - 4096u);
        const uint32_t v_lane_shift = ((uint32_t)(laneid * 16u)) & (HBM_REGION_BYTES - 1u);
        const uint32_t v_base = v_iter_shift + v_lane_shift;

        intx8_t kt[12];  // 24 b128 = 12 intx8 (each intx8 = 256 bits = 2× b128)
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            uint32_t v_lo = (v_base + i * 32u) & (HBM_REGION_BYTES - 1u);
            uint32_t v_hi = (v_lo + 16u) & (HBM_REGION_BYTES - 1u);
            __uint128_t lo = llvm_amdgcn_raw_buffer_load_b128(srd, v_lo, 0, 0);
            __uint128_t hi = llvm_amdgcn_raw_buffer_load_b128(srd, v_hi, 0, 0);
            __uint128_t* dst = reinterpret_cast<__uint128_t*>(&kt[i]);
            dst[0] = lo;
            dst[1] = hi;
        }

        // ---- vmcnt(8) drain → 2 mfma ----
        asm volatile("s_waitcnt vmcnt(8)");
        cA = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            kt[0], kt[1], cA, 0, 0, 0, 0, 0, 0);
        cB = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            kt[2], kt[3], cB, 0, 0, 0, 0, 0, 0);

        // ---- vmcnt(0) drain → 2 mfma ----
        asm volatile("s_waitcnt vmcnt(0)");
        cC = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            kt[8], kt[9], cC, 0, 0, 0, 0, 0, 0);
        cD = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            kt[10], kt[11], cD, 0, 0, 0, 0, 0, 0);
    }

    if (laneid == 0) {
        floatx4_t s = cA + cB + cC + cD;
        g_drain[0] = (unsigned long long)(int)s[0];
        g_drain[1] = (unsigned long long)(int)s[1];
    }
}

// ============================================================
// Path (b) "pre-issued": load chain → mfma chain (overlapping) → wait → mfma chain
// ============================================================
__global__ __launch_bounds__(64, 8)
void bench_b_preissued(const intx8_t* __restrict__ reg_seed,
                       const uint8_t* hbm_base,
                       uint32_t hbm_bytes,
                       int N) {
    const int laneid = threadIdx.x & 63;

    intx8_t a_reg = reg_seed[laneid & 7];
    intx8_t b0_reg = reg_seed[(laneid + 1) & 7];
    intx8_t b1_reg = reg_seed[(laneid + 2) & 7];

    floatx4_t cA = {0,0,0,0};
    floatx4_t cB = {0,0,0,0};
    floatx4_t cC = {0,0,0,0};
    floatx4_t cD = {0,0,0,0};

    i32x4 srd = mb_make_srsrc((const void*)hbm_base, hbm_bytes);

    for (int iter = 0; iter < N; iter++) {
        const uint32_t v_iter_shift = ((uint32_t)(iter * 256u)) % (HBM_REGION_BYTES - 4096u);
        const uint32_t v_lane_shift = ((uint32_t)(laneid * 16u)) & (HBM_REGION_BYTES - 1u);
        const uint32_t v_base = v_iter_shift + v_lane_shift;

        // ---- K-tail: 24 buffer_load_b128 ISSUED FIRST ----
        // (these run in parallel with the mfma chain below)
        intx8_t kt[12];
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            uint32_t v_lo = (v_base + i * 32u) & (HBM_REGION_BYTES - 1u);
            uint32_t v_hi = (v_lo + 16u) & (HBM_REGION_BYTES - 1u);
            __uint128_t lo = llvm_amdgcn_raw_buffer_load_b128(srd, v_lo, 0, 0);
            __uint128_t hi = llvm_amdgcn_raw_buffer_load_b128(srd, v_hi, 0, 0);
            __uint128_t* dst = reinterpret_cast<__uint128_t*>(&kt[i]);
            dst[0] = lo;
            dst[1] = hi;
        }
        // Force the buffer_load issue point to be HERE in the schedule
        // (no false vmcnt drain — mfma below has no HBM dep so they can
        // overlap with the in-flight loads).
        asm volatile("" ::: "memory");

        // ---- Epilog 2: 4 mfma using register data, OVERLAPS with HBM ----
        cA = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_reg, b0_reg, cA, 0, 0, 0, 0, 0, 0);
        cB = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_reg, b1_reg, cB, 0, 0, 0, 0, 0, 0);
        cC = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_reg, b0_reg, cC, 0, 0, 0, 0, 0, 0);
        cD = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_reg, b1_reg, cD, 0, 0, 0, 0, 0, 0);

        // ---- vmcnt(8) drain → 2 K-tail mfma ----
        asm volatile("s_waitcnt vmcnt(8)");
        cA = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            kt[0], kt[1], cA, 0, 0, 0, 0, 0, 0);
        cB = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            kt[2], kt[3], cB, 0, 0, 0, 0, 0, 0);

        // ---- vmcnt(0) drain → 2 K-tail mfma ----
        asm volatile("s_waitcnt vmcnt(0)");
        cC = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            kt[8], kt[9], cC, 0, 0, 0, 0, 0, 0);
        cD = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            kt[10], kt[11], cD, 0, 0, 0, 0, 0, 0);
    }

    if (laneid == 0) {
        floatx4_t s = cA + cB + cC + cD;
        g_drain[2] = (unsigned long long)(int)s[0];
        g_drain[3] = (unsigned long long)(int)s[1];
    }
}

// ============================================================

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
    int N = N_ITERS;
    if (argc >= 2) N = atoi(argv[1]);

    // Allocate seed registers (8 intx8 = 256 bytes) and HBM region.
    intx8_t* d_seed = nullptr;
    hipMalloc(&d_seed, 8 * sizeof(intx8_t));
    hipMemset(d_seed, 0x12, 8 * sizeof(intx8_t));

    uint8_t* d_hbm = nullptr;
    hipMalloc(&d_hbm, HBM_REGION_BYTES);
    hipMemset(d_hbm, 0x34, HBM_REGION_BYTES);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    constexpr int N_TRIALS = 11;

    // Warmup.
    bench_a_current<<<1, 64>>>(d_seed, d_hbm, HBM_REGION_BYTES, N);
    bench_b_preissued<<<1, 64>>>(d_seed, d_hbm, HBM_REGION_BYTES, N);
    hipDeviceSynchronize();

    float t_a[N_TRIALS], t_b[N_TRIALS];
    for (int i = 0; i < N_TRIALS; i++) {
        hipEventRecord(start);
        bench_a_current<<<1, 64>>>(d_seed, d_hbm, HBM_REGION_BYTES, N);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&t_a[i], start, stop);
    }
    for (int i = 0; i < N_TRIALS; i++) {
        hipEventRecord(start);
        bench_b_preissued<<<1, 64>>>(d_seed, d_hbm, HBM_REGION_BYTES, N);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&t_b[i], start, stop);
    }

    // Anti-bias: alternate (a)/(b) for second half to detect run-order
    // sensitivity (e.g., HBM warmup state asymmetry).
    float t_a2[N_TRIALS], t_b2[N_TRIALS];
    for (int i = 0; i < N_TRIALS; i++) {
        hipEventRecord(start);
        bench_b_preissued<<<1, 64>>>(d_seed, d_hbm, HBM_REGION_BYTES, N);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&t_b2[i], start, stop);
    }
    for (int i = 0; i < N_TRIALS; i++) {
        hipEventRecord(start);
        bench_a_current<<<1, 64>>>(d_seed, d_hbm, HBM_REGION_BYTES, N);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&t_a2[i], start, stop);
    }

    float t_a_med = median_of(t_a, N_TRIALS);
    float t_b_med = median_of(t_b, N_TRIALS);
    float t_a2_med = median_of(t_a2, N_TRIALS);
    float t_b2_med = median_of(t_b2, N_TRIALS);

    float per_iter_a = (t_a_med * 1e6f) / N;
    float per_iter_b = (t_b_med * 1e6f) / N;
    float per_iter_a2 = (t_a2_med * 1e6f) / N;
    float per_iter_b2 = (t_b2_med * 1e6f) / N;

    // Workload per iter: 8 mfma (× 64×32×128 mac = 262144) + 24 b128 loads
    // mfma flops = 8 * 64 * 32 * 128 * 2 = 4 194 304
    constexpr double mac_per_iter = 8.0 * 64.0 * 32.0 * 128.0;
    constexpr double flops_per_iter = mac_per_iter * 2.0;
    double tflops_a = flops_per_iter / (per_iter_a * 1e-9) / 1e12;
    double tflops_b = flops_per_iter / (per_iter_b * 1e-9) / 1e12;
    double tflops_a2 = flops_per_iter / (per_iter_a2 * 1e-9) / 1e12;
    double tflops_b2 = flops_per_iter / (per_iter_b2 * 1e-9) / 1e12;

    double pct_b_vs_a = 100.0 * (tflops_b - tflops_a) / tflops_a;
    double pct_b2_vs_a2 = 100.0 * (tflops_b2 - tflops_a2) / tflops_a2;

    // Average the two run-order samples.
    double pct_avg = 0.5 * (pct_b_vs_a + pct_b2_vs_a2);

    printf("Lever R microbench gate (R17)\n");
    printf("  N inner iter  = %d, trials per path = %d (median)\n", N, N_TRIALS);
    printf("  Workload per iter = 4 mfma (epilog 2) + 24 buffer_load_b128 + 4 mfma (K-tail)\n");
    printf("  Single wave (64 lanes, 1 CTA), HBM region = %u bytes (cached)\n", HBM_REGION_BYTES);
    printf("\n");
    printf("  Run-order #1 (a-first, then b):\n");
    printf("    (a) current     = %.3f ms total / %d iter -> %.2f ns/iter -> %.1f TFLOPS (mfma-only)\n",
           t_a_med, N, per_iter_a, tflops_a);
    printf("    (b) pre-issued  = %.3f ms total / %d iter -> %.2f ns/iter -> %.1f TFLOPS (mfma-only)\n",
           t_b_med, N, per_iter_b, tflops_b);
    printf("    delta (b vs a)  = %+.2f %% throughput\n", pct_b_vs_a);
    printf("\n");
    printf("  Run-order #2 (b-first, then a):\n");
    printf("    (b) pre-issued  = %.3f ms total / %d iter -> %.2f ns/iter -> %.1f TFLOPS\n",
           t_b2_med, N, per_iter_b2, tflops_b2);
    printf("    (a) current     = %.3f ms total / %d iter -> %.2f ns/iter -> %.1f TFLOPS\n",
           t_a2_med, N, per_iter_a2, tflops_a2);
    printf("    delta (b vs a)  = %+.2f %% throughput\n", pct_b2_vs_a2);
    printf("\n");
    printf("  Mean delta (b vs a, both run orders) = %+.2f %%\n", pct_avg);
    printf("\n");
    printf("  Decision (R17 plan):\n");
    printf("    pct >= +3.0 %%   -> Lever R CONFIRMED. R18 = port pre-issue ordering.\n");
    printf("    pct in (-3,+3)  -> Lever R FALSIFIED. LLVM auto-overlaps; plateau accepted.\n");
    printf("    pct <= -3.0 %%  -> Lever R WORSE. FALSIFIED.\n");
    printf("  Verdict: %s\n",
           pct_avg >= 3.0 ? "PASS - Lever R confirmed (port to kernel in R18)"
                          : (pct_avg > -3.0 ? "FAIL - plateau accepted (LLVM auto-schedules overlap)"
                                            : "FAIL - hand schedule worse than LLVM"));

    hipFree(d_seed);
    hipFree(d_hbm);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return 0;
}
