// R43 Dev B — M=1 single-token decode MXFP8 fastpath, RRR + CRR layouts.
//
// This is the layout-extension companion to R42 Dev A's RCR-only M=1
// fastpath (mxfp8_decode_m1_fastpath.inc). Addresses R42 Dev A's deferred
// RRR/CRR coverage. Same single-wavefront BLK_M=1 BLK_N=64 GEMV geometry,
// re-targeted for the alternate B layouts used by LLaMA SwiGLU (RRR) and
// the rare CRR shape (A col-major).
//
// Why a separate file?
// --------------------
// The B-coalescing pattern differs materially across layouts:
//   * RCR: B is (N, K) row-major → B[col, kk]; adjacent lanes (cols)
//          hit DIFFERENT cache lines (handled by Dev A's RCR file).
//   * RRR: B is (K, N) row-major → B[kk, col]; adjacent lanes hit
//          ADJACENT bytes within the same K row → naturally coalesced
//          across the wavefront. We loop kk one at a time and each lane
//          reads its byte; 64 lanes × 1 byte = 1 cache line per K-iter.
//   * CRR: A is (K, M) col-major → A[kk, 0] = a.raw_ptr[kk * g.m + 0]
//          (with g.m=1, A is just K-strided 1-byte reads which we still
//          stage to LDS once for reuse). B identical to RRR.
//
// Thus RRR and CRR both want the "lane-strided B" inner loop. Since the
// per-iter load is 1 byte/lane (kk advances in 1-element steps), we lose
// the packed-4 vectorization Dev A used for RCR. To regain throughput we
// process 4 K-elements per iter via a uint pre-shuffle: load 4 K rows into
// 4 separate scalar lanes, then FMA × 4. This keeps the K-loop trip count
// at K/4 per scale-block of 32 → 8 micro-iters per scale-block (matching
// Dev A's count), but each micro-iter is 1 byte-load × 4 (vs Dev A's 4
// bytes packed per load).
//
// Bandwidth: at K=4096, total B bytes = N*K (16 MB at N=4096 / 64 MB at
// N=8192) — IDENTICAL to RCR (bandwidth lower bound is the same).
// Throughput parity vs Dev A's RCR is therefore the design target.
//
// Numerical equivalence: identical to V1 tail kernel (FMA-order matches
// scale-block-major layout — same as Dev A). Expect max|delta|=0 vs V1.
//
// Default 8192³ build invariance
// ------------------------------
// Whole file is gated by MXFP8_DECODE_M1_RRR_CRR_ENABLE. Default builds do
// not define this macro → kernel symbols never instantiated, dispatcher
// branches compile to dead code. Verified via nm: zero
// `gemv_m1_decode_rrr` / `gemv_m1_decode_crr` symbols when default.

#ifndef MXFP8_DECODE_M1_RRR_CRR_ENABLE
#define MXFP8_DECODE_M1_RRR_CRR_ENABLE 0
#endif

#if MXFP8_DECODE_M1_RRR_CRR_ENABLE

namespace mxfp8_decode_m1_rrr_crr {

constexpr int DECODE_BLK_M = 1;
constexpr int DECODE_BLK_N = 64;
constexpr int DECODE_THREADS = 64;   // one wavefront per WG

}  // namespace mxfp8_decode_m1_rrr_crr

// gemv_m1_decode_rrr_crr_kernel<L, PRESHUFFLED_QUANT>
// ----------------------------------------------------
// Grid: (N / DECODE_BLK_N, 1, 1). Block: (DECODE_THREADS=64, 1, 1).
// Each WG produces 64 contiguous output cols starting at bc*DECODE_BLK_N.
// Layouts supported: RRR, CRR (RCR uses the Dev A file).
template<Layout L, bool PRESHUFFLED_QUANT>
__global__ void gemv_m1_decode_rrr_crr_kernel(const layout_globals g) {
    static_assert(L == Layout::RRR || L == Layout::CRR,
                  "M=1 RRR/CRR fastpath: use Dev A's RCR file for RCR");

    constexpr int BLK_N = mxfp8_decode_m1_rrr_crr::DECODE_BLK_N;
    constexpr int N_THR = mxfp8_decode_m1_rrr_crr::DECODE_THREADS;
    const int lane = threadIdx.x;                // 0..63
    const int col_base = blockIdx.x * BLK_N;     // 0..N-BLK_N
    const int col = col_base + lane;             // this lane's output column

    if (col >= g.n) return;

    const int K = g.k;
    const int K_BLOCKS = K / 32;                 // number of e8m0 scale blocks per row

    // ---------- Cooperative A + A_scale prefetch into LDS ----------
    // Both RRR and CRR consume A as a single M=1 row of K bytes; the
    // memory layout differs (RRR row-major (M=1,K) → contiguous K bytes;
    // CRR col-major (K,M=1) → ALSO contiguous K bytes since stride along
    // M is 1). We use coord<>(0, i) for RRR and coord<>(i, 0) for CRR.
    extern __shared__ unsigned char smem[];
    fp8e4m3* a_lds = reinterpret_cast<fp8e4m3*>(smem);
    const int a_lds_bytes = ((K + 15) / 16) * 16;
    uint8_t* a_scale_lds = reinterpret_cast<uint8_t*>(smem) + a_lds_bytes;

    #pragma unroll 4
    for (int i = lane; i < K; i += N_THR) {
        if constexpr (L == Layout::RRR) {
            a_lds[i] = g.a[coord<>(0, i)];
        } else {  // CRR
            a_lds[i] = g.a[coord<>(i, 0)];
        }
    }
    #pragma unroll 4
    for (int i = lane; i < K_BLOCKS; i += N_THR) {
        if constexpr (PRESHUFFLED_QUANT) {
            const int offset = preshuffled_scale_offset(0, i);
            a_scale_lds[i] = static_cast<uint8_t>(g.a_scale[coord<>(0, offset)]);
        } else {
            a_scale_lds[i] = static_cast<uint8_t>(g.a_scale[coord<>(0, i)]);
        }
    }

    __syncthreads();

    // ---------- Per-lane K-loop ----------
    // For RRR/CRR the B layout is (K, N) row-major — adjacent lanes hit
    // adjacent bytes in the same K row. Inner loop walks 32 K elements
    // per scale block, doing 1-byte global B load + 1-byte LDS A load
    // per lane per kk. The 64-lane wavefront issues 1 coalesced cache
    // line per kk (64 contiguous bytes at b + kk*N + col_base).
    //
    // Outer loop (kb, scale block): hoists scale fetches once per 32 kk.
    // Inner loop (kk within kb): 32 FMAs per scale block (matches Dev A
    // throughput count: 8 packed-4 chunks × 4 elems = 32 FMAs).
    float acc = 0.0f;

    const int N = g.n;

    #pragma unroll 1
    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        const float sa = decode_scale_raw_e8m0(a_scale_lds[kb]);

        float sb;
        if constexpr (PRESHUFFLED_QUANT) {
            sb = load_scale_scalar_preshuffled(g.b_scale, col, kb);
        } else {
            sb = load_scale_scalar(g.b_scale, col, kb);
        }
        const float sab = sa * sb;

        const int k0 = kb * 32;

        // Inner: 32 K elements, fully unrolled.
        // Each kk: B[kk, col] = b.raw_ptr[kk * N + col] (1 byte/lane;
        // 64 contiguous lane bytes = 1 cache line per kk for both RRR
        // and CRR which share the same B layout).
        #pragma unroll
        for (int dk = 0; dk < 32; ++dk) {
            const int kk = k0 + dk;
            const fp8e4m3 a_raw = a_lds[kk];
            // B uses the layout-correct coord ordering.
            // For RRR and CRR, B is (K, N) row-major: coord<>(kk, col).
            const fp8e4m3 b_raw = g.b[coord<>(kk, col)];
            const float a_f =
                base_types::convertor<float, fp8e4m3>::convert(a_raw);
            const float b_f =
                base_types::convertor<float, fp8e4m3>::convert(b_raw);
            acc = fmaf(a_f * b_f, sab, acc);
        }
    }

    // ---------- Epilogue ----------
    const float scaled = acc * g.scale;
    g.c[coord<>(0, col)] = base_types::convertor<bf16, float>::convert(scaled);
}

// Host-side predicate. Triggers only on M=1, RRR or CRR, N divisible by
// BLK_N=64, K a multiple of 32 (scale block).
__host__ inline bool can_use_decode_m1_rrr_crr(const layout_globals& g) {
    return (g.m == 1)
        && (g.n > 0) && (g.n % mxfp8_decode_m1_rrr_crr::DECODE_BLK_N == 0)
        && (g.k > 0) && (g.k % 32 == 0);
}

template<Layout L, bool PRESHUFFLED_QUANT>
__host__ void dispatch_decode_m1_rrr_crr(layout_globals g) {
    constexpr int BLK_N = mxfp8_decode_m1_rrr_crr::DECODE_BLK_N;
    dim3 grid(g.n / BLK_N, 1, 1);
    dim3 block(mxfp8_decode_m1_rrr_crr::DECODE_THREADS, 1, 1);
    const int a_lds_bytes = ((g.k + 15) / 16) * 16;
    const int a_scale_lds_bytes = (g.k / 32) + 16;
    const size_t shmem = static_cast<size_t>(a_lds_bytes) +
                         static_cast<size_t>(a_scale_lds_bytes);
    gemv_m1_decode_rrr_crr_kernel<L, PRESHUFFLED_QUANT>
        <<<grid, block, shmem, g.stream>>>(g);
}

#endif // MXFP8_DECODE_M1_RRR_CRR_ENABLE
