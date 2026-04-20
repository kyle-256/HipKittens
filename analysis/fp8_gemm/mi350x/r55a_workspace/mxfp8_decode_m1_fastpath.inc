// R42 Dev A — M=1 single-token decode MXFP8 fastpath (RCR layout only).
//
// Geometry
// --------
// BLK_M=1, BLK_N=64. One wavefront (64 lanes) per workgroup. Each lane
// computes ONE output element c[0, col] = sum_k A[0, k] * B[col, k] *
// scale_a[0, k/32] * scale_b[col, k/32]. Total grid = (N / 64) workgroups.
//
// Why this geometry?
// ------------------
// At M=1 the problem is a GEMV. MFMA primitives have a minimum M=16 tile
// (16x16x128 fp8 MFMA), wasting 15/16 of the M axis. A pure SIMD GEMV
// (one lane = one column) avoids MFMA entirely and unlocks the bandwidth-
// bound regime: B is N*K bytes (16 MB at 4096x4096 / 64 MB at 8192x8192)
// which dominates total time. Theoretical lower bound on 8B 1-tok at
// 5 TB/s HBM is ~3.2 us; current V1-LEGACY-FALLBACK measures 519 us
// (R41 Dev C). Headroom is ~150x.
//
// Coalescing
// ----------
// B is row-major (N, K) so within a row (single column) K-direction reads
// are contiguous. Adjacent lanes read DIFFERENT rows -> separate cache
// lines per lane. This is intentional: the memory subsystem services 64
// concurrent column streams in parallel, saturating MEM ports across CUs.
// Total B bytes / WG = 64 * K bytes (256 KB at K=4096), * (N/64) WGs =
// N * K bytes -> matches problem size. No oversubscribed reads.
//
// A and A_scale broadcast
// -----------------------
// A is M=1 row (K bytes; 4 KB at K=4096). A_scale is M=1 row (K/32 bytes;
// 128 bytes). Both are tiny and shared across all output cols. We load
// them ONCE per WG into LDS at kernel start; subsequent K-iterations
// service A from LDS (no DRAM hit).
//
// Scale layout (PRESHUFFLED_QUANT)
// -------------------------------
// PRESHUFFLED_QUANT=true: V1-PQ layout (matches load_scale_scalar_preshuffled
// in this same TU). We use the existing helper for both A_scale (during
// the LDS prefetch) and B_scale (per-lane in the K-loop).
// PRESHUFFLED_QUANT=false: row-major (M, K/32) and (N, K/32) -- raw e8m0.
//
// Default 8192x8192x8192 build invariance
// ---------------------------------------
// This entire file is gated by MXFP8_DECODE_M1_ENABLE. Default builds do
// not define this macro -> kernel symbol is never instantiated, predicate
// helper is never declared, dispatcher branches compile to dead code.
// Verified via nm on default .so: zero `gemv_m1_decode_kernel` symbols.

#ifndef MXFP8_DECODE_M1_ENABLE
#define MXFP8_DECODE_M1_ENABLE 0
#endif

#if MXFP8_DECODE_M1_ENABLE

namespace mxfp8_decode_m1 {

constexpr int DECODE_BLK_M = 1;
constexpr int DECODE_BLK_N = 64;
constexpr int DECODE_THREADS = 64;   // one wavefront per WG

}  // namespace mxfp8_decode_m1

// gemv_m1_decode_kernel<L=RCR, PRESHUFFLED_QUANT>
// -----------------------------------------------
// Grid: (N / DECODE_BLK_N, 1, 1). Block: (DECODE_THREADS=64, 1, 1).
// Each WG produces 64 contiguous output cols starting at bc*DECODE_BLK_N.
template<Layout L, bool PRESHUFFLED_QUANT>
__global__ void gemv_m1_decode_kernel(const layout_globals g) {
    static_assert(L == Layout::RCR,
                  "M=1 decode fastpath only implemented for RCR layout in R42 Dev A");

    constexpr int BLK_N = mxfp8_decode_m1::DECODE_BLK_N;
    constexpr int N_THR = mxfp8_decode_m1::DECODE_THREADS;
    const int lane = threadIdx.x;                // 0..63
    const int col_base = blockIdx.x * BLK_N;     // 0..N-BLK_N
    const int col = col_base + lane;             // this lane's output column

    if (col >= g.n) return;

    const int K = g.k;
    const int K_BLOCKS = K / 32;                 // number of e8m0 scale blocks per row

    // ---------- Cooperative A + A_scale prefetch into LDS ----------
    extern __shared__ unsigned char smem[];
    fp8e4m3* a_lds = reinterpret_cast<fp8e4m3*>(smem);
    // Pad A to 16-byte boundary so a_scale_lds is uint4-safe.
    const int a_lds_bytes = ((K + 15) / 16) * 16;
    uint8_t* a_scale_lds = reinterpret_cast<uint8_t*>(smem) + a_lds_bytes;

    // Each lane copies (K / 64) fp8 elements from A.
    #pragma unroll 4
    for (int i = lane; i < K; i += N_THR) {
        a_lds[i] = g.a[coord<>(0, i)];
    }
    #pragma unroll 4
    for (int i = lane; i < K_BLOCKS; i += N_THR) {
        if constexpr (PRESHUFFLED_QUANT) {
            // V1-PQ preshuffled layout: row-group 0, offset given by helper.
            const int offset = preshuffled_scale_offset(0, i);
            a_scale_lds[i] = static_cast<uint8_t>(g.a_scale[coord<>(0, offset)]);
        } else {
            a_scale_lds[i] = static_cast<uint8_t>(g.a_scale[coord<>(0, i)]);
        }
    }

    __syncthreads();

    // ---------- Per-lane K-loop ----------
    // Each lane computes acc = sum_k A[0,k] * scale_a[k/32] * B[col,k] * scale_b[col,k/32].
    // Inner loop walks scale-block by scale-block (32 K elements per block).
    float acc = 0.0f;

    // Per-row B base pointer (lane-private, fp8e4m3* arithmetic = 1 byte/elem).
    const fp8e4m3* b_row_base = g.b.raw_ptr +
        static_cast<uint64_t>(col) * static_cast<uint64_t>(K);

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

        // 32 K elements per scale block; load as 2 x packed-4 (8 elems each chunk).
        // -> 8 packed-4 loads per scale block (4 from A, 4 from B); each gives float4.
        const int k0 = kb * 32;
        #pragma unroll
        for (int chunk = 0; chunk < 8; ++chunk) {
            const int k = k0 + chunk * 4;
            // Read 4 packed fp8 from A LDS (4 bytes = 1 dword).
            const fp8e4m3_4 a_pack = *reinterpret_cast<const fp8e4m3_4*>(a_lds + k);
            // Read 4 packed fp8 from B[col, k..k+3].
            const fp8e4m3_4 b_pack = *reinterpret_cast<const fp8e4m3_4*>(b_row_base + k);

            const float4 a_f = base_types::convertor<float4, fp8e4m3_4>::convert(a_pack);
            const float4 b_f = base_types::convertor<float4, fp8e4m3_4>::convert(b_pack);

            const float prod0 = a_f.x * b_f.x;
            const float prod1 = a_f.y * b_f.y;
            const float prod2 = a_f.z * b_f.z;
            const float prod3 = a_f.w * b_f.w;
            acc = fmaf(prod0, sab, acc);
            acc = fmaf(prod1, sab, acc);
            acc = fmaf(prod2, sab, acc);
            acc = fmaf(prod3, sab, acc);
        }
    }

    // ---------- Epilogue ----------
    const float scaled = acc * g.scale;
    g.c[coord<>(0, col)] = base_types::convertor<bf16, float>::convert(scaled);
}

// Host-side predicate. Triggers only on M=1, RCR, N divisible by BLK_N=64,
// K a multiple of 32 (scale block).
__host__ inline bool can_use_decode_m1(const layout_globals& g) {
    return (g.m == 1)
        && (g.n > 0) && (g.n % mxfp8_decode_m1::DECODE_BLK_N == 0)
        && (g.k > 0) && (g.k % 32 == 0);
}

template<Layout L, bool PRESHUFFLED_QUANT>
__host__ void dispatch_decode_m1(layout_globals g) {
    constexpr int BLK_N = mxfp8_decode_m1::DECODE_BLK_N;
    dim3 grid(g.n / BLK_N, 1, 1);
    dim3 block(mxfp8_decode_m1::DECODE_THREADS, 1, 1);
    // Shared memory: A row (K bytes, padded to 16) + A_scale row (K/32 bytes + 16 pad).
    const int a_lds_bytes = ((g.k + 15) / 16) * 16;
    const int a_scale_lds_bytes = (g.k / 32) + 16;
    const size_t shmem = static_cast<size_t>(a_lds_bytes) +
                         static_cast<size_t>(a_scale_lds_bytes);
    gemv_m1_decode_kernel<L, PRESHUFFLED_QUANT>
        <<<grid, block, shmem, g.stream>>>(g);
}

#endif // MXFP8_DECODE_M1_ENABLE
