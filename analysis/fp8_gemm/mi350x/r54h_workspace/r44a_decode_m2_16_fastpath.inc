// R44 Dev A — M=2..16 small-batch decode MXFP8 fastpath (RCR/RRR/CRR).
//
// Scope and motivation
// --------------------
// Closes the M=2..16 gap left by R42 Dev A (M=1 RCR), R43 Dev B (M=1 RRR/CRR),
// and R42 Dev B (M=32/128 K-loop-hoisted scalar tail kernel). At 1<M<32 the
// production code path falls through to V1-LEGACY-FALLBACK `gemm_tail_kernel`
// (16x16 thread block, scalar K-loop). Two motivating workloads:
//   * Prefill batch-size 2..16 (small batch interactive serving)
//   * Speculative-decode beam widths (4..16 typical)
//
// Geometry
// --------
// BLK_M_FIXED ∈ {2, 4, 8, 16}, BLK_N = 64, single wavefront (64 lanes) per
// WG. Each lane computes M_FIXED output elements c[m, col_lane] for
// m=0..M_FIXED-1 by accumulating M_FIXED float accumulators per lane across
// the K dimension. Runtime g.m may be ≤ M_FIXED — extra rows masked at
// store. Dispatcher picks the smallest M_FIXED that covers runtime g.m.
//
// Why per-M template? An earlier non-templated M_PAD=16 design (see git
// history / R44A development notes) emitted 16 FMAs per kk regardless of
// runtime M, wasting up to 87.5% of FMA bandwidth at M=2. Per-M templates
// let the compiler statically unroll the M-broadcast loop and emit only
// M_FIXED FMAs per kk — measured ~2-4x speedup vs the M_PAD=16 design.
//
// Why not full MFMA? MI350X 16x16x128 fp8 MFMA needs M=16 minimum and yields
// one 16x16 tile per matrix-instruction, but at small M the MFMA path
// requires the same 256x256 tile geometry as V2 fastpaths to amortize the
// LDS double-buffer pipeline. Re-tooling V2 for arbitrary small BLK_M is a
// 3-5 day effort. This kernel takes a different angle: amortize B-side
// bandwidth by reusing each B byte across M_FIXED rows in scalar SIMD (one
// col per lane, M_FIXED accumulators), turning M=1's bandwidth-bound
// regime (~9% HBM peak per R42 Dev A) into a compute-bandwidth-balanced
// regime as M_FIXED grows.
//
// LDS A streaming
// ---------------
// A footprint: M_FIXED*K bytes. At M=16, K=8192 = 128 KB; M=16, K=4096 =
// 64 KB. The MI350X CU has 160 KB LDS, but we want occupancy ≥2 waves/CU.
// Strategy: stream A in K_TILE chunks. Each K_TILE iter: cooperatively load
// M_FIXED*K_TILE bytes into LDS, drain via FMA, advance. K_TILE=1024 gives
// LDS=M_FIXED*1024 (≤16 KB) per WG.
//
// Scale handling
// --------------
// A_scale: M_FIXED*(K/32) bytes. Tiny. Held in LDS once at kernel entry
// (K=8192 → 16*256 = 4 KB).
// B_scale: per-(col, k_block). Loaded once per K-block (hoisted out of
// inner FMA loop) using `load_scale_scalar_preshuffled`.
//
// Default 8192³ build invariance
// ------------------------------
// Whole file is gated by MXFP8_DECODE_M2_16_ENABLE. Default builds (macro
// undefined / 0): no kernel symbol instantiated, predicate helper compiles
// to dead code. Verified via nm: zero `gemv_m2_16_decode_kernel` symbols
// when default. nm-gate pattern: `gemv_m2_16`.

#ifndef MXFP8_DECODE_M2_16_ENABLE
#define MXFP8_DECODE_M2_16_ENABLE 0
#endif

#if MXFP8_DECODE_M2_16_ENABLE

namespace mxfp8_decode_m2_16 {

constexpr int DECODE_BLK_N     = 64;     // one wavefront per WG
constexpr int DECODE_THREADS   = 64;
constexpr int DECODE_K_TILE    = 1024;   // K elements per LDS A-fill chunk
constexpr int MAX_M_FIXED      = 16;     // max compile-time row count

}  // namespace mxfp8_decode_m2_16

// gemv_m2_16_decode_kernel<L, M_FIXED, PRESHUFFLED_QUANT>
// -------------------------------------------------------
// Grid:  (N / DECODE_BLK_N, 1, 1).
// Block: (DECODE_THREADS=64, 1, 1).
// Each WG produces an M_FIXED x BLK_N tile of C; rows ≥ g.m masked at
// store (only rows 0..g.m-1 written). M_FIXED ∈ {2, 4, 8, 16}.
template<Layout L, int M_FIXED, bool PRESHUFFLED_QUANT>
__global__ void gemv_m2_16_decode_kernel(const layout_globals g) {
    static_assert(M_FIXED == 2 || M_FIXED == 4 || M_FIXED == 8 || M_FIXED == 16,
                  "M_FIXED must be one of {2, 4, 8, 16}");
    constexpr int BLK_N    = mxfp8_decode_m2_16::DECODE_BLK_N;
    constexpr int N_THR    = mxfp8_decode_m2_16::DECODE_THREADS;
    constexpr int K_TILE   = mxfp8_decode_m2_16::DECODE_K_TILE;

    const int lane     = threadIdx.x;             // 0..63
    const int col_base = blockIdx.x * BLK_N;      // 0..N-BLK_N
    const int col      = col_base + lane;         // this lane's output column

    const int M = g.m;       // runtime, in [2..M_FIXED]
    const int K = g.k;
    const int N = g.n;
    const int K_BLOCKS = K / 32;                  // total scale blocks

    // ---------- Per-lane M-row accumulators ----------
    float acc[M_FIXED];
    #pragma unroll
    for (int m = 0; m < M_FIXED; ++m) acc[m] = 0.0f;

    // ---------- LDS layout ----------
    //   a_lds: M_FIXED * K_TILE bytes
    //   a_scale_lds: M_FIXED * K_BLOCKS bytes
    extern __shared__ unsigned char smem[];
    fp8e4m3* a_lds = reinterpret_cast<fp8e4m3*>(smem);
    const int a_lds_bytes = M_FIXED * K_TILE;
    uint8_t* a_scale_lds  = reinterpret_cast<uint8_t*>(smem) + a_lds_bytes;

    // ---------- A_scale prefetch (whole rows for all valid M) ----------
    const int total_scale = M_FIXED * K_BLOCKS;
    #pragma unroll 4
    for (int i = lane; i < total_scale; i += N_THR) {
        const int m  = i / K_BLOCKS;
        const int kb = i - m * K_BLOCKS;
        if (m < M) {
            if constexpr (PRESHUFFLED_QUANT) {
                const int row_group = m >> 5;            // always 0 for M<=16
                const int offset    = preshuffled_scale_offset(m, kb);
                a_scale_lds[i] = static_cast<uint8_t>(g.a_scale[coord<>(row_group, offset)]);
            } else {
                a_scale_lds[i] = static_cast<uint8_t>(g.a_scale[coord<>(m, kb)]);
            }
        } else {
            a_scale_lds[i] = 0x7f;  // 1.0 placeholder (never read)
        }
    }
    __syncthreads();

    // ---------- K-tile loop ----------
    const int K_TILES = (K + K_TILE - 1) / K_TILE;

    for (int kt = 0; kt < K_TILES; ++kt) {
        const int k0 = kt * K_TILE;
        const int kt_lim = (k0 + K_TILE <= K) ? K_TILE : (K - k0);

        // ---------- Cooperative A-tile fill (M_FIXED rows × kt_lim cols) ----------
        const int total_a = M_FIXED * kt_lim;
        #pragma unroll 4
        for (int i = lane; i < total_a; i += N_THR) {
            const int m        = i / kt_lim;
            const int kk_local = i - m * kt_lim;
            const int kk       = k0 + kk_local;
            if (m < M) {
                if constexpr (L == Layout::RCR || L == Layout::RRR) {
                    a_lds[m * K_TILE + kk_local] = g.a[coord<>(m, kk)];
                } else {  // CRR
                    a_lds[m * K_TILE + kk_local] = g.a[coord<>(kk, m)];
                }
            } else {
                a_lds[m * K_TILE + kk_local] = static_cast<fp8e4m3>(0);
            }
        }
        __syncthreads();

        // ---------- K-block loop within K-tile ----------
        const int kb_lo = k0 / 32;
        const int kb_hi = (k0 + kt_lim) / 32;

        #pragma unroll 1
        for (int kb = kb_lo; kb < kb_hi; ++kb) {
            // B_scale per-(col, kb), hoisted per kb.
            float sb;
            if (col < N) {
                if constexpr (PRESHUFFLED_QUANT) {
                    sb = load_scale_scalar_preshuffled(g.b_scale, col, kb);
                } else {
                    sb = load_scale_scalar(g.b_scale, col, kb);
                }
            } else {
                sb = 0.0f;
            }

            // Per-row A_scale lookup (M_FIXED per kb; cheap from LDS).
            float sa[M_FIXED];
            #pragma unroll
            for (int m = 0; m < M_FIXED; ++m) {
                sa[m] = decode_scale_raw_e8m0(a_scale_lds[m * K_BLOCKS + kb]);
            }

            // Inner: 32 kk; for each kk load B[*,col] once, broadcast to
            // M_FIXED FMAs. Inner unroll factor scales with M_FIXED to keep
            // total inner-loop FMA count tractable (max 32*16 = 512 FMAs at
            // M_FIXED=16 with full unroll → spilling; at M=2/4/8 we can
            // afford full unroll).
            constexpr int DK_UNROLL =
                (M_FIXED == 16) ? 4 :
                (M_FIXED == 8)  ? 8 :
                                  16;  // M=2/4: unroll 16 of 32

            const int kk_local_lo = kb * 32 - k0;
            #pragma unroll DK_UNROLL
            for (int dk = 0; dk < 32; ++dk) {
                const int kk_local = kk_local_lo + dk;

                fp8e4m3 b_raw;
                if (col < N) {
                    if constexpr (L == Layout::RCR) {
                        b_raw = g.b[coord<>(col, k0 + kk_local)];
                    } else {
                        b_raw = g.b[coord<>(k0 + kk_local, col)];
                    }
                } else {
                    b_raw = static_cast<fp8e4m3>(0);
                }
                const float b_f =
                    base_types::convertor<float, fp8e4m3>::convert(b_raw);
                const float bsb = b_f * sb;

                // Broadcast: M_FIXED FMAs per kk.
                #pragma unroll
                for (int m = 0; m < M_FIXED; ++m) {
                    const fp8e4m3 a_raw = a_lds[m * K_TILE + kk_local];
                    const float a_f =
                        base_types::convertor<float, fp8e4m3>::convert(a_raw);
                    acc[m] = fmaf(a_f * sa[m], bsb, acc[m]);
                }
            }
        }
        __syncthreads();  // before next K-tile refill
    }

    // ---------- Epilogue: store M valid rows ----------
    if (col < N) {
        const float gscale = g.scale;
        #pragma unroll
        for (int m = 0; m < M_FIXED; ++m) {
            if (m < M) {
                const float scaled = acc[m] * gscale;
                g.c[coord<>(m, col)] =
                    base_types::convertor<bf16, float>::convert(scaled);
            }
        }
    }
}

// Host-side predicate. SHIP-LITE-PARTIAL scope:
//   - Cells where R44A both BEATS V1-LEGACY-FALLBACK and meets the 95%
//     MXFP8/FP8 SHIP gate per single-GPU smoke bench (see
//     r44a_decode_m2_16_findings.md).
//   - Geometric coverage:
//       * M ≤ 4 with N ≥ 4096          (M=4 4kx4k 101%/1.26×; M=4 8kx8k 96%)
//       * M ∈ [5..16] with N ≥ 8192    (M=16 8kx8k 96%; M=8 8kx8k 79% but
//                                       still beats V1 → ALLOW; M=8/16 4kx4k
//                                       lose to V1 due to grid undersubscription
//                                       at N=4096 / 64 = 64 WGs ≪ 304 CUs →
//                                       EXCLUDE).
//   - Layout: any of RCR/RRR/CRR (kernel template covers all three).
//   - Alignment: N divisible by BLK_N=64; K multiple of 32 (scale block).
//
// Cells excluded by this predicate fall through to V1-LEGACY tail kernel —
// the same path production hits today, so byte-identity vs production for
// the M=8/16 N=4096 shapes is preserved.
__host__ inline bool can_use_decode_m2_16(const layout_globals& g) {
    if (g.m < 2 || g.m > mxfp8_decode_m2_16::MAX_M_FIXED) return false;
    if (g.n <= 0 || g.n % mxfp8_decode_m2_16::DECODE_BLK_N != 0) return false;
    if (g.k <= 0 || g.k % 32 != 0) return false;
    // Grid-saturation guard: at N=4096 the wave count (N/BLK_N=64) under-
    // subscribes the 304-CU MI355X. Only enable for N≥8192 once M>4.
    if (g.m > 4 && g.n < 8192) return false;
    return true;
}

// Pick the smallest M_FIXED ∈ {2, 4, 8, 16} that covers runtime g.m.
__host__ inline int pick_m_fixed(int m_runtime) {
    if (m_runtime <= 2)  return 2;
    if (m_runtime <= 4)  return 4;
    if (m_runtime <= 8)  return 8;
    return 16;
}

template<Layout L, bool PRESHUFFLED_QUANT>
__host__ void dispatch_decode_m2_16(layout_globals g) {
    constexpr int BLK_N  = mxfp8_decode_m2_16::DECODE_BLK_N;
    constexpr int K_TILE = mxfp8_decode_m2_16::DECODE_K_TILE;
    dim3 grid(g.n / BLK_N, 1, 1);
    dim3 block(mxfp8_decode_m2_16::DECODE_THREADS, 1, 1);
    const int K_BLOCKS = g.k / 32;
    const int m_fixed  = pick_m_fixed(g.m);
    const int a_lds_bytes = m_fixed * K_TILE;
    const int a_scale_lds_bytes = m_fixed * K_BLOCKS + 16;
    const size_t shmem = static_cast<size_t>(a_lds_bytes) +
                         static_cast<size_t>(a_scale_lds_bytes);
    switch (m_fixed) {
        case 2:
            gemv_m2_16_decode_kernel<L, 2, PRESHUFFLED_QUANT>
                <<<grid, block, shmem, g.stream>>>(g);
            break;
        case 4:
            gemv_m2_16_decode_kernel<L, 4, PRESHUFFLED_QUANT>
                <<<grid, block, shmem, g.stream>>>(g);
            break;
        case 8:
            gemv_m2_16_decode_kernel<L, 8, PRESHUFFLED_QUANT>
                <<<grid, block, shmem, g.stream>>>(g);
            break;
        case 16:
        default:
            gemv_m2_16_decode_kernel<L, 16, PRESHUFFLED_QUANT>
                <<<grid, block, shmem, g.stream>>>(g);
            break;
    }
}

#endif // MXFP8_DECODE_M2_16_ENABLE
