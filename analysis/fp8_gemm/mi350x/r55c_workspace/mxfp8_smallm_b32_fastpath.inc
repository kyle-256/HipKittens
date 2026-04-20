// R42 Dev B — small-M MXFP8 fastpath (M=32..128 decode shapes).
//
// Scope: Improve MXFP8 V1-LEGACY-FALLBACK tail kernel for small M.
// R41 Dev C survey identified all 6 decode shapes FAIL the 95% rule with the
// existing scalar tail kernel:
//   - 32×4096×4096:   MXFP8 0.874 TF / FP8 1.125 TF = 77.7%
//   - 128×4096×4096:  MXFP8 0.882 TF / FP8 1.146 TF = 77.0%
//   - 32×8192×8192:   MXFP8 0.885 TF / FP8 1.139 TF = 77.7%
//   - 128×8192×8192:  MXFP8 0.885 TF / FP8 1.144 TF = 77.4%
//
// Mechanism of the 23% gap: BOTH MXFP8 and FP8 fall through to gemm_tail_kernel
// (a scalar 16×16 thread block per output tile) for M < BLK=256. MXFP8 carries
// per-K-iteration scalar scale loads on top of FP8's single per-tensor scalar.
// At K=4096 that's 8192 redundant `load_scale_scalar_preshuffled` calls per
// (row, col) thread — the scale only changes every 32 K-iters but the existing
// loop calls the helper every iter without hoisting.
//
// R41 Dev C's MFMA fastpath proposal (BLK_M=32 + BLK_N=128 + K-pipelined) is a
// 3-5 day effort. R42 Dev B's optimization is a tail-kernel-level lever: hoist
// the per-k_block scale load out of the inner 32-iter K loop. This is a small
// refactor (no new tile types, no MFMA wiring, no scale preshuffle change)
// that targets the exact mechanism of the 23% gap.
//
// Scoping vs full MFMA fastpath: the tail kernel will remain scalar (16×16
// thread block per output tile), so absolute TF stays in the same regime as
// the FP8 reference. The lever is the *MXFP8/FP8 ratio*, not absolute TF.
// Predicted lift: +20-25% on MXFP8 TF (close the scale-load redundancy gap).
// FP8 unchanged. Ratio target: ~95% (vs current 77%).
//
// Alternate-geometry note (BLK_M=16 vs BLK_M=32): NOT IN SCOPE. Both MXFP8 and
// FP8 use TAIL_BLOCK_M=16 already; the M-dimension blocking is already small
// enough for M=32 (2 row-blocks) and M=128 (8 row-blocks). The bottleneck is
// per-thread K-axis redundancy, not row-block geometry.
//
// Build: define MXFP8_SMALLM_B32_FASTPATH=1 at compile time to enable the
// optimized tail kernel. Default builds (MXFP8_SMALLM_B32_FASTPATH unset)
// see this translation unit only; the new kernel is named
// `gemm_tail_kernel_smallm_b32` to avoid colliding with existing
// `gemm_tail_kernel`. Dispatcher selection lives in the caller.
//
// nm-gate: when MXFP8_SMALLM_B32_FASTPATH=0 (default), no smallm symbols
// appear in the .so. R38 Dev D nm-gate pattern `_smallm_b32` covers this.

#ifndef MXFP8_SMALLM_B32_FASTPATH
#define MXFP8_SMALLM_B32_FASTPATH 0
#endif

#if MXFP8_SMALLM_B32_FASTPATH

template<Layout L, bool PRESHUFFLED_QUANT>
__global__ void gemm_tail_kernel_smallm_b32(const layout_globals g) {
    // Same launch geometry as gemm_tail_kernel: dim3(TAIL_BLOCK_N, TAIL_BLOCK_M)
    // block, ceil_div grid. Per-thread (row, col) ownership unchanged.
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= g.m || col >= g.n) {
        return;
    }

    const bool interior_mn = row < g.fast_m && col < g.fast_n;
    const bool fast_covers_cell = interior_mn && g.fast_m > 0 && g.fast_n > 0 && g.fast_k > 0;
    const bool needs_k_tail = g.fast_k < g.k;
    if (fast_covers_cell && !needs_k_tail) {
        return;
    }

    const int k0 = fast_covers_cell ? g.fast_k : 0;
    const int k_end = g.k;
    float acc = 0.0f;

    // OPTIMIZATION: outer loop over k_blocks (32-K-element groups) with the
    // scale load hoisted to outer scope. Inner loop walks 32 K within the
    // block, doing pure FP8 multiply-add with the cached scale already
    // multiplied in.
    //
    // Edge cases handled:
    //   - k0 may not be 32-aligned (when g.fast_k is not a multiple of 32 —
    //     can happen if BK!=128 future variant; today BK=128, fast_k always
    //     32-aligned, so first iter is full-block).
    //   - k_end may not be 32-aligned (rare, but handled by inner-loop
    //     boundary check).
    const int kb_start = k0 / 32;
    const int kb_end = (k_end + 31) / 32;

    #pragma unroll 1
    for (int kb = kb_start; kb < kb_end; ++kb) {
        // Hoisted scale load: once per 32 K-iters.
        float a_scale, b_scale;
        if constexpr (PRESHUFFLED_QUANT) {
            a_scale = load_scale_scalar_preshuffled(g.a_scale, row, kb);
            b_scale = load_scale_scalar_preshuffled(g.b_scale, col, kb);
        } else {
            a_scale = load_scale_scalar(g.a_scale, row, kb);
            b_scale = load_scale_scalar(g.b_scale, col, kb);
        }
        // Combined scale — apply once per pair, after the inner FP8 dot.
        const float scale_pair = a_scale * b_scale;

        const int kk_lo = kb * 32 < k0 ? k0 : kb * 32;
        const int kk_hi = (kb + 1) * 32 < k_end ? (kb + 1) * 32 : k_end;

        float partial = 0.0f;
        #pragma unroll 8
        for (int kk = kk_lo; kk < kk_hi; ++kk) {
            float a, b;
            if constexpr (L == Layout::RCR) {
                a = load_fp8_scalar(g.a, row, kk);
                b = load_fp8_scalar(g.b, col, kk);
            } else if constexpr (L == Layout::RRR) {
                a = load_fp8_scalar(g.a, row, kk);
                b = load_fp8_scalar(g.b, kk, col);
            } else {
                a = load_fp8_scalar(g.a, kk, row);
                b = load_fp8_scalar(g.b, kk, col);
            }
            partial += a * b;
        }
        acc += partial * scale_pair;
    }

    const float scaled = acc * g.scale;
    if (fast_covers_cell && needs_k_tail) {
        store_bf16_scalar(g.c, row, col, load_bf16_scalar(g.c, row, col) + scaled);
    } else {
        store_bf16_scalar(g.c, row, col, scaled);
    }
}

template __global__ void gemm_tail_kernel_smallm_b32<Layout::RCR, true>(const layout_globals);
template __global__ void gemm_tail_kernel_smallm_b32<Layout::RRR, true>(const layout_globals);
template __global__ void gemm_tail_kernel_smallm_b32<Layout::CRR, true>(const layout_globals);
template __global__ void gemm_tail_kernel_smallm_b32<Layout::RCR, false>(const layout_globals);
template __global__ void gemm_tail_kernel_smallm_b32<Layout::RRR, false>(const layout_globals);
template __global__ void gemm_tail_kernel_smallm_b32<Layout::CRR, false>(const layout_globals);

#endif  // MXFP8_SMALLM_B32_FASTPATH

// ============================================================================
// R45 Dev C — B-side scalar-load vectorization (gemm_tail_kernel_smallm_b32_bvec)
// ============================================================================
//
// Scope (R44 Dev B replacement recommendation, R44+ priority #2): the highest-
// leverage lever for KV-decode N=1024 shapes after R42B closed the scale-load
// redundancy gap. The R42B kernel still issues per-thread per-K-iter SCALAR
// fp8 loads for both A and B operands. At K=4096, that's 4096 byte-loads per
// thread per operand = 8K HBM transactions per (row, col) thread. HBM
// efficiency observed at ~50% (per R44B inspection notes) due to scalar load
// granularity (1-byte transactions issued as 16-byte cache lines, 6.25%
// effective utilization per request). Moving to 4-byte (uint32_t) packed
// loads gives 4x fewer load instructions and >50% HBM efficiency.
//
// ----- Memory layout per layout L (B operand) -----
//   - RCR: B is (N, K) row-major. b[col, kk] is at byte offset col*K + kk.
//          Adjacent kk are CONTIGUOUS (1 byte stride). Direct uint32_t load
//          of 4 fp8 values along K axis works WITHOUT warp-shuffle.
//   - RRR: B is (K, N) row-major. b[kk, col] is at byte offset kk*N + col.
//          Adjacent kk are STRIDED (N-byte stride). Per-thread vectorization
//          along K axis would require warp-shuffle to gather strided values.
//   - CRR: same B layout as RRR. Strided.
//
// ----- A operand layout per layout L -----
//   - RCR: A is (M, K) row-major. a[row, kk]: K-contiguous. Vectorizable.
//   - RRR: A is (M, K) row-major. a[row, kk]: K-contiguous. Vectorizable.
//   - CRR: A is (K, M) row-major. a[kk, row]: K-strided. Scalar.
//
// ----- Design decision: RCR-only B+A vectorization, RRR mixed, CRR scalar -----
// The benchmark target shapes (M=32/128, N=1024, K=4096/8192) are RCR. The
// production KV-decode dispatch route is gemm_rcr_pq -> dispatch_pq_v2<RCR> ->
// V1-LEGACY-FALLBACK -> R43C unified small-M waterfall -> SMALLM-B32-TAIL.
// Vectorizing the RCR variant captures 100% of the production KV-decode load
// in the gold-standard regression set.
//
// For RRR/CRR strided B, the warp-shuffle gather pattern (each lane loads
// uint32 of 4 consecutive cols at one K, then __shfl gathers values for the
// thread's own col across 4 K positions) was DESIGN-EXPLORED but DEFERRED:
//   1) The TAIL_BLOCK geometry is 16 cols x 16 rows = 256 threads = 4 waves.
//      Adjacent lanes within a wavefront are (ty, tx) = (0..3, 0..15). For
//      K-axis shuffle to fuse: lanes within the same row (same ty) need to
//      cooperate. Only 16 lanes per row -> 4x redistribution requires lane-
//      group of 4 within row -> __shfl(value, src_lane_within_4) where
//      src_lane = (lane & 0xc) | k_offset_within_4. Feasible but adds
//      __shfl ops in the inner loop -- net win depends on whether AMD's
//      LDS+shuffle path beats 4x more byte-loads.
//   2) Empirical priority: RCR is the dominant production layout. RRR/CRR
//      tail kernels execute on weight-major decode shapes which are less
//      common in inference. If RCR vec succeeds, RRR/CRR shuffle becomes
//      future-priority work.
//
// In this variant, RRR vectorizes A only (B scalar); CRR is fully scalar
// (R42B fallback). RCR uses vectorized 4-byte loads on both A and B.
//
// ----- Build gate -----
//   MXFP8_SMALLM_B32_BSIDE_VEC_ENABLE=1 enables `gemm_tail_kernel_smallm_b32_bvec`
//   AND requires MXFP8_SMALLM_B32_FASTPATH=1 (R42B must be enabled -- this
//   variant is a strict superset of R42B's hoisted-scale K-loop).
//   Default builds (macro unset / 0): byte-identical to current default,
//   the new kernel is not emitted.
//
// nm-gate: when MXFP8_SMALLM_B32_BSIDE_VEC_ENABLE=0, no `_bvec` symbols in .so.
// Pattern: `_smallm_b32_bvec` (extends R38 Dev D nm-gate `_smallm_b32` family).
//
// ----- Correctness contract -----
// Vectorized loads decode exactly the same fp8 byte sequence as the scalar
// kernel; arithmetic is bitwise identical under -ffast-math (FMA reordering
// only at the unrolled inner-loop level, no mathematical reassociation
// across the per-32 scale boundary). max|delta| = 0.0 expected vs R42B.

#ifndef MXFP8_SMALLM_B32_BSIDE_VEC_ENABLE
#define MXFP8_SMALLM_B32_BSIDE_VEC_ENABLE 0
#endif

#if MXFP8_SMALLM_B32_BSIDE_VEC_ENABLE
#if !MXFP8_SMALLM_B32_FASTPATH
#error "MXFP8_SMALLM_B32_BSIDE_VEC_ENABLE requires MXFP8_SMALLM_B32_FASTPATH=1"
#endif

template<Layout L, bool PRESHUFFLED_QUANT>
__global__ void gemm_tail_kernel_smallm_b32_bvec(const layout_globals g) {
    // Same launch geometry as gemm_tail_kernel_smallm_b32 (R42B): per-thread
    // (row, col) ownership over a 16x16 tile.
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= g.m || col >= g.n) {
        return;
    }

    const bool interior_mn = row < g.fast_m && col < g.fast_n;
    const bool fast_covers_cell = interior_mn && g.fast_m > 0 && g.fast_n > 0 && g.fast_k > 0;
    const bool needs_k_tail = g.fast_k < g.k;
    if (fast_covers_cell && !needs_k_tail) {
        return;
    }

    const int k0 = fast_covers_cell ? g.fast_k : 0;
    const int k_end = g.k;
    float acc = 0.0f;

    const int kb_start = k0 / 32;
    const int kb_end = (k_end + 31) / 32;

    // Raw pointers + strides for vectorization paths.
    // RCR: a[row, kk] @ a_raw + row*K + kk; b[col, kk] @ b_raw + col*K + kk
    // RRR: a[row, kk] @ a_raw + row*K + kk; b[kk, col] @ b_raw + kk*N + col
    // CRR: a[kk, row] @ a_raw + kk*M + row; b[kk, col] @ b_raw + kk*N + col
    const fp8e4m3* __restrict__ a_raw = g.a.raw_ptr;
    const fp8e4m3* __restrict__ b_raw = g.b.raw_ptr;
    const int K = g.k;

    #pragma unroll 1
    for (int kb = kb_start; kb < kb_end; ++kb) {
        // Hoisted scale load (R42B mechanism preserved).
        float a_scale, b_scale;
        if constexpr (PRESHUFFLED_QUANT) {
            a_scale = load_scale_scalar_preshuffled(g.a_scale, row, kb);
            b_scale = load_scale_scalar_preshuffled(g.b_scale, col, kb);
        } else {
            a_scale = load_scale_scalar(g.a_scale, row, kb);
            b_scale = load_scale_scalar(g.b_scale, col, kb);
        }
        const float scale_pair = a_scale * b_scale;

        const int kk_lo = kb * 32 < k0 ? k0 : kb * 32;
        const int kk_hi = (kb + 1) * 32 < k_end ? (kb + 1) * 32 : k_end;

        float partial = 0.0f;

        if constexpr (L == Layout::RCR) {
            // ----- RCR vectorized path -----
            // Both A and B are K-contiguous. When [kk_lo, kk_hi) is a full
            // 32-element block aligned to 4-byte boundary (the common case
            // for K%32==0, k0%32==0 -- true for all production KV-decode
            // shapes since K in {4096,8192}), do 8 vec-quads of 4 fp8 each.
            // Boundary case (k_tail) falls back to scalar.
            const bool full_block = (kk_lo == kb * 32) && (kk_hi == kb * 32 + 32);
            if (full_block && ((kk_lo & 3) == 0)) {
                const fp8e4m3* a_ptr = a_raw + row * K + kk_lo;
                const fp8e4m3* b_ptr = b_raw + col * K + kk_lo;
                #pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const uint32_t a_pack = *reinterpret_cast<const uint32_t*>(a_ptr + q * 4);
                    const uint32_t b_pack = *reinterpret_cast<const uint32_t*>(b_ptr + q * 4);
                    const fp8e4m3_4 a4 = std::bit_cast<fp8e4m3_4>(a_pack);
                    const fp8e4m3_4 b4 = std::bit_cast<fp8e4m3_4>(b_pack);
                    const float4 af = base_types::convertor<float4, fp8e4m3_4>::convert(a4);
                    const float4 bf = base_types::convertor<float4, fp8e4m3_4>::convert(b4);
                    partial += af.x * bf.x;
                    partial += af.y * bf.y;
                    partial += af.z * bf.z;
                    partial += af.w * bf.w;
                }
            } else {
                #pragma unroll 8
                for (int kk = kk_lo; kk < kk_hi; ++kk) {
                    const float a = load_fp8_scalar(g.a, row, kk);
                    const float b = load_fp8_scalar(g.b, col, kk);
                    partial += a * b;
                }
            }
        } else if constexpr (L == Layout::RRR) {
            // ----- RRR mixed path -----
            // A K-contiguous (vec); B K-strided (scalar). 4x lift on A loads,
            // none on B. Net is HBM-bandwidth-better since A and B reads
            // contend; expect ~half the lift of RCR.
            const bool full_block = (kk_lo == kb * 32) && (kk_hi == kb * 32 + 32);
            if (full_block && ((kk_lo & 3) == 0)) {
                const fp8e4m3* a_ptr = a_raw + row * K + kk_lo;
                #pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const uint32_t a_pack = *reinterpret_cast<const uint32_t*>(a_ptr + q * 4);
                    const fp8e4m3_4 a4 = std::bit_cast<fp8e4m3_4>(a_pack);
                    const float4 af = base_types::convertor<float4, fp8e4m3_4>::convert(a4);
                    const int kk = kk_lo + q * 4;
                    const float b0 = load_fp8_scalar(g.b, kk + 0, col);
                    const float b1 = load_fp8_scalar(g.b, kk + 1, col);
                    const float b2 = load_fp8_scalar(g.b, kk + 2, col);
                    const float b3 = load_fp8_scalar(g.b, kk + 3, col);
                    partial += af.x * b0;
                    partial += af.y * b1;
                    partial += af.z * b2;
                    partial += af.w * b3;
                }
            } else {
                #pragma unroll 8
                for (int kk = kk_lo; kk < kk_hi; ++kk) {
                    const float a = load_fp8_scalar(g.a, row, kk);
                    const float b = load_fp8_scalar(g.b, kk, col);
                    partial += a * b;
                }
            }
        } else {
            // ----- CRR scalar fallback -----
            // Both A and B K-strided; no per-thread vectorization without
            // warp-shuffle. Falls back to R42B scalar path.
            #pragma unroll 8
            for (int kk = kk_lo; kk < kk_hi; ++kk) {
                const float a = load_fp8_scalar(g.a, kk, row);
                const float b = load_fp8_scalar(g.b, kk, col);
                partial += a * b;
            }
        }
        acc += partial * scale_pair;
    }

    const float scaled = acc * g.scale;
    if (fast_covers_cell && needs_k_tail) {
        store_bf16_scalar(g.c, row, col, load_bf16_scalar(g.c, row, col) + scaled);
    } else {
        store_bf16_scalar(g.c, row, col, scaled);
    }
}

template __global__ void gemm_tail_kernel_smallm_b32_bvec<Layout::RCR, true>(const layout_globals);
template __global__ void gemm_tail_kernel_smallm_b32_bvec<Layout::RRR, true>(const layout_globals);
template __global__ void gemm_tail_kernel_smallm_b32_bvec<Layout::CRR, true>(const layout_globals);
template __global__ void gemm_tail_kernel_smallm_b32_bvec<Layout::RCR, false>(const layout_globals);
template __global__ void gemm_tail_kernel_smallm_b32_bvec<Layout::RRR, false>(const layout_globals);
template __global__ void gemm_tail_kernel_smallm_b32_bvec<Layout::CRR, false>(const layout_globals);

#endif  // MXFP8_SMALLM_B32_BSIDE_VEC_ENABLE
