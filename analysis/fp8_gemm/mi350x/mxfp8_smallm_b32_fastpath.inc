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
