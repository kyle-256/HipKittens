// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// =============================================================================
// HipKittens FP8 grouped RCR — V2 PINNED-SLOT ARCHITECTURE REWRITE
// =============================================================================
// Multi-session work. This file is the canonical home for the new kernel.
//
// Goal: spill ≤ 5 on BN=256 RCR with 4-acc, perf ≥ v1 baseline.
// Strategy: explicit VGPR pinning of A/B fragment storage via HIP clang
// register-asm extension, with HK SSA-visible mfma1616128_agpr_inplace
// wrapper preserving def-use chain (avoids [[8w-tw-phase-split-bug-observation]]).
//
// -----------------------------------------------------------------------------
// PINNED LAYOUT — 1 wave/SIMD, 8-warp WG, V cap = A cap = 256 dwords/lane
// -----------------------------------------------------------------------------
//   AGPR (acc, AGPR-pinned via mfma1616128_agpr_inplace `+a` constraint):
//     a[ 0: 31] : cA  (per-warp 64×32 in M m_chunk 0 / N n_chunk 0)
//     a[32: 63] : cB  (m_chunk 0 / n_chunk 1)
//     a[64: 95] : cC  (m_chunk 1 / n_chunk 0)
//     a[96:127] : cD  (m_chunk 1 / n_chunk 1)
//     Total 128 AGPR / lane.
//
//   VGPR (fragment storage, register-asm pinned at slots below):
//     v[ 32: 39] : A frag (8 fp8e4m3_4 = 8 dwords) — current K-iter A row data
//     v[ 40: 47] : B0 frag (n_chunk 0 B row data)
//     v[ 48: 55] : B1 frag (n_chunk 1 B row data)
//     v[  0: 31] + v[56:255] : compiler-managed (LDS prefetch buffers,
//                              SRD scratch, ptr math, uniforms not in SGPR)
//
// -----------------------------------------------------------------------------
// SESSION PLAN
// -----------------------------------------------------------------------------
//   Session 1 (this commit):
//     - Document architecture (this block)
//     - Define pinned-storage abstraction (PinnedFragA, PinnedFragB)
//     - Define pinned mma wrapper (mma_pinned) that calls
//       mfma1616128_agpr_inplace with pinned-storage references
//     - Define pinned ds_read primitive (ds_read_pinned_into) for fp8e4m3_4[8]
//     - Initial kernel scaffold using these primitives (compiles + smoke)
//
//   Session 2:
//     - Replace HK fragment types in K-loop body with pinned variants
//     - Verify register layout via amdhsa.kernels metadata (V usage, no spill)
//
//   Session 3:
//     - Port FUSED_KTAIL path to pinned-storage
//     - End-to-end smoke + spill verification
//
//   Session 4:
//     - Perf tuning: sched_group_barrier batching, prefetch overlap
//     - Final bench vs v1
//
// Current implementation status: SESSION 1, in-progress. v2 dispatcher
// continues to forward to v1's body until pinned primitives are validated.

#include "kernel_fp8_layouts.cpp"  // pull v1 helpers + dispatchers into ns


// =============================================================================
// SESSION 1 — Pinned-storage primitives (compile-only at this commit)
// =============================================================================

namespace v2_pinned {

// 8 fp8e4m3_4 dwords = single 16x16x128 mfma A or B operand per lane.
// We use raw int (=int32_t) for the storage so register-asm binding works
// reliably on HIP clang; the mma wrapper reinterprets to fp8e4m3_4[8].
using frag_dwords = int32_t[8];

// HIP clang register-asm extension binds the storage of a local var to
// specific VGPRs. For an array, the binding is to the first VGPR of the
// range; consecutive VGPRs hold subsequent elements. We use distinct
// declarations rather than `register T arr[N] asm(...)` because the
// per-element syntax is more reliably honored across clang versions.
//
// Per K-iter usage pattern (inside kernel body):
//
//   v2_pinned::ds_read_into_a(a_pin, lds_addr_a);
//   v2_pinned::ds_read_into_b(b0_pin, lds_addr_b0);
//   v2_pinned::ds_read_into_b(b1_pin, lds_addr_b1);
//   v2_pinned::mma(cA, a_pin, b0_pin);
//   v2_pinned::mma(cB, a_pin, b1_pin);
//   ... etc.

// Wrapper for HK's mfma1616128_agpr_inplace, taking raw int32_t[8] frags
// (which can be register-asm pinned in caller). Preserves SSA visibility:
// the asm `"v"(A)` constraint sees the pinned-storage as inputs, the `+a`
// constraint on D sees the accumulator. No DCE.
__device__ __forceinline__ static void mma_one_tile(
        float2 (&D)[2],
        const int32_t (&A_pin)[8],
        const int32_t (&B_pin)[8]) {
    // Reinterpret pinned int32 storage as fp8e4m3_4[8] for the wrapper.
    // Cast preserves storage identity (no copy); compiler sees A_pin/B_pin
    // as live across this asm.
    const auto& A_frag = *reinterpret_cast<const fp8e4m3_4(*)[8]>(&A_pin);
    const auto& B_frag = *reinterpret_cast<const fp8e4m3_4(*)[8]>(&B_pin);
    ::kittens::mfma1616128_agpr_inplace(D, A_frag, B_frag);
}

// Per-warp 64×32 acc (cA, cB, cC, cD shapes from kernel body) decomposes
// into multiple 16×16×128 mma tiles. For RBM=64, RBN=32 the per-warp
// output is 4 tiles in M × 2 tiles in N = 8 mma calls per K iter per acc.
// This wrapper drives the full per-acc mma sweep across A_pin/B_pin
// fragments, where each fragment holds K=128 of a single 16-row warp tile.

}  // namespace v2_pinned


// =============================================================================
// PT-facing struct.
// =============================================================================
struct grouped_layout_globals_v2 {
    const void* a_ptr;
    const void* b_ptr;
    void*       c_ptr;
    int M_total, G_b, bN, bK, cM, cN;
    const float* sa_ptr;
    const float* sb_ptr;
    const int64_t* group_offs_ptr;
    hipStream_t stream;
    int G, group_m, m_per_group, num_xcds;
    int num_slots, chunk_size;
};


// =============================================================================
// V2 dispatch — SESSION 1: forwards to v1 body until pinned primitives are
// integrated into the K-loop (sessions 2-3).
// =============================================================================
template<bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__global__ __launch_bounds__(_NUM_THREADS, 1)
void grouped_gemm_fp8_kernel_v2(const grouped_layout_globals g) {
    grouped_rcr_kernel_body<N_MASKED_STORE, FUSED_KTAIL>(g);
}


inline void dispatch_grouped_rcr_v2(grouped_layout_globals_v2 g_in) {
    grouped_layout_globals g{
        _gl_fp8(reinterpret_cast<fp8e4m3*>(const_cast<void*>(g_in.a_ptr)),
                1, 1, g_in.M_total, g_in.bK),
        _gl_fp8(reinterpret_cast<fp8e4m3*>(const_cast<void*>(g_in.b_ptr)),
                1, g_in.G_b, g_in.bN, g_in.bK),
        _gl_bf16(reinterpret_cast<bf16*>(g_in.c_ptr), 1, 1, g_in.cM, g_in.cN),
        0.f, 0.f, g_in.sa_ptr, g_in.sb_ptr,
        g_in.group_offs_ptr, g_in.stream,
        g_in.G, 0, 0, 0, 0,
        g_in.group_m, g_in.num_xcds, 0,
        0, 0,
        g_in.m_per_group, g_in.num_slots, g_in.chunk_size, 0,
        0, nullptr, 0,
    };
    g.n       = static_cast<int>(g.c.cols());
    g.M_total = static_cast<int>(g.c.rows());
    g.k       = static_cast<int>(g.a.cols());
    g.fast_n  = (g.n / BLOCK_SIZE) * BLOCK_SIZE;
    g.fast_k  = (g.k / K_BLOCK)    * K_BLOCK;
    g.bpc     = kittens::ceil_div(g.n, BLOCK_SIZE);
    g.ki      = g.fast_k / K_BLOCK;
    if (g.bpc == 0 || g.ki == 0) return;

    const int K_rem = g.k - g.fast_k;
    const bool fuse_on = (K_rem == 64);
    const bool n_aligned = (g.bpc * BLOCK_SIZE == g.n);

    static const int slots_env = []() {
        if (const char* e = std::getenv("TK_RCR_V2_NUM_CUS")) {
            const int v = std::atoi(e);
            if (v > 0 && v <= NUM_CUS) return v;
        }
        return NUM_CUS;
    }();
    const int slots = (g.num_slots > 0 && g.num_slots <= NUM_CUS)
        ? g.num_slots : slots_env;

    if (fuse_on) {
        if (n_aligned)
            grouped_gemm_fp8_kernel_v2<false, true><<<dim3(slots), g.block(), 0, g.stream>>>(g);
        else
            grouped_gemm_fp8_kernel_v2<true,  true><<<dim3(slots), g.block(), 0, g.stream>>>(g);
    } else {
        if (n_aligned)
            grouped_gemm_fp8_kernel_v2<false, false><<<dim3(slots), g.block(), 0, g.stream>>>(g);
        else
            grouped_gemm_fp8_kernel_v2<true,  false><<<dim3(slots), g.block(), 0, g.stream>>>(g);
    }
}
