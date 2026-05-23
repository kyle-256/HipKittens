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

// =============================================================================
// PROBE — validates HIP clang register-asm honors specific VGPR slot for
// int4 storage + that mfma1616128_agpr_inplace reads from those slots
// (i.e., the SSA def-use chain is preserved across the cast).
//
// Build this as an instantiated kernel symbol so amdhsa.kernels metadata
// reveals whether vNN slots are honored. Per-symbol V usage should show
// the pinned slots; if compiler ignores the binding, V usage will spread.
// =============================================================================
__device__ __forceinline__ static void mma_probe_one_iter(
        float2 (&D)[2]) {
    // Pinned VGPR slots: A at v32, B at v40 (8 dwords each = 8 VGPRs).
    // HIP clang `register T asm("vNN")` is supposed to bind to consecutive
    // VGPRs starting at vNN. We probe int4 (4 dwords) granularity since
    // int4 is a primitive HIP type with reliable codegen.
    register int4 a_lo asm("v32");
    register int4 a_hi asm("v36");
    register int4 b_lo asm("v40");
    register int4 b_hi asm("v44");
    // Synthetic data — keeps SSA edges in place so allocator can't DCE.
    a_lo = {1, 2, 3, 4};
    a_hi = {5, 6, 7, 8};
    b_lo = {9, 10, 11, 12};
    b_hi = {13, 14, 15, 16};
    int32_t A_arr[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};
    int32_t B_arr[8] = {b_lo.x, b_lo.y, b_lo.z, b_lo.w,
                        b_hi.x, b_hi.y, b_hi.z, b_hi.w};
    mma_one_tile(D, A_arr, B_arr);
}

}  // namespace v2_pinned

// Instantiate the probe as a global symbol so its register usage shows up
// in amdhsa.kernels metadata for inspection.
__global__ void __probe_v2_pinned_mma_one_iter(float2* out) {
    float2 acc[2] = {{0.f, 0.f}, {0.f, 0.f}};
    v2_pinned::mma_probe_one_iter(acc);
    out[threadIdx.x * 2 + 0] = acc[0];
    out[threadIdx.x * 2 + 1] = acc[1];
}


// =============================================================================
// SESSION 3 PROBE — full pinned 1-acc K-loop with raw int4 storage
// =============================================================================
// Goal: validate that the architectural primitives (register-asm int4
// pinning + ds_read into pinned slots + mma_one_tile from pinned slots)
// produce the target register layout — V usage low, no spill — when
// composed at full per-acc scale.
//
// Per-warp 1 acc (64×32) needs 8 mma 16x16x128 calls per K iter.
// A frag: 32 dwords (8 int4 = 8 × 4 dwords) pinned at v[0:31]
// B frag: 16 dwords (4 int4) pinned at v[32:47]
// Acc:    32 floats/lane in AGPR, pinned via mma_one_tile's `+a` constraint
// =============================================================================

// =============================================================================
// SESSION 4 PROBE — DCE-resistant. Drops int32_t pack (which broke SSA);
// passes pinned int4 vars directly to a new mma_int4 wrapper that builds
// the intx8_t operand via vector init inside the wrapper, so register-asm
// pinning is preserved end-to-end. Adds volatile LDS reads + per-iter
// global writes so compiler can't prove acc dead.
// =============================================================================
namespace v2_pinned {
__device__ __forceinline__ static void mma_int4(
        float2 (&D)[2],
        int4 A_lo, int4 A_hi,
        int4 B_lo, int4 B_hi) {
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;
    intx8_t A = {A_lo.x, A_lo.y, A_lo.z, A_lo.w, A_hi.x, A_hi.y, A_hi.z, A_hi.w};
    intx8_t B = {B_lo.x, B_lo.y, B_lo.z, B_lo.w, B_hi.x, B_hi.y, B_hi.z, B_hi.w};
    // Direct deref into D — mirrors HK's mfma1616128_agpr_inplace pattern;
    // local-var indirection (the prior version) broke the AGPR `+a` binding.
    asm volatile(
        "v_mfma_f32_16x16x128_f8f6f4 %0, %1, %2, %0"
        : "+a"(*(floatx4_t*)D)
        : "v"(A), "v"(B));
}
}  // namespace v2_pinned

extern "C" __global__ __launch_bounds__(_NUM_THREADS, 1)
void __probe_v2_dce_resistant(
        const int4* __restrict__ a_lds_in,
        const int4* __restrict__ b_lds_in,
        float* __restrict__ out,
        int ki) {
    register int4 a_p0 asm("v0");
    register int4 a_p1 asm("v4");
    register int4 b_p0 asm("v32");
    register int4 b_p1 asm("v36");

    float2 acc[2] = {{0.f, 0.f}, {0.f, 0.f}};

    const int tid = threadIdx.x;
    for (int k = 0; k < ki; ++k) {
        // Real loads from kernel-arg pointer.
        a_p0 = a_lds_in[tid * 2     + k * 1024];
        a_p1 = a_lds_in[tid * 2 + 1 + k * 1024];
        b_p0 = b_lds_in[tid * 2     + k * 1024];
        b_p1 = b_lds_in[tid * 2 + 1 + k * 1024];

        v2_pinned::mma_int4(acc, a_p0, a_p1, b_p0, b_p1);

        // Per-iter visible write keeps acc live across loop, prevents DCE.
        ((volatile float*)out)[tid] = acc[0].x + acc[1].y;
    }
    out[tid     ] = acc[0].x;
    out[tid + 64] = acc[0].y;
    out[tid +128] = acc[1].x;
    out[tid +192] = acc[1].y;
}


extern "C" __global__ __launch_bounds__(_NUM_THREADS, 1)
void __probe_v2_pinned_full_kloop(
        const fp8e4m3* __restrict__ a_lds_ptr,
        const fp8e4m3* __restrict__ b_lds_ptr,
        float2* out,
        int ki) {
    // Pinned A fragment storage (per-warp 64 M × 128 K = 32 dwords/lane).
    // 8 int4 vars × 4 dwords each = 32 dwords; bind starting at v0.
    register int4 a_p0 asm("v0");
    register int4 a_p1 asm("v4");
    register int4 a_p2 asm("v8");
    register int4 a_p3 asm("v12");
    register int4 a_p4 asm("v16");
    register int4 a_p5 asm("v20");
    register int4 a_p6 asm("v24");
    register int4 a_p7 asm("v28");
    // Pinned B fragment storage (per-warp 32 N × 128 K = 16 dwords/lane).
    // 4 int4 vars; bind starting at v32.
    register int4 b_p0 asm("v32");
    register int4 b_p1 asm("v36");
    register int4 b_p2 asm("v40");
    register int4 b_p3 asm("v44");

    // Acc: 1 per-warp acc covering 64×32 = 8 mma tiles each with
    // float2[2] = 4 floats per tile. 8 × 4 = 32 floats / lane (AGPR).
    float2 acc[8][2];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { acc[i][0] = {0,0}; acc[i][1] = {0,0}; }

    const uint32_t lds_stride = 128;  // synthetic, for probe only
    const uint32_t a_off = static_cast<uint32_t>(threadIdx.x * 16);
    const uint32_t b_off = static_cast<uint32_t>(threadIdx.x * 16 + 4096);

    #pragma unroll 1
    for (int k = 0; k < ki; ++k) {
        // ds_read into pinned A slots
        a_p0 = *reinterpret_cast<const int4*>(a_lds_ptr + a_off + lds_stride * 0);
        a_p1 = *reinterpret_cast<const int4*>(a_lds_ptr + a_off + lds_stride * 1);
        a_p2 = *reinterpret_cast<const int4*>(a_lds_ptr + a_off + lds_stride * 2);
        a_p3 = *reinterpret_cast<const int4*>(a_lds_ptr + a_off + lds_stride * 3);
        a_p4 = *reinterpret_cast<const int4*>(a_lds_ptr + a_off + lds_stride * 4);
        a_p5 = *reinterpret_cast<const int4*>(a_lds_ptr + a_off + lds_stride * 5);
        a_p6 = *reinterpret_cast<const int4*>(a_lds_ptr + a_off + lds_stride * 6);
        a_p7 = *reinterpret_cast<const int4*>(a_lds_ptr + a_off + lds_stride * 7);
        // ds_read into pinned B slots
        b_p0 = *reinterpret_cast<const int4*>(b_lds_ptr + b_off + lds_stride * 0);
        b_p1 = *reinterpret_cast<const int4*>(b_lds_ptr + b_off + lds_stride * 1);
        b_p2 = *reinterpret_cast<const int4*>(b_lds_ptr + b_off + lds_stride * 2);
        b_p3 = *reinterpret_cast<const int4*>(b_lds_ptr + b_off + lds_stride * 3);

        // Pack consecutive int4s into int32_t[8] arrays for mma_one_tile.
        // Cast through unions in inline scope so compiler still sees the
        // pinned vars as the storage (SSA chain intact).
        int32_t A_pack[2][8] = {
            {a_p0.x, a_p0.y, a_p0.z, a_p0.w, a_p1.x, a_p1.y, a_p1.z, a_p1.w},
            {a_p2.x, a_p2.y, a_p2.z, a_p2.w, a_p3.x, a_p3.y, a_p3.z, a_p3.w},
        };
        int32_t B_pack[8] = {b_p0.x, b_p0.y, b_p0.z, b_p0.w,
                             b_p1.x, b_p1.y, b_p1.z, b_p1.w};

        // 8 mma per K iter for 1 acc (4 m-tiles × 2 n-tiles).
        // Synthetic mapping — probe correctness not the point; just validate
        // codegen + register layout.
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            v2_pinned::mma_one_tile(acc[t], A_pack[t & 1], B_pack);
        }
    }
    // Store acc to output (keeps SSA chain alive)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[(threadIdx.x * 8 + i) * 2 + 0] = acc[i][0];
        out[(threadIdx.x * 8 + i) * 2 + 1] = acc[i][1];
    }
}


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
// =============================================================================
// SESSION 2 — pinned 4-acc K-loop body (in-file copy of v1 with register-asm
// declarations on HK fragment types `A_row_reg` / `B_row_reg`).
//
// Test hypothesis: HIP clang accepts `register T t asm("vNN")` on HK
// fragment struct types (A_row_reg = rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>);
// binding the fragments to specific VGPR slots may give the register
// allocator a more favorable layout and reduce spill from baseline 37.
// =============================================================================
template<bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__device__ __forceinline__
void grouped_rcr_kernel_body_pinned(const grouped_layout_globals g) {
    using ST_rcr = ST_v2;
    __shared__ ST_rcr As[2][2];
    __shared__ ST_rcr Bs[2][2];
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;

    // [PINNED] Fragments bound to specific VGPR slots via register-asm.
    // If HIP clang honors this on HK struct types, the K-loop hot path's
    // fragment storage lives at fixed VGPRs, freeing compiler from
    // shuffling fragment data through general-purpose allocation.
    register A_row_reg a   asm("v32");
    register B_row_reg b0  asm("v40");
    register B_row_reg b1  asm("v48");
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;

    const int slots_eff = gridDim.x;
    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    const int chunk_size_eff = g.chunk_size > 0 ? g.chunk_size : 64;
    int pid = chiplet_transform_chunked(blockIdx.x, slots_eff, xcds_eff, chunk_size_eff);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;
    const int ki_dyn   = g.ki;

    init_group_cumsum_smem<MAX_G_PLUS_1>(g, s_offs, s_cum_tiles, s_total_tiles,
                                         num_pid_n, /*M_BLOCK_DIV=*/BLOCK_SIZE);
    const int total_tiles = s_total_tiles;

    constexpr int bpt = ST_rcr::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * _NUM_THREADS;
    constexpr int mpt = ST_rcr::rows * ST_rcr::cols * sizeof(fp8e4m3) / bpm;
    uint32_t soA[mpt], soB[mpt];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    for (int gt = pid; gt < total_tiles; gt += slots_eff) {
        int group_idx, m_start_g, M_g, bpr_g, br, bc;
        if (!dispatch_tile_in_group<MAX_G_PLUS_1>(
                gt, s_cum_tiles, s_offs, num_pid_n, g.group_m,
                /*M_BLOCK_DIV=*/BLOCK_SIZE,
                group_idx, m_start_g, M_g, bpr_g, br, bc)) continue;

        auto a_gl_g = g.a;
        auto c_gl_g = g.c;
        patch_per_group_gl_view(a_gl_g, c_gl_g, m_start_g, M_g);
        constexpr int m_subtile_A = 0;
        constexpr int m_subtile_C = 0;
        const int m_limit = M_g;

        auto a_co = [&](int s, int k) -> coord<ST_rcr> { return {0, 0, m_subtile_A + s, k}; };
        auto b_co = [&](int s, int k) -> coord<ST_rcr> { return {0, group_idx, s, k}; };

        auto load_a = [&](A_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_row_reg& dst, ST_rcr& tile, int wi) {
            auto sub = subtile_inplace<RBN, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto b_tile = [&](int stage, int which) -> ST_rcr& { return Bs[stage][which]; };

        zero(cA); zero(cB); zero(cC); zero(cD);

        int tic = 0, toc = 1;
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2,   0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], a_gl_g, a_co(br*2,   0), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, 0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][1], a_gl_g, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RCR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 0), g.b, b_co(bc*2,   1), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[toc][0], a_gl_g, a_co(br*2,   1), soA);
        rcr_8w_load_hoist<_NUM_THREADS>(b_tile(toc, 1), g.b, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RCR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        TK_PRAGMA_UNROLL(RCR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], a_gl_g, a_co(br*2+1, k+1), soA);
            TK_WAIT_LGKM(RCR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], a_gl_g, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
            TK_WAIT_VMCNT(RCR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // Epilog 1
        {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], a_gl_g, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(RCR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, b_tile(toc, 0), wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        // Epilog 2
        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_agpr_t<!FUSED_KTAIL>(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1);
            rcr_mma_agpr_t<!FUSED_KTAIL>(cC, a, b0);
            rcr_mma_agpr_t<!FUSED_KTAIL>(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // FUSED_KTAIL skipped in this pinned body for session 2; full port in S3.
        // For K_rem=64 shapes, dispatcher routes to v1 body via FUSED template
        // until S3 lands.

        const float combined_scale = resolve_combined_scale_grp(g);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        const int r0 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+wm);
        const int r1 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+WARPS_M+wm);
        const int c0 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+wn);
        const int c1 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+WARPS_N+wn);
        mul(cA, cA, combined_scale);
        store_c_tile_mn_masked_grouped(c_gl_g, cA, /*group_idx=*/0, r0, c0, m_limit, g.n);
        mul(cB, cB, combined_scale);
        store_c_tile_mn_masked_grouped(c_gl_g, cB, /*group_idx=*/0, r0, c1, m_limit, g.n);
        mul(cC, cC, combined_scale);
        store_c_tile_mn_masked_grouped(c_gl_g, cC, /*group_idx=*/0, r1, c0, m_limit, g.n);
        mul(cD, cD, combined_scale);
        store_c_tile_mn_masked_grouped(c_gl_g, cD, /*group_idx=*/0, r1, c1, m_limit, g.n);

        MAYBE_DRAIN_LGKM();
        __builtin_amdgcn_s_barrier();
    }
}


template<bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__global__ __launch_bounds__(_NUM_THREADS, 1)
void grouped_gemm_fp8_kernel_v2(const grouped_layout_globals g) {
    // Session 2: only K_rem==0 shapes use the pinned body; K_rem=64 (FUSED)
    // still routes to v1 body until session 3 ports the FUSED block.
    if constexpr (!FUSED_KTAIL) {
        grouped_rcr_kernel_body_pinned<N_MASKED_STORE, false>(g);
    } else {
        grouped_rcr_kernel_body<N_MASKED_STORE, true>(g);
    }
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
