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

// R12: builtin mfma path — compiler chooses register class for D (typically
// VGPR-acc when used). Use for cD only to test if spreading acc between
// AGPR (cA/cB/cC via +a) and VGPR (cD via builtin) reduces overall spill.
__device__ __forceinline__ static void mma_int4_vacc(
        float2 (&D)[2],
        int4 A_lo, int4 A_hi,
        int4 B_lo, int4 B_hi) {
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;
    intx8_t A = {A_lo.x, A_lo.y, A_lo.z, A_lo.w, A_hi.x, A_hi.y, A_hi.z, A_hi.w};
    intx8_t B = {B_lo.x, B_lo.y, B_lo.z, B_lo.w, B_hi.x, B_hi.y, B_hi.z, B_hi.w};
    asm volatile(
        "v_mfma_f32_16x16x128_f8f6f4 %0, %1, %2, %0"
        : "+v"(*(floatx4_t*)D)
        : "v"(A), "v"(B));
}

}  // namespace v2_pinned

// =============================================================================
// SESSION 5 — MINIMAL REAL PINNED GEMM (16x16x128, single warp, single mma)
// =============================================================================
// First end-to-end test that the pinned primitives actually produce mfma
// codegen (A>0 in amdhsa metadata) when the data path is forced concrete:
//   - 64-thread (single warp) WG
//   - Loads 8 fp8e4m3_4 / lane of A (16 rows × 128 K) and B (16 cols × 128 K)
//     into pinned int4 slots v0:v7 and v8:v15
//   - One mfma 16x16x128 with `+a` AGPR-bound acc
//   - Stores acc to HBM as float4/lane
//
// If this kernel shows A > 0 + spill = 0 in metadata, the pinned mfma
// asm pattern is functional. We can then scale to per-warp 64×32 and
// integrate into the K-loop.
// =============================================================================
// R5: 8-warp WG (512 thread) — production thread config, per-warp same pinned layout
extern "C" __global__ __launch_bounds__(512, 1)
void __probe_v2_8warp_pinned(
        const int4* __restrict__ A_g,
        const int4* __restrict__ B_g,
        float4* __restrict__ C,
        int ki) {
    __shared__ int4 A_lds[2048];
    __shared__ int4 B_lds[1024];
    const int tid = threadIdx.x;
    const int wid = tid / 64;
    const int lid = tid % 64;

    register int4 a00 asm("v0");  register int4 a01 asm("v4");
    register int4 a02 asm("v8");  register int4 a03 asm("v12");
    register int4 a04 asm("v16"); register int4 a05 asm("v20");
    register int4 a06 asm("v24"); register int4 a07 asm("v28");
    register int4 a10 asm("v32"); register int4 a11 asm("v36");
    register int4 a12 asm("v40"); register int4 a13 asm("v44");
    register int4 a14 asm("v48"); register int4 a15 asm("v52");
    register int4 a16 asm("v56"); register int4 a17 asm("v60");
    register int4 b00 asm("v64"); register int4 b01 asm("v68");
    register int4 b02 asm("v72"); register int4 b03 asm("v76");
    register int4 b10 asm("v80"); register int4 b11 asm("v84");
    register int4 b12 asm("v88"); register int4 b13 asm("v92");

    float2 cA[8][2], cB[8][2], cC[8][2], cD[8][2];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        cA[i][0]={0,0}; cA[i][1]={0,0}; cB[i][0]={0,0}; cB[i][1]={0,0};
        cC[i][0]={0,0}; cC[i][1]={0,0}; cD[i][0]={0,0}; cD[i][1]={0,0};
    }

    #pragma unroll 1
    for (int k = 0; k < ki; ++k) {
        // Cooperative load gmem → smem (512 threads load 2048 int4 + 1024 int4)
        #pragma unroll
        for (int i = 0; i < 4; ++i) A_lds[tid + i*512] = A_g[k*2048 + tid + i*512];
        #pragma unroll
        for (int i = 0; i < 2; ++i) B_lds[tid + i*512] = B_g[k*1024 + tid + i*512];
        __syncthreads();

        // Per-warp ds_read using wid offset
        a00 = A_lds[wid*1024 + lid*16+ 0]; a01 = A_lds[wid*1024 + lid*16+ 1];
        a02 = A_lds[wid*1024 + lid*16+ 2]; a03 = A_lds[wid*1024 + lid*16+ 3];
        a04 = A_lds[wid*1024 + lid*16+ 4]; a05 = A_lds[wid*1024 + lid*16+ 5];
        a06 = A_lds[wid*1024 + lid*16+ 6]; a07 = A_lds[wid*1024 + lid*16+ 7];
        a10 = A_lds[wid*1024 + lid*16+ 8]; a11 = A_lds[wid*1024 + lid*16+ 9];
        a12 = A_lds[wid*1024 + lid*16+10]; a13 = A_lds[wid*1024 + lid*16+11];
        a14 = A_lds[wid*1024 + lid*16+12]; a15 = A_lds[wid*1024 + lid*16+13];
        a16 = A_lds[wid*1024 + lid*16+14]; a17 = A_lds[wid*1024 + lid*16+15];
        b00 = B_lds[wid*512 + lid*8+0]; b01 = B_lds[wid*512 + lid*8+1];
        b02 = B_lds[wid*512 + lid*8+2]; b03 = B_lds[wid*512 + lid*8+3];
        b10 = B_lds[wid*512 + lid*8+4]; b11 = B_lds[wid*512 + lid*8+5];
        b12 = B_lds[wid*512 + lid*8+6]; b13 = B_lds[wid*512 + lid*8+7];

        v2_pinned::mma_int4(cA[0], a00, a01, b00, b01);
        v2_pinned::mma_int4(cA[1], a00, a01, b02, b03);
        v2_pinned::mma_int4(cA[2], a02, a03, b00, b01);
        v2_pinned::mma_int4(cA[3], a02, a03, b02, b03);
        v2_pinned::mma_int4(cA[4], a04, a05, b00, b01);
        v2_pinned::mma_int4(cA[5], a04, a05, b02, b03);
        v2_pinned::mma_int4(cA[6], a06, a07, b00, b01);
        v2_pinned::mma_int4(cA[7], a06, a07, b02, b03);
        v2_pinned::mma_int4(cB[0], a00, a01, b10, b11);
        v2_pinned::mma_int4(cB[1], a00, a01, b12, b13);
        v2_pinned::mma_int4(cB[2], a02, a03, b10, b11);
        v2_pinned::mma_int4(cB[3], a02, a03, b12, b13);
        v2_pinned::mma_int4(cB[4], a04, a05, b10, b11);
        v2_pinned::mma_int4(cB[5], a04, a05, b12, b13);
        v2_pinned::mma_int4(cB[6], a06, a07, b10, b11);
        v2_pinned::mma_int4(cB[7], a06, a07, b12, b13);
        v2_pinned::mma_int4(cC[0], a10, a11, b00, b01);
        v2_pinned::mma_int4(cC[1], a10, a11, b02, b03);
        v2_pinned::mma_int4(cC[2], a12, a13, b00, b01);
        v2_pinned::mma_int4(cC[3], a12, a13, b02, b03);
        v2_pinned::mma_int4(cC[4], a14, a15, b00, b01);
        v2_pinned::mma_int4(cC[5], a14, a15, b02, b03);
        v2_pinned::mma_int4(cC[6], a16, a17, b00, b01);
        v2_pinned::mma_int4(cC[7], a16, a17, b02, b03);
        v2_pinned::mma_int4(cD[0], a10, a11, b10, b11);
        v2_pinned::mma_int4(cD[1], a10, a11, b12, b13);
        v2_pinned::mma_int4(cD[2], a12, a13, b10, b11);
        v2_pinned::mma_int4(cD[3], a12, a13, b12, b13);
        v2_pinned::mma_int4(cD[4], a14, a15, b10, b11);
        v2_pinned::mma_int4(cD[5], a14, a15, b12, b13);
        v2_pinned::mma_int4(cD[6], a16, a17, b10, b11);
        v2_pinned::mma_int4(cD[7], a16, a17, b12, b13);
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        C[tid*32+i   ] = *(float4*)cA[i];
        C[tid*32+i+ 8] = *(float4*)cB[i];
        C[tid*32+i+16] = *(float4*)cC[i];
        C[tid*32+i+24] = *(float4*)cD[i];
    }
}


// R4: full K-loop + LDS + 4-acc 32-mma per iter (per-warp 128M×64N)
extern "C" __global__ __launch_bounds__(64, 1)
void __probe_v2_kloop_pinned(
        const int4* __restrict__ A_g,
        const int4* __restrict__ B_g,
        float4* __restrict__ C,
        int ki) {
    __shared__ int4 A_lds[1024];
    __shared__ int4 B_lds[512];
    const int tid = threadIdx.x;

    register int4 a00 asm("v0");  register int4 a01 asm("v4");
    register int4 a02 asm("v8");  register int4 a03 asm("v12");
    register int4 a04 asm("v16"); register int4 a05 asm("v20");
    register int4 a06 asm("v24"); register int4 a07 asm("v28");
    register int4 a10 asm("v32"); register int4 a11 asm("v36");
    register int4 a12 asm("v40"); register int4 a13 asm("v44");
    register int4 a14 asm("v48"); register int4 a15 asm("v52");
    register int4 a16 asm("v56"); register int4 a17 asm("v60");
    register int4 b00 asm("v64"); register int4 b01 asm("v68");
    register int4 b02 asm("v72"); register int4 b03 asm("v76");
    register int4 b10 asm("v80"); register int4 b11 asm("v84");
    register int4 b12 asm("v88"); register int4 b13 asm("v92");

    float2 cA[8][2], cB[8][2], cC[8][2], cD[8][2];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        cA[i][0]={0,0}; cA[i][1]={0,0}; cB[i][0]={0,0}; cB[i][1]={0,0};
        cC[i][0]={0,0}; cC[i][1]={0,0}; cD[i][0]={0,0}; cD[i][1]={0,0};
    }

    #pragma unroll 1
    for (int k = 0; k < ki; ++k) {
        // Load gmem → LDS
        #pragma unroll
        for (int i = 0; i < 16; ++i) A_lds[tid*16+i] = A_g[k*1024+tid*16+i];
        #pragma unroll
        for (int i = 0; i < 8;  ++i) B_lds[tid*8+i]  = B_g[k*512+tid*8+i];
        __syncthreads();

        // ds_read into pinned slots
        a00 = A_lds[tid*16+ 0]; a01 = A_lds[tid*16+ 1];
        a02 = A_lds[tid*16+ 2]; a03 = A_lds[tid*16+ 3];
        a04 = A_lds[tid*16+ 4]; a05 = A_lds[tid*16+ 5];
        a06 = A_lds[tid*16+ 6]; a07 = A_lds[tid*16+ 7];
        a10 = A_lds[tid*16+ 8]; a11 = A_lds[tid*16+ 9];
        a12 = A_lds[tid*16+10]; a13 = A_lds[tid*16+11];
        a14 = A_lds[tid*16+12]; a15 = A_lds[tid*16+13];
        a16 = A_lds[tid*16+14]; a17 = A_lds[tid*16+15];
        b00 = B_lds[tid*8+0]; b01 = B_lds[tid*8+1];
        b02 = B_lds[tid*8+2]; b03 = B_lds[tid*8+3];
        b10 = B_lds[tid*8+4]; b11 = B_lds[tid*8+5];
        b12 = B_lds[tid*8+6]; b13 = B_lds[tid*8+7];

        // 32 mma per K iter
        v2_pinned::mma_int4(cA[0], a00, a01, b00, b01);
        v2_pinned::mma_int4(cA[1], a00, a01, b02, b03);
        v2_pinned::mma_int4(cA[2], a02, a03, b00, b01);
        v2_pinned::mma_int4(cA[3], a02, a03, b02, b03);
        v2_pinned::mma_int4(cA[4], a04, a05, b00, b01);
        v2_pinned::mma_int4(cA[5], a04, a05, b02, b03);
        v2_pinned::mma_int4(cA[6], a06, a07, b00, b01);
        v2_pinned::mma_int4(cA[7], a06, a07, b02, b03);
        v2_pinned::mma_int4(cB[0], a00, a01, b10, b11);
        v2_pinned::mma_int4(cB[1], a00, a01, b12, b13);
        v2_pinned::mma_int4(cB[2], a02, a03, b10, b11);
        v2_pinned::mma_int4(cB[3], a02, a03, b12, b13);
        v2_pinned::mma_int4(cB[4], a04, a05, b10, b11);
        v2_pinned::mma_int4(cB[5], a04, a05, b12, b13);
        v2_pinned::mma_int4(cB[6], a06, a07, b10, b11);
        v2_pinned::mma_int4(cB[7], a06, a07, b12, b13);
        v2_pinned::mma_int4(cC[0], a10, a11, b00, b01);
        v2_pinned::mma_int4(cC[1], a10, a11, b02, b03);
        v2_pinned::mma_int4(cC[2], a12, a13, b00, b01);
        v2_pinned::mma_int4(cC[3], a12, a13, b02, b03);
        v2_pinned::mma_int4(cC[4], a14, a15, b00, b01);
        v2_pinned::mma_int4(cC[5], a14, a15, b02, b03);
        v2_pinned::mma_int4(cC[6], a16, a17, b00, b01);
        v2_pinned::mma_int4(cC[7], a16, a17, b02, b03);
        v2_pinned::mma_int4(cD[0], a10, a11, b10, b11);
        v2_pinned::mma_int4(cD[1], a10, a11, b12, b13);
        v2_pinned::mma_int4(cD[2], a12, a13, b10, b11);
        v2_pinned::mma_int4(cD[3], a12, a13, b12, b13);
        v2_pinned::mma_int4(cD[4], a14, a15, b10, b11);
        v2_pinned::mma_int4(cD[5], a14, a15, b12, b13);
        v2_pinned::mma_int4(cD[6], a16, a17, b10, b11);
        v2_pinned::mma_int4(cD[7], a16, a17, b12, b13);
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        C[tid*32+i   ] = *(float4*)cA[i];
        C[tid*32+i+ 8] = *(float4*)cB[i];
        C[tid*32+i+16] = *(float4*)cC[i];
        C[tid*32+i+24] = *(float4*)cD[i];
    }
}


// R3: per-warp pinned + LDS storage + ds_read intrinsic
extern "C" __global__ __launch_bounds__(64, 1)
void __probe_v2_lds_pinned(
        const int4* __restrict__ A_g,
        const int4* __restrict__ B_g,
        float4* __restrict__ C) {
    __shared__ int4 A_lds[128];  // 128 int4 = 2KB
    __shared__ int4 B_lds[64];
    const int tid = threadIdx.x;
    // cooperative load gmem→smem (just bulk via lane writes — each lane loads 2 A int4 + 1 B int4)
    A_lds[tid*2  ] = A_g[tid*2  ];
    A_lds[tid*2+1] = A_g[tid*2+1];
    B_lds[tid    ] = B_g[tid    ];
    __syncthreads();

    register int4 a_lo asm("v0");
    register int4 a_hi asm("v4");
    register int4 b_lo asm("v8");
    register int4 b_hi asm("v12");
    // ds_read from LDS into pinned slots
    a_lo = A_lds[tid*2  ];
    a_hi = A_lds[tid*2+1];
    b_lo = B_lds[tid    ];
    b_hi = B_lds[tid + 32];

    float2 acc[2] = {{0.f, 0.f}, {0.f, 0.f}};
    v2_pinned::mma_int4(acc, a_lo, a_hi, b_lo, b_hi);
    C[tid] = *(float4*)acc;
}


// R2: per-warp 128M×64N (full v1 per-warp area), 4 acc × 8 mma = 32 mma per iter
extern "C" __global__ __launch_bounds__(64, 1)
void __probe_v2_4acc_pinned(
        const int4* __restrict__ A,   // 128 rows × 128 K = 256 int4 = 16 int4/lane × 16
        const int4* __restrict__ B,   // 64 cols × 128 K = 128 int4 = 8 int4/lane × 16
        float4* __restrict__ C) {     // 128×64 = 8192 floats = 32 float4/lane × 64
    const int tid = threadIdx.x;
    // A pinned: 16 int4 (2 m_chunks × 8 int4)
    register int4 a00 asm("v0");  register int4 a01 asm("v4");
    register int4 a02 asm("v8");  register int4 a03 asm("v12");
    register int4 a04 asm("v16"); register int4 a05 asm("v20");
    register int4 a06 asm("v24"); register int4 a07 asm("v28");
    register int4 a10 asm("v32"); register int4 a11 asm("v36");
    register int4 a12 asm("v40"); register int4 a13 asm("v44");
    register int4 a14 asm("v48"); register int4 a15 asm("v52");
    register int4 a16 asm("v56"); register int4 a17 asm("v60");
    // B pinned: 8 int4 (2 n_chunks × 4 int4)
    register int4 b00 asm("v64"); register int4 b01 asm("v68");
    register int4 b02 asm("v72"); register int4 b03 asm("v76");
    register int4 b10 asm("v80"); register int4 b11 asm("v84");
    register int4 b12 asm("v88"); register int4 b13 asm("v92");

    a00 = A[tid*16+ 0]; a01 = A[tid*16+ 1]; a02 = A[tid*16+ 2]; a03 = A[tid*16+ 3];
    a04 = A[tid*16+ 4]; a05 = A[tid*16+ 5]; a06 = A[tid*16+ 6]; a07 = A[tid*16+ 7];
    a10 = A[tid*16+ 8]; a11 = A[tid*16+ 9]; a12 = A[tid*16+10]; a13 = A[tid*16+11];
    a14 = A[tid*16+12]; a15 = A[tid*16+13]; a16 = A[tid*16+14]; a17 = A[tid*16+15];
    b00 = B[tid*8+0]; b01 = B[tid*8+1]; b02 = B[tid*8+2]; b03 = B[tid*8+3];
    b10 = B[tid*8+4]; b11 = B[tid*8+5]; b12 = B[tid*8+6]; b13 = B[tid*8+7];

    // 4 acc, each 8 tiles, each tile 4 floats = 32 AGPR/lane per acc; 128 AGPR total
    float2 cA[8][2], cB[8][2], cC[8][2], cD[8][2];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        cA[i][0]={0,0}; cA[i][1]={0,0}; cB[i][0]={0,0}; cB[i][1]={0,0};
        cC[i][0]={0,0}; cC[i][1]={0,0}; cD[i][0]={0,0}; cD[i][1]={0,0};
    }

    // cA = m_chunk 0 × n_chunk 0, 8 tiles (4 m × 2 n)
    v2_pinned::mma_int4(cA[0], a00, a01, b00, b01);
    v2_pinned::mma_int4(cA[1], a00, a01, b02, b03);
    v2_pinned::mma_int4(cA[2], a02, a03, b00, b01);
    v2_pinned::mma_int4(cA[3], a02, a03, b02, b03);
    v2_pinned::mma_int4(cA[4], a04, a05, b00, b01);
    v2_pinned::mma_int4(cA[5], a04, a05, b02, b03);
    v2_pinned::mma_int4(cA[6], a06, a07, b00, b01);
    v2_pinned::mma_int4(cA[7], a06, a07, b02, b03);
    // cB = m_chunk 0 × n_chunk 1
    v2_pinned::mma_int4(cB[0], a00, a01, b10, b11);
    v2_pinned::mma_int4(cB[1], a00, a01, b12, b13);
    v2_pinned::mma_int4(cB[2], a02, a03, b10, b11);
    v2_pinned::mma_int4(cB[3], a02, a03, b12, b13);
    v2_pinned::mma_int4(cB[4], a04, a05, b10, b11);
    v2_pinned::mma_int4(cB[5], a04, a05, b12, b13);
    v2_pinned::mma_int4(cB[6], a06, a07, b10, b11);
    v2_pinned::mma_int4(cB[7], a06, a07, b12, b13);
    // cC = m_chunk 1 × n_chunk 0
    v2_pinned::mma_int4(cC[0], a10, a11, b00, b01);
    v2_pinned::mma_int4(cC[1], a10, a11, b02, b03);
    v2_pinned::mma_int4(cC[2], a12, a13, b00, b01);
    v2_pinned::mma_int4(cC[3], a12, a13, b02, b03);
    v2_pinned::mma_int4(cC[4], a14, a15, b00, b01);
    v2_pinned::mma_int4(cC[5], a14, a15, b02, b03);
    v2_pinned::mma_int4(cC[6], a16, a17, b00, b01);
    v2_pinned::mma_int4(cC[7], a16, a17, b02, b03);
    // cD = m_chunk 1 × n_chunk 1
    v2_pinned::mma_int4(cD[0], a10, a11, b10, b11);
    v2_pinned::mma_int4(cD[1], a10, a11, b12, b13);
    v2_pinned::mma_int4(cD[2], a12, a13, b10, b11);
    v2_pinned::mma_int4(cD[3], a12, a13, b12, b13);
    v2_pinned::mma_int4(cD[4], a14, a15, b10, b11);
    v2_pinned::mma_int4(cD[5], a14, a15, b12, b13);
    v2_pinned::mma_int4(cD[6], a16, a17, b10, b11);
    v2_pinned::mma_int4(cD[7], a16, a17, b12, b13);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        C[tid*32 + i     ] = *(float4*)cA[i];
        C[tid*32 + i + 8 ] = *(float4*)cB[i];
        C[tid*32 + i + 16] = *(float4*)cC[i];
        C[tid*32 + i + 24] = *(float4*)cD[i];
    }
}


// R1: per-warp 64×32 output, 8 mma per K iter, 8 acc tiles (32 AGPR total)
extern "C" __global__ __launch_bounds__(64, 1)
void __probe_v2_per_warp_pinned(
        const int4* __restrict__ A,   // 64 rows × 128 K = 128 int4
        const int4* __restrict__ B,   // 32 cols × 128 K =  64 int4
        float4* __restrict__ C) {     // 64×32 = 2048 floats = 512 float4
    const int tid = threadIdx.x;
    // Pinned A: 8 int4 per lane = 32 V dwords (4 m-tiles × 8 dwords each)
    register int4 a0 asm("v0");  register int4 a1 asm("v4");
    register int4 a2 asm("v8");  register int4 a3 asm("v12");
    register int4 a4 asm("v16"); register int4 a5 asm("v20");
    register int4 a6 asm("v24"); register int4 a7 asm("v28");
    // Pinned B: 4 int4 per lane = 16 V dwords (2 n-tiles × 8 dwords each)
    register int4 b0 asm("v32"); register int4 b1 asm("v36");
    register int4 b2 asm("v40"); register int4 b3 asm("v44");

    a0 = A[tid * 8 + 0]; a1 = A[tid * 8 + 1];
    a2 = A[tid * 8 + 2]; a3 = A[tid * 8 + 3];
    a4 = A[tid * 8 + 4]; a5 = A[tid * 8 + 5];
    a6 = A[tid * 8 + 6]; a7 = A[tid * 8 + 7];
    b0 = B[tid * 4 + 0]; b1 = B[tid * 4 + 1];
    b2 = B[tid * 4 + 2]; b3 = B[tid * 4 + 3];

    // 8 acc tiles, each float2[2] = 4 floats/lane = 4 AGPR/lane each
    float2 acc[8][2];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { acc[i][0] = {0,0}; acc[i][1] = {0,0}; }

    // m_tile 0..3, n_tile 0..1 -- 8 mma calls
    // Tile (m=0, n=0): a0,a1 × b0,b1
    v2_pinned::mma_int4(acc[0], a0, a1, b0, b1);
    // Tile (m=0, n=1): a0,a1 × b2,b3
    v2_pinned::mma_int4(acc[1], a0, a1, b2, b3);
    // Tile (m=1, n=0): a2,a3 × b0,b1
    v2_pinned::mma_int4(acc[2], a2, a3, b0, b1);
    v2_pinned::mma_int4(acc[3], a2, a3, b2, b3);
    v2_pinned::mma_int4(acc[4], a4, a5, b0, b1);
    v2_pinned::mma_int4(acc[5], a4, a5, b2, b3);
    v2_pinned::mma_int4(acc[6], a6, a7, b0, b1);
    v2_pinned::mma_int4(acc[7], a6, a7, b2, b3);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        C[(tid * 8) + i] = *(float4*)acc[i];
    }
}


extern "C" __global__ __launch_bounds__(64, 1)
void __probe_v2_minimal_pinned_gemm(
        const int4* __restrict__ A,   // packed fp8: 16 rows × 128 K = 32 int4 (2 per thread)
        const int4* __restrict__ B,   // 16 cols × 128 K = 32 int4
        float4* __restrict__ C) {     // 16x16 = 256 floats = 64 float4 (1 per thread)
    const int tid = threadIdx.x;
    register int4 a_lo asm("v0");
    register int4 a_hi asm("v4");
    register int4 b_lo asm("v8");
    register int4 b_hi asm("v12");
    a_lo = A[tid * 2 + 0];
    a_hi = A[tid * 2 + 1];
    b_lo = B[tid * 2 + 0];
    b_hi = B[tid * 2 + 1];

    typedef __attribute__((__vector_size__(8 * sizeof(int))))    int   intx8_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;
    intx8_t A_vec = {a_lo.x, a_lo.y, a_lo.z, a_lo.w, a_hi.x, a_hi.y, a_hi.z, a_hi.w};
    intx8_t B_vec = {b_lo.x, b_lo.y, b_lo.z, b_lo.w, b_hi.x, b_hi.y, b_hi.z, b_hi.w};

    float2 acc[2] = {{0.f, 0.f}, {0.f, 0.f}};
    asm volatile(
        "v_mfma_f32_16x16x128_f8f6f4 %0, %1, %2, %0"
        : "+a"(*(floatx4_t*)acc)
        : "v"(A_vec), "v"(B_vec));

    C[tid] = *(float4*)acc;
}


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
// R12: wrapper using mma_int4_vacc (builtin path → compiler chooses VGPR for D)
template<bool USE_AGPR>
__device__ __forceinline__ void rcr_mma_v2_vacc_wrapper(
        rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
        A_row_reg& a, B_row_reg& b) {
    constexpr int M_TILES = RBM / 16;
    constexpr int N_TILES = RBN / 16;
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        #pragma unroll
        for (int n = 0; n < N_TILES; ++n) {
            int4* a_p = reinterpret_cast<int4*>(&a.tiles[m][0].data[0]);
            int4* b_p = reinterpret_cast<int4*>(&b.tiles[n][0].data[0]);
            v2_pinned::mma_int4_vacc(
                *reinterpret_cast<float2(*)[2]>(&acc.tiles[m][n].data[0]),
                a_p[0], a_p[1], b_p[0], b_p[1]);
        }
    }
}

// R6: mma wrapper using mma_int4 (raw int4 pinned pattern) instead of
// HK's rcr_mma_agpr_t. Reinterprets HK fragment storage as int4 in-register.
template<bool USE_AGPR>
__device__ __forceinline__ void rcr_mma_v2_wrapper(
        rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
        A_row_reg& a, B_row_reg& b) {
    constexpr int M_TILES = RBM / 16;
    constexpr int N_TILES = RBN / 16;
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        #pragma unroll
        for (int n = 0; n < N_TILES; ++n) {
            int4* a_p = reinterpret_cast<int4*>(&a.tiles[m][0].data[0]);
            int4* b_p = reinterpret_cast<int4*>(&b.tiles[n][0].data[0]);
            v2_pinned::mma_int4(
                *reinterpret_cast<float2(*)[2]>(&acc.tiles[m][n].data[0]),
                a_p[0], a_p[1], b_p[0], b_p[1]);
        }
    }
}

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
    A_row_reg a;
    B_row_reg b0, b1;
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
            __builtin_amdgcn_s_setprio(1); rcr_mma_v2_wrapper<!FUSED_KTAIL>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_v2_wrapper<!FUSED_KTAIL>(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], a_gl_g, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_v2_wrapper<!FUSED_KTAIL>(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
            TK_WAIT_VMCNT(RCR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_setprio(1); rcr_mma_v2_vacc_wrapper<!FUSED_KTAIL>(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // Epilog 1
        {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], a_gl_g, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_v2_wrapper<!FUSED_KTAIL>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_v2_wrapper<!FUSED_KTAIL>(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(RCR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_v2_wrapper<!FUSED_KTAIL>(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, b_tile(toc, 0), wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_v2_vacc_wrapper<!FUSED_KTAIL>(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        // Epilog 2
        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_v2_wrapper<!FUSED_KTAIL>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rcr_mma_v2_wrapper<!FUSED_KTAIL>(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1);
            rcr_mma_v2_wrapper<!FUSED_KTAIL>(cC, a, b0);
            rcr_mma_v2_vacc_wrapper<!FUSED_KTAIL>(cD, a, b1);
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
__attribute__((amdgpu_waves_per_eu(1, 1)))
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
