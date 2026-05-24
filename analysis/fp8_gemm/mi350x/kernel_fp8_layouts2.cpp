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

// R155-real
#undef RCR_EPILOGUE_VMCNT
#define RCR_EPILOGUE_VMCNT 2































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

// =============================================================================
// P1.2 STEP 1 (R159) — 32x32x64 mfma + native 32x64 frag types
//
// ISA root-cause finding: with 16x16x128 + 4-acc layout, compiler emits
//   ~207 scratch_load + 199 v_accvgpr_write + 103 v_accvgpr_read per
//   kernel call = ~2450 cycle overhead per call (= 18% of total runtime).
// Foundation R57 probe confirmed 32x32x64 + 4-acc at 8-warp WG achieves
//   spill=0 + A=0 (no AGPR shuffle).
// This step lands the production-grade wrapper that operates on native
//   32x64 (M-rows × K) fragments. Per acc (64×32 per-warp area) becomes
//   2 M-tiles × 1 N-tile × 2 K-halves = 4 mma calls (vs 8 with 16x16).
// =============================================================================

// 32×64 fragment storage (row-major for A, col-major effective for B in RCR).
// 32 rows × 64 cols of fp8 = 2048 bytes/tile = 32 fp8e4m3_4 per lane
// distributed across 64-lane warp via mfma_32x32x64 operand convention:
// 8 fp8e4m3_4 per lane (= 8 int32 dwords).
using A_row_reg_32 = rt_fp8e4m3<32, 64, row_l, rt_32x64_s>;
using B_row_reg_32 = rt_fp8e4m3<32, 64, row_l, rt_32x64_s>;

// 32×32 acc tile = 16 floats/lane = float2[8]
using acc32_t = rt_fl<32, 32, col_l, rt_32x32_s>;

// Native 32x32 wrapper: accepts HK 32x64 frag types, emits 1 mfma_32x32x64.
// Per-acc 64x32 area requires 2 M-tiles × 2 K-halves = 4 calls.
__device__ __forceinline__ static void mma_32_frag(
        float2 (&D)[8],
        const A_row_reg_32& a, int m_tile, int k_half,
        const B_row_reg_32& b, int n_tile) {
    // tiles[m_tile][0] gives the 32x64 sub-tile (HK rt_32x64_s stores K=64
    // per tile element; for K=128 we have 2 tiles per row → m_tile selects
    // M-tile, [0] selects K-slot. With BK=128 we have 2 sub-tiles per A_row_reg
    // — actually A_row_reg_32 is templated with K=64 fixed; we use two
    // separate fragments to cover K=128.)
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;
    const int4* a_p = reinterpret_cast<const int4*>(&a.tiles[m_tile][0].data[k_half * 4]);
    const int4* b_p = reinterpret_cast<const int4*>(&b.tiles[n_tile][0].data[k_half * 4]);
    intx8_t A_vec = {a_p[0].x, a_p[0].y, a_p[0].z, a_p[0].w,
                     a_p[1].x, a_p[1].y, a_p[1].z, a_p[1].w};
    intx8_t B_vec = {b_p[0].x, b_p[0].y, b_p[0].z, b_p[0].w,
                     b_p[1].x, b_p[1].y, b_p[1].z, b_p[1].w};
    asm volatile(
        "v_mfma_f32_32x32x64_f8f6f4 %0, %1, %2, %0"
        : "+a"(*(floatx16_t*)D)
        : "v"(A_vec), "v"(B_vec));
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

// R473 (P1-RRR): vacc wrapper for RRR — same as RCR vacc wrapper but uses
// B_col_reg (RRR's col-layout B fragment) with b.tiles[0][n] instead of
// b.tiles[n][0]. Mirrors R167 RCR breakthrough (all 4 acc as vacc).
template<bool USE_AGPR>
__device__ __forceinline__ void rrr_mma_v2_vacc_wrapper(
        rt_fl<RBM, RBN, col_l, rt_16x16_s>& acc,
        A_row_reg& a, B_col_reg& b) {
    constexpr int M_TILES = RBM / 16;
    constexpr int N_TILES = RBN / 16;
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        #pragma unroll
        for (int n = 0; n < N_TILES; ++n) {
            int4* a_p = reinterpret_cast<int4*>(&a.tiles[m][0].data[0]);
            // RRR layout: b.tiles[k][n] with K width=1 (BK=128 = 1 base tile in K)
            int4* b_p = reinterpret_cast<int4*>(&b.tiles[0][n].data[0]);
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


// R52/R53 — P1.2 multi-session building block: 32×32×64 mfma direct wrapper.
namespace v2_pinned {
__device__ __forceinline__ static void mma_32_int4(
        float2 (&D)[8],
        int4 A_lo, int4 A_hi,
        int4 B_lo, int4 B_hi) {
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;
    intx8_t A = {A_lo.x, A_lo.y, A_lo.z, A_lo.w, A_hi.x, A_hi.y, A_hi.z, A_hi.w};
    intx8_t B = {B_lo.x, B_lo.y, B_lo.z, B_lo.w, B_hi.x, B_hi.y, B_hi.z, B_hi.w};
    asm volatile(
        "v_mfma_f32_32x32x64_f8f6f4 %0, %1, %2, %0"
        : "+a"(*(floatx16_t*)D)
        : "v"(A), "v"(B));
}
}  // namespace v2_pinned

// R53: isolation probe. Expect spill=0 with V<128 A=64 (1 mfma acc).
extern "C" __global__ __launch_bounds__(64, 1)
void __probe_v2_mma_32_isolated(
        const int4* __restrict__ A,
        const int4* __restrict__ B,
        float* __restrict__ C) {
    const int tid = threadIdx.x;
    int4 a_lo = A[tid * 2 + 0];
    int4 a_hi = A[tid * 2 + 1];
    int4 b_lo = B[tid * 2 + 0];
    int4 b_hi = B[tid * 2 + 1];
    float2 acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { acc[i] = {0.f, 0.f}; }
    v2_pinned::mma_32_int4(acc, a_lo, a_hi, b_lo, b_hi);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        C[tid * 16 + i * 2 + 0] = acc[i].x;
        C[tid * 16 + i * 2 + 1] = acc[i].y;
    }
}

// R58: 8-warp + 4-acc + 22 iter + LDS double-buffer prefetch — closest
// approximation of real production body register pressure.
extern "C" __global__ __launch_bounds__(512, 1)
void __probe_v2_mma_32_8w_4acc_lds_k22(
        const int4* __restrict__ A_gl,  // [22 * 2048] int4
        const int4* __restrict__ B_gl,  // [22 * 2048] int4
        float* __restrict__ C) {
    constexpr int LDS_TILE_INT4 = 1024;  // 16 KB per tile
    __shared__ int4 As[2][LDS_TILE_INT4];
    __shared__ int4 Bs[2][LDS_TILE_INT4];
    const int tid = threadIdx.x;
    const int wid = tid / 64;
    const int lid = tid % 64;
    float2 acc0[8], acc1[8], acc2[8], acc3[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc0[i] = {0.f,0.f}; acc1[i] = {0.f,0.f};
        acc2[i] = {0.f,0.f}; acc3[i] = {0.f,0.f};
    }
    // Initial prefetch tic=0
    if (tid < LDS_TILE_INT4) {
        As[0][tid] = A_gl[tid];
        Bs[0][tid] = B_gl[tid];
    }
    __builtin_amdgcn_s_barrier();
    int tic = 0;
    #pragma unroll 1
    for (int k = 0; k < 22; ++k, tic ^= 1) {
        int toc = tic ^ 1;
        // Prefetch next iter
        if (k + 1 < 22 && tid < LDS_TILE_INT4) {
            As[toc][tid] = A_gl[(k + 1) * LDS_TILE_INT4 * 2 + tid];
            Bs[toc][tid] = B_gl[(k + 1) * LDS_TILE_INT4 * 2 + tid];
        }
        // Read current
        int4 a0_lo = As[tic][wid * 256 + lid * 2 + 0];
        int4 a0_hi = As[tic][wid * 256 + lid * 2 + 1];
        int4 a1_lo = As[tic][wid * 256 + lid * 2 + 128];
        int4 a1_hi = As[tic][wid * 256 + lid * 2 + 129];
        int4 b0_lo = Bs[tic][wid * 256 + lid * 2 + 0];
        int4 b0_hi = Bs[tic][wid * 256 + lid * 2 + 1];
        int4 b1_lo = Bs[tic][wid * 256 + lid * 2 + 128];
        int4 b1_hi = Bs[tic][wid * 256 + lid * 2 + 129];
        __builtin_amdgcn_s_barrier();
        v2_pinned::mma_32_int4(acc0, a0_lo, a0_hi, b0_lo, b0_hi);
        v2_pinned::mma_32_int4(acc1, a0_lo, a0_hi, b1_lo, b1_hi);
        v2_pinned::mma_32_int4(acc2, a1_lo, a1_hi, b0_lo, b0_hi);
        v2_pinned::mma_32_int4(acc3, a1_lo, a1_hi, b1_lo, b1_hi);
        __builtin_amdgcn_s_barrier();
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        C[tid * 64 + i*2 + 0]  = acc0[i].x; C[tid * 64 + i*2 + 1]  = acc0[i].y;
        C[tid * 64 + i*2 + 16] = acc1[i].x; C[tid * 64 + i*2 + 17] = acc1[i].y;
        C[tid * 64 + i*2 + 32] = acc2[i].x; C[tid * 64 + i*2 + 33] = acc2[i].y;
        C[tid * 64 + i*2 + 48] = acc3[i].x; C[tid * 64 + i*2 + 49] = acc3[i].y;
    }
}

// R57: 8-warp WG probe — production thread topology + 4 acc + 22 iter.
// 8 warps × 64 threads = 512 threads/WG, 1 wave/SIMD per launch_bounds(_,1).
extern "C" __global__ __launch_bounds__(512, 1)
void __probe_v2_mma_32_8w_4acc_k22(
        const int4* __restrict__ A,
        const int4* __restrict__ B,
        float* __restrict__ C) {
    const int tid = threadIdx.x;
    const int wid = tid / 64;
    const int lid = tid % 64;
    float2 acc0[8], acc1[8], acc2[8], acc3[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc0[i] = {0.f,0.f}; acc1[i] = {0.f,0.f};
        acc2[i] = {0.f,0.f}; acc3[i] = {0.f,0.f};
    }
    #pragma unroll 1
    for (int k = 0; k < 22; ++k) {
        // Each warp reads its own slice via wid offset
        int4 a0_lo = A[k * 2048 + wid * 256 + lid * 2 + 0];
        int4 a0_hi = A[k * 2048 + wid * 256 + lid * 2 + 1];
        int4 a1_lo = A[k * 2048 + wid * 256 + lid * 2 + 128];
        int4 a1_hi = A[k * 2048 + wid * 256 + lid * 2 + 129];
        int4 b0_lo = B[k * 2048 + wid * 256 + lid * 2 + 0];
        int4 b0_hi = B[k * 2048 + wid * 256 + lid * 2 + 1];
        int4 b1_lo = B[k * 2048 + wid * 256 + lid * 2 + 128];
        int4 b1_hi = B[k * 2048 + wid * 256 + lid * 2 + 129];
        v2_pinned::mma_32_int4(acc0, a0_lo, a0_hi, b0_lo, b0_hi);
        v2_pinned::mma_32_int4(acc1, a0_lo, a0_hi, b1_lo, b1_hi);
        v2_pinned::mma_32_int4(acc2, a1_lo, a1_hi, b0_lo, b0_hi);
        v2_pinned::mma_32_int4(acc3, a1_lo, a1_hi, b1_lo, b1_hi);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        C[tid * 64 + i*2 + 0]  = acc0[i].x; C[tid * 64 + i*2 + 1]  = acc0[i].y;
        C[tid * 64 + i*2 + 16] = acc1[i].x; C[tid * 64 + i*2 + 17] = acc1[i].y;
        C[tid * 64 + i*2 + 32] = acc2[i].x; C[tid * 64 + i*2 + 33] = acc2[i].y;
        C[tid * 64 + i*2 + 48] = acc3[i].x; C[tid * 64 + i*2 + 49] = acc3[i].y;
    }
}

// R56: 4-acc K-chain probe — production acc count (cA/cB/cC/cD).
extern "C" __global__ __launch_bounds__(64, 1)
void __probe_v2_mma_32_4acc_k22(
        const int4* __restrict__ A,
        const int4* __restrict__ B,
        float* __restrict__ C) {
    const int tid = threadIdx.x;
    float2 acc0[8], acc1[8], acc2[8], acc3[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc0[i] = {0.f,0.f}; acc1[i] = {0.f,0.f};
        acc2[i] = {0.f,0.f}; acc3[i] = {0.f,0.f};
    }
    #pragma unroll 1
    for (int k = 0; k < 22; ++k) {
        int4 a0_lo = A[k * 256 + tid * 2 + 0];
        int4 a0_hi = A[k * 256 + tid * 2 + 1];
        int4 a1_lo = A[k * 256 + tid * 2 + 128];
        int4 a1_hi = A[k * 256 + tid * 2 + 129];
        int4 b0_lo = B[k * 256 + tid * 2 + 0];
        int4 b0_hi = B[k * 256 + tid * 2 + 1];
        int4 b1_lo = B[k * 256 + tid * 2 + 128];
        int4 b1_hi = B[k * 256 + tid * 2 + 129];
        v2_pinned::mma_32_int4(acc0, a0_lo, a0_hi, b0_lo, b0_hi);
        v2_pinned::mma_32_int4(acc1, a0_lo, a0_hi, b1_lo, b1_hi);
        v2_pinned::mma_32_int4(acc2, a1_lo, a1_hi, b0_lo, b0_hi);
        v2_pinned::mma_32_int4(acc3, a1_lo, a1_hi, b1_lo, b1_hi);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        C[tid * 64 + i*2 + 0]  = acc0[i].x; C[tid * 64 + i*2 + 1]  = acc0[i].y;
        C[tid * 64 + i*2 + 16] = acc1[i].x; C[tid * 64 + i*2 + 17] = acc1[i].y;
        C[tid * 64 + i*2 + 32] = acc2[i].x; C[tid * 64 + i*2 + 33] = acc2[i].y;
        C[tid * 64 + i*2 + 48] = acc3[i].x; C[tid * 64 + i*2 + 49] = acc3[i].y;
    }
}

// R55: 2-acc K-chain probe — does spill stay 0 with 2 acc × 22 iter?
extern "C" __global__ __launch_bounds__(64, 1)
void __probe_v2_mma_32_2acc_k22(
        const int4* __restrict__ A,
        const int4* __restrict__ B,
        float* __restrict__ C) {
    const int tid = threadIdx.x;
    float2 acc0[8], acc1[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { acc0[i] = {0.f, 0.f}; acc1[i] = {0.f, 0.f}; }
    #pragma unroll 1
    for (int k = 0; k < 22; ++k) {
        int4 a_lo = A[k * 128 + tid * 2 + 0];
        int4 a_hi = A[k * 128 + tid * 2 + 1];
        int4 b0_lo = B[k * 256 + tid * 2 + 0];
        int4 b0_hi = B[k * 256 + tid * 2 + 1];
        int4 b1_lo = B[k * 256 + tid * 2 + 128];
        int4 b1_hi = B[k * 256 + tid * 2 + 129];
        v2_pinned::mma_32_int4(acc0, a_lo, a_hi, b0_lo, b0_hi);
        v2_pinned::mma_32_int4(acc1, a_lo, a_hi, b1_lo, b1_hi);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        C[tid * 32 + i * 2 + 0]  = acc0[i].x;
        C[tid * 32 + i * 2 + 1]  = acc0[i].y;
        C[tid * 32 + i * 2 + 16] = acc1[i].x;
        C[tid * 32 + i * 2 + 17] = acc1[i].y;
    }
}

// R163-real: probe HK load(rt_32x64_s, st_32x64) compile
extern "C" __global__ __launch_bounds__(64, 1)
void __probe_v2_st_32x64_load(float* __restrict__ C) {
    using ST_A = st_fp8e4m3<32, 64, st_32x64_s>;
    __shared__ ST_A As;
    using RT_A = rt_fp8e4m3<32, 64, row_l, rt_32x64_s>;
    RT_A a;
    const int tid = threadIdx.x;
    int4* Ap = reinterpret_cast<int4*>(&As);
    if (tid < 32) {
        Ap[tid * 4 + 0] = make_int4(tid, tid, tid, tid);
        Ap[tid * 4 + 1] = make_int4(tid, tid, tid, tid);
    }
    __builtin_amdgcn_s_barrier();
    load(a, As);
    int v = reinterpret_cast<int*>(&a.tiles[0][0].data[0])[0];
    C[tid] = (float)v;
}

// R187-real: probe HK mma_ABt with rt_32x32 acc + rt_32x64 frag
extern "C" __global__ __launch_bounds__(64, 1)
void __probe_v2_mma_ABt_32x32(float* __restrict__ C) {
    using ST_A = st_fp8e4m3<32, 64, st_32x64_s>;
    using ST_B = st_fp8e4m3<32, 64, st_32x64_s>;
    __shared__ ST_A As;
    __shared__ ST_B Bs;
    rt_fp8e4m3<32, 64, row_l, rt_32x64_s> a;
    rt_fp8e4m3<32, 64, row_l, rt_32x64_s> b;
    rt_fl<32, 32, col_l, rt_32x32_s> acc;
    zero(acc);
    const int tid = threadIdx.x;
    int4* Ap = reinterpret_cast<int4*>(&As);
    int4* Bp = reinterpret_cast<int4*>(&Bs);
    if (tid < 32) {
        Ap[tid * 4 + 0] = make_int4(tid, 1, 2, 3);
        Bp[tid * 4 + 0] = make_int4(tid, 1, 2, 3);
    }
    __builtin_amdgcn_s_barrier();
    load(a, As);
    load(b, Bs);
    __builtin_amdgcn_s_barrier();
    mma_ABt(acc, a, b, acc);
    C[tid] = acc.tiles[0][0].data[0].x;
}

// R165-real: production-shape ST composed of st_32x64 subtiles
extern "C" __global__ __launch_bounds__(64, 1)
void __probe_v2_st_128x128_32x64subtile(float* __restrict__ C) {
    using ST_BIG = st_fp8e4m3<128, 128, st_32x64_s>;
    __shared__ ST_BIG As;
    using RT_A = rt_fp8e4m3<64, 128, row_l, rt_32x64_s>;  // per-warp 64M × 128K
    RT_A a;
    const int tid = threadIdx.x;
    int4* Ap = reinterpret_cast<int4*>(&As);
    if (tid < 64) {
        for (int i = 0; i < 16; ++i) Ap[tid * 16 + i] = make_int4(tid, i, i, tid);
    }
    __builtin_amdgcn_s_barrier();
    auto sub = subtile_inplace<64, 128>(As, {0, 0});
    load(a, sub);
    int v = reinterpret_cast<int*>(&a.tiles[0][0].data[0])[0];
    C[tid] = (float)v;
}

// R54: K-chain probe — accumulate over 22 K-iter validating spill stays 0
// as K-loop scale grows (production gpt_oss has ki=22).
extern "C" __global__ __launch_bounds__(64, 1)
void __probe_v2_mma_32_kchain_22(
        const int4* __restrict__ A,
        const int4* __restrict__ B,
        float* __restrict__ C) {
    const int tid = threadIdx.x;
    float2 acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { acc[i] = {0.f, 0.f}; }
    #pragma unroll 1
    for (int k = 0; k < 22; ++k) {
        int4 a_lo = A[k * 128 + tid * 2 + 0];
        int4 a_hi = A[k * 128 + tid * 2 + 1];
        int4 b_lo = B[k * 128 + tid * 2 + 0];
        int4 b_hi = B[k * 128 + tid * 2 + 1];
        v2_pinned::mma_32_int4(acc, a_lo, a_hi, b_lo, b_hi);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        C[tid * 16 + i * 2 + 0] = acc[i].x;
        C[tid * 16 + i * 2 + 1] = acc[i].y;
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
    // R456: per-shape chunk_size auto-pick if caller passes 0.
    // Heuristic from R452 sweep: large-K large-N (dsv3 up) benefits from cs=48,
    // others stay at cs=32. Trigger: K*N > 1.5M heuristic.
    int chunk_size_eff;
    if (g.chunk_size > 0) {
        chunk_size_eff = g.chunk_size;
    } else {
        const int n_cols = static_cast<int>(g.c.cols());
        const int k_cols = static_cast<int>(g.a.cols());
        chunk_size_eff = (k_cols >= 4096 && n_cols >= 4096) ? 48 : 32;
    }
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
            rcr_mma_v2_vacc_wrapper<!FUSED_KTAIL>(cA, a, b0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 0), g.b, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            rcr_mma_v2_vacc_wrapper<!FUSED_KTAIL>(cB, a, b1);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], a_gl_g, a_co(br*2, k+2), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            rcr_mma_v2_vacc_wrapper<!FUSED_KTAIL>(cC, a, b0);
            __builtin_amdgcn_s_barrier();

            rcr_8w_load_hoist<_NUM_THREADS>(b_tile(tic, 1), g.b, b_co(bc*2+1, k+2), soB);
            TK_WAIT_VMCNT(RCR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            rcr_mma_v2_vacc_wrapper<!FUSED_KTAIL>(cD, a, b1);
            __builtin_amdgcn_s_barrier();
        }

        // Epilog 1
        {
            load_b(b0, b_tile(tic, 0), wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], a_gl_g, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            rcr_mma_v2_vacc_wrapper<!FUSED_KTAIL>(cA, a, b0);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            rcr_mma_v2_vacc_wrapper<!FUSED_KTAIL>(cB, a, b1);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(RCR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            rcr_mma_v2_vacc_wrapper<!FUSED_KTAIL>(cC, a, b0);
            __builtin_amdgcn_s_barrier();

            load_b(b0, b_tile(toc, 0), wn);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            rcr_mma_v2_vacc_wrapper<!FUSED_KTAIL>(cD, a, b1);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        // Epilog 2
        {
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            rcr_mma_v2_vacc_wrapper<!FUSED_KTAIL>(cA, a, b0);
            __builtin_amdgcn_s_barrier();

            load_b(b1, b_tile(tic, 1), wn);
            __builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();
            MAYBE_DRAIN_LGKM();
            rcr_mma_v2_vacc_wrapper<!FUSED_KTAIL>(cB, a, b1);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1);
            rcr_mma_v2_vacc_wrapper<!FUSED_KTAIL>(cC, a, b0);
            rcr_mma_v2_vacc_wrapper<!FUSED_KTAIL>(cD, a, b1);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // R22: FUSED_KTAIL block — K_rem=64 tail. Mirrors v1 FUSED block
        // structure but routes mma through v2 wrappers (cA vacc, B/C/D AGPR).
        if constexpr (FUSED_KTAIL) {
            if (g.fast_k < g.k) {
                const int laneid     = kittens::laneid();
                const int row_lane   = laneid % 16;
                const int k_lane_byte = (laneid / 16) * 32;
                const bool both_valid = (laneid < 32);
                constexpr uint32_t SENTINEL = 0xFFFF0000u;
                const fp8e4m3* a_base_ptr = (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
                const fp8e4m3* b_base_ptr = (const fp8e4m3*)&g.b[{0, 0, 0, 0}];
                const int a_row_stride_bytes_kt = static_cast<int>(g.a.template stride<2>()) * sizeof(*g.a.raw_ptr);
                const int b_row_stride_bytes = g.b.template stride<2>();
                const uint32_t a_total_bytes =
                    static_cast<uint32_t>(g.M_total) * static_cast<uint32_t>(a_row_stride_bytes_kt);
                const uint32_t b_per_group_bytes =
                    static_cast<uint32_t>(group_idx + 1) *
                    static_cast<uint32_t>(g.n) * static_cast<uint32_t>(b_row_stride_bytes);
                i32x4 a_srsrc_kt = make_srsrc((const void*)a_base_ptr, a_total_bytes);
                i32x4 b_srsrc_kt = make_srsrc((const void*)b_base_ptr, b_per_group_bytes);
                const uint32_t K_tail_base_bytes = static_cast<uint32_t>(g.fast_k);
                const uint32_t b_group_byte_base =
                    static_cast<uint32_t>(group_idx) *
                    static_cast<uint32_t>(g.n) * static_cast<uint32_t>(b_row_stride_bytes);

                auto load_a_kt = [&](A_row_reg& A_tile, int slab) __attribute__((always_inline)) {
                    const int M_warp_base = m_start_g + (br * 2 + slab) * HB + wm * RBM;
                    #pragma unroll
                    for (int h = 0; h < A_row_reg::height; ++h) {
                        const int A_row_idx = M_warp_base + h * 16 + row_lane;
                        const uint32_t v_base = static_cast<uint32_t>(
                            A_row_idx * a_row_stride_bytes_kt + K_tail_base_bytes + k_lane_byte);
                        const uint32_t v_lo = both_valid ? v_base : SENTINEL;
                        const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
                        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(a_srsrc_kt, v_lo, 0, 0);
                        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(a_srsrc_kt, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[0]) = v0;
                        *reinterpret_cast<__uint128_t*>(&A_tile.tiles[h][0].data[4]) = v1;
                    }
                };
                auto load_b_kt = [&](B_row_reg& B_tile, int n_strip) __attribute__((always_inline)) {
                    const int N_warp_base = (bc * 2 + n_strip) * HB + wn * RBN;
                    #pragma unroll
                    for (int h_b = 0; h_b < B_row_reg::height; ++h_b) {
                        const int B_row_idx_in_group = N_warp_base + h_b * 16 + row_lane;
                        const uint32_t v_base = b_group_byte_base + static_cast<uint32_t>(
                            B_row_idx_in_group * b_row_stride_bytes + K_tail_base_bytes + k_lane_byte);
                        const uint32_t v_lo = both_valid ? v_base : SENTINEL;
                        const uint32_t v_hi = both_valid ? (v_base + 16) : SENTINEL;
                        __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(b_srsrc_kt, v_lo, 0, 0);
                        __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(b_srsrc_kt, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[0]) = v0;
                        *reinterpret_cast<__uint128_t*>(&B_tile.tiles[h_b][0].data[4]) = v1;
                    }
                };
                load_b_kt(b0, 0);
                load_b_kt(b1, 1);
                load_a_kt(a,  0);
                asm volatile("s_waitcnt vmcnt(0)");
                rcr_mma_v2_vacc_wrapper<true>(cA, a, b0);
                rcr_mma_v2_vacc_wrapper<true>(cB, a, b1);
                load_a_kt(a,  1);
                asm volatile("s_waitcnt vmcnt(0)");
                rcr_mma_v2_vacc_wrapper<true>(cC, a, b0);
                rcr_mma_v2_vacc_wrapper<true>(cD, a, b1);
            }
        }

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
    // R22: FUSED=true now also goes through pinned body (FUSED block ported)
    grouped_rcr_kernel_body_pinned<N_MASKED_STORE, FUSED_KTAIL>(g);
}

// R188-step1: P1.2 32x32x64 mfma body skeleton. Uses HK rt_32x64/rt_32x32 frag
// types + st_32x64 LDS swizzle + mma_ABt high-level API. Per-acc 64×32 area
// = 2 M-tiles × 1 N-tile × 2 K-halves = 4 mfma (vs 8 with 16x16x128).
// Half mfma count per acc. Expected lower compiler scheduling pressure +
// foundation R57 probe validated V=104 A=0 spill=0 at 8-warp scale.
template<bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__device__ __forceinline__
void grouped_rcr_kernel_body_pinned_32(const grouped_layout_globals g) {
    using ST_rcr = st_fp8e4m3<HB, BK, st_32x64_s>;
    __shared__ ST_rcr As[2][2];
    __shared__ ST_rcr Bs[2][2];
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;

    using A_row_reg_32 = rt_fp8e4m3<RBM, BK, row_l, rt_32x64_s>;
    using B_row_reg_32 = rt_fp8e4m3<RBN, BK, row_l, rt_32x64_s>;
    using AccTile_32 = rt_fl<RBM, RBN, col_l, rt_32x32_s>;
    A_row_reg_32 a;
    B_row_reg_32 b0, b1;
    AccTile_32 cA, cB, cC, cD;

    const int slots_eff = gridDim.x;
    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    const int chunk_size_eff = g.chunk_size > 0 ? g.chunk_size : 32;
    int pid = chiplet_transform_chunked(blockIdx.x, slots_eff, xcds_eff, chunk_size_eff);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;
    const int ki_dyn   = g.ki;

    init_group_cumsum_smem<MAX_G_PLUS_1>(g, s_offs, s_cum_tiles, s_total_tiles,
                                         num_pid_n, /*M_BLOCK_DIV=*/BLOCK_SIZE);
    const int total_tiles = s_total_tiles;

    // R188-step1: Simple main loop without prefetch optimization first
    // (validate correctness + spill metadata before tuning pipeline)
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
        constexpr int m_subtile_C = 0;
        const int m_limit = M_g;

        zero(cA); zero(cB); zero(cC); zero(cD);

        int tic = 0;
        for (int k = 0; k < ki_dyn; ++k, tic ^= 1) {
            // Load tiles for current K-iter
            coord<ST_rcr> a0_co = {0, 0, br*2,   k};
            coord<ST_rcr> a1_co = {0, 0, br*2+1, k};
            coord<ST_rcr> b0_co = {0, group_idx, bc*2,   k};
            coord<ST_rcr> b1_co = {0, group_idx, bc*2+1, k};
            // R195-real: try G::load instead of rcr_8w_load_hoist
            G::load(As[tic][0], a_gl_g, a0_co, soA);
            G::load(As[tic][1], a_gl_g, a1_co, soA);
            G::load(Bs[tic][0], g.b, b0_co, soB);
            G::load(Bs[tic][1], g.b, b1_co, soB);
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();
            auto a_sub_m0 = subtile_inplace<RBM, BK>(As[tic][0], {wm, 0});
            load(a, a_sub_m0);
            auto b0_sub = subtile_inplace<RBN, BK>(Bs[tic][0], {wn, 0});
            load(b0, b0_sub);
            auto b1_sub = subtile_inplace<RBN, BK>(Bs[tic][1], {wn, 0});
            load(b1, b1_sub);
            __builtin_amdgcn_s_barrier();
            mma_ABt(cA, a, b0, cA);
            mma_ABt(cB, a, b1, cB);
            __builtin_amdgcn_s_barrier();
            auto a_sub_m1 = subtile_inplace<RBM, BK>(As[tic][1], {wm, 0});
            load(a, a_sub_m1);
            __builtin_amdgcn_s_barrier();
            mma_ABt(cC, a, b0, cC);
            mma_ABt(cD, a, b1, cD);
            __builtin_amdgcn_s_barrier();
        }

        const float combined_scale = resolve_combined_scale_grp(g);
        if (wm == 0) __builtin_amdgcn_s_barrier();
        // R389 REVERT: r_tile is in units of RT::rows (= 64 for rt_32x32 cA height=2).
        // Original formula already gives non-overlapping rows: wm=0 → 0,128; wm=1 → 64,192.
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
        __builtin_amdgcn_s_barrier();
    }
}

template<bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__global__ __launch_bounds__(_NUM_THREADS, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void grouped_gemm_fp8_kernel_v2_32(const grouped_layout_globals g) {
    grouped_rcr_kernel_body_pinned_32<N_MASKED_STORE, FUSED_KTAIL>(g);
}


template<bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__device__ __forceinline__
void grouped_rrr_kernel_body_pinned(const grouped_layout_globals g) {
    __shared__ ST_row As[2][2];
    __shared__ ST_v2  Bs[2][2];
    constexpr int MAX_G_PLUS_1 = 65;
    __shared__ int s_offs[MAX_G_PLUS_1];
    __shared__ int s_cum_tiles[MAX_G_PLUS_1];
    __shared__ int s_total_tiles;
    // Uniform-group fast path (mirror reference at
    // mxfp8/Primus-Turbo/csrc/kernels/grouped_gemm/turbo/turbo_grouped_gemm_mxfp8_kernel.h).
    // If all groups have the same M (gpt_oss / DSV3 / most MoE training),
    // skip the 6-level binary search and use a single division per persistent
    // iteration to map gt → (group_idx, local_tile).
    __shared__ int s_uniform_M;     // > 0 iff all groups have same M; else -1
    __shared__ int s_tiles_per_g;   // (uniform_M / BLOCK_SIZE) * num_pid_n
    // K-tail cross-lane shuffle scratch (16 KB) — only allocated for
    // FUSED_KTAIL=true. Moved inside the if-constexpr branch below so
    // the FUSED_KTAIL=false instantiation can lift launch_bounds 1 → 2
    // (occupancy parity with dense gemm_kernel<RRR>).

    A_row_reg a;
    // 2026-05-22 PMC analysis showed H6 single-b0 causes +25% LDS reads
    // (72 vs dense's 48 ds_read_b64_tr_b8) which stalls mfma issue
    // (mfma throughput -18% per CU). Reverting to b0+b1 split (dense
    // pattern) eliminates the redundant Bs[tic][0] reload for cC. Spill
    // may increase 37→58 but mfma throughput gain dominates.
    B_col_reg b0, b1;
    rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;

    // Round-2 (FP8 backward unblock): mirror RCR — read host-side
    // ``g.num_xcds`` knob, fall back to the default 8 when unset.
    const int xcds_eff = g.num_xcds > 0 ? g.num_xcds : BLOCK_SWIZZLE_NUM_XCDS;
    // 2026-05-20: use gridDim.x not NUM_CUS so we can launch with grid =
    // total_tiles (TK_*_GRID_MODE=tile experiment).
    const int slots_eff = gridDim.x;
    // R475: per-shape chunk_size for RRR v2 (mirror R456 RCR pattern)
    int chunk_size_eff;
    if (g.chunk_size > 0) {
        chunk_size_eff = g.chunk_size;
    } else {
        const int n_cols = static_cast<int>(g.c.cols());
        const int k_cols = static_cast<int>(g.a.cols());
        chunk_size_eff = (k_cols >= 4096 && n_cols >= 4096) ? 48 : 64;
    }
    int pid = chiplet_transform_chunked(
        blockIdx.x, slots_eff, xcds_eff, chunk_size_eff);

    int wm = warpid() / WARPS_N;
    int wn = warpid() % WARPS_N;
    const int num_pid_n = g.bpc;
    const int ki_dyn   = g.ki;

    if (threadIdx.x <= g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_offs[threadIdx.x] = static_cast<int>(g.group_offs[threadIdx.x]);
    }
    if (threadIdx.x > g.G && threadIdx.x < MAX_G_PLUS_1) {
        s_cum_tiles[threadIdx.x] = 0x7FFFFFFF;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int prev = s_offs[0];
        s_cum_tiles[0] = 0;
        int t = 0;
        const int M0 = s_offs[1] - prev;
        bool uniform = (g.G > 0);
        #pragma unroll 1
        for (int gi = 0; gi < g.G; ++gi) {
            const int next = s_offs[gi + 1];
            const int M_g_i = next - prev;
            if (M_g_i != M0) uniform = false;
            // ceil_div: partial last M-tile (M_g % BLOCK_SIZE != 0) gets its
            // own (br, bc) tile; per-group shifted gl view + m_limit masked
            // store below handle the OOB rows. Floor div silently skipped
            // small groups (M_g < BLOCK_SIZE) → garbage / zero output.
            t += kittens::ceil_div(M_g_i, BLOCK_SIZE) * num_pid_n;
            s_cum_tiles[gi + 1] = t;
            prev = next;
        }
        s_total_tiles  = t;
        s_uniform_M    = uniform ? M0 : -1;
        s_tiles_per_g  = uniform ? kittens::ceil_div(M0, BLOCK_SIZE) * num_pid_n : 0;
    }
    __syncthreads();
    const int total_tiles  = s_total_tiles;
    const int uniform_M    = s_uniform_M;
    const int tiles_per_g  = s_tiles_per_g;

    constexpr int bptA = ST_row::underlying_subtile_bytes_per_thread;
    constexpr int bpmA = bptA * _NUM_THREADS;
    constexpr int mptA = ST_row::rows * ST_row::cols * sizeof(fp8e4m3) / bpmA;
    uint32_t soA[mptA];
    G::prefill_swizzled_offsets(As[0][0], g.a, soA);

    constexpr int bptB = ST_v2::underlying_subtile_bytes_per_thread;
    constexpr int bpmB = bptB * _NUM_THREADS;
    constexpr int mptB = ST_v2::rows * ST_v2::cols * sizeof(fp8e4m3) / bpmB;
    uint32_t soB[mptB];
    G::prefill_swizzled_offsets(Bs[0][0], g.b, soB);

    for (int gt = pid; gt < total_tiles; gt += slots_eff) {
        int group_idx, local_tile;
        if (uniform_M > 0) {
            // Uniform-group fast path: single division per persistent iter.
            // tiles_per_g is identical across groups so simple integer
            // division gives group_idx; remainder gives local_tile.
            group_idx  = gt / tiles_per_g;
            local_tile = gt - group_idx * tiles_per_g;
        } else {
            // Variable-M fallback: 6-level binary search over s_cum_tiles.
            int lo = 0;
            int hi = MAX_G_PLUS_1 - 1;
            #pragma unroll
            for (int level = 0; level < 6; ++level) {
                const int mid = (lo + hi + 1) >> 1;
                if (gt >= s_cum_tiles[mid]) lo = mid;
                else hi = mid - 1;
            }
            group_idx  = lo;
            local_tile = gt - s_cum_tiles[lo];
        }
        const int m_start_g = s_offs[group_idx];
        const int M_g = s_offs[group_idx + 1] - m_start_g;
        const int bpr_g = kittens::ceil_div(M_g, BLOCK_SIZE);

        int br, bc;
        if (g.bpc > bpr_g) {
            const int WGN = g.group_m;
            const int num_wgid_in_group = bpr_g * WGN;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_n = group_id * WGN;
            int group_size_n = min(num_pid_n - first_pid_n, WGN);
            if (group_size_n <= 0) continue;
            bc = first_pid_n + ((local_tile % num_wgid_in_group) % group_size_n);
            br = (local_tile % num_wgid_in_group) / group_size_n;
        } else {
            const int WGM = g.group_m;
            const int num_wgid_in_group = WGM * num_pid_n;
            int group_id = local_tile / num_wgid_in_group;
            int first_pid_m = group_id * WGM;
            int group_size_m = min(bpr_g - first_pid_m, WGM);
            if (group_size_m <= 0) continue;
            br = first_pid_m + ((local_tile % num_wgid_in_group) % group_size_m);
            bc = (local_tile % num_wgid_in_group) / group_size_m;
        }
        if (br >= bpr_g || bc >= num_pid_n) continue;

        auto a_gl_g = g.a;
        auto c_gl_g = g.c;
        patch_per_group_gl_view(a_gl_g, c_gl_g, m_start_g, M_g);
        const int a_row_stride_bytes = static_cast<int>(g.a.template stride<2>()) * sizeof(*g.a.raw_ptr);
        // H3 (2026-05-20): outer/inner ptr split for B-tensor — shift the
        // grouped B SRD base by `group_idx * K * N * sizeof(fp8)` once per
        // tile, then drop `group_idx` from `b_co` so it matches dense's
        // `{0, 0, k, s}`. Theory: lets the compiler reuse dense's exact
        // register/MMA schedule for B-loads, freeing the per-tile slot
        // currently held by `group_idx` inside the b_co lambda capture.
        // Safety: SRD bound (full-tensor depth=G*K*N) extends past tensor
        // end by (G-group_idx-1)*K*N bytes; HW returns 0 for those OOB
        // reads, and the k/bc loops stay within the current group's K*N
        // window so no garbage is ever read.
        auto b_gl_g = g.b;
        {
            using ptr_t = decltype(b_gl_g.raw_ptr);
            const int64_t b_group_stride_bytes =
                static_cast<int64_t>(g.b.template stride<1>()) * sizeof(*g.b.raw_ptr);
            auto* base = reinterpret_cast<uint8_t*>(b_gl_g.raw_ptr);
            b_gl_g.raw_ptr = reinterpret_cast<ptr_t>(
                base + static_cast<int64_t>(group_idx) * b_group_stride_bytes);
        }
        // 2026-05-20: a-load switched from G::load(.., g.a, ..) + m_subtile_A
        // shift to rcr_8w_load_hoist(.., a_gl_g, ..) — same loader RCR uses,
        // and the only one that survives the compiler optimizing away a
        // patched gl<>::raw_ptr (it computes tile_byte_offset =
        // global_ptr - tensor_base explicitly).
        //
        // Combined with the patched view (rows_internal = M_g), the SRD bound
        // becomes per-group byte-level — HW raw_buffer_load clamps OOB rows
        // to 0 so unbalanced M_g works without prior-iter HBM garbage leak.
        // (Pre-2026-05-20 state: SRD bound was full-tensor M_total, OOB rows
        // returned arbitrary VRAM contents that were freed-but-not-zeroed
        // from prior iters → reproducible memory access fault on iter 2 of
        // the HIPKITTEN sweep.)
        //
        // FUSED_KTAIL path still uses g.a + m_subtile_A unit-coord encoding;
        // see below.
        const int m_subtile_A = m_start_g / HB;  // FUSED_KTAIL only
        constexpr int m_subtile_C = 0;
        const int m_limit = M_g;

        // RRR coord conventions (mirror of dense gemm_kernel<RRR>):
        //   a_co(s, k) : A is per-group [M_g, K]  → unit_coord row index 0..bpr_g-1.
        //                Pointer-base shift via patched a_gl_g handles m_start_g.
        //   b_co(s, k) : B is per-group-shifted view (b_gl_g) — drop the
        //                group_idx dim to match dense's coord (H3 hoist).
        auto a_co = [&](int s, int k) -> coord<ST_row> {
            return {0, 0, s, k};
        };
        auto b_co = [&](int s, int k) -> coord<ST_v2> {
            return {0, 0, k, s};
        };

        auto load_a = [&](A_row_reg& dst, ST_row& tile, int wi) {
            auto sub = subtile_inplace<RBM, BK>(tile, {wi, 0});
            load(dst, sub);
        };
        auto load_b = [&](B_col_reg& dst, ST_v2& tile, int wi) {
            load_col_from_st(dst, tile, wi * RBN);
        };

        zero(cA); zero(cB); zero(cC); zero(cD);

        int tic = 0, toc = 1;
        // Prologue: tile-0 + tile-1 (mirrors dense gemm_kernel<RRR>
        // lines 1421-1435). a-load uses g.a (unpatched) per fix above.
        G::load(Bs[tic][0], b_gl_g, b_co(bc*2,   0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], a_gl_g, a_co(br*2,   0), soA);
        G::load(Bs[tic][1], b_gl_g, b_co(bc*2+1, 0), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[tic][1], a_gl_g, a_co(br*2+1, 0), soA);

        if (wm == 1) __builtin_amdgcn_s_barrier();
        TK_WAIT_VMCNT(RRR_INIT0_VMCNT);
        __builtin_amdgcn_s_barrier();

        G::load(Bs[toc][0], b_gl_g, b_co(bc*2,   1), soB);
        rcr_8w_load_hoist<_NUM_THREADS>(As[toc][0], a_gl_g, a_co(br*2,   1), soA);
        G::load(Bs[toc][1], b_gl_g, b_co(bc*2+1, 1), soB);

        TK_WAIT_VMCNT(RRR_INIT1_VMCNT);
        __builtin_amdgcn_s_barrier();

        // 2026-05-22 Round-3 PMC-driven: b0+b1 split (dense pattern, mirrors
        // dense main loop at lines 1380-1412). mma order cA→cB→cC→cD reuses
        // b0 (Bs[tic][0]) for cA+cC and b1 (Bs[tic][1]) for cB+cD without
        // reload. Eliminates 1 redundant LDS read per iter (cC's b0 reload
        // from strip0 in H6+H7). PMC: was 72 ds_read_b64_tr_b8, target ~48.
        TK_PRAGMA_UNROLL(RRR_MAIN_UNROLL)
        for (int k = 0; k < ki_dyn - 2; k++, tic ^= 1, toc ^= 1) {
            // Phase 1: cA = mma(slab0, strip0). Load both b0 and b1 upfront.
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], a_gl_g, a_co(br*2+1, k+1), soA);
            // R17 (2026-05-22): remove s_setprio around mfma. The +1/-1 prio
            // sequence creates an implicit scheduling barrier in LLVM AMDGPU
            // backend, preventing instruction interleaving across mfma issue
            // boundaries. Removing lets scheduler pack more LDS prefetch into
            // mfma latency cycles.
            TK_WAIT_LGKM(RRR_PREFETCH_LGKM); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            rrr_mma_v2_vacc_wrapper<false>(cA, a, b0);
            __builtin_amdgcn_s_barrier();

            // Phase 2: cB = mma(slab0, strip1).
            load_b(b1, Bs[tic][1], wn);
            G::load(Bs[tic][0], b_gl_g, b_co(bc*2, k+2), soB);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            rrr_mma_v2_vacc_wrapper<false>(cB, a, b1);
            __builtin_amdgcn_s_barrier();

            // Phase 3: cC = mma(slab1, strip0).
            load_a(a, As[tic][1], wm);
            G::load(Bs[tic][1], b_gl_g, b_co(bc*2+1, k+2), soB);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            rrr_mma_v2_vacc_wrapper<false>(cC, a, b0);
            __builtin_amdgcn_s_barrier();

            // Phase 4: cD = mma(slab1, strip1).
            rcr_8w_load_hoist<_NUM_THREADS>(As[tic][0], a_gl_g, a_co(br*2, k+2), soA);
            TK_WAIT_VMCNT(RRR_STEADY_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            rrr_mma_v2_vacc_wrapper<false>(cD, a, b1);
            __builtin_amdgcn_s_barrier();
            RRR_SCHED_BARRIER();
        }

        // Epilog 1 (Round-4: dense-pattern b0 prefetch at end → epilog 2 saves 1 load_b).
        {
            load_b(b0, Bs[tic][0], wn);
            load_b(b1, Bs[tic][1], wn);
            load_a(a, As[tic][0], wm);
            rcr_8w_load_hoist<_NUM_THREADS>(As[toc][1], a_gl_g, a_co(br*2+1, ki_dyn-1), soA);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_v2_vacc_wrapper<false>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_v2_vacc_wrapper<false>(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            TK_WAIT_VMCNT(RRR_EPILOGUE_VMCNT); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_v2_vacc_wrapper<false>(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            // Dense-pattern prefetch: load b0 for epilog 2 NOW (Bs[toc][0]
            // post-flip = Bs[new tic][0]). Hides LDS read latency under cD mma.
            load_b(b0, Bs[toc][0], wn);
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_v2_vacc_wrapper<false>(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            RRR_SCHED_BARRIER();
            tic ^= 1; toc ^= 1;
        }

        // Epilog 2 (Round-4: b0 prefetched from epilog 1; only load b1 fresh).
        {
            load_b(b1, Bs[tic][1], wn);
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_v2_vacc_wrapper<false>(cA, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_v2_vacc_wrapper<false>(cB, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            load_a(a, As[tic][1], wm);
            __builtin_amdgcn_s_barrier();
            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_v2_vacc_wrapper<false>(cC, a, b0); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();

            MAYBE_DRAIN_LGKM();
            __builtin_amdgcn_s_setprio(1); rrr_mma_v2_vacc_wrapper<false>(cD, a, b1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }

        // ===== FUSED_KTAIL (RRR FP8) =====
        // K-tail K_REM=64 reduction fused into main kernel epilog. Both A
        // AND B loaded direct-to-register via raw_buffer_load_b8 to avoid
        // the load_col_from_st (ds_read_b64_tr_b8) path that triggers a
        // mfma operand-forwarding quirk for RRR. Per-group SRD bounds
        // protect against cross-group/OOB garbage reads (K_row >= g.k → 0).
        // K-aligned-to-32 chunks: lane chunks 0,1 carry real data (K=0..64),
        // chunks 2,3 zeroed (K=64..128, mfma's K=128 zero-pad).
        if constexpr (FUSED_KTAIL) {
            // Per-warp scratch for cross-lane B shuffle (see step-2..4 below).
            // Layout per warp: 64 K-rows × 16 N-cols × 1 byte = 1024 bytes per
            // (warp, j); 8 warps × 2 j = 16 KB total. Only allocated for the
            // FUSED_KTAIL=true template instantiation.
            __shared__ uint8_t kt_b_scratch[8][2][1024];

            if (g.fast_k < g.k) {
                typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
                B_col_reg b0_kt, b1_kt;

                const int laneid_fk  = kittens::laneid();
                const int row_lane_fk = laneid_fk % 16;
                const int chunk_fk    = laneid_fk / 16;          // 0..3
                const int k_lane_byte_fk = chunk_fk * 32;
                const bool ab_chunk_valid = (chunk_fk < 2);      // K=0..64 chunks have real K-tail data
                constexpr uint32_t SENTINEL_FK = 0xFFFF0000u;

                // ---- Per-group SRDs (so OOB voffset returns 0) ----
                const fp8e4m3* a_base_ptr_fk =
                    (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
                fp8e4m3* b_grp_base = const_cast<fp8e4m3*>(
                    &g.b[{0, group_idx, 0, 0}]);
                const uint32_t a_row_stride_bytes_fk = g.a.template stride<2>();
                const uint32_t b_row_stride_bytes_fk = g.b.template stride<2>();
                const uint32_t a_total_bytes_fk =
                    static_cast<uint32_t>(g.M_total) * a_row_stride_bytes_fk;
                const uint32_t b_grp_total_bytes_fk =
                    static_cast<uint32_t>(g.k) * b_row_stride_bytes_fk;
                i32x4 a_srsrc_fk = make_srsrc(
                    (const void*)a_base_ptr_fk, a_total_bytes_fk);
                i32x4 b_srsrc_fk = make_srsrc(
                    (const void*)b_grp_base, b_grp_total_bytes_fk);
                const uint32_t K_tail_byte_fk =
                    static_cast<uint32_t>(g.fast_k);

                // ---- Custom A K-tail load (direct-to-reg, mirror RCR) ----
                auto load_a_kt_fk = [&](A_row_reg& A_tile, int slab)
                        __attribute__((always_inline)) {
                    const int M_warp_base =
                        (m_subtile_A + br * 2 + slab) * HB + wm * RBM;
                    #pragma unroll
                    for (int h = 0; h < A_row_reg::height; ++h) {
                        const int A_row_idx =
                            M_warp_base + h * 16 + row_lane_fk;
                        const uint32_t v_base = static_cast<uint32_t>(
                            A_row_idx * a_row_stride_bytes_fk +
                            K_tail_byte_fk + k_lane_byte_fk);
                        const uint32_t v_lo = ab_chunk_valid ? v_base : SENTINEL_FK;
                        const uint32_t v_hi = ab_chunk_valid ? (v_base + 16) : SENTINEL_FK;
                        __uint128_t va0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_fk, v_lo, 0, 0);
                        __uint128_t va1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            a_srsrc_fk, v_hi, 0, 0);
                        *reinterpret_cast<__uint128_t*>(
                            &A_tile.tiles[h][0].data[0]) = va0;
                        *reinterpret_cast<__uint128_t*>(
                            &A_tile.tiles[h][0].data[4]) = va1;
                    }
                };

                // ---- B K-tail load: cross-lane shuffle via LDS scratch ----
                // OLD: 32 strided byte_b8 loads per lane × 16 lanes = 512 ops/warp/(j,strip)
                // NEW: 1 b128/lane (16 contig N bytes for 1 K row) + LDS write +
                //      32 byte-reads from LDS column-strided.
                // Memory traffic same; instruction count cut ~2-3x; LDS bandwidth
                // is much higher than strided global so net throughput up.
                //
                // gfx950 quirk note: ds_read_b64_tr_b8 in mma chain b128(A)+ds_read(B)+mfma
                // silently zeros mfma. We use NORMAL ds_read (not _tr_b8 transposed)
                // so quirk should not fire.
                const int warp_id_fk = kittens::warpid();
                auto load_b_kt_fk = [&](B_col_reg& B_tile, int n_strip)
                        __attribute__((always_inline)) {
                    const int N_warp_base =
                        (bc * 2 + n_strip) * HB + wn * RBN;
                    #pragma unroll
                    for (int j = 0; j < B_col_reg::width; ++j) {
                        // STEP 1: ALL 64 lanes load — lane l loads K=K_tail+l
                        // (covers K=K_tail+0..63 = the full real K-tail range).
                        const uint32_t K_pos = K_tail_byte_fk + laneid_fk;
                        const uint32_t N_col_start =
                            static_cast<uint32_t>(N_warp_base + j * 16);
                        const uint32_t voffset =
                            K_pos * b_row_stride_bytes_fk + N_col_start;
                        __uint128_t v = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                            b_srsrc_fk, voffset, 0, 0);
                        // STEP 2: write to LDS column-major: byte at (K, N) →
                        // LDS[N*64 + K]. Each lane scatters its 16 bytes:
                        // byte i of lane l → LDS[i*64 + l]. This makes per-lane
                        // strided writes (16 ds_write_b8/lane) but enables 8
                        // ds_read_b32/lane on read side (32 contig K bytes).
                        // Net: 24 LDS ops/lane vs 33 in row-major scheme.
                        uint8_t* v_bytes = (uint8_t*)&v;
                        uint8_t* lds_base = &kt_b_scratch[warp_id_fk][j][0];
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            lds_base[i * 64 + laneid_fk] = v_bytes[i];
                        }
                        // STEP 3: drain LDS commits.
                        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");

                        // STEP 4: contiguous 32-byte read per lane.
                        // Chunk 0/1 lane reads its 32 K bytes for N=row_lane_fk,
                        // K offset = chunk_fk*32. LDS layout col-major puts
                        // (K=0..63, N=row_lane_fk) at LDS[row_lane_fk*64..+63].
                        // Lane reads 32 contig bytes = 8 ds_read_b32.
                        intx8_t b_pack = intx8_t{};
                        if (ab_chunk_valid) {
                            const int K_offset = chunk_fk * 32;
                            uint8_t* lds_col_base =
                                &kt_b_scratch[warp_id_fk][j][row_lane_fk * 64 + K_offset];
                            // 32 contig bytes = 2 b128 reads.
                            __uint128_t r0 =
                                *reinterpret_cast<__uint128_t*>(lds_col_base);
                            __uint128_t r1 =
                                *reinterpret_cast<__uint128_t*>(lds_col_base + 16);
                            uint8_t* bp = (uint8_t*)&b_pack;
                            *reinterpret_cast<__uint128_t*>(bp) = r0;
                            *reinterpret_cast<__uint128_t*>(bp + 16) = r1;
                        }
                        *reinterpret_cast<intx8_t*>(
                            &B_tile.tiles[0][j].data[0]) = b_pack;
                    }
                };

                // ---- Runtime knob: LDS-based B load (avoids 32 byte-loads/lane).
                // Quirk-free strategy: pre-zero Bs[tic], coop-load B via LDS
                // with per-group SRD bound = K_REM*N (so K=K_REM..127 voffsets
                // OOB → no-op → preserve zero), keep A as direct-to-reg b128
                // (chunks 2,3 may read next-M-row garbage but multiplied by
                // B=0 in those chunks → harmless). Avoids the b128(A) →
                // ds_read(B) → mfma quirk because B's chunks 2,3 are zero
                // (so wrong contribution = 0).
                // Compile-time gated experiment. Toggle via #define.
                #ifndef HK_FP8_RRR_FUSED_KTAIL_LDS
                #define HK_FP8_RRR_FUSED_KTAIL_LDS 0
                #endif
                constexpr bool ktail_lds_active = (HK_FP8_RRR_FUSED_KTAIL_LDS != 0);

                if constexpr (ktail_lds_active) {
                    // ---- Cooperative pre-zero Bs[tic][0/1] ----
                    constexpr int ST_V2_B128 = (sizeof(ST_v2) / 16);
                    const int tid_z = threadIdx.x;
                    __uint128_t* Bs0_ptr = reinterpret_cast<__uint128_t*>(&Bs[tic][0].data[0]);
                    __uint128_t* Bs1_ptr = reinterpret_cast<__uint128_t*>(&Bs[tic][1].data[0]);
                    #pragma unroll
                    for (int idx = tid_z; idx < ST_V2_B128; idx += _NUM_THREADS) {
                        Bs0_ptr[idx] = 0;
                        Bs1_ptr[idx] = 0;
                    }
                    __syncthreads();

                    // ---- Custom coop load B with per-group SRD ----
                    // Per-group bound = K_REM * N bytes. Voffsets where K_local
                    // >= K_REM (chunks 2,3) auto-no-op → preserves zero.
                    const uint32_t b_row_stride = g.b.template stride<2>();
                    const uint32_t b_per_group_bound =
                        static_cast<uint32_t>(g.k - g.fast_k) * b_row_stride;

                    constexpr int bptB_lds = ST_v2::underlying_subtile_bytes_per_thread;
                    constexpr int bpwB_lds = bptB_lds * kittens::WARP_THREADS;
                    constexpr int mptB_lds = ST_v2::rows * ST_v2::cols * sizeof(fp8e4m3) / (bptB_lds * _NUM_THREADS);
                    constexpr int nwarps_lds = _NUM_THREADS / kittens::WARP_THREADS;
                    const int laneid_lds = kittens::laneid();
                    const int warpid_lds = kittens::warpid() % nwarps_lds;

                    #pragma unroll
                    for (int strip = 0; strip < 2; ++strip) {
                        // global_ptr at K-tail tile origin within this group
                        coord<ST_v2> tile_idx{0, group_idx, ki_dyn, bc*2+strip};
                        coord<> unit = tile_idx.template unit_coord<2, 3>();
                        fp8e4m3* gp = (fp8e4m3*)&g.b[unit];
                        i32x4 b_srsrc_lds = make_srsrc((const void*)gp, b_per_group_bound);

                        ST_v2& Bs_dst = Bs[tic][strip];
                        const uintptr_t lds_tile_base = reinterpret_cast<uintptr_t>(&Bs_dst.data[0]);

                        #pragma unroll
                        for (int i = 0; i < mptB_lds; i++) {
                            const int lane_byte_offset = (laneid_lds * bptB_lds) + (warpid_lds * bpwB_lds) + (i * nwarps_lds * bpwB_lds);
                            const int subtile_id = lane_byte_offset / ST_v2::underlying_subtile_bytes;
                            const int subtile_row = subtile_id / ST_v2::underlying_subtiles_per_row;
                            const int subtile_col = subtile_id % ST_v2::underlying_subtiles_per_row;
                            const int subtile_lane_byte_offset = lane_byte_offset % ST_v2::underlying_subtile_bytes;

                            const int row = subtile_lane_byte_offset / ST_v2::underlying_subtile_row_bytes;
                            const int col = (subtile_lane_byte_offset % ST_v2::underlying_subtile_row_bytes) / sizeof(fp8e4m3);

                            const uint32_t swizzled_shared_byte_offset = Bs_dst.swizzle({row, col});

                            const int swizzled_global_row = (swizzled_shared_byte_offset / ST_v2::underlying_subtile_row_bytes) + subtile_row * ST_v2::underlying_subtile_rows;
                            const int swizzled_global_col = (swizzled_shared_byte_offset % ST_v2::underlying_subtile_row_bytes) / sizeof(fp8e4m3) + subtile_col * ST_v2::underlying_subtile_cols;
                            const uint32_t swizzled_global_byte_offset =
                                (swizzled_global_row * b_row_stride + swizzled_global_col) * sizeof(fp8e4m3);

                            const int warp_linear_offset = (warpid_lds * bpwB_lds) + (i * nwarps_lds * bpwB_lds);
                            const int lds_subtile_id = warp_linear_offset / ST_v2::underlying_subtile_bytes;
                            uintptr_t lds_addr = lds_tile_base + warp_linear_offset + lds_subtile_id * ST_v2::subtile_padding;
                            kittens::as3_uint32_ptr lds_ptr = (kittens::as3_uint32_ptr)(lds_addr);

                            kittens::llvm_amdgcn_raw_buffer_load_lds(
                                b_srsrc_lds, lds_ptr, bptB_lds,
                                swizzled_global_byte_offset, 0, 0,
                                static_cast<int>(kittens::coherency::cache_all));
                        }
                    }

                    // A side: keep direct-to-reg b128 (working, fast).
                    // 2026-05-16 spill fix: serialize slabs to reuse `a`.
                    load_a_kt_fk(a, 0);

                    // Drain global loads (B coop load + A slab 0)
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                    __syncthreads();

                    // ds_read B from LDS (col-layout via tr_b8)
                    load_col_from_st(b0_kt, Bs[tic][0], wn * RBN);
                    load_col_from_st(b1_kt, Bs[tic][1], wn * RBN);
                    MAYBE_DRAIN_LGKM();

                    rrr_mma(cA, a, b0_kt);
                    rrr_mma(cB, a, b1_kt);

                    load_a_kt_fk(a, 1);
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                    rrr_mma(cC, a, b0_kt);
                    rrr_mma(cD, a, b1_kt);
                    __builtin_amdgcn_s_barrier();
                } else {
                    // Production direct-to-reg byte-load path (slow but quirk-safe).
                    // 2026-05-16 spill fix: serialize slabs to reuse `a` (saves
                    // ~32 VGPRs of A_row_reg pressure from a_kt0+a_kt1).
                    load_b_kt_fk(b0_kt, 0);
                    load_b_kt_fk(b1_kt, 1);
                    load_a_kt_fk(a, 0);
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                    rrr_mma(cA, a, b0_kt);
                    rrr_mma(cB, a, b1_kt);
                    load_a_kt_fk(a, 1);
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                    rrr_mma(cC, a, b0_kt);
                    rrr_mma(cD, a, b1_kt);
                    __builtin_amdgcn_s_barrier();
                }
            }
        }

#if FP8_RRR_FUSE_PROBE
        if (g.fast_k < g.k) {
            // ---- Cooperative pre-zero of Bs[tic][0/1] ----
            // ST_v2 has swizzle padding (= 17408 bytes); strip to 16-byte
            // alignment so we can b128-store the entire region incl. padding.
            constexpr int ST_V2_B128 = (sizeof(ST_v2) / 16);
            const int tid = threadIdx.x;
            __uint128_t* Bs0_ptr = reinterpret_cast<__uint128_t*>(&Bs[tic][0].data[0]);
            __uint128_t* Bs1_ptr = reinterpret_cast<__uint128_t*>(&Bs[tic][1].data[0]);
            #pragma unroll
            for (int idx = tid; idx < ST_V2_B128; idx += _NUM_THREADS) {
                Bs0_ptr[idx] = 0;
                Bs1_ptr[idx] = 0;
            }
            __syncthreads();

            // ---- B side: cooperative G::load on K-tail iter ----
            // OOB voffsets (k_row >= K_global) no-op on raw_buffer_load_lds →
            // pre-zeroed bytes preserved → effective zero-pad for K=[K_global,
            // fast_k + K_BLOCK). Cross-group contamination is OK for the G=1
            // probe shape (full-tensor SRD == per-group SRD); production hybrid
            // would need per-group SRD construction.
            G::load(Bs[tic][0], g.b, b_co(bc*2,   ki_dyn), soB);
            G::load(Bs[tic][1], g.b, b_co(bc*2+1, ki_dyn), soB);

            const int laneid = kittens::laneid();
            const int row_lane = laneid % 16;
            const int k_lane_byte = (laneid / 16) * 32;
            const int K_REM = g.k - g.fast_k;
            const bool b128_lo_valid = (k_lane_byte + 16) <= K_REM;
            const bool b128_hi_valid = (k_lane_byte + 32) <= K_REM;
            constexpr uint32_t SENTINEL = 0xFFFF0000u;
            // Original absolute A SRD; m_start_g shifted into the byte
            // offset (mirrors grouped_rcr_kernel FUSED_KTAIL).
            const fp8e4m3* a_base_ptr = (const fp8e4m3*)&g.a[{0, 0, 0, 0}];
            const uint32_t a_total_bytes =
                static_cast<uint32_t>(g.M_total) *
                static_cast<uint32_t>(a_row_stride_bytes);
            i32x4 a_srsrc_kt = make_srsrc((const void*)a_base_ptr, a_total_bytes);
            const uint32_t K_tail_base_bytes =
                static_cast<uint32_t>(g.fast_k);

            auto load_a_kt = [&](int slab) __attribute__((always_inline)) {
                const int M_warp_base =
                    m_start_g + (br * 2 + slab) * HB + wm * RBM;
                #pragma unroll
                for (int h = 0; h < A_row_reg::height; ++h) {
                    const int A_row_idx = M_warp_base + h * 16 + row_lane;
                    const uint32_t v_base = static_cast<uint32_t>(
                        A_row_idx * a_row_stride_bytes +
                        K_tail_base_bytes + k_lane_byte);
                    const uint32_t v_lo = b128_lo_valid ? v_base : SENTINEL;
                    const uint32_t v_hi = b128_hi_valid ? (v_base + 16) : SENTINEL;
                    __uint128_t v0 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                        a_srsrc_kt, v_lo, 0, 0);
                    __uint128_t v1 = ::kittens::llvm_amdgcn_raw_buffer_load_b128(
                        a_srsrc_kt, v_hi, 0, 0);
                    *reinterpret_cast<__uint128_t*>(&a.tiles[h][0].data[0]) = v0;
                    *reinterpret_cast<__uint128_t*>(&a.tiles[h][0].data[4]) = v1;
                }
            };

            // Wait B G::load + LDS visibility before B reads.
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
            __syncthreads();

#if !FP8_RRR_FUSE_PROBE_SKIP_A_LOAD
            load_b(b0, Bs[tic][0], wn);
            load_a_kt(0);
            asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)" ::: "memory");
            rrr_mma(cA, a, b0);

            load_b(b1, Bs[tic][1], wn);
            rrr_mma(cB, a, b1);

            load_a_kt(1);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            rrr_mma(cC, a, b0);
            rrr_mma(cD, a, b1);
#endif
            __builtin_amdgcn_s_barrier();
        }
#endif

        const float combined_scale = resolve_combined_scale_grp(g);
        mul(cA, cA, combined_scale);
        mul(cB, cB, combined_scale);
        mul(cC, cC, combined_scale);
        mul(cD, cD, combined_scale);

        if (wm == 0) __builtin_amdgcn_s_barrier();
        const int r0 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+wm);
        const int r1 = __builtin_amdgcn_readfirstlane(m_subtile_C + br*WARPS_M*2+WARPS_M+wm);
        const int c0 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+wn);
        const int c1 = __builtin_amdgcn_readfirstlane(bc*WARPS_N*2+WARPS_N+wn);
        // Masked grouped store on the per-group shifted c_gl_g: m_limit=M_g
        // drops partial-last-M-tile OOB rows, n_limit=g.n drops partial-N.
        store_c_tile_mn_masked_grouped(c_gl_g, cA, /*group_idx=*/0, r0, c0, m_limit, g.n);
        store_c_tile_mn_masked_grouped(c_gl_g, cB, /*group_idx=*/0, r0, c1, m_limit, g.n);
        store_c_tile_mn_masked_grouped(c_gl_g, cC, /*group_idx=*/0, r1, c0, m_limit, g.n);
        store_c_tile_mn_masked_grouped(c_gl_g, cD, /*group_idx=*/0, r1, c1, m_limit, g.n);

        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
}
// R474: kernel symbol + dispatch update — replace v1 forward with v2 body

template<bool N_MASKED_STORE = false, bool FUSED_KTAIL = false>
__global__ __launch_bounds__(_NUM_THREADS, 1)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void grouped_rrr_kernel_v2(const grouped_layout_globals g) {
    grouped_rrr_kernel_body_pinned<N_MASKED_STORE, FUSED_KTAIL>(g);
}

// =============================================================================
// R473: v2 RRR entry point (forwards to v1 dispatch_grouped_rrr initially).
// PT-facing struct mirrors RCR's but with bK/bN swapped (RRR B is [G, K, N]).
// Future rounds: replace v1 forward with v2 RRR body using rrr_mma_v2_vacc_wrapper.
// =============================================================================
struct grouped_layout_globals_v2_rrr {
    const void* a_ptr;
    const void* b_ptr;
    void*       c_ptr;
    int M_total, G_b, bK, bN, cM, cN;
    const float* sa_ptr;
    const float* sb_ptr;
    const int64_t* group_offs_ptr;
    hipStream_t stream;
    int G, group_m, m_per_group, num_xcds;
    int num_slots, chunk_size;
    int bn_block;
};

inline void dispatch_grouped_rrr_v2(grouped_layout_globals_v2_rrr g_in) {
    grouped_layout_globals g{
        _gl_fp8(reinterpret_cast<fp8e4m3*>(const_cast<void*>(g_in.a_ptr)),
                1, 1, g_in.M_total, g_in.bK),
        _gl_fp8(reinterpret_cast<fp8e4m3*>(const_cast<void*>(g_in.b_ptr)),
                1, g_in.G_b, g_in.bK, g_in.bN),
        _gl_bf16(reinterpret_cast<bf16*>(g_in.c_ptr), 1, 1, g_in.cM, g_in.cN),
        0.f, 0.f, g_in.sa_ptr, g_in.sb_ptr,
        g_in.group_offs_ptr, g_in.stream,
        g_in.G, 0, 0, 0, 0,
        g_in.group_m, g_in.num_xcds, 0,
        0, 0,
        g_in.m_per_group, g_in.num_slots, g_in.chunk_size, 0,
        0, nullptr, g_in.bn_block,
    };
    // R474: use v2 RRR body (vacc wrapper for cA-cD, matches RCR R167 pattern).
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
        if (const char* e = std::getenv("TK_RRR_V2_NUM_CUS")) {
            const int v = std::atoi(e);
            if (v > 0 && v <= NUM_CUS) return v;
        }
        return NUM_CUS;
    }();
    const int slots = (g.num_slots > 0 && g.num_slots <= NUM_CUS)
        ? g.num_slots : slots_env;

    if (fuse_on) {
        if (n_aligned)
            grouped_rrr_kernel_v2<false, true><<<dim3(slots), g.block(), 0, g.stream>>>(g);
        else
            grouped_rrr_kernel_v2<true,  true><<<dim3(slots), g.block(), 0, g.stream>>>(g);
    } else {
        if (n_aligned)
            grouped_rrr_kernel_v2<false, false><<<dim3(slots), g.block(), 0, g.stream>>>(g);
        else
            grouped_rrr_kernel_v2<true,  false><<<dim3(slots), g.block(), 0, g.stream>>>(g);
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

    // R189: env knob to select 32x32 K-loop body
    static const bool use_32 = []() {
        if (const char* e = std::getenv("TK_RCR_V2_USE_32")) return std::atoi(e) > 0;
        return false;
    }();

    if (use_32) {
        if (fuse_on) {
            if (n_aligned)
                grouped_gemm_fp8_kernel_v2_32<false, true><<<dim3(slots), g.block(), 0, g.stream>>>(g);
            else
                grouped_gemm_fp8_kernel_v2_32<true,  true><<<dim3(slots), g.block(), 0, g.stream>>>(g);
        } else {
            if (n_aligned)
                grouped_gemm_fp8_kernel_v2_32<false, false><<<dim3(slots), g.block(), 0, g.stream>>>(g);
            else
                grouped_gemm_fp8_kernel_v2_32<true,  false><<<dim3(slots), g.block(), 0, g.stream>>>(g);
        }
    } else if (fuse_on) {
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
